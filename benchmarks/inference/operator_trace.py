"""Profile the explicit FP16 reference path and rank its expensive operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from benchmarks.common import environment_record
from engine.model import ExplicitDecodeRunner, load_model


def _device_time_us(event: object) -> float:
    """Handle profiler field naming across supported PyTorch releases."""
    value = getattr(event, "self_device_time_total", None)
    if value is None:
        value = getattr(event, "self_cuda_time_total", 0.0)
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile explicit prefill/decode operators")
    parser.add_argument("--prompt", default="Explain paged KV caching in one sentence.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/operator_trace_t4.json"))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.warmup_runs < 0 or args.top < 1:
        parser.error("token count/top must be positive and warmup runs non-negative")

    loaded = load_model(args.model, revision=args.revision, dtype=args.dtype)
    runner = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    for _ in range(args.warmup_runs):
        runner.generate(args.prompt, max_new_tokens=args.max_new_tokens)

    torch.cuda.synchronize(loaded.device)
    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function("explicit_generate"):
            result = runner.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    torch.cuda.synchronize(loaded.device)

    if args.trace is not None:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))

    events = list(prof.key_averages(group_by_input_shape=True))
    events.sort(key=_device_time_us, reverse=True)
    top_events = []
    for event in events[: args.top]:
        top_events.append(
            {
                "operator": event.key,
                "count": int(event.count),
                "self_device_time_us": _device_time_us(event),
                "device_time_us": float(
                    getattr(
                        event,
                        "device_time_total",
                        getattr(event, "cuda_time_total", 0.0),
                    )
                ),
                "self_cpu_time_us": float(event.self_cpu_time_total),
                "cpu_time_us": float(event.cpu_time_total),
                "input_shapes": event.input_shapes,
            }
        )

    record = {
        **environment_record(loaded.device),
        "model": {
            "name": loaded.model_name,
            "requested_revision": loaded.requested_revision,
            "resolved_revision": loaded.resolved_revision,
            "dtype": str(loaded.dtype),
        },
        "workload": {
            "prompt_tokens": len(loaded.tokenizer(args.prompt).input_ids),
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
        },
        "generation": {
            "token_ids": result.token_ids,
            "metrics": result.metrics.as_dict(),
        },
        "top_operators_by_self_device_time": top_events,
        "chrome_trace": str(args.trace) if args.trace is not None else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
