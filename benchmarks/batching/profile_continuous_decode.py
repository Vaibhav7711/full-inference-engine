"""Profile the optimized physical continuous-decode path after synchronization cleanup.

Prefill is deliberately completed before profiling so the report isolates batched
single-token decode. Profiler timings are diagnostic and must not be compared with the
unprofiled throughput benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import torch

from benchmarks.batching.continuous_throughput import PROMPT_POOL
from engine.runtime import GenerationRequest, RequestState


def _event_device_time_us(event) -> float:
    value = getattr(event, "self_device_time_total", None)
    if value is None:
        value = getattr(event, "self_cuda_time_total", 0.0)
    return float(value or 0.0)


def _rows(events, key) -> list[dict]:
    rows = [
        {
            "name": event.key,
            "calls": event.count,
            "self_gpu_us": round(_event_device_time_us(event), 1),
            "self_cpu_us": round(float(event.self_cpu_time_total), 1),
            "input_shapes": [list(shape) for shape in getattr(event, "input_shapes", [])],
            "stack": list(getattr(event, "stack", [])),
        }
        for event in events
    ]
    rows.sort(key=key, reverse=True)
    return rows


def _print_rows(title: str, rows: list[dict], metric: str, limit: int) -> None:
    print(f"\n{title}")
    print(f"{'operator':<58} {'calls':>8} {metric:>14}")
    print("-" * 82)
    for row in rows[:limit]:
        print(f"{row['name'][:58]:<58} {row['calls']:>8} {row[metric]:>14.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="group profiler events by input shape and print aten::cat diagnostics",
    )
    parser.add_argument(
        "--record-stacks",
        action="store_true",
        help="group profiler events by Python stack and print aten::cat call sites",
    )
    parser.add_argument("--output", default="results/profile_continuous_decode.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")

    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engine.batching.continuous_batching import ContinuousBatchingEngine

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    engine = ContinuousBatchingEngine(
        model,
        tokenizer,
        "cuda",
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_active=args.concurrency,
    )
    prompts = [PROMPT_POOL[i % len(PROMPT_POOL)] for i in range(args.concurrency)]

    print("Warmup...")
    engine.generate(prompts[: min(4, len(prompts))], max_new_tokens=8)
    engine.reset()

    requests = []
    for index, prompt in enumerate(prompts):
        token_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        request = GenerationRequest(
            request_id=f"profile-{index}",
            prompt_token_count=len(token_ids),
            max_new_tokens=args.max_new_tokens,
            prompt_token_ids=token_ids,
        )
        requests.append(request)
        engine.scheduler.submit(request)
    admitted = engine.scheduler.admit_available(max_active_requests=args.concurrency)
    if len(admitted) != len(requests):
        raise RuntimeError("profiling pool could not admit every request")
    for request in admitted:
        engine.prefill(request)

    torch.cuda.synchronize()
    print("Profiling decode only...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=args.record_shapes,
        with_stack=args.record_stacks,
    ) as profiler:
        while True:
            active = [
                request
                for request in engine.scheduler.active.values()
                if request.state is RequestState.DECODING
            ]
            if not active:
                break
            engine.decode_step(active)
    torch.cuda.synchronize()

    events = list(
        profiler.key_averages(
            group_by_input_shape=args.record_shapes,
            group_by_stack_n=8 if args.record_stacks else 0,
        )
    )
    gpu_rows = _rows(events, key=lambda row: row["self_gpu_us"])
    cpu_rows = _rows(events, key=lambda row: row["self_cpu_us"])
    _print_rows("Top GPU operators", gpu_rows, "self_gpu_us", args.top)
    _print_rows("Top CPU operators", cpu_rows, "self_cpu_us", args.top)
    if args.record_shapes or args.record_stacks:
        cat_rows = [row for row in gpu_rows if row["name"] == "aten::cat"]
        print("\naten::cat diagnostic groups")
        for row in cat_rows:
            print(
                f"calls={row['calls']} self_gpu_us={row['self_gpu_us']:.1f} "
                f"input_shapes={row['input_shapes']}"
            )
            if args.record_stacks:
                for frame in row["stack"]:
                    print(f"  {frame}")

    properties = torch.cuda.get_device_properties(0)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_gib": round(properties.total_memory / 2**30, 2),
        },
        "config": vars(args),
        "profile_scope": "decode_only",
        "generated_tokens": sum(len(request.output_token_ids) for request in requests),
        "top_gpu_operators": gpu_rows[: args.top],
        "top_cpu_operators": cpu_rows[: args.top],
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(result, file, indent=2)
    print(f"\nSaved -> {args.output}")
    print("Profiler timings are diagnostic; use continuous_throughput for performance claims.")


if __name__ == "__main__":
    main()
