from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from engine.batching import StaticBatchRunner
from engine.metrics import latency_summary_ms
from engine.model import load_model


def parse_batch_sizes(value: str) -> list[int]:
    try:
        sizes = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch sizes must be comma-separated integers") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 9 static batching benchmark")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-sizes", type=parse_batch_sizes, default=[1, 2, 4, 8])
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, default=Path("results/static_batching.json"))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.warmup_runs < 0 or args.runs < 1:
        parser.error("invalid generation or run count")
    loaded = load_model(args.model)
    runner = StaticBatchRunner(loaded.model, loaded.tokenizer, loaded.device)
    measurements: list[dict[str, object]] = []
    for batch_size in args.batch_sizes:
        prompts = [args.prompt] * batch_size
        for _ in range(args.warmup_runs):
            runner.generate(prompts, max_new_tokens=args.max_new_tokens)
        runs = [runner.generate(prompts, max_new_tokens=args.max_new_tokens) for _ in range(args.runs)]
        total_ms = [run.total_ms for run in runs]
        decode_steps = [item for run in runs for item in run.decode_ms]
        measurements.append({
            "batch_size": batch_size,
            "latency_ms": latency_summary_ms(total_ms),
            "aggregate_output_tokens_per_second": sum(run.output_tokens for run in runs) / (sum(total_ms) / 1000),
            "prefill_ms": latency_summary_ms([run.prefill_ms for run in runs]),
            "decode_ms_per_step": latency_summary_ms(decode_steps) if decode_steps else None,
        })
    props = torch.cuda.get_device_properties(loaded.device)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"gpu": props.name, "total_memory_bytes": props.total_memory},
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__},
        "model": {"name": args.model, "dtype": str(loaded.dtype)},
        "workload": {"prompt_tokens": len(loaded.tokenizer(args.prompt).input_ids), "max_new_tokens": args.max_new_tokens, "sampling": "greedy", "warmup_runs": args.warmup_runs, "measured_runs": args.runs},
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
