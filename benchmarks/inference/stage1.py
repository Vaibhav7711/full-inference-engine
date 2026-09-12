from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from engine.model import ExplicitDecodeRunner, load_model
from engine.metrics import latency_summary_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 explicit decode benchmark")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/stage1.json"))
    args = parser.parse_args()
    if args.warmup_runs < 0 or args.runs < 1:
        parser.error("--warmup-runs must be non-negative and --runs must be at least 1")
    loaded = load_model(args.model, revision=args.revision)
    runner = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    for _ in range(args.warmup_runs):
        runner.generate(args.prompt, max_new_tokens=args.max_new_tokens)
    results = [runner.generate(args.prompt, max_new_tokens=args.max_new_tokens) for _ in range(args.runs)]
    properties = torch.cuda.get_device_properties(loaded.device)
    output_tokens = [len(result.token_ids) for result in results]
    total_ms = [result.metrics.total_ms for result in results]
    ttft_ms = [result.metrics.ttft_ms for result in results]
    decode_ms = [value for result in results for value in result.metrics.decode_ms]
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"gpu": properties.name, "total_memory_bytes": properties.total_memory},
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__},
        "model": {"name": args.model, "revision": args.revision, "dtype": str(loaded.dtype)},
        "workload": {"batch_size": 1, "concurrency": 1, "prompt_tokens": len(loaded.tokenizer(args.prompt).input_ids), "max_new_tokens": args.max_new_tokens, "sampling": "greedy", "warmup_runs": args.warmup_runs, "measured_runs": args.runs},
        "summary": {
            "ttft_ms": latency_summary_ms(ttft_ms),
            "end_to_end_ms": latency_summary_ms(total_ms),
            "decode_ms_per_token": latency_summary_ms(decode_ms) if decode_ms else None,
            "aggregate_output_tokens_per_second": sum(output_tokens) / (sum(total_ms) / 1000),
            "peak_allocated_bytes_max": max(result.metrics.peak_allocated_bytes for result in results),
            "peak_reserved_bytes_max": max(result.metrics.peak_reserved_bytes for result in results),
        },
        "runs": [
            {"output_tokens": len(result.token_ids), "metrics": result.metrics.as_dict()}
            for result in results
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
