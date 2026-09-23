"""Run isolated FP16 and optional BF16 Stage-1 measurements on a Colab T4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _run_stage1(args: argparse.Namespace, dtype: str, output: Path) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.inference.stage1",
        "--prompt",
        args.prompt,
        "--model",
        args.model,
        "--dtype",
        dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-runs",
        str(args.warmup_runs),
        "--runs",
        str(args.runs),
        "--output",
        str(output),
    ]
    if args.revision:
        command.extend(("--revision", args.revision))
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode != 0:
        return {
            "status": "failed",
            "returncode": completed.returncode,
            "stderr": completed.stderr[-8000:],
            "stdout": completed.stdout[-2000:],
        }
    return {"status": "ok", "record": json.loads(output.read_text())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated dtype baseline for a Colab T4")
    parser.add_argument("--prompt", default="Explain paged KV caching in one sentence.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--include-bfloat16", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/baseline_t4.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dtype_outputs = {
        "float16": args.output.with_name(f"{args.output.stem}_float16.json"),
    }
    if args.include_bfloat16:
        dtype_outputs["bfloat16"] = args.output.with_name(
            f"{args.output.stem}_bfloat16.json"
        )

    variants = {
        dtype: _run_stage1(args, dtype, path) for dtype, path in dtype_outputs.items()
    }
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "T4 dtype and explicit-decode baseline",
        "variants": variants,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
