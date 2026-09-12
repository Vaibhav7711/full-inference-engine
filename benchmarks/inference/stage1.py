from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from engine.model import ExplicitDecodeRunner, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 explicit decode benchmark")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, default=Path("results/stage1.json"))
    args = parser.parse_args()
    loaded = load_model(args.model)
    result = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device).generate(
        args.prompt, max_new_tokens=args.max_new_tokens
    )
    properties = torch.cuda.get_device_properties(loaded.device)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": {"gpu": properties.name, "total_memory_bytes": properties.total_memory},
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__},
        "model": {"name": args.model, "dtype": str(loaded.dtype)},
        "workload": {"batch_size": 1, "concurrency": 1, "prompt_tokens": len(loaded.tokenizer(args.prompt).input_ids), "max_new_tokens": args.max_new_tokens, "sampling": "greedy"},
        "metrics": result.metrics.as_dict(),
        "output_tokens": len(result.token_ids),
        "output_tokens_per_second": len(result.token_ids) / (result.metrics.total_ms / 1000),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
