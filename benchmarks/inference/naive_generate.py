"""Stock Hugging Face ``model.generate`` baseline for serving comparisons."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/naive_generate.json"))
    args = parser.parse_args()
    if min(args.max_new_tokens, args.runs) < 1 or args.warmup_runs < 0:
        parser.error("invalid run or token count")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        for _ in range(args.warmup_runs):
            model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        elapsed_s, output_tokens = [], []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
            torch.cuda.synchronize()
            elapsed_s.append(time.perf_counter() - started)
            output_tokens.append(int(output.shape[1] - inputs.input_ids.shape[1]))
    record = {
        "implementation": "stock_transformers_model_generate",
        "workload": {"batch_size": 1, "prompt_tokens": int(inputs.input_ids.shape[1]), "max_new_tokens": args.max_new_tokens, "warmup_runs": args.warmup_runs, "measured_runs": args.runs, "sampling": "greedy"},
        "summary": {"end_to_end_ms_p50": statistics.median(elapsed_s) * 1000.0, "end_to_end_ms_mean": statistics.mean(elapsed_s) * 1000.0, "aggregate_output_tokens_per_second": sum(output_tokens) / sum(elapsed_s)},
        "runs": [{"output_tokens": tokens, "end_to_end_ms": seconds * 1000.0} for seconds, tokens in zip(elapsed_s, output_tokens)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
