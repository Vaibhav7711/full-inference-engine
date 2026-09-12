from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from engine.model import ExplicitDecodeRunner, load_model
from engine.quantization import model_storage_bytes, quantize_linear_modules


def timed_prefill(model: torch.nn.Module, inputs: object) -> tuple[torch.Tensor, float]:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False, return_dict=True).logits
    end.record(); end.synchronize()
    return logits, start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference INT8 weight-only experiment")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", type=Path, default=Path("results/int8_weight_only.json"))
    args = parser.parse_args()
    loaded = load_model(args.model)
    inputs = loaded.tokenizer(args.prompt, return_tensors="pt").to(loaded.device)
    baseline_logits, baseline_prefill_ms = timed_prefill(loaded.model, inputs)
    baseline_bytes = model_storage_bytes(loaded.model)

    quantized_model = copy.deepcopy(loaded.model).eval()
    replaced_modules = quantize_linear_modules(quantized_model)
    quantized_logits, quantized_prefill_ms = timed_prefill(quantized_model, inputs)
    quantized_bytes = model_storage_bytes(quantized_model)
    baseline_tokens = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device).generate(
        args.prompt, max_new_tokens=args.max_new_tokens
    ).token_ids
    quantized_tokens = ExplicitDecodeRunner(quantized_model, loaded.tokenizer, loaded.device).generate(
        args.prompt, max_new_tokens=args.max_new_tokens
    ).token_ids
    token_matches = sum(left == right for left, right in zip(baseline_tokens, quantized_tokens))
    record = {
        "model": args.model,
        "weight_storage_bytes": {"baseline": baseline_bytes, "int8_reference": quantized_bytes, "savings_fraction": 1 - (quantized_bytes / baseline_bytes)},
        "prefill_ms": {"baseline": baseline_prefill_ms, "int8_reference": quantized_prefill_ms},
        "logits": {"max_abs_error": float((baseline_logits.float() - quantized_logits.float()).abs().max()), "mean_abs_error": float((baseline_logits.float() - quantized_logits.float()).abs().mean())},
        "generation": {"max_new_tokens": args.max_new_tokens, "baseline_token_ids": baseline_tokens, "int8_token_ids": quantized_tokens, "token_agreement_fraction": token_matches / max(len(baseline_tokens), len(quantized_tokens))},
        "replaced_linear_modules": replaced_modules,
        "note": "Reference dequantize-then-F.linear path: memory/quality experiment, not an optimized INT8 throughput result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
