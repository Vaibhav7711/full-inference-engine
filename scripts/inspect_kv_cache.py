"""Inspect estimated and physical KV-cache memory for the Stage 1 model path."""

from __future__ import annotations

import argparse
import json

import torch

from engine.cache import KVCacheGeometry, observed_kv_cache_bytes
from engine.model import ExplicitDecodeRunner, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure model KV-cache geometry")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Explain KV cache memory.")
    parser.add_argument("--budget-gib", type=float, default=4.0)
    args = parser.parse_args()
    if args.budget_gib <= 0:
        parser.error("--budget-gib must be positive")

    loaded = load_model(args.model)
    runner = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    inputs = loaded.tokenizer(args.prompt, return_tensors="pt").to(loaded.device)
    with torch.inference_mode():
        state = runner.prefill(inputs.input_ids, inputs.attention_mask)
    geometry = KVCacheGeometry.from_model_config(loaded.model.config, loaded.dtype)
    prompt_tokens = inputs.input_ids.shape[1]
    budget_bytes = int(args.budget_gib * 2**30)
    expected_bytes = geometry.bytes_for_tokens(prompt_tokens)
    observed_bytes = observed_kv_cache_bytes(state.past_key_values)
    record = {
        "model": args.model,
        "prompt_tokens": prompt_tokens,
        "geometry": geometry.as_dict(),
        "estimated_kv_bytes": expected_bytes,
        "observed_kv_bytes": observed_bytes,
        "observed_to_estimated_ratio": observed_bytes / expected_bytes,
        "budget_bytes": budget_bytes,
        "max_tokens_at_budget": geometry.max_tokens_for_budget(budget_bytes),
        "max_requests_at_prompt_length": geometry.max_concurrent_requests(budget_bytes, prompt_tokens),
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
