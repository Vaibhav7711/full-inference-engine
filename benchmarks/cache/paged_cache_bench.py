"""Compare DynamicCache and the reference PagedCache with identical decode loops."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
from transformers.cache_utils import DynamicCache

from benchmarks.common import assert_repeatable_tokens, environment_record
from engine.cache.paged_cache import PagedCache
from engine.model import load_model


def _eos_ids(model, tokenizer) -> set[int]:
    configured = model.generation_config.eos_token_id
    if configured is None:
        configured = tokenizer.eos_token_id
    if isinstance(configured, int):
        return {configured}
    return set(configured or ())


@torch.inference_mode()
def _run_once(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
    cache_factory: Callable[[], object],
) -> dict[str, object]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = cache_factory()
    eos = _eos_ids(model, tokenizer)
    generated: list[int] = []

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    output = model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
    next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for step in range(max_new_tokens):
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id in eos or step == max_new_tokens - 1:
            break
        output = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        next_token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    end.record()
    end.synchronize()
    snapshot = cache.snapshot() if hasattr(cache, "snapshot") else None
    return {
        "token_ids": generated,
        "elapsed_ms": start.elapsed_time(end),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "cache_snapshot": snapshot,
    }


def _measure(
    run: Callable[[], dict[str, object]], warmup_runs: int, measured_runs: int
) -> dict[str, object]:
    for _ in range(warmup_runs):
        run()
    measured = [run() for _ in range(measured_runs)]
    token_runs = [item["token_ids"] for item in measured]
    assert_repeatable_tokens(token_runs)
    times = [float(item["elapsed_ms"]) for item in measured]
    output_tokens = len(token_runs[0])
    median_ms = statistics.median(times)
    return {
        "token_ids": token_runs[0],
        "output_tokens": output_tokens,
        "elapsed_ms": {
            "mean": statistics.mean(times),
            "median": median_ms,
            "min": min(times),
            "max": max(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        },
        "tokens_per_second_from_median": output_tokens / (median_ms / 1000),
        "peak_allocated_bytes_max": max(int(item["peak_allocated_bytes"]) for item in measured),
        "peak_reserved_bytes_max": max(int(item["peak_reserved_bytes"]) for item in measured),
        "cache_snapshot": measured[-1]["cache_snapshot"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair DynamicCache/PagedCache comparison")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prompt", default="Explain paged KV caching in one sentence.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-sizes", default="8,16,32")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/paged_cache_bench.json"))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.warmup_runs < 0 or args.runs < 1:
        parser.error("invalid token count or run count")
    block_sizes = [int(value) for value in args.block_sizes.split(",")]
    if not block_sizes or any(value <= 0 for value in block_sizes):
        parser.error("block sizes must be positive")

    loaded = load_model(args.model, revision=args.revision, dtype=args.dtype)
    model, tokenizer, device = loaded.model, loaded.tokenizer, loaded.device
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"

    common = (model, tokenizer, device, args.prompt, args.max_new_tokens)
    baseline = _measure(
        lambda: _run_once(*common, cache_factory=DynamicCache),
        args.warmup_runs,
        args.runs,
    )
    variants: dict[str, object] = {"dynamic_cache": baseline}
    for block_size in block_sizes:
        result = _measure(
            lambda size=block_size: _run_once(
                *common,
                cache_factory=lambda: PagedCache(
                    num_layers=model.config.num_hidden_layers,
                    block_size_tokens=size,
                    initial_blocks=4,
                ),
            ),
            args.warmup_runs,
            args.runs,
        )
        if result["token_ids"] != baseline["token_ids"]:
            raise RuntimeError(f"PagedCache block size {block_size} changed generated tokens")
        baseline_ms = float(baseline["elapsed_ms"]["median"])
        result["latency_change_pct"] = (
            float(result["elapsed_ms"]["median"]) / baseline_ms - 1.0
        ) * 100
        variants[f"paged_block_{block_size}"] = result

    record = {
        **environment_record(device),
        "model": {
            "name": loaded.model_name,
            "requested_revision": loaded.requested_revision,
            "resolved_revision": loaded.resolved_revision,
            "dtype": str(loaded.dtype),
        },
        "workload": {
            "prompt_tokens": len(tokenizer(args.prompt).input_ids),
            "max_new_tokens": args.max_new_tokens,
            "warmup_runs": args.warmup_runs,
            "measured_runs": args.runs,
            "block_sizes": block_sizes,
        },
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
