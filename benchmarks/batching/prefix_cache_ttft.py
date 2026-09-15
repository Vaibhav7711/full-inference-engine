"""Same-engine miss/hit TTFT benchmark for block-aligned paged prefix reuse."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch


def _timed_generate(engine, prompt: str) -> tuple[float, list[int]]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = engine.generate([prompt], max_new_tokens=1)[0]
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt-repeats", type=int, default=32)
    parser.add_argument("--hit-repeats", type=int, default=5)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--prefix-cache-blocks", type=int, default=256)
    parser.add_argument("--output", default="results/prefix_cache_ttft.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    if args.hit_repeats <= 0:
        raise SystemExit("--hit-repeats must be positive")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    engine = ContinuousBatchingEngine(
        model, tokenizer, "cuda", num_blocks=args.num_blocks, block_size=16,
        max_active=1, prefix_cache_blocks=args.prefix_cache_blocks,
    )
    unit = (
        "You are a reliable assistant using a shared system prompt. Explain GPU inference "
        "with precise attention to memory, scheduling, correctness, and latency. "
    )
    prompt = unit * args.prompt_repeats + "Question: What is prefix caching?"
    warm_prompt = unit.replace("reliable", "methodical") * args.prompt_repeats + "Warmup."

    # Warm the same long-shape SDPA path, then reset allocator/cache state.
    _timed_generate(engine, warm_prompt)
    engine.reset()
    miss_ms, reference = _timed_generate(engine, prompt)
    # First hit compiles the residual paged-prefill shape and is excluded from timing.
    _, warm_hit = _timed_generate(engine, prompt)
    if warm_hit != reference:
        raise RuntimeError("prefix hit changed greedy output tokens")

    hit_times = []
    for _ in range(args.hit_repeats):
        elapsed_ms, output = _timed_generate(engine, prompt)
        if output != reference:
            raise RuntimeError("prefix hit changed greedy output tokens")
        hit_times.append(elapsed_ms)
    hit_median = statistics.median(hit_times)
    snapshot = engine.prefix_cache.snapshot()
    result = {
        "config": vars(args),
        "prompt_tokens": len(tokenizer(prompt, return_tensors="pt").input_ids[0]),
        "miss_ttft_ms": miss_ms,
        "hit_ttft_ms": hit_times,
        "hit_ttft_median_ms": hit_median,
        "speedup": miss_ms / hit_median,
        "prefix_cache": snapshot,
        "block_manager": engine.block_manager.snapshot(),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)

    print("\nPaged prefix-cache TTFT")
    print(f"Prompt tokens:       {result['prompt_tokens']}")
    print(f"Reusable hit tokens: {snapshot['hit_tokens'] // max(snapshot['hits'], 1)} average")
    print(f"Cache miss TTFT:     {miss_ms:.2f} ms")
    print(f"Cache hit TTFT p50:  {hit_median:.2f} ms")
    print(f"TTFT speedup:        {result['speedup']:.2f}x")
    print(f"Cache blocks:        {snapshot['cached_blocks']}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
