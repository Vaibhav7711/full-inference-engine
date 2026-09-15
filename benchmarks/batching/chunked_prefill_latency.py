"""Measure decode latency isolation while a long prompt is prefilling."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from engine.runtime import GenerationRequest


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _request(tokenizer, request_id: str, prompt: str, max_new_tokens: int):
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    return GenerationRequest(
        request_id, len(ids), max_new_tokens, prompt_token_ids=ids
    )


def _scenario(engine, tokenizer, long_prompt: str, chunk_size: int) -> dict:
    engine.reset()
    engine.prefill_chunk_size = chunk_size
    engine.max_prefill_tokens_per_iteration = chunk_size
    anchor = _request(tokenizer, "decode-anchor", "Count upward carefully: 1, 2, 3,", 24)
    long_request = _request(tokenizer, "long-prefill", long_prompt, 1)
    engine.submit(anchor)
    engine.step()  # establish a decoding request before the long prompt arrives
    engine.submit(long_request)
    while engine.has_unfinished_requests:
        engine.step()
    torch.cuda.synchronize()

    intervals_ms = [
        (right - left) / 1_000_000
        for left, right in zip(anchor.token_timestamps_ns, anchor.token_timestamps_ns[1:])
    ]
    return {
        "chunk_size": chunk_size,
        "long_prompt_tokens": long_request.prompt_token_count,
        "anchor_tokens": len(anchor.output_token_ids),
        "anchor_itl_p50_ms": statistics.median(intervals_ms) if intervals_ms else 0.0,
        "anchor_itl_p95_ms": _percentile(intervals_ms, 0.95),
        "anchor_itl_max_ms": max(intervals_ms, default=0.0),
        "long_prompt_ttft_ms": long_request.time_to_first_token_ms(),
        "free_blocks_after": engine.block_manager.snapshot()["free_blocks"],
    }


def _warm_chunked_path(engine, tokenizer, long_prompt: str, chunk_size: int) -> float:
    """Compile the real paged-prefill path without contaminating measured state."""
    engine.reset()
    engine.prefill_chunk_size = chunk_size
    engine.max_prefill_tokens_per_iteration = chunk_size
    request = _request(tokenizer, "chunk-warmup", long_prompt, 1)
    engine.submit(request)
    torch.cuda.synchronize()
    started = time.perf_counter()
    engine.step()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    if request.request_id in engine.scheduler.active:
        engine.cancel(request.request_id, reason="WARMUP_COMPLETE")
    engine.reset()
    return elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--long-prompt-repeats", type=int, default=40)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--output", default="results/chunked_prefill_latency.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    engine = ContinuousBatchingEngine(
        model, tokenizer, "cuda", num_blocks=args.num_blocks, block_size=16,
        max_active=2, prefill_chunk_size=args.chunk_size,
        max_prefill_tokens_per_iteration=args.chunk_size,
    )

    # Warm the short SDPA/decode path, then explicitly compile the paged chunk path.
    engine.generate(["Warm up the inference engine."], max_new_tokens=4)
    long_prompt = (
        "Describe a robust GPU inference runtime including scheduling, paged memory, "
        "attention kernels, cancellation, observability, and failure handling. "
        * args.long_prompt_repeats
    )
    long_tokens = len(tokenizer(long_prompt, return_tensors="pt").input_ids[0])
    chunk_compile_warmup_ms = _warm_chunked_path(
        engine, tokenizer, long_prompt, args.chunk_size
    )
    unchunked = _scenario(engine, tokenizer, long_prompt, long_tokens)
    chunked = _scenario(engine, tokenizer, long_prompt, args.chunk_size)
    result = {
        "config": vars(args),
        "chunk_compile_warmup_ms": chunk_compile_warmup_ms,
        "unchunked": unchunked,
        "chunked": chunked,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)

    print("\nChunked-prefill decode latency isolation")
    print(f"{'mode':<12} {'chunk':>8} {'ITL p50':>12} {'ITL p95':>12} {'ITL max':>12} {'long TTFT':>12}")
    for name, row in (("unchunked", unchunked), ("chunked", chunked)):
        print(
            f"{name:<12} {row['chunk_size']:>8} {row['anchor_itl_p50_ms']:>11.2f} "
            f"{row['anchor_itl_p95_ms']:>11.2f} {row['anchor_itl_max_ms']:>11.2f} "
            f"{row['long_prompt_ttft_ms']:>11.2f}"
        )
    print(f"\nSaved -> {args.output}")
    if chunked["anchor_itl_max_ms"] < unchunked["anchor_itl_max_ms"]:
        reduction = 100 * (
            1 - chunked["anchor_itl_max_ms"] / unchunked["anchor_itl_max_ms"]
        )
        print(f"Interpretation: chunking reduced worst-case decode ITL by {reduction:.1f}%.")
    else:
        regression = 100 * (
            chunked["anchor_itl_max_ms"] / unchunked["anchor_itl_max_ms"] - 1
        )
        print(f"Interpretation: chunking regressed worst-case decode ITL by {regression:.1f}%.")
    print("Long-prompt TTFT and typical decoder ITL are explicit tradeoffs above.")


if __name__ == "__main__":
    main()
