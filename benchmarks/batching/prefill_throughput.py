"""Same-engine sequential versus batched prefill throughput."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from benchmarks.batching.continuous_throughput import PROMPT_POOL
from engine.runtime import GenerationRequest


def _make_prompts(count: int, repeat: int) -> list[str]:
    return [" ".join([PROMPT_POOL[index % len(PROMPT_POOL)]] * repeat) for index in range(count)]


def _admit(engine, tokenizer, prompts, run_id):
    requests = []
    for index, prompt in enumerate(prompts):
        token_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        request = GenerationRequest(
            request_id=f"{run_id}-{index}",
            prompt_token_count=len(token_ids),
            max_new_tokens=8,
            prompt_token_ids=token_ids,
        )
        requests.append(request)
        engine.scheduler.submit(request)
    admitted = engine.scheduler.admit_available(max_active_requests=len(requests))
    if admitted != requests:
        raise RuntimeError("KV pool could not admit the requested prefill batch")
    return requests


def _time(engine, tokenizer, prompts, *, batched: bool, run_id: str):
    engine.reset()
    requests = _admit(engine, tokenizer, prompts, run_id)
    torch.cuda.synchronize()
    started = time.perf_counter()
    if batched:
        engine.prefill_batch(requests)
    else:
        for request in requests:
            engine.prefill(request)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    outputs = [request.output_token_ids[0] for request in requests]
    tokens = sum(request.prompt_token_count for request in requests)
    return elapsed, tokens, outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", default="results/prefill_throughput.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    if args.concurrency < 1 or args.repeat < 1 or args.rounds < 1:
        raise SystemExit("concurrency, repeat, and rounds must be positive")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    ).eval()
    engine = ContinuousBatchingEngine(
        model, tokenizer, "cuda", num_blocks=args.num_blocks,
        block_size=args.block_size, max_active=args.concurrency,
    )
    prompts = _make_prompts(args.concurrency, args.repeat)

    # Compile both batch shapes before measurement.
    _time(engine, tokenizer, prompts, batched=False, run_id="warm-seq")
    _time(engine, tokenizer, prompts, batched=True, run_id="warm-batch")

    sequential, batched = [], []
    for round_index in range(args.rounds):
        # Alternate order to limit drift bias.
        order = [False, True] if round_index % 2 == 0 else [True, False]
        for use_batch in order:
            elapsed, tokens, outputs = _time(
                engine, tokenizer, prompts, batched=use_batch,
                run_id=f"r{round_index}-{'batch' if use_batch else 'seq'}",
            )
            (batched if use_batch else sequential).append(elapsed)
            if use_batch:
                batched_outputs = outputs
            else:
                sequential_outputs = outputs
    if batched_outputs != sequential_outputs:
        raise RuntimeError("batched prefill changed first generated tokens")

    seq_median = statistics.median(sequential)
    batch_median = statistics.median(batched)
    result = {
        "config": vars(args),
        "prompt_tokens": tokens,
        "sequential_median_ms": seq_median * 1000,
        "batched_median_ms": batch_median * 1000,
        "sequential_tok_s": tokens / seq_median,
        "batched_tok_s": tokens / batch_median,
        "speedup": seq_median / batch_median,
        "sequential_rounds_s": sequential,
        "batched_rounds_s": batched,
    }
    print("\nBatched Prefill A/B")
    print(f"prompt tokens: {tokens}")
    print(f"sequential: {result['sequential_median_ms']:.2f} ms  {result['sequential_tok_s']:.1f} tok/s")
    print(f"batched:    {result['batched_median_ms']:.2f} ms  {result['batched_tok_s']:.1f} tok/s")
    print(f"speedup:    {result['speedup']:.2f}x")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as file:
        json.dump(result, file, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
