"""End-to-end long-context continuous decode A/B for FP16 versus INT8 paged KV."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from engine.runtime import GenerationRequest


def _prompt_ids(tokenizer, target_tokens: int) -> list[int]:
    fragment = tokenizer(
        "Paged KV attention serves many long requests by mapping logical token blocks to physical memory. ",
        return_tensors="pt",
    ).input_ids[0].tolist()
    return (fragment * ((target_tokens + len(fragment) - 1) // len(fragment)))[:target_tokens]


def _run_round(model, tokenizer, *, kv_cache_dtype: str, prompt_ids: list[int], batch_size: int,
               decode_steps: int, num_blocks: int) -> tuple[float, list[list[int]]]:
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    engine = ContinuousBatchingEngine(
        model, tokenizer, "cuda", num_blocks=num_blocks, block_size=16,
        max_active=batch_size, prefix_cache_blocks=0, kv_cache_dtype=kv_cache_dtype,
    )
    # Fixed work is essential for a decode throughput comparison.
    engine.eos_ids.clear()
    requests = [
        GenerationRequest(
            request_id=f"{kv_cache_dtype}-{index}", prompt_token_count=len(prompt_ids),
            max_new_tokens=decode_steps + 2, prompt_token_ids=prompt_ids,
        )
        for index in range(batch_size)
    ]
    for request in requests:
        assert engine.scheduler.submit(request)
    admitted = engine.scheduler.admit_available(max_active_requests=batch_size)
    assert len(admitted) == batch_size
    engine.prefill_batch(admitted)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(decode_steps):
        engine.decode_step(admitted)
    end.record()
    end.synchronize()
    return start.elapsed_time(end), [request.output_token_ids[:] for request in admitted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--output", default="results/int8_kv_continuous_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    if min(args.prompt_tokens, args.batch_size, args.decode_steps, args.rounds, args.num_blocks) <= 0:
        parser.error("all size arguments must be positive")
    required_blocks = args.batch_size * ((args.prompt_tokens + args.decode_steps + 1 + 15) // 16)
    if args.num_blocks < required_blocks:
        parser.error(f"--num-blocks must be at least {required_blocks} for this fixed workload")

    from engine.model import load_model
    loaded = load_model(args.model)
    prompt_ids = _prompt_ids(loaded.tokenizer, args.prompt_tokens)
    results: dict[str, object] = {"config": vars(args), "rows": {}}
    outputs: dict[str, list[list[int]]] = {}
    print(f"\nEnd-to-end INT8 KV continuous decode ({len(prompt_ids)}-token prompt, batch {args.batch_size})")
    for mode in ("fp16", "int8"):
        # One unrecorded round compiles the current storage-mode kernels and warms the
        # long-context model shape before collecting the reported samples.
        _run_round(loaded.model, loaded.tokenizer, kv_cache_dtype=mode, prompt_ids=prompt_ids,
                   batch_size=args.batch_size, decode_steps=4, num_blocks=args.num_blocks)
        samples, final_outputs = [], None
        for _ in range(args.rounds):
            elapsed_ms, final_outputs = _run_round(
                loaded.model, loaded.tokenizer, kv_cache_dtype=mode, prompt_ids=prompt_ids,
                batch_size=args.batch_size, decode_steps=args.decode_steps, num_blocks=args.num_blocks,
            )
            samples.append(elapsed_ms)
        median_ms = statistics.median(samples)
        rows = {
            "median_decode_ms": median_ms, "round_decode_ms": samples,
            "tokens_per_second": args.batch_size * args.decode_steps / (median_ms / 1000),
        }
        results["rows"][mode] = rows
        outputs[mode] = final_outputs
        print(f"  {mode:>4}: {median_ms:8.2f} ms  {rows['tokens_per_second']:7.1f} tok/s")

    fp16_tokens, int8_tokens = outputs["fp16"], outputs["int8"]
    compared = sum(len(row) for row in fp16_tokens)
    matches = sum(left == right for fp_row, int_row in zip(fp16_tokens, int8_tokens)
                  for left, right in zip(fp_row, int_row))
    speedup = results["rows"]["fp16"]["median_decode_ms"] / results["rows"]["int8"]["median_decode_ms"]
    results["int8_over_fp16_speedup"] = speedup
    results["greedy_token_agreement"] = matches / max(compared, 1)
    print(f"  INT8 speedup: {speedup:.2f}x")
    print(f"  Greedy token agreement: {results['greedy_token_agreement']:.1%}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
