"""T4 calibration sweep for the live batched paged-decode attention kernel."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch


def _parse_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or any(item <= 0 for item in result):
        raise ValueError("expected a non-empty comma-separated list of positive integers")
    return result


def _parse_configs(value: str) -> list[tuple[int, int]]:
    configs = []
    for item in value.split(","):
        tile, warps = item.split("x")
        configs.append((int(tile), int(warps)))
    return configs


def _build_pool(batch: int, seq_len: int, *, block_size: int, kv_heads: int, head_dim: int):
    blocks_per_sequence = (seq_len + block_size - 1) // block_size
    total_blocks = batch * blocks_per_sequence + 8
    table = torch.empty((batch, blocks_per_sequence), dtype=torch.int32, device="cuda")
    # Descending physical IDs make the benchmark exercise non-contiguous addressing.
    table.copy_(torch.arange(batch * blocks_per_sequence - 1, -1, -1, device="cuda", dtype=torch.int32).view(batch, -1))
    key = torch.randn(total_blocks, block_size, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    query = torch.randn(batch, 16, 1, head_dim, device="cuda", dtype=torch.float16)
    lengths = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")
    return query, key, value, table, lengths


def _time(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="64,128,256,512,1024,2048")
    parser.add_argument("--batches", default="1,4,8,16")
    parser.add_argument("--configs", default="16x2,32x2,32x4,64x4,64x8,128x4,128x8")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="results/paged_decode_regime_sweep.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")

    from engine.kernels.paged_decode_batched import paged_decode_batched

    seq_lens = _parse_ints(args.seq_lens)
    batches = _parse_ints(args.batches)
    configs = _parse_configs(args.configs)
    result = {"config": vars(args), "rows": []}
    print("\nPaged decode regime sweep (Qwen3-0.6B geometry: H=16, KVH=8, D=128)")
    print(f"{'context':>8} {'batch':>6} {'tile':>6} {'warps':>6} {'median ms':>11} {'tok/s':>12}")
    for seq_len in seq_lens:
        for batch in batches:
            tensors = _build_pool(batch, seq_len, block_size=args.block_size, kv_heads=8, head_dim=128)
            for block_n, num_warps in configs:
                def run(block_n=block_n, num_warps=num_warps):
                    return paged_decode_batched(*tensors, block_n=block_n, num_warps=num_warps)
                try:
                    elapsed_ms = _time(run, warmup=args.warmup, repeats=args.repeats)
                except Exception as error:
                    row = {"seq_len": seq_len, "batch": batch, "block_n": block_n,
                           "num_warps": num_warps, "error": repr(error)}
                    result["rows"].append(row)
                    print(f"{seq_len:>8} {batch:>6} {block_n:>6} {num_warps:>6} {'ERROR':>11}")
                    continue
                row = {"seq_len": seq_len, "batch": batch, "block_n": block_n,
                       "num_warps": num_warps, "median_ms": elapsed_ms,
                       "tokens_per_second": batch / (elapsed_ms / 1000)}
                result["rows"].append(row)
                print(f"{seq_len:>8} {batch:>6} {block_n:>6} {num_warps:>6} "
                      f"{elapsed_ms:>11.4f} {row['tokens_per_second']:>12.0f}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
