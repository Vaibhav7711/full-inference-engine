"""Validate the RTX 4060 decode dispatcher against the established per-head regime."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from benchmarks.common import environment_record, git_record
from benchmarks.kernels.paged_decode_regime_sweep import _build_pool, _parse_ints


def _time(fn, warmup: int, repeats: int) -> float:
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
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", default="64,128,256,512,1024,2048,4096")
    parser.add_argument("--batches", default="1,4,8,16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default="results/rtx4060/paged_decode_ada_sweep.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    if torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("This dispatcher is calibrated only for sm_89")

    from engine.kernels.paged_decode_ada import paged_decode_ada
    from engine.kernels.paged_decode_batched import paged_decode_batched
    from engine.kernels.paged_decode_config import (
        select_ada_decode_config,
        select_paged_decode_config,
        use_ada_paired_gqa,
    )

    result = {
        "config": vars(args),
        "git": git_record(),
        "environment": environment_record(),
        "device": torch.cuda.get_device_name(),
        "rows": [],
    }
    print(f"{'context':>8} {'batch':>6} {'baseline':>11} {'ada':>11} "
          f"{'speedup':>9} {'path':>10} {'config':>8} {'max diff':>10}")
    for seq_len in _parse_ints(args.seq_lens):
        for batch in _parse_ints(args.batches):
            tensors = _build_pool(
                batch, seq_len, block_size=args.block_size, kv_heads=8, head_dim=128,
            )
            old_tile, old_warps = select_paged_decode_config(seq_len, batch)
            new_tile, new_warps = select_ada_decode_config(seq_len, batch)

            def baseline():
                return paged_decode_batched(
                    *tensors, block_n=old_tile, num_warps=old_warps,
                )

            def candidate():
                return paged_decode_ada(*tensors, max_sequence_length=seq_len)

            # Interleave once before timing so both compiled variants are resident.
            expected, actual = baseline(), candidate()
            max_diff = float((expected - actual).abs().max().item())
            baseline_ms = _time(baseline, args.warmup, args.repeats)
            ada_ms = _time(candidate, args.warmup, args.repeats)
            paired = use_ada_paired_gqa(seq_len, batch)
            row = {
                "seq_len": seq_len,
                "batch": batch,
                "baseline": {"block_n": old_tile, "num_warps": old_warps,
                             "median_ms": baseline_ms},
                "ada": {"block_n": new_tile, "num_warps": new_warps,
                        "path": "paired_gqa" if paired else "per_head",
                        "median_ms": ada_ms},
                "speedup": baseline_ms / ada_ms,
                "max_abs_diff": max_diff,
            }
            result["rows"].append(row)
            print(f"{seq_len:8d} {batch:6d} {baseline_ms:11.4f} {ada_ms:11.4f} "
                  f"{row['speedup']:9.3f} {row['ada']['path']:>10} "
                  f"{new_tile}x{new_warps:<4} {max_diff:10.6f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
