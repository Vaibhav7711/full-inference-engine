"""Interleaved Ada paged-decode sweep for register and grid pressure.

Compares the established per-head launch with alternative tiles and split-K at
fixed Qwen3 geometry. This is an isolated kernel experiment; promotion to a
serving policy still requires the full-engine token and latency A/B gates.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import torch

from benchmarks.common import environment_record, git_record
from benchmarks.kernels.paged_decode_regime_sweep import _build_pool, _parse_configs, _parse_ints
from engine.kernels.paged_decode_batched import paged_decode_batched
from engine.kernels.paged_decode_split_k import paged_decode_split_k


def _time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", default="256,1024,4096")
    parser.add_argument("--batches", default="1,8")
    parser.add_argument("--per-head-configs", default="64x2,128x4")
    parser.add_argument("--split-configs", default="64x2,128x4")
    parser.add_argument("--splits", default="2,4,8")
    parser.add_argument("--prefetch-value", action="store_true",
                        help="also measure the split-K variant that loads V before QK")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--max-error", type=float, default=0.002)
    parser.add_argument("--output", default="results/rtx4060/paged_decode_occupancy_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.warmup < 0 or args.rounds <= 0 or args.max_error <= 0:
        parser.error("warmup must be nonnegative; rounds and max-error must be positive")
    if torch.cuda.get_device_capability() != (8, 9):
        parser.error("this sweep is for the RTX 4060 sm_89")
    per_head_configs = _parse_configs(args.per_head_configs)
    split_configs = _parse_configs(args.split_configs)
    split_counts = _parse_ints(args.splits)
    if any(count & (count - 1) for count in split_counts):
        parser.error("split counts must be powers of two")

    result = {
        "config": vars(args), "environment": environment_record(),
        "git": git_record(), "device": torch.cuda.get_device_name(), "rows": [],
    }
    rng = random.Random(4060)
    for seq_len in _parse_ints(args.seq_lens):
        for batch in _parse_ints(args.batches):
            tensors = _build_pool(batch, seq_len, block_size=16, kv_heads=8, head_dim=128)
            reference = paged_decode_batched(*tensors, block_n=128, num_warps=4)
            arms = []
            for block_n, warps in per_head_configs:
                name = f"per_head_{block_n}x{warps}"
                arms.append((name, lambda bn=block_n, nw=warps: paged_decode_batched(
                    *tensors, block_n=bn, num_warps=nw,
                )))
            for block_n, warps in split_configs:
                for splits in split_counts:
                    if seq_len < splits * block_n:
                        continue
                    name = f"split{splits}_{block_n}x{warps}"
                    arms.append((name, lambda bn=block_n, nw=warps, sp=splits: paged_decode_split_k(
                        *tensors, block_n=bn, num_warps=nw,
                        max_sequence_length=seq_len, splits=sp,
                    )))
                    if args.prefetch_value:
                        arms.append((f"{name}_prefetch_v", lambda bn=block_n, nw=warps, sp=splits: paged_decode_split_k(
                            *tensors, block_n=bn, num_warps=nw,
                            max_sequence_length=seq_len, splits=sp, prefetch_value=True,
                        )))

            timings = {name: [] for name, _ in arms}
            errors = {}
            for name, fn in arms:
                actual = fn()
                max_abs = float((reference - actual).abs().max().item())
                errors[name] = max_abs
                if not torch.isfinite(actual).all() or max_abs > args.max_error:
                    raise RuntimeError(f"{name} failed parity at batch={batch}, context={seq_len}: {max_abs}")
                for _ in range(args.warmup):
                    fn()
            torch.cuda.synchronize()

            for _ in range(args.rounds):
                order = arms[:]
                rng.shuffle(order)
                for name, fn in order:
                    timings[name].append(_time_ms(fn))

            baseline = statistics.median(timings["per_head_128x4"])
            rows = []
            for name, _ in arms:
                samples = timings[name]
                median = statistics.median(samples)
                row = {
                    "batch": batch, "seq_len": seq_len, "arm": name,
                    "median_ms": median, "min_ms": min(samples), "max_ms": max(samples),
                    "speedup_vs_128x4": baseline / median,
                    "max_abs_diff": errors[name],
                }
                result["rows"].append(row)
                rows.append(row)
            print(f"batch={batch} context={seq_len} baseline={baseline:.4f} ms")
            for row in sorted(rows, key=lambda item: item["median_ms"]):
                print(f"  {row['arm']:<20} {row['median_ms']:.4f} ms "
                      f"{row['speedup_vs_128x4']:.3f}x err={row['max_abs_diff']:.6f}")

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
