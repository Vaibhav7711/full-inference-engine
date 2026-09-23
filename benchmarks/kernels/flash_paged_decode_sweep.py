"""Compare FA2 paged decode against the engine kernel and tune ``num_splits``.

This is intentionally an attention-only benchmark. Full-engine latency is dominated by
reading model weights, so it can hide whether Flash itself won or lost at a particular
batch/context operating point.

    python -m benchmarks.kernels.flash_paged_decode_sweep \
        --output results/rtx4060/flash_paged_decode_sweep.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def _ints(value: str, *, allow_zero: bool = False) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    floor = 0 if allow_zero else 1
    if not values or any(item < floor for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _time(function, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _tensors(batch: int, context: int, *, page_size: int, dtype: torch.dtype):
    q_heads, kv_heads, head_dim = 16, 8, 128
    pages_per_row = (context + page_size - 1) // page_size
    total_pages = batch * pages_per_row
    key = torch.randn(total_pages, page_size, kv_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    query = torch.randn(batch, q_heads, 1, head_dim, device="cuda", dtype=dtype)
    table = torch.arange(total_pages - 1, -1, -1, device="cuda", dtype=torch.int32).view(
        batch, pages_per_row,
    )
    lengths = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    return query, key, value, table, lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", default="128,512,2048,4096")
    parser.add_argument("--batches", default="1,4,8,16")
    parser.add_argument("--splits", default="0,1,2,4,8")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", default="results/rtx4060/flash_paged_decode_sweep.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    from engine.kernels.flash_paged import flash_paged_decode, supports
    from engine.kernels.paged_decode_batched import paged_decode_batched

    reason = supports(args.page_size, 128, torch.float16)
    if reason:
        raise SystemExit(reason)

    contexts = _ints(args.contexts)
    batches = _ints(args.batches)
    splits = _ints(args.splits, allow_zero=True)
    result = {
        "config": vars(args),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": [],
    }
    print(f"{'context':>7} {'batch':>5} {'kernel':>12} {'splits':>6} "
          f"{'median_ms':>10} {'vs_triton':>10} {'max_error':>10}")
    for context in contexts:
        for batch in batches:
            tensors = _tensors(batch, context, page_size=args.page_size, dtype=torch.float16)
            baselines = []
            baseline_outputs = {}
            for block_n in (64, 128):
                def run_triton(block_n=block_n):
                    return paged_decode_batched(*tensors, block_n=block_n, num_warps=4)
                elapsed = _time(run_triton, args.warmup, args.repeats)
                baselines.append((elapsed, block_n))
                baseline_outputs[block_n] = run_triton()
            baseline_ms, block_n = min(baselines)
            reference = baseline_outputs[block_n]
            result["rows"].append({
                "context": context, "batch": batch, "kernel": "triton_per_head",
                "block_n": block_n, "median_ms": baseline_ms,
                "tokens_per_second": batch * 1000 / baseline_ms,
            })
            print(f"{context:>7} {batch:>5} {'triton':>12} {'-':>6} "
                  f"{baseline_ms:>10.4f} {1.0:>10.3f} {'-':>10}")
            for num_splits in splits:
                def run_flash(num_splits=num_splits):
                    return flash_paged_decode(*tensors, num_splits=num_splits)
                output = run_flash()
                max_error = (output.float() - reference.float()).abs().max().item()
                elapsed = _time(run_flash, args.warmup, args.repeats)
                row = {
                    "context": context, "batch": batch, "kernel": "flash",
                    "num_splits": num_splits, "median_ms": elapsed,
                    "ratio_vs_triton": elapsed / baseline_ms,
                    "tokens_per_second": batch * 1000 / elapsed,
                    "max_abs_error_vs_triton": max_error,
                }
                result["rows"].append(row)
                print(f"{context:>7} {batch:>5} {'flash':>12} {num_splits:>6} "
                      f"{elapsed:>10.4f} {row['ratio_vs_triton']:>10.3f} {max_error:>10.5f}")
            del tensors, reference, baseline_outputs
            torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
