"""Measure FA2 paged prefill against the engine's SDPA and Triton paths."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def _ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
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


def _build(batch: int, prefix: int, chunk: int, page_size: int):
    total = prefix + chunk
    pages_per_row = (total + page_size - 1) // page_size
    total_pages = batch * pages_per_row
    key = torch.randn(total_pages, page_size, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    query = torch.randn(batch, 16, chunk, 128, device="cuda", dtype=torch.float16)
    tables = torch.arange(total_pages - 1, -1, -1, device="cuda", dtype=torch.int32).view(
        batch, pages_per_row,
    )
    starts = torch.full((batch,), prefix, device="cuda", dtype=torch.int32)
    chunks = torch.full((batch,), chunk, device="cuda", dtype=torch.int32)
    return query, key, value, tables, starts, chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--prefixes", default="1,512,2048")
    parser.add_argument("--chunks", default="32,128,512")
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", default="results/rtx4060/flash_paged_prefill_sweep.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    from engine.kernels.flash_paged import flash_paged_prefill, supports
    from engine.kernels.sdpa_prefill import sdpa_paged_prefill
    from engine.kernels.tiled_paged_prefill import tiled_paged_prefill

    reason = supports(args.page_size, 128, torch.float16)
    if reason:
        raise SystemExit(reason)

    result = {"config": vars(args), "device": torch.cuda.get_device_name(0), "rows": []}
    print(f"{'prefix':>6} {'chunk':>5} {'batch':>5} {'kernel':>7} "
          f"{'median_ms':>10} {'vs_sdpa':>8} {'max_error':>10}")
    for prefix in _ints(args.prefixes):
        for chunk in _ints(args.chunks):
            for batch in _ints(args.batches):
                tensors = _build(batch, prefix, chunk, args.page_size)
                total = prefix + chunk
                calls = {
                    "sdpa": lambda: sdpa_paged_prefill(*tensors, total_len=total),
                    "tiled": lambda: tiled_paged_prefill(*tensors),
                    "flash": lambda: flash_paged_prefill(*tensors, total_len=total),
                }
                reference = calls["sdpa"]()
                timings = {}
                for name, call in calls.items():
                    output = call()
                    error = (output.float() - reference.float()).abs().max().item()
                    timings[name] = _time(call, args.warmup, args.repeats)
                    result["rows"].append({
                        "prefix": prefix, "chunk": chunk, "batch": batch,
                        "kernel": name, "median_ms": timings[name],
                        "ratio_vs_sdpa": timings[name] / timings["sdpa"] if "sdpa" in timings else 1.0,
                        "max_abs_error_vs_sdpa": error,
                    })
                    print(f"{prefix:>6} {chunk:>5} {batch:>5} {name:>7} "
                          f"{timings[name]:>10.4f} {timings[name] / timings['sdpa']:>8.3f} "
                          f"{error:>10.5f}")
                del tensors, reference
                torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
