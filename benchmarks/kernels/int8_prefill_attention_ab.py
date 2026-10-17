"""Offline search: INT8 paged prefill attention kernels against the FP16 paths.

The mixed-arrival INT8-KV A/B found INT8 worse than FP16 whenever memory was not binding
(ITL p99 +72%, TTFT p50 +32% on the chat profile). The suspected cause is the INT8
chunked-prefill attention: one program per (row, head, query token), scoring on CUDA
cores, while the FP16 arm uses SDPA. This measures every candidate per cell:

- fp16_sdpa: the FP16 engine default (gather + torch SDPA)
- fp16_tiled: the Triton FlashAttention-structured FP16 kernel
- int8_per_token: the current INT8 kernel
- int8_tiled[BM,BN,warps,stages]: the tensor-core INT8 tile, one entry per config

Every candidate is validated against the exact-dequantization FP16 reference before it is
timed; failures (compile, resources, numerics) are recorded, not fatal. Candidates are
timed in shuffled order per round with CUDA events; compile time is measured separately.

    python -m benchmarks.kernels.int8_prefill_attention_ab --out results/rtx4060/int8_prefill_search.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from engine.kernels.int8_paged_kv import paged_prefill_int8, tiled_paged_prefill_int8
from engine.kernels.sdpa_prefill import sdpa_paged_prefill
from engine.kernels.tiled_paged_prefill import tiled_paged_prefill

TILE_CONFIGS = [
    (16, 64, 4, 2), (16, 128, 4, 2), (32, 32, 4, 2), (32, 64, 4, 2), (32, 128, 4, 2),
    (64, 32, 4, 2), (64, 64, 4, 2), (64, 64, 4, 3), (64, 64, 8, 2), (64, 128, 8, 2),
    (128, 64, 8, 2), (128, 32, 8, 2),
]


def build(rows, q_heads, kv_heads, start, chunk, block_size=16, dim=128, seed=0, smoothing=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = start + chunk
    blocks_per_row = -(-total // block_size)
    num_blocks = rows * blocks_per_row + 3
    permutation = torch.randperm(num_blocks, device="cuda", generator=generator)[:rows * blocks_per_row]
    tables = permutation.view(rows, blocks_per_row).to(torch.int32)
    key_int8 = torch.randint(-127, 128, (num_blocks, block_size, kv_heads, dim), device="cuda",
                             dtype=torch.int8, generator=generator)
    value_int8 = torch.randint(-127, 128, key_int8.shape, device="cuda", dtype=torch.int8,
                               generator=generator)
    key_scale = (torch.rand(key_int8.shape[:-1], device="cuda", generator=generator) * 0.02 + 0.005).half()
    value_scale = (torch.rand(key_int8.shape[:-1], device="cuda", generator=generator) * 0.02 + 0.005).half()
    query = torch.randn(rows, q_heads, chunk, dim, device="cuda", dtype=torch.float16, generator=generator)
    smooth = None
    if smoothing:
        smooth = (torch.rand(kv_heads, dim, device="cuda", generator=generator) * 4 + 0.5)
        half = dim // 2
        smooth[:, half:] = smooth[:, :half]
        smooth = smooth.contiguous()
    starts = torch.full((rows,), start, device="cuda", dtype=torch.int32)
    lengths = torch.full((rows,), chunk, device="cuda", dtype=torch.int32)
    # Exact dequantization reference: what an INT8 kernel must reproduce up to rounding.
    key_ref = (key_int8.float() * key_scale.float()[..., None])
    if smooth is not None:
        key_ref = key_ref * smooth[None, None]
    key_ref = key_ref.half()
    value_ref = (value_int8.float() * value_scale.float()[..., None]).half()
    return dict(query=query, key_int8=key_int8, value_int8=value_int8, key_scale=key_scale,
                value_scale=value_scale, tables=tables, starts=starts, lengths=lengths,
                key_ref=key_ref, value_ref=value_ref, smooth=smooth, total=total)


def candidates(case, family: str = "int8"):
    c = case
    chunk = c["query"].shape[2]
    if family == "fp16":
        # SDPA as the engine runs it: the mask/page index is built once per step and
        # reused by every layer, so one warm cache is shared across timed calls.
        shared_cache: dict = {}
        out = {
            "fp16_sdpa_cached": lambda: sdpa_paged_prefill(
                c["query"], c["key_ref"], c["value_ref"], c["tables"], c["starts"], c["lengths"],
                total_len=c["total"], cache=shared_cache),
            "fp16_tiled": lambda: tiled_paged_prefill(
                c["query"], c["key_ref"], c["value_ref"], c["tables"], c["starts"], c["lengths"]),
        }
        for bm, bn, warps, stages in TILE_CONFIGS:
            if bm > max(16, triton_next_pow2(chunk)):
                continue
            out[f"fp16_tiled_{bm}x{bn}_w{warps}_s{stages}"] = (
                lambda bm=bm, bn=bn, warps=warps, stages=stages: tiled_paged_prefill(
                    c["query"], c["key_ref"], c["value_ref"], c["tables"], c["starts"],
                    c["lengths"], block_m=bm, block_n=bn, num_warps=warps, num_stages=stages))
        return out
    out = {
        "fp16_sdpa": lambda: sdpa_paged_prefill(
            c["query"], c["key_ref"], c["value_ref"], c["tables"], c["starts"], c["lengths"],
            total_len=c["total"], cache={}),
        "fp16_tiled": lambda: tiled_paged_prefill(
            c["query"], c["key_ref"], c["value_ref"], c["tables"], c["starts"], c["lengths"]),
        "int8_per_token": lambda: paged_prefill_int8(
            c["query"], c["key_int8"], c["value_int8"], c["key_scale"], c["value_scale"],
            c["tables"], c["starts"], c["lengths"], key_smoothing=c["smooth"]),
    }
    for bm, bn, warps, stages in TILE_CONFIGS:
        if bm > max(16, triton_next_pow2(chunk)):
            continue
        out[f"int8_tiled_{bm}x{bn}_w{warps}_s{stages}"] = (
            lambda bm=bm, bn=bn, warps=warps, stages=stages: tiled_paged_prefill_int8(
                c["query"], c["key_int8"], c["value_int8"], c["key_scale"], c["value_scale"],
                c["tables"], c["starts"], c["lengths"], key_smoothing=c["smooth"],
                block_m=bm, block_n=bn, num_warps=warps, num_stages=stages))
    return out


def triton_next_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def time_events(fn, iters: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--chunks", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--starts", type=int, nargs="+", default=[0, 512, 1024, 2048, 4096])
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--smoothing", action="store_true", default=True)
    parser.add_argument("--max-rel-error", type=float, default=2e-2)
    parser.add_argument("--family", choices=("int8", "fp16"), default="int8",
                        help="int8: INT8 tile search plus FP16 baselines; fp16: FP16 tile search")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rng = random.Random(0)
    compile_ms: dict[str, float] = {}
    cells = []
    for rows in args.rows:
        for chunk in args.chunks:
            for start in args.starts:
                case = build(rows, args.q_heads, args.kv_heads, start, chunk,
                             smoothing=args.smoothing and args.family == "int8")
                reference = None
                fns = candidates(case, args.family)
                status, results = {}, {}
                # Reference: FP16 tiled is not the decision-maker, SDPA in fp32 math is.
                reference = sdpa_paged_prefill(
                    case["query"].float().half(), case["key_ref"], case["value_ref"], case["tables"],
                    case["starts"], case["lengths"], total_len=case["total"], cache={}).float()
                scale_ref = reference.abs().mean().clamp_min(1e-6)
                valid = {}
                for name, fn in fns.items():
                    try:
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        out = fn()
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter() - started) * 1000
                        key = name if name.startswith("int8_tiled") else name
                        compile_ms.setdefault(key, elapsed)
                        if not torch.isfinite(out).all():
                            status[name] = "non-finite output"
                            continue
                        error = float((out.float() - reference).abs().max() / scale_ref)
                        if error > args.max_rel_error:
                            status[name] = f"numerical error {error:.3e}"
                            continue
                        status[name] = f"ok (max err {error:.2e})"
                        valid[name] = fn
                    except Exception as error:  # compile/resources failures are data here
                        status[name] = f"failed: {type(error).__name__}: {str(error).splitlines()[0][:160]}"
                        torch.cuda.synchronize()
                for fn in valid.values():
                    time_events(fn, 3)
                samples = {name: [] for name in valid}
                for _ in range(args.rounds):
                    order = list(valid)
                    rng.shuffle(order)
                    for name in order:
                        samples[name].append(time_events(valid[name], args.iters))
                for name, values in samples.items():
                    median = statistics.median(values)
                    results[name] = {"median_ms": median, "min_ms": min(values), "max_ms": max(values),
                                     "spread": (max(values) - min(values)) / median}
                prefix = f"{args.family}_tiled_"
                tiled = {n: r for n, r in results.items() if n.startswith(prefix)}
                best = min(tiled, key=lambda n: tiled[n]["median_ms"]) if tiled else None
                cell = {"rows": rows, "chunk": chunk, "start": start, "results": results,
                        "status": status, f"best_{args.family}_tiled": best}
                cells.append(cell)
                line = "  ".join(f"{n.replace(prefix, 't')}={r['median_ms']*1000:.0f}us"
                                 for n, r in results.items() if not n.startswith(prefix) or n == best)
                print(f"rows={rows} chunk={chunk:3d} start={start:4d}: {line}", flush=True)
                del case, fns, valid
                torch.cuda.empty_cache()
    payload = {"benchmark": "int8_prefill_attention_ab", "gpu": torch.cuda.get_device_name(),
               "args": vars(args), "tile_configs": TILE_CONFIGS,
               "first_call_ms_including_compile": compile_ms, "cells": cells,
               "note": "per-layer attention call; multiply by num_layers for a step"}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
