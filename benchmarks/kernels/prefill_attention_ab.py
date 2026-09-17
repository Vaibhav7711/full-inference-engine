"""Measure the prefill attention kernel in isolation and fit its cost per token.

The engine sweep measures a whole step: attention, MLP, weight reads, the decode the step
also performs, and scheduling. That is the right number for deciding what to build, but the
wrong one for iterating on a kernel, because a 2x kernel win shows up as a fraction of a
step and is easy to lose in noise.

This runs the attention kernel alone across chunk sizes and prefix lengths, reports
effective KV bandwidth, and fits `ms = a + b * chunk_tokens` per kernel so the two can be
compared on the same axis the Phase B sweep used.

    python -m benchmarks.kernels.prefill_attention_ab
    python -m benchmarks.kernels.prefill_attention_ab --sweep-tiles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmarks.reliability.sweep import fit_line
from engine.kernels.paged_prefill import paged_prefill
from engine.kernels.tiled_paged_prefill import tiled_paged_prefill

HEAD_DIM = 128
BLOCK_SIZE = 16
LAYERS = 28          # Qwen3-0.6B: a step runs the attention kernel once per layer
BYTES = 2


def _time_ms(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _build(batch, q_heads, kv_heads, start, chunk, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total = start + chunk
    pages = -(-total // BLOCK_SIZE) + 2
    key_pages = torch.randn(pages * batch, BLOCK_SIZE, kv_heads, HEAD_DIM,
                            device="cuda", dtype=torch.float16, generator=generator)
    value_pages = torch.randn_like(key_pages)
    tables = torch.arange(pages * batch, device="cuda",
                          dtype=torch.int32).view(batch, pages)
    query = torch.randn(batch, q_heads, chunk, HEAD_DIM,
                        device="cuda", dtype=torch.float16, generator=generator)
    starts = torch.full((batch,), start, device="cuda", dtype=torch.int32)
    chunks = torch.full((batch,), chunk, device="cuda", dtype=torch.int32)
    return query, key_pages, value_pages, tables, starts, chunks


def ideal_kv_bytes(batch, q_heads, start, chunk, block_m):
    """KV a correctly tiled kernel must read: each Q tile loads its prefix once."""
    tiles = -(-chunk // block_m)
    entries = sum(start + min((i + 1) * block_m, chunk) for i in range(tiles))
    return entries * batch * q_heads * 2 * HEAD_DIM * BYTES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--prefix", type=int, default=896)
    parser.add_argument("--chunks", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--sweep-tiles", action="store_true")
    parser.add_argument("--bandwidth-gbps", type=float, default=258.8,
                        help="measured achieved bandwidth, from roofline.py")
    parser.add_argument("--out", default="results/prefill_attention_ab.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("requires CUDA")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"batch={args.batch} q_heads={args.q_heads} kv_heads={args.kv_heads} "
          f"prefix={args.prefix}\n")

    payload: dict = {"config": vars(args), "points": [], "fits": {}}
    header = f"{'chunk':>6} {'old ms':>9} {'new ms':>9} {'speedup':>8} {'new GB/s':>9} {'ideal GB':>9}"
    print(header)
    print("-" * len(header))
    old_ms, new_ms = [], []
    for chunk in args.chunks:
        tensors = _build(args.batch, args.q_heads, args.kv_heads, args.prefix, chunk)
        old = _time_ms(lambda t=tensors: paged_prefill(*t))
        new = _time_ms(lambda t=tensors: tiled_paged_prefill(
            *t, block_m=args.block_m, block_n=args.block_n))
        ideal = ideal_kv_bytes(args.batch, args.q_heads, args.prefix, chunk, args.block_m)
        gbps = ideal / (new / 1000) / 1e9
        old_ms.append(old)
        new_ms.append(new)
        print(f"{chunk:>6} {old:>9.3f} {new:>9.3f} {old / new:>7.1f}x "
              f"{gbps:>9.1f} {ideal / 1e9:>9.4f}")
        payload["points"].append({
            "chunk": chunk, "old_ms": old, "new_ms": new, "speedup": old / new,
            "ideal_kv_bytes": ideal, "effective_gbps": gbps,
        })

    # Scale one layer's attention to a whole step, so b is on the Phase B axis.
    print(f"\nper-step cost model, attention only, x{LAYERS} layers, "
          f"batch {args.batch} rows sharing the step")
    for name, series in (("old", old_ms), ("new", new_ms)):
        fit = fit_line([float(c) for c in args.chunks],
                       [ms * LAYERS / args.batch for ms in series])
        payload["fits"][name] = fit
        print(f"  {name:4s} ms = {fit['a']:7.3f} + {fit['b']:.5f} * chunk_tokens   "
              f"(R^2={fit['r2']:.3f})  b = {fit['b'] / 0.018:5.1f}x compute floor")
    if payload["fits"]["new"]["b"] > 0:
        ratio = payload["fits"]["old"]["b"] / payload["fits"]["new"]["b"]
        print(f"\n  marginal cost per prefill token improved {ratio:.1f}x")
        target = 0.05
        verdict = ("MEETS" if payload["fits"]["new"]["b"] <= target else "MISSES")
        print(f"  {verdict} the Phase D2 target of b <= {target} ms/token")

    if args.sweep_tiles:
        print("\ntile shape sweep at chunk 128")
        tensors = _build(args.batch, args.q_heads, args.kv_heads, args.prefix, 128)
        best = None
        for block_m in (16, 32, 64, 128):
            for block_n in (32, 64, 128):
                for warps in (2, 4, 8):
                    try:
                        ms = _time_ms(lambda: tiled_paged_prefill(
                            *tensors, block_m=block_m, block_n=block_n, num_warps=warps))
                    except Exception as error:  # unsupported shape on this device
                        print(f"  BLOCK_M={block_m:3d} BLOCK_N={block_n:3d} warps={warps} "
                              f"-> {type(error).__name__}")
                        continue
                    marker = ""
                    if best is None or ms < best[0]:
                        best, marker = (ms, block_m, block_n, warps), "  <-- best"
                    print(f"  BLOCK_M={block_m:3d} BLOCK_N={block_n:3d} warps={warps} "
                          f"-> {ms:7.3f} ms{marker}")
        if best:
            print(f"\n  best: BLOCK_M={best[1]} BLOCK_N={best[2]} num_warps={best[3]} "
                  f"at {best[0]:.3f} ms")
            payload["best_tile"] = {"block_m": best[1], "block_n": best[2],
                                    "num_warps": best[3], "ms": best[0]}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
