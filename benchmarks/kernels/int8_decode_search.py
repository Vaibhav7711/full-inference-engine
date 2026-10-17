"""Offline INT8 paged-decode search: per-head launch space and the paired-GQA kernel.

Each call reads the next of R per-layer pools (R chosen so the ring is >= 64 MB), so K/V
stream from DRAM as they do across a 28-layer forward; one layer's pool at short context
otherwise sits in the 24 MB L2 and times unrealistically. Every candidate is validated
against the current INT8 kernel; FP16 per-head over the same tokens is the speed reference.

    python -m benchmarks.kernels.int8_decode_search --out results/rtx4060/int8_decode_search.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics

import torch


def build_ring(batch, context, kv_heads=8, dim=128, block=16, min_bytes=64 << 20, max_ring=64, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    blocks_per_row = -(-(context + 1) // block)
    num_blocks = batch * blocks_per_row
    per_layer = num_blocks * block * kv_heads * (dim + 2) * 2
    ring = max(1, min(max_ring, -(-min_bytes // per_layer)))
    layers = []
    for _ in range(ring):
        kp = torch.randint(-127, 128, (num_blocks, block, kv_heads, dim), device="cuda", dtype=torch.int8,
                           generator=generator)
        vp = torch.randint(-127, 128, kp.shape, device="cuda", dtype=torch.int8, generator=generator)
        ks = (torch.rand(kp.shape[:-1], device="cuda", generator=generator) * 0.02 + 0.002).half()
        vs = (torch.rand(kp.shape[:-1], device="cuda", generator=generator) * 0.02 + 0.002).half()
        layers.append((kp, vp, ks, vs))
    tables = torch.randperm(num_blocks, device="cuda", generator=generator).view(batch, blocks_per_row).to(torch.int32)
    lengths = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    return layers, tables, lengths, per_layer


def time_ring(fn, ring_size, iters):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        fn(i % ring_size)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=56)
    parser.add_argument("--max-layer-bytes", type=float, default=600e6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from engine.kernels.int8_decode_tuned import paged_decode_int8_paired, paged_decode_int8_per_head
    from engine.kernels.int8_paged_kv import paged_decode_batched_int8
    from engine.kernels.paged_decode_batched import paged_decode_batched
    from engine.kernels.paged_decode_config import select_paged_decode_config

    rng = random.Random(0)
    cells = []
    for batch in args.batches:
        for context in args.contexts:
            if batch * context * 8 * 130 * 2 > args.max_layer_bytes:
                print(f"skip b{batch} ctx{context}: one layer exceeds the memory cap")
                continue
            layers, tables, lengths, per_layer = build_ring(batch, context)
            ring = len(layers)
            query = torch.randn(batch, 16, 1, 128, device="cuda", dtype=torch.float16)
            fp16_layers = [((kp.float() * ks.float()[..., None]).half(), (vp.float() * vs.float()[..., None]).half())
                           for kp, vp, ks, vs in layers[:max(1, min(ring, int(512e6 // (2 * per_layer))))]]
            block_n, warps = select_paged_decode_config(context, batch, "per_head")
            candidates = {
                "int8_current": lambda i: paged_decode_batched_int8(query, *layers[i], tables, lengths,
                                                                    block_n=block_n, num_warps=warps),
                "fp16_per_head_reference": lambda i: paged_decode_batched(
                    query, *fp16_layers[i % len(fp16_layers)], tables, lengths, block_n=block_n, num_warps=warps),
            }
            for bn, w, st in itertools.product((32, 64, 128, 256), (2, 4, 8), (1, 2, 3)):
                candidates[f"per_head_{bn}_w{w}_s{st}"] = (
                    lambda i, bn=bn, w=w, st=st: paged_decode_int8_per_head(
                        query, *layers[i], tables, lengths, block_n=bn, num_warps=w, num_stages=st))
            for bn, w, st in itertools.product((32, 64, 128), (2, 4, 8), (1, 2, 3)):
                candidates[f"paired_{bn}_w{w}_s{st}"] = (
                    lambda i, bn=bn, w=w, st=st: paged_decode_int8_paired(
                        query, *layers[i], tables, lengths, block_n=bn, num_warps=w, num_stages=st))
            reference = candidates["int8_current"](0).float()
            status, valid = {}, {}
            for key, fn in candidates.items():
                try:
                    out = fn(0).float()
                    error = float((out - reference).abs().max())
                    if key != "fp16_per_head_reference" and error > 2e-3:
                        status[key] = f"rejected: max abs error {error:.2e}"
                        continue
                    status[key] = f"ok {error:.1e}"
                    valid[key] = fn
                except Exception as error:
                    status[key] = f"failed: {type(error).__name__}: {str(error).splitlines()[0][:120]}"
            for fn in valid.values():
                time_ring(fn, ring, 4)
            samples = {key: [] for key in valid}
            for _ in range(args.rounds):
                order = list(valid)
                rng.shuffle(order)
                for key in order:
                    size = len(fp16_layers) if key == "fp16_per_head_reference" else ring
                    samples[key].append(time_ring(valid[key], size, args.iters))
            results = {k: {"median_ms": statistics.median(v), "spread": (max(v) - min(v)) / statistics.median(v)}
                       for k, v in samples.items()}
            tuned = {k: r for k, r in results.items() if k.startswith(("per_head_", "paired_"))}
            best = min(tuned, key=lambda k: tuned[k]["median_ms"])
            best_per_head = min((k for k in tuned if k.startswith("per_head_")), key=lambda k: tuned[k]["median_ms"])
            cur = results["int8_current"]["median_ms"]
            cells.append({"batch": batch, "context": context, "ring": ring, "results": results,
                          "status": status, "best": best, "best_per_head": best_per_head,
                          "current_config": [block_n, warps]})
            print(f"b{batch:2d} ctx{context:5d} ring{ring:2d}: current {cur*1000:6.1f}us  "
                  f"fp16 {results['fp16_per_head_reference']['median_ms']*1000:6.1f}us  "
                  f"best {best} {tuned[best]['median_ms']*1000:6.1f}us ({cur / tuned[best]['median_ms']:.2f}x)  "
                  f"best per-head {best_per_head} ({cur / tuned[best_per_head]['median_ms']:.2f}x)", flush=True)
            del layers, fp16_layers, candidates, valid
            torch.cuda.empty_cache()
    with open(args.out, "w") as handle:
        json.dump({"benchmark": "int8_decode_search", "gpu": torch.cuda.get_device_name(),
                   "args": vars(args), "methodology": "DRAM-streamed per-layer ring >= 64 MB",
                   "cells": cells}, handle, indent=2)
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
