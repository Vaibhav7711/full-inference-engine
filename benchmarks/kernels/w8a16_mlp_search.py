"""Offline W8A16 MLP search on real Qwen3-1.7B gate/up/down weights.

Candidates per row count M: the engine's current kernel (16x128x64, 4 warps), cuBLAS
FP16 on the original weights (speed reference; not an option, it needs 2x the bytes) and
the search kernel in `engine.kernels.w8a16_tuned` over tiles, warps, stages, scale
placement and program grouping. Every candidate is validated against an FP32
dequantization reference before timing; failures are recorded. Rounds are shuffled.
The decision metric is the summed layer cost gate + up + down; gate/up share a shape.

Each timed call reads a different copy from a ring of `--ring` weight buffers (>= 100 MB
per shape): one 12.5 MB INT8 projection otherwise stays resident in the 24 MB L2 across
repeated calls and times at an impossible ~540 GB/s, which a real 28-layer forward that
streams its weights never sees.

    python -m benchmarks.kernels.w8a16_mlp_search --out results/rtx4060/w8a16_mlp_search_1.7b.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import time

import torch


def decode_configs():
    for bn, bk, warps, stages, after in itertools.product((32, 64, 128), (64, 128, 256), (2, 4),
                                                          (2, 3, 4), (False, True)):
        yield dict(block_m=16, block_n=bn, block_k=bk, num_warps=warps, num_stages=stages,
                   group_m=1, scale_after=after)


def prefill_configs(m: int):
    for bm, bn, bk, warps, stages, after, group in itertools.product(
            (16, 32, 64), (64, 128), (64, 128), (4, 8), (2, 3), (False, True), (1, 8)):
        if bm <= m:
            yield dict(block_m=bm, block_n=bn, block_k=bk, num_warps=warps, num_stages=stages,
                       group_m=group, scale_after=after)
    if m >= 128:
        for bn, stages, group in itertools.product((64, 128), (2, 3), (1, 8)):
            yield dict(block_m=128, block_n=bn, block_k=64, num_warps=8, num_stages=stages,
                       group_m=group, scale_after=True)


def label(config) -> str:
    return (f"{config['block_m']}x{config['block_n']}x{config['block_k']}_w{config['num_warps']}"
            f"_s{config['num_stages']}_g{config['group_m']}_{'after' if config['scale_after'] else 'inloop'}")


def time_events(fn, iters):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 128, 256])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-rel-error", type=float, default=1e-2)
    parser.add_argument("--ring", type=int, default=8, help="distinct weight copies cycled per call")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from engine.kernels.w8a16_linear import quantize_weight_per_channel, w8a16_linear
    from engine.kernels.w8a16_tuned import w8a16_linear_tuned
    from engine.model import load_model

    loaded = load_model(args.model, dtype="float16")
    mlp = loaded.model.model.layers[args.layer].mlp
    shapes = {}
    for name in ("gate_proj", "down_proj"):  # up_proj has gate_proj's shape
        weight = getattr(mlp, name).weight.detach().clone()
        q, s = quantize_weight_per_channel(weight)
        shapes[name] = ([weight.clone() for _ in range(args.ring)], [q.clone() for _ in range(args.ring)],
                        [s.clone() for _ in range(args.ring)])
    del loaded, mlp
    torch.cuda.empty_cache()

    rng = random.Random(0)
    compile_ms, cells = {}, []
    for m in args.rows:
        for name, (weights, qs, ss) in shapes.items():
            weight, q, s = weights[0], qs[0], ss[0]
            x = torch.randn(m, weight.shape[1], device="cuda", dtype=torch.float16)
            reference = x.float() @ (q.float() * s.float()[:, None]).T
            scale = reference.abs().mean()
            ring = {"i": 0}

            def nxt():
                ring["i"] = (ring["i"] + 1) % args.ring
                return ring["i"]

            candidates = {
                "engine_current": lambda: (lambda i: w8a16_linear(x, qs[i], ss[i]))(nxt()),
                "cublas_fp16_reference": lambda: (lambda i: x @ weights[i].T)(nxt()),
            }
            for config in (decode_configs() if m <= 16 else prefill_configs(m)):
                candidates[label(config)] = lambda config=config: (
                    lambda i: w8a16_linear_tuned(x, qs[i], ss[i], **config))(nxt())
            status, valid = {}, {}
            for key, fn in candidates.items():
                try:
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    out = fn()
                    torch.cuda.synchronize()
                    compile_ms.setdefault(key, (time.perf_counter() - started) * 1000)
                    error = float((out.float() - reference).abs().max() / scale)
                    if not torch.isfinite(out).all() or (error > args.max_rel_error and key != "cublas_fp16_reference"):
                        status[key] = f"rejected: error {error:.2e}"
                        continue
                    status[key] = f"ok {error:.1e}"
                    valid[key] = fn
                except Exception as error:  # resources/compile failures are results here
                    status[key] = f"failed: {type(error).__name__}: {str(error).splitlines()[0][:120]}"
            for fn in valid.values():
                time_events(fn, 3)
            samples = {key: [] for key in valid}
            for _ in range(args.rounds):
                order = list(valid)
                rng.shuffle(order)
                for key in order:
                    samples[key].append(time_events(valid[key], args.iters))
            results = {key: {"median_ms": statistics.median(v), "spread": (max(v) - min(v)) / statistics.median(v)}
                       for key, v in samples.items()}
            tuned = {k: r for k, r in results.items() if k not in ("engine_current", "cublas_fp16_reference")}
            best = min(tuned, key=lambda k: tuned[k]["median_ms"])
            cells.append({"rows": m, "shape": name, "N": weight.shape[0], "K": weight.shape[1],
                          "results": results, "status": status, "best": best})
            print(f"M={m:3d} {name:9s}: current {results['engine_current']['median_ms']*1000:7.1f} us  "
                  f"cublas_fp16 {results['cublas_fp16_reference']['median_ms']*1000:7.1f} us  "
                  f"best {best} {tuned[best]['median_ms']*1000:7.1f} us  "
                  f"({results['engine_current']['median_ms'] / tuned[best]['median_ms']:.2f}x)", flush=True)
    payload = {"benchmark": "w8a16_mlp_search", "model": args.model, "layer": args.layer,
               "gpu": torch.cuda.get_device_name(), "args": vars(args),
               "methodology": "DRAM-streamed: ring of distinct weight copies per call",
               "first_call_ms_including_compile": compile_ms, "cells": cells}
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
