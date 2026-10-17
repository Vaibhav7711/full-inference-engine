"""Offline search: Triton W4A16 GEMM over tinygemm int4pack weights, prefill row counts.

Real Qwen3-4B-AWQ layer-0 projections (q/k/v/o and gate/up/down), each timed against
`torch.ops.aten._weight_int4pack_mm` (the engine's path) and cuBLAS FP16 on the
dequantized weight (reference only: it needs 4x the bytes). A ring of weight copies keeps
every call DRAM-streamed. Every candidate is validated against an exact FP32
dequantization; the decision metric is the summed seven-projection layer cost.

    python -m benchmarks.kernels.int4pack_prefill_search --out results/rtx4060/int4pack_prefill_search_4b.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics

import torch


def dequantize(packed: torch.Tensor, qparams: torch.Tensor) -> torch.Tensor:
    """Exact FP32 `[N, K]` weight from int4pack buffers, using the derived layout."""
    n_tiles, k_tiles = packed.shape[:2]
    N, K = n_tiles * 8, k_tiles * 128
    n = torch.arange(N, device=packed.device)[:, None]
    k = torch.arange(K, device=packed.device)[None, :]
    lane = 4 * (n % 8) + (k % 128 % 8) // 2
    word = (k % 128) // 32
    nibble = (k % 32) // 8 + 4 * (k % 2)
    flat = packed.reshape(-1)
    address = (((n // 8) * k_tiles + k // 128) * 32 + lane) * 4 + word
    q = ((flat[address] >> (4 * nibble)) & 15).float()
    scale = qparams[..., 0].float().T.repeat_interleave(128, 1)
    zero = qparams[..., 1].float().T.repeat_interleave(128, 1)
    return (q - 8) * scale + zero


def time_ring(fn, ring, iters):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(iters):
        fn(i % ring)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-4B-AWQ")
    parser.add_argument("--rows", type=int, nargs="+", default=[16, 32, 64, 128, 132, 256, 260])
    parser.add_argument("--ring", type=int, default=6)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=18)
    parser.add_argument("--kernel", choices=("v1", "v2", "v3"), default="v3",
                        help="v1: per-element gather (rejected); v2: coalesced packed loads; "
                             "v3: v2 with bit-trick dequant, plus the dequant+cuBLAS candidate")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from engine.kernels.int4pack_gemm import (
        int4pack_gemm, int4pack_gemm_dequant_cublas, int4pack_gemm_v2, int4pack_gemm_v3,
    )
    from engine.model import load_awq_tinygemm_model

    loaded = load_awq_tinygemm_model(args.model)
    layer = loaded.model.model.layers[0]
    projections = {
        "q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj, "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj, "up_proj": layer.mlp.up_proj, "down_proj": layer.mlp.down_proj,
    }
    shapes = {}
    for name, module in projections.items():
        packed, qp = module.packed_weight.detach(), module.scale_and_zero.detach()
        dense = dequantize(packed, qp)
        shapes[name] = ([packed.clone() for _ in range(args.ring)], [qp.clone() for _ in range(args.ring)],
                        [dense.half() for _ in range(2)], dense)
    del loaded, layer, projections
    torch.cuda.empty_cache()

    if args.kernel == "v1":
        gemm = int4pack_gemm
        configs = [dict(block_m=bm, block_n=bn, block_k=bk, num_warps=w, num_stages=st, group_m=g)
                   for bm, bn, bk, w, st, g in itertools.product((16, 32, 64, 128), (64, 128), (32, 64),
                                                                 (4, 8), (2, 3), (1, 8))]
    elif args.kernel == "v2":
        gemm = int4pack_gemm_v2
        configs = [dict(block_m=bm, block_n=bn, block_k=128, num_warps=w, num_stages=st, group_m=g)
                   for bm, bn, w, st, g in itertools.product((16, 32, 64, 128), (32, 64, 128),
                                                             (4, 8), (2, 3), (1, 8))]
    else:
        gemm = int4pack_gemm_v3
        # BLOCK_N 128 won every v2 cell; the v3 sweep keeps it and varies the rest.
        configs = [dict(block_m=bm, block_n=128, block_k=128, num_warps=w, num_stages=st, group_m=g)
                   for bm, w, st, g in itertools.product((16, 32, 64, 128), (4, 8), (2, 3), (1, 8))]
    rng = random.Random(0)
    cells = []
    for m in args.rows:
        for name, (packs, qps, denses, exact) in shapes.items():
            K = exact.shape[1]
            x = torch.randn(m, K, device="cuda", dtype=torch.float16)
            reference = x.float() @ exact.T
            scale = reference.abs().mean()
            candidates = {
                "tinygemm": lambda i: torch.ops.aten._weight_int4pack_mm(
                    x.to(torch.bfloat16), packs[i], 128, qps[i]),
                "cublas_fp16_reference": lambda i: x @ denses[i % 2].T,
            }
            if args.kernel == "v3":
                candidates["dequant_cublas"] = lambda i: int4pack_gemm_dequant_cublas(x, packs[i], qps[i])
            for c in configs:
                if c["block_m"] > max(16, 1 << (m - 1).bit_length()):
                    continue
                key = f"{c['block_m']}x{c['block_n']}x{c['block_k']}_w{c['num_warps']}_s{c['num_stages']}_g{c['group_m']}"
                kwargs = {k: v for k, v in c.items() if not (args.kernel in ("v2", "v3") and k == "block_k")}
                candidates[key] = lambda i, kwargs=kwargs: gemm(x, packs[i], qps[i], **kwargs)
            status, valid = {}, {}
            for key, fn in candidates.items():
                try:
                    error = float((fn(0).float() - reference).abs().max() / scale)
                    # FP16-weight paths (cuBLAS reference, dequant+cuBLAS) carry the
                    # fp16 rounding of every weight, ~1e-2 relative at K=4096; the fused
                    # kernels keep q-8 exact and apply scales in FP32, so they are held
                    # to the tighter bound.
                    bound = 2e-2 if key in ("cublas_fp16_reference", "dequant_cublas") else 1e-2
                    if key != "tinygemm" and error > bound:
                        status[key] = f"rejected: error {error:.2e}"
                        continue
                    status[key] = f"ok {error:.1e}"
                    valid[key] = fn
                except Exception as error:
                    status[key] = f"failed: {type(error).__name__}: {str(error).splitlines()[0][:100]}"
            for fn in valid.values():
                time_ring(fn, args.ring, 3)
            samples = {key: [] for key in valid}
            for _ in range(args.rounds):
                order = list(valid)
                rng.shuffle(order)
                for key in order:
                    samples[key].append(time_ring(valid[key], 2 if key == "cublas_fp16_reference" else args.ring,
                                                  args.iters))
            results = {k: {"median_ms": statistics.median(v), "spread": (max(v) - min(v)) / statistics.median(v)}
                       for k, v in samples.items()}
            ours = {k: r for k, r in results.items() if k not in ("tinygemm", "cublas_fp16_reference", "dequant_cublas")}
            best = min(ours, key=lambda k: ours[k]["median_ms"])
            cells.append({"rows": m, "projection": name, "N": exact.shape[0], "K": K,
                          "results": results, "status": status, "best": best})
            for key, state in status.items():
                if not state.startswith("ok"):
                    print(f"    M={m} {name} {key}: {state}", flush=True)
        def layer_cost(key):
            parts = [c["results"].get(key, {}).get("median_ms") for c in cells if c["rows"] == m]
            return sum(parts) if all(part is not None for part in parts) else float("nan")
        common = set.intersection(*[set(c["results"]) for c in cells if c["rows"] == m]) - {"tinygemm", "cublas_fp16_reference", "dequant_cublas"}
        shared = min(common, key=layer_cost)
        extra = (f"  dequant_cublas {layer_cost('dequant_cublas')*1000:7.1f} us"
                 if args.kernel == "v3" else "")
        print(f"M={m:4d} layer: tinygemm {layer_cost('tinygemm')*1000:7.1f} us  "
              f"cublas_fp16 {layer_cost('cublas_fp16_reference')*1000:7.1f} us  "
              f"int4pack {shared} {layer_cost(shared)*1000:7.1f} us  "
              f"({layer_cost('tinygemm') / layer_cost(shared):.2f}x vs tinygemm){extra}", flush=True)
    with open(args.out, "w") as handle:
        json.dump({"benchmark": "int4pack_prefill_search", "kernel": args.kernel, "model": args.model,
                   "gpu": torch.cuda.get_device_name(), "args": vars(args),
                   "methodology": "DRAM-streamed weight ring; exact-dequant validation", "cells": cells},
                  handle, indent=2)
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
