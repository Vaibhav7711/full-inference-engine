"""Per-projection timings for the W4A16 paths at the Qwen3-8B shapes, on this device.

The first 8B run on T4 put the decode step at 439 ms GPU for a mean batch of 5.5, against
an 18 ms sweep of the packed weights. This isolates the projection paths so the step can
be read as a sum: Triton dequant-GEMV (`w4a16_linear`), dequant kernel + cuBLAS over the
scratch (`dense`), and fp16 cuBLAS over the same tile as the floor that would not fit
whole. Times are CUDA-event medians; the `model` row scales each shape by its count per
step (36 layers, one lm_head) so the per-step cost of each path is explicit.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import torch.nn.functional as F

SHAPES_8B = {           # name: (in_features, out_features, count per step)
    "q_proj":   (4096, 4096, 36),
    "k_proj":   (4096, 1024, 36),
    "v_proj":   (4096, 1024, 36),
    "o_proj":   (4096, 4096, 36),
    "gate":     (4096, 12288, 36),
    "up":       (4096, 12288, 36),
    "down":     (12288, 4096, 36),
    "lm_head":  (4096, 151936, 1),
}


def _time(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--block-n", type=int, nargs="+", default=[64])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    from engine.kernels.w4a16_format import quantize_weight_w4_grouped
    from engine.kernels.w4a16_linear import w4_dequantize_into, w4a16_linear
    from engine.quantization.w4a16 import dense_scratch

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"{props.name} sm_{props.major}{props.minor}")
    rows_out: list[dict] = []
    per_step: dict[tuple[int, str], float] = {}
    for name, (k, n, count) in SHAPES_8B.items():
        weight = torch.randn(n, k, device=device, dtype=torch.float16)
        packed, scales = quantize_weight_w4_grouped(weight)
        packed_mib = (packed.numel() + scales.numel() * 2) / 2**20
        for m in args.rows:
            x = torch.randn(m, k, device=device, dtype=torch.float16)
            timings = {"fp16_cublas": _time(lambda: F.linear(x, weight))}
            for block_n in args.block_n:
                timings[f"triton_gemv_bn{block_n}"] = _time(
                    lambda: w4a16_linear(x, packed, scales, None, block_n=block_n))
            scratch = dense_scratch(device, n, k)
            timings["dequant_only"] = _time(lambda: w4_dequantize_into(scratch, packed, scales))
            timings["dense_dequant_cublas"] = _time(
                lambda: F.linear(x, w4_dequantize_into(scratch, packed, scales)))
            floor_ms = packed_mib * 2**20 / 258e9 * 1e3
            row = {"shape": name, "m": m, "k": k, "n": n, "packed_mib": round(packed_mib, 1),
                   "floor_ms_258gbs": round(floor_ms, 3), **{p: round(t, 3) for p, t in timings.items()}}
            rows_out.append(row)
            for path, t in timings.items():
                per_step[(m, path)] = per_step.get((m, path), 0.0) + t * count
            print(f"{name:8s} M={m:<4d} " + "  ".join(f"{p}={t:7.3f}ms" for p, t in timings.items())
                  + f"  floor={floor_ms:.3f}ms", flush=True)
        del weight, packed, scales
        torch.cuda.empty_cache()
    print("\nper decode/prefill step (36 layers + lm_head), ms:")
    paths = sorted({p for _, p in per_step})
    for m in args.rows:
        print(f"  M={m:<4d} " + "  ".join(f"{p}={per_step[(m, p)]:8.1f}" for p in paths))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump({"device": props.name, "rows": rows_out,
                       "per_step_ms": {f"{m}:{p}": v for (m, p), v in per_step.items()}}, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
