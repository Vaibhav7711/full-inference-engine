"""Qwen decode-shape A/B for FP16 linear versus experimental W8A8."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch


def _median(callable_, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); callable_(); end.record(); end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default="results/w8a8_linear_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")

    from engine.kernels.w8a16_linear import quantize_weight_per_channel
    from engine.kernels.w8a8_linear import w8a8_linear

    hidden = 1024
    shapes = {"attention_1024": 1024, "mlp_3072": 3072}
    rows = []
    print(f"\nW8A8 decode linear A/B (input width {hidden}; dynamic per-32 activation scales)")
    print(f"{'shape':>18} {'batch':>6} {'fp16 ms':>10} {'w8a8 ms':>10} {'speedup':>9} {'rel err':>9}")
    for name, output_width in shapes.items():
        for batch in [int(item) for item in args.batches.split(",") if item]:
            inputs = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
            weight = torch.randn(output_width, hidden, device="cuda", dtype=torch.float16)
            qweight, scales = quantize_weight_per_channel(weight)
            fp16 = lambda: torch.nn.functional.linear(inputs, weight)
            w8a8 = lambda: w8a8_linear(inputs, qweight, scales)
            fp16_output, w8a8_output = fp16(), w8a8()
            fp16_ms = _median(fp16, args.warmup, args.repeats)
            w8a8_ms = _median(w8a8, args.warmup, args.repeats)
            error = float((fp16_output.float() - w8a8_output.float()).abs().mean()
                          / fp16_output.float().abs().mean().clamp_min(1e-5))
            row = {"shape": name, "batch": batch, "fp16_ms": fp16_ms,
                   "w8a8_ms": w8a8_ms, "speedup": fp16_ms / w8a8_ms,
                   "relative_error": error}
            rows.append(row)
            print(f"{name:>18} {batch:>6} {fp16_ms:>10.4f} {w8a8_ms:>10.4f} "
                  f"{row['speedup']:>8.2f}x {error:>9.4f}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump({"config": vars(args), "rows": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
