"""Qwen decode-shape A/B for FP16 linear versus fused-scale W8A16."""

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
    values = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); callable_(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--output", default="results/w8a16_linear_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    from engine.kernels.w8a16_linear import quantize_weight_per_channel, w8a16_linear

    hidden_size = 1024  # Qwen3-0.6B hidden width on the measured runtime.
    shapes = {"attention_1024": hidden_size, "mlp_3072": 3072, "lm_head_151936": 151936}
    rows = []
    print(f"\nW8A16 decode linear A/B (input width {hidden_size})")
    print(f"{'shape':>18} {'batch':>6} {'fp16 ms':>10} {'w8 ms':>10} {'speedup':>9} {'rel err':>9}")
    for name, out_features in shapes.items():
        # Avoid a multi-gigabyte lm-head test on a constrained T4 unless explicitly
        # requested by a later dedicated benchmark.
        if out_features > 10000:
            continue
        for batch in [int(value) for value in args.batches.split(",") if value]:
            x = torch.randn(batch, hidden_size, device="cuda", dtype=torch.float16)
            weight = torch.randn(out_features, hidden_size, device="cuda", dtype=torch.float16)
            qweight, scales = quantize_weight_per_channel(weight)
            fp16 = lambda: torch.nn.functional.linear(x, weight)
            w8 = lambda: w8a16_linear(x, qweight, scales)
            fp16_output, w8_output = fp16(), w8()
            fp16_ms, w8_ms = _median(fp16, args.warmup, args.repeats), _median(w8, args.warmup, args.repeats)
            error = float((fp16_output.float() - w8_output.float()).abs().mean() / fp16_output.float().abs().mean().clamp_min(1e-5))
            row = {"shape": name, "batch": batch, "fp16_ms": fp16_ms, "w8a16_ms": w8_ms,
                   "speedup": fp16_ms / w8_ms, "relative_error": error}
            rows.append(row)
            print(f"{name:>18} {batch:>6} {fp16_ms:>10.4f} {w8_ms:>10.4f} {row['speedup']:>8.2f}x {error:>9.4f}")
    result = {"config": vars(args), "rows": rows}
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
