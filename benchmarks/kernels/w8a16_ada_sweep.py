"""Tile/warp sweep for the experimental W8A16 decode projection on Ada."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch


def _time(fn, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,8,16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", default="results/rtx4060/w8a16_ada_sweep.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")

    from engine.kernels.w8a16_linear import quantize_weight_per_channel, w8a16_linear

    configs = [
        (bm, bn, bk, warps)
        for bm in (8, 16, 32)
        for bn in (64, 128, 256)
        for bk in (32, 64, 128)
        for warps in (2, 4, 8)
    ]
    rows = []
    for out_features in (1024, 3072):
        for batch in (int(v) for v in args.batches.split(",") if v):
            x = torch.randn(batch, 1024, device="cuda", dtype=torch.float16)
            weight = torch.randn(out_features, 1024, device="cuda", dtype=torch.float16)
            qweight, scales = quantize_weight_per_channel(weight)
            fp16_ms = _time(lambda: torch.nn.functional.linear(x, weight), args.warmup, args.repeats)
            best = None
            for block_m, block_n, block_k, warps in configs:
                fn = lambda bm=block_m, bn=block_n, bk=block_k, nw=warps: w8a16_linear(
                    x, qweight, scales, block_m=bm, block_n=bn, block_k=bk, num_warps=nw,
                )
                try:
                    elapsed = _time(fn, args.warmup, args.repeats)
                except Exception as error:
                    rows.append({"out_features": out_features, "batch": batch,
                                 "config": [block_m, block_n, block_k, warps],
                                 "error": repr(error)})
                    continue
                row = {"out_features": out_features, "batch": batch,
                       "config": [block_m, block_n, block_k, warps],
                       "fp16_ms": fp16_ms, "w8a16_ms": elapsed,
                       "speedup": fp16_ms / elapsed}
                rows.append(row)
                if best is None or elapsed < best[0]:
                    best = (elapsed, row)
            assert best is not None
            row = best[1]
            print(f"N={out_features:4d} M={batch:2d}: fp16 {fp16_ms:.4f} ms, "
                  f"best W8 {best[0]:.4f} ms ({row['speedup']:.2f}x), "
                  f"tile={row['config']}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump({"config": vars(args), "device": torch.cuda.get_device_name(), "rows": rows},
                  handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
