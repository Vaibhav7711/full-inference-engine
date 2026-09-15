"""A/B repeated scalar versus one batched decode-token device-to-host transfer."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch


def _median(callable_, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        callable_()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", default="1,2,4,8,16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", default="results/phase15_token_transfer_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    widths = [int(value) for value in args.widths.split(",") if value]
    rows = []
    print("\nDecode-token host-transfer A/B")
    print(f"{'width':>7} {'scalar ms':>11} {'batched ms':>12} {'speedup':>9}")
    for width in widths:
        tokens = torch.arange(width, device="cuda", dtype=torch.int64)
        scalar = lambda: [int(tokens[index].item()) for index in range(width)]
        batched = lambda: tokens.tolist()
        if scalar() != batched():
            raise RuntimeError("batched token transfer changed values")
        scalar_ms = _median(scalar, args.warmup, args.repeats)
        batched_ms = _median(batched, args.warmup, args.repeats)
        row = {"width": width, "scalar_ms": scalar_ms, "batched_ms": batched_ms,
               "speedup": scalar_ms / batched_ms}
        rows.append(row)
        print(f"{width:>7} {scalar_ms:>11.5f} {batched_ms:>12.5f} {row['speedup']:>8.2f}x")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump({"config": vars(args), "rows": rows}, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
