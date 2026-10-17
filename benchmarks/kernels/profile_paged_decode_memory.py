"""One warmed paged-decode launch for Nsight Compute memory/occupancy counters.

The NVTX range encloses exactly one kernel. Use --cache-control none in Nsight
Compute so its replay does not deliberately flush L1/L2 between metric passes.
This isolates one attention layer; use trace_live_decode.py for full-step context.
"""

from __future__ import annotations

import argparse

import torch

from benchmarks.kernels.paged_decode_regime_sweep import _build_pool
from engine.kernels.paged_decode_config import select_paged_decode_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--backend", choices=("per_head", "ada", "gqa", "split_k"), default="per_head")
    parser.add_argument("--splits", type=int, default=8,
                        help="split count when --backend split_k")
    parser.add_argument("--block-n", type=int, default=0,
                        help="override the policy tile size")
    parser.add_argument("--num-warps", type=int, default=0,
                        help="override the policy warp count")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--target-launches", type=int, default=1,
                        help="launches inside the NVTX range; use 2 with NCU --launch-skip 1")
    args = parser.parse_args()
    if min(args.batch, args.context, args.page_size, args.target_launches) <= 0 or args.warmup < 0:
        parser.error("batch, context and page-size must be positive; warmup cannot be negative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    tensors = _build_pool(
        args.batch, args.context, block_size=args.page_size,
        kv_heads=8, head_dim=128,
    )
    block_n, num_warps = select_paged_decode_config(
        args.context, args.batch, args.backend,
    )
    block_n = args.block_n or block_n
    num_warps = args.num_warps or num_warps
    if args.backend == "split_k" and (args.splits < 2 or args.splits & (args.splits - 1)):
        parser.error("split_k requires --splits to be a power of two at least 2")
    if args.backend == "ada":
        from engine.kernels.paged_decode_ada import paged_decode_ada

        def launch():
            return paged_decode_ada(
                *tensors, max_sequence_length=args.context,
                block_n=block_n, num_warps=num_warps,
            )
    elif args.backend == "split_k":
        from engine.kernels.paged_decode_split_k import paged_decode_split_k

        def launch():
            return paged_decode_split_k(
                *tensors, max_sequence_length=args.context, splits=args.splits,
                block_n=block_n, num_warps=num_warps,
            )
    elif args.backend == "gqa":
        from engine.kernels.paged_decode_gqa import paged_decode_gqa

        def launch():
            return paged_decode_gqa(*tensors, block_n=block_n, num_warps=num_warps)
    else:
        from engine.kernels.paged_decode_batched import paged_decode_batched

        def launch():
            return paged_decode_batched(*tensors, block_n=block_n, num_warps=num_warps)

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    print({
        "gpu": torch.cuda.get_device_name(), "backend": args.backend,
        "batch": args.batch, "context": args.context,
        "page_size": args.page_size, "block_n": block_n, "num_warps": num_warps,
        "splits": args.splits if args.backend == "split_k" else None,
        "kv_pool_bytes": (tensors[1].numel() + tensors[2].numel()) * 2,
    }, flush=True)

    torch.cuda.nvtx.range_push("profile_target")
    try:
        for _ in range(args.target_launches):
            output = launch()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
    if not torch.isfinite(output).all():
        raise RuntimeError("paged decode produced non-finite output")


if __name__ == "__main__":
    main()
