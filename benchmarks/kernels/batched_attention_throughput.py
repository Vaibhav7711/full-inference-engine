"""Path A: batched vs sequential paged-attention throughput.

The defensible win: running S sequences' decode attention in ONE K4 kernel launch is
faster than running S separate single-sequence launches. One comparison, one number.

    BASELINE (sequential): S separate single-sequence paged-attention calls (K2 path),
                           summed. This is "process one sequence at a time".
    BATCHED (K4):          all S sequences in ONE kernel launch.

    speedup = sequential_time / batched_time, swept over S = 1,2,4,8,16,32.

What this proves: batching amortizes per-launch overhead and keeps the GPU busy across
all sequences at once instead of idling between single-sequence launches. This is the
core throughput benefit continuous batching relies on, measured at the attention level.

What this does NOT claim: a full serving engine. Just the batched-attention win.

Interview one-liner: "I measured batched vs sequential paged attention — batching N
sequences into one launch gave an X times speedup, because it amortizes launch overhead
and keeps the GPU busy across sequences."

Usage:
    python -m benchmarks.kernels.batched_attention_throughput \
        --seq-len 256 --concurrencies 1,2,4,8,16,32 \
        --warmup 5 --iters 20 --output results/batched_attention_throughput.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from datetime import datetime, timezone

import torch


def _device_info() -> dict:
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(p.total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
        })
    try:
        import triton
        info["triton_version"] = triton.__version__
    except Exception:
        pass
    return info


def _build_pool(S, seq_len, block_size, kv_heads, D, device, dtype, seed=7):
    """Build a shared pool with S sequences of the given length, shuffled block tables."""
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = S * blocks_per_seq + 5
    max_blocks = blocks_per_seq

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_blocks, generator=g).tolist()

    key_pool = torch.randn((total_blocks, block_size, kv_heads, D), device=device, dtype=dtype)
    value_pool = torch.randn_like(key_pool)
    block_tables = torch.zeros((S, max_blocks), dtype=torch.int32, device=device)
    seq_lens = torch.full((S,), seq_len, dtype=torch.int32, device=device)

    cursor = 0
    for s in range(S):
        for lb in range(blocks_per_seq):
            block_tables[s, lb] = perm[cursor]
            cursor += 1
    return key_pool, value_pool, block_tables, seq_lens


def _time_cuda(fn, warmup, iters):
    """Median wall time (ms) of fn over iters, after warmup, using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256, help="KV length per sequence")
    parser.add_argument("--concurrencies", default="1,2,4,8,16,32")
    parser.add_argument("--num-q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--output", default="results/batched_attention_throughput.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA.")

    device = "cuda"
    dtype = torch.float16
    concurrencies = [int(c) for c in args.concurrencies.split(",")]

    from engine.kernels.paged_decode_batched import paged_decode_batched
    from engine.kernels.paged_attention_kernel import paged_attention

    H, kvH, D, bs = args.num_q_heads, args.kv_heads, args.head_dim, args.block_size

    results = {"device_info": _device_info(), "config": vars(args), "sweep": []}

    print(f"\n{'='*66}")
    print(f"Batched vs Sequential Paged Attention  (seq_len={args.seq_len}, "
          f"heads={H}/{kvH}, D={D})")
    print(f"{'='*66}")
    print(f"{'S':>4} {'sequential_ms':>14} {'batched_ms':>12} {'speedup':>10} "
          f"{'tok/s_seq':>12} {'tok/s_batch':>12}")
    print("-" * 66)

    for S in concurrencies:
        key_pool, value_pool, block_tables, seq_lens = _build_pool(
            S, args.seq_len, bs, kvH, D, device, dtype,
        )
        # One query token per sequence
        query = torch.randn(S, H, 1, D, device=device, dtype=dtype)

        # --- BATCHED: one K4 launch over all S sequences ---
        def batched():
            return paged_decode_batched(query, key_pool, value_pool,
                                        block_tables, seq_lens, block_n=64)

        batched_ms = _time_cuda(batched, args.warmup, args.iters)

        # --- SEQUENTIAL: S separate single-sequence launches (K2 path) ---
        # Expand each sequence's KV heads for K2 (K2 assumes equal heads). To keep the
        # comparison apples-to-apples at the attention level, we run K2 per sequence with
        # GQA expansion done via repeat on the pool slice. Simplest: build per-seq pages.
        n_rep = H // kvH

        def repeat_pool_heads(pool):
            nb, b, h, d = pool.shape
            if n_rep == 1:
                return pool
            return pool[:, :, :, None, :].expand(nb, b, h, n_rep, d).reshape(nb, b, h * n_rep, d)

        key_pool_exp = repeat_pool_heads(key_pool)
        value_pool_exp = repeat_pool_heads(value_pool)

        def sequential():
            for s in range(S):
                q_s = query[s]                       # [H, 1, D]
                bt_s = block_tables[s].contiguous()
                paged_attention(q_s, key_pool_exp, value_pool_exp, bt_s,
                                kv_len=args.seq_len, causal=False)

        sequential_ms = _time_cuda(sequential, args.warmup, args.iters)

        speedup = sequential_ms / batched_ms if batched_ms > 0 else 0.0
        # Throughput: S tokens produced per call (one per sequence)
        tok_s_seq = S / (sequential_ms / 1000.0)
        tok_s_batch = S / (batched_ms / 1000.0)

        results["sweep"].append({
            "concurrency": S,
            "sequential_ms": round(sequential_ms, 4),
            "batched_ms": round(batched_ms, 4),
            "speedup": round(speedup, 3),
            "tokens_per_sec_sequential": round(tok_s_seq, 1),
            "tokens_per_sec_batched": round(tok_s_batch, 1),
        })

        print(f"{S:>4} {sequential_ms:>14.4f} {batched_ms:>12.4f} {speedup:>9.2f}x "
              f"{tok_s_seq:>12.0f} {tok_s_batch:>12.0f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {args.output}")
    print("\nDefensible claim: batching S sequences' paged attention into one launch is")
    print("faster than S separate launches — the throughput benefit continuous batching")
    print("relies on, measured at the attention level. NOT a full serving engine.")


if __name__ == "__main__":
    main()
