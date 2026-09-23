"""M3: profile the PagedCache decode path to locate the ~11% overhead.

M2 measured that vectorized paged storage costs ~11% vs stock DynamicCache. Before
deciding whether a Triton kernel is justified (M4), we must know WHERE that 11% goes.

Two suspects inside PagedLayer:
    1. The write path — a per-token Python loop that scatters new K,V into pages.
       During decode this loop is length-1 (1 token/step), but it still does per-step
       Python-level indexing and small tensor assignments.
    2. The gather — flatten + slice + transpose that allocates a fresh contiguous
       [1, H, S, D] tensor every step. As S grows, this copies more data each step.

This harness separates them three ways:
    A. torch.profiler over full paged decode — the operator-level timeline.
    B. Micro-timing of write() vs gather() in isolation, via CUDA events, swept over
       sequence length — shows how each scales as the cache grows.
    C. A/B against stock DynamicCache decode under the same profiler, so the paged-only
       operators stand out.

Output feeds the M4 decision: if gather dominates and grows with sequence length, a
fused paged-attention kernel (read K,V directly from blocks, skip the contiguous copy)
is justified. If the write loop dominates, vectorizing the scatter is the cheaper fix.
If overhead is flat and small, no kernel is warranted (a defensible finding).

Usage:
    python -m benchmarks.cache.profile_paged \
        --prompt "Explain KV caching in one sentence." \
        --decode-steps 64 --block-size 16 \
        --output results/profile_paged.json
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
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# B. Micro-timing: write() vs gather() in isolation, swept over seq length
# ---------------------------------------------------------------------------

def micro_time_paged_layer(block_size: int, seq_lengths: list[int], device: str,
                           num_kv_heads: int = 8, head_dim: int = 128,
                           inner_iters: int = 50) -> dict:
    """Time write() and gather() at various pre-filled sequence lengths.

    For each target length L: pre-fill a PagedLayer to L tokens, then repeatedly time
    (a) one decode-style write of 1 token, and (b) one gather of the full L+1 tokens.
    Shows how each operation scales as the cache grows — the key question for M4.
    """
    from engine.cache.paged_cache import PagedLayer

    results = {"seq_lengths": seq_lengths, "write_us": [], "gather_us": []}

    for L in seq_lengths:
        layer = PagedLayer(block_size_tokens=block_size, initial_blocks=max(1, L // block_size + 2))
        # Pre-fill to L tokens with one big write (not timed)
        if L > 0:
            k0 = torch.randn(1, num_kv_heads, L, head_dim, dtype=torch.float16, device=device)
            layer.update(k0, k0)

        one_k = torch.randn(1, num_kv_heads, 1, head_dim, dtype=torch.float16, device=device)

        # --- time write of 1 token ---
        # (re-fill each iter so seq_len stays ~L; we measure the marginal write cost)
        torch.cuda.synchronize(device)
        ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(inner_iters)]
        ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(inner_iters)]
        # Use a scratch layer per iter to avoid unbounded growth
        write_times = []
        for i in range(inner_iters):
            scratch = PagedLayer(block_size_tokens=block_size,
                                 initial_blocks=max(1, L // block_size + 2))
            if L > 0:
                scratch.update(k0, k0)
            torch.cuda.synchronize(device)
            ev_s[i].record()
            scratch._write(one_k, one_k)
            ev_e[i].record()
        torch.cuda.synchronize(device)
        for i in range(inner_iters):
            write_times.append(ev_s[i].elapsed_time(ev_e[i]) * 1000.0)  # ms->us

        # --- time gather of full sequence ---
        layer.update(one_k, one_k)  # now L+1 tokens
        gather_times = []
        ev_gs = [torch.cuda.Event(enable_timing=True) for _ in range(inner_iters)]
        ev_ge = [torch.cuda.Event(enable_timing=True) for _ in range(inner_iters)]
        torch.cuda.synchronize(device)
        for i in range(inner_iters):
            ev_gs[i].record()
            layer._gather()
            ev_ge[i].record()
        torch.cuda.synchronize(device)
        for i in range(inner_iters):
            gather_times.append(ev_gs[i].elapsed_time(ev_ge[i]) * 1000.0)

        results["write_us"].append(round(statistics.median(write_times), 2))
        results["gather_us"].append(round(statistics.median(gather_times), 2))

    return results


# ---------------------------------------------------------------------------
# A + C. torch.profiler over full decode (paged vs stock)
# ---------------------------------------------------------------------------

def profile_decode(model, tokenizer, prompt, decode_steps, device, cache_factory, label):
    """Run a decode loop under torch.profiler; return top ops by CUDA time."""
    from torch.profiler import profile, ProfilerActivity

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Warmup
    for _ in range(2):
        cache = cache_factory()
        with torch.inference_mode():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
            nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(8):
                out = model(input_ids=nt, past_key_values=cache, use_cache=True, return_dict=True)
                nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    cache = cache_factory()
    torch.cuda.synchronize(device)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        with torch.inference_mode():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
            nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(decode_steps - 1):
                out = model(input_ids=nt, past_key_values=cache, use_cache=True, return_dict=True)
                nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)

    # Extract top ops by self CUDA time
    events = prof.key_averages()
    rows = []
    for evt in events:
        cuda_us = getattr(evt, "self_device_time_total", None)
        if cuda_us is None:
            cuda_us = getattr(evt, "self_cuda_time_total", 0)
        rows.append({
            "name": evt.key,
            "cuda_us_total": round(cuda_us, 1),
            "cpu_us_total": round(evt.self_cpu_time_total, 1),
            "count": evt.count,
        })
    rows.sort(key=lambda r: r["cuda_us_total"], reverse=True)

    return {"label": label, "decode_steps": decode_steps, "top_ops": rows[:25]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Explain KV caching in one sentence.")
    parser.add_argument("--decode-steps", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", default="results/profile_paged.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA.")

    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.cache.paged_cache import PagedCache

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    model.config._attn_implementation = "sdpa"
    num_layers = model.config.num_hidden_layers

    results = {"device_info": _device_info(), "config": vars(args)}

    # === B. Micro-timing sweep: write vs gather as seq length grows ===
    print("\n=== Micro-timing: write() vs gather() vs sequence length ===")
    seq_lengths = [16, 32, 64, 128, 256, 512]
    micro = micro_time_paged_layer(args.block_size, seq_lengths, device)
    results["micro_timing"] = micro
    print(f"{'seq_len':>8} {'write_us':>10} {'gather_us':>10}")
    for i, L in enumerate(seq_lengths):
        print(f"{L:>8} {micro['write_us'][i]:>10.2f} {micro['gather_us'][i]:>10.2f}")

    # === A. Profile paged decode ===
    print("\n=== torch.profiler: paged decode ===")
    paged_prof = profile_decode(
        model, tokenizer, args.prompt, args.decode_steps, device,
        cache_factory=lambda: PagedCache(num_layers=num_layers,
                                         block_size_tokens=args.block_size, initial_blocks=8),
        label="paged",
    )
    results["profile_paged"] = paged_prof
    print(f"{'op':<40} {'cuda_us':>12} {'count':>8}")
    for r in paged_prof["top_ops"][:15]:
        print(f"{r['name'][:40]:<40} {r['cuda_us_total']:>12.1f} {r['count']:>8}")

    # === C. Profile stock decode for comparison ===
    print("\n=== torch.profiler: stock DynamicCache decode ===")
    from transformers.cache_utils import DynamicCache
    stock_prof = profile_decode(
        model, tokenizer, args.prompt, args.decode_steps, device,
        cache_factory=lambda: DynamicCache(),
        label="stock",
    )
    results["profile_stock"] = stock_prof
    print(f"{'op':<40} {'cuda_us':>12} {'count':>8}")
    for r in stock_prof["top_ops"][:15]:
        print(f"{r['name'][:40]:<40} {r['cuda_us_total']:>12.1f} {r['count']:>8}")

    # === Interpretation hint ===
    print(f"\n{'='*66}")
    print("Reading the result")
    print(f"{'='*66}")
    w = micro["write_us"]
    g = micro["gather_us"]
    write_growth = w[-1] / w[0] if w[0] > 0 else 0
    gather_growth = g[-1] / g[0] if g[0] > 0 else 0
    print(f"write()  scaling 16->512 tokens: {write_growth:.1f}x  (flat = good)")
    print(f"gather() scaling 16->512 tokens: {gather_growth:.1f}x  (grows with seq len)")
    print()
    if g[-1] > w[-1] * 2:
        print("-> gather dominates and grows with sequence length.")
        print("   A fused paged-attention kernel (read blocks directly, skip the")
        print("   contiguous copy) would target the real bottleneck. M4 justified.")
    elif w[-1] > g[-1] * 2:
        print("-> write path dominates. Vectorizing the scatter loop is the cheaper")
        print("   fix than a Triton kernel. Try that before M4.")
    else:
        print("-> write and gather are comparable. Check absolute magnitude against")
        print("   total decode time before committing to a kernel.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
