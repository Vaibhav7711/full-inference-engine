"""M2 benchmark: PagedCache (vectorized paged storage) vs stock DynamicCache.

The headline comparison: M1's naive Python-loop gather cost ~51% overhead. M2's gather
is vectorized (flatten + slice + transpose, no per-token loop). This benchmark measures
how much of that 51% the vectorized path recovers.

Three-way comparison:
    1. stock DynamicCache   — the baseline (model.generate default path)
    2. PagedCache           — our authoritative block-structured store, batch=1
    (M1's paged read path numbers are in results/paged_read_path.json for reference)

Note: M2's write path still contains a small per-token Python loop for the scatter
(handling block-straddling writes explicitly). The decode phase writes 1 token/step, so
that loop is length-1 during decode — cheap. Prefill writes N tokens once. We measure
the real end-to-end effect.

Usage:
    python -m benchmarks.cache.paged_cache_bench \
        --prompt "Explain KV caching in one sentence." \
        --max-new-tokens 32 --block-sizes 8,16,32 \
        --warmup-runs 2 --runs 5 --output results/paged_cache_bench.json
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
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_gb": round(props.total_memory / 1e9, 2),
            "cuda_version": torch.version.cuda,
        })
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        pass
    return info


def _time_stock(model, tokenizer, prompt, max_new_tokens, device):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = ids.shape[1]
    torch.cuda.synchronize(device)
    ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev_s.record()
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None)
    ev_e.record()
    torch.cuda.synchronize(device)
    return out.shape[1] - prompt_len, ev_s.elapsed_time(ev_e)


def _time_paged(model, tokenizer, prompt, max_new_tokens, device, num_layers, block_size):
    from engine.cache.paged_cache import PagedCache

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    eos_ids = set()
    cfg_eos = model.generation_config.eos_token_id
    if isinstance(cfg_eos, int):
        eos_ids.add(cfg_eos)
    elif isinstance(cfg_eos, (list, tuple)):
        eos_ids.update(cfg_eos)

    torch.cuda.synchronize(device)
    ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev_s.record()

    cache = PagedCache(num_layers=num_layers, block_size_tokens=block_size, initial_blocks=4)
    generated = []
    with torch.inference_mode():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(nt.item()))
        for _ in range(max_new_tokens - 1):
            if generated[-1] in eos_ids:
                break
            out = model(input_ids=nt, past_key_values=cache, use_cache=True, return_dict=True)
            nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(nt.item()))

    ev_e.record()
    torch.cuda.synchronize(device)
    return len(generated), ev_s.elapsed_time(ev_e)


def _run(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    totals, gen = [], 0
    for _ in range(runs):
        gen, ms = fn()
        totals.append(ms)
    return {
        "generated_tokens": gen,
        "total_ms_mean": statistics.mean(totals),
        "total_ms_std": statistics.stdev(totals) if len(totals) > 1 else 0.0,
        "tokens_per_sec": gen / (statistics.mean(totals) / 1000.0),
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="Explain KV caching in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--block-sizes", default="8,16,32")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="results/paged_cache_bench.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark requires CUDA.")

    device = torch.device("cuda")
    block_sizes = [int(b) for b in args.block_sizes.split(",")]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    model.config._attn_implementation = "sdpa"
    num_layers = model.config.num_hidden_layers

    results = {"device_info": _device_info(), "config": vars(args), "runs": {}}

    # Baseline: stock DynamicCache
    print("\n=== Baseline (stock DynamicCache) ===")
    baseline = _run(
        lambda: _time_stock(model, tokenizer, args.prompt, args.max_new_tokens, device),
        args.warmup_runs, args.runs,
    )
    results["runs"]["dynamic_cache_baseline"] = baseline
    print(f"  {baseline['tokens_per_sec']:.1f} tok/s  total={baseline['total_ms_mean']:.1f}ms "
          f"(+/- {baseline['total_ms_std']:.1f})")

    # PagedCache per block size
    for bs in block_sizes:
        print(f"\n=== PagedCache (block_size={bs}) ===")
        res = _run(
            lambda bs=bs: _time_paged(model, tokenizer, args.prompt, args.max_new_tokens,
                                      device, num_layers, bs),
            args.warmup_runs, args.runs,
        )
        overhead = (res["total_ms_mean"] / baseline["total_ms_mean"] - 1.0) * 100
        res["overhead_pct_vs_baseline"] = overhead
        results["runs"][f"paged_block{bs}"] = res
        print(f"  {res['tokens_per_sec']:.1f} tok/s  total={res['total_ms_mean']:.1f}ms  "
              f"overhead=+{overhead:.1f}% vs DynamicCache")

    # Summary
    print(f"\n{'='*66}")
    print("M2 PagedCache — Vectorized Paged Storage vs Stock DynamicCache")
    print(f"{'='*66}")
    print(f"{'Config':<26} {'Tok/s':>8} {'Total ms':>10} {'Overhead':>12}")
    print("-" * 58)
    b = results["runs"]["dynamic_cache_baseline"]
    print(f"{'dynamic_cache_baseline':<26} {b['tokens_per_sec']:>8.1f} {b['total_ms_mean']:>10.1f} {'—':>12}")
    for bs in block_sizes:
        r = results["runs"][f"paged_block{bs}"]
        print(f"{'paged_block'+str(bs):<26} {r['tokens_per_sec']:>8.1f} {r['total_ms_mean']:>10.1f} "
              f"{'+'+format(r['overhead_pct_vs_baseline'],'.1f')+'%':>12}")

    print("\nCompare against M1 naive-gather overhead (~51%) in results/paged_read_path.json")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
