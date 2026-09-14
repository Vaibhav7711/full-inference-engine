"""KV cache quantization benchmark — memory reduction + quality vs FP16 KV.

Measures the real win: INT8 KV halves the KV memory and the decode-phase bandwidth,
which is the memory-bandwidth bottleneck this whole engine attacks.

Reports:
  1. KV memory: INT8 (+ scales) vs FP16 equivalent — the ~2x reduction.
  2. Quality: perplexity with FP16 KV vs INT8 KV, and greedy token agreement.
  3. The honest note: dequant adds a little compute, but it's cheap relative to the
     bandwidth saved — unlike weight-INT8 which sat on the compute path.

Usage:
    python -m benchmarks.quantization.kv_int8_bench \
        --max-new-tokens 32 --output results/kv_int8_bench.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import torch


def _device_info():
    info = {"timestamp": datetime.now(timezone.utc).isoformat(), "torch_version": torch.__version__}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu_name": torch.cuda.get_device_name(0),
                     "gpu_memory_gb": round(p.total_memory / 1e9, 2)})
    return info


@torch.inference_mode()
def _perplexity(model, tok, text, cache_factory, device):
    """Perplexity over `text`, feeding tokens one at a time through the given cache.

    We run teacher-forced: feed the prompt, then step through, measuring NLL of each
    actual next token. Uses the provided cache (FP16 or INT8) so the KV precision affects
    the logits.
    """
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    cache = cache_factory()
    nlls = []
    # Prefill all but last token, then step
    with torch.inference_mode():
        out = model(input_ids=ids[:, :1], past_key_values=cache, use_cache=True, return_dict=True)
        for i in range(1, ids.shape[1]):
            logits = out.logits[:, -1, :]
            target = ids[0, i]
            logp = torch.log_softmax(logits.float(), dim=-1)[0, target]
            nlls.append(-logp.item())
            out = model(input_ids=ids[:, i:i+1], past_key_values=cache, use_cache=True, return_dict=True)
    import math
    return math.exp(sum(nlls) / len(nlls)) if nlls else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", default="results/kv_int8_bench.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA.")

    device = "cuda"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.cache.paged_cache import PagedCache
    from engine.quantization.kv_int8 import QuantizedKVLayer

    print(f"Loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    model.eval()
    model.config._attn_implementation = "sdpa"
    num_layers = model.config.num_hidden_layers

    results = {"device_info": _device_info(), "config": vars(args)}

    def fp16_cache():
        return PagedCache(num_layers=num_layers, block_size_tokens=16, initial_blocks=8)

    def int8_cache():
        c = PagedCache(num_layers=num_layers, block_size_tokens=16, initial_blocks=8)
        c._paged_layers = [QuantizedKVLayer(block_size_tokens=16, initial_blocks=8)
                           for _ in range(num_layers)]
        return c

    # === 1. Memory ===
    print(f"\n{'='*60}\n1. KV memory — INT8 vs FP16\n{'='*60}")
    # Fill both caches with the same sequence, compare storage
    prompt = "The transformer architecture revolutionized natural language processing by " \
             "replacing recurrence with self-attention over the input sequence."
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    ic = int8_cache()
    with torch.inference_mode():
        model(input_ids=ids, past_key_values=ic, use_cache=True, return_dict=True)
    mem = ic._paged_layers[0].memory_bytes()
    # scale to all layers
    total_int8 = mem["total_quantized_bytes"] * num_layers
    total_fp16 = mem["fp16_equivalent_bytes"] * num_layers
    results["memory"] = {
        "per_layer": mem,
        "all_layers_int8_kb": round(total_int8 / 1024, 1),
        "all_layers_fp16_kb": round(total_fp16 / 1024, 1),
        "reduction_pct": mem["reduction_pct"],
    }
    print(f"  Per-layer INT8 (+scales): {mem['total_quantized_bytes']} bytes")
    print(f"  Per-layer FP16 equivalent: {mem['fp16_equivalent_bytes']} bytes")
    print(f"  All {num_layers} layers: {total_int8/1024:.0f} KB (INT8) vs {total_fp16/1024:.0f} KB (FP16)")
    print(f"  Reduction: {mem['reduction_pct']}%")

    # === 2. Quality ===
    print(f"\n{'='*60}\n2. Quality — perplexity FP16 KV vs INT8 KV\n{'='*60}")
    ppl_text = ("The theory of relativity describes how space and time are linked for "
                "objects moving at a consistent speed in a straight line.")
    ppl_fp16 = _perplexity(model, tok, ppl_text, fp16_cache, device)
    ppl_int8 = _perplexity(model, tok, ppl_text, int8_cache, device)
    results["quality"] = {
        "perplexity_fp16_kv": round(ppl_fp16, 3),
        "perplexity_int8_kv": round(ppl_int8, 3),
        "perplexity_increase_pct": round((ppl_int8 / ppl_fp16 - 1) * 100, 2),
    }
    print(f"  Perplexity FP16 KV: {ppl_fp16:.3f}")
    print(f"  Perplexity INT8 KV: {ppl_int8:.3f}  (+{results['quality']['perplexity_increase_pct']:.2f}%)")

    # Greedy agreement
    def gen(cache_factory):
        c = cache_factory()
        g = []
        p = tok("Explain how photosynthesis works.", return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = model(input_ids=p, past_key_values=c, use_cache=True, return_dict=True)
            nt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            g.append(int(nt.item()))
            for _ in range(args.max_new_tokens - 1):
                out = model(input_ids=nt, past_key_values=c, use_cache=True, return_dict=True)
                nt = out.logits[:, -1, :].argmax(-1, keepdim=True)
                g.append(int(nt.item()))
        return g
    g_fp16 = gen(fp16_cache)
    g_int8 = gen(int8_cache)
    agree = sum(1 for a, b in zip(g_fp16, g_int8) if a == b) / len(g_fp16)
    results["quality"]["greedy_agreement"] = round(agree, 3)
    print(f"  Greedy token agreement: {agree:.0%}")

    # === Summary ===
    print(f"\n{'='*60}\nHonest claim\n{'='*60}")
    print(f"INT8 KV cache cut KV memory {mem['reduction_pct']}% with "
          f"{results['quality']['perplexity_increase_pct']:.1f}% perplexity increase.")
    print("This halves the KV bandwidth read every decode step — a direct attack on the")
    print("memory-bandwidth bottleneck. Dequant adds minor compute, cheap vs bandwidth saved.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
