"""Verify INT8 quantization on its REAL axes: memory + quality (+ honest latency).

The existing benchmark (int8_weight_only.py) measures storage bytes and prefill latency.
What's missing is the measurement that actually matters for a weight-only quantization
whose forward dequantizes before F.linear: OUTPUT QUALITY. The honest claim for this
implementation is:

    "INT8 weight-only quantization ~halves model weight memory while preserving output
     quality (low perplexity increase). It does NOT speed up inference — the forward
     dequantizes to FP16 before the matmul, so decode latency is neutral or slightly
     worse. It is a memory optimization, not a latency one."

This benchmark proves that claim with three measurements:

1. MEMORY — model_storage_bytes FP16 vs INT8. The real, provable win.
2. QUALITY — perplexity of both models on a fixed text, plus greedy output agreement
   on several prompts. Shows quantization didn't break the model.
3. LATENCY — decode latency FP16 vs INT8, reported honestly (expected neutral/slower).

Uses the confirmed interface:
    from engine.quantization import quantize_linear_modules, model_storage_bytes
    from engine.model import load_model  (returns .model .tokenizer .device)

Usage:
    python -m benchmarks.quantization.verify_int8 \
        --output results/verify_int8.json
"""

from __future__ import annotations

import argparse
import copy
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
# Quality: perplexity on fixed text
# ---------------------------------------------------------------------------

@torch.inference_mode()
def perplexity(model, tokenizer, text: str, device) -> float:
    """Standard token-level perplexity: exp(mean negative log-likelihood).

    Lower is better. A small increase FP16 -> INT8 means quantization preserved the
    model's language modeling ability. A large increase means it broke it.
    """
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    # Shift for next-token prediction
    out = model(input_ids=ids, use_cache=False, return_dict=True)
    logits = out.logits[:, :-1, :]           # predict token t+1 from position t
    targets = ids[:, 1:]                      # the actual next tokens
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = -token_lp.mean().item()
    return float(torch.exp(torch.tensor(nll)))


@torch.inference_mode()
def greedy_tokens(model, tokenizer, prompt: str, max_new_tokens: int, device) -> list[int]:
    """Greedy generation via explicit decode (no model.generate), returns token ids."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = []
    cur = ids
    pkv = None
    for _ in range(max_new_tokens):
        out = model(input_ids=cur if pkv is None else cur[:, -1:],
                    past_key_values=pkv, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(nt.item()))
        cur = torch.cat([cur, nt], dim=1)
    return generated


# ---------------------------------------------------------------------------
# Latency: decode timing
# ---------------------------------------------------------------------------

@torch.inference_mode()
def decode_latency_ms(model, tokenizer, prompt, decode_steps, device, warmup=2, runs=3):
    """Median per-token decode latency over runs."""
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    def one_run():
        pkv = None
        cur = ids
        # prefill
        out = model(input_ids=cur, use_cache=True, return_dict=True)
        pkv = out.past_key_values
        nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        torch.cuda.synchronize(device)
        ev_s = [torch.cuda.Event(enable_timing=True) for _ in range(decode_steps)]
        ev_e = [torch.cuda.Event(enable_timing=True) for _ in range(decode_steps)]
        for i in range(decode_steps):
            ev_s[i].record()
            out = model(input_ids=nt, past_key_values=pkv, use_cache=True, return_dict=True)
            pkv = out.past_key_values
            nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ev_e[i].record()
        torch.cuda.synchronize(device)
        per_tok = [ev_s[i].elapsed_time(ev_e[i]) for i in range(decode_steps)]
        return statistics.median(per_tok)

    for _ in range(warmup):
        one_run()
    return statistics.median([one_run() for _ in range(runs)])


PERPLEXITY_TEXT = (
    "The transformer architecture revolutionized natural language processing by "
    "replacing recurrence with self-attention. Each layer computes scaled dot-product "
    "attention over queries, keys, and values, allowing the model to weigh the "
    "importance of every token when producing a representation. This parallelism, "
    "combined with residual connections and layer normalization, enables training of "
    "very deep networks on large corpora, which in turn produces the emergent "
    "capabilities observed in modern large language models."
)

AGREEMENT_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "To write a for loop in Python, you",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--output", default="results/verify_int8.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA.")

    device = "cuda"
    from engine.model import load_model
    from engine.quantization import quantize_linear_modules, model_storage_bytes

    print(f"Loading {args.model} (FP16 baseline)...")
    loaded = load_model(args.model)
    fp16_model = loaded.model
    tokenizer = loaded.tokenizer
    dev = loaded.device

    print("Quantizing a copy to INT8...")
    int8_model = copy.deepcopy(fp16_model).eval()
    replaced = quantize_linear_modules(int8_model)

    results = {"device_info": _device_info(), "config": vars(args)}

    # === 1. MEMORY ===
    print(f"\n{'='*60}")
    print("1. Memory — the real win")
    print(f"{'='*60}")
    fp16_bytes = model_storage_bytes(fp16_model)
    int8_bytes = model_storage_bytes(int8_model)
    results["memory"] = {
        "fp16_mb": round(fp16_bytes / 1e6, 1),
        "int8_mb": round(int8_bytes / 1e6, 1),
        "reduction_pct": round((1 - int8_bytes / fp16_bytes) * 100, 1),
        "modules_quantized": len(replaced) if hasattr(replaced, "__len__") else replaced,
    }
    print(f"  FP16 weights: {results['memory']['fp16_mb']:.1f} MB")
    print(f"  INT8 weights: {results['memory']['int8_mb']:.1f} MB")
    print(f"  Reduction:    {results['memory']['reduction_pct']:.1f}%")

    # === 2. QUALITY ===
    print(f"\n{'='*60}")
    print("2. Quality — did quantization break the model?")
    print(f"{'='*60}")
    ppl_fp16 = perplexity(fp16_model, tokenizer, PERPLEXITY_TEXT, dev)
    ppl_int8 = perplexity(int8_model, tokenizer, PERPLEXITY_TEXT, dev)

    # Greedy agreement across prompts
    agree = 0
    agreement_detail = []
    for prompt in AGREEMENT_PROMPTS:
        t_fp16 = greedy_tokens(fp16_model, tokenizer, prompt, 20, dev)
        t_int8 = greedy_tokens(int8_model, tokenizer, prompt, 20, dev)
        match = t_fp16 == t_int8
        # token-level agreement fraction
        matched = sum(1 for a, b in zip(t_fp16, t_int8) if a == b)
        frac = matched / max(len(t_fp16), 1)
        agreement_detail.append({"prompt": prompt[:40], "exact_match": match,
                                 "token_agreement": round(frac, 3)})
        if match:
            agree += 1

    results["quality"] = {
        "perplexity_fp16": round(ppl_fp16, 3),
        "perplexity_int8": round(ppl_int8, 3),
        "perplexity_increase_pct": round((ppl_int8 / ppl_fp16 - 1) * 100, 2),
        "exact_match_prompts": f"{agree}/{len(AGREEMENT_PROMPTS)}",
        "agreement_detail": agreement_detail,
    }
    print(f"  Perplexity FP16: {ppl_fp16:.3f}")
    print(f"  Perplexity INT8: {ppl_int8:.3f}  "
          f"(+{results['quality']['perplexity_increase_pct']:.2f}%)")
    print(f"  Greedy exact match: {agree}/{len(AGREEMENT_PROMPTS)} prompts")
    for d in agreement_detail:
        print(f"    {d['prompt']:<42} match={d['exact_match']} "
              f"tok_agree={d['token_agreement']:.2f}")

    # === 3. LATENCY (honest) ===
    print(f"\n{'='*60}")
    print("3. Latency — reported honestly (expected neutral/slower)")
    print(f"{'='*60}")
    lat_fp16 = decode_latency_ms(fp16_model, tokenizer,
                                 "Explain KV caching.", args.decode_steps, dev)
    lat_int8 = decode_latency_ms(int8_model, tokenizer,
                                 "Explain KV caching.", args.decode_steps, dev)
    results["latency"] = {
        "fp16_ms_per_token": round(lat_fp16, 3),
        "int8_ms_per_token": round(lat_int8, 3),
        "int8_relative": round(lat_int8 / lat_fp16, 3),
        "note": "INT8 dequantizes to FP16 before matmul, so no compute speedup is expected.",
    }
    print(f"  FP16 decode: {lat_fp16:.2f} ms/token")
    print(f"  INT8 decode: {lat_int8:.2f} ms/token  ({lat_int8/lat_fp16:.2f}x)")
    if lat_int8 > lat_fp16:
        print(f"  -> INT8 is SLOWER (dequant overhead). Expected for weight-only.")
    else:
        print(f"  -> INT8 is neutral/faster (memory bandwidth savings on weight reads).")

    # === Honest summary ===
    print(f"\n{'='*60}")
    print("Honest claim")
    print(f"{'='*60}")
    print(f"INT8 weight-only cut model memory {results['memory']['reduction_pct']:.0f}% "
          f"with {results['quality']['perplexity_increase_pct']:.1f}% perplexity increase.")
    print(f"Latency is {results['latency']['int8_relative']:.2f}x FP16 — this is a MEMORY")
    print(f"optimization, not a speed one (dequant-before-matmul forward path).")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
