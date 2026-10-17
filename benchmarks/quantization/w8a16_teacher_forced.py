"""Teacher-forced quality of selective MLP W8A16 against the FP16 model.

Greedy first-divergence positions (13 and 20 on the identity prompts) say only when a
near-tie flipped. Teacher forcing over natural text measures the drift itself: top-1
agreement, mean KL(FP16 || W8A16) and perplexity on the same tokens, at a short and a long
context (long: the last `tail` positions scored to bound logit memory).

    python -m benchmarks.quantization.w8a16_teacher_forced --out results/rtx4060/w8a16_teacher_forced_1.7b.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch


@torch.inference_mode()
def tail_hidden(model, ids, tail):
    return model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -tail - 1:-1]


@torch.inference_mode()
def compare(fp16_model, w8_model, ids, tail, slice_rows=128):
    """Top-1 matches, summed KL and summed NLLs, scoring the vocabulary in row slices."""
    h_ref, h_out = tail_hidden(fp16_model, ids, tail), tail_hidden(w8_model, ids, tail)
    targets = ids[0, -tail:]
    matches = kl = nll_ref = nll_out = 0.0
    for i in range(0, tail, slice_rows):
        ref = torch.log_softmax(fp16_model.lm_head(h_ref[i:i + slice_rows]).float(), -1)
        out = torch.log_softmax(w8_model.lm_head(h_out[i:i + slice_rows]).float(), -1)
        t = targets[i:i + slice_rows, None]
        matches += float((ref.argmax(-1) == out.argmax(-1)).sum())
        kl += float((ref.exp() * (ref - out)).sum())
        nll_ref += float(-ref.gather(1, t).sum())
        nll_out += float(-out.gather(1, t).sum())
    return matches / tail, kl / tail, nll_ref / tail, nll_out / tail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lm-head", action="store_true", help="also quantize lm_head to W8A16")
    parser.add_argument("--int8-kv-smoothing", action="store_true",
                        help="also fake-quantize K/V as the smoothed per-token INT8 cache does "
                             "(exact under teacher forcing: per-token quantization is causal-free)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from benchmarks.batching.int8_kv_pressure_ab import _corpus_ids
    from engine.kernels.rope import stock_rope
    from engine.kernels.w8a16_linear import quantize_mlp_w8a16
    from engine.model import load_model

    fp16 = load_model(args.model, dtype="float16")
    w8 = load_model(args.model, dtype="float16")
    quantize_mlp_w8a16(w8.model)
    if args.lm_head:
        from engine.kernels.w8a16_tuned import quantize_lm_head_w8a16
        quantize_lm_head_w8a16(w8.model)
    for loaded in (fp16, w8):
        loaded.model.config._attn_implementation = "sdpa"
    if args.int8_kv_smoothing:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        from benchmarks.batching.int8_kv_pressure_ab import calibration_batches
        from engine.quantization.kv_smoothing import calibrate_key_smoothing

        with stock_rope():
            smoothing = calibrate_key_smoothing(w8.model, calibration_batches(w8.tokenizer)).cuda()

        def per_token(x):
            scale = (x.float().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8)
            return torch.round(x.float() / scale).clamp(-127, 127) * scale.half().float()

        def int8_kv_attention(module, query, key, value, mask, **kwargs):
            factor = smoothing[module.layer_idx][None, :, None, :]
            key = (per_token(key.float() / factor) * factor).to(key.dtype)
            value = per_token(value).to(value.dtype)
            return sdpa_attention_forward(module, query, key, value, mask, **kwargs)

        ALL_ATTENTION_FUNCTIONS["int8_kv_fake_quant"] = int8_kv_attention
        w8.model.config._attn_implementation = "int8_kv_fake_quant"
    corpus = _corpus_ids(fp16.tokenizer)
    payload = {"model": args.model, "lm_head_w8a16": args.lm_head,
               "int8_kv_smoothed_fake_quant": args.int8_kv_smoothing, "settings": {}}
    for length, tail, count in ((1024, 1023, 16), (4096, 512, 8)):
        starts = [int(i * (len(corpus) - length) / (count - 1)) for i in range(count)]
        top1, kl, nll_fp, nll_w8 = [], [], [], []
        with stock_rope():
            for start in starts:
                ids = torch.tensor([corpus[start:start + length]], device="cuda")
                agreement, divergence, nll_a, nll_b = compare(fp16.model, w8.model, ids, tail)
                top1.append(agreement)
                kl.append(divergence)
                nll_fp.append(nll_a)
                nll_w8.append(nll_b)
                torch.cuda.empty_cache()
        row = {"sequences": count, "scored_tail": tail, "top1_agreement": statistics.fmean(top1),
               "mean_kl": statistics.fmean(kl), "ppl_fp16": math.exp(statistics.fmean(nll_fp)),
               "ppl_w8a16": math.exp(statistics.fmean(nll_w8))}
        payload["settings"][str(length)] = row
        print(f"context {length}: top1 {row['top1_agreement']:.4f}  KL {row['mean_kl']:.5f}  "
              f"PPL {row['ppl_fp16']:.3f} -> {row['ppl_w8a16']:.3f}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
