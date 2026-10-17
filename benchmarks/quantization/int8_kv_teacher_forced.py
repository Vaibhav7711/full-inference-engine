"""Teacher-forced INT8-KV quality: one causal forward with fake-quantized K/V.

Per-token K/V quantization is independent of other tokens, so a single full-sequence
forward with quantized keys/values reproduces exactly the logits a cache would give
under teacher forcing. Metrics vs FP16: top-1 agreement, mean KL, perplexity.
"""
import json, sys, torch, statistics
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from benchmarks.batching.int8_kv_pressure_ab import _corpus_ids, calibration_batches
from engine.model import load_model
from engine.kernels.rope import stock_rope
from engine.quantization.kv_smoothing import calibrate_key_smoothing

model_name = sys.argv[1]
out_path = sys.argv[2]
# Optional: sequence length and how many trailing positions to score (long contexts
# would otherwise need [L, vocab] fp32 logits several GB in size).
L = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
TAIL = int(sys.argv[4]) if len(sys.argv) > 4 else L - 1
COUNT = int(sys.argv[5]) if len(sys.argv) > 5 else 16
loaded = load_model(model_name, dtype="float16")
model, tok = loaded.model, loaded.tokenizer
with stock_rope():
    smooth = calibrate_key_smoothing(model, calibration_batches(tok)).cuda()

def q(x, group=None):
    shape = x.shape
    if group:
        x = x.reshape(*shape[:-1], shape[-1] // group, group)
    s = (x.float().abs().amax(-1, keepdim=True) / 127).clamp_min(1e-8)
    return (torch.round(x.float() / s).clamp(-127, 127) * s.half().float()).reshape(shape)

STATE = {"v": None}
def attn(module, qy, k, v, mask, **kw):
    mode = STATE["v"]
    if mode is not None:
        li = module.layer_idx
        if mode in ("token", "group32"):
            k = q(k, 32 if mode == "group32" else None).to(k.dtype)
        elif mode == "vonly":
            pass
        elif mode.startswith("smooth"):
            s = smooth[li][None, :, None, :]
            k = (q(k.float() / s, 32 if mode.endswith("g32") else None) * s).to(k.dtype)
        v = q(v).to(v.dtype)
    return sdpa_attention_forward(module, qy, k, v, mask, **kw)
ALL_ATTENTION_FUNCTIONS["fq_tf"] = attn
model.config._attn_implementation = "fq_tf"

corpus = _corpus_ids(tok)
starts = [int(i * (len(corpus) - L) / (COUNT - 1)) for i in range(COUNT)]
variants = [None, "token", "group32", "smooth", "smooth_g32", "fp16k"]
res = {str(v): {"top1": [], "kl": [], "nll": []} for v in variants}
with stock_rope(), torch.inference_mode():
    for st in starts:
        ids = torch.tensor([corpus[st:st + L]], device="cuda")
        ref = None
        for v in variants:
            STATE["v"] = v if v != "fp16k" else None
            if v == "fp16k":
                # FP16 K, INT8 V: reuse the path with K untouched
                STATE["v"] = "vonly"
            hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0, -TAIL - 1:-1]
            logp = torch.log_softmax(model.lm_head(hidden).float(), -1)
            nll = -logp.gather(1, ids[0, -TAIL:, None]).mean().item()
            if ref is None:
                ref = logp
            top1 = (logp.argmax(-1) == ref.argmax(-1)).float().mean().item()
            kl = sum(float((ref[i:i+128].exp() * (ref[i:i+128] - logp[i:i+128])).sum()) for i in range(0, ref.shape[0], 128)) / ref.shape[0]
            del logp
            res[str(v)]["top1"].append(top1); res[str(v)]["kl"].append(kl); res[str(v)]["nll"].append(nll)
summary = {}
for v, r in res.items():
    import math
    summary[v] = {"top1_agreement": statistics.fmean(r["top1"]), "mean_kl": statistics.fmean(r["kl"]),
                  "ppl": math.exp(statistics.fmean(r["nll"]))}
    print(f"{v:12s} top1 {summary[v]['top1_agreement']:.4f}  KL {summary[v]['mean_kl']:.5f}  ppl {summary[v]['ppl']:.3f}")
json.dump({"model": model_name, "sequences": len(starts), "length": L, "scored_tail": TAIL, "summary": summary,
           "note": "teacher-forced; fake-quant mirrors engine per-token INT8 KV; 'vonly' = FP16 K + INT8 V"},
          open(out_path, "w"), indent=2)
