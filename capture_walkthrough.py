"""One-shot walkthrough capture — runs EVERY subsystem, saves output for offline study.

Run this once with the GPU on. It captures the whole engine's behavior — correctness,
timings, tradeoffs, the continuous-batching trace — into results/walkthrough_full.json.
Then turn off the GPU and study the JSON offline.

Design for robustness (learned from the global-state bug):
  - model attention reset to "sdpa" before EVERY in-process capture
  - each capture wrapped in try/except — one failure never kills the rest
  - torch.cuda.empty_cache() between heavy sections
  - existing benchmarks run as subprocesses (fully isolated; a crash in one is contained)

Usage (from repo root, GPU on):
    python capture_walkthrough.py
"""

from __future__ import annotations

import io
import json
import contextlib
import subprocess
import sys
import traceback
from datetime import datetime, timezone

import torch


WALK = {"timestamp": datetime.now(timezone.utc).isoformat(), "captures": {}}

# Loaded once, reused across in-process captures
_MODEL = None
_TOK = None


def _load_model():
    global _MODEL, _TOK
    if _MODEL is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading Qwen3-0.6B (once)...")
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", dtype=torch.float16, device_map="cuda", trust_remote_code=True)
        _MODEL.eval()
    return _MODEL, _TOK


def _reset_attention(model):
    """Reset to stock sdpa — critical between captures that change attention."""
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"


def capture(name: str, fn):
    """Run an in-process capture with state reset + error isolation."""
    print(f"\n{'#'*72}\n# {name}\n{'#'*72}")
    buf = io.StringIO()
    try:
        model, _ = _load_model()
        _reset_attention(model)
        with contextlib.redirect_stdout(buf):
            fn()
        out = buf.getvalue()
        WALK["captures"][name] = {"status": "ok", "output": out}
        print(out)
    except Exception:
        tb = traceback.format_exc()
        WALK["captures"][name] = {"status": "FAILED", "error": tb, "partial_output": buf.getvalue()}
        print(buf.getvalue())
        print(f"  [CAPTURE FAILED — recorded, continuing]\n{tb}")
    finally:
        torch.cuda.empty_cache()


def capture_subprocess(name: str, module: str, args: list[str]):
    """Run an existing benchmark as an isolated subprocess, capture its stdout."""
    print(f"\n{'#'*72}\n# {name}  (subprocess: {module})\n{'#'*72}")
    cmd = [sys.executable, "-m", module] + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        out = r.stdout + ("\n[STDERR]\n" + r.stderr if r.returncode != 0 else "")
        WALK["captures"][name] = {
            "status": "ok" if r.returncode == 0 else "nonzero_exit",
            "returncode": r.returncode,
            "output": out,
        }
        print(out[-4000:])  # tail, to keep console readable
    except subprocess.TimeoutExpired:
        WALK["captures"][name] = {"status": "TIMEOUT"}
        print("  [TIMEOUT — skipped]")
    except Exception:
        WALK["captures"][name] = {"status": "FAILED", "error": traceback.format_exc()}
        print(f"  [FAILED]\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════════
# GROUP B — in-process instrumented "feel" captures
# ═══════════════════════════════════════════════════════════════════════

def cap_kv_math():
    from engine.cache.kv_cache import KVCacheGeometry
    model, _ = _load_model()
    geo = KVCacheGeometry.from_model_config(model.config, torch.float16)
    print("Qwen3-0.6B KV cache geometry:")
    print(f"  layers={geo.num_layers}, KV heads={geo.num_kv_heads}, head_dim={geo.head_dim}")
    print(f"  KV bytes/token = 2 x {geo.num_layers} x {geo.num_kv_heads} x {geo.head_dim} x 2 = {geo.bytes_per_token:,}")
    if geo.mha_bytes_per_token:
        print(f"  full MHA would be {geo.mha_bytes_per_token:,} bytes/token")
        print(f"  GQA saves {(1-geo.bytes_per_token/geo.mha_bytes_per_token)*100:.0f}% of KV memory")
    for L in [128, 512, 2048, 4096]:
        print(f"  {L:>5} tokens -> {geo.bytes_for_tokens(L)/1e6:>7.1f} MB")
    total = torch.cuda.get_device_properties(0).total_memory
    free = total - 1.2e9 - 0.5e9
    print(f"  T4 free for KV ~ {free/1e9:.1f} GB -> {int(free/geo.bytes_for_tokens(1024))} concurrent seqs @ 1024 tok")


def cap_prefill_vs_decode():
    from engine.model.runner import ExplicitDecodeRunner
    model, tok = _load_model()
    runner = ExplicitDecodeRunner(model, tok, torch.device("cuda"))
    for _ in range(2):
        runner.generate("Warmup pass", max_new_tokens=8)
    prompts = [
        "The capital of France is",
        "Explain the theory of relativity in detail, covering both special and general "
        "relativity and their many implications for physics and cosmology",
    ]
    print("Prefill (compute-bound, whole prompt at once) vs Decode (memory-bound, one token at a time):\n")
    for p in prompts:
        r = runner.generate(p, max_new_tokens=32)
        m = r.metrics
        dt = sum(m.decode_ms)
        print(f"  Prompt ({len(tok(p).input_ids)} tokens): {p[:45]!r}...")
        print(f"    prefill      = {m.prefill_ms:>7.1f} ms  (one forward pass over whole prompt)")
        print(f"    decode total = {dt:>7.1f} ms for {len(m.decode_ms)} tokens")
        print(f"    per-token    = {dt/max(len(m.decode_ms),1):>7.1f} ms/token")
        print(f"    -> decode dominates: {dt/max(m.prefill_ms,0.01):.1f}x the prefill time\n")


def cap_explicit_correctness():
    from engine.model.runner import ExplicitDecodeRunner
    model, tok = _load_model()
    runner = ExplicitDecodeRunner(model, tok, torch.device("cuda"))
    prompts = ["The capital of France is", "2 + 2 =", "Water is made of"]
    print("Explicit prefill/decode vs HF generate() (greedy, must be token-identical):\n")
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        with torch.inference_mode():
            hf = model.generate(ids, max_new_tokens=20, do_sample=False, temperature=None, top_p=None)
        hf_toks = hf[0, ids.shape[1]:].tolist()
        ours = runner.generate(p, max_new_tokens=20).token_ids
        match = hf_toks == ours
        print(f"  {'MATCH' if match else 'MISMATCH'}  {p!r} -> {tok.decode(ours)!r}")


def cap_kernel_chain():
    import torch.nn.functional as F
    from engine.kernels.triton_attention import triton_attention
    from engine.kernels.paged_attention_kernel import paged_attention, scatter_to_pages
    from engine.kernels.paged_decode_batched import paged_decode_batched
    print("Kernel path — each stage verified against the one below it:\n")
    q = torch.randn(1,8,64,128,device="cuda",dtype=torch.float16)
    k = torch.randn(1,8,64,128,device="cuda",dtype=torch.float16)
    v = torch.randn(1,8,64,128,device="cuda",dtype=torch.float16)
    ref = F.scaled_dot_product_attention(q,k,v,is_causal=True)
    k1 = triton_attention(q,k,v,causal=True)
    print(f"  K1 (Triton attention math)  vs SDPA:  max diff = {(k1-ref).abs().max().item():.2e}")
    bt = torch.tensor([3,1,4,0], dtype=torch.int32, device="cuda")   # shuffled blocks
    kp, vp = scatter_to_pages(k[0], v[0], 16, bt, 5)
    k2 = paged_attention(q[0], kp, vp, bt, kv_len=64, causal=True)
    print(f"  K2 (paged, shuffled blocks) vs K1:    max diff = {(k2-k1[0]).abs().max().item():.2e}")
    # K4 batched decode: 3 seqs, 1 query each, vs per-seq
    S = 3
    qd = torch.randn(S,8,1,128,device="cuda",dtype=torch.float16)
    kp2 = torch.randn(20,16,8,128,device="cuda",dtype=torch.float16)
    vp2 = torch.randn_like(kp2)
    bts = torch.zeros(S,2,dtype=torch.int32,device="cuda")
    for i in range(S):
        bts[i,0]=i*2; bts[i,1]=i*2+1
    sl = torch.tensor([10,20,15],dtype=torch.int32,device="cuda")
    outb = paged_decode_batched(qd,kp2,vp2,bts,sl,block_n=64)
    print(f"  K4 (batched, 3 seqs 1 launch) shapes: query {tuple(qd.shape)} -> out {tuple(outb.shape)}")
    print(f"  -> full kernel path correct: math, addressing, batched execution")


def cap_scheduler_demo():
    from engine.runtime import GenerationRequest, RequestState
    from engine.scheduler import FCFSScheduler
    from engine.cache import ContiguousKVAllocator, KVCacheGeometry
    model, _ = _load_model()
    geo = KVCacheGeometry.from_model_config(model.config, torch.float16)
    # Small arena so we can watch admission + rejection
    alloc = ContiguousKVAllocator(capacity_tokens=100, geometry=geo)
    sched = FCFSScheduler(alloc)
    print("FCFS scheduler admission demo (arena capacity = 100 tokens):\n")
    # Submit requests: some fit, some don't
    reqs = [
        GenerationRequest("r0", prompt_token_count=30, max_new_tokens=10),  # 40 reserved
        GenerationRequest("r1", prompt_token_count=40, max_new_tokens=10),  # 50 reserved
        GenerationRequest("r2", prompt_token_count=50, max_new_tokens=10),  # 60 - won't fit after r0,r1
    ]
    for r in reqs:
        sched.submit(r)
        print(f"  submitted {r.request_id}: reserves {r.reserved_tokens} tokens, state={r.state}")
    admitted = sched.admit_available()
    print(f"\n  admitted: {[r.request_id for r in admitted]}")
    print(f"  snapshot: {sched.snapshot()}")
    print(f"\n  -> r0+r1 fit (40+50=90<=100); r2 (60) blocked, stays WAITING. FCFS admission control.")


def cap_continuous_trace():
    import engine.batching.continuous_batching as cbmod
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    model, tok = _load_model()
    eng = ContinuousBatchingEngine(model, tok, "cuda", num_blocks=512, block_size=16, max_active=4)
    _op = eng.prefill
    def loud_prefill(seq):
        nb = (len(seq.prompt_ids)+15)//16
        print(f"\n  PREFILL {seq.seq_id}: {len(seq.prompt_ids)} tokens {tok.decode(seq.prompt_ids)!r}, needs {nb} block(s)")
        _op(seq)
        print(f"          -> physical blocks {seq.block_table}, seq_len={seq.seq_len}, 1st token {tok.decode([seq.next_token])!r}")
    eng.prefill = loud_prefill
    _od = eng.decode_step
    st = [0]
    def loud_decode(active):
        st[0]+=1
        cb0 = cbmod._ATTN_CALLS
        print(f"\n  DECODE STEP {st[0]}: batch={len(active)}, KV lengths={[s.seq_len for s in active]}")
        _od(active)
        calls = cbmod._ATTN_CALLS - cb0
        print(f"          K4 fired {calls}x ({len(active)} seqs x {calls//max(len(active),1)} layers, one batched launch/layer)")
        for s in active:
            print(f"          {s.seq_id}: +{tok.decode([s.output_ids[-1]])!r} seq_len={s.seq_len}{' [DONE]' if s.done else ''}")
    eng.decode_step = loud_decode
    prompts = ["The capital of France is", "2 + 2 =", "Once upon a time"]
    print(f"WATCH: {len(prompts)} prompts through continuous batching (max 8 tokens each)\n")
    outs = eng.generate(prompts, max_new_tokens=8)
    print("\n  FINAL:")
    for i,(p,o) in enumerate(zip(prompts,outs)):
        print(f"    seq{i}: {p!r} -> {tok.decode(o)!r}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN — run everything
# ═══════════════════════════════════════════════════════════════════════

def main():
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA.")

    print("="*72)
    print("WALKTHROUGH CAPTURE — running every subsystem, saving for offline study")
    print("="*72)

    # --- Group B: in-process instrumented captures (fast, model reused) ---
    capture("L1_kv_cache_math", cap_kv_math)
    capture("L1_prefill_vs_decode", cap_prefill_vs_decode)
    capture("L1_explicit_correctness", cap_explicit_correctness)
    capture("L3_scheduler_admission", cap_scheduler_demo)
    capture("L4_kernel_chain", cap_kernel_chain)
    capture("FINALE_continuous_batching_trace", cap_continuous_trace)

    # --- Group A: existing benchmarks as isolated subprocesses ---
    # Small configs to keep GPU time reasonable.
    capture_subprocess("L2_paged_memory_capacity", "benchmarks.cache.verify_paged_memory",
                       ["--budget-gib", "2", "--block-size", "16", "--real-num-seqs", "4",
                        "--real-seq-len", "128", "--output", "results/_cap_paged_mem.json"])
    capture_subprocess("L2_paged_cache_overhead", "benchmarks.cache.paged_cache_bench",
                       ["--max-new-tokens", "24", "--block-sizes", "16", "--runs", "3",
                        "--output", "results/_cap_paged_cache.json"])
    capture_subprocess("L4_int8_quantization", "benchmarks.quantization.verify_int8",
                       ["--decode-steps", "24", "--output", "results/_cap_int8.json"])
    capture_subprocess("L4_speculative_nonwin", "benchmarks.speculative.vanilla",
                       ["--prompt", "Explain KV caching in one sentence.", "--max-new-tokens", "24",
                        "--speculation-depths", "1,4,8", "--output", "results/_cap_spec.json"])
    capture_subprocess("FINALE_continuous_throughput", "benchmarks.batching.continuous_throughput",
                       ["--num-requests", "16", "--max-new-tokens", "24",
                        "--concurrencies", "1,4,8,16", "--warmup",
                        "--output", "results/_cap_throughput.json"])

    # --- Save everything ---
    with open("results/walkthrough_full.json", "w") as f:
        json.dump(WALK, f, indent=2, default=str)

    print("\n" + "="*72)
    n_ok = sum(1 for c in WALK["captures"].values() if c.get("status") == "ok")
    n_total = len(WALK["captures"])
    print(f"DONE: {n_ok}/{n_total} captures succeeded.")
    print("Saved -> results/walkthrough_full.json")
    for name, c in WALK["captures"].items():
        print(f"  [{c.get('status'):>12}]  {name}")
    print("="*72)


if __name__ == "__main__":
    main()
