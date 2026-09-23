"""Batched speculation correctness — with the CORRECT oracle.

WHY THE OLD TEST WAS WRONG (and what the research says):
    We compared batched-spec against SINGLE-sequence vanilla spec. Those run on different
    GPU kernel paths (batched GEMM vs single-row GEMM) with different accumulation orders.
    In FP16 the logits differ slightly; at a near-tie the argmax flips; autoregressive
    decoding then amplifies one flipped token into a fully divergent (but coherent) output.
    This is "batch non-invariance" — documented (HF issue #26869, LLM-42 paper 2026) and
    NOT fixable by position_ids (confirmed by others and by our own zero-effect fix).

THE PRACTICAL FP16 ORACLE:
    A depth-1 speculative round and batched greedy use the same one-token target forward,
    so they must be token-identical. A depth-K verification forward has a different GEMM
    shape from K one-token forwards, however; in FP16 a genuine top-2 near-tie can flip.
    A depth-K divergence is therefore accepted only when the two choices are the target
    path's top two logits with a small measured gap. Anything else is a cache/rollback bug.

Tests (run in order):
    test_batched_spec_matches_batched_greedy   -> depth-1 exact gate plus depth-K near-tie gate.
    test_near_tie_diagnostic                   -> proves single-vs-batched divergence is a
                                                  near-tie (small logit gap), not corruption.
    test_fp32_parity_with_single_sequence      -> in FP32 the numerical drift vanishes, so
                                                  batched-spec should match single vanilla.
                                                  (Confirms the FP16 explanation end to end.)

Run:
    python -m pytest tests/batching/test_batched_speculative.py -v -m cuda
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
requires_spec_fp32_memory = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 12 * 1024**3,
    reason="batched speculative FP32 parity needs two models and at least 12 GiB of VRAM",
)

TARGET = "Qwen/Qwen3-1.7B"
DRAFT = "Qwen/Qwen3-0.6B"

PROMPTS = ["The capital of France is", "Once upon a time in a land far away"]


def _load(dtype=torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TARGET, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(
        TARGET, dtype=dtype, device_map="cuda", trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT, dtype=dtype, device_map="cuda", trust_remote_code=True).eval()
    target.config._attn_implementation = "sdpa"
    draft.config._attn_implementation = "sdpa"
    return target, draft, tok


def _batched_greedy(engine, prompts, max_new):
    """THE ORACLE: greedy decode of the target on the SAME left-padded batch, through the
    engine's own _batched_prefill/_batched_decode (same kernel path as batched-spec)."""
    tok, device = engine.tok, engine.device
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    cache, nxt, mask = engine._batched_prefill(engine.target, enc.input_ids, enc.attention_mask)
    N = len(prompts)
    outs = [[] for _ in range(N)]
    done = [False] * N
    for _ in range(max_new):
        toks = nxt.tolist()
        for i in range(N):
            if not done[i]:
                outs[i].append(toks[i][0])
                if toks[i][0] in engine.eos_ids:
                    done[i] = True
        if all(done):
            break
        nxt, cache, mask = engine._batched_decode(engine.target, nxt, cache, mask)
    return outs


# ---------------------------------------------------------------------------
# THE GATE: depth-1 exactness, then depth-K numerical classification
# ---------------------------------------------------------------------------

@cuda
@requires_cuda
def test_batched_spec_matches_batched_greedy():
    from engine.batching.batched_speculative import BatchedSpeculativeEngine

    target, draft, tok = _load()
    eng = BatchedSpeculativeEngine(target, draft, tok, torch.device("cuda"))
    max_new, depth = 20, 4

    greedy = _batched_greedy(eng, PROMPTS, max_new)
    depth_one = eng.generate(PROMPTS, max_new_tokens=max_new, speculation_depth=1).outputs
    spec = eng.generate(PROMPTS, max_new_tokens=max_new, speculation_depth=depth).outputs

    assert depth_one == greedy, "depth-1 speculation must exactly reproduce batched greedy"
    for i, (s, g) in enumerate(zip(spec, greedy)):
        divergence = next((j for j, pair in enumerate(zip(s, g)) if pair[0] != pair[1]), None)
        if divergence is None:
            continue
        prefix = tok(PROMPTS[i], return_tensors="pt").input_ids[0].tolist() + g[:divergence]
        with torch.inference_mode():
            logits = target(
                input_ids=torch.tensor([prefix], device="cuda"), use_cache=False,
            ).logits[0, -1].float()
        top2 = torch.topk(logits, 2)
        gap = (top2.values[0] - top2.values[1]).item()
        assert set(top2.indices.tolist()) == {g[divergence], s[divergence]}, (
            f"depth-{depth} divergence for seq {i} is not a target near-tie: "
            f"greedy={g[divergence]}, speculative={s[divergence]}, "
            f"top2={top2.indices.tolist()}"
        )
        assert gap < 1.0, f"depth-{depth} divergence has a non-near-tie gap of {gap:.3f}"


# ---------------------------------------------------------------------------
# DIAGNOSTIC: single-vs-batched divergence is a near-tie, not corruption
# ---------------------------------------------------------------------------

@cuda
@requires_cuda
def test_near_tie_diagnostic():
    """Where batched-greedy and single-greedy diverge, the top-2 logit gap is small.

    This PROVES the divergence is FP16 numerics (a near-tie flipped by accumulation
    order), not a corrupted cache (which would show a large gap toward a wrong token).
    """
    from engine.batching.batched_speculative import BatchedSpeculativeEngine

    target, draft, tok = _load()
    eng = BatchedSpeculativeEngine(target, draft, tok, torch.device("cuda"))
    max_new = 20

    batched = _batched_greedy(eng, PROMPTS, max_new)

    # Single-sequence greedy for seq 0 (different kernel path)
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        single_out = target.generate(ids, max_new_tokens=max_new, do_sample=False,
                                     temperature=None, top_p=None)
    single = single_out[0, ids.shape[1]:].tolist()

    b = batched[0]
    n = min(len(b), len(single))
    div = next((i for i in range(n) if b[i] != single[i]), None)
    if div is None:
        print("\nbatched and single agree fully on seq 0 — no near-tie reached in 20 tokens")
        return

    # Teacher-force the SHARED prefix (up to divergence) and inspect the top-2 gap
    prefix = torch.tensor([ids[0].tolist() + single[:div]], device="cuda")
    with torch.inference_mode():
        logits = target(input_ids=prefix, use_cache=False, return_dict=True).logits[0, -1].float()
    top2 = torch.topk(logits, 2)
    gap = (top2.values[0] - top2.values[1]).item()
    print(f"\nseq 0 diverges at index {div}: single={single[div]} batched={b[div]}")
    print(f"top-2 candidates at that position: {top2.indices.tolist()}, logit gap = {gap:.4f}")
    # Both candidates should be exactly the two tokens the two paths chose.
    assert set(top2.indices.tolist()) == {single[div], b[div]}, \
        "divergent tokens are not the top-2 — this would indicate corruption, not a near-tie"
    assert gap < 1.0, f"logit gap {gap:.3f} is too large to be a near-tie flip"


# ---------------------------------------------------------------------------
# FP32 PARITY: with no FP16 drift, batched-spec should match single vanilla spec
# ---------------------------------------------------------------------------

@cuda
@requires_cuda
@requires_spec_fp32_memory
def test_fp32_parity_with_single_sequence():
    """In FP32 the batched-vs-single drift vanishes, so batched-spec == single vanilla spec.

    This closes the loop: if the FP16 mismatch were a logic bug, FP32 would ALSO mismatch.
    If FP32 matches, the FP16 mismatch is confirmed numerical.
    """
    from engine.batching.batched_speculative import BatchedSpeculativeEngine
    from engine.speculative.vanilla import VanillaSpeculativeDecoder

    target, draft, tok = _load(dtype=torch.float32)
    eng = BatchedSpeculativeEngine(target, draft, tok, torch.device("cuda"))
    max_new, depth = 20, 4

    spec = eng.generate(PROMPTS, max_new_tokens=max_new, speculation_depth=depth).outputs

    van = VanillaSpeculativeDecoder(target, draft, tok, torch.device("cuda"))
    for i, p in enumerate(PROMPTS):
        ref = van.generate(p, max_new_tokens=max_new, speculation_depth=depth).token_ids
        n = min(len(ref), len(spec[i]))
        assert spec[i][:n] == ref[:n], (
            f"FP32 batched-spec != single vanilla for seq {i} — would indicate a real bug\n"
            f"  ref:  {ref[:n]}\n  spec: {spec[i][:n]}"
        )
