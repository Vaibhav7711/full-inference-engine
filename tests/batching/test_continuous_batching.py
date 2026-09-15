"""Staged continuous-batching tests — run IN ORDER so failures localize.

    test_d1_prefill_stores_correct_kv   -> D1: prefill scatters right K,V into the pool
    test_d2_decode_step_matches_per_seq -> D2: one batched step == per-sequence decode
    test_d3_full_generation_matches_ref -> D3: full loop == per-sequence reference

Because they're staged, a D2 failure means D1 already passed (bug is in the batched
decode), and a D3 failure means D2 passed (bug is in the loop). Same discipline that made
the kernel path clean.

Run:
    python -m pytest tests/batching/test_continuous_batching.py -v -m cuda
    # or one stage at a time:
    python -m pytest tests/batching/test_continuous_batching.py::test_d1_prefill_stores_correct_kv -v
"""

from __future__ import annotations

import pytest
import torch

from engine.runtime import GenerationRequest

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MODEL_NAME = "Qwen/Qwen3-0.6B"


def _load():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tok


def _reference_greedy(model, tok, prompt, max_new_tokens):
    """Per-sequence reference: stock sdpa + DynamicCache via generate."""
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None)
    return out[0, ids.shape[1]:].tolist()


def _admit(eng, request_id, prompt_ids, max_new_tokens):
    request = GenerationRequest(
        request_id,
        prompt_token_count=len(prompt_ids),
        max_new_tokens=max_new_tokens,
        prompt_token_ids=prompt_ids,
    )
    eng.scheduler.submit(request)
    assert eng.scheduler.admit_available(max_active_requests=eng.max_active) == [request]
    return request


# ---------------------------------------------------------------------------
# D1: prefill stores correct K,V into the pool
# ---------------------------------------------------------------------------

@cuda
@requires_cuda
def test_d1_prefill_stores_correct_kv():
    """After prefill, the pool holds the same K,V a single-sequence PagedCache would."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.cache.paged_cache import PagedCache

    model, tok = _load()
    eng = ContinuousBatchingEngine(model, tok, "cuda", num_blocks=512, block_size=16)

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    seq = _admit(eng, "s0", ids, 8)
    eng.prefill(seq)

    prompt_len = len(ids)
    assert seq.allocation.sequence_length == prompt_len

    # Reference: prefill the same prompt with a standalone PagedCache
    model.config._attn_implementation = "sdpa"
    ref_cache = PagedCache(num_layers=eng.num_layers, block_size_tokens=16, initial_blocks=8)
    with torch.inference_mode():
        model(input_ids=torch.tensor([ids], device="cuda"),
              past_key_values=ref_cache, use_cache=True, return_dict=True)

    # Compare pool K,V (at seq's blocks) to the reference cache's stored K,V, per layer
    for layer_idx in range(eng.num_layers):
        ref_pl = ref_cache._paged_layers[layer_idx]
        ref_k = ref_pl.key_pages.flatten(0, 1)[:prompt_len]     # [prompt_len, kv_heads, D]
        ref_v = ref_pl.value_pages.flatten(0, 1)[:prompt_len]
        for pos in range(prompt_len):
            lb = pos // 16
            off = pos % 16
            pb = seq.block_table[lb]
            got_k = eng.key_pool[layer_idx][pb, off]
            got_v = eng.value_pool[layer_idx][pb, off]
            assert torch.equal(got_k, ref_k[pos]), f"layer {layer_idx} pos {pos} K mismatch"
            assert torch.equal(got_v, ref_v[pos]), f"layer {layer_idx} pos {pos} V mismatch"


@cuda
@requires_cuda
def test_d1_batched_prefill_matches_individual_reference():
    """Mixed prompt lengths share one prefill forward without changing first tokens."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    eng = ContinuousBatchingEngine(model, tok, "cuda", num_blocks=1024, block_size=16)
    prompts = [
        "Hi",
        "The capital of France is",
        "Explain why continuous batching improves inference throughput in one sentence.",
    ]
    refs = [_reference_greedy(model, tok, prompt, 1)[0] for prompt in prompts]
    requests = []
    for index, prompt in enumerate(prompts):
        token_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        request = GenerationRequest(
            f"batch-{index}", len(token_ids), 8, prompt_token_ids=token_ids
        )
        eng.scheduler.submit(request)
        requests.append(request)
    admitted = eng.scheduler.admit_available(max_active_requests=eng.max_active)
    assert admitted == requests

    eng.prefill_batch(admitted)

    assert [request.output_token_ids[0] for request in requests] == refs
    assert all(request.state.name == "DECODING" for request in requests)


# ---------------------------------------------------------------------------
# D2: one batched decode step matches per-sequence decode
# ---------------------------------------------------------------------------

@cuda
@requires_cuda
def test_d2_decode_step_matches_per_seq():
    """One batched decode step over N sequences == each sequence's own next token.

    We prefill N sequences, then run ONE batched decode step, and check each sequence's
    produced token equals what stock generation gives as that sequence's first decode
    token. (The first decode token after prefill is deterministic greedy.)
    """
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    eng = ContinuousBatchingEngine(model, tok, "cuda", num_blocks=1024, block_size=16)

    prompts = [
        "The capital of France is",
        "Once upon a time in a distant land",
        "2 + 2 =",
    ]
    # Reference: each prompt's first TWO greedy tokens (prefill token + first decode token)
    refs = [_reference_greedy(model, tok, p, max_new_tokens=2) for p in prompts]

    # Prefill all (gives each its first token = refs[i][0])
    seqs = []
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids[0].tolist()
        s = _admit(eng, f"s{i}", ids, 8)
        eng.prefill(s)
        seqs.append(s)
        assert s.output_token_ids[0] == refs[i][0], (
            f"prefill token mismatch seq {i}: got {s.output_token_ids[0]}, ref {refs[i][0]}"
        )

    # One batched decode step -> each seq's SECOND token
    eng.decode_step(seqs)

    for i, s in enumerate(seqs):
        assert s.output_token_ids[1] == refs[i][1], (
            f"batched decode token mismatch seq {i}: "
            f"got {s.output_token_ids[1]}, ref {refs[i][1]}\n"
            f"  prompt: {prompts[i]}"
        )


# ---------------------------------------------------------------------------
# D3: full generation matches per-sequence reference
# ---------------------------------------------------------------------------

@cuda
@requires_cuda
def test_d3_full_generation_matches_ref():
    """Full continuous-batching generation == per-sequence stock generation, token-identical."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    eng = ContinuousBatchingEngine(model, tok, "cuda", num_blocks=2048, block_size=16, max_active=8)

    prompts = [
        "The capital of France is",
        "Once upon a time in a distant land",
        "2 + 2 =",
        "The transformer architecture works by",
    ]
    max_new = 24

    # Reference per sequence
    refs = [_reference_greedy(model, tok, p, max_new_tokens=max_new) for p in prompts]

    # Continuous batching
    outs = eng.generate(prompts, max_new_tokens=max_new)

    snapshot = eng.block_manager.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["free_blocks"] == snapshot["num_blocks"]

    for i, (out, ref) in enumerate(zip(outs, refs)):
        assert out == ref, (
            f"continuous batching diverged from reference for prompt {i}\n"
            f"  prompt: {prompts[i]}\n"
            f"  ref:    {ref}\n"
            f"  cb:     {out}"
        )


@cuda
@requires_cuda
def test_d3_mixed_lengths_and_staggered():
    """Sequences of very different prompt lengths + generation lengths batched together."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    eng = ContinuousBatchingEngine(model, tok, "cuda", num_blocks=2048, block_size=16, max_active=8)

    prompts = [
        "Hi",  # very short
        "The transformer architecture, introduced in the paper Attention Is All You Need, "
        "revolutionized natural language processing by",  # long
        "42",
    ]
    max_new = 20
    refs = [_reference_greedy(model, tok, p, max_new_tokens=max_new) for p in prompts]
    outs = eng.generate(prompts, max_new_tokens=max_new)

    snapshot = eng.block_manager.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["free_blocks"] == snapshot["num_blocks"]

    for i, (out, ref) in enumerate(zip(outs, refs)):
        assert out == ref, (
            f"mixed-length continuous batching diverged for prompt {i}\n"
            f"  ref: {ref}\n  cb:  {out}"
        )
