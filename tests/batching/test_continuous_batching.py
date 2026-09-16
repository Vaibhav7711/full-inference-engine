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


_STOCK_MODEL = None


def _stock_model():
    """A second, never-patched copy of the checkpoint for token-identical references.

    The engine installs Triton RMSNorm/SwiGLU on the model object it is given and a
    process-global Triton RoPE, so references must not share that object or kernels.
    """
    global _STOCK_MODEL
    if _STOCK_MODEL is None:
        _STOCK_MODEL, _ = _load()
    return _STOCK_MODEL


def _reference_greedy(tok, prompt, max_new_tokens):
    """Per-sequence reference: stock Transformers kernels, SDPA, DynamicCache."""
    from engine.kernels.rope import stock_rope

    model = _stock_model()
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    with stock_rope(), torch.inference_mode():
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
    refs = [_reference_greedy(tok, prompt, 1)[0] for prompt in prompts]
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
    refs = [_reference_greedy(tok, p, max_new_tokens=2) for p in prompts]

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
    refs = [_reference_greedy(tok, p, max_new_tokens=max_new) for p in prompts]

    # Continuous batching
    outs = eng.generate(prompts, max_new_tokens=max_new)

    snapshot = eng.block_manager.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["used_blocks"] == eng.prefix_cache.snapshot()["cached_blocks"]

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
    refs = [_reference_greedy(tok, p, max_new_tokens=max_new) for p in prompts]
    outs = eng.generate(prompts, max_new_tokens=max_new)

    snapshot = eng.block_manager.snapshot()
    assert snapshot["active_requests"] == 0
    assert snapshot["used_blocks"] == eng.prefix_cache.snapshot()["cached_blocks"]

    for i, (out, ref) in enumerate(zip(outs, refs)):
        assert out == ref, (
            f"mixed-length continuous batching diverged for prompt {i}\n"
            f"  ref: {ref}\n  cb:  {out}"
        )


@cuda
@requires_cuda
def test_int8_kv_mode_runs_chunked_prefill_then_decode():
    """Opt-in INT8 storage covers both resumable prefill and fused decode attention."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=1024, block_size=16, max_active=4,
        prefill_chunk_size=16, kv_cache_dtype="int8", prefix_cache_blocks=0,
    )
    prompt = ("Paged attention stores key and value vectors in blocks for efficient " * 12)
    token_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    assert len(token_ids) > 16
    request = _admit(eng, "int8", token_ids, 4)

    # The first partial chunk forces the custom paged-prefill kernel rather than the
    # full-prompt SDPA fast path. Complete it in two or more resumable iterations.
    while request.remaining_prefill_tokens:
        eng.prefill_chunks([(request, min(16, request.remaining_prefill_tokens))])

    assert request.state.name == "DECODING"
    assert eng.key_pool[0].dtype is torch.int8
    assert eng.key_scale_pool is not None
    assert eng.key_scale_pool[0].dtype is torch.float16
    before = request.allocation.sequence_length
    eng.decode_step([request])
    assert request.allocation.sequence_length == before + 1
    assert len(request.output_token_ids) >= 2


@cuda
@requires_cuda
def test_fixed_width_cuda_graph_bucket_matches_reference():
    """A captured paged-decode bucket remains token-identical through real state updates."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    prompts = ["The capital of France is", "2 + 2 ="]
    refs = [_reference_greedy(tok, prompt, 3) for prompt in prompts]
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=512, max_active=2, prefix_cache_blocks=0,
        cuda_graph_batch_size=2,
    )
    outputs = eng.generate(prompts, max_new_tokens=3)
    assert outputs == refs
    assert eng._decode_graphs


@cuda
@requires_cuda
def test_padded_cuda_graph_bucket_matches_reference():
    """Three live rows may safely replay a width-four graph with one dummy KV page."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    prompts = ["The capital of France is", "2 + 2 =", "Water is composed of"]
    refs = [_reference_greedy(tok, prompt, 3) for prompt in prompts]
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=512, max_active=4, prefix_cache_blocks=0,
        cuda_graph_batch_sizes=(2, 4),
    )
    outputs = eng.generate(prompts, max_new_tokens=3)
    assert outputs == refs
    assert any(key[0] == 4 for key in eng._decode_graphs)
    assert len(eng._graph_dummy_blocks) == 3


@cuda
@requires_cuda
def test_d4_chunked_prefill_matches_reference_and_releases_blocks():
    """A prompt spanning several resumable chunks remains token-identical."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    prompt = (
        "Explain how paged attention, continuous batching, and chunked prefill work "
        "together in a production inference engine. Include scheduling and memory details."
    )
    max_new = 12
    reference = _reference_greedy(tok, prompt, max_new)
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=512, block_size=16, max_active=4,
        prefill_chunk_size=8, max_prefill_tokens_per_iteration=8,
    )
    actual = eng.generate([prompt], max_new_tokens=max_new)[0]
    assert actual == reference
    assert eng.block_manager.snapshot()["used_blocks"] == eng.prefix_cache.snapshot()["cached_blocks"]


@cuda
@requires_cuda
def test_d4_partial_prefill_can_be_cancelled():
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=128, block_size=16,
        prefill_chunk_size=4, max_prefill_tokens_per_iteration=4,
    )
    ids = tok("A deliberately longer prompt for cancellation", return_tensors="pt").input_ids[0].tolist()
    request = _admit(eng, "cancel-me", ids, 8)
    eng.prefill_chunks([(request, min(4, len(ids) - 1))])
    assert request.state.name == "PREFILLING"
    assert request.prefilled_token_count > 0
    eng.cancel(request.request_id)
    assert request.state.name == "CANCELLED"
    assert eng.block_manager.snapshot()["used_blocks"] == 0


@cuda
@requires_cuda
def test_d5_repeated_prompt_reuses_prefix_without_changing_tokens():
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    prompt = (
        "You are a careful inference-engine reviewer. Discuss correctness, scheduling, "
        "paged KV ownership, kernel numerical accuracy, and production reliability. "
        "Answer the following request precisely: explain prefix caching."
    )
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=512, block_size=16,
        prefix_cache_blocks=128,
    )
    first = eng.generate([prompt], max_new_tokens=8)[0]
    before = eng.prefix_cache.snapshot()
    second = eng.generate([prompt], max_new_tokens=8)[0]
    after = eng.prefix_cache.snapshot()
    assert second == first
    assert after["hits"] == before["hits"] + 1
    assert after["hit_tokens"] > before["hit_tokens"]
    assert after["exact_entries"] >= 1
    assert eng.block_manager.snapshot()["active_requests"] == 0


# ---------------------------------------------------------------------------
# D6: KV exhaustion preempts the newest request and resumes it token-identically
# ---------------------------------------------------------------------------

def _run_to_completion(eng, prompts, max_new):
    requests = []
    for index, prompt in enumerate(prompts):
        ids = eng.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        request = GenerationRequest(f"d6-{index}", prompt_token_count=len(ids),
                                    max_new_tokens=max_new, prompt_token_ids=ids)
        requests.append(request)
        assert eng.submit(request)
    steps = 0
    while eng.has_unfinished_requests:
        eng.step()
        steps += 1
        assert steps < 10_000, "engine did not converge"
    return requests


@cuda
@requires_cuda
def test_d6_pool_pressure_preempts_and_resumes_without_changing_tokens():
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import RequestState

    model, tok = _load()
    prompts = [
        "The capital of France is",
        "Once upon a time in a distant land",
        "2 + 2 =",
        "The transformer architecture works by",
    ]
    max_new = 40
    refs = [_reference_greedy(tok, p, max_new) for p in prompts]
    # Four requests need ~4 blocks each at full length; ten blocks cannot hold them all,
    # so the newest must yield under pressure and be recomputed later.
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=10, block_size=16, max_active=4,
        prefix_cache_blocks=0,
    )
    requests = _run_to_completion(eng, prompts, max_new)

    assert eng.scheduler.snapshot()["preemption_count"] > 0
    assert all(r.state is RequestState.FINISHED for r in requests), [
        (r.request_id, r.state, r.finish_reason) for r in requests
    ]
    assert any(r.preempted_count > 0 for r in requests)
    for request, ref in zip(requests, refs):
        assert request.output_token_ids == ref, request.request_id
    assert eng.block_manager.snapshot()["used_blocks"] == 0


@cuda
@requires_cuda
def test_d6_pool_too_small_for_one_request_fails_it_instead_of_looping():
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import RequestState

    model, tok = _load()
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=2, block_size=16, max_active=4, prefix_cache_blocks=0,
    )
    [request] = _run_to_completion(eng, ["Explain paged attention in detail."], max_new=64)
    assert request.state is RequestState.FAILED
    assert request.finish_reason == "KV_POOL_EXHAUSTED"
    assert eng.block_manager.snapshot()["used_blocks"] == 0


@cuda
@requires_cuda
def test_d6_preempted_request_reattaches_its_published_prefix():
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import RequestState

    model, tok = _load()
    prompt = (
        "You are a careful inference-engine reviewer. Discuss correctness, scheduling, "
        "paged KV ownership, kernel numerical accuracy, and production reliability."
    )
    max_new = 24
    ref = _reference_greedy(tok, prompt, max_new)
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=12, block_size=16, max_active=4,
        prefix_cache_blocks=4,
    )
    requests = _run_to_completion(eng, [prompt, prompt, prompt], max_new)
    assert all(r.state is RequestState.FINISHED for r in requests)
    assert all(r.output_token_ids == ref for r in requests)
    snapshot = eng.prefix_cache.snapshot()
    assert snapshot["hits"] >= 1
