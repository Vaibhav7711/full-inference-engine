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


def _load_fresh():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return model, tok


_ENGINE_MODEL = None


def _load():
    """One checkpoint shared by every engine in this module, as the A/B harness does.

    The engine's kernel installers are idempotent and nothing here mutates the model
    irreversibly, so a fresh 1.2 GB load per test bought nothing but time and, with the
    caching allocator holding earlier blocks, an OOM by the twentieth test on a T4.
    """
    global _ENGINE_MODEL
    if _ENGINE_MODEL is None:
        _ENGINE_MODEL = _load_fresh()
    return _ENGINE_MODEL


_STOCK_MODEL = None


def _stock_model():
    """A second, never-patched copy of the checkpoint for token-identical references.

    The engine installs Triton RMSNorm/SwiGLU on the model object it is given and a
    process-global Triton RoPE, so references must not share that object or kernels.
    """
    global _STOCK_MODEL
    if _STOCK_MODEL is None:
        _STOCK_MODEL, _ = _load_fresh()
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
@pytest.mark.parametrize("disabled", ["triton_rmsnorm", "triton_rope", "triton_swiglu"])
def test_d3_each_fusion_toggle_off_matches_ref(disabled):
    """A fusion switched off must fall back to stock kernels, token-identically, and the
    switch must take effect on a model object a previous engine already patched: the A/B
    harness builds every arm on one shared checkpoint."""
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    from engine.batching.continuous_batching import ContinuousBatchingEngine

    model, tok = _load()
    # A fully patched engine first, so the toggled engine has to undo its work.
    ContinuousBatchingEngine(model, tok, "cuda", num_blocks=256, block_size=16, max_active=4)
    eng = ContinuousBatchingEngine(model, tok, "cuda", num_blocks=2048, block_size=16,
                                   max_active=8, **{disabled: False})

    norms = [m for m in model.modules() if m.__class__.__name__.lower().endswith("rmsnorm")]
    mlps = [m for m in model.modules() if m.__class__.__name__.lower() == "qwen3mlp"]
    rmsnorm_on = any(hasattr(m, "_pre_triton_rmsnorm_forward") for m in norms)
    swiglu_on = any(hasattr(m, "_pre_triton_swiglu_forward") for m in mlps)
    rope_on = hasattr(modeling_qwen3, "_pre_triton_apply_rotary_pos_emb")
    assert rmsnorm_on == (disabled != "triton_rmsnorm")
    assert swiglu_on == (disabled != "triton_swiglu")
    assert rope_on == (disabled != "triton_rope")

    prompts = ["The capital of France is", "The transformer architecture works by"]
    refs = [_reference_greedy(tok, p, max_new_tokens=24) for p in prompts]
    outs = eng.generate(prompts, max_new_tokens=24)
    assert outs == refs, f"{disabled}=False diverged from stock reference: {outs} vs {refs}"

    # Leave the shared model fully patched for the tests that follow.
    ContinuousBatchingEngine(model, tok, "cuda", num_blocks=256, block_size=16, max_active=4)


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


def _staggered_prompts():
    """Prompts whose prefill spans several steps while earlier ones are decoding."""
    long = (
        "Explain how paged attention, continuous batching, and chunked prefill work "
        "together in a production inference engine. Include scheduling and memory details, "
        "and describe what happens when the KV pool runs out of free pages."
    )
    return ["The capital of France is", long, "2 + 2 =", long + " Be concise."]


def _fused_engine(model, tok, **overrides):
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    settings = dict(
        num_blocks=1024, block_size=16, max_active=4, prefix_cache_blocks=0,
        prefill_chunk_size=16, max_prefill_tokens_per_iteration=16,
    )
    settings.update(overrides)
    return ContinuousBatchingEngine(model, tok, "cuda", **settings)


@cuda
@requires_cuda
@pytest.mark.parametrize("graphs", [None, (2, 4)], ids=["eager", "graphed"])
def test_fused_step_runs_decode_and_prefill_in_one_forward(graphs):
    """A step carrying both decode rows and chunk rows runs one forward, token-identical
    to the two-forward engine and to stock Transformers."""
    prompts = _staggered_prompts()
    max_new = 12
    model, tok = _load()
    refs = [_reference_greedy(tok, p, max_new) for p in prompts]
    separate = _fused_engine(model, tok, fused_step=False, cuda_graph_batch_sizes=graphs)
    fused = _fused_engine(model, tok, fused_step=True, cuda_graph_batch_sizes=graphs)
    expected = separate.generate(prompts, max_new_tokens=max_new)
    assert separate.fused_steps == 0
    actual = fused.generate(prompts, max_new_tokens=max_new)
    assert fused.fused_steps > 0, "no step carried decode and prefill together"
    assert fused.prefill_steps >= fused.fused_steps
    if graphs:
        assert fused._fused_graphs, "fused steps never replayed a graph"
        assert not fused._prefill_graph_unsupported
    for prompt, out, ref, sep in zip(prompts, actual, refs, expected):
        assert out == sep, f"fused diverged from the two-forward engine on {prompt!r}"
        assert out == ref, f"fused diverged from stock on {prompt!r}"
    assert fused.block_manager.snapshot()["used_blocks"] == fused.prefix_cache.snapshot()["cached_blocks"]


@cuda
@requires_cuda
def test_fused_step_survives_preemption():
    """Prefill capacity acquired after the decode rows may evict one of them; the fused
    forward must then run without that row and every request must still finish."""
    from engine.runtime import RequestState

    model, tok = _load()
    max_new = 24
    refs = [_reference_greedy(tok, p, max_new) for p in _PRESSURE_PROMPTS]
    eng = _fused_engine(
        model, tok, max_active=4, prefill_chunk_size=32, max_prefill_tokens_per_iteration=32,
        num_blocks=_pressure_blocks(tok, _PRESSURE_PROMPTS, max_new, reserved=3),
        cuda_graph_batch_sizes=(2, 4),
    )
    requests = _run_to_completion(eng, _PRESSURE_PROMPTS, max_new)
    assert eng.scheduler.preemption_count > 0
    assert all(r.state is RequestState.FINISHED for r in requests)
    for request, ref in zip(requests, refs):
        assert request.output_token_ids == ref, request.request_id


@cuda
@requires_cuda
@pytest.mark.parametrize("prefill_attention", ["per_token", "sdpa"])
def test_d4_chunked_prefill_matches_reference_and_releases_blocks(prefill_attention):
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
        prefill_attention=prefill_attention,
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
def test_d6_pool_too_small_for_one_request_is_rejected_before_it_enters_the_engine():
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import RequestState

    model, tok = _load()
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=2, block_size=16, max_active=4, prefix_cache_blocks=0,
    )
    [request] = _run_to_completion(eng, ["Explain paged attention in detail."], max_new=64)
    # A request whose declared prompt + generation budget cannot fit even in an empty
    # pool is an admission error, not an active request that should be preempted.
    assert request.state is RequestState.REJECTED
    assert request.finish_reason == "KV_CAPACITY_EXCEEDED"
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
    max_new = 40
    ref = _reference_greedy(tok, prompt, max_new)
    eng = ContinuousBatchingEngine(
        # Three complete sequences need more than eight pages, while one fits. This
        # forces real recompute preemption rather than merely exercising prefix hits.
        model, tok, "cuda", num_blocks=8, block_size=16, max_active=4,
        prefix_cache_blocks=4,
    )
    requests = _run_to_completion(eng, [prompt, prompt, prompt], max_new)
    assert all(r.state is RequestState.FINISHED for r in requests)
    assert all(r.output_token_ids == ref for r in requests)
    snapshot = eng.prefix_cache.snapshot()
    assert snapshot["hits"] >= 1
    # Cache reuse between serialized requests would satisfy `hits` on its own. Require
    # that a request which actually yielded came back through the prefix path.
    assert any(r.preempted_count > 0 for r in requests)
    assert eng.recompute_report()["recomputed_tokens"] > 0


# ---------------------------------------------------------------------------
# Gate 1B: recompute preemption interacts safely with every other subsystem
# ---------------------------------------------------------------------------

def _pressure_blocks(tok, prompts, max_new, *, block_size=16, fraction=0.6, reserved=0):
    """Pool size that is guaranteed to force preemption for this exact workload.

    Derived from the real tokenization rather than hard-coded, because a pool that merely
    *looks* tight silently stops testing anything when a prompt or tokenizer changes —
    which is exactly how the first version of these tests passed without preempting.

    Returns a pool that is a fraction of the blocks all requests need at peak, but never
    smaller than the largest single request (otherwise admission rejects it instead).
    """
    per_request = [
        (len(tok(prompt, return_tensors="pt").input_ids[0]) + max_new + block_size - 1)
        // block_size
        for prompt in prompts
    ]
    peak = sum(per_request)
    squeezed = max(1, int(peak * fraction))
    usable = max(squeezed, max(per_request))
    assert usable < peak, "workload cannot be squeezed; lengthen prompts or max_new"
    return usable + reserved


_PRESSURE_PROMPTS = [
    "The capital of France is",
    "Once upon a time in a distant land there lived a careful engineer who",
    "2 + 2 =",
    "The transformer architecture works by attending over previous positions, which",
]


@cuda
@requires_cuda
def test_g1b_mixed_length_pressure_stays_token_identical():
    """Uneven prompts under pressure: every request must match the stock reference."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import RequestState

    model, tok = _load()
    max_new = 32
    refs = [_reference_greedy(tok, p, max_new) for p in _PRESSURE_PROMPTS]
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", block_size=16, max_active=4, prefix_cache_blocks=0,
        num_blocks=_pressure_blocks(tok, _PRESSURE_PROMPTS, max_new),
    )
    requests = _run_to_completion(eng, _PRESSURE_PROMPTS, max_new)
    print("\nmixed-length pressure:", eng.recompute_report())
    assert eng.scheduler.preemption_count > 0
    assert all(r.state is RequestState.FINISHED for r in requests)
    for request, ref in zip(requests, refs):
        assert request.output_token_ids == ref, request.request_id


@cuda
@requires_cuda
def test_g1b_preemption_under_cuda_graphs_stays_token_identical():
    """Yielding changes the live row count, so replay must cross graph buckets safely."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import RequestState

    model, tok = _load()
    max_new = 32
    refs = [_reference_greedy(tok, p, max_new) for p in _PRESSURE_PROMPTS]
    # Graph dummy rows are carved out of the pool, so ask for them on top of the squeezed
    # size; clause 5 then has to subtract them again for admission to stay honest.
    num_blocks = _pressure_blocks(tok, _PRESSURE_PROMPTS, max_new, reserved=3)
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=num_blocks, block_size=16, max_active=4,
        prefix_cache_blocks=0, cuda_graph_batch_sizes=(2, 4),
    )
    assert eng.scheduler.reserved_blocks == len(eng._graph_dummy_blocks) == 3
    assert eng.scheduler.effective_capacity_tokens == (num_blocks - 3) * 16

    requests = _run_to_completion(eng, _PRESSURE_PROMPTS, max_new)
    print("\ncuda-graph pressure:", eng.recompute_report())
    assert eng.scheduler.preemption_count > 0
    assert all(r.state is RequestState.FINISHED for r in requests)
    for request, ref in zip(requests, refs):
        assert request.output_token_ids == ref, request.request_id
    assert eng._decode_graphs, "no graph was captured under pressure"


@cuda
@requires_cuda
def test_g1b_int8_kv_preemption_matches_int8_without_preemption():
    """INT8 rebuilds per-block scales on recompute.

    INT8 storage is not bit-identical to fp16, so the reference here is the same engine
    with a pool large enough that nothing ever yields.
    """
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import RequestState

    model, tok = _load()
    max_new = 24
    roomy = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=256, block_size=16, max_active=4,
        kv_cache_dtype="int8", prefix_cache_blocks=0,
    )
    baseline = _run_to_completion(roomy, _PRESSURE_PROMPTS, max_new)
    assert roomy.scheduler.preemption_count == 0

    tight = ContinuousBatchingEngine(
        model, tok, "cuda", block_size=16, max_active=4,
        kv_cache_dtype="int8", prefix_cache_blocks=0,
        num_blocks=_pressure_blocks(tok, _PRESSURE_PROMPTS, max_new),
    )
    pressured = _run_to_completion(tight, _PRESSURE_PROMPTS, max_new)
    print("\nint8 pressure:", tight.recompute_report())
    assert tight.scheduler.preemption_count > 0
    assert all(r.state is RequestState.FINISHED for r in pressured)
    for under_pressure, reference in zip(pressured, baseline):
        assert under_pressure.output_token_ids == reference.output_token_ids


@cuda
@requires_cuda
def test_g1b_cancelling_a_yielded_request_releases_everything():
    """Cancellation must reach a request parked in the queue after a yield."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import RequestState

    model, tok = _load()
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", block_size=16, max_active=4, prefix_cache_blocks=0,
        num_blocks=_pressure_blocks(tok, _PRESSURE_PROMPTS, 64),
    )
    requests = []
    for index, prompt in enumerate(_PRESSURE_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        request = GenerationRequest(f"g1b-cancel-{index}", prompt_token_count=len(ids),
                                    max_new_tokens=64, prompt_token_ids=ids)
        requests.append(request)
        assert eng.submit(request)

    yielded = None
    for _ in range(400):
        if not eng.has_unfinished_requests:
            break
        eng.step()
        yielded = next(
            (r for r in eng.scheduler.waiting if r.preempted_count > 0 and not r.done), None
        )
        if yielded is not None:
            break
    assert yielded is not None, "pressure never produced a yielded request"

    eng.cancel(yielded.request_id, reason="CLIENT_DISCONNECTED")
    assert yielded.state is RequestState.CANCELLED
    assert yielded.finish_reason == "CLIENT_DISCONNECTED"
    assert yielded.request_id not in eng.scheduler.active
    assert all(r.request_id != yielded.request_id for r in eng.scheduler.waiting)

    while eng.has_unfinished_requests:
        eng.step()
    # No customer page may survive the run, cancelled or not.
    assert eng.block_manager.snapshot()["used_blocks"] == len(eng._graph_dummy_blocks)


@cuda
@requires_cuda
def test_g1b_recompute_cost_is_measured_not_just_survived():
    """Recompute must be observable: counts, tokens, and wall time, per request."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.runtime import RequestState

    model, tok = _load()
    max_new = 32
    eng = ContinuousBatchingEngine(
        model, tok, "cuda", block_size=16, max_active=4, prefix_cache_blocks=0,
        num_blocks=_pressure_blocks(tok, _PRESSURE_PROMPTS, max_new),
    )
    requests = _run_to_completion(eng, _PRESSURE_PROMPTS, max_new)
    assert all(r.state is RequestState.FINISHED for r in requests)

    report = eng.recompute_report()
    assert report["preemptions"] > 0
    assert report["recomputed_tokens"] > 0
    assert report["recompute_ms"] > 0
    assert report["progress_epoch"] == len(requests)

    preempted = [r for r in requests if r.preempted_count > 0]
    assert preempted
    for request in preempted:
        overhead = request.recompute_overhead()
        assert overhead["recomputed_tokens"] > 0
        assert overhead["recompute_ms"] > 0
        assert overhead["recompute_tokens_per_output_token"] > 0
    print("\nGate 1B recompute cost:", report)
    for request in requests:
        print(" ", request.request_id, request.recompute_overhead())
