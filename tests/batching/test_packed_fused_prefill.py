"""Packed fused prefill: multi-row chunk batches without per-row padding."""

from __future__ import annotations

import pytest
import torch

from engine.runtime import GenerationRequest

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_DECODE_PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "A haiku about autumn leaves:",
    "The speed of light in vacuum is",
]
# Two chunked prompts whose final and first chunks share one 128-token budget, so the
# fused step carries two partial prefill rows.
_CHUNKED_PROMPTS = [
    "Explain how a paged KV cache differs from a contiguous one, in detail. " * 11,
    "Write a short story about a lighthouse keeper who " + "kept a very long diary, " * 30,
]


@pytest.fixture(scope="module")
def model_and_references():
    """One fresh model for this module, freed at teardown.

    Importing `_load` from test_continuous_batching created a second copy of that module
    under pytest's import mode, pinning two extra models for the rest of the session and
    running later CUDA tests out of memory. Stock references are taken first, before any
    engine patches the model's modules in place.
    """
    import gc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from engine.kernels.rope import stock_rope

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16,
                                                 device_map="cuda").eval()
    model.config._attn_implementation = "sdpa"
    references = []
    with stock_rope(), torch.inference_mode():
        for prompt, count in [(p, 24) for p in _DECODE_PROMPTS] + [(p, 16) for p in _CHUNKED_PROMPTS]:
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            out = model.generate(ids, max_new_tokens=count, do_sample=False, temperature=None,
                                 top_p=None, top_k=None)
            references.append(out[0, ids.shape[1]:].tolist())
    yield model, tok, references
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _run(model, tok, *, packed: bool, graphs: bool, **engine_kwargs):
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    eng = ContinuousBatchingEngine(
        model, tok, "cuda", num_blocks=256, block_size=16, max_active=8,
        prefix_cache_blocks=0, packed_fused_prefill=packed,
        cuda_graph_batch_sizes=(1, 2, 4, 8) if graphs else None, **engine_kwargs,
    )
    if graphs:
        eng.warmup()
    multi_row_fused = 0
    original = eng._plan_prefill_chunks

    def plan():
        nonlocal multi_row_fused
        plans = original()
        if len(plans) > 1 and eng.scheduler.active and any(
                r.state.name == "DECODING" for r in eng.scheduler.active.values()):
            multi_row_fused += 1
        return plans

    eng._plan_prefill_chunks = plan
    requests = []
    for index, prompt in enumerate(_DECODE_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        requests.append(GenerationRequest(f"d{index}", len(ids), 24, prompt_token_ids=ids))
        assert eng.submit(requests[-1])
    for _ in range(3):
        eng.step()
    for index, prompt in enumerate(_CHUNKED_PROMPTS):
        ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        requests.append(GenerationRequest(f"c{index}", len(ids), 16, prompt_token_ids=ids))
        assert eng.submit(requests[-1])
    while eng.has_unfinished_requests:
        eng.step()
    del eng._plan_prefill_chunks  # drop the instance override and its reference cycle
    assert eng.block_manager.snapshot()["used_blocks"] == len(eng._graph_dummy_blocks)
    result = [r.output_token_ids for r in requests], multi_row_fused, eng.lazy_graph_captures
    _release(eng)
    return result


def _release(eng) -> None:
    """Free the engine's KV pools and graph pools before the next test allocates."""
    import gc

    del eng
    gc.collect()
    torch.cuda.empty_cache()


@cuda
@requires_cuda
@pytest.mark.parametrize("buckets", [False, True])
@pytest.mark.parametrize("graphs", [False, True])
def test_packed_fused_prefill_matches_stock_greedy(graphs: bool, buckets: bool, model_and_references) -> None:
    model, tok, references = model_and_references
    outputs, multi_row, lazy = _run(model, tok, packed=True, graphs=graphs,
                                    packed_prefill_width_buckets=buckets)
    assert multi_row > 0, "workload never produced a multi-row fused prefill step"
    assert lazy == 0
    for produced, expected in zip(outputs, references):
        # Leading-token gate: packing changes GEMM row counts, so a late fp16 near-tie
        # may flip, but a wrong gather/scatter fails within the first tokens.
        assert produced[:8] == expected[:8]


@cuda
@requires_cuda
def test_packed_fused_prefill_with_smoothed_int8_kv_matches_padded_layout(model_and_references) -> None:
    """Packing composes with the INT8 pool: same writes, same tiled INT8 attention."""
    from engine.quantization.kv_smoothing import calibrate_key_smoothing

    model, tok, _ = model_and_references
    calibration = tok(" ".join(_CHUNKED_PROMPTS) * 4, return_tensors="pt").input_ids[0].tolist()[:1024]
    smoothing = calibrate_key_smoothing(model, [calibration])
    kwargs = dict(kv_cache_dtype="int8", kv_key_smoothing=smoothing, int8_prefill_attention="tiled")
    padded, _, _ = _run(model, tok, packed=False, graphs=True, **kwargs)
    packed, multi_row, lazy = _run(model, tok, packed=True, graphs=True, **kwargs)
    assert multi_row > 0 and lazy == 0
    bucketed, _, lazy = _run(model, tok, packed=True, graphs=True, packed_prefill_width_buckets=True, **kwargs)
    assert lazy == 0
    for left, right in zip(bucketed, padded):
        assert left[:8] == right[:8]
    for left, right in zip(packed, padded):
        assert left[:8] == right[:8]


@cuda
@requires_cuda
def test_idle_whole_prompt_prefill_takes_one_step_and_matches_stock(model_and_references) -> None:
    """With nothing decoding, a 2k prompt is prefilled in one forward, not sixteen chunks."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.kernels.rope import stock_rope

    model, tok, _ = model_and_references
    prompt = ("Paged attention maps logical blocks to physical memory and lets requests share "
              "a pool. ") * 110
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    assert 1500 < ids.shape[1] < 4096
    # Earlier tests use this module-scoped model inside an engine, which selects one of
    # the registered paged-attention hooks on the shared config. Restore stock attention
    # before asking Transformers for an independent reference.
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"
    with stock_rope(), torch.inference_mode():
        reference = model.generate(ids, max_new_tokens=16, do_sample=False, temperature=None,
                                   top_p=None, top_k=None)[0, ids.shape[1]:].tolist()
    eng = ContinuousBatchingEngine(model, tok, "cuda", num_blocks=320, block_size=16, max_active=2,
                                   prefix_cache_blocks=0, cuda_graph_batch_sizes=(1, 2))
    eng.warmup()
    out = eng.generate([prompt], max_new_tokens=16)[0]
    assert eng.prefill_sdpa_calls == 1 and eng.prefill_chunked_calls == 0
    assert eng.prefill_steps == 1
    assert out[:8] == reference[:8]
    # A second fresh prompt while the first decodes must still chunk at the budget.
    eng.reset()
    eng.eos_ids.clear()
    from engine.runtime import GenerationRequest
    first = GenerationRequest("a", 40, 64, prompt_token_ids=ids[0, :40].tolist())
    assert eng.submit(first)
    while first.state.name != "DECODING":
        eng.step()
    second = GenerationRequest("b", ids.shape[1], 4, prompt_token_ids=ids[0].tolist())
    assert eng.submit(second)
    chunked_before = eng.prefill_chunked_calls
    while second.state.name == "PREFILLING" or second.allocation is None:
        eng.step()
    assert eng.prefill_chunked_calls > chunked_before
    while eng.has_unfinished_requests:
        eng.step()
    _release(eng)
