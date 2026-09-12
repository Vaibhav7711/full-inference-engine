"""M1 correctness: the paged read path must match the HuggingFace reference exactly.

Two levels of verification:

1. Unit — block_round_trip(kv) == kv for many shapes and block sizes. This proves the
   scatter/gather addressing is correct in isolation.

2. Integration — a full greedy generation with paged attention enabled produces the
   exact same token IDs as the same generation with stock sdpa attention. This proves
   the paged path is correct *inside real model execution*, which is the M1 milestone.

Run:
    python -m pytest tests/cache/test_paged_attention_integration.py -v
    python -m pytest tests/cache/test_paged_attention_integration.py -v -m cuda
"""

from __future__ import annotations

import pytest
import torch

from engine.cache.paged_attention import (
    block_round_trip,
    get_paged_config,
    enable_paged_attention_on_model,
    PAGED_ATTENTION_NAME,
)


# ---------------------------------------------------------------------------
# Unit: block round trip is exact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_size", [8, 16, 32, 64])
@pytest.mark.parametrize("seq_len", [1, 5, 15, 16, 17, 63, 64, 100])
def test_block_round_trip_is_exact(block_size: int, seq_len: int) -> None:
    """scatter -> gather must reproduce the input exactly for all shapes."""
    batch, heads, head_dim = 1, 8, 128
    kv = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float16)
    gathered = block_round_trip(kv, block_size_tokens=block_size)
    assert gathered.shape == kv.shape
    assert torch.equal(gathered, kv), (
        f"round trip mismatch: seq_len={seq_len}, block_size={block_size}, "
        f"max_diff={(gathered - kv).abs().max().item()}"
    )


@pytest.mark.parametrize("block_size", [8, 16])
def test_block_round_trip_batched_multihead(block_size: int) -> None:
    """Round trip must preserve batch and head structure independently."""
    kv = torch.randn(2, 8, 40, 128, dtype=torch.float16)
    gathered = block_round_trip(kv, block_size_tokens=block_size)
    assert torch.equal(gathered, kv)


# ---------------------------------------------------------------------------
# Integration: paged path == HF reference (token-identical greedy)
# ---------------------------------------------------------------------------

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MODEL_NAME = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "The capital of France is",
    "Explain KV caching in one sentence.",
    "2 + 2 =",
    "The transformer architecture works by",
]


@cuda
@requires_cuda
@pytest.mark.parametrize("block_size", [8, 16, 32])
def test_paged_path_matches_reference_greedy(block_size: int) -> None:
    """Full generation with paged attention == stock sdpa, token for token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    max_new_tokens = 32

    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        # --- Reference: stock sdpa attention ---
        model.config._attn_implementation = "sdpa"
        if hasattr(model.config, "_attn_implementation_internal"):
            model.config._attn_implementation_internal = "sdpa"
        with torch.inference_mode():
            ref_out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
            )
        ref_tokens = ref_out[0, ids.shape[1]:].tolist()

        # --- Paged read path ---
        enable_paged_attention_on_model(model, block_size_tokens=block_size, verify=True)
        with torch.inference_mode():
            paged_out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
            )
        paged_tokens = paged_out[0, ids.shape[1]:].tolist()

        assert paged_tokens == ref_tokens, (
            f"paged path diverged from reference\n"
            f"  prompt: {prompt}\n"
            f"  block_size: {block_size}\n"
            f"  ref:   {ref_tokens}\n"
            f"  paged: {paged_tokens}"
        )

    # The paged config should have run and verified without raising
    cfg = get_paged_config()
    assert cfg.calls > 0, "paged attention function was never called"
    assert cfg.max_abs_diff == 0.0, (
        f"gather introduced numerical difference: {cfg.max_abs_diff}"
    )


@cuda
@requires_cuda
def test_paged_attention_actually_engaged() -> None:
    """Sanity: confirm the model is really using our function, not stock sdpa.

    We enable paged attention with verify=True and a deliberately broken block size of 1
    is still correct (every token is its own block), so instead we check the call
    counter increments, proving our code path executed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    enable_paged_attention_on_model(model, block_size_tokens=16, verify=True)
    assert model.config._attn_implementation == PAGED_ATTENTION_NAME

    ids = tokenizer("Hello world", return_tensors="pt").input_ids.cuda()
    cfg = get_paged_config()
    calls_before = cfg.calls

    with torch.inference_mode():
        model(input_ids=ids, use_cache=True, return_dict=True)

    # One forward pass over 28 layers => 28 attention calls
    assert cfg.calls - calls_before == model.config.num_hidden_layers, (
        f"expected {model.config.num_hidden_layers} attention calls, "
        f"got {cfg.calls - calls_before}"
    )
