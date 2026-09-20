"""M2 correctness: PagedCache must match the stock DynamicCache reference exactly.

This is the M2 milestone gate. Where M1 proved block addressing was correct in parallel,
M2 proves our page tensors work as the SOLE authoritative KV store: generation driven by
PagedCache must produce token-identical output to generation driven by HF's DynamicCache.

Because M1 already verified the gather addressing, a failure here points at the storage
write path (or the cache-interface plumbing), not the gather — one new thing to debug.

Run:
    python -m pytest tests/cache/test_paged_cache.py -v
    python -m pytest tests/cache/test_paged_cache.py -v -m cuda
"""

from __future__ import annotations

import pytest
import torch

from engine.cache.paged_cache import PagedCache, PagedLayer


cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MODEL_NAME = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "The capital of France is",
    "Explain KV caching in one sentence.",
    "2 + 2 =",
    "The transformer architecture works by",
]


# ---------------------------------------------------------------------------
# Unit: PagedLayer write/gather on GPU tensors (shape + value correctness)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("chunks", [
    [5],                      # single prefill
    [1, 1, 1, 1, 1],          # pure decode pattern
    [3, 1, 1, 1],             # prefill then decode
    [16, 1],                  # prefill fills a block exactly, then grow
    [20],                     # prefill larger than initial capacity
])
def test_paged_layer_matches_concat(block_size, chunks):
    """PagedLayer.update over chunks == torch.cat of the same chunks (the DynamicLayer behavior)."""
    num_kv_heads, head_dim = 8, 128
    layer = PagedLayer(block_size_tokens=block_size, initial_blocks=1)

    ref_keys = []
    ref_values = []
    out_keys = out_values = None

    pos = 0
    for n in chunks:
        k = torch.randn(1, num_kv_heads, n, head_dim)
        v = torch.randn(1, num_kv_heads, n, head_dim)
        ref_keys.append(k)
        ref_values.append(v)
        out_keys, out_values = layer.update(k, v)
        pos += n

    expected_k = torch.cat(ref_keys, dim=2)
    expected_v = torch.cat(ref_values, dim=2)

    assert out_keys.shape == expected_k.shape, f"{out_keys.shape} != {expected_k.shape}"
    assert torch.equal(out_keys, expected_k), "paged key storage diverged from concat"
    assert torch.equal(out_values, expected_v), "paged value storage diverged from concat"
    assert layer.seq_len == sum(chunks)


def test_paged_layer_grows_at_boundaries():
    """Growth only happens when capacity is exceeded."""
    layer = PagedLayer(block_size_tokens=4, initial_blocks=1)  # capacity 4
    k = torch.randn(1, 8, 4, 128)
    layer.update(k, k)                       # exactly fills block 0
    assert layer.num_blocks == 1
    k1 = torch.randn(1, 8, 1, 128)
    layer.update(k1, k1)                      # needs block 1 -> grow
    assert layer.num_blocks >= 2


# ---------------------------------------------------------------------------
# Integration: PagedCache == DynamicCache reference (token-identical greedy)
# ---------------------------------------------------------------------------

@cuda
@requires_cuda
@pytest.mark.parametrize("block_size", [8, 16, 32])
def test_paged_cache_matches_reference_greedy(block_size):
    """Full generation driven by PagedCache == stock DynamicCache, token for token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    # Keep stock sdpa attention; M2 changes storage, not the attention function.
    model.config._attn_implementation = "sdpa"

    max_new_tokens = 32
    num_layers = model.config.num_hidden_layers

    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        # --- Reference: stock DynamicCache (model.generate default) ---
        with torch.inference_mode():
            ref_out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None,
            )
        ref_tokens = ref_out[0, ids.shape[1]:].tolist()

        # --- Paged storage path: drive generation manually with PagedCache ---
        paged_tokens = _generate_with_paged_cache(
            model, ids, max_new_tokens, num_layers, block_size,
        )

        assert paged_tokens == ref_tokens, (
            f"paged cache diverged from reference\n"
            f"  prompt: {prompt}\n  block_size: {block_size}\n"
            f"  ref:   {ref_tokens}\n  paged: {paged_tokens}"
        )


def _generate_with_paged_cache(model, input_ids, max_new_tokens, num_layers, block_size):
    """Greedy generation using PagedCache as the KV store. Mirrors the explicit runner."""
    eos_ids = set()
    cfg_eos = model.generation_config.eos_token_id
    if isinstance(cfg_eos, int):
        eos_ids.add(cfg_eos)
    elif isinstance(cfg_eos, (list, tuple)):
        eos_ids.update(cfg_eos)

    cache = PagedCache(num_layers=num_layers, block_size_tokens=block_size, initial_blocks=4)
    generated = []

    with torch.inference_mode():
        # Prefill
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True, return_dict=True)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(int(next_token.item()))

        # Decode
        for _ in range(max_new_tokens - 1):
            if generated[-1] in eos_ids:
                break
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True, return_dict=True)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(next_token.item()))

    return generated


@cuda
@requires_cuda
def test_paged_cache_is_authoritative_store():
    """Confirm K,V physically live in our pages: seq_len tracks generation, blocks grow."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    model.config._attn_implementation = "sdpa"

    num_layers = model.config.num_hidden_layers
    cache = PagedCache(num_layers=num_layers, block_size_tokens=16, initial_blocks=1)

    ids = tokenizer("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids.cuda()
    prompt_len = ids.shape[1]

    with torch.inference_mode():
        out = model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
        # After prefill, our cache must hold exactly prompt_len tokens
        assert cache.get_seq_length(0) == prompt_len, (
            f"expected seq_len {prompt_len}, got {cache.get_seq_length(0)}"
        )

        # Decode 20 tokens, verify seq_len grows and blocks were allocated
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for _ in range(20):
            out = model(input_ids=next_token, past_key_values=cache, use_cache=True, return_dict=True)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    snap = cache.snapshot()
    assert snap["seq_len"] == prompt_len + 20
    assert snap["blocks_per_layer"] >= 2, "cache should have grown beyond initial block"
    assert snap["total_growths"] > 0, "cache should have grown at least once"


def test_mask_sizes_are_plain_ints_for_both_cache_interfaces() -> None:
    """transformers >= 4.53 passes a cache_position tensor; older builds pass an int.

    Either way the result must be Python ints: the mask builder evaluates
    `kv_length + kv_offset - width > 0`, which raises on a tensor with several values.
    """
    import torch

    from engine.cache.paged_cache import PagedCache
    from engine.cache.pool_cache import BatchedPoolBackedPrefillCache

    pool = [torch.zeros(2, 4, 1, 2)]
    batched = BatchedPoolBackedPrefillCache(pool, pool, torch.zeros(1, 2, dtype=torch.int32),
                                            torch.tensor([3], dtype=torch.int32), padded_length=3)
    for query in (torch.arange(3), 3):
        kv_length, kv_offset = batched.get_mask_sizes(query, 0)
        assert (kv_length, kv_offset) == (3, 0) and type(kv_length) is int

    paged = PagedCache(num_layers=1, block_size_tokens=4)
    paged.layers[0].seq_len = 5
    for query in (torch.arange(2), 2):
        kv_length, kv_offset = paged.get_mask_sizes(query, 0)
        assert (kv_length, kv_offset) == (7, 0) and type(kv_length) is int
