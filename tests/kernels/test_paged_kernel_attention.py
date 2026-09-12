"""K3b correctness: paged kernel reading live PagedCache blocks == stock sdpa reference.

This is the K3b gate — the culmination. Full greedy generation where the K2 kernel is
the model's attention function, reading K,V directly from the PagedCache's physical page
tensors, must produce token-identical output to stock sdpa generation.

If this passes:
    - K1 attention math is correct (already proven)
    - K2 block addressing is correct (already proven)
    - K3b: the kernel reads live paged blocks during generation and matches reference
    => a real paged-attention path, kernel over physical blocks, end to end.

Run:
    python -m pytest tests/kernels/test_paged_kernel_attention.py -v -m cuda
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

MODEL_NAME = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "The capital of France is",
    "Explain KV caching in one sentence.",
    "2 + 2 =",
    "The transformer architecture works by",
]


def _generate_stock(model, tokenizer, ids, max_new_tokens):
    """Reference: stock sdpa + default DynamicCache via model.generate."""
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             temperature=None, top_p=None)
    return out[0, ids.shape[1]:].tolist()


def _generate_paged_kernel(model, tokenizer, ids, max_new_tokens, num_layers, block_size):
    """Drive generation with the paged KERNEL attention over a live PagedCache."""
    from engine.cache.paged_cache import PagedCache
    from engine.kernels.paged_kernel_attention import (
        enable_paged_kernel_attention, set_active_paged_cache, clear_active_paged_cache,
        reset_kernel_call_count, kernel_call_count,
    )

    eos_ids = set()
    ce = model.generation_config.eos_token_id
    if isinstance(ce, int):
        eos_ids.add(ce)
    elif isinstance(ce, (list, tuple)):
        eos_ids.update(ce)

    enable_paged_kernel_attention(model)
    cache = PagedCache(num_layers=num_layers, block_size_tokens=block_size, initial_blocks=4)
    set_active_paged_cache(cache)
    reset_kernel_call_count()

    generated = []
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
            nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(int(nt.item()))
            for _ in range(max_new_tokens - 1):
                if generated[-1] in eos_ids:
                    break
                out = model(input_ids=nt, past_key_values=cache, use_cache=True, return_dict=True)
                nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(int(nt.item()))
    finally:
        clear_active_paged_cache()

    calls = kernel_call_count()
    return generated, calls


@cuda
@requires_cuda
@pytest.mark.parametrize("block_size", [16, 32])
def test_paged_kernel_matches_stock(block_size):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    max_new_tokens = 32

    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        ref = _generate_stock(model, tokenizer, ids, max_new_tokens)
        paged, calls = _generate_paged_kernel(
            model, tokenizer, ids, max_new_tokens, num_layers, block_size,
        )

        assert calls > 0, "paged kernel attention was never called"
        assert paged == ref, (
            f"paged kernel diverged from stock sdpa\n"
            f"  prompt: {prompt}\n  block_size: {block_size}\n"
            f"  ref:   {ref}\n  paged: {paged}"
        )


@cuda
@requires_cuda
def test_paged_kernel_actually_runs_per_layer():
    """Confirm the kernel executes once per layer per forward pass (not stock sdpa)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.cache.paged_cache import PagedCache
    from engine.kernels.paged_kernel_attention import (
        enable_paged_kernel_attention, set_active_paged_cache, clear_active_paged_cache,
        reset_kernel_call_count, kernel_call_count,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    num_layers = model.config.num_hidden_layers

    enable_paged_kernel_attention(model)
    cache = PagedCache(num_layers=num_layers, block_size_tokens=16, initial_blocks=4)
    set_active_paged_cache(cache)
    reset_kernel_call_count()

    ids = tokenizer("Hello world", return_tensors="pt").input_ids.cuda()
    try:
        with torch.inference_mode():
            model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
    finally:
        clear_active_paged_cache()

    # One prefill forward => one kernel call per layer
    assert kernel_call_count() == num_layers, (
        f"expected {num_layers} kernel calls, got {kernel_call_count()}"
    )
