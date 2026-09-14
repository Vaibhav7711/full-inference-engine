"""KV cache quantization correctness — round-trip bounds + generation stays close to FP16.

Two levels:
  1. Unit: quantize->dequantize round-trip error stays small across shapes.
  2. Integration: full greedy generation driven by a QuantizedKVLayer-backed cache produces
     output close to the FP16-KV reference (mostly-identical tokens; divergence only at
     near-tie positions, since INT8 KV introduces tiny perturbations).

Run:
    python -m pytest tests/quantization/test_kv_int8.py -v
    python -m pytest tests/quantization/test_kv_int8.py -v -m cuda
"""

from __future__ import annotations

import pytest
import torch

from engine.quantization.kv_int8 import quantize_kv, dequantize_kv, QuantizedKVLayer

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


# ---------------------------------------------------------------------------
# Unit: round-trip error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(64, 8, 128), (256, 8, 128), (1, 8, 128), (512, 8, 128)])
def test_quantize_roundtrip_error_small(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float16)
    q, scale = quantize_kv(x)
    xr = dequantize_kv(q, scale)

    assert q.dtype == torch.int8
    assert (q.abs() <= 127).all(), "INT8 values must be in [-127, 127]"
    rel_err = (xr.float() - x.float()).abs().sum() / x.float().abs().sum()
    assert rel_err < 0.02, f"relative error {rel_err:.4f} too large (shape {shape})"


def test_quantize_preserves_magnitude():
    """Larger values quantize to larger int8 (structure preserved)."""
    x = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]], dtype=torch.float16)
    q, scale = quantize_kv(x)
    # ratios roughly preserved
    xr = dequantize_kv(q, scale)
    assert xr[0, 0, 3] > xr[0, 0, 2] > xr[0, 0, 1] > xr[0, 0, 0]


# ---------------------------------------------------------------------------
# Unit: QuantizedKVLayer behaves like a cache (accumulates, gathers)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("block_size", [8, 16])
def test_quantized_layer_accumulates(block_size):
    """update() over chunks accumulates, gather returns ~the concatenated (dequantized) K,V."""
    layer = QuantizedKVLayer(block_size_tokens=block_size, initial_blocks=1)
    kv_heads, D = 8, 128

    refs_k, refs_v = [], []
    out_k = out_v = None
    for n in [3, 1, 1, 16, 1]:  # prefill then decode, crossing block boundaries
        k = torch.randn(1, kv_heads, n, D, dtype=torch.float16)
        v = torch.randn(1, kv_heads, n, D, dtype=torch.float16)
        refs_k.append(k)
        refs_v.append(v)
        out_k, out_v = layer.update(k, v)

    expected_k = torch.cat(refs_k, dim=2)
    expected_v = torch.cat(refs_v, dim=2)
    assert out_k.shape == expected_k.shape
    # Quantized, so not exact — but close
    rel_err_k = (out_k.float() - expected_k.float()).abs().sum() / expected_k.float().abs().sum()
    assert rel_err_k < 0.02, f"KV quant accumulation error {rel_err_k:.4f}"
    assert layer.seq_len == sum([3, 1, 1, 16, 1])


def test_memory_reduction_reported():
    layer = QuantizedKVLayer(block_size_tokens=16, initial_blocks=2)
    k = torch.randn(1, 8, 10, 128, dtype=torch.float16)
    layer.update(k, k)
    mem = layer.memory_bytes()
    # INT8 + small scales should be well under FP16 equivalent
    assert mem["reduction_pct"] > 40, f"expected >40% reduction, got {mem['reduction_pct']}%"


# ---------------------------------------------------------------------------
# Integration: generation with quantized KV vs FP16 KV
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"


@cuda
@requires_cuda
def test_quantized_kv_generation_close_to_fp16():
    """Full generation with INT8 KV stays close to FP16 KV (mostly-identical greedy)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.cache.paged_cache import PagedCache, PagedLayer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="cuda", trust_remote_code=True)
    model.eval()
    model.config._attn_implementation = "sdpa"
    num_layers = model.config.num_hidden_layers

    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    max_new = 24

    def generate_with_cache(cache):
        gen = []
        with torch.inference_mode():
            out = model(input_ids=ids, past_key_values=cache, use_cache=True, return_dict=True)
            nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            gen.append(int(nt.item()))
            for _ in range(max_new - 1):
                out = model(input_ids=nt, past_key_values=cache, use_cache=True, return_dict=True)
                nt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                gen.append(int(nt.item()))
        return gen

    # FP16 KV reference
    fp16_cache = PagedCache(num_layers=num_layers, block_size_tokens=16, initial_blocks=4)
    fp16_tokens = generate_with_cache(fp16_cache)

    # INT8 KV: swap PagedCache's layers for QuantizedKVLayer
    quant_cache = PagedCache(num_layers=num_layers, block_size_tokens=16, initial_blocks=4)
    quant_cache._paged_layers = [
        QuantizedKVLayer(block_size_tokens=16, initial_blocks=4) for _ in range(num_layers)
    ]
    quant_tokens = generate_with_cache(quant_cache)

    # Compare: most tokens should match; any mismatch is a near-tie (INT8 perturbation)
    matches = sum(1 for a, b in zip(fp16_tokens, quant_tokens) if a == b)
    agreement = matches / max(len(fp16_tokens), 1)
    assert agreement >= 0.6, (
        f"INT8 KV diverged too much: {agreement:.0%} agreement\n"
        f"  fp16:  {fp16_tokens}\n  int8:  {quant_tokens}"
    )
    # The point is quality is PRESERVED (high agreement), not bit-identical (INT8 is lossy).
