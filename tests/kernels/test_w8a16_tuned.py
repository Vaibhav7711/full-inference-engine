"""The searched W8A16 dispatch matches the reference dequantized matmul at every bucket."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
@pytest.mark.parametrize("rows", [1, 7, 16, 33, 64, 100, 256])
@pytest.mark.parametrize("shape", [(6144, 2048), (2048, 6144), (3072, 1024)])
def test_dispatched_w8a16_matches_dequantized_reference(rows, shape) -> None:
    from engine.kernels.w8a16_linear import quantize_weight_per_channel, w8a16_linear
    from engine.kernels.w8a16_tuned import w8a16_linear_dispatched

    torch.manual_seed(rows)
    n, k = shape
    weight = (torch.randn(n, k, device="cuda") * 0.02).half()
    qweight, scales = quantize_weight_per_channel(weight)
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    reference = x.float() @ (qweight.float() * scales.float()[:, None]).T
    tuned = w8a16_linear_dispatched(x, qweight, scales).float()
    current = w8a16_linear(x, qweight, scales).float()
    scale = reference.abs().mean()
    assert float((tuned - reference).abs().max() / scale) < 1e-2
    # No worse than the kernel it replaces.
    assert float((tuned - reference).abs().max()) <= float((current - reference).abs().max()) * 1.5 + 1e-3


@cuda
@requires_cuda
def test_install_tuned_w8a16_round_trips_the_class() -> None:
    from torch import nn

    from engine.kernels.w8a16_linear import W8A16Linear
    from engine.kernels.w8a16_tuned import TunedW8A16Linear, install_tuned_w8a16

    model = nn.Sequential(W8A16Linear(nn.Linear(64, 32, bias=False).cuda().half()))
    assert install_tuned_w8a16(model) == 1 and type(model[0]) is TunedW8A16Linear
    x = torch.randn(3, 64, device="cuda", dtype=torch.float16)
    tuned = model(x)
    assert install_tuned_w8a16(model, enabled=False) == 1 and type(model[0]) is W8A16Linear
    torch.testing.assert_close(tuned, model(x), atol=2e-2, rtol=2e-2)


@cuda
@requires_cuda
def test_quantize_lm_head_w8a16_keeps_the_fp16_embedding_and_matches_logits() -> None:
    from torch import nn

    from engine.kernels.w8a16_tuned import TunedW8A16Linear, quantize_lm_head_w8a16

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(40000, 256).cuda().half()
            self.lm_head = nn.Linear(256, 40000, bias=False).cuda().half()
            self.lm_head.weight = self.embed_tokens.weight  # tied, as in Qwen3

    model = Tiny()
    x = torch.randn(5, 256, device="cuda", dtype=torch.float16)
    reference = model.lm_head(x).float()
    modules = quantize_lm_head_w8a16(model)
    assert type(model.lm_head) is TunedW8A16Linear and modules["fp16"] is not None
    assert model.embed_tokens.weight.dtype is torch.float16  # lookups untouched
    quantized = model.lm_head(x).float()
    lm_head = model.lm_head
    exact = x.float() @ (lm_head.qweight.float() * lm_head.scales.float()[:, None]).T
    # Kernel correctness against the exact dequantized weight; INT8 quantization noise
    # against the FP16 logits is reported by the teacher-forced quality benchmark.
    assert float((quantized - exact).abs().max() / exact.abs().mean()) < 1e-2
    assert float((quantized - reference).abs().mean() / reference.abs().mean()) < 2e-2
    assert quantize_lm_head_w8a16(model)["fp16"] is None  # idempotent


@cuda
@requires_cuda
def test_quantize_attention_w8a16_swaps_every_projection_and_restores() -> None:
    from torch import nn

    from engine.kernels.w8a16_tuned import TunedW8A16Linear, quantize_attention_w8a16, swap_projections

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(256, 512, bias=False)
            self.k_proj = nn.Linear(256, 128, bias=False)
            self.v_proj = nn.Linear(256, 128, bias=False)
            self.o_proj = nn.Linear(512, 256, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()

    model = nn.ModuleDict({"layers": nn.ModuleList([Layer(), Layer()])}).cuda().half()
    x = torch.randn(5, 256, device="cuda", dtype=torch.float16)
    before = model["layers"][1].self_attn.q_proj(x).float()
    mapping = quantize_attention_w8a16(model)
    assert len(mapping) == 8 and all(type(w) is TunedW8A16Linear for _, w in mapping.values())
    after = model["layers"][1].self_attn.q_proj(x).float()
    assert float((after - before).abs().mean() / before.abs().mean()) < 2e-2
    swap_projections(model, mapping, use_w8a16=False)
    torch.testing.assert_close(model["layers"][1].self_attn.q_proj(x).float(), before)
    swap_projections(model, mapping, use_w8a16=True)
    assert type(model["layers"][0].self_attn.o_proj) is TunedW8A16Linear
    # Idempotent: already-quantized projections are skipped rather than re-quantized.
    assert quantize_attention_w8a16(model, ("q_proj",)) == {}
