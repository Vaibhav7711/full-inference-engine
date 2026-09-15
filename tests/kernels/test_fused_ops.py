from __future__ import annotations

import pytest
import torch


cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
@pytest.mark.parametrize("sequence", [1, 7, 33])
def test_fused_qk_rope_matches_reference(sequence: int) -> None:
    from engine.kernels.rope import triton_rope_qk

    torch.manual_seed(31)
    batch, q_heads, k_heads, head_dim = 3, 16, 8, 128
    query = torch.randn(batch, q_heads, sequence, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn(batch, k_heads, sequence, head_dim, device="cuda", dtype=torch.float16)
    cos = torch.randn(batch, sequence, head_dim, device="cuda", dtype=torch.float16)
    sin = torch.randn_like(cos)

    def reference(tensor):
        first, second = tensor.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return tensor * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)

    query_out, key_out = triton_rope_qk(query, key, cos, sin)
    torch.testing.assert_close(query_out, reference(query), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(key_out, reference(key), rtol=2e-3, atol=2e-3)


@cuda
@requires_cuda
@pytest.mark.parametrize("shape", [(16, 3072), (2, 11, 3072)])
def test_fused_swiglu_matches_reference(shape) -> None:
    from engine.kernels.swiglu import triton_swiglu

    torch.manual_seed(37)
    gate = torch.randn(*shape, device="cuda", dtype=torch.float16)
    up = torch.randn_like(gate)
    reference = torch.nn.functional.silu(gate) * up
    actual = triton_swiglu(gate, up)
    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


@cuda
@requires_cuda
def test_qwen_swiglu_installer_covers_every_layer() -> None:
    from transformers import AutoModelForCausalLM
    from engine.kernels.swiglu import install_triton_qwen_swiglu, uninstall_triton_qwen_swiglu

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float16, device_map="cuda", trust_remote_code=True
    ).eval()
    assert install_triton_qwen_swiglu(model) == 28
    assert uninstall_triton_qwen_swiglu(model) == 28
