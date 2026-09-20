from __future__ import annotations

import pytest
import torch


cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
@pytest.mark.parametrize("sequence", [1, 7, 33])
@pytest.mark.parametrize("broadcast_tables", [False, True])
def test_fused_qk_rope_matches_reference(sequence: int, broadcast_tables: bool) -> None:
    from engine.kernels.rope import triton_rope_qk

    torch.manual_seed(31)
    batch, q_heads, k_heads, head_dim = 3, 16, 8, 128
    query = torch.randn(batch, q_heads, sequence, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn(batch, k_heads, sequence, head_dim, device="cuda", dtype=torch.float16)
    table_batch = 1 if broadcast_tables else batch
    cos = torch.randn(table_batch, sequence, head_dim, device="cuda", dtype=torch.float16)
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
@pytest.mark.parametrize("shape", [(16, 3072), (2, 11, 3072)])
def test_fused_swiglu_reads_split_projection_halves_in_place(shape) -> None:
    """The fused gate/up projection yields strided halves; no copy may be needed."""
    from engine.kernels.swiglu import _as_rows, triton_swiglu

    torch.manual_seed(41)
    fused = torch.randn(*shape[:-1], 2 * shape[-1], device="cuda", dtype=torch.float16)
    gate, up = fused.split(shape[-1], dim=-1)
    assert not gate.is_contiguous()
    assert _as_rows(gate).data_ptr() == gate.data_ptr(), "the strided half was copied"
    reference = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(triton_swiglu(gate, up), reference, rtol=2e-3, atol=2e-3)


@cuda
@requires_cuda
def test_qwen_swiglu_installer_covers_every_layer() -> None:
    from transformers import AutoModelForCausalLM
    from engine.kernels.swiglu import install_triton_qwen_swiglu, uninstall_triton_qwen_swiglu

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float16, device_map="cuda", trust_remote_code=True
    ).eval()
    mlps = [module for module in model.modules() if module.__class__.__name__.lower() == "qwen3mlp"]
    hidden = torch.randn(2, model.config.hidden_size, device="cuda", dtype=torch.float16)
    reference = mlps[0](hidden)
    assert install_triton_qwen_swiglu(model, fuse_gate_up=True) == 28
    assert mlps[0].gate_proj is None
    assert mlps[0].up_proj is None
    torch.testing.assert_close(mlps[0](hidden), reference, rtol=3e-3, atol=3e-3)
    assert uninstall_triton_qwen_swiglu(model) == 28
    torch.testing.assert_close(mlps[0](hidden), reference, rtol=2e-3, atol=2e-3)
