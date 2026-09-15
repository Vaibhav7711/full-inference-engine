from __future__ import annotations

import pytest
import torch


cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
@pytest.mark.parametrize(
    "shape",
    [
        (16, 1024),
        (16, 8, 1, 128),
        (1, 23, 1024),
    ],
)
def test_triton_rmsnorm_matches_fp32_reference(shape) -> None:
    from engine.kernels.rmsnorm import triton_rmsnorm

    torch.manual_seed(29)
    hidden_states = torch.randn(*shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
    epsilon = 1e-6

    values = hidden_states.float()
    reference = (
        values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + epsilon)
    ).to(hidden_states.dtype) * weight
    actual = triton_rmsnorm(hidden_states, weight, epsilon)

    torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)


@cuda
@requires_cuda
def test_qwen_rmsnorm_installer_matches_original_modules() -> None:
    from transformers import AutoModelForCausalLM

    from engine.kernels.rmsnorm import install_triton_rmsnorm, uninstall_triton_rmsnorm

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float16, device_map="cuda", trust_remote_code=True
    ).eval()
    rmsnorm_modules = [
        module
        for module in model.modules()
        if module.__class__.__name__.lower().endswith("rmsnorm")
    ]
    assert len(rmsnorm_modules) == 113

    samples = []
    for module in rmsnorm_modules:
        width = module.weight.numel()
        hidden_states = torch.randn(4, width, device="cuda", dtype=torch.float16)
        samples.append((module, hidden_states, module(hidden_states)))

    assert install_triton_rmsnorm(model) == 113
    for module, hidden_states, reference in samples:
        torch.testing.assert_close(
            module(hidden_states), reference, rtol=2e-3, atol=2e-3
        )
    assert uninstall_triton_rmsnorm(model) == 113
