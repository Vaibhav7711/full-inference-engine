import torch
from torch import nn

from engine.quantization import Int8Linear, model_storage_bytes, quantize_linear_modules


def test_int8_linear_tracks_reference_with_bounded_error() -> None:
    torch.manual_seed(0)
    reference = nn.Linear(16, 8, bias=True, dtype=torch.float32)
    quantized = Int8Linear(reference)
    inputs = torch.randn(4, 16)
    assert torch.allclose(quantized(inputs), reference(inputs), atol=0.02, rtol=0.02)


def test_recursive_quantization_replaces_linear_modules_and_reduces_storage() -> None:
    model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8)).to(torch.float16)
    before = model_storage_bytes(model)
    assert quantize_linear_modules(model) == 2
    assert all(not isinstance(module, nn.Linear) for module in model.modules())
    assert model_storage_bytes(model) < before
