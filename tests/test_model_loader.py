from __future__ import annotations

import pytest
import torch

from engine.model.loader import resolve_dtype


def test_explicit_dtype_resolution() -> None:
    device = torch.device("cpu")
    assert resolve_dtype("float16", device) is torch.float16
    assert resolve_dtype("bfloat16", device) is torch.bfloat16
    assert resolve_dtype(torch.float32, device) is torch.float32


def test_auto_uses_float32_without_cuda() -> None:
    assert resolve_dtype("auto", torch.device("cpu")) is torch.float32


def test_auto_uses_float16_on_turing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (7, 5))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_dtype("auto", torch.device("cuda")) is torch.float16


def test_auto_uses_bfloat16_when_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_dtype("auto", torch.device("cuda")) is torch.bfloat16


def test_auto_honors_measured_rtx4060_fp16_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 9))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_dtype("auto", torch.device("cuda")) is torch.float16


def test_unknown_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        resolve_dtype("float8", torch.device("cpu"))


def test_tie_output_embeddings_shares_storage_only_when_identical() -> None:
    import torch
    from torch import nn

    from engine.model.loader import tie_output_embeddings

    class Tiny(nn.Module):
        def __init__(self, tie: bool, same_values: bool):
            super().__init__()
            self.config = type("Config", (), {"tie_word_embeddings": tie})()
            self.embed = nn.Embedding(8, 4)
            self.head = nn.Linear(4, 8, bias=False)
            with torch.no_grad():
                self.head.weight.copy_(self.embed.weight if same_values else self.embed.weight + 1)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    model = Tiny(tie=True, same_values=True)
    assert tie_output_embeddings(model)
    assert model.head.weight.data_ptr() == model.embed.weight.data_ptr()

    model = Tiny(tie=True, same_values=False)
    assert not tie_output_embeddings(model)
    assert model.head.weight.data_ptr() != model.embed.weight.data_ptr()

    model = Tiny(tie=False, same_values=True)
    assert not tie_output_embeddings(model)
