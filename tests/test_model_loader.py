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


def test_unknown_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        resolve_dtype("float8", torch.device("cpu"))
