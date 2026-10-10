"""W4A16 installed on a model, and loaded without an fp16 detour - on CPU, no Triton.

A two-layer Qwen3 the size of a thumbnail (hidden 64, intermediate 128, head_dim 16,
vocab 256) is enough to exercise every path: structural discovery of the 15 target
Linears (7 per layer + the head), in-place quantisation, the reference matmul, the
`meta`-device streaming loader against a `save_pretrained` folder, rotary re-init, and
the tied-head case. The reference path is what the Triton kernel is tested against on
CUDA, so agreement here is the contract the GPU inherits.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from engine.kernels.w4a16_format import unpack_w4_reference  # noqa: E402
from engine.quantization.w4a16 import (  # noqa: E402
    W4A16Linear, describe, install_w4a16, load_w4a16_model, reference_mode, w4a16_modules,
)

GROUP = 32


def _tiny_config(tie: bool = False):
    from transformers import Qwen3Config

    return Qwen3Config(
        hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=256, max_position_embeddings=128, tie_word_embeddings=tie,
    )


def _tiny_model(tie: bool = False):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_tiny_config(tie), dtype=torch.float32)
    return model.eval()


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False).logits


def test_install_replaces_every_projection_and_the_head():
    model = _tiny_model()
    report = install_w4a16(model, group_size=GROUP)
    assert report.replaced == 2 * 7 + 1
    assert report.by_kind == {"attention": 8, "mlp": 6, "lm_head": 1}
    assert report.skipped == []
    assert all(isinstance(m, W4A16Linear) for m in w4a16_modules(model))
    assert len(w4a16_modules(model)) == 15
    assert 3.0 < report.ratio < 4.0          # fp16 bytes over packed+scales bytes
    assert describe(model)["w4a16_modules"] == 15


def test_installed_layer_matches_linear_over_its_own_dequantised_weight():
    model = _tiny_model()
    layer = model.model.layers[0].mlp.down_proj
    weight = layer.weight.detach().clone()
    install_w4a16(model, group_size=GROUP)
    shell = model.model.layers[0].mlp.down_proj
    assert isinstance(shell, W4A16Linear)
    x = torch.randn(3, 5, shell.in_features)
    dequantised = unpack_w4_reference(shell.packed, shell.scales, shell.in_features, GROUP).float()
    torch.testing.assert_close(shell(x), F.linear(x, dequantised), rtol=1e-5, atol=1e-5)
    # And the dequantised weight is the quantisation of the original, within a half step.
    error = (dequantised - weight).abs().reshape(shell.out_features, -1, GROUP).amax(-1)
    assert torch.all(error <= shell.scales.float() / 2 * 1.01 + 1e-3)


def test_whole_model_runs_and_is_close_to_fp32():
    model = _tiny_model()
    ids = torch.randint(0, 256, (2, 9))
    before = _logits(model, ids)
    install_w4a16(model, group_size=GROUP)
    after = _logits(model, ids)
    assert after.shape == before.shape and torch.isfinite(after).all()
    # 4-bit on a random-init thumbnail is lossy; the check is that it is the same model.
    assert torch.corrcoef(torch.stack([before.flatten(), after.flatten()]))[0, 1] > 0.9


def test_reference_mode_is_a_context_and_restores():
    model = _tiny_model()
    install_w4a16(model, group_size=GROUP)
    modules = w4a16_modules(model)
    assert not any(m.reference for m in modules)
    with reference_mode(model):
        assert all(m.reference for m in modules)
    assert not any(m.reference for m in modules)


def test_streaming_loader_matches_install_and_leaves_nothing_on_meta(tmp_path):
    # The loader serves fp16, so it casts each tensor to fp16 before quantising. Give the
    # in-place installer the same fp16 bits, or the two paths quantise different inputs.
    model = _tiny_model().to(torch.float16)
    model.save_pretrained(tmp_path, safe_serialization=True)
    ids = torch.randint(0, 256, (1, 7))
    install_w4a16(model, group_size=GROUP)
    expected = _logits(model.float(), ids)

    loaded, report = load_w4a16_model(str(tmp_path), device="cpu", group_size=GROUP)
    assert report.replaced == 15 and report.skipped == []
    assert not any(p.device.type == "meta" for p in loaded.parameters())
    assert not any(b.device.type == "meta" for b in loaded.buffers())
    # Rotary tables were rebuilt on the real device, not left as meta placeholders.
    rotary = [m for m in loaded.modules() if hasattr(m, "inv_freq")]
    assert rotary and all(m.inv_freq.device.type == "cpu" for m in rotary)
    # Same quantiser, same weights: the two paths agree exactly.
    torch.testing.assert_close(_logits(loaded.float(), ids), expected, rtol=1e-4, atol=1e-4)


def test_streaming_loader_quantises_a_tied_head_from_the_embedding(tmp_path):
    model = _tiny_model(tie=True)
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded, report = load_w4a16_model(str(tmp_path), device="cpu", group_size=GROUP)
    head = loaded.get_output_embeddings()
    assert isinstance(head, W4A16Linear) and head.loaded
    assert report.by_kind["lm_head"] == 1
    ids = torch.randint(0, 256, (1, 5))
    assert torch.isfinite(_logits(loaded.float(), ids)).all()


def test_group_size_that_does_not_divide_is_skipped_not_crashed():
    model = _tiny_model()
    report = install_w4a16(model, group_size=48)      # 64 % 48 != 0, 128 % 48 != 0
    assert report.replaced == 0
    assert len(report.skipped) == 15
