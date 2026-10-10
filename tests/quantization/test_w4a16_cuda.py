"""W4A16 installed on a real checkpoint, on CUDA: kernel vs reference, engine vs gate.

These run on the 0.6B so they are cheap, and they are the gate-2 proof that the 8B path
works before a session is spent loading it. A separate copy is loaded rather than the
shared `loaded` fixture, because installation replaces modules in place.
"""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(scope="module")
def w4():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from engine.model import load_model

    loaded = load_model("Qwen/Qwen3-0.6B", quantize="w4a16")
    assert loaded.quantization == "w4a16"
    assert loaded.quantization_report["replaced"] == 28 * 7 + 1
    yield loaded
    del loaded
    torch.cuda.empty_cache()


@cuda
@requires_cuda
@pytest.mark.parametrize("shape, path", [((1, 8), "kernel"), ((2, 24), "dense")],
                         ids=["M=8 triton kernel", "M=48 cuBLAS over dequantised fp16"])
def test_installed_path_agrees_with_torch_over_the_same_packed_weights(w4, shape, path):
    from engine.quantization.w4a16 import kernel_available, reference_mode, w4a16_modules

    assert kernel_available(), "Triton kernel must be the live path on a CUDA box"
    modules = w4a16_modules(w4.model)
    assert not any(m.reference for m in modules)
    rows = shape[0] * shape[1]
    # Below the threshold the Triton GEMV runs; at or above it, cuBLAS over a dequantised
    # tile (on sm_75 tl.dot is FMA, so wide batches must reach the tensor cores).
    assert all((rows < m.dense_threshold) == (path == "kernel") for m in modules)
    torch.manual_seed(0)
    ids = torch.randint(1000, 100_000, shape, device=w4.device)
    with torch.inference_mode():
        kernel = w4.model(input_ids=ids, use_cache=False).logits.float()
        with reference_mode(w4.model):
            reference = w4.model(input_ids=ids, use_cache=False).logits.float()
    # Same nibbles, same scales; only the GEMV's fp16 accumulation order differs across
    # 28 layers. Logits are O(10), so this is a few-ulp-per-layer budget.
    torch.testing.assert_close(kernel, reference, rtol=2e-2, atol=1e-1)
    agree = (kernel.argmax(-1) == reference.argmax(-1)).float().mean().item()
    assert agree > 0.95, f"argmax agreement {agree:.3f}"


@cuda
@requires_cuda
def test_engine_on_w4a16_passes_the_identity_gate_against_its_reference(w4):
    from benchmarks.reliability.ab import IDENTITY_PROMPTS, _first_divergence, _stock_reference
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    # The gate's reference: stock attention, stock RoPE, torch over the packed weights.
    reference, margins = _stock_reference(w4, IDENTITY_PROMPTS, 24)
    engine = ContinuousBatchingEngine(
        w4.model, w4.tokenizer, w4.device, max_active=4, num_blocks=256, block_size=16,
        prefix_cache_blocks=0, cuda_graph_batch_sizes=(1, 2, 4),
    )
    engine.warmup()
    outputs = engine.generate(list(IDENTITY_PROMPTS), max_new_tokens=24)
    divergence = _first_divergence(outputs, reference, margins)
    early = [i for i in divergence if i is not None and i < 8]
    assert not early, f"W4A16 engine diverges from its reference within 8 tokens: {divergence}"
    assert engine.lazy_graph_captures == 0
    report = engine.graph_replay_report()
    assert report["captured"] > 0 and report["replays_total"] > 0
