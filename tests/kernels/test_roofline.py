"""The floor must be measured, and the measurement must be sane enough to optimise toward."""

from __future__ import annotations

import pytest
import torch

from benchmarks.kernels.roofline import (
    kv_bytes_per_step, measure_bandwidth, measure_weight_bytes,
)

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@cuda
@requires_cuda
def test_bandwidth_probes_are_plausible_and_consistent():
    probes = {p.name: p for p in measure_bandwidth(size_mb=256)}
    for name, probe in probes.items():
        print(f"  {name:10s} {probe.gb_per_s:7.1f} GB/s")
    # Any modern discrete GPU clears 50 GB/s; nothing clears 10 TB/s. A probe outside
    # that band means the timing or the byte count is wrong, not that the card is exotic.
    for name, probe in probes.items():
        assert 50 < probe.gb_per_s < 10_000, f"{name} reported {probe.gb_per_s:.1f} GB/s"
    # The GEMV probe is the one the floor is built on, so it must not be wildly below a
    # plain sweep - that would mean it is compute-bound and unrepresentative.
    assert probes["gemv"].gb_per_s > 0.4 * probes["reduction"].gb_per_s


@cuda
@requires_cuda
def test_decode_step_reads_lm_head_but_not_the_whole_embedding_table():
    from engine.model import load_model

    loaded = load_model("Qwen/Qwen3-0.6B")
    weights = measure_weight_bytes(loaded.model)
    print(f"  layers={weights.layers/1e6:.1f}MB lm_head={weights.lm_head/1e6:.1f}MB "
          f"other={weights.other/1e6:.1f}MB tied={weights.lm_head_is_tied}")
    assert weights.layers > 0 and weights.lm_head > 0
    # Qwen3-0.6B ties lm_head to the embedding table: same memory, read in full each step
    # for logits, gathered a row at a time for input embeddings.
    assert weights.lm_head_is_tied
    total = weights.read_per_decode_step
    assert total > weights.layers, "lm_head must be counted in the per-step read"
    assert total < weights.layers + weights.lm_head + weights.other + weights.embedding_table


@cuda
@requires_cuda
def test_kv_traffic_scales_with_batch_and_context():
    from engine.model import load_model

    model = load_model("Qwen/Qwen3-0.6B").model
    base = kv_bytes_per_step(model, batch=1, context_tokens=128)
    assert kv_bytes_per_step(model, batch=4, context_tokens=128) == 4 * base
    assert kv_bytes_per_step(model, batch=1, context_tokens=512) == 4 * base
    # At short context KV traffic must be small beside the weights, which is the premise
    # of calling a small-batch decode step weight-bound in the first place.
    weights = measure_weight_bytes(model).read_per_decode_step
    assert kv_bytes_per_step(model, batch=8, context_tokens=128) < weights
