"""The measured RTX 4060 profiles build, warm up with every graph, and generate sanely."""

from __future__ import annotations

import gc

import pytest
import torch

cuda = pytest.mark.cuda
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_profile_names_and_block_defaults_are_consistent() -> None:
    from engine.profiles import PROFILES, default_num_blocks

    assert PROFILES == ("optimized", "lossless")
    for model in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "models/Qwen3-4B-AWQ"):
        for profile in PROFILES:
            assert default_num_blocks(model, profile) > 0
    # An INT8 pool holds about twice the tokens per byte, so the optimized profiles for
    # the FP16-weight models get more blocks than their lossless twins.
    assert default_num_blocks("Qwen/Qwen3-0.6B", "optimized") > default_num_blocks("Qwen/Qwen3-0.6B", "lossless")


def test_calibration_corpus_yields_fixed_length_batches() -> None:
    from engine.quantization.calibration_corpus import CALIBRATION_TEXT, default_calibration_batches

    class Tok:
        def __call__(self, text):
            return type("R", (), {"input_ids": list(range(len(text.split())))})()

    batches = default_calibration_batches(Tok(), length=64, count=3)
    assert [len(batch) for batch in batches] == [64, 64, 64]
    assert len(CALIBRATION_TEXT.split()) > 800


@cuda
@requires_cuda
@pytest.mark.parametrize("profile", ["lossless", "optimized"])
def test_profile_engine_warms_up_and_generates(profile: str) -> None:
    from engine.profiles import ProfileReport, build_engine

    report = ProfileReport("", "", 0, "", 0)
    engine = build_engine("Qwen/Qwen3-0.6B", profile=profile, max_active=2, num_blocks=96,
                          graph_buckets=(1, 2), report=report)
    try:
        assert report.kv_dtype == ("int8" if profile == "optimized" else "fp16")
        assert engine.lazy_graph_captures == 0
        if profile == "optimized":
            assert engine.int8_prefill_attention == "tiled"
        else:
            assert engine.prefill_attention == "tiled"
        assert engine.packed_fused_prefill
        out = engine.generate(["The capital of France is"], max_new_tokens=6)[0]
        assert len(out) == 6
        text = engine.tokenizer.decode(out)
        assert "Paris" in text, text
        # A greedy prompt publishes its exact KV prefix, so the cache intentionally
        # retains one or more blocks after the request allocation itself is released.
        expected = len(engine._graph_dummy_blocks) + engine.prefix_cache.snapshot()["cached_blocks"]
        assert engine.block_manager.snapshot()["used_blocks"] == expected
    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
