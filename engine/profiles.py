"""Measured RTX 4060 serving profiles, one switch per model.

Every setting here was accepted by an interleaved end-to-end A/B on the RTX 4060 and the
stock-token gate (docs/rtx4060-rnd-priorities.md). The engine's own defaults stay
conservative; a profile turns on the opt-in paths that won for a given checkpoint:

- ``optimized``: the fastest measured configuration. It is lossy: INT8 KV with calibrated
  key smoothing, W8A16 MLP and vocabulary projections on Qwen3-0.6B and 1.7B, and W8A16
  attention projections on 1.7B (rejected on 0.6B by the early-token gate). Expect about
  97-98% top-1 agreement with the FP16 model and unchanged perplexity.
- ``lossless``: FP16 weights and KV with every lossless scheduling win (tiled prefill,
  packed fused rows, idle whole-prompt prefill).

    from engine.profiles import build_engine
    engine = build_engine("Qwen/Qwen3-1.7B", profile="optimized", max_active=8)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

PROFILES = ("optimized", "lossless")


@dataclass
class ProfileReport:
    model_name: str
    profile: str
    num_blocks: int
    kv_dtype: str
    weights_bytes: int
    engine_kwargs: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _is_awq(model_name: str) -> bool:
    return "awq" in model_name.lower()


def default_num_blocks(model_name: str, profile: str) -> int:
    """Pool sizes that fit the 8 GB card with the profile's weights and graph pools."""
    if _is_awq(model_name):
        return 1024                      # FP16 KV, 2.54 GiB weights: ~5.0 GiB total
    if "1.7b" in model_name.lower():
        return 2048 if profile == "optimized" else 1024
    return 2560 if profile == "optimized" else 1536   # 0.6B


def build_engine(model_name: str, *, profile: str = "optimized", max_active: int = 8,
                 num_blocks: int | None = None, block_size: int = 16,
                 graph_buckets: tuple[int, ...] = (1, 2, 4, 8),
                 max_waiting_requests: int | None = None, prefix_cache_blocks: int = 64,
                 warmup: bool = True, report: ProfileReport | None = None, **overrides):
    """Load the checkpoint and build the engine for `profile`; see the module docstring."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    from engine.batching.continuous_batching import ContinuousBatchingEngine

    notes: list[str] = []
    kwargs: dict = dict(packed_fused_prefill=True)
    if _is_awq(model_name):
        from engine.model import load_awq_tinygemm_model

        loaded = load_awq_tinygemm_model(model_name)   # Triton int4pack prefill by default
        # Width buckets are what makes AWQ prefill graphs a measured win (ITL p999 -34%).
        kwargs["packed_prefill_width_buckets"] = True
        notes.append("AWQ: Triton int4pack prefill for >= 64 rows, packed rows with width "
                     "buckets so prefill/fused graphs are captured")
        if profile == "optimized":
            notes.append("INT8 KV is not yet validated on the 4B checkpoint; FP16 KV kept")
    else:
        from engine.model import load_model

        loaded = load_model(model_name, dtype="float16")
        if profile == "optimized":
            from engine.kernels.w8a16_linear import quantize_mlp_w8a16
            from engine.kernels.w8a16_tuned import quantize_attention_w8a16, quantize_lm_head_w8a16

            quantize_mlp_w8a16(loaded.model)
            quantize_lm_head_w8a16(loaded.model)
            notes.append("W8A16 MLP (searched dispatch) and W8A16 lm_head")
            if "1.7b" in model_name.lower():
                # Decode -13% to -15% end to end on 1.7B; on 0.6B the same change flipped a
                # token at position 1 (margin 0.078) and is rejected by the early-token gate.
                quantize_attention_w8a16(loaded.model)
                notes.append("W8A16 q/k/v/o attention projections")
            torch.cuda.empty_cache()
            from engine.quantization.calibration_corpus import default_calibration_batches
            from engine.quantization.kv_smoothing import calibrate_key_smoothing

            smoothing = calibrate_key_smoothing(
                loaded.model, default_calibration_batches(loaded.tokenizer))
            kwargs.update(kv_cache_dtype="int8", kv_key_smoothing=smoothing,
                          int8_prefill_attention="tiled", int8_decode_tuned=True)
            notes.append("INT8 KV with calibrated key smoothing, tiled INT8 prefill, "
                         "tuned INT8 decode launch")
    kwargs.update(overrides)
    blocks = num_blocks if num_blocks is not None else default_num_blocks(model_name, profile)
    buckets = tuple(size for size in graph_buckets if size <= max_active) or (1,)
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, loaded.device, num_blocks=blocks, block_size=block_size,
        max_active=max_active, cuda_graph_batch_sizes=buckets,
        max_waiting_requests=max_waiting_requests, prefix_cache_blocks=prefix_cache_blocks,
        **kwargs,
    )
    if warmup:
        engine.warmup()
    if report is not None:
        from engine.quantization.int8 import model_storage_bytes

        report.model_name = model_name
        report.profile = profile
        report.num_blocks = blocks
        report.kv_dtype = kwargs.get("kv_cache_dtype", "fp16")
        report.weights_bytes = model_storage_bytes(loaded.model)
        report.engine_kwargs = {k: v for k, v in kwargs.items() if k != "kv_key_smoothing"}
        report.notes = notes
    engine.loaded_model = loaded
    return engine
