"""Static per-channel key smoothing for the INT8 paged KV cache.

Per-token absmax INT8 quantization of Qwen3 keys loses most of its precision to a few
outlier channels: on Qwen3-0.6B the largest post-RoPE key channel is 24-81x the median
channel in the first layers. A fake-quantized stock model on natural-text prompts showed
a median first-token logit error of 1.6 with per-token keys, versus 0.4 when keys are
divided by a calibrated per-(kv head, channel) factor before quantization (and queries
multiplied by it). See `docs/rtx4060-rnd-priorities.md`, Priority 4 quality gate.

The factor is static: computed once from calibration text, then passed to the engine as
``kv_key_smoothing``. It must be equal across each RoPE pair (channel ``i`` and
``i + head_dim / 2``), because RoPE mixes those two channels by a position-dependent
rotation; a pair-equal factor commutes with the rotation.
"""

from __future__ import annotations

import torch


def pair_equal(factor: torch.Tensor) -> torch.Tensor:
    """Make ``[..., head_dim]`` factors equal across each rotary pair (max of the pair)."""
    half = factor.shape[-1] // 2
    paired = torch.maximum(factor[..., :half], factor[..., half:])
    return torch.cat((paired, paired), dim=-1)


@torch.inference_mode()
def calibrate_key_smoothing(
    model, token_batches: list[list[int]], *, floor: float = 1e-2,
) -> torch.Tensor:
    """Per-layer post-RoPE key absmax per `(kv_head, channel)`, as `[layers, heads, dim]`.

    Runs the model's own SDPA forward on each calibration sequence and records the keys
    the attention function receives, i.e. after `k_norm` and RoPE - exactly the values
    the INT8 write kernels quantize. Factors are pair-equalized, normalized to a unit
    median per layer and clamped below at ``floor`` so no channel is divided by ~0.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not token_batches:
        raise ValueError("calibration needs at least one token sequence")
    config = model.config
    layers = int(config.num_hidden_layers)
    absmax: list[torch.Tensor | None] = [None] * layers

    def recording_attention(module, query, key, value, attention_mask, **kwargs):
        observed = key.detach().float().abs().amax(dim=(0, 2))  # [kv_heads, head_dim]
        index = module.layer_idx
        absmax[index] = observed if absmax[index] is None else torch.maximum(absmax[index], observed)
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

    name = "__kv_smoothing_calibration__"
    ALL_ATTENTION_FUNCTIONS[name] = recording_attention
    previous = config._attn_implementation
    config._attn_implementation = name
    if hasattr(config, "_attn_implementation_internal"):
        config._attn_implementation_internal = name
    device = next(model.parameters()).device
    try:
        for tokens in token_batches:
            model(input_ids=torch.tensor([tokens], device=device), use_cache=False)
    finally:
        config._attn_implementation = previous
        if hasattr(config, "_attn_implementation_internal"):
            config._attn_implementation_internal = previous
        del ALL_ATTENTION_FUNCTIONS[name]
    if any(entry is None for entry in absmax):
        raise RuntimeError("calibration did not observe every attention layer")
    factors = []
    for entry in absmax:
        factor = pair_equal(entry)
        factor = factor / factor.median()
        factors.append(factor.clamp_min(floor))
    return torch.stack(factors).contiguous()
