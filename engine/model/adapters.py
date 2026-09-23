"""Model-family hooks: what the engine needs to know about an architecture.

The engine's fast paths were written against Qwen3 and, until this module, said so in
three places: the SwiGLU installer matched the class name `Qwen3MLP`, the RoPE installer
imported `transformers.models.qwen3.modeling_qwen3` by path, and nothing checked whether
a checkpoint's geometry was one the paged kernels can serve. Loading a Llama checkpoint
would have silently installed nothing and served it with stock attention at stock speed,
or worse, run a kernel outside its supported head dimension.

The hook is deliberately small, because most of what the engine needs is already in the
config or derivable from the model object:

* the MLP modules are found structurally (a module with `gate_proj`, `up_proj` and
  `down_proj`), which covers every Llama-style feed-forward including Qwen, Mistral and
  Gemma, rather than by a name list that goes stale;
* the RoPE function is patched in *the model's own modeling module*
  (`type(model).__module__`), so a family needs no entry here at all;
* the norm modules are found by class-name suffix, since every RMSNorm variant in
  transformers ends in `RMSNorm`.

What does need declaring is what the engine cannot serve: `SUPPORT_CHECKS` turns an
unsupported geometry into one clear error at load time instead of a wrong answer or a
kernel launch failure at the first request.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable


@dataclass(frozen=True)
class ModelGeometry:
    """The shapes every kernel and the KV pool are sized from."""

    num_layers: int
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int | None

    @property
    def gqa_group(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    def kv_bytes_per_token(self, layers: int | None = None, dtype_bytes: int = 2) -> int:
        """K and V for one token across all layers - the number that sizes the pool."""
        layers = self.num_layers if layers is None else layers
        return 2 * layers * self.num_kv_heads * self.head_dim * dtype_bytes


def geometry_of(config) -> ModelGeometry:
    """Read a transformers config into the shapes the engine uses."""
    q_heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", q_heads))
    head_dim = int(getattr(config, "head_dim", 0) or config.hidden_size // q_heads)
    return ModelGeometry(
        num_layers=int(config.num_hidden_layers),
        hidden_size=int(config.hidden_size),
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        vocab_size=int(getattr(config, "vocab_size", 0)),
        max_position_embeddings=getattr(config, "max_position_embeddings", None),
    )


# Each check returns a reason the engine cannot serve this config, or None. They are the
# difference between "this model is not supported, here is why" at load and a wrong
# answer in production.
SUPPORT_CHECKS: list[Callable[[object, ModelGeometry], str | None]] = []


def support_check(function: Callable[[object, ModelGeometry], str | None]):
    SUPPORT_CHECKS.append(function)
    return function


@support_check
def _head_dim_within_kernel_limit(config, geometry: ModelGeometry) -> str | None:
    # Every paged kernel loads a whole head per program into registers, with the head
    # dimension as a compile-time constant; 128 is the largest that fits the register
    # budget the tiles were chosen for.
    if geometry.head_dim > 128:
        return (f"head_dim {geometry.head_dim} exceeds the 128 the paged kernels support; "
                f"serve this model with prefill_attention='sdpa' and a stock decode path, "
                f"or re-tile the kernels")
    if geometry.head_dim % 8:
        return f"head_dim {geometry.head_dim} is not a multiple of 8"
    return None


@support_check
def _grouped_query_attention_is_uniform(config, geometry: ModelGeometry) -> str | None:
    if geometry.num_q_heads % geometry.num_kv_heads:
        return (f"{geometry.num_q_heads} query heads do not divide into "
                f"{geometry.num_kv_heads} KV heads")
    return None


@support_check
def _no_sliding_window(config, geometry: ModelGeometry) -> str | None:
    window = getattr(config, "sliding_window", None)
    uses_window = getattr(config, "use_sliding_window", window is not None)
    if window and uses_window:
        # The paged kernels attend to every cached key of a row. A windowed model would
        # answer differently from its reference, which the identity gate would catch -
        # but only after a GPU run, and only if someone looked.
        return (f"sliding-window attention (window {window}) is not implemented; the "
                f"paged kernels attend to the full prefix")
    return None


@support_check
def _dense_feed_forward(config, geometry: ModelGeometry) -> str | None:
    experts = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
    if experts:
        return f"mixture-of-experts ({experts} experts) is not implemented"
    return None


@support_check
def _absorbed_attention_unsupported(config, geometry: ModelGeometry) -> str | None:
    # DeepSeek-style MLA keeps a compressed latent per token rather than K/V heads; the
    # pool layout and every kernel assume [blocks, slots, kv_heads, head_dim].
    if getattr(config, "kv_lora_rank", None):
        return "multi-head latent attention (MLA) checkpoints are not supported"
    return None


def unsupported_reason(config) -> str | None:
    """The first reason this config cannot be served, or None."""
    geometry = geometry_of(config)
    for check in SUPPORT_CHECKS:
        reason = check(config, geometry)
        if reason is not None:
            return reason
    return None


def ensure_supported(config) -> ModelGeometry:
    reason = unsupported_reason(config)
    if reason is not None:
        model_type = getattr(config, "model_type", "model")
        raise ValueError(f"{model_type}: {reason}")
    return geometry_of(config)


# ---------------------------------------------------------------------------
# Structural discovery: which modules the fused kernels replace
# ---------------------------------------------------------------------------

def mlp_modules(model, *, gate: str = "gate_proj", up: str = "up_proj",
                down: str = "down_proj") -> list:
    """Every SwiGLU feed-forward in the model, found by shape rather than by name."""
    found = []
    for module in model.modules():
        if all(hasattr(module, attribute) for attribute in (gate, up, down)):
            # A module that has already been fused keeps the attributes set to None.
            if getattr(module, gate) is None and not hasattr(module, "fused_gate_up_proj"):
                continue
            found.append(module)
    return found


def norm_modules(model, *, suffix: str = "rmsnorm") -> list:
    """Every RMSNorm-shaped module: one weight, an epsilon, no bias."""
    found = []
    for module in model.modules():
        if not module.__class__.__name__.lower().endswith(suffix):
            continue
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 1:
            continue
        found.append(module)
    return found


def modeling_module(model) -> ModuleType | None:
    """The transformers module that defines this model, which owns `apply_rotary_pos_emb`.

    `Qwen3ForCausalLM.__module__` is `transformers.models.qwen3.modeling_qwen3`, and the
    same holds for every family, so the RoPE hook needs no per-family table.
    """
    for klass in type(model).__mro__:
        module = sys.modules.get(klass.__module__)
        if module is not None and hasattr(module, "apply_rotary_pos_emb"):
            return module
    return None


def describe(model) -> dict:
    """What the engine detected about a checkpoint. Printed by `scripts/check_hooks.py`
    so an unfamiliar model's support can be inspected before it is served."""
    geometry = geometry_of(model.config)
    module = modeling_module(model)
    return {
        "model_type": getattr(model.config, "model_type", "unknown"),
        "class": type(model).__name__,
        "geometry": geometry.__dict__ | {"gqa_group": geometry.gqa_group},
        "kv_bytes_per_token": geometry.kv_bytes_per_token(),
        "mlp_modules": len(mlp_modules(model)),
        "norm_modules": len(norm_modules(model)),
        "rope_module": module.__name__ if module is not None else None,
        "unsupported_reason": unsupported_reason(model.config),
    }
