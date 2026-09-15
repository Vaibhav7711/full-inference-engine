"""Combined Q/K rotary-position embedding kernel for Qwen decode and prefill."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_qk_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_cb, stride_cs, stride_cd,
    stride_sb, stride_ss, stride_sd,
    stride_qob, stride_qoh, stride_qos, stride_qod,
    stride_kob, stride_koh, stride_kos, stride_kod,
    q_heads, k_heads,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    token = tl.program_id(1)
    head = tl.program_id(2)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    half = HEAD_DIM // 2
    rotated_dims = tl.where(dims < half, dims + half, dims - half)
    signs = tl.where(dims < half, -1.0, 1.0)

    cos = tl.load(
        cos_ptr + batch * stride_cb + token * stride_cs + dims * stride_cd,
        mask=dim_mask, other=0.0,
    )
    sin = tl.load(
        sin_ptr + batch * stride_sb + token * stride_ss + dims * stride_sd,
        mask=dim_mask, other=0.0,
    )

    q_mask = dim_mask & (head < q_heads)
    q_base = q_ptr + batch * stride_qb + head * stride_qh + token * stride_qs
    q = tl.load(q_base + dims * stride_qd, mask=q_mask, other=0.0)
    q_rotated = tl.load(
        q_base + rotated_dims * stride_qd, mask=q_mask, other=0.0
    ) * signs
    q_out_base = (
        q_out_ptr + batch * stride_qob + head * stride_qoh + token * stride_qos
    )
    tl.store(q_out_base + dims * stride_qod, q * cos + q_rotated * sin, mask=q_mask)

    k_mask = dim_mask & (head < k_heads)
    k_base = k_ptr + batch * stride_kb + head * stride_kh + token * stride_ks
    k = tl.load(k_base + dims * stride_kd, mask=k_mask, other=0.0)
    k_rotated = tl.load(
        k_base + rotated_dims * stride_kd, mask=k_mask, other=0.0
    ) * signs
    k_out_base = (
        k_out_ptr + batch * stride_kob + head * stride_koh + token * stride_kos
    )
    tl.store(k_out_base + dims * stride_kod, k * cos + k_rotated * sin, mask=k_mask)


def triton_rope_qk(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Qwen half-rotation RoPE to `[B,H,S,D]` Q and K in one launch."""
    if unsqueeze_dim != 1:
        raise ValueError("fused Qwen RoPE supports [B,H,S,D] layout only")
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must have [B,H,S,D] shapes")
    batch, q_heads, sequence, head_dim = query.shape
    kb, k_heads, ks, kd = key.shape
    if (kb, ks, kd) != (batch, sequence, head_dim):
        raise ValueError("query and key batch/sequence/head dimensions must match")
    if head_dim % 2 or head_dim > 256:
        raise ValueError("RoPE head dimension must be even and at most 256")
    if cos.shape != (batch, sequence, head_dim) or sin.shape != cos.shape:
        raise ValueError("cos and sin must have [B,S,D] shapes")
    if not all(t.device == query.device for t in (key, cos, sin)):
        raise ValueError("Q/K/cos/sin must share a device")
    if query.device.type != "cuda" or key.dtype != query.dtype:
        raise ValueError("Q/K must share a CUDA dtype")

    query_out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    key_out = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    block_d = triton.next_power_of_2(head_dim)
    _rope_qk_kernel[(batch, sequence, max(q_heads, k_heads))](
        query, key, cos, sin, query_out, key_out,
        *query.stride(), *key.stride(), *cos.stride(), *sin.stride(),
        *query_out.stride(), *key_out.stride(),
        q_heads, k_heads,
        HEAD_DIM=head_dim, BLOCK_D=block_d, num_warps=4,
    )
    return query_out, key_out


def install_triton_qwen_rope() -> None:
    """Replace Qwen3's module-global RoPE function used by every attention layer."""
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    if not hasattr(modeling_qwen3, "_pre_triton_apply_rotary_pos_emb"):
        modeling_qwen3._pre_triton_apply_rotary_pos_emb = modeling_qwen3.apply_rotary_pos_emb
        modeling_qwen3.apply_rotary_pos_emb = triton_rope_qk


def uninstall_triton_qwen_rope() -> None:
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    original = getattr(modeling_qwen3, "_pre_triton_apply_rotary_pos_emb", None)
    if original is not None:
        modeling_qwen3.apply_rotary_pos_emb = original
        del modeling_qwen3._pre_triton_apply_rotary_pos_emb
