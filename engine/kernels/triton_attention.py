"""K1: Triton attention kernel matching SDPA (contiguous K,V, no paging).

This is the FOUNDATION of the paged-attention kernel, built in isolation so the
attention math is proven correct BEFORE paged addressing is added (that's K2).

Flash-attention style:
    - Tiled over the KV sequence (never materializes the full q_len x kv_len matrix)
    - Online softmax: running max m_i and running sum l_i, rescaled per KV tile
    - Causal masking
    - GQA handled by the CALLER (repeat_kv before calling); kernel sees equal Q/KV heads

One program instance per (flattened batch*head, q_tile). Head dim <= 128 (Qwen3 = 128).

Correctness gate (K1): output == torch.nn.functional.scaled_dot_product_attention.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qz, stride_qm, stride_qd,   # Q: [Z, M, D] where Z = B*H
    stride_kz, stride_kn, stride_kd,   # K: [Z, N, D]
    stride_vz, stride_vn, stride_vd,   # V: [Z, N, D]
    stride_oz, stride_om, stride_od,   # out: [Z, M, D]
    q_len, kv_len,
    scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)   # which q tile
    pid_z = tl.program_id(1)   # flattened (batch, head) index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M] query positions
    offs_d = tl.arange(0, HEAD_DIM)                      # [HEAD_DIM]
    offs_n = tl.arange(0, BLOCK_N)                       # [BLOCK_N] key positions (per tile)

    # Base pointer for this (batch,head) slice
    q_base = q_ptr + pid_z * stride_qz
    k_base = k_ptr + pid_z * stride_kz
    v_base = v_ptr + pid_z * stride_vz
    o_base = out_ptr + pid_z * stride_oz

    # Load Q tile: [BLOCK_M, HEAD_DIM]
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < q_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax state
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, kv_len, BLOCK_N):
        cur_n = start_n + offs_n
        n_mask = cur_n < kv_len

        # K tile: [BLOCK_N, HEAD_DIM]
        k_ptrs = k_base + cur_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        # scores = Q @ K^T * scale  -> [BLOCK_M, BLOCK_N]
        # Keep the FP32 correctness path genuinely IEEE FP32.  On Ada, Triton otherwise
        # selects TF32 input precision for float32 dot products, which is fast but exceeds
        # this kernel's tight FP32 error contract.  FP16/BF16 inputs still select tensor
        # core MMA as before.
        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        scores = tl.where(n_mask[None, :], scores, float("-inf"))
        if CAUSAL:
            causal_mask = offs_m[:, None] >= cur_n[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        # Online softmax update
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        # V tile: [BLOCK_N, HEAD_DIM]
        v_ptrs = v_base + cur_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        acc = acc * alpha[:, None] + tl.dot(
            p.to(v.dtype), v, input_precision="ieee",
        )
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Guard against fully-masked rows (l_i == 0) to avoid nan
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=q_mask)


def _auto_block_sizes(head_dim: int, dtype: torch.dtype) -> tuple[int, int]:
    """Pick (block_m, block_n) tiles that fit the GPU shared-memory budget.

    Calibrated from MEASURED T4 (64KB shared mem) launches rather than a formula, because
    Triton's real shared-memory usage (buffer reuse, pipelining) is hard to predict
    analytically. Measured data points on a T4 with head_dim=128:
        - FP16 64x64 tiles launch fine (fit in 64KB).
        - FP32 64x64 tiles need ~82KB and overflow; 32x32 scales to ~38KB and fits.
    FP32 tiles are ~2x FP16 (4 vs 2 bytes/element), so we halve tile dims for FP32.

    For smaller head_dim (e.g. 64), tiles use proportionally less memory, so the FP16
    default is safe there too.
    """
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    if dtype_bytes >= 4:          # FP32 / higher precision
        return (32, 32)
    return (64, 64)               # FP16 / BF16 — measured to fit on T4 at head_dim 128


def triton_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    causal: bool = True,
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    """Flash-attention-style attention via Triton. Contiguous K,V (no paging).

    Args:
        query: [B, H, M, D]
        key:   [B, H, N, D]  (GQA already expanded so H matches query)
        value: [B, H, N, D]
        scale: softmax scale (default 1/sqrt(D))
        causal: apply causal mask
    Returns:
        out: [B, H, M, D]
    """
    B, H, M, D = query.shape
    N = key.shape[2]
    assert key.shape[1] == H, "K heads must match Q heads (expand GQA before calling)"
    assert D <= 128, "kernel assumes head_dim <= 128"

    # Auto-select tile sizes to fit this GPU's shared memory, unless caller forced them.
    if block_m is None or block_n is None:
        auto_m, auto_n = _auto_block_sizes(D, query.dtype)
        block_m = block_m if block_m is not None else auto_m
        block_n = block_n if block_n is not None else auto_n

    if scale is None:
        scale = 1.0 / (D ** 0.5)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    out = torch.empty_like(query)

    # Flatten (B, H) -> Z for a clean 3D [Z, seq, D] view
    q_v = query.view(B * H, M, D)
    k_v = key.view(B * H, N, D)
    v_v = value.view(B * H, N, D)
    o_v = out.view(B * H, M, D)

    grid = (triton.cdiv(M, block_m), B * H)

    _attention_kernel[grid](
        q_v, k_v, v_v, o_v,
        q_v.stride(0), q_v.stride(1), q_v.stride(2),
        k_v.stride(0), k_v.stride(1), k_v.stride(2),
        v_v.stride(0), v_v.stride(1), v_v.stride(2),
        o_v.stride(0), o_v.stride(1), o_v.stride(2),
        M, N,
        scale,
        HEAD_DIM=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        CAUSAL=causal,
    )
    return out
