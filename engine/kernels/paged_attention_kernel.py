"""K2: paged attention — kernel reads K,V from physical blocks via a block table.

Built on K1 (proven attention math). The ONLY change is where K,V come from: instead of
contiguous [H, N, D] tensors, the kernel loads K,V rows from a paged store
[num_blocks, block_size, H, D] using a block table that maps logical block -> physical
block. Everything else (online softmax, tiling, causal masking) is identical to K1.

This is what vLLM's paged-attention kernel does: attention reads directly from
non-contiguous KV blocks, no gather, no copy.

Single sequence (batch handling is K4). Shapes:
    query:       [H, M, D]                          one sequence, M query positions
    key_pages:   [num_blocks, block_size, H, D]     shared paged K store
    value_pages: [num_blocks, block_size, H, D]     shared paged V store
    block_table: [num_logical_blocks]  int32        logical block i -> physical block
    kv_len:      int                                 actual logical KV length

Physical row for logical KV position p:
    logical_block = p // block_size
    offset        = p %  block_size
    physical_row  = block_table[logical_block] * block_size + offset

Correctness gate (K2): output == K1 output for the same logical K,V data.
Since K1 == SDPA, transitively K2 == SDPA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention_kernel(
    q_ptr, kp_ptr, vp_ptr, out_ptr, bt_ptr,
    # Q strides [H, M, D]
    stride_qh, stride_qm, stride_qd,
    # paged K,V strides [num_blocks, block_size, H, D]
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    # out strides [H, M, D]
    stride_oh, stride_om, stride_od,
    # block table stride [num_logical_blocks]
    stride_bt,
    q_len, kv_len,
    scale,
    BLOCK_SIZE: tl.constexpr,     # tokens per physical block
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)   # which q tile
    pid_h = tl.program_id(1)   # which head

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    # Q base for this head
    q_base = q_ptr + pid_h * stride_qh
    o_base = out_ptr + pid_h * stride_oh

    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q_mask = offs_m[:, None] < q_len
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, kv_len, BLOCK_N):
        cur_n = start_n + offs_n                    # [BLOCK_N] logical KV positions
        n_mask = cur_n < kv_len

        # --- paged address computation ---
        logical_block = cur_n // BLOCK_SIZE          # [BLOCK_N]
        offset = cur_n % BLOCK_SIZE                   # [BLOCK_N]
        # Look up physical block for each logical block (guard OOB with 0)
        bt_ptrs = bt_ptr + logical_block * stride_bt
        phys_block = tl.load(bt_ptrs, mask=n_mask, other=0)   # [BLOCK_N]
        # physical row within the flattened [num_blocks*block_size, H, D] view is
        # phys_block * block_size + offset, but we address pages as [nb, bs, H, D]:
        #   row pointer = phys_block*stride_kb + offset*stride_ks + head*stride_kh
        # K tile: [BLOCK_N, HEAD_DIM]
        k_row = phys_block[:, None] * stride_kb + offset[:, None] * stride_ks + pid_h * stride_kh
        k_ptrs = kp_ptr + k_row + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(n_mask[None, :], scores, float("-inf"))
        if CAUSAL:
            causal_mask = offs_m[:, None] >= cur_n[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        v_row = phys_block[:, None] * stride_vb + offset[:, None] * stride_vs + pid_h * stride_vh
        v_ptrs = vp_ptr + v_row + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=q_mask)


def _auto_block_sizes(head_dim: int, dtype: torch.dtype) -> tuple[int, int]:
    """Same measured-T4 calibration as K1: FP16 64x64, FP32 32x32 (64KB shared-mem)."""
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    if dtype_bytes >= 4:
        return (32, 32)
    return (64, 64)


def paged_attention(
    query: torch.Tensor,        # [H, M, D]
    key_pages: torch.Tensor,    # [num_blocks, block_size, H, D]
    value_pages: torch.Tensor,  # [num_blocks, block_size, H, D]
    block_table: torch.Tensor,  # [num_logical_blocks] int
    kv_len: int,
    scale: float | None = None,
    causal: bool = True,
    block_m: int | None = None,
    block_n: int | None = None,
) -> torch.Tensor:
    """Paged attention for a single sequence. K,V read from blocks via block_table.

    Returns out: [H, M, D].
    """
    H, M, D = query.shape
    num_blocks, block_size, H_kv, D_kv = key_pages.shape
    assert H_kv == H, "paged K,V head count must match query heads (expand GQA first)"
    assert D_kv == D and D <= 128
    assert block_table.dtype in (torch.int32, torch.int64)

    if scale is None:
        scale = 1.0 / (D ** 0.5)
    if block_m is None or block_n is None:
        am, an = _auto_block_sizes(D, query.dtype)
        block_m = block_m if block_m is not None else am
        block_n = block_n if block_n is not None else an

    query = query.contiguous()
    key_pages = key_pages.contiguous()
    value_pages = value_pages.contiguous()
    block_table = block_table.contiguous().to(torch.int32)
    out = torch.empty_like(query)

    grid = (triton.cdiv(M, block_m), H)

    _paged_attention_kernel[grid](
        query, key_pages, value_pages, out, block_table,
        query.stride(0), query.stride(1), query.stride(2),
        key_pages.stride(0), key_pages.stride(1), key_pages.stride(2), key_pages.stride(3),
        value_pages.stride(0), value_pages.stride(1), value_pages.stride(2), value_pages.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_table.stride(0),
        M, kv_len,
        scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        CAUSAL=causal,
    )
    return out


def scatter_to_pages(
    key: torch.Tensor,          # [H, N, D] contiguous
    value: torch.Tensor,        # [H, N, D]
    block_size: int,
    block_table: torch.Tensor,  # [num_logical_blocks] logical -> physical
    num_physical_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter contiguous [H, N, D] K,V into paged [num_blocks, block_size, H, D] store.

    Helper for the correctness gate: builds paged storage from contiguous K,V so we can
    verify the paged kernel reproduces the contiguous (K1) result.
    """
    H, N, D = key.shape
    kp = key.new_zeros((num_physical_blocks, block_size, H, D))
    vp = value.new_zeros((num_physical_blocks, block_size, H, D))
    for logical_pos in range(N):
        lb = logical_pos // block_size
        off = logical_pos % block_size
        phys = int(block_table[lb].item())
        kp[phys, off] = key[:, logical_pos, :]   # [H, D]
        vp[phys, off] = value[:, logical_pos, :]
    return kp, vp
