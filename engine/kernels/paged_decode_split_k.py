"""Split-K paged decode attention: several programs per (row, head), one merge pass.

The per-head kernel runs one program per (row, query head), and each walks that row's
whole KV history serially. At the measured chat operating point that is 128 programs on
40 SMs moving 33 MB per layer in 0.21 ms - about 160 GB/s, half of what the T4 delivers
to a well-shaped kernel. The kernel is not short of bandwidth, it is short of
parallelism, and at batch 1 (a single interactive request, the case a local GPU actually
serves) it is 16 programs for the whole device.

This is the standard answer, FlashDecoding's: split the key range into `SPLITS` slices,
give each its own program, and reduce the partial softmaxes afterwards. Each program
keeps the same online-softmax state as before but only over its slice, and writes
`(m, l, acc)` to a workspace. The merge kernel rescales every slice to the global maximum
and sums - exactly the update the single kernel applies tile by tile, done once across
slices instead.

Cost of the split: one `[rows, heads, splits, head_dim]` fp32 workspace plus a second
launch. At 16 rows, 16 heads, 8 splits, head_dim 128 that is 1 MB and ~10 us, so the
split only pays when it buys more than that in occupancy. `choose_splits` refuses to
split when the grid is already wide enough, and the wrapper then calls the per-head
kernel directly, which is why this file is a strict extension rather than a replacement.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.kernels.paged_decode_config import choose_splits

__all__ = ["paged_decode_split_k", "choose_splits"]


@triton.jit
def _split_decode_kernel(
    q_ptr, kp_ptr, vp_ptr, bt_ptr, sl_ptr,
    acc_ptr, m_ptr, l_ptr,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kh, stride_kd,
    stride_vb, stride_vs, stride_vh, stride_vd,
    stride_as, stride_ah, stride_ak, stride_ad,
    stride_ms, stride_mh, stride_mk,
    stride_bts, stride_btb,
    stride_sl,
    split_len,
    num_q_heads, num_kv_heads, scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LENGTH_OFFSET: tl.constexpr,
):
    pid_s = tl.program_id(0)    # sequence
    pid_h = tl.program_id(1)    # query head
    pid_k = tl.program_id(2)    # which slice of this row's keys

    kv_head = pid_h // (num_q_heads // num_kv_heads)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)

    seq_len = tl.load(sl_ptr + pid_s * stride_sl) + LENGTH_OFFSET
    start = pid_k * split_len
    end = tl.minimum(seq_len, start + split_len)

    q = tl.load(q_ptr + pid_s * stride_qs + pid_h * stride_qh + offs_d * stride_qd)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    bt_base = bt_ptr + pid_s * stride_bts
    # An empty slice (this row is shorter than the split boundary) runs zero iterations
    # and stores the identity state, which the merge ignores.
    for block_start in range(start, end, BLOCK_N):
        cur_n = block_start + offs_n
        n_mask = cur_n < end
        logical_block = cur_n // BLOCK_SIZE
        offset = cur_n % BLOCK_SIZE
        phys_block = tl.load(bt_base + logical_block * stride_btb, mask=n_mask, other=0)

        k_row = phys_block[:, None] * stride_kb + offset[:, None] * stride_ks + kv_head * stride_kh
        k = tl.load(kp_ptr + k_row + offs_d[None, :] * stride_kd,
                    mask=n_mask[:, None], other=0.0)
        scores = tl.sum(q[None, :] * k, axis=1) * scale
        scores = tl.where(n_mask, scores, float("-inf"))

        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)

        v_row = phys_block[:, None] * stride_vb + offset[:, None] * stride_vs + kv_head * stride_vh
        v = tl.load(vp_ptr + v_row + offs_d[None, :] * stride_vd,
                    mask=n_mask[:, None], other=0.0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    acc_base = acc_ptr + pid_s * stride_as + pid_h * stride_ah + pid_k * stride_ak
    tl.store(acc_base + offs_d * stride_ad, acc)
    state = pid_s * stride_ms + pid_h * stride_mh + pid_k * stride_mk
    tl.store(m_ptr + state, m_i)
    tl.store(l_ptr + state, l_i)


@triton.jit
def _merge_splits_kernel(
    acc_ptr, m_ptr, l_ptr, out_ptr,
    stride_as, stride_ah, stride_ak, stride_ad,
    stride_ms, stride_mh, stride_mk,
    stride_os, stride_oh, stride_od,
    HEAD_DIM: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Combine the slices' partial softmaxes into one output vector.

    Each slice holds `m` (its maximum score), `l` (its denominator) and `acc` (its
    weighted value sum), all relative to its own maximum. Rescaling every slice by
    `exp(m_k - max_k m_k)` puts them on one scale, after which the sums are the
    single-pass kernel's final state.
    """
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_k = tl.arange(0, SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)

    state = pid_s * stride_ms + pid_h * stride_mh + offs_k * stride_mk
    m = tl.load(m_ptr + state)
    l = tl.load(l_ptr + state)
    m_max = tl.max(m, axis=0)
    # A slice that saw no keys has m = -inf and l = 0; exp(-inf - finite) is 0, so it
    # contributes nothing. If every slice is empty the row has no keys at all, and the
    # guarded division below returns zeros rather than NaN.
    rescale = tl.exp(m - m_max)
    rescale = tl.where(l > 0.0, rescale, 0.0)
    denominator = tl.sum(l * rescale, axis=0)

    acc = tl.load(
        acc_ptr + pid_s * stride_as + pid_h * stride_ah
        + offs_k[:, None] * stride_ak + offs_d[None, :] * stride_ad
    )
    combined = tl.sum(acc * rescale[:, None], axis=0)
    combined = combined / tl.where(denominator > 0.0, denominator, 1.0)
    tl.store(out_ptr + pid_s * stride_os + pid_h * stride_oh + offs_d * stride_od,
             combined.to(out_ptr.dtype.element_ty))


def paged_decode_split_k(
    query: torch.Tensor,        # [S, H, 1, D]
    key_pages: torch.Tensor,    # [num_blocks, block_size, kv_heads, D]
    value_pages: torch.Tensor,
    block_tables: torch.Tensor, # [S, max_blocks] int
    seq_lens: torch.Tensor,     # [S] int
    scale: float | None = None,
    block_n: int = 64,
    num_warps: int = 4,
    length_offset: int = 0,
    *,
    max_sequence_length: int | None = None,
    splits: int | None = None,
) -> torch.Tensor:
    """Same contract as `paged_decode_batched`, with the key range split across programs.

    `max_sequence_length` is the longest row in the batch, which the caller knows
    host-side; reading it from `seq_lens` would be a device synchronization per layer.
    Without it the block table's width is used as an upper bound, which is correct but
    splits more coarsely than necessary.
    """
    from engine.kernels.paged_decode_batched import paged_decode_batched

    S, H, one, D = query.shape
    if one != 1:
        raise ValueError("decode: query length must be 1 per sequence")
    num_blocks, block_size, kv_heads, D_kv = key_pages.shape
    if D_kv != D or D > 128:
        raise ValueError("head_dim mismatch or above 128")
    if H % kv_heads:
        raise ValueError("num_q_heads must be a multiple of kv_heads")
    if block_n not in {16, 32, 64, 128}:
        raise ValueError("block_n must be one of 16, 32, 64, or 128")
    if num_warps not in {2, 4, 8}:
        raise ValueError("num_warps must be one of 2, 4, or 8")

    bound = max_sequence_length or block_tables.shape[1] * block_size
    if splits is None:
        from engine.kernels.device import current_device

        profile = current_device()
        splits = choose_splits(
            S, H, bound, block_n=block_n,
            multiprocessors=profile.multiprocessors if profile else 40,
        )
    if splits <= 1:
        # Nothing to gain: the single-pass kernel has no workspace and no merge launch.
        return paged_decode_batched(
            query, key_pages, value_pages, block_tables, seq_lens, scale=scale,
            block_n=block_n, num_warps=num_warps, length_offset=length_offset,
        )
    if splits & (splits - 1):
        raise ValueError("splits must be a power of two")

    if scale is None:
        scale = 1.0 / (D ** 0.5)
    query = query.contiguous()
    key_pages = key_pages.contiguous()
    value_pages = value_pages.contiguous()
    block_tables = block_tables.contiguous().to(torch.int32)
    seq_lens = seq_lens.contiguous().to(torch.int32)

    q_view = query.view(S, H, D)
    out = torch.empty_like(query)
    out_view = out.view(S, H, D)
    partial = torch.empty((S, H, splits, D), device=query.device, dtype=torch.float32)
    partial_m = torch.empty((S, H, splits), device=query.device, dtype=torch.float32)
    partial_l = torch.empty_like(partial_m)
    split_len = -(-bound // splits)

    _split_decode_kernel[(S, H, splits)](
        q_view, key_pages, value_pages, block_tables, seq_lens,
        partial, partial_m, partial_l,
        q_view.stride(0), q_view.stride(1), q_view.stride(2),
        key_pages.stride(0), key_pages.stride(1), key_pages.stride(2), key_pages.stride(3),
        value_pages.stride(0), value_pages.stride(1), value_pages.stride(2), value_pages.stride(3),
        partial.stride(0), partial.stride(1), partial.stride(2), partial.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        block_tables.stride(0), block_tables.stride(1),
        seq_lens.stride(0),
        split_len,
        H, kv_heads, scale,
        BLOCK_SIZE=block_size, HEAD_DIM=D, BLOCK_N=block_n,
        LENGTH_OFFSET=length_offset, num_warps=num_warps,
    )
    _merge_splits_kernel[(S, H)](
        partial, partial_m, partial_l, out_view,
        partial.stride(0), partial.stride(1), partial.stride(2), partial.stride(3),
        partial_m.stride(0), partial_m.stride(1), partial_m.stride(2),
        out_view.stride(0), out_view.stride(1), out_view.stride(2),
        HEAD_DIM=D, SPLITS=splits, num_warps=4,
    )
    return out
