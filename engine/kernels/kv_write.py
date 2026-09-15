"""Triton kernels for writing contiguous K/V directly into paged storage."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _write_decode_kv_kernel(
    key_ptr,
    value_ptr,
    key_pool_ptr,
    value_pool_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vd,
    stride_bt_b,
    stride_kpb,
    stride_kps,
    stride_kph,
    stride_kpd,
    stride_vpb,
    stride_vps,
    stride_vph,
    stride_vpd,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    offsets_d = tl.arange(0, BLOCK_D)
    mask = offsets_d < HEAD_DIM

    position = tl.load(seq_lens_ptr + batch)
    logical_block = position // BLOCK_SIZE
    block_offset = position % BLOCK_SIZE
    physical_block = tl.load(
        block_tables_ptr + batch * stride_bt_b + logical_block
    )

    key = tl.load(
        key_ptr + batch * stride_kb + head * stride_kh + offsets_d * stride_kd,
        mask=mask,
        other=0.0,
    )
    value = tl.load(
        value_ptr + batch * stride_vb + head * stride_vh + offsets_d * stride_vd,
        mask=mask,
        other=0.0,
    )
    key_destination = (
        key_pool_ptr
        + physical_block * stride_kpb
        + block_offset * stride_kps
        + head * stride_kph
        + offsets_d * stride_kpd
    )
    value_destination = (
        value_pool_ptr
        + physical_block * stride_vpb
        + block_offset * stride_vps
        + head * stride_vph
        + offsets_d * stride_vpd
    )
    tl.store(key_destination, key, mask=mask)
    tl.store(value_destination, value, mask=mask)


@triton.jit
def _write_prefill_kv_batched_kernel(
    key_ptr, value_ptr, key_pool_ptr, value_pool_ptr, block_tables_ptr, seq_lens_ptr,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_btb, stride_btl,
    stride_kpb, stride_kps, stride_kph, stride_kpd,
    stride_vpb, stride_vps, stride_vph, stride_vpd,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    token = tl.program_id(1)
    head = tl.program_id(2)
    dims = tl.arange(0, BLOCK_D)
    token_valid = token < tl.load(seq_lens_ptr + batch)
    valid = token_valid & (dims < HEAD_DIM)
    logical_block = token // BLOCK_SIZE
    block_offset = token % BLOCK_SIZE
    physical_block = tl.load(
        block_tables_ptr + batch * stride_btb + logical_block * stride_btl,
        mask=token_valid,
        other=0,
    )
    key = tl.load(
        key_ptr + batch * stride_kb + head * stride_kh + token * stride_kt + dims * stride_kd,
        mask=valid, other=0.0,
    )
    value = tl.load(
        value_ptr + batch * stride_vb + head * stride_vh + token * stride_vt + dims * stride_vd,
        mask=valid, other=0.0,
    )
    key_dst = (
        key_pool_ptr + physical_block * stride_kpb + block_offset * stride_kps
        + head * stride_kph + dims * stride_kpd
    )
    value_dst = (
        value_pool_ptr + physical_block * stride_vpb + block_offset * stride_vps
        + head * stride_vph + dims * stride_vpd
    )
    tl.store(key_dst, key, mask=valid)
    tl.store(value_dst, value, mask=valid)


def write_decode_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Write one decode token per request directly into the shared paged pool.

    Inputs are `[N, H, 1, D]`; `seq_lens[n]` is the destination token position
    before this decode step, and `block_tables[n]` maps its logical blocks.
    """
    if key.ndim != 4 or value.shape != key.shape or key.shape[2] != 1:
        raise ValueError("key/value must have matching [N, H, 1, D] shapes")
    if key_pool.shape != value_pool.shape or key_pool.ndim != 4:
        raise ValueError("K/V pools must have matching [blocks, block, H, D] shapes")
    batch_size, heads, _, head_dim = key.shape
    if key_pool.shape[2:] != (heads, head_dim):
        raise ValueError("pool head geometry does not match incoming K/V")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
        raise ValueError("block tables must have shape [N, max_blocks]")
    if seq_lens.ndim != 1 or seq_lens.numel() != batch_size:
        raise ValueError("sequence lengths must have shape [N]")
    if head_dim > 256:
        raise ValueError("KV write kernel supports head dimensions up to 256")

    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=key.device)
    seq_lens = seq_lens.contiguous().to(dtype=torch.int32, device=key.device)
    block_d = triton.next_power_of_2(head_dim)
    _write_decode_kv_kernel[(batch_size, heads)](
        key,
        value,
        key_pool,
        value_pool,
        block_tables,
        seq_lens,
        key.stride(0),
        key.stride(1),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(3),
        block_tables.stride(0),
        key_pool.stride(0),
        key_pool.stride(1),
        key_pool.stride(2),
        key_pool.stride(3),
        value_pool.stride(0),
        value_pool.stride(1),
        value_pool.stride(2),
        value_pool.stride(3),
        BLOCK_SIZE=key_pool.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


def write_prefill_kv_batched(
    key: torch.Tensor,
    value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Write valid tokens from padded `[B,H,S,D]` K/V into per-request blocks."""
    if key.ndim != 4 or value.shape != key.shape:
        raise ValueError("key/value must have matching [B,H,S,D] shapes")
    batch, heads, padded_length, head_dim = key.shape
    if key_pool.shape != value_pool.shape or key_pool.ndim != 4:
        raise ValueError("K/V pools must have matching [blocks,block,H,D] shapes")
    if key_pool.shape[2:] != (heads, head_dim):
        raise ValueError("pool head geometry does not match incoming K/V")
    if block_tables.ndim != 2 or block_tables.shape[0] != batch:
        raise ValueError("block tables must have shape [B,max_blocks]")
    if seq_lens.ndim != 1 or seq_lens.numel() != batch:
        raise ValueError("sequence lengths must have shape [B]")
    if head_dim > 256:
        raise ValueError("KV write kernel supports head dimensions up to 256")

    block_tables = block_tables.contiguous().to(dtype=torch.int32, device=key.device)
    seq_lens = seq_lens.contiguous().to(dtype=torch.int32, device=key.device)
    block_d = triton.next_power_of_2(head_dim)
    _write_prefill_kv_batched_kernel[(batch, padded_length, heads)](
        key, value, key_pool, value_pool, block_tables, seq_lens,
        *key.stride(), *value.stride(), *block_tables.stride(),
        *key_pool.stride(), *value_pool.stride(),
        BLOCK_SIZE=key_pool.shape[1], HEAD_DIM=head_dim, BLOCK_D=block_d,
        num_warps=4,
    )
