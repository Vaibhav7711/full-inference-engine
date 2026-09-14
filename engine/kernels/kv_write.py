"""Triton kernels for writing contiguous K/V directly into paged storage."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _write_paged_kv_kernel(
    key_ptr,
    value_ptr,
    key_pool_ptr,
    value_pool_ptr,
    block_table_ptr,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_kpb,
    stride_kps,
    stride_kph,
    stride_kpd,
    stride_vpb,
    stride_vps,
    stride_vph,
    stride_vpd,
    sequence_length,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets_d = tl.arange(0, BLOCK_D)
    mask = (token < sequence_length) & (offsets_d < HEAD_DIM)

    logical_block = token // BLOCK_SIZE
    block_offset = token % BLOCK_SIZE
    physical_block = tl.load(block_table_ptr + logical_block)

    key = tl.load(
        key_ptr + head * stride_kh + token * stride_kt + offsets_d * stride_kd,
        mask=mask,
        other=0.0,
    )
    value = tl.load(
        value_ptr + head * stride_vh + token * stride_vt + offsets_d * stride_vd,
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


def write_paged_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_pool: torch.Tensor,
    value_pool: torch.Tensor,
    block_table: torch.Tensor,
) -> None:
    """Write `[1, H, S, D]` K/V into `[blocks, block, H, D]` pools."""
    if key.ndim != 4 or value.shape != key.shape or key.shape[0] != 1:
        raise ValueError("key/value must have matching [1, H, S, D] shapes")
    if key_pool.shape != value_pool.shape or key_pool.ndim != 4:
        raise ValueError("K/V pools must have matching [blocks, block, H, D] shapes")
    _, heads, sequence_length, head_dim = key.shape
    if key_pool.shape[2:] != (heads, head_dim):
        raise ValueError("pool head geometry does not match incoming K/V")
    block_size = key_pool.shape[1]
    blocks_needed = (sequence_length + block_size - 1) // block_size
    if block_table.ndim != 1 or block_table.numel() < blocks_needed:
        raise ValueError("block table does not cover the incoming sequence")
    if head_dim > 256:
        raise ValueError("KV write kernel supports head dimensions up to 256")

    block_table = block_table.contiguous().to(dtype=torch.int32, device=key.device)
    block_d = triton.next_power_of_2(head_dim)
    _write_paged_kv_kernel[(sequence_length, heads)](
        key,
        value,
        key_pool,
        value_pool,
        block_table,
        key.stride(1),
        key.stride(2),
        key.stride(3),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        key_pool.stride(0),
        key_pool.stride(1),
        key_pool.stride(2),
        key_pool.stride(3),
        value_pool.stride(0),
        value_pool.stride(1),
        value_pool.stride(2),
        value_pool.stride(3),
        sequence_length,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
