"""Stage 12 reference paged-KV addressing and gather path."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .allocator import BlockAllocator
from .block_table import BlockTable


@dataclass
class PagedRequest:
    table: BlockTable
    sequence_length: int = 0


class PagedKVCacheManager:
    """Owns block tables and capacity accounting, not model-specific attention tensors."""

    def __init__(self, num_blocks: int, block_size_tokens: int):
        self.allocator = BlockAllocator(num_blocks, block_size_tokens)
        self.block_size_tokens = block_size_tokens
        self.requests: dict[str, PagedRequest] = {}

    def reserve(self, request_id: str, capacity_tokens: int, *, sequence_length: int = 0) -> BlockTable | None:
        if capacity_tokens <= 0 or sequence_length < 0 or sequence_length > capacity_tokens:
            raise ValueError("invalid capacity_tokens or sequence_length")
        blocks_needed = (capacity_tokens + self.block_size_tokens - 1) // self.block_size_tokens
        block_ids = self.allocator.allocate(request_id, blocks_needed)
        if block_ids is None:
            return None
        table = BlockTable(request_id, self.block_size_tokens, block_ids)
        self.requests[request_id] = PagedRequest(table, sequence_length)
        return table

    def set_sequence_length(self, request_id: str, sequence_length: int) -> None:
        request = self.requests[request_id]
        if not 0 <= sequence_length <= request.table.capacity_tokens:
            raise ValueError("sequence_length exceeds reserved block capacity")
        request.sequence_length = sequence_length

    def release(self, request_id: str) -> BlockTable:
        request = self.requests.pop(request_id)
        self.allocator.release(request_id)
        return request.table

    def snapshot(self) -> dict[str, object]:
        allocated_capacity = sum(request.table.capacity_tokens for request in self.requests.values())
        used_tokens = sum(request.sequence_length for request in self.requests.values())
        return {
            "num_blocks": self.allocator.num_blocks,
            "block_size_tokens": self.block_size_tokens,
            "free_blocks": self.allocator.free_block_count,
            "used_blocks": self.allocator.used_block_count,
            "active_requests": len(self.requests),
            "allocated_capacity_tokens": allocated_capacity,
            "used_sequence_tokens": used_tokens,
            "internal_fragmentation_tokens": allocated_capacity - used_tokens,
        }


def gather_paged_tokens(pages: torch.Tensor, table: BlockTable, sequence_length: int) -> torch.Tensor:
    """Reference gather of logical `[token, ...]` data from physical `[block, token, ...]` pages."""
    if pages.ndim < 2:
        raise ValueError("pages must be shaped [physical_block, block_token, ...]")
    if pages.shape[1] != table.block_size_tokens:
        raise ValueError("page token dimension must match block-table size")
    if not 0 <= sequence_length <= table.capacity_tokens:
        raise ValueError("sequence_length exceeds block-table capacity")
    indices = torch.tensor(table.physical_block_ids, device=pages.device, dtype=torch.long)
    logical_pages = pages.index_select(0, indices)
    return logical_pages.flatten(0, 1)[:sequence_length]
