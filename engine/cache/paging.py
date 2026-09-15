"""Production block ownership and reference paged-KV gather operations."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .allocator import BlockAllocator


@dataclass
class KVBlockAllocation:
    """Mutable logical-to-physical mapping owned by exactly one request."""

    request_id: str
    block_size_tokens: int
    physical_block_ids: list[int]
    sequence_length: int = 0

    @property
    def capacity_tokens(self) -> int:
        return len(self.physical_block_ids) * self.block_size_tokens

    def physical_location(self, token_index: int) -> tuple[int, int]:
        if not 0 <= token_index < self.capacity_tokens:
            raise IndexError(f"token index {token_index} exceeds block allocation capacity")
        logical_block, offset = divmod(token_index, self.block_size_tokens)
        return self.physical_block_ids[logical_block], offset


class KVBlockManager:
    """Single owner of request block mappings, sequence lengths, and capacity."""

    def __init__(self, num_blocks: int, block_size_tokens: int):
        self.allocator = BlockAllocator(num_blocks, block_size_tokens)
        self.block_size_tokens = block_size_tokens
        self.requests: dict[str, KVBlockAllocation] = {}

    def reserve(
        self, request_id: str, capacity_tokens: int, *, sequence_length: int = 0
    ) -> KVBlockAllocation | None:
        if capacity_tokens <= 0 or sequence_length < 0 or sequence_length > capacity_tokens:
            raise ValueError("invalid capacity_tokens or sequence_length")
        if request_id in self.requests:
            raise ValueError(f"request {request_id!r} already owns KV blocks")
        blocks_needed = (capacity_tokens + self.block_size_tokens - 1) // self.block_size_tokens
        block_ids = self.allocator.allocate(request_id, blocks_needed)
        if block_ids is None:
            return None
        allocation = KVBlockAllocation(
            request_id, self.block_size_tokens, list(block_ids), sequence_length
        )
        self.requests[request_id] = allocation
        return allocation

    def attach_prefix(
        self, request_id: str, physical_block_ids: list[int], sequence_length: int
    ) -> KVBlockAllocation:
        """Attach a request to immutable full prefix blocks already in the pool."""
        if request_id in self.requests:
            raise ValueError(f"request {request_id!r} already owns KV blocks")
        if not physical_block_ids or sequence_length <= 0:
            raise ValueError("a shared prefix requires blocks and a positive length")
        capacity = len(physical_block_ids) * self.block_size_tokens
        if sequence_length != capacity:
            raise ValueError("only complete blocks may be shared")
        self.allocator.attach(request_id, physical_block_ids)
        allocation = KVBlockAllocation(
            request_id, self.block_size_tokens, list(physical_block_ids), sequence_length
        )
        self.requests[request_id] = allocation
        return allocation

    def set_sequence_length(self, request_id: str, sequence_length: int) -> None:
        allocation = self.requests[request_id]
        if not 0 <= sequence_length <= allocation.capacity_tokens:
            raise ValueError("sequence_length exceeds reserved block capacity")
        allocation.sequence_length = sequence_length

    def ensure_capacity(self, request_id: str, target_length: int) -> bool:
        """Ensure a request can store ``target_length`` tokens without committing them."""
        allocation = self.requests[request_id]
        if target_length < allocation.sequence_length:
            raise ValueError("target_length cannot be shorter than the committed sequence")
        blocks_needed = (target_length + self.block_size_tokens - 1) // self.block_size_tokens
        extra_blocks = blocks_needed - len(allocation.physical_block_ids)
        if extra_blocks <= 0:
            return True
        new_ids = self.allocator.extend(request_id, extra_blocks)
        if new_ids is None:
            return False
        allocation.physical_block_ids.extend(new_ids)
        return True

    def append_tokens(self, request_id: str, count: int = 1) -> bool:
        """Grow a sequence, allocating new physical blocks only at block boundaries."""
        if count <= 0:
            raise ValueError("count must be positive")
        allocation = self.requests[request_id]
        target_length = allocation.sequence_length + count
        if not self.ensure_capacity(request_id, target_length):
            return False
        allocation.sequence_length = target_length
        return True

    def release(self, request_id: str) -> KVBlockAllocation:
        allocation = self.requests.pop(request_id)
        self.allocator.release(request_id)
        return allocation

    def snapshot(self) -> dict[str, object]:
        allocated_capacity = sum(request.capacity_tokens for request in self.requests.values())
        used_tokens = sum(request.sequence_length for request in self.requests.values())
        return {
            "num_blocks": self.allocator.num_blocks,
            "block_size_tokens": self.block_size_tokens,
            "free_blocks": self.allocator.free_block_count,
            "used_blocks": self.allocator.used_block_count,
            "shared_blocks": self.allocator.shared_block_count,
            "block_references": self.allocator.total_references,
            "active_requests": len(self.requests),
            "allocated_capacity_tokens": allocated_capacity,
            "used_sequence_tokens": used_tokens,
            "internal_fragmentation_tokens": allocated_capacity - used_tokens,
        }


def gather_paged_tokens(
    pages: torch.Tensor, table: KVBlockAllocation, sequence_length: int
) -> torch.Tensor:
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
