"""Stage 5: first-fit contiguous allocation baseline for KV-cache capacity."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .kv_cache import KVCacheGeometry


@dataclass(frozen=True)
class ContiguousAllocation:
    request_id: str
    start_token: int
    capacity_tokens: int


class ContiguousKVAllocator:
    """Logical KV arena that reserves one contiguous token range per request.

    This deliberately simple policy provides a measurable fragmentation baseline before
    fixed-size blocks and paged execution are introduced. It accounts for capacity but
    does not yet replace Transformers' physical cache tensors.
    """

    def __init__(self, capacity_tokens: int, geometry: KVCacheGeometry):
        if capacity_tokens <= 0:
            raise ValueError("capacity_tokens must be positive")
        self.capacity_tokens = capacity_tokens
        self.geometry = geometry
        self._free_ranges: list[tuple[int, int]] = [(0, capacity_tokens)]
        self._allocations: dict[str, ContiguousAllocation] = {}

    def allocate(self, request_id: str, capacity_tokens: int) -> ContiguousAllocation | None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if capacity_tokens <= 0:
            raise ValueError("capacity_tokens must be positive")
        if request_id in self._allocations:
            raise ValueError(f"request {request_id!r} already owns an allocation")
        for index, (start, length) in enumerate(self._free_ranges):
            if length < capacity_tokens:
                continue
            allocation = ContiguousAllocation(request_id, start, capacity_tokens)
            self._allocations[request_id] = allocation
            remainder = length - capacity_tokens
            if remainder:
                self._free_ranges[index] = (start + capacity_tokens, remainder)
            else:
                self._free_ranges.pop(index)
            return allocation
        return None

    def release(self, request_id: str) -> ContiguousAllocation:
        try:
            allocation = self._allocations.pop(request_id)
        except KeyError as error:
            raise KeyError(f"request {request_id!r} has no allocation") from error
        self._free_ranges.append((allocation.start_token, allocation.capacity_tokens))
        self._free_ranges.sort()
        merged: list[tuple[int, int]] = []
        for start, length in self._free_ranges:
            if merged and merged[-1][0] + merged[-1][1] == start:
                previous_start, previous_length = merged[-1]
                merged[-1] = (previous_start, previous_length + length)
            else:
                merged.append((start, length))
        self._free_ranges = merged
        return allocation

    @property
    def free_tokens(self) -> int:
        return sum(length for _, length in self._free_ranges)

    @property
    def used_tokens(self) -> int:
        return self.capacity_tokens - self.free_tokens

    @property
    def largest_free_range_tokens(self) -> int:
        return max((length for _, length in self._free_ranges), default=0)

    @property
    def external_fragmentation(self) -> float:
        """0 means all free capacity is contiguous; 1 means maximally split."""
        if not self.free_tokens:
            return 0.0
        return 1.0 - (self.largest_free_range_tokens / self.free_tokens)

    def snapshot(self) -> dict[str, object]:
        return {
            "capacity_tokens": self.capacity_tokens,
            "used_tokens": self.used_tokens,
            "free_tokens": self.free_tokens,
            "largest_free_range_tokens": self.largest_free_range_tokens,
            "external_fragmentation": self.external_fragmentation,
            "active_allocations": len(self._allocations),
            "capacity_bytes": self.geometry.bytes_for_tokens(self.capacity_tokens),
            "used_bytes": self.geometry.bytes_for_tokens(self.used_tokens),
            "free_bytes": self.geometry.bytes_for_tokens(self.free_tokens),
            "free_ranges": list(self._free_ranges),
        }


class BlockAllocator:
    """Fixed-size physical KV block allocator used by the paged-cache manager."""

    def __init__(self, num_blocks: int, block_size_tokens: int):
        if num_blocks <= 0 or block_size_tokens <= 0:
            raise ValueError("num_blocks and block_size_tokens must be positive")
        self.num_blocks = num_blocks
        self.block_size_tokens = block_size_tokens
        self._free_blocks: list[int] = list(range(num_blocks))
        self._allocations: dict[str, tuple[int, ...]] = {}

    def allocate(self, request_id: str, count: int) -> tuple[int, ...] | None:
        if not request_id or count <= 0:
            raise ValueError("request_id and count must be positive")
        if request_id in self._allocations:
            raise ValueError(f"request {request_id!r} already owns blocks")
        if count > len(self._free_blocks):
            return None
        block_ids = tuple(self._free_blocks.pop() for _ in range(count))
        self._allocations[request_id] = block_ids
        return block_ids

    def release(self, request_id: str) -> tuple[int, ...]:
        try:
            block_ids = self._allocations.pop(request_id)
        except KeyError as error:
            raise KeyError(f"request {request_id!r} has no block allocation") from error
        self._free_blocks.extend(block_ids)
        self._free_blocks.sort()
        return block_ids

    def extend(self, request_id: str, count: int) -> tuple[int, ...] | None:
        """Append blocks to an existing allocation without requiring contiguity."""
        if count <= 0:
            raise ValueError("count must be positive")
        if request_id not in self._allocations:
            raise KeyError(f"request {request_id!r} has no block allocation")
        if count > len(self._free_blocks):
            return None
        new_blocks = tuple(self._free_blocks.pop() for _ in range(count))
        self._allocations[request_id] += new_blocks
        return new_blocks

    @property
    def free_block_count(self) -> int:
        return len(self._free_blocks)

    @property
    def used_block_count(self) -> int:
        return self.num_blocks - self.free_block_count
