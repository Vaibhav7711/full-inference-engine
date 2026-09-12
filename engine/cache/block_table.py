"""Logical sequence blocks mapped to non-contiguous physical KV blocks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockTable:
    request_id: str
    block_size_tokens: int
    physical_block_ids: tuple[int, ...]

    @property
    def capacity_tokens(self) -> int:
        return len(self.physical_block_ids) * self.block_size_tokens

    def physical_location(self, token_index: int) -> tuple[int, int]:
        if not 0 <= token_index < self.capacity_tokens:
            raise IndexError(f"token index {token_index} exceeds block-table capacity")
        logical_block, offset = divmod(token_index, self.block_size_tokens)
        return self.physical_block_ids[logical_block], offset
