"""Block-aligned radix prefix cache over the shared paged KV allocator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .paging import KVBlockAllocation, KVBlockManager


@dataclass(frozen=True)
class _CacheOwner:
    node_id: int


@dataclass(frozen=True)
class _ExactOwner:
    entry_id: int


@dataclass
class _PrefixNode:
    node_id: int
    parent_id: int | None
    token_block: tuple[int, ...]
    physical_block_id: int
    owner: _CacheOwner
    children: set[int] = field(default_factory=set)
    last_access: int = 0


@dataclass
class _ExactEntry:
    entry_id: int
    token_ids: tuple[int, ...]
    physical_block_ids: tuple[int, ...]
    next_token_id: int
    owner: _ExactOwner
    last_access: int = 0


@dataclass(frozen=True)
class PrefixMatch:
    physical_block_ids: tuple[int, ...] = ()
    token_count: int = 0
    next_token_id: int | None = None
    exact: bool = False


class PrefixCache:
    """A radix tree whose nodes own one immutable, complete physical KV block."""

    def __init__(self, block_manager: KVBlockManager, max_blocks: int):
        if max_blocks < 0:
            raise ValueError("max_blocks must be non-negative")
        self.block_manager = block_manager
        self.max_blocks = max_blocks
        self._nodes: dict[int, _PrefixNode] = {}
        self._edges: dict[tuple[int | None, tuple[int, ...]], int] = {}
        self._exact: dict[tuple[int, ...], _ExactEntry] = {}
        self._next_node_id = 0
        self._next_entry_id = 0
        self._clock = 0
        self.lookups = 0
        self.hits = 0
        self.hit_tokens = 0
        self.evictions = 0

    def lookup(self, token_ids: list[int]) -> PrefixMatch:
        """Return the longest complete-block prefix, always leaving one prompt token."""
        self.lookups += 1
        exact = self._exact.get(tuple(token_ids))
        if exact is not None:
            self._touch(exact)
            self.hits += 1
            self.hit_tokens += len(token_ids)
            return PrefixMatch(
                exact.physical_block_ids, len(token_ids), exact.next_token_id, True
            )
        reusable_tokens = max(0, len(token_ids) - 1)
        complete_blocks = reusable_tokens // self.block_manager.block_size_tokens
        parent_id = None
        physical_blocks = []
        for block_index in range(complete_blocks):
            start = block_index * self.block_manager.block_size_tokens
            token_block = tuple(
                token_ids[start:start + self.block_manager.block_size_tokens]
            )
            node_id = self._edges.get((parent_id, token_block))
            if node_id is None:
                break
            node = self._nodes[node_id]
            self._touch(node)
            physical_blocks.append(node.physical_block_id)
            parent_id = node_id
        token_count = len(physical_blocks) * self.block_manager.block_size_tokens
        if token_count:
            self.hits += 1
            self.hit_tokens += token_count
        return PrefixMatch(tuple(physical_blocks), token_count)

    def publish(
        self,
        token_ids: list[int],
        allocation: KVBlockAllocation,
        next_token_id: int | None = None,
    ) -> int:
        """Retain newly computed complete blocks and return the number published."""
        if self.max_blocks == 0:
            return 0
        complete_blocks = min(
            len(token_ids) // self.block_manager.block_size_tokens,
            len(allocation.physical_block_ids),
        )
        parent_id = None
        published = 0
        for block_index in range(complete_blocks):
            start = block_index * self.block_manager.block_size_tokens
            token_block = tuple(
                token_ids[start:start + self.block_manager.block_size_tokens]
            )
            edge = (parent_id, token_block)
            existing_id = self._edges.get(edge)
            if existing_id is not None:
                node = self._nodes[existing_id]
                self._touch(node)
                parent_id = existing_id
                continue
            physical_block = allocation.physical_block_ids[block_index]
            node_id = self._next_node_id
            self._next_node_id += 1
            owner = _CacheOwner(node_id)
            self.block_manager.allocator.attach(owner, [physical_block])
            node = _PrefixNode(
                node_id, parent_id, token_block, physical_block, owner
            )
            self._touch(node)
            self._nodes[node_id] = node
            self._edges[edge] = node_id
            if parent_id is not None:
                self._nodes[parent_id].children.add(node_id)
            parent_id = node_id
            published += 1
        # Exact hits can bypass residual prefill only when the cache owns every prompt
        # block plus the first-token decision produced by that exact KV state.
        exact_key = tuple(token_ids)
        prompt_blocks = (
            len(token_ids) + self.block_manager.block_size_tokens - 1
        ) // self.block_manager.block_size_tokens
        if (
            next_token_id is not None
            and prompt_blocks <= self.max_blocks
            and exact_key not in self._exact
        ):
            block_ids = tuple(allocation.physical_block_ids[:prompt_blocks])
            entry_id = self._next_entry_id
            self._next_entry_id += 1
            owner = _ExactOwner(entry_id)
            self.block_manager.allocator.attach(owner, block_ids)
            entry = _ExactEntry(
                entry_id, exact_key, block_ids, int(next_token_id), owner
            )
            self._touch(entry)
            self._exact[exact_key] = entry
        self._evict_to_limit()
        return published

    def evict_until_free(self, required_free_blocks: int) -> int:
        """Drop LRU leaf ownership until allocator pressure is satisfied."""
        if required_free_blocks < 0:
            raise ValueError("required_free_blocks must be non-negative")
        before = self.block_manager.allocator.free_block_count
        while (
            self.block_manager.allocator.free_block_count < required_free_blocks
            and self._evict_one()
        ):
            pass
        return self.block_manager.allocator.free_block_count - before

    def clear(self) -> None:
        while self._evict_one():
            pass

    def snapshot(self) -> dict[str, int | float]:
        return {
            "max_blocks": self.max_blocks,
            "cached_blocks": len(self._cached_physical_blocks()),
            "radix_nodes": len(self._nodes),
            "exact_entries": len(self._exact),
            "lookups": self.lookups,
            "hits": self.hits,
            "hit_tokens": self.hit_tokens,
            "hit_rate": self.hits / self.lookups if self.lookups else 0.0,
            "evictions": self.evictions,
        }

    def _touch(self, node: _PrefixNode | _ExactEntry) -> None:
        self._clock += 1
        node.last_access = self._clock

    def _evict_to_limit(self) -> None:
        while len(self._cached_physical_blocks()) > self.max_blocks and self._evict_one():
            pass

    def _cached_physical_blocks(self) -> set[int]:
        blocks = {node.physical_block_id for node in self._nodes.values()}
        for entry in self._exact.values():
            blocks.update(entry.physical_block_ids)
        return blocks

    def _evict_one(self) -> bool:
        leaves = [node for node in self._nodes.values() if not node.children]
        candidates: list[_PrefixNode | _ExactEntry] = leaves + list(self._exact.values())
        if not candidates:
            return False
        victim = min(
            candidates,
            key=lambda item: (
                item.last_access,
                item.node_id if isinstance(item, _PrefixNode) else item.entry_id,
            ),
        )
        self.block_manager.allocator.release(victim.owner)
        if isinstance(victim, _ExactEntry):
            self._exact.pop(victim.token_ids)
        else:
            self._edges.pop((victim.parent_id, victim.token_block))
            self._nodes.pop(victim.node_id)
            if victim.parent_id is not None and victim.parent_id in self._nodes:
                self._nodes[victim.parent_id].children.discard(victim.node_id)
        self.evictions += 1
        return True
