from engine.cache import KVBlockManager, PrefixCache


def test_prefix_cache_retains_shares_and_releases_blocks() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    cache = PrefixCache(manager, max_blocks=4)
    first = manager.reserve("first", 8, sequence_length=8)
    assert first is not None
    original_blocks = list(first.physical_block_ids)
    assert cache.publish(list(range(8)), first) == 2
    manager.release("first")
    assert manager.snapshot()["used_blocks"] == 2

    match = cache.lookup(list(range(9)))
    assert match.token_count == 8
    assert list(match.physical_block_ids) == original_blocks
    second = manager.attach_prefix("second", list(match.physical_block_ids), match.token_count)
    assert second.sequence_length == 8
    assert all(manager.allocator.refcount(block) == 2 for block in original_blocks)

    cache.clear()
    assert manager.snapshot()["used_blocks"] == 2
    manager.release("second")
    assert manager.snapshot()["free_blocks"] == 8


def test_prefix_cache_uses_complete_blocks_and_leaves_a_residual_token() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    cache = PrefixCache(manager, max_blocks=8)
    allocation = manager.reserve("source", 12, sequence_length=12)
    assert allocation is not None
    cache.publish(list(range(12)), allocation)
    assert cache.lookup(list(range(12))).token_count == 8
    assert cache.lookup(list(range(13))).token_count == 12


def test_prefix_cache_evicts_lru_leaves_to_its_block_budget() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    cache = PrefixCache(manager, max_blocks=2)
    allocation = manager.reserve("source", 12, sequence_length=12)
    assert allocation is not None
    cache.publish(list(range(12)), allocation)
    assert cache.snapshot()["cached_blocks"] == 2
    assert cache.snapshot()["evictions"] == 1
    assert cache.lookup(list(range(13))).token_count == 8


def test_exact_prompt_hit_reuses_partial_tail_and_first_token() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    cache = PrefixCache(manager, max_blocks=4)
    source = manager.reserve("source", 6, sequence_length=6)
    assert source is not None
    cache.publish(list(range(6)), source, next_token_id=42)
    source_blocks = tuple(source.physical_block_ids)
    manager.release("source")

    match = cache.lookup(list(range(6)))
    assert match.exact
    assert match.token_count == 6
    assert match.next_token_id == 42
    assert match.physical_block_ids == source_blocks
    attached = manager.attach_prefix("hit", list(match.physical_block_ids), 6)
    old_tail = attached.physical_block_ids[-1]
    copied = manager.copy_on_write_tail("hit")
    assert copied is not None and copied[0] == old_tail
    assert attached.physical_block_ids[-1] != old_tail
    assert manager.allocator.refcount(old_tail) >= 1


def test_pressure_eviction_skips_entries_pinned_by_active_requests() -> None:
    from engine.cache import KVBlockManager, PrefixCache
    from engine.runtime import GenerationRequest

    manager = KVBlockManager(num_blocks=4, block_size_tokens=4)
    cache = PrefixCache(manager, max_blocks=4)
    shared = manager.reserve("shared-owner", 8)
    manager.append_tokens("shared-owner", 8)
    cache.publish(list(range(1, 9)), shared, next_token_id=9)
    # The publishing request is still active: its blocks are pinned (refcount > 1).
    assert cache.snapshot()["cached_blocks"] == 2
    freed = cache.evict_until_free(4)
    assert freed == 0
    assert cache.snapshot()["cached_blocks"] == 2  # nothing wiped for no gain

    manager.release("shared-owner")
    freed = cache.evict_until_free(4)
    assert freed == 2 and manager.allocator.free_block_count == 4
    assert cache.snapshot()["cached_blocks"] == 0
