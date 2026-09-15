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
