import torch

from engine.cache import KVBlockAllocation, KVBlockManager, gather_paged_tokens


def test_block_table_maps_logical_token_to_physical_location() -> None:
    table = KVBlockAllocation("r", 4, [7, 2, 9])
    assert table.physical_location(0) == (7, 0)
    assert table.physical_location(5) == (2, 1)
    assert table.physical_location(11) == (9, 3)


def test_paged_gather_matches_logical_sequence() -> None:
    pages = torch.arange(5 * 4 * 2).reshape(5, 4, 2)
    table = KVBlockAllocation("r", 4, [3, 1, 4])
    expected = torch.cat([pages[3], pages[1], pages[4]], dim=0)[:9]
    assert torch.equal(gather_paged_tokens(pages, table, 9), expected)


def test_paged_manager_reuses_blocks_and_reports_internal_fragmentation() -> None:
    manager = KVBlockManager(num_blocks=4, block_size_tokens=4)
    first = manager.reserve("first", capacity_tokens=5, sequence_length=5)
    second = manager.reserve("second", capacity_tokens=4, sequence_length=2)
    assert first is not None and second is not None
    snapshot = manager.snapshot()
    assert snapshot["used_blocks"] == 3
    assert snapshot["internal_fragmentation_tokens"] == 5
    released = manager.release("first")
    replacement = manager.reserve("replacement", capacity_tokens=8, sequence_length=1)
    assert replacement is not None
    assert set(released.physical_block_ids).issubset(set(replacement.physical_block_ids))


def test_paged_manager_grows_only_when_crossing_a_block_boundary() -> None:
    manager = KVBlockManager(num_blocks=3, block_size_tokens=4)
    table = manager.reserve("request", capacity_tokens=2, sequence_length=2)
    assert table is not None
    assert manager.append_tokens("request", 2)
    assert len(manager.requests["request"].physical_block_ids) == 1
    assert manager.append_tokens("request", 1)
    assert len(manager.requests["request"].physical_block_ids) == 2
    assert manager.requests["request"].sequence_length == 5


def test_failed_growth_does_not_change_request_ownership() -> None:
    manager = KVBlockManager(num_blocks=2, block_size_tokens=4)
    allocation = manager.reserve("request", capacity_tokens=4, sequence_length=4)
    assert allocation is not None
    original_blocks = list(allocation.physical_block_ids)
    assert not manager.ensure_capacity("request", 12)
    assert allocation.physical_block_ids == original_blocks
    assert allocation.sequence_length == 4
