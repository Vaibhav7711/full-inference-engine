import torch

from engine.cache import BlockTable, PagedKVCacheManager, gather_paged_tokens


def test_block_table_maps_logical_token_to_physical_location() -> None:
    table = BlockTable("r", 4, (7, 2, 9))
    assert table.physical_location(0) == (7, 0)
    assert table.physical_location(5) == (2, 1)
    assert table.physical_location(11) == (9, 3)


def test_paged_gather_matches_logical_sequence() -> None:
    pages = torch.arange(5 * 4 * 2).reshape(5, 4, 2)
    table = BlockTable("r", 4, (3, 1, 4))
    expected = torch.cat([pages[3], pages[1], pages[4]], dim=0)[:9]
    assert torch.equal(gather_paged_tokens(pages, table, 9), expected)


def test_paged_manager_reuses_blocks_and_reports_internal_fragmentation() -> None:
    manager = PagedKVCacheManager(num_blocks=4, block_size_tokens=4)
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
