import pytest
import torch

from engine.cache import ContiguousKVAllocator, KVCacheGeometry


@pytest.fixture
def allocator() -> ContiguousKVAllocator:
    return ContiguousKVAllocator(10, KVCacheGeometry(1, 1, 1, torch.float16))


def test_contiguous_allocator_exposes_fragmentation(allocator: ContiguousKVAllocator) -> None:
    assert allocator.allocate("a", 3)
    assert allocator.allocate("b", 3)
    assert allocator.allocate("c", 3)
    allocator.release("b")
    assert allocator.free_tokens == 4
    assert allocator.largest_free_range_tokens == 3
    assert allocator.external_fragmentation == 0.25
    assert allocator.allocate("large", 4) is None


def test_release_merges_adjacent_ranges(allocator: ContiguousKVAllocator) -> None:
    allocator.allocate("a", 3)
    allocator.allocate("b", 3)
    allocator.release("a")
    allocator.release("b")
    assert allocator.snapshot()["free_ranges"] == [(0, 10)]
