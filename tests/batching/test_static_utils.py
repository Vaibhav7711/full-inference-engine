import pytest
import torch

from engine.batching.static import StaticBatchRunner


def _engine_class():
    """The engine module imports Triton at module scope; these tests exercise pure
    policy on the class, so they skip rather than error where Triton is absent."""
    return pytest.importorskip(
        "engine.batching.continuous_batching", reason="requires Triton",
    ).ContinuousBatchingEngine


def test_left_padded_position_ids_ignore_padding() -> None:
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    positions = StaticBatchRunner._prefill_positions(mask)
    assert positions.tolist() == [[0, 0, 0, 1], [0, 0, 1, 2]]


def test_graph_dummy_slots_are_distinct_and_fit_their_pages() -> None:
    """Padded CUDA-graph rows need one KV slot each, not one page each.

    At 16-token pages the old per-row page reservation was invisible; at the 256-token
    pages FlashAttention's paged API requires, seven padded rows cost 1,792 tokens of a
    16,384-token pool to store seven. The reservation is now ceil(rows / block_size)
    pages with a distinct slot per row.
    """
    ContinuousBatchingEngine = _engine_class()
    from engine.cache import KVBlockManager

    for block_size, buckets in ((16, (2, 4, 8)), (256, (1, 2, 4, 8)), (16, (8, 16, 32))):
        required = max(buckets) - 1
        expected_pages = -(-required // block_size)

        class _Fake:
            pass

        fake = _Fake()
        fake.cuda_graph_batch_sizes = buckets
        fake.block_size = block_size
        fake.block_manager = KVBlockManager(num_blocks=64, block_size_tokens=block_size)
        ContinuousBatchingEngine._reserve_graph_dummy_blocks(fake)

        assert len(fake._graph_dummy_blocks) == expected_pages, block_size
        assert len(fake._graph_dummy_slots) == required
        # No two padded rows may target the same (page, slot): nothing reads those writes,
        # but two rows writing one address is a hazard nobody should have to reason about.
        assert len(set(fake._graph_dummy_slots)) == required
        for page, slot in fake._graph_dummy_slots:
            assert page in fake._graph_dummy_blocks
            assert 0 <= slot < block_size


def test_unreachable_capture_shapes_are_skipped() -> None:
    """A chunk batch needs rows * context tokens of KV; the pool bounds the product."""
    reachable = _engine_class()._shape_is_reachable
    pool = 16384
    assert reachable(1, 16384, pool)        # one row may use the whole pool
    assert not reachable(2, 16384, pool)    # two cannot
    assert reachable(16, 1024, pool)        # 16 x 1024 == pool exactly
    assert not reachable(16, 2048, pool)    # twice the pool: impossible, never captured
    # The Triton prefill paths read their lengths on the device, so they are unconstrained.
    assert reachable(16, 0, pool)
