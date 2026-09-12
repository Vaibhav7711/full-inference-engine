import torch

from engine.batching.static import StaticBatchRunner


def test_left_padded_position_ids_ignore_padding() -> None:
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    positions = StaticBatchRunner._prefill_positions(mask)
    assert positions.tolist() == [[0, 0, 0, 1], [0, 0, 1, 2]]
