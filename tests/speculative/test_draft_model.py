from types import SimpleNamespace

import torch
from torch import nn

from engine.speculative import DraftModelProposer


class _Cache:
    def __init__(self, tokens=()):
        self.tokens = list(tokens)

    def crop(self, length: int) -> None:
        del self.tokens[length:]


class _NextTokenModel(nn.Module):
    """Tiny cache-aware model whose greedy token is ``input + 1 mod vocab``."""

    def __init__(self, vocab_size: int = 10):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, past_key_values=None, **_):
        cache = past_key_values if past_key_values is not None else _Cache()
        cache.tokens.extend(input_ids[0].tolist())
        logits = torch.full(
            (*input_ids.shape, self.vocab_size), -100.0, device=input_ids.device,
        )
        expected = (input_ids + 1) % self.vocab_size
        logits.scatter_(2, expected.unsqueeze(-1), 0.0)
        assert attention_mask.shape[1] == len(cache.tokens)
        return SimpleNamespace(past_key_values=cache, logits=logits)


def test_draft_proposer_rolls_back_rejected_suffix_and_advances_correction() -> None:
    proposer = DraftModelProposer(_NextTokenModel(), "cpu")
    first = proposer.propose([1, 2], 3, request_id="r")
    assert first.token_ids == (3, 4, 5)

    # Target accepted 3, rejected the speculative 4, and committed correction 9.
    second = proposer.propose([1, 2, 3, 9], 2, request_id="r")
    assert second.token_ids == (0, 1)
    assert proposer.rollback_tokens == 2
    assert proposer._states["r"].cache.tokens == [1, 2, 3, 9, 0, 1]


def test_draft_proposer_requires_identity_and_forgets_terminal_state() -> None:
    proposer = DraftModelProposer(_NextTokenModel(), "cpu")
    try:
        proposer.propose([1], 1)
    except ValueError as error:
        assert "request_id" in str(error)
    else:
        raise AssertionError("stateful draft proposal accepted no request identity")

    proposer.propose([1], 1, request_id="r")
    proposer.forget("r")
    assert "r" not in proposer._states
