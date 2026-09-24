import pytest

from engine.speculative import plan_greedy_commit


@pytest.mark.parametrize("mismatch", range(4))
def test_commit_rejects_at_every_position(mismatch: int) -> None:
    draft = [10, 11, 12, 13]
    target = draft.copy()
    target[mismatch] = 99
    result = plan_greedy_commit(draft, target, 77, remaining_tokens=8)
    assert result.emitted == tuple(draft[:mismatch] + [99])
    assert result.accepted_draft_tokens == mismatch
    assert result.cached_input_count == mismatch + 1
    assert not result.fully_accepted


def test_commit_full_acceptance_emits_bonus() -> None:
    result = plan_greedy_commit([1, 2, 3], [1, 2, 3], 4, remaining_tokens=4)
    assert result.emitted == (1, 2, 3, 4)
    assert result.accepted_draft_tokens == 3
    assert result.cached_input_count == 4
    assert result.fully_accepted
    assert result.terminal_reason == "LENGTH"


@pytest.mark.parametrize(
    ("terminal", "kwargs", "reason"),
    [
        (2, {"eos_token_ids": {2}}, "EOS"),
        (2, {"stop_token_ids": {2}}, "STOP"),
        (2, {"eos_token_ids": {2}, "ignore_eos": True}, None),
    ],
)
def test_commit_truncates_at_terminal(terminal, kwargs, reason) -> None:
    result = plan_greedy_commit([1, terminal, 3], [1, terminal, 3], 4,
                                remaining_tokens=8, **kwargs)
    if reason is None:
        assert result.emitted == (1, 2, 3, 4)
    else:
        assert result.emitted == (1, 2)
        assert result.cached_input_count == 2
    assert result.terminal_reason == reason


def test_stop_precedes_eos() -> None:
    result = plan_greedy_commit([2], [2], 3, remaining_tokens=2,
                                eos_token_ids={2}, stop_token_ids={2})
    assert result.terminal_reason == "STOP"


def test_remaining_budget_truncates_full_round() -> None:
    result = plan_greedy_commit([1, 2, 3], [1, 2, 3], 4, remaining_tokens=2)
    assert result.emitted == (1, 2)
    assert result.cached_input_count == 2
    assert result.terminal_reason == "LENGTH"
