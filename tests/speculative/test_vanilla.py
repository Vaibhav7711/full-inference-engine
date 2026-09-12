from engine.speculative import greedy_accept


def test_greedy_accepts_all_and_emits_bonus() -> None:
    result = greedy_accept([1, 2, 3], [1, 2, 3], 4)
    assert result.emitted == [1, 2, 3, 4]
    assert result.accepted_draft_tokens == 3
    assert result.fully_accepted


def test_greedy_rejects_at_first_mismatch() -> None:
    result = greedy_accept([1, 2, 3], [1, 9, 3], 4)
    assert result.emitted == [1, 9]
    assert result.accepted_draft_tokens == 1
    assert not result.fully_accepted
