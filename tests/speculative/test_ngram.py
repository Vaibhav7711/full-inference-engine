from engine.speculative import NgramProposer


def test_ngram_prefers_longest_then_most_recent_match() -> None:
    proposer = NgramProposer(min_match=2, max_match=4)
    # Current suffix [1,2] occurred twice; the most recent continuation is [8,9].
    result = proposer.propose([1, 2, 7, 1, 2, 8, 9, 1, 2], 3)
    assert result.token_ids == (8, 9, 1)
    assert result.matched_tokens == 2


def test_ngram_returns_no_proposal_without_observed_continuation() -> None:
    proposer = NgramProposer(min_match=2, max_match=4)
    assert proposer.propose([1, 2, 3, 4], 4).token_ids == ()
    assert proposer.propose([1, 2, 1, 2], 0).token_ids == ()


def test_ngram_validates_match_range() -> None:
    try:
        NgramProposer(min_match=0, max_match=4)
    except ValueError as error:
        assert "min_match" in str(error)
    else:
        raise AssertionError("invalid n-gram range was accepted")
