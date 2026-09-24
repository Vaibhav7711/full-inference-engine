"""Deterministic prompt/output lookup proposer with no model or GPU state."""

from __future__ import annotations

from collections.abc import Sequence

from .proposer import Proposal


class NgramProposer:
    """Copy the continuation of the most recent longest suffix match.

    The lookup corpus is the request's prompt plus generated output. For each n-gram
    length, longest first, the current suffix is searched in the earlier history and the
    continuation following its most recent occurrence is proposed. The current suffix
    itself is excluded as a match, and a match without an observed continuation is not a
    proposal.
    """

    name = "ngram"

    def __init__(self, *, min_match: int = 2, max_match: int = 4):
        if min_match <= 0 or max_match < min_match:
            raise ValueError("require 0 < min_match <= max_match")
        self.min_match = min_match
        self.max_match = max_match

    def propose(
        self, history: Sequence[int], max_tokens: int, *, request_id: str | None = None,
    ) -> Proposal:
        if max_tokens <= 0:
            return Proposal((), self.name)
        tokens = tuple(int(token) for token in history)
        largest = min(self.max_match, len(tokens) // 2)
        for width in range(largest, self.min_match - 1, -1):
            suffix_start = len(tokens) - width
            suffix = tokens[suffix_start:]
            # A prior match must end before the current suffix begins and must have at
            # least one observed token after it.
            for start in range(suffix_start - width, -1, -1):
                if tokens[start:start + width] != suffix:
                    continue
                continuation_start = start + width
                continuation_end = min(continuation_start + max_tokens, len(tokens))
                continuation = tokens[continuation_start:continuation_end]
                if continuation:
                    return Proposal(continuation, self.name, matched_tokens=width)
        return Proposal((), self.name)

    def forget(self, request_id: str) -> None:
        """N-gram lookup is stateless; present for the common proposer lifecycle."""
