"""Common contracts for engine-native speculative token proposers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Proposal:
    """A proposal for tokens *after* a request's already-emitted pending token."""

    token_ids: tuple[int, ...]
    method: str
    matched_tokens: int = 0

    def __post_init__(self) -> None:
        if any(token < 0 for token in self.token_ids):
            raise ValueError("proposal token ids must be non-negative")
        if self.matched_tokens < 0:
            raise ValueError("matched_tokens must be non-negative")

    @property
    def depth(self) -> int:
        return len(self.token_ids)


class TokenProposer(Protocol):
    """CPU-facing proposal interface used by the live scheduler."""

    name: str

    def propose(
        self, history: Sequence[int], max_tokens: int, *, request_id: str | None = None,
    ) -> Proposal:
        """Return at most ``max_tokens`` continuations after ``history``."""

    def forget(self, request_id: str) -> None:
        """Release optional request-local state after a terminal transition."""
