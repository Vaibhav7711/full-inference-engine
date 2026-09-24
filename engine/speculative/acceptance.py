"""Pure greedy acceptance and transactional commit planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Sequence


@dataclass(frozen=True)
class GreedyCommit:
    """Result of one target verification round.

    ``cached_input_count`` is how many verified inputs become logically visible in KV.
    The final emitted token remains the next pending input and is therefore not included
    in that count; the old pending token is, so the count equals the number emitted.
    """

    emitted: tuple[int, ...]
    accepted_draft_tokens: int
    cached_input_count: int
    fully_accepted: bool
    terminal_reason: str | None = None


def plan_greedy_commit(
    proposals: Sequence[int],
    target_predictions: Sequence[int],
    bonus_token: int,
    *,
    remaining_tokens: int,
    eos_token_ids: Collection[int] = (),
    stop_token_ids: Collection[int] = (),
    ignore_eos: bool = False,
) -> GreedyCommit:
    """Accept a draft prefix and plan the exact logical KV/output commit.

    ``target_predictions[i]`` is the target token after the old pending input and the
    first ``i`` accepted proposals. ``bonus_token`` is the prediction after all proposal
    tokens have themselves been verified and written.
    """

    draft = tuple(int(token) for token in proposals)
    target = tuple(int(token) for token in target_predictions)
    if not draft or len(draft) != len(target):
        raise ValueError("proposals and target_predictions must be non-empty and equal length")
    if remaining_tokens <= 0:
        raise ValueError("remaining_tokens must be positive")

    accepted = 0
    for draft_token, target_token in zip(draft, target):
        if draft_token != target_token:
            break
        accepted += 1

    fully_accepted = accepted == len(draft)
    if fully_accepted:
        candidates = draft + (int(bonus_token),)
    else:
        candidates = draft[:accepted] + (target[accepted],)

    emitted: list[int] = []
    terminal_reason = None
    eos = set(eos_token_ids)
    stop = set(stop_token_ids)
    for token in candidates[:remaining_tokens]:
        emitted.append(token)
        if token in stop:
            terminal_reason = "STOP"
            break
        if not ignore_eos and token in eos:
            terminal_reason = "EOS"
            break
    if terminal_reason is None and len(emitted) == remaining_tokens:
        terminal_reason = "LENGTH"

    return GreedyCommit(
        emitted=tuple(emitted),
        accepted_draft_tokens=min(accepted, len(emitted)),
        cached_input_count=len(emitted),
        fully_accepted=fully_accepted,
        terminal_reason=terminal_reason,
    )
