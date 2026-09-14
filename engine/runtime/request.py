"""Stage 7 request lifecycle and state-transition invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter_ns

from engine.cache import KVBlockAllocation


class RequestState(StrEnum):
    WAITING = "WAITING"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    FINISHED = "FINISHED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


_ALLOWED_TRANSITIONS = {
    RequestState.WAITING: {RequestState.PREFILLING, RequestState.CANCELLED, RequestState.REJECTED},
    RequestState.PREFILLING: {RequestState.DECODING, RequestState.CANCELLED, RequestState.FAILED},
    RequestState.DECODING: {RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED},
    RequestState.FINISHED: set(),
    RequestState.CANCELLED: set(),
    RequestState.FAILED: set(),
    RequestState.REJECTED: set(),
}


@dataclass
class GenerationRequest:
    request_id: str
    prompt_token_count: int
    max_new_tokens: int
    state: RequestState = RequestState.WAITING
    created_ns: int = field(default_factory=perf_counter_ns)
    admitted_ns: int | None = None
    prompt_token_ids: list[int] = field(default_factory=list)
    allocation: KVBlockAllocation | None = None
    output_token_ids: list[int] = field(default_factory=list)
    next_token_id: int | None = None
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.prompt_token_ids and len(self.prompt_token_ids) != self.prompt_token_count:
            raise ValueError("prompt_token_ids length must equal prompt_token_count")

    @property
    def reserved_tokens(self) -> int:
        return self.prompt_token_count + self.max_new_tokens

    @property
    def block_table(self) -> list[int]:
        if self.allocation is None:
            return []
        return self.allocation.physical_block_ids

    @property
    def done(self) -> bool:
        return self.state in {
            RequestState.FINISHED,
            RequestState.CANCELLED,
            RequestState.FAILED,
            RequestState.REJECTED,
        }

    def transition(self, next_state: RequestState, *, reason: str | None = None) -> None:
        if next_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"invalid request transition: {self.state} -> {next_state}")
        self.state = next_state
        if next_state is RequestState.PREFILLING:
            self.admitted_ns = perf_counter_ns()
        if next_state in {RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED, RequestState.REJECTED}:
            self.finish_reason = reason

    def append_token(self, token_id: int) -> None:
        if self.state is not RequestState.DECODING:
            raise RuntimeError("tokens can only be appended while DECODING")
        if len(self.output_token_ids) >= self.max_new_tokens:
            raise RuntimeError("request has reached max_new_tokens")
        self.output_token_ids.append(token_id)

    def queue_time_ms(self) -> float | None:
        if self.admitted_ns is None:
            return None
        return (self.admitted_ns - self.created_ns) / 1_000_000
