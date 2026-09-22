"""Stage 7 request lifecycle and state-transition invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter_ns

from engine.cache import KVBlockAllocation
from engine.runtime.sampling import GREEDY, SamplingParams


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
    # PREFILLING/DECODING -> WAITING is preemption: the request yields its KV blocks under
    # memory pressure and re-enters the queue head to be recomputed later.
    RequestState.PREFILLING: {
        RequestState.DECODING, RequestState.WAITING, RequestState.CANCELLED, RequestState.FAILED,
    },
    RequestState.DECODING: {
        RequestState.FINISHED, RequestState.WAITING, RequestState.CANCELLED, RequestState.FAILED,
    },
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
    # How this request's logits become tokens. The default is greedy decoding, which is
    # what every benchmark and the token-identity gate assume.
    sampling: SamplingParams = GREEDY
    # Per generated token, [(token_id, logprob), ...] when the request asked for
    # logprobs; empty otherwise. Positionally aligned with `output_token_ids`.
    output_logprobs: list[list[tuple[int, float]]] = field(default_factory=list)
    next_token_id: int | None = None
    finish_reason: str | None = None
    prefilled_token_count: int = 0
    first_token_ns: int | None = None
    last_token_ns: int | None = None
    token_timestamps_ns: list[int] = field(default_factory=list)
    cached_prefix_tokens: int = 0
    cached_next_token_id: int | None = None
    preempted_count: int = 0
    # True while a preempted request is being recomputed. Its prefill sequence is then
    # the prompt plus every generated token except the last, which is the pending input.
    resuming: bool = False
    # True from the moment a request is preempted until its rebuilt KV re-enters decode.
    # Unlike ``resuming`` this also covers a request preempted before it produced any
    # token, whose replayed prompt is recompute work too.
    recomputing: bool = False
    # Scheduler progress epoch at which this request last yielded. It is not re-admitted
    # until the epoch advances, so a retry only happens after memory is actually freed.
    preempted_at_epoch: int = 0
    # Gate 1B recompute cost accounting.
    recomputed_token_count: int = 0
    recompute_ns: int = 0
    preempted_wait_ns: int = 0
    preempted_ns: int | None = None
    resume_started_ns: int | None = None
    # Time a request spent *not* decoding between its first and last output token,
    # because it was preempted and rebuilt. Separating this lets generation time be
    # reported as wall clock and as pure decode time, instead of conflating the two.
    stalled_ns: int = 0
    # True when this request still owned KV pages at the moment it terminated, i.e. its
    # exit returned memory to the pool. False for a request rejected at admission or
    # cancelled while parked in the queue, which held nothing.
    held_pages_at_exit: bool = False

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must not be empty")
        if self.prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.prompt_token_ids and len(self.prompt_token_ids) != self.prompt_token_count:
            raise ValueError("prompt_token_ids length must equal prompt_token_count")
        if not 0 <= self.prefilled_token_count <= self.prompt_token_count:
            raise ValueError("prefilled_token_count must be within the prompt")
        if not 0 <= self.cached_prefix_tokens <= self.prefilled_token_count:
            raise ValueError("cached_prefix_tokens must be committed prefill tokens")

    @property
    def reserved_tokens(self) -> int:
        return self.prompt_token_count + self.max_new_tokens

    @property
    def prefill_token_ids(self) -> list[int]:
        """Token sequence whose KV must exist before this request can decode."""
        if self.resuming and self.output_token_ids:
            return self.prompt_token_ids + self.output_token_ids[:-1]
        return self.prompt_token_ids

    @property
    def prefill_token_count(self) -> int:
        if self.resuming and self.output_token_ids:
            return self.prompt_token_count + len(self.output_token_ids) - 1
        return self.prompt_token_count

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

    @property
    def remaining_prefill_tokens(self) -> int:
        return self.prefill_token_count - self.prefilled_token_count

    def preempt(self, *, reason: str = "PREEMPTED", epoch: int = 0) -> None:
        """Yield all KV state and return to the queue; generated tokens are kept."""
        if self.state not in {RequestState.PREFILLING, RequestState.DECODING}:
            raise RuntimeError("only PREFILLING or DECODING requests can be preempted")
        self.transition(RequestState.WAITING, reason=reason)
        self.allocation = None
        self.prefilled_token_count = 0
        self.cached_prefix_tokens = 0
        self.cached_next_token_id = None
        self.resuming = bool(self.output_token_ids)
        self.recomputing = True
        self.preempted_count += 1
        self.preempted_at_epoch = epoch
        self.preempted_ns = perf_counter_ns()
        self.resume_started_ns = None

    def complete_resumption(self) -> None:
        """Restore ordinary prompt accounting after rebuilt KV enters decode.

        While ``resuming`` is true, prefill accounting includes generated tokens except
        the pending decode input. The `PREFILLING -> DECODING` guard must observe that
        expanded sequence first. Only after the transition may normal prompt accounting
        be restored.
        """
        if self.state is not RequestState.DECODING or not self.resuming:
            raise RuntimeError("only a resumed DECODING request can complete resumption")
        self.resuming = False
        self.prefilled_token_count = self.prompt_token_count

    def advance_prefill(self, count: int) -> None:
        if self.state is not RequestState.PREFILLING:
            raise RuntimeError("prefill can advance only while PREFILLING")
        if count <= 0 or count > self.remaining_prefill_tokens:
            raise ValueError("invalid prefill token count")
        if self.recomputing:
            self.recomputed_token_count += count
        self.prefilled_token_count += count

    def transition(self, next_state: RequestState, *, reason: str | None = None) -> None:
        if next_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(f"invalid request transition: {self.state} -> {next_state}")
        if next_state is RequestState.DECODING and self.remaining_prefill_tokens:
            raise RuntimeError("request cannot decode before its prompt is fully prefetched")
        self.state = next_state
        now = perf_counter_ns()
        if next_state is RequestState.PREFILLING:
            if self.admitted_ns is None:
                self.admitted_ns = now
            if self.preempted_ns is not None:
                # Queue time accrued since the yield ends at re-admission, not at the
                # end of the rebuild; the rebuild itself is charged to recompute_ns.
                waited = now - self.preempted_ns
                self.preempted_wait_ns += waited
                if self.first_token_ns is not None:
                    self.stalled_ns += waited
                self.preempted_ns = None
                self.resume_started_ns = now
        if next_state is RequestState.DECODING and self.recomputing:
            self.recomputing = False
            if self.resume_started_ns is not None:
                spent = now - self.resume_started_ns
                self.recompute_ns += spent
                if self.first_token_ns is not None:
                    self.stalled_ns += spent
                self.resume_started_ns = None
        if next_state is not RequestState.WAITING and self.preempted_ns is not None:
            # Cancelled or failed while parked after a yield: close the wait interval so
            # reported totals stay consistent for terminal requests.
            self.preempted_wait_ns += now - self.preempted_ns
            self.preempted_ns = None
        if next_state in {RequestState.FINISHED, RequestState.CANCELLED, RequestState.FAILED, RequestState.REJECTED}:
            self.finish_reason = reason
            # Drop the allocation handle with the state. The scheduler has already
            # returned the pages, so keeping the reference would leave the request
            # pointing at block ids that now belong to somebody else - harmless until
            # something reads `block_table` on a finished request, then not harmless.
            self.allocation = None

    def append_token(self, token_id: int) -> None:
        if self.state is not RequestState.DECODING:
            raise RuntimeError("tokens can only be appended while DECODING")
        if len(self.output_token_ids) >= self.max_new_tokens:
            raise RuntimeError("request has reached max_new_tokens")
        self.output_token_ids.append(token_id)
        now = perf_counter_ns()
        if self.first_token_ns is None:
            self.first_token_ns = now
        self.last_token_ns = now
        self.token_timestamps_ns.append(now)

    def queue_time_ms(self) -> float | None:
        """Time from arrival to *first* admission. Excludes later preemption waits."""
        if self.admitted_ns is None:
            return None
        return (self.admitted_ns - self.created_ns) / 1_000_000

    def total_queue_time_ms(self) -> float | None:
        """Every millisecond this request spent queued, including after a yield.

        `queue_time_ms` alone understates a preempted request, because `admitted_ns` is
        recorded once and never moves. Report this whenever queueing is the question.
        """
        first = self.queue_time_ms()
        if first is None:
            return None
        return first + self.preempted_wait_time_ms()

    def time_to_first_token_ms(self) -> float | None:
        if self.first_token_ns is None:
            return None
        return (self.first_token_ns - self.created_ns) / 1_000_000

    def generation_time_ms(self) -> float | None:
        """Wall-clock span from first to last token, stalls included."""
        if self.first_token_ns is None or self.last_token_ns is None:
            return None
        return (self.last_token_ns - self.first_token_ns) / 1_000_000

    def stall_time_ms(self) -> float:
        """Generation time lost to preemption: queued plus rebuilding, after token one."""
        return self.stalled_ns / 1_000_000

    def decode_time_ms(self) -> float | None:
        """Generation time with preemption stalls removed - time actually spent decoding."""
        wall = self.generation_time_ms()
        if wall is None:
            return None
        return max(0.0, wall - self.stall_time_ms())

    def mean_inter_token_latency_ms(self) -> float | None:
        """Average gap between tokens, excluding preemption stalls.

        A preempted request has one enormous gap in `token_timestamps_ns`. Including it
        makes a mean meaningless, so it is removed here; consumers that want the
        user-visible worst case should read the raw timestamps and `stall_time_ms`.
        """
        produced = len(self.output_token_ids)
        decode_ms = self.decode_time_ms()
        if decode_ms is None or produced < 2:
            return None
        return decode_ms / (produced - 1)

    def latency_report(self) -> dict[str, float | int | None]:
        """Every timing number for one request, with queueing and stalls separated."""
        return {
            "queue_ms": self.queue_time_ms(),
            "total_queue_ms": self.total_queue_time_ms(),
            "ttft_ms": self.time_to_first_token_ms(),
            "generation_ms": self.generation_time_ms(),
            "decode_ms": self.decode_time_ms(),
            "stall_ms": self.stall_time_ms(),
            "mean_itl_ms": self.mean_inter_token_latency_ms(),
            "output_tokens": len(self.output_token_ids),
        }

    def recompute_time_ms(self) -> float:
        """Wall time spent rebuilding KV after preemption (prefill only, not queueing)."""
        return self.recompute_ns / 1_000_000

    def preempted_wait_time_ms(self) -> float:
        """Wall time spent parked in the queue because of preemption."""
        return self.preempted_wait_ns / 1_000_000

    def recompute_overhead(self) -> dict[str, float | int]:
        """Per-request recompute cost, for soak reports and the metrics endpoint."""
        produced = len(self.output_token_ids)
        return {
            "preempted_count": self.preempted_count,
            "recomputed_tokens": self.recomputed_token_count,
            "recompute_ms": self.recompute_time_ms(),
            "preempted_wait_ms": self.preempted_wait_time_ms(),
            # Rebuilt prompt tokens per produced output token: the headline waste ratio.
            "recompute_tokens_per_output_token": (
                self.recomputed_token_count / produced if produced else 0.0
            ),
        }
