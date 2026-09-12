"""Stage 10 continuous-batching control plane.

This module owns admission, departure, and batch composition. Physical batched KV
execution remains deliberately separate until cache compaction/paging semantics exist.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.runtime import GenerationRequest, RequestState
from engine.scheduler import FCFSScheduler


@dataclass(frozen=True)
class BatchPlan:
    iteration: int
    prefill_request_ids: tuple[str, ...]
    decode_request_ids: tuple[str, ...]

    @property
    def occupancy(self) -> int:
        return len(self.prefill_request_ids) + len(self.decode_request_ids)


class ContinuousBatcher:
    """Dynamically refill fixed request slots as work completes.

    A request admitted during an iteration is planned for prefill only. Existing
    decoding requests contribute one decode token each. Once a request completes, its
    allocation is released and the next waiting request can enter at the next plan.
    """

    def __init__(self, scheduler: FCFSScheduler, max_active_requests: int):
        if max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive")
        self.scheduler = scheduler
        self.max_active_requests = max_active_requests
        self.iteration = 0
        self.occupancy_history: list[int] = []

    def submit(self, request: GenerationRequest) -> None:
        self.scheduler.submit(request)

    def plan_next_iteration(self) -> BatchPlan:
        admitted = self.scheduler.admit_available(max_active_requests=self.max_active_requests)
        decoding = tuple(
            request_id
            for request_id, request in self.scheduler.active.items()
            if request.state is RequestState.DECODING
        )
        plan = BatchPlan(
            iteration=self.iteration,
            prefill_request_ids=tuple(request.request_id for request in admitted),
            decode_request_ids=decoding,
        )
        self.iteration += 1
        self.occupancy_history.append(plan.occupancy)
        return plan

    def mark_prefill_complete(self, request_id: str) -> GenerationRequest:
        return self.scheduler.mark_decoding(request_id)

    def record_decoded_token(self, request_id: str, token_id: int, *, terminal: bool = False) -> GenerationRequest:
        request = self.scheduler.active[request_id]
        request.append_token(token_id)
        if terminal or len(request.output_token_ids) == request.max_new_tokens:
            reason = "EOS" if terminal else "LENGTH"
            return self.scheduler.finish(request_id, reason=reason)
        return request

    def cancel(self, request_id: str, reason: str = "CANCELLED_BY_CLIENT") -> GenerationRequest:
        return self.scheduler.cancel(request_id, reason=reason)

    def snapshot(self) -> dict[str, object]:
        history = self.occupancy_history
        return {
            "iterations": self.iteration,
            "max_active_requests": self.max_active_requests,
            "mean_batch_occupancy": sum(history) / len(history) if history else 0.0,
            "max_batch_occupancy": max(history, default=0),
            "scheduler": self.scheduler.snapshot(),
        }
