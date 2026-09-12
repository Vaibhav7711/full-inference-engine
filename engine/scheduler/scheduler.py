"""Stage 8 FCFS admission scheduler over the contiguous KV baseline."""

from __future__ import annotations

from collections import deque

from engine.cache import ContiguousKVAllocator
from engine.runtime import GenerationRequest, RequestState


class FCFSScheduler:
    def __init__(self, allocator: ContiguousKVAllocator):
        self.allocator = allocator
        self.waiting: deque[GenerationRequest] = deque()
        self.active: dict[str, GenerationRequest] = {}
        self.rejected_count = 0
        self.admission_count = 0

    def submit(self, request: GenerationRequest) -> None:
        if request.state is not RequestState.WAITING:
            raise ValueError("only WAITING requests may be submitted")
        if request.request_id in self.active or any(item.request_id == request.request_id for item in self.waiting):
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        self.waiting.append(request)

    def admit_available(self, *, max_active_requests: int | None = None) -> list[GenerationRequest]:
        """Admit requests in arrival order; a fragmented head blocks later requests."""
        if max_active_requests is not None and max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive when provided")
        admitted: list[GenerationRequest] = []
        while self.waiting:
            if max_active_requests is not None and len(self.active) >= max_active_requests:
                break
            request = self.waiting[0]
            if request.reserved_tokens > self.allocator.capacity_tokens:
                self.waiting.popleft()
                request.transition(RequestState.REJECTED, reason="KV_CAPACITY_EXCEEDED")
                self.rejected_count += 1
                continue
            allocation = self.allocator.allocate(request.request_id, request.reserved_tokens)
            if allocation is None:
                break
            self.waiting.popleft()
            request.allocation = allocation
            request.transition(RequestState.PREFILLING)
            self.active[request.request_id] = request
            self.admission_count += 1
            admitted.append(request)
        return admitted

    def mark_decoding(self, request_id: str) -> GenerationRequest:
        request = self.active[request_id]
        request.transition(RequestState.DECODING)
        return request

    def finish(self, request_id: str, reason: str = "EOS") -> GenerationRequest:
        request = self.active.pop(request_id)
        self.allocator.release(request_id)
        request.transition(RequestState.FINISHED, reason=reason)
        return request

    def cancel(self, request_id: str, reason: str = "CANCELLED_BY_CLIENT") -> GenerationRequest:
        for request in self.waiting:
            if request.request_id == request_id:
                self.waiting.remove(request)
                request.transition(RequestState.CANCELLED, reason=reason)
                return request
        request = self.active.pop(request_id)
        self.allocator.release(request_id)
        request.transition(RequestState.CANCELLED, reason=reason)
        return request

    def snapshot(self) -> dict[str, object]:
        return {
            "waiting_requests": len(self.waiting),
            "active_requests": len(self.active),
            "admission_count": self.admission_count,
            "rejected_count": self.rejected_count,
            "head_request_id": self.waiting[0].request_id if self.waiting else None,
            "allocator": self.allocator.snapshot(),
        }
