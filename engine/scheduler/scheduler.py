"""FCFS admission and lifecycle scheduler over production KV blocks."""

from __future__ import annotations

from collections import deque

from engine.cache import KVBlockManager, PrefixCache
from engine.runtime import GenerationRequest, RequestState


class FCFSScheduler:
    def __init__(
        self,
        block_manager: KVBlockManager,
        *,
        max_waiting_requests: int | None = None,
        prefix_cache: PrefixCache | None = None,
    ):
        if max_waiting_requests is not None and max_waiting_requests <= 0:
            raise ValueError("max_waiting_requests must be positive when provided")
        self.block_manager = block_manager
        self.max_waiting_requests = max_waiting_requests
        self.prefix_cache = prefix_cache
        self.waiting: deque[GenerationRequest] = deque()
        self.active: dict[str, GenerationRequest] = {}
        self._prefill_order: deque[str] = deque()
        self.rejected_count = 0
        self.admission_count = 0

    def submit(self, request: GenerationRequest) -> bool:
        if request.state is not RequestState.WAITING:
            raise ValueError("only WAITING requests may be submitted")
        if request.request_id in self.active or any(item.request_id == request.request_id for item in self.waiting):
            raise ValueError(f"duplicate request_id {request.request_id!r}")
        if self.max_waiting_requests is not None and len(self.waiting) >= self.max_waiting_requests:
            request.transition(RequestState.REJECTED, reason="QUEUE_FULL")
            self.rejected_count += 1
            return False
        self.waiting.append(request)
        return True

    def admit_available(self, *, max_active_requests: int | None = None) -> list[GenerationRequest]:
        """Admit requests in arrival order while request slots and blocks are available."""
        if max_active_requests is not None and max_active_requests <= 0:
            raise ValueError("max_active_requests must be positive when provided")
        admitted: list[GenerationRequest] = []
        while self.waiting:
            if max_active_requests is not None and len(self.active) >= max_active_requests:
                break
            request = self.waiting[0]
            if request.reserved_tokens > (
                self.block_manager.allocator.num_blocks * self.block_manager.block_size_tokens
            ):
                self.waiting.popleft()
                request.transition(RequestState.REJECTED, reason="KV_CAPACITY_EXCEEDED")
                self.rejected_count += 1
                continue
            match = (
                self.prefix_cache.lookup(request.prompt_token_ids)
                if self.prefix_cache is not None and request.prompt_token_ids
                else None
            )
            if match is not None and match.token_count:
                allocation = self.block_manager.attach_prefix(
                    request.request_id, list(match.physical_block_ids), match.token_count
                )
                request.prefilled_token_count = match.token_count
                request.cached_prefix_tokens = match.token_count
            else:
                allocation = self.block_manager.reserve(
                    request.request_id,
                    min(request.prompt_token_count, self.block_manager.block_size_tokens),
                    sequence_length=0,
                )
                if allocation is None and self.prefix_cache is not None:
                    self.prefix_cache.evict_until_free(1)
                    allocation = self.block_manager.reserve(
                        request.request_id,
                        min(request.prompt_token_count, self.block_manager.block_size_tokens),
                        sequence_length=0,
                    )
            if allocation is None:
                break
            self.waiting.popleft()
            request.allocation = allocation
            request.transition(RequestState.PREFILLING)
            self.active[request.request_id] = request
            self._prefill_order.append(request.request_id)
            self.admission_count += 1
            admitted.append(request)
        return admitted

    def mark_decoding(self, request_id: str) -> GenerationRequest:
        request = self.active[request_id]
        request.transition(RequestState.DECODING)
        return request

    def finish(self, request_id: str, reason: str = "EOS") -> GenerationRequest:
        request = self.active.pop(request_id)
        self._remove_prefill_order(request_id)
        self.block_manager.release(request_id)
        request.transition(RequestState.FINISHED, reason=reason)
        return request

    def fail(self, request_id: str, reason: str) -> GenerationRequest:
        """Release an active request and record a terminal engine failure."""
        request = self.active.pop(request_id)
        self._remove_prefill_order(request_id)
        self.block_manager.release(request_id)
        request.transition(RequestState.FAILED, reason=reason)
        return request

    def cancel(self, request_id: str, reason: str = "CANCELLED_BY_CLIENT") -> GenerationRequest:
        for request in self.waiting:
            if request.request_id == request_id:
                self.waiting.remove(request)
                request.transition(RequestState.CANCELLED, reason=reason)
                return request
        request = self.active.pop(request_id)
        self._remove_prefill_order(request_id)
        self.block_manager.release(request_id)
        request.transition(RequestState.CANCELLED, reason=reason)
        return request

    def plan_prefill(
        self, *, chunk_size: int, token_budget: int
    ) -> list[tuple[GenerationRequest, int]]:
        """Round-robin at most one prompt chunk per active request."""
        if chunk_size <= 0 or token_budget <= 0:
            raise ValueError("prefill chunk size and token budget must be positive")
        plans: list[tuple[GenerationRequest, int]] = []
        visits = len(self._prefill_order)
        for _ in range(visits):
            request_id = self._prefill_order.popleft()
            request = self.active.get(request_id)
            if request is not None:
                self._prefill_order.append(request_id)
            if request is None or request.state is not RequestState.PREFILLING:
                continue
            count = min(request.remaining_prefill_tokens, chunk_size, token_budget)
            if count:
                plans.append((request, count))
                token_budget -= count
            if token_budget == 0:
                break
        return plans

    def _remove_prefill_order(self, request_id: str) -> None:
        try:
            self._prefill_order.remove(request_id)
        except ValueError:
            pass

    def snapshot(self) -> dict[str, object]:
        return {
            "waiting_requests": len(self.waiting),
            "active_requests": len(self.active),
            "admission_count": self.admission_count,
            "rejected_count": self.rejected_count,
            "head_request_id": self.waiting[0].request_id if self.waiting else None,
            "block_manager": self.block_manager.snapshot(),
            "prefix_cache": (
                self.prefix_cache.snapshot() if self.prefix_cache is not None else None
            ),
        }
