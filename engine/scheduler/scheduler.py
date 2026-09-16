"""FCFS admission and lifecycle scheduler over production KV blocks.

Preemption policy (Gate 1B) — *strict-LIFO recompute with progress-gated readmission*:

1. Victim order is strictly by arrival, newest first. The oldest active request is never
   chosen as a victim, so it always runs to completion and therefore always releases its
   blocks. That is the system's progress guarantee.
2. A request that needs capacity and is itself the newest active request yields itself,
   because it is the correct LIFO victim. If it is the *only* active request, nothing
   else can free memory for it and the engine fails it instead of looping.
3. A yielded request is not re-admitted until the progress epoch advances past the epoch
   at which it yielded. The epoch advances only when a request reaches a terminal state,
   i.e. when blocks are permanently released. Retrying before then is pure recompute
   waste. The gate is ignored when no request is active, so the GPU never idles.
4. There is no preemption-count limit. Termination follows from (1)-(3): the oldest
   request always completes, every completion advances the epoch, and every yield
   strictly reduces the active set.
5. Admission is checked against *effective* capacity - the pool minus blocks reserved
   permanently for engine use (CUDA-Graph dummy rows) - so a request that is admitted can
   in principle be served once it is alone.
6. A yielded request re-enters the queue in arrival order, never blindly at the head, so
   it cannot overtake an older request that is still waiting for its first admission.
"""

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
        reserved_blocks: int = 0,
    ):
        if max_waiting_requests is not None and max_waiting_requests <= 0:
            raise ValueError("max_waiting_requests must be positive when provided")
        if not 0 <= reserved_blocks < block_manager.allocator.num_blocks:
            raise ValueError("reserved_blocks must leave at least one usable block")
        self.block_manager = block_manager
        self.max_waiting_requests = max_waiting_requests
        self.prefix_cache = prefix_cache
        self.reserved_blocks = reserved_blocks
        self.waiting: deque[GenerationRequest] = deque()
        self.active: dict[str, GenerationRequest] = {}
        self._prefill_order: deque[str] = deque()
        self.rejected_count = 0
        self.admission_count = 0
        self.preemption_count = 0
        self.recomputed_tokens_total = 0
        self.recompute_ns_total = 0
        # Advances whenever a request reaches a terminal state and its blocks are gone
        # for good. Yielded requests wait for this to move before they are retried.
        self.progress_epoch = 0

    @property
    def effective_capacity_tokens(self) -> int:
        """Pool capacity a single request can actually obtain, alone, once drained."""
        usable = self.block_manager.allocator.num_blocks - self.reserved_blocks
        return usable * self.block_manager.block_size_tokens

    def _advance_progress(self, request: GenerationRequest) -> None:
        """Record a terminal request: bank its recompute cost and unblock yielded peers."""
        request.held_pages_at_exit = True
        self.progress_epoch += 1
        self.recomputed_tokens_total += request.recomputed_token_count
        self.recompute_ns_total += request.recompute_ns

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
            if request.reserved_tokens > self.effective_capacity_tokens:
                self.waiting.popleft()
                request.transition(RequestState.REJECTED, reason="KV_CAPACITY_EXCEEDED")
                self.rejected_count += 1
                continue
            if (
                request.recomputing
                and request.preempted_at_epoch >= self.progress_epoch
                and self.active
            ):
                # Nothing has been released since this request yielded, so re-admitting it
                # would rebuild its KV only to yield again. Hold the queue: FCFS forbids
                # skipping past it, and the running requests will advance the epoch.
                break
            prefill_ids = request.prefill_token_ids
            match = (
                self.prefix_cache.lookup(prefill_ids)
                if self.prefix_cache is not None and prefill_ids
                else None
            )
            if match is not None and match.token_count:
                allocation = self.block_manager.attach_prefix(
                    request.request_id, list(match.physical_block_ids), match.token_count
                )
                request.prefilled_token_count = match.token_count
                request.cached_prefix_tokens = match.token_count
                request.cached_next_token_id = match.next_token_id
            else:
                initial_tokens = min(request.prefill_token_count, self.block_manager.block_size_tokens)
                allocation = self.block_manager.reserve(
                    request.request_id, initial_tokens, sequence_length=0,
                )
                if allocation is None and self.prefix_cache is not None:
                    self.prefix_cache.evict_until_free(1)
                    allocation = self.block_manager.reserve(
                        request.request_id, initial_tokens, sequence_length=0,
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
        self._advance_progress(request)
        return request

    def fail(self, request_id: str, reason: str) -> GenerationRequest:
        """Release an active request and record a terminal engine failure."""
        request = self.active.pop(request_id)
        self._remove_prefill_order(request_id)
        self.block_manager.release(request_id)
        request.transition(RequestState.FAILED, reason=reason)
        self._advance_progress(request)
        return request

    def newest_active(self) -> GenerationRequest | None:
        """The active request with the lowest FCFS priority (latest arrival)."""
        if not self.active:
            return None
        return max(self.active.values(), key=lambda item: (item.created_ns, item.request_id))

    def preempt(self, request_id: str, reason: str = "PREEMPTED") -> GenerationRequest:
        """Release an active request's KV blocks and requeue it in arrival order.

        Recompute-style preemption: generated tokens are retained on the request, and its
        KV is rebuilt by a later prefill of the prompt plus generated prefix. Yielding
        does not advance the progress epoch - no memory was permanently freed, the blocks
        simply changed hands - which is what keeps the request parked until a peer ends.
        """
        request = self.active.pop(request_id)
        self._remove_prefill_order(request_id)
        self.block_manager.release(request_id)
        request.preempt(reason=reason, epoch=self.progress_epoch)
        self._requeue_in_arrival_order(request)
        self.preemption_count += 1
        return request

    def _requeue_in_arrival_order(self, request: GenerationRequest) -> None:
        """Insert a yielded request ahead of every later arrival, behind every earlier one."""
        key = (request.created_ns, request.request_id)
        for index, queued in enumerate(self.waiting):
            if (queued.created_ns, queued.request_id) > key:
                self.waiting.insert(index, request)
                return
        self.waiting.append(request)

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
        self._advance_progress(request)
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
            "preemption_count": self.preemption_count,
            "progress_epoch": self.progress_epoch,
            "effective_capacity_tokens": self.effective_capacity_tokens,
            "recomputed_tokens_total": self.recomputed_tokens_total,
            "recompute_ms_total": self.recompute_ns_total / 1_000_000,
            "head_request_id": self.waiting[0].request_id if self.waiting else None,
            "block_manager": self.block_manager.snapshot(),
            "prefix_cache": (
                self.prefix_cache.snapshot() if self.prefix_cache is not None else None
            ),
        }
