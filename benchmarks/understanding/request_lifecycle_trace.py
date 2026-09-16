"""Trace the real CPU request, scheduler, and KV-block state machines.

This intentionally does not load a model. It makes scheduling decisions and block
ownership visible without CUDA execution or timing noise obscuring the transitions.
"""

from __future__ import annotations

import json

from engine.cache import KVBlockManager
from engine.runtime import GenerationRequest, RequestState
from engine.scheduler import FCFSScheduler


def _snapshot(label: str, scheduler: FCFSScheduler, requests: list[GenerationRequest]) -> None:
    rows = []
    for request in requests:
        allocation = request.allocation
        rows.append(
            {
                "id": request.request_id,
                "state": request.state.value,
                "prefill": f"{request.prefilled_token_count}/{request.prompt_token_count}",
                "outputs": list(request.output_token_ids),
                "blocks": list(request.block_table),
                "kv_length": allocation.sequence_length if allocation is not None else None,
                "finish_reason": request.finish_reason,
            }
        )
    state = {
        "waiting": [request.request_id for request in scheduler.waiting],
        "active": list(scheduler.active),
        "prefill_order": list(scheduler._prefill_order),
        "kv": scheduler.block_manager.snapshot(),
        "requests": rows,
    }
    print(f"\n=== {label} ===")
    print(json.dumps(state, indent=2))


def main() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    scheduler = FCFSScheduler(manager, max_waiting_requests=3)
    requests = [
        GenerationRequest("r0", prompt_token_count=6, max_new_tokens=2),
        GenerationRequest("r1", prompt_token_count=5, max_new_tokens=2),
        GenerationRequest("r2", prompt_token_count=4, max_new_tokens=2),
    ]

    for request in requests:
        assert scheduler.submit(request)
    _snapshot("submitted: WAITING owns no KV blocks", scheduler, requests)

    scheduler.admit_available(max_active_requests=3)
    _snapshot("admitted: one initial block per request", scheduler, requests)

    iteration = 0
    while any(request.state is RequestState.PREFILLING for request in requests):
        iteration += 1
        plans = scheduler.plan_prefill(chunk_size=3, token_budget=4)
        print("planned chunks:", [(request.request_id, count) for request, count in plans])
        for request, count in plans:
            assert manager.append_tokens(request.request_id, count)
            request.advance_prefill(count)
            if request.remaining_prefill_tokens == 0:
                # This mirrors the post-forward state changes in prefill_chunks().
                request.next_token_id = 100 + iteration
                scheduler.mark_decoding(request.request_id)
                request.append_token(request.next_token_id)
        _snapshot(f"prefill iteration {iteration}", scheduler, requests)

    # A first output token was predicted from the final prompt position. It is present
    # in output_token_ids but has not entered KV yet. One decode step consumes it,
    # commits one more KV position, predicts output token two, and finishes each request.
    active = list(scheduler.active.values())
    # decode_step() first proves/grows capacity for every row before launching the
    # batched forward. Only after that forward does it commit lengths and finish rows.
    for request in active:
        assert request.allocation is not None
        assert manager.ensure_capacity(request.request_id, request.allocation.sequence_length + 1)
    for request in active:
        assert manager.append_tokens(request.request_id)
        request.append_token(200)
        scheduler.finish(request.request_id, reason="LENGTH")
    _snapshot("decode once, finish, and release all blocks", scheduler, requests)

    # Cancellation has a different path depending on whether work is waiting or active.
    waiting = GenerationRequest("waiting-cancel", 2, 1)
    active = GenerationRequest("active-cancel", 6, 1)
    scheduler.submit(waiting)
    scheduler.submit(active)
    scheduler.cancel("waiting-cancel", reason="CLIENT_DISCONNECTED")
    scheduler.admit_available(max_active_requests=1)
    assert manager.append_tokens("active-cancel", 3)
    active.advance_prefill(3)
    scheduler.cancel("active-cancel", reason="CLIENT_DISCONNECTED")
    _snapshot("waiting and partial-prefill cancellation", scheduler, requests + [waiting, active])


if __name__ == "__main__":
    main()
