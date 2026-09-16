"""LIFO preemption: the newest request yields blocks, keeps its tokens, and resumes first."""

from __future__ import annotations

from engine.cache import KVBlockManager, PrefixCache
from engine.runtime import GenerationRequest, RequestState
from engine.scheduler import FCFSScheduler


def _request(request_id: str, prompt: int, max_new: int = 4) -> GenerationRequest:
    return GenerationRequest(request_id, prompt_token_count=prompt, max_new_tokens=max_new,
                             prompt_token_ids=list(range(1, prompt + 1)))


def test_preempt_releases_blocks_and_requeues_at_head() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    older, newer = _request("older", 8), _request("newer", 8)
    scheduler.submit(older)
    scheduler.submit(newer)
    scheduler.submit(_request("queued", 4))
    assert [r.request_id for r in scheduler.admit_available(max_active_requests=2)] == ["older", "newer"]
    manager.append_tokens("newer", 8)
    used_before = manager.snapshot()["used_blocks"]

    assert scheduler.newest_active() is newer
    scheduler.preempt("newer")

    assert newer.state is RequestState.WAITING and newer.preempted_count == 1
    assert manager.snapshot()["used_blocks"] < used_before
    assert "newer" not in scheduler.active
    assert scheduler.waiting[0] is newer and scheduler.waiting[1].request_id == "queued"
    assert scheduler.snapshot()["preemption_count"] == 1
    # It resumes before the request that arrived after it.
    assert [r.request_id for r in scheduler.admit_available()] == ["newer", "queued"]


def test_resumed_request_keeps_its_arrival_priority() -> None:
    manager = KVBlockManager(num_blocks=64, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    first, second = _request("first", 4), _request("second", 4)
    scheduler.submit(first)
    scheduler.submit(second)
    scheduler.admit_available()
    scheduler.preempt("first")
    scheduler.admit_available()  # first re-enters active after second in dict order
    assert list(scheduler.active) == ["second", "first"]
    # Victim selection is by arrival, so the resumed request is not the next victim.
    assert scheduler.newest_active() is second


def test_resumed_request_prefills_prompt_plus_generated_prefix_and_reattaches_prefix() -> None:
    manager = KVBlockManager(num_blocks=64, block_size_tokens=4)
    cache = PrefixCache(manager, max_blocks=16)
    scheduler = FCFSScheduler(manager, prefix_cache=cache)
    request = _request("r", 8, max_new=6)
    scheduler.submit(request)
    scheduler.admit_available()
    manager.append_tokens("r", 8)
    request.advance_prefill(8)
    cache.publish(request.prompt_token_ids, request.allocation, next_token_id=100)
    scheduler.mark_decoding("r")
    for token in (100, 101, 102):
        request.append_token(token)
        manager.append_tokens("r", 1)

    scheduler.preempt("r")
    assert request.prefill_token_ids == request.prompt_token_ids + [100, 101]
    admitted = scheduler.admit_available()
    assert admitted == [request]
    # The two complete prompt blocks came back from the prefix cache; only the tail remains.
    assert request.prefilled_token_count == 8
    assert request.remaining_prefill_tokens == 2


def test_cancel_finds_a_preempted_request_in_the_queue() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    request = _request("r", 4)
    scheduler.submit(request)
    scheduler.admit_available()
    scheduler.preempt("r")
    cancelled = scheduler.cancel("r")
    assert cancelled is request and request.state is RequestState.CANCELLED
    assert not scheduler.waiting and manager.snapshot()["used_blocks"] == 0
