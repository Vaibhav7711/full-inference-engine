"""Gate 1B: strict-LIFO recompute preemption with progress-gated readmission.

Each test pins one clause of the policy documented on FCFSScheduler.
"""

from __future__ import annotations

import pytest

from engine.cache import KVBlockManager, PrefixCache
from engine.runtime import GenerationRequest, RequestState
from engine.scheduler import FCFSScheduler


def _request(request_id: str, prompt: int, max_new: int = 4) -> GenerationRequest:
    return GenerationRequest(request_id, prompt_token_count=prompt, max_new_tokens=max_new,
                             prompt_token_ids=list(range(1, prompt + 1)))


# --------------------------------------------------------------------- clause 1 and 2
def test_preempt_releases_blocks_and_keeps_generated_tokens() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    older, newer = _request("older", 8), _request("newer", 8)
    scheduler.submit(older)
    scheduler.submit(newer)
    assert [r.request_id for r in scheduler.admit_available()] == ["older", "newer"]
    manager.append_tokens("newer", 8)
    used_before = manager.snapshot()["used_blocks"]

    assert scheduler.newest_active() is newer  # strict LIFO victim
    scheduler.preempt("newer")

    assert newer.state is RequestState.WAITING and newer.preempted_count == 1
    assert newer.recomputing is True
    assert manager.snapshot()["used_blocks"] < used_before
    assert "newer" not in scheduler.active
    assert scheduler.snapshot()["preemption_count"] == 1
    # Yielding does not free memory permanently, so it must not advance the epoch.
    assert scheduler.progress_epoch == 0


def test_oldest_active_request_is_never_the_victim() -> None:
    manager = KVBlockManager(num_blocks=64, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    for name in ("a", "b", "c"):
        scheduler.submit(_request(name, 4))
    scheduler.admit_available()
    for expected in ("c", "b"):
        victim = scheduler.newest_active()
        assert victim.request_id == expected
        scheduler.preempt(victim.request_id)
    assert list(scheduler.active) == ["a"]


# ----------------------------------------------------------------------------- clause 3
def test_yielded_request_is_not_readmitted_until_the_epoch_advances() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    older, newer = _request("older", 8), _request("newer", 8)
    scheduler.submit(older)
    scheduler.submit(newer)
    scheduler.admit_available()
    older.advance_prefill(older.remaining_prefill_tokens)
    scheduler.mark_decoding("older")
    scheduler.preempt("newer")

    # Nothing has terminated: retrying would rebuild KV only to yield again.
    assert scheduler.admit_available() == []
    assert newer.state is RequestState.WAITING

    scheduler.finish("older", reason="LENGTH")
    assert scheduler.progress_epoch == 1
    assert [r.request_id for r in scheduler.admit_available()] == ["newer"]


def test_epoch_gate_is_ignored_when_nothing_is_active() -> None:
    """The GPU must never idle behind the gate."""
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    request = _request("solo", 8)
    scheduler.submit(request)
    scheduler.admit_available()
    scheduler.preempt("solo")
    assert not scheduler.active
    assert [r.request_id for r in scheduler.admit_available()] == ["solo"]


def test_a_cancelled_peer_also_advances_the_epoch() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    scheduler.submit(_request("older", 8))
    scheduler.submit(_request("newer", 8))
    scheduler.admit_available()
    scheduler.preempt("newer")
    assert scheduler.admit_available() == []
    scheduler.cancel("older")
    assert [r.request_id for r in scheduler.admit_available()] == ["newer"]


# ----------------------------------------------------------------------------- clause 4
def test_no_preemption_count_limit_exists() -> None:
    """Yield the same request many times; it must stay serviceable, never failed."""
    manager = KVBlockManager(num_blocks=64, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    keeper = _request("keeper", 4, max_new=200)
    scheduler.submit(keeper)
    victim = _request("victim", 4, max_new=200)
    scheduler.submit(victim)
    scheduler.admit_available()
    for round_index in range(100):
        scheduler.preempt("victim")
        scheduler.progress_epoch += 1  # stand in for a peer completing each round
        assert [r.request_id for r in scheduler.admit_available()] == ["victim"]
    assert victim.preempted_count == 100
    assert victim.state is RequestState.PREFILLING
    assert keeper.state is RequestState.PREFILLING


# ----------------------------------------------------------------------------- clause 5
def test_admission_uses_capacity_minus_permanently_reserved_blocks() -> None:
    manager = KVBlockManager(num_blocks=8, block_size_tokens=4)
    scheduler = FCFSScheduler(manager, reserved_blocks=4)
    assert scheduler.effective_capacity_tokens == 16
    fits = _request("fits", 8, max_new=8)
    too_big = _request("too-big", 8, max_new=12)  # 20 tokens > 16 usable
    scheduler.submit(fits)
    scheduler.submit(too_big)
    admitted = scheduler.admit_available()
    assert [r.request_id for r in admitted] == ["fits"]
    assert too_big.state is RequestState.REJECTED
    assert too_big.finish_reason == "KV_CAPACITY_EXCEEDED"


def test_reserved_blocks_must_leave_usable_capacity() -> None:
    manager = KVBlockManager(num_blocks=4, block_size_tokens=4)
    with pytest.raises(ValueError):
        FCFSScheduler(manager, reserved_blocks=4)


# ----------------------------------------------------------------------------- clause 6
def test_yielded_request_requeues_in_arrival_order_not_at_the_head() -> None:
    manager = KVBlockManager(num_blocks=64, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    early = _request("early", 4)
    late = _request("late", 4)
    scheduler.submit(early)
    scheduler.submit(late)
    # Only `late` is admitted; `early` stays queued behind a full active slot table.
    scheduler.admit_available(max_active_requests=1)
    assert list(scheduler.active) == ["early"]
    scheduler.submit(_request("later-still", 4))
    scheduler.preempt("early")
    # `early` arrived before both queued requests, so it goes back in front of them.
    assert [r.request_id for r in scheduler.waiting] == ["early", "late", "later-still"]


def test_a_yielded_request_does_not_overtake_an_older_waiting_request() -> None:
    manager = KVBlockManager(num_blocks=64, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    first = _request("first", 4)
    second = _request("second", 4)
    scheduler.submit(first)
    scheduler.submit(second)
    scheduler.admit_available(max_active_requests=1)  # admits `first` only
    scheduler.submit(_request("third", 4))
    # Simulate `second` having been admitted and yielded after `first` was requeued.
    scheduler.preempt("first")
    assert [r.request_id for r in scheduler.waiting] == ["first", "second", "third"]


def test_resumed_request_keeps_its_arrival_priority_as_victim_order() -> None:
    manager = KVBlockManager(num_blocks=64, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    first, second = _request("first", 4), _request("second", 4)
    scheduler.submit(first)
    scheduler.submit(second)
    scheduler.admit_available()
    scheduler.preempt("first")
    scheduler.progress_epoch += 1
    scheduler.admit_available()
    assert set(scheduler.active) == {"first", "second"}
    # Victim order is by arrival, so the resumed request is not the next victim.
    assert scheduler.newest_active() is second


# ------------------------------------------------------------------- resume correctness
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
    assert scheduler.admit_available() == [request]  # alone: gate bypassed
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


# ------------------------------------------------------------------- cost accounting
def test_recompute_cost_is_attributed_to_the_request_and_banked_on_completion() -> None:
    manager = KVBlockManager(num_blocks=64, block_size_tokens=4)
    scheduler = FCFSScheduler(manager)
    request = _request("r", 8, max_new=6)
    scheduler.submit(request)
    scheduler.admit_available()
    manager.append_tokens("r", 8)
    request.advance_prefill(8)
    scheduler.mark_decoding("r")
    for token in (100, 101, 102):
        request.append_token(token)
        manager.append_tokens("r", 1)
    assert request.recomputed_token_count == 0  # first prefill is not recompute

    scheduler.preempt("r")
    scheduler.admit_available()
    request.advance_prefill(request.remaining_prefill_tokens)
    scheduler.mark_decoding("r")
    request.complete_resumption()

    # Rebuilt prompt(8) + generated[:-1](2) = 10 tokens to recover 3 output tokens.
    assert request.recomputed_token_count == 10
    assert request.recompute_time_ms() > 0
    assert request.preempted_wait_time_ms() >= 0
    overhead = request.recompute_overhead()
    assert overhead["preempted_count"] == 1
    assert overhead["recompute_tokens_per_output_token"] == pytest.approx(10 / 3)

    scheduler.finish("r", reason="LENGTH")
    assert scheduler.recomputed_tokens_total == 10
    assert scheduler.snapshot()["recompute_ms_total"] > 0
