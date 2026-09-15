from __future__ import annotations

import pytest

from engine.runtime import RequestState
from engine.server.continuous import ContinuousBatchingService


class _FakeEngine:
    def __init__(self) -> None:
        self.requests = []

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.requests)

    def submit(self, request) -> bool:
        request.transition(RequestState.PREFILLING)
        self.requests.append(request)
        return True

    def step(self) -> None:
        for request in list(self.requests):
            if request.state is RequestState.PREFILLING:
                request.advance_prefill(request.remaining_prefill_tokens)
                request.transition(RequestState.DECODING)
            request.append_token(7)
            if len(request.output_token_ids) == request.max_new_tokens:
                request.transition(RequestState.FINISHED, reason="LENGTH")
                self.requests.remove(request)

    def cancel(self, request_id, reason="CANCELLED") -> None:
        for request in list(self.requests):
            if request.request_id == request_id:
                request.transition(RequestState.CANCELLED, reason=reason)
                self.requests.remove(request)
                return request
        raise KeyError(request_id)


def test_background_service_completes_request_without_handler_touching_engine() -> None:
    service = ContinuousBatchingService(_FakeEngine())
    service.start()
    try:
        handle = service.submit([1, 2, 3], max_new_tokens=3)
        assert handle.completed.wait(1)
        assert handle.error is None
        assert handle.request.output_token_ids == [7, 7, 7]
        assert handle.request.finish_reason == "LENGTH"
    finally:
        service.stop()


def test_background_service_routes_cancellation_to_worker_owned_engine() -> None:
    engine = _FakeEngine()
    service = ContinuousBatchingService(engine)
    handle = service.submit([1, 2, 3], max_new_tokens=3)
    service.cancel(handle, reason="TEST_CANCEL")
    service.start()
    try:
        assert handle.completed.wait(1)
        assert handle.request.state is RequestState.CANCELLED
        assert handle.request.finish_reason == "TEST_CANCEL"
        assert service.snapshot()["cancelled_requests"] == 1
    finally:
        service.stop()


def test_submission_queue_applies_backpressure_before_worker_start() -> None:
    service = ContinuousBatchingService(_FakeEngine(), max_pending_submissions=1)
    service.submit([1], max_new_tokens=1)
    with pytest.raises(RuntimeError, match="submission queue is full"):
        service.submit([2], max_new_tokens=1)
