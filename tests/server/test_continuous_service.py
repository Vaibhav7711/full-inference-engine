from __future__ import annotations

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
