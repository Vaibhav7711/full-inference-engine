"""Gate 1 chaos suite for the HTTP layer: every failure ends in a terminal state with the
right status code, and only engine-level failures take the worker down.

Runs on CPU against a scripted fake engine so the API contract is tested independently
of the GPU. The real-engine counterpart is tests/batching/test_preemption_cuda.py.
"""

from __future__ import annotations

import time
from threading import Lock

import pytest
from fastapi.testclient import TestClient

from engine.runtime import RequestState
from engine.server.api import create_app


class _FakeTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt"):
        ids = [len(word) for word in prompt.split()]

        class _Row(list):
            def tolist(self):
                return list(self)

        class _Batch:
            input_ids = [_Row(ids)]

        return _Batch()

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(f"<{i}>" for i in ids)


class _ScriptedEngine:
    """Minimal stand-in for ContinuousBatchingEngine with failure knobs."""

    def __init__(self, *, step_delay_s: float = 0.0, reject_reason: str | None = None,
                 fail_reason: str | None = None, explode: bool = False,
                 max_model_len: int | None = 64):
        self.tokenizer = _FakeTokenizer()
        self.max_model_len = max_model_len
        self.step_delay_s = step_delay_s
        self.reject_reason = reject_reason
        self.fail_reason = fail_reason
        self.explode = explode
        self.requests = []
        self.cancelled: list[tuple[str, str]] = []
        self._lock = Lock()

    @property
    def has_unfinished_requests(self) -> bool:
        with self._lock:
            return bool(self.requests)

    def submit(self, request) -> bool:
        if self.reject_reason is not None:
            request.transition(RequestState.REJECTED, reason=self.reject_reason)
            return False
        request.transition(RequestState.PREFILLING)
        with self._lock:
            self.requests.append(request)
        return True

    def step(self) -> None:
        if self.explode:
            raise RuntimeError("simulated CUDA failure")
        time.sleep(self.step_delay_s)
        with self._lock:
            batch = list(self.requests)
        for request in batch:
            if request.state is RequestState.PREFILLING:
                request.advance_prefill(request.remaining_prefill_tokens)
                if self.fail_reason is not None:
                    request.transition(RequestState.FAILED, reason=self.fail_reason)
                    self._remove(request)
                    continue
                request.transition(RequestState.DECODING)
            request.append_token(7)
            if len(request.output_token_ids) >= request.max_new_tokens:
                request.transition(RequestState.FINISHED, reason="LENGTH")
                self._remove(request)

    def _remove(self, request) -> None:
        with self._lock:
            if request in self.requests:
                self.requests.remove(request)

    def cancel(self, request_id: str, reason: str = "CANCELLED"):
        with self._lock:
            for request in list(self.requests):
                if request.request_id == request_id:
                    request.transition(RequestState.CANCELLED, reason=reason)
                    self.requests.remove(request)
                    self.cancelled.append((request_id, reason))
                    return request
        raise KeyError(request_id)


def _app(engine: _ScriptedEngine, **kwargs):
    return create_app(engine_factory=lambda: engine, max_prompt_tokens=8,
                      request_timeout_s=kwargs.pop("request_timeout_s", 5.0),
                      drain_timeout_s=kwargs.pop("drain_timeout_s", 1.0), **kwargs)


def test_successful_generation_carries_request_id_and_metrics() -> None:
    with TestClient(_app(_ScriptedEngine())) as client:
        response = client.post("/generate", json={"prompt": "a bb ccc", "max_new_tokens": 3})
        assert response.status_code == 200
        body = response.json()
        assert body["token_ids"] == [7, 7, 7] and body["finish_reason"] == "LENGTH"
        assert response.headers["x-request-id"] == body["request_id"]
        assert body["metrics"]["preemptions"] == 0


def test_zero_token_and_oversized_prompts_are_client_errors() -> None:
    with TestClient(_app(_ScriptedEngine())) as client:
        assert client.post("/generate", json={"prompt": "   ", "max_new_tokens": 1}).status_code == 400
        too_long = " ".join(["x"] * 9)
        assert client.post("/generate", json={"prompt": too_long, "max_new_tokens": 1}).status_code == 413
        # 8 prompt tokens + 60 requested exceeds the fake model context of 64.
        assert client.post("/generate", json={"prompt": " ".join(["x"] * 8), "max_new_tokens": 60}).status_code == 400


def test_scheduler_rejections_map_to_429_and_413() -> None:
    with TestClient(_app(_ScriptedEngine(reject_reason="QUEUE_FULL"))) as client:
        response = client.post("/generate", json={"prompt": "a b", "max_new_tokens": 1})
        assert response.status_code == 429 and response.json()["error"] == "QUEUE_FULL"
    with TestClient(_app(_ScriptedEngine(reject_reason="KV_CAPACITY_EXCEEDED"))) as client:
        assert client.post("/generate", json={"prompt": "a b", "max_new_tokens": 1}).status_code == 413


def test_pool_exhaustion_failure_is_503_and_worker_survives() -> None:
    engine = _ScriptedEngine(fail_reason="KV_POOL_EXHAUSTED")
    with TestClient(_app(engine)) as client:
        response = client.post("/generate", json={"prompt": "a b", "max_new_tokens": 2})
        assert response.status_code == 503 and response.json()["error"] == "KV_POOL_EXHAUSTED"
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        engine.fail_reason = None
        assert client.post("/generate", json={"prompt": "a b", "max_new_tokens": 2}).status_code == 200


def test_stream_rejections_are_status_codes_not_broken_streams() -> None:
    with TestClient(_app(_ScriptedEngine(reject_reason="QUEUE_FULL"))) as client:
        response = client.post("/generate/stream", json={"prompt": "a b", "max_new_tokens": 1})
        assert response.status_code == 429
    with TestClient(_app(_ScriptedEngine())) as client:
        too_long = " ".join(["x"] * 9)
        assert client.post("/generate/stream", json={"prompt": too_long, "max_new_tokens": 1}).status_code == 413


def test_stream_delivers_every_token_then_a_terminal_event() -> None:
    with TestClient(_app(_ScriptedEngine())) as client:
        with client.stream("POST", "/generate/stream", json={"prompt": "a b", "max_new_tokens": 3}) as response:
            assert response.status_code == 200
            events = [line for line in response.iter_lines() if line.startswith("data: ")]
    assert len(events) == 4
    assert '"finish_reason": "LENGTH"' in events[-1]


def test_client_disconnect_mid_stream_cancels_the_request() -> None:
    """Starlette's TestClient buffers streams, so drive the ASGI app directly and send
    http.disconnect after the first token event, exactly as a vanished socket would."""
    import asyncio
    import json as _json

    engine = _ScriptedEngine(step_delay_s=0.05, max_model_len=None)
    app = _app(engine)
    with TestClient(app) as client:  # runs the lifespan so the service exists
        async def drive() -> list[bytes]:
            body = _json.dumps({"prompt": "a b", "max_new_tokens": 200}).encode()
            scope = {
                "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                "method": "POST", "scheme": "http", "path": "/generate/stream",
                "raw_path": b"/generate/stream", "query_string": b"", "root_path": "",
                "headers": [(b"content-type", b"application/json"),
                            (b"content-length", str(len(body)).encode())],
                "client": ("testclient", 1), "server": ("testserver", 80),
            }
            sent_request = False
            first_token = asyncio.Event()
            chunks: list[bytes] = []

            async def receive():
                nonlocal sent_request
                if not sent_request:
                    sent_request = True
                    return {"type": "http.request", "body": body, "more_body": False}
                await first_token.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.body" and message.get("body"):
                    chunks.append(message["body"])
                    first_token.set()

            await app(scope, receive, send)
            return chunks

        chunks = asyncio.run(drive())
        assert chunks and chunks[0].startswith(b"data: ")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not engine.cancelled:
            time.sleep(0.02)
        assert engine.cancelled and engine.cancelled[0][1] == "CLIENT_DISCONNECTED"
        assert not engine.has_unfinished_requests
        assert client.get("/health").status_code == 200


def test_deadline_exceeded_is_504_and_cancels() -> None:
    engine = _ScriptedEngine(step_delay_s=0.05, max_model_len=None)
    with TestClient(_app(engine, request_timeout_s=0.2)) as client:
        response = client.post("/generate", json={"prompt": "a b", "max_new_tokens": 500})
        assert response.status_code == 504
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not engine.cancelled:
            time.sleep(0.02)
        assert engine.cancelled[0][1] == "TIMEOUT"


def test_engine_level_failure_marks_worker_dead_and_fails_open_requests() -> None:
    engine = _ScriptedEngine(explode=True)
    with TestClient(_app(engine)) as client:
        response = client.post("/generate", json={"prompt": "a b", "max_new_tokens": 2})
        assert response.status_code == 500
        assert "simulated CUDA failure" in response.json()["error"]
        health = client.get("/health")
        assert health.status_code == 503 and health.json()["worker_failed"] is True
        assert client.get("/ready").status_code == 503
        assert client.post("/generate", json={"prompt": "a b", "max_new_tokens": 2}).status_code == 503


def test_shutdown_drains_in_flight_requests_and_completes_stragglers() -> None:
    engine = _ScriptedEngine(step_delay_s=0.02, max_model_len=None)
    app = _app(engine, drain_timeout_s=0.3)
    client = TestClient(app)
    client.__enter__()
    import threading

    results: dict[str, int] = {}

    def _long_request():
        results["status"] = client.post(
            "/generate", json={"prompt": "a b", "max_new_tokens": 4000}
        ).status_code

    thread = threading.Thread(target=_long_request)
    thread.start()
    time.sleep(0.15)
    client.__exit__(None, None, None)  # lifespan shutdown -> drain -> cancel stragglers
    thread.join(timeout=5)
    assert results["status"] == 503
    assert engine.cancelled and engine.cancelled[0][1] == "SERVER_SHUTDOWN"


@pytest.mark.parametrize("path", ["/health", "/ready"])
def test_probes_report_unavailable_before_startup(path: str) -> None:
    app = _app(_ScriptedEngine())
    client = TestClient(app)  # no lifespan: service is None
    assert client.get(path).status_code == 503
