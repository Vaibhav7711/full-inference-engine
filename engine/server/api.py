"""HTTP/SSE serving surface over the continuous-batching service.

Status mapping (request-level outcomes never take the worker down):
    400  prompt tokenizes to nothing, or prompt + max_new_tokens exceeds the model context
    413  prompt exceeds the configured token limit, or can never fit the KV pool
    429  submission queue or scheduler queue is full
    499  client disconnected (no body is sent)
    503  engine unavailable, draining, or a request was dropped by the pool/shutdown
    504  per-request deadline exceeded
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field


_DEMO_UI = Path(__file__).with_name("static") / "index.html"

from engine.runtime import RequestState
from engine.server.continuous import (
    ContinuousBatchingService, RequestHandle, ServerShutdown, SubmitError,
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=4096)


_TERMINAL_STATUS: dict[tuple[str, str | None], int] = {
    ("REJECTED", "QUEUE_FULL"): 429,
    ("REJECTED", "KV_CAPACITY_EXCEEDED"): 413,
    ("FAILED", "KV_POOL_EXHAUSTED"): 503,
    ("CANCELLED", "TIMEOUT"): 504,
    ("CANCELLED", "SERVER_SHUTDOWN"): 503,
    ("CANCELLED", "CLIENT_DISCONNECTED"): 499,
}


def terminal_status(handle: RequestHandle) -> int | None:
    """HTTP status for a handle, or None when it is in flight or finished normally."""
    request = handle.request
    if request.done:
        if request.state is RequestState.FINISHED:
            return None
        key = (request.state.name, request.finish_reason)
        if key in _TERMINAL_STATUS:
            return _TERMINAL_STATUS[key]
        if request.state is RequestState.REJECTED:
            return 429
        if request.state is RequestState.CANCELLED:
            return 499
        return 500
    if handle.error is not None:
        # The worker abandoned the request: shutdown (retry elsewhere) or engine failure.
        return 503 if isinstance(handle.error, ServerShutdown) else 500
    return None


def default_engine_factory(
    model_name: str, *, max_active: int, num_blocks: int,
    graph_buckets: tuple[int, ...], max_pending_requests: int,
):
    """Load the checkpoint and build the GPU engine. Imported lazily so the API module
    stays importable (and unit-testable) without CUDA, Triton, or a checkpoint."""
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(model_name)
    buckets = tuple(size for size in graph_buckets if size <= max_active)
    return ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, loaded.device, max_active=max_active,
        num_blocks=num_blocks, cuda_graph_batch_sizes=buckets or None,
        max_waiting_requests=max_pending_requests,
    )


def create_app(
    model_name: str = "Qwen/Qwen3-0.6B", *, max_active: int = 16,
    num_blocks: int = 1024, graph_buckets: tuple[int, ...] = (2, 4, 8, 16),
    max_pending_requests: int = 256, max_prompt_tokens: int = 4096,
    request_timeout_s: float = 120.0, drain_timeout_s: float = 30.0,
    engine_factory: Callable[[], object] | None = None,
) -> FastAPI:
    if min(max_active, num_blocks, max_pending_requests, max_prompt_tokens) <= 0:
        raise ValueError("server capacity limits must be positive")
    if request_timeout_s <= 0 or drain_timeout_s < 0:
        raise ValueError("request_timeout_s must be positive and drain_timeout_s non-negative")
    service: ContinuousBatchingService | None = None

    def build_engine():
        if engine_factory is not None:
            return engine_factory()
        return default_engine_factory(
            model_name, max_active=max_active, num_blocks=num_blocks,
            graph_buckets=graph_buckets, max_pending_requests=max_pending_requests,
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal service
        engine = await asyncio.to_thread(build_engine)
        service = ContinuousBatchingService(engine, max_pending_submissions=max_pending_requests)
        service.start()
        try:
            yield
        finally:
            # SIGTERM arrives here via uvicorn's lifespan shutdown: stop admitting,
            # let in-flight generations finish, cancel stragglers, complete every handle.
            await asyncio.to_thread(service.stop, drain_timeout_s=drain_timeout_s)
            service = None

    app = FastAPI(title="full-inference-engine", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def demo_ui() -> FileResponse:
        return FileResponse(_DEMO_UI)

    def require_service() -> ContinuousBatchingService:
        if service is None:
            raise HTTPException(status_code=503, detail="model is not ready")
        return service

    async def tokenize(current: ContinuousBatchingService, prompt: str) -> list[int]:
        tokenizer = current.engine.tokenizer
        # Tokenization is CPU work; keep it off the event loop under load.
        return await asyncio.to_thread(
            lambda: tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        )

    async def submit(payload: GenerateRequest) -> tuple[ContinuousBatchingService, RequestHandle]:
        current = require_service()
        if not current.ready:
            raise HTTPException(status_code=503, detail="server is not accepting requests")
        token_ids = await tokenize(current, payload.prompt)
        if len(token_ids) > max_prompt_tokens:
            raise HTTPException(
                status_code=413,
                detail=f"prompt exceeds the {max_prompt_tokens}-token server limit",
            )
        max_model_len = getattr(current.engine, "max_model_len", None)
        if max_model_len is not None and len(token_ids) + payload.max_new_tokens > max_model_len:
            raise HTTPException(
                status_code=400,
                detail=f"prompt plus max_new_tokens exceeds the model context of {max_model_len}",
            )
        try:
            return current, current.submit(token_ids, payload.max_new_tokens)
        except SubmitError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error

    async def _await_event(register, timeout_s: float) -> bool:
        """Await a handle event without occupying an executor thread per request."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()

        def _fire(_: RequestHandle) -> None:
            loop.call_soon_threadsafe(lambda: future.done() or future.set_result(None))

        register(_fire)
        try:
            await asyncio.wait_for(future, timeout_s)
        except asyncio.TimeoutError:
            return False
        return True

    async def wait_for(handle: RequestHandle, timeout_s: float) -> bool:
        return await _await_event(handle.on_complete, timeout_s)

    async def wait_for_admission(handle: RequestHandle, timeout_s: float) -> bool:
        return await _await_event(handle.on_accept, timeout_s)

    def metrics(handle: RequestHandle) -> dict[str, float | int | None]:
        """Per-request timing with queueing, stalls, and recompute cost separated."""
        request = handle.request
        report = dict(request.latency_report())
        report.update(request.recompute_overhead())
        return report

    def failure_response(handle: RequestHandle) -> JSONResponse | None:
        status = terminal_status(handle)
        if status is None:
            return None
        detail = handle.request.finish_reason or (
            str(handle.error) if handle.error is not None else "request failed"
        )
        return JSONResponse(
            status_code=status,
            content={"error": detail, "request_id": handle.request.request_id},
            headers={"x-request-id": handle.request.request_id},
        )

    @app.post("/generate")
    async def generate(payload: GenerateRequest):
        service_, handle = await submit(payload)
        completed = await wait_for(handle, request_timeout_s)
        if not completed:
            service_.cancel(handle, reason="TIMEOUT")
            return JSONResponse(
                status_code=504,
                content={"error": "generation deadline exceeded", "request_id": handle.request.request_id},
                headers={"x-request-id": handle.request.request_id},
            )
        failure = failure_response(handle)
        if failure is not None:
            return failure
        tokens = handle.request.output_token_ids
        return JSONResponse(
            content={
                "request_id": handle.request.request_id,
                "text": service_.engine.tokenizer.decode(tokens, skip_special_tokens=True),
                "token_ids": tokens, "finish_reason": handle.request.finish_reason,
                "metrics": metrics(handle),
            },
            headers={"x-request-id": handle.request.request_id},
        )

    @app.get("/health")
    async def health():
        """Liveness: the worker is running. A dead worker means restart the process."""
        current = service
        if current is None or not current.alive:
            detail = {"status": "unavailable"}
            if current is not None:
                detail.update(current.snapshot())
                if current.fatal_error is not None:
                    detail["error"] = str(current.fatal_error)
            return JSONResponse(status_code=503, content=detail)
        return {"status": "ok", **current.snapshot()}

    @app.get("/ready")
    async def ready():
        """Readiness: accepting requests (loaded, alive, not draining)."""
        current = service
        if current is None or not current.ready:
            content = {"status": "not_ready"}
            if current is not None:
                content.update(current.snapshot())
            return JSONResponse(status_code=503, content=content)
        return {"status": "ready", **current.snapshot()}

    @app.post("/generate/stream")
    async def generate_stream(payload: GenerateRequest, request: Request):
        # Submit and wait for the admission decision before the response starts, so
        # limit, queue, and capacity errors are real status codes rather than events
        # inside an already-open stream.
        service_, handle = await submit(payload)
        if not await wait_for_admission(handle, request_timeout_s):
            service_.cancel(handle, reason="TIMEOUT")
            raise HTTPException(status_code=504, detail="admission deadline exceeded")
        failure = failure_response(handle)
        if failure is not None:
            return failure
        tokenizer = service_.engine.tokenizer

        async def events() -> AsyncIterator[str]:
            sent = 0
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                while not handle.completed.is_set():
                    if await request.is_disconnected():
                        return
                    if loop.time() - started >= request_timeout_s:
                        service_.cancel(handle, reason="TIMEOUT")
                        yield f"data: {json.dumps({'error': 'generation deadline exceeded', 'finish_reason': 'TIMEOUT'})}\n\n"
                        return
                    tokens = handle.request.output_token_ids
                    while sent < len(tokens):
                        token_id = tokens[sent]
                        data = {"token_id": token_id,
                                "text": tokenizer.decode([token_id], skip_special_tokens=True),
                                "index": sent, "finish_reason": None}
                        sent += 1
                        yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0.005)
                status = terminal_status(handle)
                if status is not None:
                    detail = handle.request.finish_reason or str(handle.error)
                    yield f"data: {json.dumps({'error': detail, 'status': status, 'finish_reason': handle.request.finish_reason})}\n\n"
                    return
                tokens = handle.request.output_token_ids
                while sent < len(tokens):
                    token_id = tokens[sent]
                    data = {"token_id": token_id,
                            "text": tokenizer.decode([token_id], skip_special_tokens=True),
                            "index": sent, "finish_reason": None}
                    sent += 1
                    yield f"data: {json.dumps(data)}\n\n"
                yield f"data: {json.dumps({'token_id': None, 'text': '', 'index': None, 'finish_reason': handle.request.finish_reason})}\n\n"
            finally:
                # Starlette cancels the generator when a socket disappears, which may
                # bypass the explicit is_disconnected branch. Always reclaim unfinished
                # scheduler/KV state through the worker-owned cancellation queue.
                if not handle.completed.is_set():
                    service_.cancel(handle)

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"x-request-id": handle.request.request_id},
        )

    return app
