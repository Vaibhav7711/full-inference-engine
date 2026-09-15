"""Stage 6 minimal HTTP/SSE serving surface for the reference runtime."""

from __future__ import annotations

import json
import asyncio
from contextlib import asynccontextmanager
from time import monotonic
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.batching.continuous_batching import ContinuousBatchingEngine
from engine.model import load_model
from engine.server.continuous import ContinuousBatchingService, RequestHandle


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=4096)


def create_app(
    model_name: str = "Qwen/Qwen3-0.6B", *, max_active: int = 16,
    num_blocks: int = 1024, graph_buckets: tuple[int, ...] = (2, 4, 8, 16),
    max_pending_requests: int = 256, max_prompt_tokens: int = 4096,
    request_timeout_s: float = 120.0,
) -> FastAPI:
    if min(max_active, num_blocks, max_pending_requests, max_prompt_tokens) <= 0:
        raise ValueError("server capacity limits must be positive")
    if request_timeout_s <= 0:
        raise ValueError("request_timeout_s must be positive")
    service: ContinuousBatchingService | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal service
        loaded = load_model(model_name)
        buckets = tuple(size for size in graph_buckets if size <= max_active)
        engine = ContinuousBatchingEngine(
            loaded.model, loaded.tokenizer, loaded.device, max_active=max_active,
            num_blocks=num_blocks, cuda_graph_batch_sizes=buckets or None,
            max_waiting_requests=max_pending_requests,
        )
        service = ContinuousBatchingService(
            engine, max_pending_submissions=max_pending_requests,
        )
        service.start()
        yield
        service.stop()
        service = None

    app = FastAPI(title="full-inference-engine", lifespan=lifespan)

    def require_service() -> ContinuousBatchingService:
        if service is None:
            raise HTTPException(status_code=503, detail="model is not ready")
        return service

    def submit(payload: GenerateRequest) -> tuple[ContinuousBatchingService, RequestHandle]:
        current = require_service()
        token_ids = current.engine.tokenizer(payload.prompt, return_tensors="pt").input_ids[0].tolist()
        if len(token_ids) > max_prompt_tokens:
            raise HTTPException(
                status_code=413,
                detail=f"prompt exceeds the {max_prompt_tokens}-token server limit",
            )
        try:
            return current, current.submit(token_ids, payload.max_new_tokens)
        except RuntimeError as error:
            raise HTTPException(status_code=429, detail=str(error)) from error

    @staticmethod
    def metrics(handle: RequestHandle) -> dict[str, float | None]:
        request = handle.request
        return {
            "queue_ms": request.queue_time_ms(), "ttft_ms": request.time_to_first_token_ms(),
            "generation_ms": request.generation_time_ms(), "output_tokens": len(request.output_token_ids),
        }

    @app.post("/generate")
    async def generate(payload: GenerateRequest) -> dict[str, object]:
        service, handle = submit(payload)
        completed = await asyncio.to_thread(handle.completed.wait, request_timeout_s)
        if not completed:
            service.cancel(handle, reason="TIMEOUT")
            raise HTTPException(status_code=504, detail="generation deadline exceeded")
        if handle.error is not None:
            raise HTTPException(status_code=500, detail=str(handle.error))
        tokens = handle.request.output_token_ids
        return {"text": service.engine.tokenizer.decode(tokens, skip_special_tokens=True),
                "token_ids": tokens, "metrics": metrics(handle)}

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ready", **require_service().snapshot()}

    @app.post("/generate/stream")
    async def generate_stream(payload: GenerateRequest, request: Request) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            service, handle = submit(payload)
            sent = 0
            started = monotonic()
            try:
                while not handle.completed.is_set():
                    if await request.is_disconnected():
                        return
                    if monotonic() - started >= request_timeout_s:
                        service.cancel(handle, reason="TIMEOUT")
                        yield f"data: {json.dumps({'error': 'generation deadline exceeded', 'finish_reason': 'TIMEOUT'})}\n\n"
                        return
                    tokens = handle.request.output_token_ids
                    while sent < len(tokens):
                        token_id = tokens[sent]
                        data = {"token_id": token_id,
                                "text": service.engine.tokenizer.decode([token_id], skip_special_tokens=True),
                                "index": sent, "finish_reason": None}
                        sent += 1
                        yield f"data: {json.dumps(data)}\n\n"
                    await asyncio.sleep(0.005)
                if handle.error is not None:
                    yield f"data: {json.dumps({'error': str(handle.error)})}\n\n"
                    return
                tokens = handle.request.output_token_ids
                while sent < len(tokens):
                    token_id = tokens[sent]
                    data = {"token_id": token_id,
                            "text": service.engine.tokenizer.decode([token_id], skip_special_tokens=True),
                            "index": sent, "finish_reason": None}
                    sent += 1
                    yield f"data: {json.dumps(data)}\n\n"
                yield f"data: {json.dumps({'token_id': None, 'text': '', 'index': None, 'finish_reason': handle.request.finish_reason})}\n\n"
            finally:
                # Starlette cancels the generator when a socket disappears, which may
                # bypass the explicit is_disconnected branch. Always reclaim unfinished
                # scheduler/KV state through the worker-owned cancellation queue.
                if not handle.completed.is_set():
                    service.cancel(handle)
        return StreamingResponse(events(), media_type="text/event-stream")

    return app
