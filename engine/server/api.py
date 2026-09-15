"""Stage 6 minimal HTTP/SSE serving surface for the reference runtime."""

from __future__ import annotations

import json
import asyncio
from contextlib import asynccontextmanager
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
) -> FastAPI:
    service: ContinuousBatchingService | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal service
        loaded = load_model(model_name)
        buckets = tuple(size for size in graph_buckets if size <= max_active)
        engine = ContinuousBatchingEngine(
            loaded.model, loaded.tokenizer, loaded.device, max_active=max_active,
            num_blocks=num_blocks, cuda_graph_batch_sizes=buckets or None,
        )
        service = ContinuousBatchingService(engine)
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
        await asyncio.to_thread(handle.completed.wait)
        if handle.error is not None:
            raise HTTPException(status_code=500, detail=str(handle.error))
        tokens = handle.request.output_token_ids
        return {"text": service.engine.tokenizer.decode(tokens, skip_special_tokens=True),
                "token_ids": tokens, "metrics": metrics(handle)}

    @app.post("/generate/stream")
    async def generate_stream(payload: GenerateRequest, request: Request) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            service, handle = submit(payload)
            sent = 0
            while not handle.completed.is_set():
                if await request.is_disconnected():
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
        return StreamingResponse(events(), media_type="text/event-stream")

    return app
