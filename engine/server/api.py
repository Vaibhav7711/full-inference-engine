"""Stage 6 minimal HTTP/SSE serving surface for the reference runtime."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from engine.model import ExplicitDecodeRunner, load_model


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(default=32, ge=1, le=4096)


def create_app(model_name: str = "Qwen/Qwen3-0.6B") -> FastAPI:
    runner: ExplicitDecodeRunner | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        nonlocal runner
        loaded = load_model(model_name)
        runner = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
        yield
        runner = None

    app = FastAPI(title="full-inference-engine", lifespan=lifespan)

    def require_runner() -> ExplicitDecodeRunner:
        if runner is None:
            raise HTTPException(status_code=503, detail="model is not ready")
        return runner

    @app.post("/generate")
    async def generate(payload: GenerateRequest) -> dict[str, object]:
        result = require_runner().generate(payload.prompt, max_new_tokens=payload.max_new_tokens)
        return {"text": result.text, "token_ids": result.token_ids, "metrics": result.metrics.as_dict()}

    @app.post("/generate/stream")
    async def generate_stream(payload: GenerateRequest, request: Request) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            for event in require_runner().stream_generate(payload.prompt, max_new_tokens=payload.max_new_tokens):
                if await request.is_disconnected():
                    return
                data = {"token_id": event.token_id, "text": event.text, "index": event.index, "finish_reason": event.finish_reason}
                yield f"data: {json.dumps(data)}\n\n"
        return StreamingResponse(events(), media_type="text/event-stream")

    return app
