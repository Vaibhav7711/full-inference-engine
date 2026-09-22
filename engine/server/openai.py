"""OpenAI-compatible routes over the continuous-batching service.

Clients, evaluation harnesses and SDKs speak the OpenAI wire format; an engine that only
answers `POST /generate` cannot be pointed at by any of them. These routes are a
translation layer and nothing more: request fields become `SamplingParams`, the engine
decides tokens, and the response is assembled in the shape the format requires. No
generation logic lives here.

Supported: `/v1/models`, `/v1/completions`, `/v1/chat/completions`, streaming for both,
`n = 1`, `logprobs`, `stop` strings and token ids, `usage` (including
`stream_options.include_usage`), `seed`, and the standard sampling knobs.

Deliberately unsupported, and rejected rather than silently ignored: `n > 1`, `echo`,
`suffix`, `best_of`, `logit_bias`, tools and structured outputs. A serving surface that
accepts a parameter it does not honour is worse than one that refuses it.
"""

from __future__ import annotations

import json
import time
from typing import AsyncIterator, Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine.runtime import SamplingParams
from engine.server.continuous import ContinuousBatchingService, RequestHandle, SubmitError


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class _SamplingFields(BaseModel):
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=2.0)
    seed: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False
    stream: bool = False
    stream_options: dict | None = None
    model: str | None = None
    # Refused rather than ignored (see the module docstring).
    n: int = 1
    echo: bool = False
    best_of: int | None = None
    logit_bias: dict | None = None


class CompletionRequest(_SamplingFields):
    prompt: str | list[str]
    logprobs: int | None = Field(default=None, ge=0, le=20)


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage] = Field(min_length=1)
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)


def _unsupported(payload: _SamplingFields) -> str | None:
    if payload.n != 1:
        return "n > 1 is not supported; issue separate requests"
    if payload.echo:
        return "echo is not supported"
    if payload.best_of not in (None, 1):
        return "best_of is not supported"
    if payload.logit_bias:
        return "logit_bias is not supported"
    return None


def sampling_from(payload: _SamplingFields, logprobs: int | None) -> SamplingParams:
    """Translate the request's sampling fields, with OpenAI's defaults and conventions."""
    return SamplingParams(
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        min_p=payload.min_p,
        presence_penalty=payload.presence_penalty,
        frequency_penalty=payload.frequency_penalty,
        repetition_penalty=payload.repetition_penalty,
        seed=payload.seed,
        stop_token_ids=frozenset(payload.stop_token_ids or ()),
        ignore_eos=payload.ignore_eos,
        logprobs=logprobs,
    )


def stop_strings(payload: _SamplingFields) -> list[str]:
    if payload.stop is None:
        return []
    return [payload.stop] if isinstance(payload.stop, str) else list(payload.stop)


def _finish_reason(handle: RequestHandle) -> str:
    """Map the engine's terminal reason onto the four OpenAI values."""
    reason = handle.request.finish_reason
    if reason == "LENGTH":
        return "length"
    return "stop"


def truncate_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    """Cut `text` at the earliest stop string. The stop string itself is not returned,
    matching the OpenAI convention."""
    cut = None
    for stop in stops:
        if not stop:
            continue
        index = text.find(stop)
        if index >= 0 and (cut is None or index < cut):
            cut = index
    if cut is None:
        return text, False
    return text[:cut], True


def install_openai_routes(
    app: FastAPI,
    *,
    require_service: Callable[[], ContinuousBatchingService],
    model_name: str,
    max_prompt_tokens: int,
    request_timeout_s: float,
    default_max_tokens: int = 128,
) -> None:
    """Register the `/v1` surface on `app`, reusing the service the native routes use."""

    async def _tokenize(service: ContinuousBatchingService, text: str) -> list[int]:
        import asyncio
        tokenizer = service.engine.tokenizer
        return await asyncio.to_thread(
            lambda: tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        )

    def _chat_prompt(service: ContinuousBatchingService, messages: list[ChatMessage]) -> str:
        """Render messages with the model's own chat template when it has one.

        A model served without its template produces subtly wrong answers that look like
        a quality problem rather than a serving bug, so the fallback is explicit and
        plain rather than an imitation of some other model's format.
        """
        tokenizer = service.engine.tokenizer
        payload = [{"role": message.role, "content": message.content} for message in messages]
        template = getattr(tokenizer, "apply_chat_template", None)
        if template is not None and getattr(tokenizer, "chat_template", None):
            return template(payload, tokenize=False, add_generation_prompt=True)
        rendered = "\n".join(f"{item['role']}: {item['content']}" for item in payload)
        return rendered + "\nassistant:"

    async def _submit(
        payload: _SamplingFields, token_ids: list[int], sampling: SamplingParams,
    ) -> tuple[ContinuousBatchingService, RequestHandle, int]:
        service = require_service()
        if not service.ready:
            raise HTTPException(status_code=503, detail="server is not accepting requests")
        refusal = _unsupported(payload)
        if refusal is not None:
            raise HTTPException(status_code=400, detail=refusal)
        if not token_ids:
            raise HTTPException(status_code=400, detail="prompt tokenized to zero tokens")
        if len(token_ids) > max_prompt_tokens:
            raise HTTPException(
                status_code=413,
                detail=f"prompt exceeds the {max_prompt_tokens}-token server limit",
            )
        max_tokens = payload.max_tokens or default_max_tokens
        max_model_len = getattr(service.engine, "max_model_len", None)
        if max_model_len is not None and len(token_ids) + max_tokens > max_model_len:
            raise HTTPException(
                status_code=400,
                detail=f"prompt plus max_tokens exceeds the model context of {max_model_len}",
            )
        try:
            handle = service.submit(token_ids, max_tokens, sampling)
        except SubmitError as error:
            raise HTTPException(status_code=error.status_code, detail=str(error)) from error
        return service, handle, len(token_ids)

    def _usage(prompt_tokens: int, handle: RequestHandle) -> dict:
        generated = len(handle.request.output_token_ids)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": generated,
            "total_tokens": prompt_tokens + generated,
        }

    def _logprob_payload(handle: RequestHandle, tokenizer, count: int) -> dict | None:
        entries = handle.request.output_logprobs
        if not entries:
            return None
        tokens, token_logprobs, top = [], [], []
        for step, options in zip(handle.request.output_token_ids, entries):
            tokens.append(tokenizer.decode([step], skip_special_tokens=False))
            token_logprobs.append(options[0][1] if options else None)
            top.append({
                tokenizer.decode([token], skip_special_tokens=False): value
                for token, value in options[:count or len(options)]
            })
        return {"tokens": tokens, "token_logprobs": token_logprobs, "top_logprobs": top,
                "text_offset": []}

    async def _collect(
        service: ContinuousBatchingService, handle: RequestHandle, stops: list[str],
    ) -> tuple[str, str]:
        """Wait for the request, applying stop strings. Returns (text, finish_reason)."""
        import asyncio
        tokenizer = service.engine.tokenizer
        loop = asyncio.get_running_loop()
        started = loop.time()
        while not handle.completed.is_set():
            if loop.time() - started >= request_timeout_s:
                service.cancel(handle, reason="TIMEOUT")
                raise HTTPException(status_code=504, detail="generation deadline exceeded")
            if stops:
                text = tokenizer.decode(handle.request.output_token_ids, skip_special_tokens=True)
                trimmed, hit = truncate_at_stop(text, stops)
                if hit:
                    service.cancel(handle, reason="STOP_SEQUENCE")
                    return trimmed, "stop"
            await asyncio.sleep(0.005)
        if handle.error is not None and not handle.request.done:
            raise HTTPException(status_code=503, detail=str(handle.error))
        text = tokenizer.decode(handle.request.output_token_ids, skip_special_tokens=True)
        if stops:
            text, hit = truncate_at_stop(text, stops)
            if hit:
                return text, "stop"
        return text, _finish_reason(handle)

    async def _stream(
        service: ContinuousBatchingService, handle: RequestHandle, stops: list[str],
        *, chunk: Callable[[str, str | None], dict], request: Request,
        prompt_tokens: int, include_usage: bool,
    ) -> AsyncIterator[str]:
        import asyncio
        tokenizer = service.engine.tokenizer
        loop = asyncio.get_running_loop()
        started = loop.time()
        sent_text = ""
        finish = None
        try:
            while finish is None:
                if await request.is_disconnected():
                    service.cancel(handle)
                    return
                if loop.time() - started >= request_timeout_s:
                    service.cancel(handle, reason="TIMEOUT")
                    finish = "length"
                    break
                done = handle.completed.is_set()
                text = tokenizer.decode(handle.request.output_token_ids, skip_special_tokens=True)
                trimmed, hit = truncate_at_stop(text, stops) if stops else (text, False)
                if len(trimmed) > len(sent_text):
                    delta = trimmed[len(sent_text):]
                    sent_text = trimmed
                    yield f"data: {json.dumps(chunk(delta, None))}\n\n"
                if hit:
                    service.cancel(handle, reason="STOP_SEQUENCE")
                    finish = "stop"
                    break
                if done:
                    finish = _finish_reason(handle)
                    break
                await asyncio.sleep(0.005)
            yield f"data: {json.dumps(chunk('', finish))}\n\n"
            if include_usage:
                payload = chunk("", None)
                payload["choices"] = []
                payload["usage"] = _usage(prompt_tokens, handle)
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Starlette can cancel this generator without running the disconnect branch.
            if not handle.completed.is_set():
                service.cancel(handle)

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "created": int(time.time()),
                      "owned_by": "local"}],
        }

    @app.post("/v1/completions")
    async def completions(payload: CompletionRequest, request: Request):
        if isinstance(payload.prompt, list):
            if len(payload.prompt) != 1:
                raise HTTPException(status_code=400, detail="batched prompts are not supported")
            prompt = payload.prompt[0]
        else:
            prompt = payload.prompt
        service = require_service()
        token_ids = await _tokenize(service, prompt)
        sampling = sampling_from(payload, payload.logprobs)
        service, handle, prompt_tokens = await _submit(payload, token_ids, sampling)
        stops = stop_strings(payload)
        created, completion_id = int(time.time()), f"cmpl-{uuid4().hex}"

        if payload.stream:
            def chunk(delta: str, finish: str | None) -> dict:
                return {"id": completion_id, "object": "text_completion", "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "text": delta, "finish_reason": finish,
                                     "logprobs": None}]}
            include_usage = bool((payload.stream_options or {}).get("include_usage"))
            return StreamingResponse(
                _stream(service, handle, stops, chunk=chunk, request=request,
                        prompt_tokens=prompt_tokens, include_usage=include_usage),
                media_type="text/event-stream",
                headers={"x-request-id": handle.request.request_id},
            )

        text, finish = await _collect(service, handle, stops)
        logprobs = _logprob_payload(handle, service.engine.tokenizer, payload.logprobs or 0)
        return JSONResponse(
            content={
                "id": completion_id, "object": "text_completion", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": finish,
                             "logprobs": logprobs}],
                "usage": _usage(prompt_tokens, handle),
            },
            headers={"x-request-id": handle.request.request_id},
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request):
        service = require_service()
        prompt = _chat_prompt(service, payload.messages)
        token_ids = await _tokenize(service, prompt)
        wanted = payload.top_logprobs if payload.logprobs else None
        sampling = sampling_from(payload, wanted)
        service, handle, prompt_tokens = await _submit(payload, token_ids, sampling)
        stops = stop_strings(payload)
        created, completion_id = int(time.time()), f"chatcmpl-{uuid4().hex}"

        if payload.stream:
            first = True

            def chunk(delta: str, finish: str | None) -> dict:
                nonlocal first
                body: dict = {"content": delta} if delta else {}
                if first and finish is None:
                    body["role"] = "assistant"
                    first = False
                return {"id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_name,
                        "choices": [{"index": 0, "delta": body, "finish_reason": finish}]}
            include_usage = bool((payload.stream_options or {}).get("include_usage"))
            return StreamingResponse(
                _stream(service, handle, stops, chunk=chunk, request=request,
                        prompt_tokens=prompt_tokens, include_usage=include_usage),
                media_type="text/event-stream",
                headers={"x-request-id": handle.request.request_id},
            )

        text, finish = await _collect(service, handle, stops)
        return JSONResponse(
            content={
                "id": completion_id, "object": "chat.completion", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": {"role": "assistant", "content": text}}],
                "usage": _usage(prompt_tokens, handle),
            },
            headers={"x-request-id": handle.request.request_id},
        )
