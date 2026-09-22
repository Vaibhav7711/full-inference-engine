"""The OpenAI-compatible surface, against a scripted engine on CPU.

What is worth testing here is the contract, not generation: the response shapes clients
parse, that sampling fields actually reach the engine as `SamplingParams`, that stop
strings cut the text and release the request, that unsupported parameters are refused
rather than ignored, and that streaming ends the way SDKs expect.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from engine.runtime import RequestState
from engine.server.api import create_app
from engine.server.openai import truncate_at_stop


class _FakeTokenizer:
    """Token id == word length; decoding joins tokens as words, so stop strings work."""

    chat_template = "{{ messages }}"

    def __call__(self, prompt: str, return_tensors: str = "pt"):
        ids = [max(1, len(word)) for word in prompt.split()]

        class _Row(list):
            def tolist(self):
                return list(self)

        class _Batch:
            input_ids = [_Row(ids)]

        return _Batch()

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(_WORDS[i % len(_WORDS)] for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(f"{m['role']} {m['content']}" for m in messages) + " assistant"


_WORDS = ["alpha ", "beta ", "gamma ", "STOPHERE ", "delta "]


class _ScriptedEngine:
    """Emits a fixed token sequence, and records the sampling params it was given."""

    def __init__(self, tokens=(0, 1, 2, 3, 4), max_model_len: int = 4096,
                 stop_at_end: bool = True, step_delay_s: float = 0.0):
        self.tokenizer = _FakeTokenizer()
        self.max_model_len = max_model_len
        self.tokens = list(tokens)
        # When False the sequence cycles until max_tokens, so a test can observe a
        # request that is still generating.
        self.stop_at_end = stop_at_end
        self.step_delay_s = step_delay_s
        self.requests: list = []
        self.seen_sampling: list = []
        self.cancelled: list[tuple[str, str]] = []

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self.requests)

    def submit(self, request) -> bool:
        self.seen_sampling.append(request.sampling)
        request.transition(RequestState.PREFILLING)
        self.requests.append(request)
        return True

    def step(self) -> None:
        if self.step_delay_s:
            time.sleep(self.step_delay_s)
        for request in list(self.requests):
            if request.state is RequestState.PREFILLING:
                request.advance_prefill(request.remaining_prefill_tokens)
                request.transition(RequestState.DECODING)
            index = len(request.output_token_ids)
            request.append_token(self.tokens[index % len(self.tokens)])
            if (len(request.output_token_ids) >= request.max_new_tokens
                    or (self.stop_at_end and index + 1 >= len(self.tokens))):
                reason = "LENGTH" if len(request.output_token_ids) >= request.max_new_tokens else "EOS"
                request.transition(RequestState.FINISHED, reason=reason)
                self.requests.remove(request)

    def stats_snapshot(self) -> dict:
        return {"waiting_requests": 0, "active_requests": len(self.requests),
                "decode_batch": len(self.requests), "kv_utilization": 0.25}

    def cancel(self, request_id: str, reason: str = "CANCELLED"):
        for request in list(self.requests):
            if request.request_id == request_id:
                request.transition(RequestState.CANCELLED, reason=reason)
                self.requests.remove(request)
                self.cancelled.append((request_id, reason))
                return request
        raise KeyError(request_id)


def _client(engine: _ScriptedEngine | None = None) -> tuple[TestClient, _ScriptedEngine]:
    engine = engine or _ScriptedEngine()
    app = create_app(engine_factory=lambda: engine, model_name="test-model",
                     request_timeout_s=5.0, drain_timeout_s=1.0)
    return TestClient(app), engine


def test_models_endpoint_lists_the_served_model() -> None:
    client, _ = _client()
    with client:
        body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "test-model"


def test_completion_returns_the_openai_shape_with_usage() -> None:
    client, _ = _client()
    with client:
        body = client.post("/v1/completions", json={"prompt": "hi there", "max_tokens": 5}).json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    choice = body["choices"][0]
    assert choice["index"] == 0 and choice["finish_reason"] in {"stop", "length"}
    assert choice["text"]
    usage = body["usage"]
    assert usage["prompt_tokens"] == 2
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completion_uses_the_template_and_returns_a_message() -> None:
    client, engine = _client()
    with client:
        body = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello world"}], "max_tokens": 4,
        }).json()
    assert body["object"] == "chat.completion"
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant" and message["content"]


def test_sampling_fields_reach_the_engine() -> None:
    client, engine = _client()
    with client:
        client.post("/v1/completions", json={
            "prompt": "hi", "max_tokens": 2, "temperature": 0.7, "top_p": 0.9,
            "top_k": 40, "presence_penalty": 0.5, "frequency_penalty": 0.25,
            "repetition_penalty": 1.1, "seed": 1234, "stop_token_ids": [9],
        })
    sampling = engine.seen_sampling[-1]
    assert sampling.temperature == 0.7 and sampling.top_p == 0.9 and sampling.top_k == 40
    assert sampling.presence_penalty == 0.5 and sampling.frequency_penalty == 0.25
    assert sampling.repetition_penalty == 1.1 and sampling.seed == 1234
    assert sampling.stop_token_ids == frozenset({9})
    assert not sampling.greedy


def test_default_request_is_greedy() -> None:
    client, engine = _client()
    with client:
        client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
    assert engine.seen_sampling[-1].greedy


def test_stop_string_truncates_the_completed_text() -> None:
    client, _ = _client()
    with client:
        body = client.post("/v1/completions", json={
            "prompt": "hi", "max_tokens": 5, "stop": "STOPHERE",
        }).json()
    choice = body["choices"][0]
    assert "STOPHERE" not in choice["text"]
    assert choice["text"].startswith("alpha")
    assert choice["finish_reason"] == "stop"


def test_stop_string_cancels_a_request_that_is_still_generating() -> None:
    # Cycles forever at ~20 ms per token, so the stop string is reached long before the
    # 50-token bound and the handler must release the request rather than wait it out.
    engine = _ScriptedEngine(stop_at_end=False, step_delay_s=0.02)
    client, engine = _client(engine)
    with client:
        body = client.post("/v1/completions", json={
            "prompt": "hi", "max_tokens": 50, "stop": "STOPHERE",
        }).json()
    assert "STOPHERE" not in body["choices"][0]["text"]
    assert body["choices"][0]["finish_reason"] == "stop"
    assert engine.cancelled and engine.cancelled[0][1] == "STOP_SEQUENCE"
    assert len(body["choices"][0]["text"]) < 200, "generation ran on after the stop string"


def test_truncate_at_stop_takes_the_earliest_match() -> None:
    assert truncate_at_stop("abcXdefY", ["Y", "X"]) == ("abc", True)
    assert truncate_at_stop("abc", ["Y"]) == ("abc", False)
    assert truncate_at_stop("abc", [""]) == ("abc", False)


def test_streaming_completion_emits_deltas_then_done() -> None:
    client, _ = _client()
    with client:
        with client.stream("POST", "/v1/completions", json={
            "prompt": "hi", "max_tokens": 5, "stream": True,
            "stream_options": {"include_usage": True},
        }) as response:
            payloads = [line[len("data: "):] for line in response.iter_lines()
                        if line.startswith("data: ")]
    assert payloads[-1] == "[DONE]"
    chunks = [json.loads(item) for item in payloads[:-1]]
    assert all(chunk["object"] == "text_completion" for chunk in chunks)
    assert "".join(chunk["choices"][0]["text"] for chunk in chunks if chunk["choices"])
    finishes = [chunk["choices"][0]["finish_reason"] for chunk in chunks if chunk["choices"]]
    assert finishes[-1] in {"stop", "length"}
    assert chunks[-1]["usage"]["completion_tokens"] > 0


def test_streaming_chat_sends_the_role_once_then_content() -> None:
    client, _ = _client()
    with client:
        with client.stream("POST", "/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "stream": True,
        }) as response:
            chunks = [json.loads(line[len("data: "):]) for line in response.iter_lines()
                      if line.startswith("data: ") and not line.endswith("[DONE]")]
    assert chunks[0]["object"] == "chat.completion.chunk"
    roles = [chunk["choices"][0]["delta"].get("role") for chunk in chunks]
    assert roles.count("assistant") == 1
    assert chunks[-1]["choices"][0]["finish_reason"] in {"stop", "length"}


@pytest.mark.parametrize("payload,detail", [
    ({"prompt": "hi", "n": 2}, "n > 1"),
    ({"prompt": "hi", "echo": True}, "echo"),
    ({"prompt": "hi", "best_of": 3}, "best_of"),
    ({"prompt": "hi", "logit_bias": {"1": 2.0}}, "logit_bias"),
])
def test_unsupported_parameters_are_refused_not_ignored(payload, detail) -> None:
    client, _ = _client()
    with client:
        response = client.post("/v1/completions", json=payload)
    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_prompt_over_the_context_is_rejected() -> None:
    client, _ = _client(_ScriptedEngine(max_model_len=4))
    with client:
        response = client.post("/v1/completions", json={"prompt": "a b c", "max_tokens": 64})
    assert response.status_code == 400
    assert "model context" in response.json()["detail"]


def test_invalid_sampling_values_are_rejected_by_validation() -> None:
    client, _ = _client()
    with client:
        assert client.post("/v1/completions", json={"prompt": "hi", "temperature": 5}).status_code == 422
        assert client.post("/v1/completions", json={"prompt": "hi", "top_p": 0}).status_code == 422


def test_metrics_endpoint_reports_after_requests() -> None:
    client, _ = _client()
    with client:
        client.post("/v1/completions", json={"prompt": "hi there", "max_tokens": 3})
        text = client.get("/metrics").text
    assert "inference_requests_total" in text
    assert "inference_generation_tokens_total" in text
    assert "inference_time_to_first_token_seconds_bucket" in text
    assert "# TYPE inference_e2e_request_latency_seconds histogram" in text
