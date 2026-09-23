"""Drive a running server the way a client would, and check what it reports about itself.

The benchmarks measure the engine; this measures the *service*: that an OpenAI client
gets the shapes it expects, that streaming delivers tokens incrementally rather than in
one lump at the end, that stop strings and seeds behave over HTTP, that concurrent
callers are served together rather than queued behind each other, and that `/metrics`
moves accordingly.

Start the server first, in its own process:

    uvicorn engine.server.api:create_app --factory --port 8000

then:

    python scripts/live_smoke.py --base-url http://127.0.0.1:8000 --out results/<run>/live_smoke.json

Only `requests` is needed (any HTTP client would do; the point is to be an outside
observer of the process, not to import the engine).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests


def _fail(message: str):
    raise AssertionError(message)


def check_ready(base: str) -> dict:
    ready = requests.get(f"{base}/ready", timeout=120)
    if ready.status_code != 200:
        _fail(f"/ready returned {ready.status_code}: {ready.text[:200]}")
    models = requests.get(f"{base}/v1/models", timeout=30).json()
    return {"ready": ready.json(), "model": models["data"][0]["id"]}


def check_completion(base: str, model: str) -> dict:
    started = time.perf_counter()
    response = requests.post(f"{base}/v1/completions", timeout=300, json={
        "model": model, "prompt": "In one sentence, what is a KV cache?",
        "max_tokens": 48, "temperature": 0.0,
    })
    elapsed = time.perf_counter() - started
    body = response.json()
    if response.status_code != 200:
        _fail(f"completion failed: {response.status_code} {body}")
    choice = body["choices"][0]
    usage = body["usage"]
    if not choice["text"].strip():
        _fail("completion returned empty text")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        _fail(f"usage does not add up: {usage}")
    if choice["finish_reason"] not in {"stop", "length"}:
        _fail(f"unexpected finish_reason {choice['finish_reason']!r}")
    return {"seconds": elapsed, "usage": usage, "finish_reason": choice["finish_reason"],
            "text": choice["text"][:200]}


def check_chat_stream(base: str, model: str) -> dict:
    """Streaming must deliver tokens as they are produced, not in one final chunk."""
    started = time.perf_counter()
    arrivals, pieces, roles, finishes = [], [], 0, []
    with requests.post(f"{base}/v1/chat/completions", timeout=300, stream=True, json={
        "model": model, "stream": True, "max_tokens": 64, "temperature": 0.0,
        "messages": [{"role": "user", "content": "Count from one to twenty in words."}],
    }) as response:
        if response.status_code != 200:
            _fail(f"chat stream failed: {response.status_code} {response.text[:200]}")
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choice = chunk["choices"][0] if chunk["choices"] else {}
            delta = choice.get("delta", {})
            if delta.get("role"):
                roles += 1
            if delta.get("content"):
                arrivals.append(time.perf_counter() - started)
                pieces.append(delta["content"])
            if choice.get("finish_reason"):
                finishes.append(choice["finish_reason"])
    if len(pieces) < 3:
        _fail(f"stream delivered {len(pieces)} content chunks; it is not incremental")
    if roles != 1:
        _fail(f"the assistant role was sent {roles} times, expected once")
    if not finishes:
        _fail("the stream never sent a finish_reason")
    span = arrivals[-1] - arrivals[0]
    if span <= 0:
        _fail("every chunk arrived at the same instant; the stream is buffered")
    return {"chunks": len(pieces), "ttft_s": arrivals[0], "span_s": span,
            "finish_reason": finishes[-1], "text": "".join(pieces)[:200]}


def check_stop_and_seed(base: str, model: str) -> dict:
    stop = requests.post(f"{base}/v1/completions", timeout=300, json={
        "model": model, "prompt": "Recite the alphabet: a b c d e f g h", "max_tokens": 64,
        "temperature": 0.0, "stop": "f",
    }).json()["choices"][0]
    if "f" in stop["text"]:
        _fail(f"stop string was not applied: {stop['text']!r}")
    if stop["finish_reason"] != "stop":
        _fail(f"stop string did not set finish_reason=stop ({stop['finish_reason']})")

    body = {"model": model, "prompt": "Write a haiku about caching.", "max_tokens": 32,
            "temperature": 0.9, "top_p": 0.95, "seed": 1234}
    first = requests.post(f"{base}/v1/completions", timeout=300, json=body).json()
    second = requests.post(f"{base}/v1/completions", timeout=300, json=body).json()
    reproducible = first["choices"][0]["text"] == second["choices"][0]["text"]
    if not reproducible:
        _fail("the same seed produced different text")
    return {"stopped_text": stop["text"][:120], "seeded_text": first["choices"][0]["text"][:120]}


def check_refusals(base: str, model: str) -> dict:
    outcomes = {}
    for name, payload in {
        "n_gt_1": {"prompt": "hi", "n": 2},
        "echo": {"prompt": "hi", "echo": True},
        "bad_temperature": {"prompt": "hi", "temperature": 9.0},
    }.items():
        response = requests.post(f"{base}/v1/completions", timeout=60,
                                 json={"model": model, **payload})
        if response.status_code not in (400, 422):
            _fail(f"{name} returned {response.status_code}, expected a 4xx refusal")
        outcomes[name] = response.status_code
    return outcomes


def check_concurrency(base: str, model: str, concurrency: int, tokens: int) -> dict:
    """Continuous batching means N callers finish in about the time one would take,
    not N times it. Compare one request against `concurrency` of them at once."""
    prompt = ("Explain paged attention, continuous batching and chunked prefill, "
              "and how a scheduler admits requests under memory pressure.")

    def one(index: int) -> float:
        started = time.perf_counter()
        response = requests.post(f"{base}/v1/completions", timeout=600, json={
            "model": model, "prompt": f"{prompt} (request {index})",
            "max_tokens": tokens, "temperature": 0.0,
        })
        if response.status_code != 200:
            _fail(f"request {index} failed: {response.status_code} {response.text[:200]}")
        return time.perf_counter() - started

    alone = one(0)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        latencies = list(pool.map(one, range(concurrency)))
    wall = time.perf_counter() - started
    speedup = (alone * concurrency) / wall if wall else 0.0
    if speedup < 1.5:
        _fail(f"{concurrency} concurrent requests took {wall:.1f}s against {alone:.1f}s "
              f"for one; batching is not happening (speedup {speedup:.2f}x)")
    return {
        "single_s": alone, "concurrent_wall_s": wall, "concurrency": concurrency,
        "speedup_vs_serial": speedup,
        "latency_p50_s": statistics.median(latencies),
        "latency_max_s": max(latencies),
        "tokens_per_second": concurrency * tokens / wall if wall else 0.0,
    }


def check_metrics(base: str) -> dict:
    text = requests.get(f"{base}/metrics", timeout=30).text
    values: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        try:
            values[name] = float(value)
        except ValueError:
            continue
    required = ["inference_requests_total", "inference_generation_tokens_total",
                "inference_time_to_first_token_seconds_count",
                "inference_e2e_request_latency_seconds_count"]
    missing = [name for name in required
               if not any(key.startswith(name) for key in values)]
    if missing:
        _fail(f"/metrics is missing {missing}")
    generated = sum(value for key, value in values.items()
                    if key.startswith("inference_generation_tokens_total"))
    if generated <= 0:
        _fail("/metrics reports no generated tokens after a load test")
    captures = values.get("inference_graph_captures_in_service", 0.0)
    return {"generation_tokens_total": generated,
            "requests_total": sum(value for key, value in values.items()
                                  if key.startswith("inference_requests_total")),
            "graph_captures_in_service": captures,
            "kv_cache_utilization": values.get("inference_kv_cache_utilization"),
            "decode_batch_size": values.get("inference_decode_batch_size")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    results: dict[str, object] = {}
    failures: list[str] = []
    identity = check_ready(base)
    model = identity["model"]
    results["service"] = identity
    print(f"serving {model} at {base}")

    for name, function in [
        ("completion", lambda: check_completion(base, model)),
        ("chat_stream", lambda: check_chat_stream(base, model)),
        ("stop_and_seed", lambda: check_stop_and_seed(base, model)),
        ("refusals", lambda: check_refusals(base, model)),
        ("concurrency", lambda: check_concurrency(base, model, args.concurrency, args.tokens)),
        ("metrics", lambda: check_metrics(base)),
    ]:
        try:
            results[name] = function()
        except Exception as error:
            failures.append(f"{name}: {error}")
            results[name] = {"error": str(error)}
            print(f"  [FAIL] {name}: {error}")
        else:
            print(f"  [PASS] {name}: {json.dumps(results[name], default=str)[:160]}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"base_url": base, "results": results,
                                   "failures": failures}, indent=2, default=str))
        print(f"Saved -> {out}")
    print(f"\n{len(results) - len(failures) - 1} passed, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
