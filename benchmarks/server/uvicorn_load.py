"""Real loopback HTTP/SSE load and disconnect test against uvicorn."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch


PROMPTS = [
    "Explain paged attention in one concise sentence.", "The capital city of Japan is",
    "Continuous batching improves throughput because", "Two plus two equals",
]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] if ordered else 0.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _json_get(port: int, path: str) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        payload = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError(f"GET {path} failed: {response.status} {payload}")
        return payload
    finally:
        connection.close()


def _stream(port: int, index: int, max_new_tokens: int, *, disconnect_after: int | None = None) -> dict:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=180)
    body = json.dumps({"prompt": PROMPTS[index % len(PROMPTS)], "max_new_tokens": max_new_tokens})
    started = time.perf_counter()
    connection.request("POST", "/generate/stream", body=body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        error = response.read().decode()
        connection.close()
        raise RuntimeError(f"stream failed: HTTP {response.status}: {error}")
    first_token = None
    count = 0
    try:
        while line := response.readline():
            text = line.decode().strip()
            if not text.startswith("data: "):
                continue
            event = json.loads(text.removeprefix("data: "))
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("token_id") is not None:
                count += 1
                if first_token is None:
                    first_token = time.perf_counter()
                if disconnect_after is not None and count >= disconnect_after:
                    connection.close()
                    return {"disconnected": True, "tokens": count}
            if event.get("finish_reason") is not None:
                break
    finally:
        connection.close()
    finished = time.perf_counter()
    if first_token is None or count != max_new_tokens:
        raise RuntimeError(f"stream produced {count}/{max_new_tokens} tokens")
    return {"ttft_ms": (first_token - started) * 1000,
            "completion_ms": (finished - started) * 1000, "tokens": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--output", default="results/server_uvicorn_phase15.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    if min(args.requests, args.max_new_tokens, args.num_blocks) <= 0:
        parser.error("requests, max-new-tokens, and num-blocks must be positive")

    import uvicorn
    from engine.server.api import create_app

    port = args.port or _free_port()
    app = create_app(args.model, max_active=16, num_blocks=args.num_blocks)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    server_thread = threading.Thread(target=server.run, name="uvicorn-load-target", daemon=True)
    server_thread.start()
    deadline = time.monotonic() + 180
    while not server.started:
        if not server_thread.is_alive():
            raise RuntimeError("uvicorn stopped during startup")
        if time.monotonic() >= deadline:
            raise TimeoutError("uvicorn/model startup exceeded 180 seconds")
        time.sleep(0.1)

    try:
        # Warm graph and transport paths, then prove a dropped socket reaches the worker.
        _stream(port, 0, min(4, args.max_new_tokens))
        cancelled_before = _json_get(port, "/health")["cancelled_requests"]
        _stream(port, 1, max(16, args.max_new_tokens), disconnect_after=1)
        cancel_deadline = time.monotonic() + 10
        while _json_get(port, "/health")["cancelled_requests"] <= cancelled_before:
            if time.monotonic() >= cancel_deadline:
                raise RuntimeError("disconnected SSE request was not cancelled by the worker")
            time.sleep(0.05)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.requests) as executor:
            samples = list(executor.map(
                lambda index: _stream(port, index, args.max_new_tokens), range(args.requests)
            ))
        elapsed = time.perf_counter() - started
        health = _json_get(port, "/health")
    finally:
        server.should_exit = True
        server_thread.join(timeout=30)

    ttft = [sample["ttft_ms"] for sample in samples]
    completion = [sample["completion_ms"] for sample in samples]
    total_tokens = sum(sample["tokens"] for sample in samples)
    result = {
        "config": vars(args), "port": port, "total_tokens": total_tokens,
        "elapsed_s": elapsed, "throughput_tok_s": total_tokens / elapsed,
        "ttft_p50_ms": statistics.median(ttft), "ttft_p95_ms": _percentile(ttft, 0.95),
        "ttft_p99_ms": _percentile(ttft, 0.99),
        "completion_p50_ms": statistics.median(completion),
        "completion_p95_ms": _percentile(completion, 0.95),
        "completion_p99_ms": _percentile(completion, 0.99),
        "disconnect_cancelled": True, "health": health, "samples": samples,
    }
    print("\nReal uvicorn SSE load")
    print(f"requests={args.requests} tokens={total_tokens} throughput={result['throughput_tok_s']:.1f} tok/s")
    print(f"TTFT p50/p95/p99: {result['ttft_p50_ms']:.2f}/{result['ttft_p95_ms']:.2f}/{result['ttft_p99_ms']:.2f} ms")
    print(f"completion p50/p95/p99: {result['completion_p50_ms']:.2f}/{result['completion_p95_ms']:.2f}/{result['completion_p99_ms']:.2f} ms")
    print("disconnect cancellation: passed")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
