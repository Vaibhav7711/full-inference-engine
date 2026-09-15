"""CUDA smoke test for FastAPI continuous batching and SSE completion."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import torch


def _sse_token_ids(response) -> list[int]:
    token_ids = []
    for line in response.iter_lines():
        if not line.startswith("data: "):
            continue
        event = json.loads(line.removeprefix("data: "))
        if event.get("token_id") is not None:
            token_ids.append(event["token_id"])
    return token_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-blocks", type=int, default=1024)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")

    from fastapi.testclient import TestClient
    from engine.server.api import create_app

    prompts = [
        "Explain continuous batching in one sentence.", "The capital of Japan is",
        "Paged KV caching means", "Two plus two equals",
    ]
    app = create_app(args.model, max_active=16, num_blocks=args.num_blocks)
    with TestClient(app) as client:
        def generate(index: int):
            response = client.post("/generate", json={
                "prompt": prompts[index % len(prompts)], "max_new_tokens": args.max_new_tokens,
            })
            response.raise_for_status()
            return response.json()

        with ThreadPoolExecutor(max_workers=args.requests) as executor:
            responses = list(executor.map(generate, range(args.requests)))
        if any(len(item["token_ids"]) != args.max_new_tokens for item in responses):
            raise RuntimeError("one or more batched API responses ended unexpectedly")
        ordinary = generate(0)
        with client.stream("POST", "/generate/stream", json={
            "prompt": prompts[0], "max_new_tokens": args.max_new_tokens,
        }) as stream:
            stream.raise_for_status()
            streamed = _sse_token_ids(stream)
        if ordinary["token_ids"] != streamed:
            raise RuntimeError("SSE token sequence diverged from /generate")
    print(f"continuous API: {args.requests} concurrent requests completed; SSE tokens identical")


if __name__ == "__main__":
    main()
