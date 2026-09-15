"""Concurrent HTTP/SSE burst-load measurement for the continuous batching service."""

from __future__ import annotations

import argparse
import json
import os
import statistics
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--output", default="results/server_burst_load_phase14.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")
    if min(args.requests, args.max_new_tokens, args.num_blocks) <= 0:
        parser.error("requests, max-new-tokens, and num-blocks must be positive")

    from fastapi.testclient import TestClient
    from engine.server.api import create_app

    app = create_app(args.model, max_active=16, num_blocks=args.num_blocks)
    with TestClient(app) as client:
        def one_request(index: int) -> dict[str, float | int]:
            payload = {"prompt": PROMPTS[index % len(PROMPTS)], "max_new_tokens": args.max_new_tokens}
            started = time.perf_counter()
            first_token_s = None
            count = 0
            with client.stream("POST", "/generate/stream", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line.removeprefix("data: "))
                    if event.get("token_id") is not None:
                        count += 1
                        if first_token_s is None:
                            first_token_s = time.perf_counter()
            finished = time.perf_counter()
            if count != args.max_new_tokens or first_token_s is None:
                raise RuntimeError("stream ended before the requested token count")
            return {"ttft_ms": (first_token_s - started) * 1000,
                    "completion_ms": (finished - started) * 1000, "tokens": count}

        # One small request compiles graph and serving paths before the measured burst.
        one_request(0)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.requests) as executor:
            samples = list(executor.map(one_request, range(args.requests)))
        elapsed = time.perf_counter() - started
    ttft = [sample["ttft_ms"] for sample in samples]
    completion = [sample["completion_ms"] for sample in samples]
    total_tokens = sum(sample["tokens"] for sample in samples)
    result = {
        "config": vars(args), "total_tokens": total_tokens, "elapsed_s": elapsed,
        "throughput_tok_s": total_tokens / elapsed,
        "ttft_p50_ms": statistics.median(ttft), "ttft_p95_ms": _percentile(ttft, 0.95),
        "completion_p50_ms": statistics.median(completion),
        "completion_p95_ms": _percentile(completion, 0.95), "samples": samples,
    }
    print("\nContinuous API burst load")
    print(f"requests={args.requests} tokens={total_tokens} throughput={result['throughput_tok_s']:.1f} tok/s")
    print(f"TTFT p50/p95: {result['ttft_p50_ms']:.2f}/{result['ttft_p95_ms']:.2f} ms")
    print(f"completion p50/p95: {result['completion_p50_ms']:.2f}/{result['completion_p95_ms']:.2f} ms")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
