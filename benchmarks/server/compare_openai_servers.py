"""Compare this local engine with any OpenAI-compatible server, including vLLM.

Start this engine on port 8000, start vLLM separately (usually in its own virtual
environment/container), then run:

  python -m benchmarks.server.compare_openai_servers \
    --peer-url http://127.0.0.1:8001/v1/chat/completions --requests 16

Both servers receive the same concurrent greedy chat workload. This avoids importing
vLLM into the engine environment and makes the comparison boundary explicit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


PROMPTS = [
    "Explain KV caching in one concise sentence.",
    "What is continuous batching?",
    "Why do CUDA graphs help repeated GPU work?",
    "Explain paged attention simply.",
]


def post(url: str, prompt: str, max_tokens: int) -> dict[str, object]:
    body = json.dumps({
        "model": "Qwen/Qwen3-0.6B",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    request = Request(url, body, headers={"content-type": "application/json"})
    with urlopen(request, timeout=180) as response:
        return json.load(response)


def run_arm(name: str, url: str, requests: int, max_tokens: int) -> dict[str, object]:
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=requests) as pool:
        replies = list(pool.map(lambda index: post(url, PROMPTS[index % len(PROMPTS)], max_tokens), range(requests)))
    elapsed_s = time.perf_counter() - started
    completion_tokens = [int(reply.get("usage", {}).get("completion_tokens", 0)) for reply in replies]
    ttfts = [float(reply["metrics"]["ttft_ms"]) for reply in replies if isinstance(reply.get("metrics"), dict) and reply["metrics"].get("ttft_ms") is not None]
    return {
        "name": name, "url": url, "requests": requests, "elapsed_s": elapsed_s,
        "completion_tokens": sum(completion_tokens),
        "throughput_tokens_per_second": sum(completion_tokens) / elapsed_s,
        "ttft_ms_p50": statistics.median(ttfts) if ttfts else None,
        "note": "Peer TTFT is null unless its API exposes a compatible metrics.ttft_ms field.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--peer-url", required=True, help="vLLM or another OpenAI-compatible chat endpoint")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/openai_server_comparison.json"))
    args = parser.parse_args()
    if min(args.requests, args.max_tokens, args.warmup_requests) < 1:
        parser.error("request and token counts must be positive")

    # Warm both sides independently; capture/JIT/cold-start does not belong in steady-state throughput.
    for name, url in (("engine", args.ours_url), ("peer", args.peer_url)):
        run_arm(name, url, args.warmup_requests, min(8, args.max_tokens))
    result = {
        "workload": {"requests": args.requests, "max_tokens": args.max_tokens, "concurrency": args.requests, "sampling": "greedy"},
        "engine": run_arm("engine", args.ours_url, args.requests, args.max_tokens),
        "peer": run_arm("peer", args.peer_url, args.requests, args.max_tokens),
    }
    result["throughput_speedup_engine_over_peer"] = (
        result["engine"]["throughput_tokens_per_second"] / result["peer"]["throughput_tokens_per_second"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
