"""Synthetic mixed-length allocation workload for contiguous vs paged KV capacity."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from engine.cache import ContiguousKVAllocator, KVCacheGeometry, PagedKVCacheManager
from engine.metrics import latency_summary_ms


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt_tokens: int
    output_tokens: int


def build_arrivals(steps: int, arrival_probability: float, seed: int) -> list[RequestSpec | None]:
    rng = random.Random(seed)
    arrivals: list[RequestSpec | None] = []
    for step in range(steps):
        if rng.random() > arrival_probability:
            arrivals.append(None)
            continue
        arrivals.append(RequestSpec(
            request_id=f"request-{step}",
            prompt_tokens=rng.randint(8, 256),
            output_tokens=rng.randint(8, 128),
        ))
    return arrivals


def simulate_contiguous(arrivals: list[RequestSpec | None], capacity_tokens: int) -> dict[str, object]:
    geometry = KVCacheGeometry(1, 1, 1, torch.float16)
    allocator = ContiguousKVAllocator(capacity_tokens, geometry)
    active: dict[str, int] = {}
    fragmentation: list[float] = []
    admitted = rejected = 0
    for spec in arrivals:
        for request_id in [key for key, remaining in active.items() if remaining == 0]:
            allocator.release(request_id)
            del active[request_id]
        if spec is not None:
            if allocator.allocate(spec.request_id, spec.prompt_tokens + spec.output_tokens) is None:
                rejected += 1
            else:
                active[spec.request_id] = spec.output_tokens
                admitted += 1
        fragmentation.append(allocator.external_fragmentation)
        active = {request_id: remaining - 1 for request_id, remaining in active.items()}
    return {
        "admitted": admitted,
        "rejected": rejected,
        "mean_external_fragmentation": sum(fragmentation) / len(fragmentation),
        "external_fragmentation": latency_summary_ms(fragmentation),
        "final": allocator.snapshot(),
    }


def simulate_paged(arrivals: list[RequestSpec | None], capacity_tokens: int, block_size_tokens: int) -> dict[str, object]:
    manager = PagedKVCacheManager(capacity_tokens // block_size_tokens, block_size_tokens)
    active: dict[str, int] = {}
    internal_fragmentation: list[int] = []
    admitted = rejected = exhausted = 0
    for spec in arrivals:
        for request_id in [key for key, remaining in active.items() if remaining == 0]:
            manager.release(request_id)
            del active[request_id]
        if spec is not None:
            if manager.reserve(spec.request_id, spec.prompt_tokens, sequence_length=spec.prompt_tokens) is None:
                rejected += 1
            else:
                active[spec.request_id] = spec.output_tokens
                admitted += 1
        for request_id in list(active):
            if not manager.append_tokens(request_id):
                manager.release(request_id)
                del active[request_id]
                exhausted += 1
        snapshot = manager.snapshot()
        internal_fragmentation.append(snapshot["internal_fragmentation_tokens"])
        active = {request_id: remaining - 1 for request_id, remaining in active.items()}
    return {
        "admitted": admitted,
        "rejected_on_admission": rejected,
        "exhausted_while_growing": exhausted,
        "internal_fragmentation_tokens": latency_summary_ms([float(value) for value in internal_fragmentation]),
        "final": manager.snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Contiguous vs paged KV allocator workload")
    parser.add_argument("--capacity-tokens", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--arrival-probability", type=float, default=0.7)
    parser.add_argument("--block-sizes", default="8,16,32,64")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/allocator_workload.json"))
    args = parser.parse_args()
    if args.capacity_tokens <= 0 or args.steps <= 0 or not 0 <= args.arrival_probability <= 1:
        parser.error("invalid capacity, steps, or arrival probability")
    block_sizes = [int(value) for value in args.block_sizes.split(",")]
    if any(size <= 0 or args.capacity_tokens % size for size in block_sizes):
        parser.error("block sizes must be positive divisors of capacity-tokens")
    arrivals = build_arrivals(args.steps, args.arrival_probability, args.seed)
    record = {
        "workload": {"capacity_tokens": args.capacity_tokens, "steps": args.steps, "arrival_probability": args.arrival_probability, "seed": args.seed},
        "contiguous": simulate_contiguous(arrivals, args.capacity_tokens),
        "paged": {str(size): simulate_paged(arrivals, args.capacity_tokens, size) for size in block_sizes},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
