"""Trace real mixed-length chunked prefill and decode-first scheduling."""

from __future__ import annotations

import argparse

import torch

from engine.runtime import GenerationRequest


def _long_prompt(tokenizer, label: str, minimum_tokens: int = 48) -> tuple[str, list[int]]:
    prompt = f"{label}: " + (
        "Paged attention stores KV state in blocks while continuous batching schedules work fairly. "
        * 8
    )
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    while len(ids) < minimum_tokens:
        prompt += " Add another scheduling detail."
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    return prompt, ids


def _rows(engine, requests: list[GenerationRequest]) -> list[dict[str, object]]:
    result = []
    for request in requests:
        allocation = request.allocation
        result.append({
            "id": request.request_id,
            "state": request.state.value,
            "prefill": f"{request.prefilled_token_count}/{request.prompt_token_count}",
            "kv_length": allocation.sequence_length if allocation else None,
            "blocks": request.block_table,
            "outputs": request.output_token_ids,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA and a Qwen checkpoint.")

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(args.model)
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, "cuda", num_blocks=args.num_blocks,
        block_size=16, max_active=3, prefix_cache_blocks=0,
        prefill_chunk_size=16, max_prefill_tokens_per_iteration=16,
    )
    engine.eos_ids.clear()
    _, long_a = _long_prompt(loaded.tokenizer, "Long request A")
    _, long_b = _long_prompt(loaded.tokenizer, "Long request B")
    short = loaded.tokenizer("Short request: explain KV paging.", return_tensors="pt").input_ids[0].tolist()
    requests = [
        GenerationRequest("long-a", len(long_a), 4, prompt_token_ids=long_a),
        GenerationRequest("long-b", len(long_b), 4, prompt_token_ids=long_b),
        GenerationRequest("short", len(short), 4, prompt_token_ids=short),
    ]
    for request in requests:
        assert engine.submit(request)
    print("real prompt lengths:", {request.request_id: request.prompt_token_count for request in requests})
    print("scheduler config:", {
        "prefill_chunk_size": engine.prefill_chunk_size,
        "prefill_token_budget": engine.max_prefill_tokens_per_iteration,
        "max_active": engine.max_active,
    })

    for tick in range(args.steps):
        before = list(engine.scheduler._prefill_order)
        engine.step()
        torch.cuda.synchronize()
        print(f"\niteration {tick}:")
        print("prefill order before step:", before)
        print("prefill order after step:", list(engine.scheduler._prefill_order))
        print("requests:", _rows(engine, requests))
        print("allocator:", engine.block_manager.snapshot())


if __name__ == "__main__":
    main()
