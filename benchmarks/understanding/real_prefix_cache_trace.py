"""Trace real Qwen exact prefix reuse and partial-tail copy-on-write."""

from __future__ import annotations

import argparse

import torch

from engine.runtime import GenerationRequest, RequestState


def _non_aligned_prompt(tokenizer, block_size: int) -> tuple[str, list[int]]:
    prompt = (
        "A shared inference service can reuse a previously computed prompt KV state. "
        "This request demonstrates exact paged prefix reuse before decoding. "
    ) * 4 + "Question: describe the cache state."
    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    while len(ids) < block_size + 1 or len(ids) % block_size == 0:
        prompt += " Add one detail."
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    return prompt, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-blocks", type=int, default=128)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA and a Qwen checkpoint.")

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    import engine.batching.continuous_batching as batching_module
    from engine.model import load_model

    loaded = load_model(args.model)
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, "cuda", num_blocks=args.num_blocks,
        block_size=16, max_active=1, prefix_cache_blocks=16,
    )
    engine.eos_ids.clear()
    prompt, ids = _non_aligned_prompt(loaded.tokenizer, engine.block_size)
    source = GenerationRequest("source", len(ids), 4, prompt_token_ids=ids)
    assert engine.submit(source)
    engine.step()  # real miss: admit + prefill + publish radix/exact entries.
    assert source.state is RequestState.DECODING
    source_table = list(source.block_table)
    source_first_token = source.output_token_ids[0]
    tail_used = len(ids) % engine.block_size
    source_tail = source_table[-1]
    # Preserve only valid prefix values from the partial final page before its owner is released.
    tail_prefix = [pool[source_tail, :tail_used].clone() for pool in engine.key_pool]
    refcounts_after_publish = [engine.block_manager.allocator.refcount(block) for block in source_table]
    engine.scheduler.finish(source.request_id, reason="TRACE_SOURCE_RELEASED")

    print("real prompt token count / block size / partial-tail used slots:", len(ids), engine.block_size, tail_used)
    print("source block table after real prefill:", source_table)
    print("source first predicted token:", source_first_token)
    print("physical block refcounts after publish, before source release:", refcounts_after_publish)
    print("prefix-cache snapshot after source release:", engine.prefix_cache.snapshot())

    hit = GenerationRequest("exact-hit", len(ids), 4, prompt_token_ids=list(ids))
    assert engine.submit(hit)
    batching_module._ATTN_CALLS = 0
    engine.step()  # exact lookup/attach only: no prefill model forward.
    assert hit.state is RequestState.DECODING
    old_tail = hit.block_table[-1]
    print("exact-hit admission:", {
        "cached_prefix_tokens": hit.cached_prefix_tokens,
        "prefilled_token_count": hit.prefilled_token_count,
        "block_table": hit.block_table,
        "first_output_from_cached_decision": hit.output_token_ids,
        "matches_source_first_token": hit.output_token_ids[0] == source_first_token,
        "attention_hook_calls_during_exact_admission": batching_module._ATTN_CALLS,
        "shared_tail_refcount": engine.block_manager.allocator.refcount(old_tail),
    })

    batching_module._ATTN_CALLS = 0
    engine.step()  # decode must copy shared partial tail, then write the consumed first output token.
    torch.cuda.synchronize()
    new_tail = hit.block_table[-1]
    copied_prefix_layers = sum(
        int(torch.equal(tail_prefix[layer], engine.key_pool[layer][new_tail, :tail_used]))
        for layer in range(engine.num_layers)
    )
    print("after first decode from exact hit:", {
        "old_shared_tail": old_tail,
        "new_private_tail": new_tail,
        "copy_on_write_happened": old_tail != new_tail,
        "copied_valid_K_prefix_layers": f"{copied_prefix_layers}/{engine.num_layers}",
        "old_tail_refcount_after_cow": engine.block_manager.allocator.refcount(old_tail),
        "new_tail_refcount_after_cow": engine.block_manager.allocator.refcount(new_tail),
        "kv_length": hit.allocation.sequence_length if hit.allocation else None,
        "outputs": hit.output_token_ids,
        "attention_hook_calls_during_decode": batching_module._ATTN_CALLS,
    })


if __name__ == "__main__":
    main()
