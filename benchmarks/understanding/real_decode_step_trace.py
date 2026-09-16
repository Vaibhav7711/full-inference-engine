"""Trace one real Qwen decode iteration through ContinuousBatchingEngine.

This is an observability probe over production engine code: it loads Qwen3-0.6B,
prefills two real prompts, then observes the following real batched decode iteration.
"""

from __future__ import annotations

import argparse

import torch

from engine.runtime import GenerationRequest, RequestState


def _target_slots(engine, requests: list[GenerationRequest]) -> list[tuple[int, int, int]]:
    """Return (request row, physical block, offset) for the imminent decode writes."""
    slots = []
    for row, request in enumerate(requests):
        assert request.allocation is not None
        position = request.allocation.sequence_length
        physical, offset = request.allocation.physical_location(position)
        slots.append((row, physical, offset))
    return slots


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
        block_size=16, max_active=2, prefix_cache_blocks=0,
        cuda_graph_batch_sizes=(),
    )
    # Prevent early EOS from shortening the trace; this does not alter model logits.
    engine.eos_ids.clear()
    prompts = [
        "Paged KV caches store each sequence in fixed physical blocks.",
        "Continuous batching schedules decode work across active requests.",
    ]
    requests = []
    for row, prompt in enumerate(prompts):
        ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        request = GenerationRequest(
            request_id=f"real-{row}", prompt_token_count=len(ids), max_new_tokens=4,
            prompt_token_ids=ids,
        )
        assert engine.submit(request)
        requests.append(request)

    # First step admits + prefills. The first generated token is selected, but has not
    # entered KV yet; that happens in the next decode step.
    engine.step()
    assert all(request.state is RequestState.DECODING for request in requests)
    torch.cuda.synchronize()

    active = [request for request in engine.scheduler.active.values() if request.state is RequestState.DECODING]
    slots = _target_slots(engine, active)
    pool_ptrs_before = [pool.data_ptr() for pool in engine.key_pool]
    # Only clone the imminent tiny write slots, not the complete persistent pools.
    before = [
        [engine.key_pool[layer][physical, offset].clone() for _, physical, offset in slots]
        for layer in range(engine.num_layers)
    ]
    print("model geometry:", {
        "layers": engine.num_layers, "q_heads": engine.num_q_heads,
        "kv_heads": engine.num_kv_heads, "head_dim": engine.head_dim,
        "pool_shape_per_layer": tuple(engine.key_pool[0].shape),
    })
    print("decode batch rows:", [request.request_id for request in active])
    print("pre-decode metadata:", [
        {
            "id": request.request_id,
            "next_input_token": request.next_token_id,
            "position": request.allocation.sequence_length if request.allocation else None,
            "block_table": request.block_table,
            "next_write_slot": slots[row][1:],
        }
        for row, request in enumerate(active)
    ])

    batching_module._ATTN_CALLS = 0
    engine.step()
    torch.cuda.synchronize()

    changed_slots = 0
    for layer in range(engine.num_layers):
        for row, (_, physical, offset) in enumerate(slots):
            changed_slots += int(not torch.equal(before[layer][row], engine.key_pool[layer][physical, offset]))
    print("custom attention-hook calls during one decode forward:", batching_module._ATTN_CALLS)
    print("changed K slots / expected slots:", changed_slots, "/", engine.num_layers * len(active))
    print("all 28 K-pool data pointers unchanged:", pool_ptrs_before == [pool.data_ptr() for pool in engine.key_pool])
    print("post-decode request state:", [
        {
            "id": request.request_id,
            "state": request.state.value,
            "kv_length": request.allocation.sequence_length if request.allocation else None,
            "outputs": request.output_token_ids,
        }
        for request in requests
    ])


if __name__ == "__main__":
    main()
