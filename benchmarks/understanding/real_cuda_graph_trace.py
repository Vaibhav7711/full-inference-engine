"""Trace real padded CUDA-graph capture and replay in ContinuousBatchingEngine."""

from __future__ import annotations

import argparse

import torch

from engine.runtime import GenerationRequest, RequestState


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
        block_size=16, max_active=4, prefix_cache_blocks=0,
        cuda_graph_batch_sizes=(2, 4),
    )
    engine.eos_ids.clear()
    prompts = [
        "CUDA graphs replay fixed-width decode work.",
        "Paged KV blocks support concurrent requests.",
        "A dummy row pads three requests to a width-four graph.",
    ]
    requests = []
    for row, prompt in enumerate(prompts):
        ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        request = GenerationRequest(f"graph-{row}", len(ids), 4, prompt_token_ids=ids)
        assert engine.submit(request)
        requests.append(request)
    engine.step()  # real admission + prefill; all rows become DECODING.
    active = [request for request in engine.scheduler.active.values() if request.state is RequestState.DECODING]
    assert len(active) == 3

    # Show exactly what will be replayed. The fourth row is real metadata but has no
    # customer request: it owns a permanent dummy physical page and sequence length zero.
    inputs, positions, tables, lengths = engine._prepare_decode_metadata(active, graph_bucket_size=4)
    torch.cuda.synchronize()
    print("configured graph buckets:", engine.cuda_graph_batch_sizes)
    print("permanently reserved dummy block IDs:", engine._graph_dummy_blocks)
    print("live rows:", [request.request_id for request in active])
    print("width-four staged GPU metadata:", {
        "input_ids": inputs.cpu().tolist(),
        "position_ids": positions.cpu().tolist(),
        "seq_lens": lengths.cpu().tolist(),
        "block_table_column_0": tables[:, 0].cpu().tolist(),
    })
    print("dummy allocator record:", engine.block_manager.requests["__cuda_graph_dummy_rows__"])

    # This step selects bucket 4, captures only on first use, then replays it. Python
    # hooks execute for warmup and capture; replay launches recorded CUDA work directly.
    batching_module._ATTN_CALLS = 0
    engine.step()
    torch.cuda.synchronize()
    print("captured graph keys:", sorted(engine._decode_graphs))
    graph = next(item for key, item in engine._decode_graphs.items() if key[0] == 4)
    print("captured graph configuration:", {
        "batch_size": graph.batch_size, "block_n": graph.block_n,
        "num_warps": graph.num_warps, "logits_shape": tuple(graph.logits.shape),
    })
    print("Python attention-hook calls while warmup + capture occurred:", batching_module._ATTN_CALLS)
    print("post-replay live request state:", [
        {
            "id": request.request_id,
            "kv_length": request.allocation.sequence_length if request.allocation else None,
            "outputs": request.output_token_ids,
        }
        for request in requests
    ])
    print("dummy allocator record remains length zero:", engine.block_manager.requests["__cuda_graph_dummy_rows__"])


if __name__ == "__main__":
    main()
