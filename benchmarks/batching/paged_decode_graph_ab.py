"""Fixed-width paged decode: ordinary model launches versus CUDA-Graph replay."""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch

from engine.runtime import GenerationRequest


def _time(callable_, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        callable_()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--output", default="results/paged_decode_graph_ab.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("Requires CUDA")

    from engine.batching.continuous_batching import _BatchContext, _clear_batch_ctx, _set_batch_ctx, ContinuousBatchingEngine
    from engine.graphs import capture_paged_decode_graph
    from engine.kernels.paged_decode_config import select_paged_decode_config
    from engine.model import load_model

    loaded = load_model(args.model)
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, "cuda", max_active=args.batch_size,
        num_blocks=args.num_blocks, prefix_cache_blocks=0,
    )
    fragment = loaded.tokenizer("CUDA graphs replay a stable paged decode bucket efficiently. ", return_tensors="pt").input_ids[0].tolist()
    prompt_ids = (fragment * ((args.prompt_tokens + len(fragment) - 1) // len(fragment)))[:args.prompt_tokens]
    requests = [GenerationRequest(f"r{i}", len(prompt_ids), 4, prompt_token_ids=prompt_ids) for i in range(args.batch_size)]
    for request in requests:
        engine.scheduler.submit(request)
    active = engine.scheduler.admit_available(max_active_requests=args.batch_size)
    engine.prefill_batch(active)
    for request in active:
        if not engine._ensure_kv_capacity(request, request.allocation.sequence_length + 1):
            raise RuntimeError("insufficient KV capacity for graph seed")
    inputs, positions, tables, lengths = engine._prepare_decode_metadata(active)
    block_n, warps = select_paged_decode_config(max(request.allocation.sequence_length + 1 for request in active), len(active))
    context = _BatchContext(engine.key_pool, engine.value_pool, tables, lengths, engine.block_size, block_n, warps,
                            engine.key_scale_pool, engine.value_scale_pool)
    engine.model.config._attn_implementation = engine.ATTN_NAME
    if hasattr(engine.model.config, "_attn_implementation_internal"):
        engine.model.config._attn_implementation_internal = engine.ATTN_NAME

    def ordinary():
        _set_batch_ctx(context)
        try:
            return engine.model(input_ids=inputs, position_ids=positions, use_cache=False, return_dict=True).logits
        finally:
            _clear_batch_ctx()

    normal_logits = ordinary()
    for _ in range(args.warmup):
        ordinary()
    torch.cuda.synchronize()
    ordinary_ms = _time(ordinary, args.repeats)
    try:
        captured = capture_paged_decode_graph(engine, batch_size=len(active), block_n=block_n, num_warps=warps)
        captured.replay()
        torch.cuda.synchronize()
        graph_ms = _time(captured.replay, args.repeats)
        graph_logits = captured.replay()
        close = bool(torch.allclose(normal_logits, graph_logits, atol=1e-3, rtol=1e-3))
        record = {
            "workload": {"batch_size": len(active), "prompt_tokens": len(prompt_ids), "block_n": block_n, "warps": warps},
            "ordinary_median_ms": ordinary_ms, "graph_median_ms": graph_ms,
            "speedup": ordinary_ms / graph_ms, "logits_close": close,
            "note": "Fixed-width replay only; request state is intentionally not advanced during this capture experiment.",
        }
    except Exception as error:
        record = {"ordinary_median_ms": ordinary_ms, "graph_capture_error": repr(error)}
    print(json.dumps(record, indent=2))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(record, handle, indent=2)


if __name__ == "__main__":
    main()
