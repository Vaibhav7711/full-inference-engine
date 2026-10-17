"""Trace a warmed, graph-replayed decode step on a real model.

Use Nsight Systems capture-range control to exclude model loading, prefill, and graph
capture from the trace. The marked range contains only steady-state decode steps.

    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
      --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      -o results/rtx4060/live_decode_b8_ctx1024 \
      .venv/bin/python -m benchmarks.batching.trace_live_decode
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from engine.batching.continuous_batching import ContinuousBatchingEngine
from engine.model import load_model
from engine.runtime import GenerationRequest, RequestState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--decode-attention", choices=("per_head", "ada", "split_k"), default="per_head")
    parser.add_argument("--decode-num-splits", type=int, default=0)
    parser.add_argument("--output", default="results/rtx4060/live_decode_trace_metadata.json")
    args = parser.parse_args()
    if min(args.batch, args.context, args.steps, args.num_blocks) <= 0:
        parser.error("batch, context, steps, and num-blocks must be positive")
    if args.decode_num_splits < 0:
        parser.error("decode-num-splits must be nonnegative")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    needed_blocks = args.batch * ((args.context + args.steps + 1 + 15) // 16)
    if needed_blocks > args.num_blocks:
        parser.error(f"at least {needed_blocks} blocks are required for this trace")

    loaded = load_model(args.model, dtype="float16")
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, loaded.device,
        num_blocks=args.num_blocks, block_size=16, max_active=args.batch,
        prefix_cache_blocks=0, cuda_graph_batch_sizes=(args.batch,),
        decode_attention=args.decode_attention, decode_num_splits=args.decode_num_splits,
        prefill_attention="sdpa",
    )
    generator = torch.Generator().manual_seed(4070)
    vocab = int(loaded.model.config.vocab_size)
    engine.eos_ids = set()
    # Early admitted rows decode while later rows consume their prefill chunks.
    # Keep those rows alive until the entire batch reaches steady decode.
    max_new_tokens = args.steps + args.batch * math.ceil(
        args.context / engine.max_prefill_tokens_per_iteration
    ) + 16
    for index in range(args.batch):
        ids = torch.randint(1, vocab, (args.context,), generator=generator).tolist()
        engine.submit(GenerationRequest(
            request_id=f"trace-{index}", prompt_token_count=len(ids),
            max_new_tokens=max_new_tokens, prompt_token_ids=ids,
        ))

    # Process real prefill and capture the graph before starting the profiler range.
    prefill_steps = 0
    while any(request.state is RequestState.PREFILLING for request in engine.scheduler.active.values()) \
            or engine.scheduler.waiting:
        engine.step()
        prefill_steps += 1
        if prefill_steps > args.batch * (args.context + 1):
            raise RuntimeError("prefill did not converge")
    active = list(engine.scheduler.active.values())
    if len(active) != args.batch or any(request.state is not RequestState.DECODING for request in active):
        raise RuntimeError("trace requires a full batch of decoding requests")
    engine.decode_step(active)
    torch.cuda.synchronize()
    if engine.lazy_graph_captures == 0:
        raise RuntimeError("decode graph was not captured before trace")
    captures_before = engine.lazy_graph_captures

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"steady_decode_b{args.batch}_ctx{args.context}")
    try:
        for index in range(args.steps):
            torch.cuda.nvtx.range_push(f"decode_step_{index}")
            try:
                active = list(engine.scheduler.active.values())
                engine.decode_step(active)
            finally:
                torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    if engine.lazy_graph_captures != captures_before:
        raise RuntimeError("a new graph was captured inside the measured range")

    result = {
        "gpu": torch.cuda.get_device_name(), "model": args.model,
        "batch": args.batch, "context": args.context, "steps": args.steps,
        "decode_attention": args.decode_attention, "prefill_steps": prefill_steps,
        "decode_num_splits": args.decode_num_splits,
        "decode_graph_keys": [list(key) for key in engine._decode_graphs],
        "lazy_graph_captures": engine.lazy_graph_captures,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
