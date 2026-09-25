"""Drive the engine as an imported library against a real checkpoint.

Everything else in this repository measures the engine against itself. This is the other
test: install the package, import it, hand it a model, and use it the way an application
would - batched generation, per-request sampling, and staggered arrivals through the
scheduler - then print what it reports about itself.

    pip install -e ".[dev,server]"
    python examples/library_usage.py --model Qwen/Qwen3-1.7B

It needs a CUDA device: the loader refuses CPU deliberately, because every timing and
cache behaviour this project records describes the GPU path.

What each section demonstrates, and what to look at:

  1. resolution   which attention backends this GPU resolved to, and why. On an
                  unmeasured architecture every reason says so.
  2. capacity     the KV arithmetic, printed rather than assumed: bytes per token, how
                  many tokens the pool holds, how much VRAM is left idle.
  3. batch        one greedy call over several prompts. Tokens per second here is the
                  batched decode rate, not a single-stream rate.
  4. sampling     two requests with the same seed produce the same text; a greedy
                  request in the same batch is unaffected by its sampled neighbour.
  5. arrivals     requests submitted while others are decoding, which is the case the
                  whole scheduler exists for. Per-request TTFT and inter-token latency
                  come from the engine's own accounting, not from a wrapper's stopwatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def human(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B",
                        help="any Llama-style causal LM; geometry the paged kernels "
                             "cannot serve is refused at load with the reason")
    parser.add_argument("--dtype", default="auto",
                        help="auto follows the measured per-device policy; float16 "
                             "keeps results comparable with the recorded runs")
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=0,
                        help="0 sizes the KV pool from free VRAM")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch

    from engine.backends import Geometry, report as backend_report
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.kernels.device import current_device, fits_in_memory
    from engine.model import load_model
    from engine.model.adapters import describe, geometry_of
    from engine.runtime import GREEDY, GenerationRequest, RequestState, SamplingParams

    record: dict = {"model": args.model}

    # ---------------------------------------------------------------- 1. resolution
    print(f"loading {args.model} (dtype={args.dtype}) ...", flush=True)
    started = perf_counter()
    loaded = load_model(args.model, dtype=args.dtype)
    print(f"  loaded in {human(perf_counter() - started)} as {loaded.dtype}")

    model_facts = describe(loaded.model)
    shapes = geometry_of(loaded.model.config)
    profile = current_device()
    print(f"\ndevice: {profile}")
    print(f"model:  {model_facts['model_type']} ({model_facts['class']}), "
          f"{shapes.num_layers} layers, {shapes.num_q_heads}/{shapes.num_kv_heads} heads, "
          f"head_dim {shapes.head_dim}")
    print(f"fusions installed structurally: {model_facts['mlp_modules']} MLP, "
          f"{model_facts['norm_modules']} norm, rope in {model_facts['rope_module']}")

    geometry = Geometry(
        num_q_heads=shapes.num_q_heads, num_kv_heads=shapes.num_kv_heads,
        head_dim=shapes.head_dim, block_size=args.block_size,
        dtype=str(loaded.dtype).removeprefix("torch."),
    )
    backends = backend_report(profile, geometry)
    print(f"\nbackends resolved for this GPU ({'measured' if backends['measured'] else 'NOT measured'}):")
    for key, value in backends["defaults"].items():
        print(f"  {key:18s} {value}")
    for key, value in backends["reasons"].items():
        print(f"    {key}: {value}")
    unavailable = [f"{row['name']} ({row['reason']})"
                   for phase in ("decode", "prefill")
                   for row in backends[f"{phase}_backends"] if not row["available"]]
    if unavailable:
        print(f"  unavailable here: {'; '.join(unavailable)}")
    record["backends"] = backends

    # ---------------------------------------------------------------- 2. capacity
    kv_per_token = shapes.kv_bytes_per_token(
        dtype_bytes=2 if loaded.dtype in (torch.float16, torch.bfloat16) else 4,
    )
    weight_bytes = sum(p.numel() * p.element_size() for p in loaded.model.parameters())
    num_blocks = args.num_blocks
    if num_blocks <= 0:
        free, total = torch.cuda.mem_get_info()
        # Leave room for activations, graph pools and the allocator's own slack. The
        # engine's own check is consulted rather than guessed at.
        budget = max(0, free - int(1.5e9))
        num_blocks = max(16, int(budget // (kv_per_token * args.block_size)))
        print(f"\nsizing the KV pool from {free / 1e9:.2f} GB free: "
              f"{num_blocks} blocks x {args.block_size} tokens")
    pool_tokens = num_blocks * args.block_size
    pool_bytes = pool_tokens * kv_per_token
    ok, why = fits_in_memory(weight_bytes, kv_pool_bytes=pool_bytes)
    print(f"\ncapacity arithmetic")
    print(f"  weights          {weight_bytes / 1e9:>7.2f} GB")
    print(f"  KV per token     {kv_per_token / 1024:>7.0f} KiB")
    print(f"  KV pool          {pool_bytes / 1e9:>7.2f} GB  ({pool_tokens} tokens, "
          f"{pool_tokens // args.max_active} per row at max_active={args.max_active})")
    print(f"  fits: {ok} - {why}")
    record["capacity"] = {"weight_bytes": weight_bytes, "kv_bytes_per_token": kv_per_token,
                          "pool_tokens": pool_tokens, "fits": ok, "note": why}
    if not ok:
        print("  refusing to continue; pass a smaller --num-blocks")
        return 1

    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, loaded.device,
        num_blocks=num_blocks, block_size=args.block_size, max_active=args.max_active,
        cuda_graph_batch_sizes=tuple(s for s in (1, 2, 4, 8, 16) if s <= args.max_active),
    )
    started = perf_counter()
    summary = engine.warmup()
    print(f"\nwarmup {human(perf_counter() - started)}: {summary['graphs']} decode graphs, "
          f"{summary['prefill_graphs']} prefill, {summary.get('fused_graphs', 0)} fused; "
          f"captures after warmup: {engine.lazy_graph_captures}")
    record["warmup"] = summary

    # ---------------------------------------------------------------- 3. batch
    prompts = [
        "Explain what a KV cache is, in two sentences.",
        "Write a haiku about garbage collection.",
        "What is the capital of Australia, and why is it not Sydney?",
        "List three reasons a GPU kernel might be memory-bound.",
    ][:args.max_active]
    print(f"\n--- batched greedy generation ({len(prompts)} prompts) ---")
    started = perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=args.max_new_tokens)
    elapsed = perf_counter() - started
    produced = sum(len(o) for o in outputs)
    print(f"{produced} tokens in {human(elapsed)} = {produced / elapsed:.0f} tok/s aggregate")
    for prompt, tokens in zip(prompts, outputs):
        text = loaded.tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"\n  > {prompt}\n    {text.strip()[:300]}")
    record["batch"] = {"prompts": len(prompts), "tokens": produced, "seconds": elapsed,
                       "tokens_per_second": produced / elapsed}

    # ---------------------------------------------------------------- 4. sampling
    print(f"\n--- per-request sampling ---")
    seeded = SamplingParams(temperature=0.9, top_p=0.95, seed=20260924)
    prompt = "In one sentence, why is continuous batching faster than static batching?"
    runs = []
    for _ in range(2):
        engine.reset()
        runs.append(engine.generate([prompt], max_new_tokens=32, sampling=seeded)[0])
    same = runs[0] == runs[1]
    print(f"  same seed twice -> identical tokens: {same}")
    print(f"    {loaded.tokenizer.decode(runs[0], skip_special_tokens=True).strip()[:200]}")

    engine.reset()
    mixed = engine.generate(
        [prompt, prompt], max_new_tokens=24,
        sampling=[GREEDY, SamplingParams(temperature=1.0, top_p=0.9, seed=7)],
    )
    engine.reset()
    alone = engine.generate([prompt], max_new_tokens=24)[0]
    print(f"  greedy row unaffected by a sampled neighbour: {mixed[0] == alone}")
    record["sampling"] = {"seed_reproducible": same, "greedy_isolated": mixed[0] == alone}

    # ---------------------------------------------------------------- 5. arrivals
    print(f"\n--- staggered arrivals through the scheduler ---")
    engine.reset()
    arriving = [
        ("short", "Name one advantage of paged attention.", 24),
        ("long", "Explain in detail how a paged KV cache, a scheduler and CUDA graphs "
                 "fit together in an inference engine, and what each one costs.", 96),
        ("short2", "What does TTFT stand for?", 16),
        ("long2", "Describe how chunked prefill keeps decode latency stable when a long "
                  "prompt arrives mid-flight.", 64),
    ]
    requests = []
    for name, text, budget in arriving:
        ids = loaded.tokenizer(text, return_tensors="pt").input_ids[0].tolist()
        requests.append(GenerationRequest(request_id=name, prompt_token_count=len(ids),
                                          max_new_tokens=budget, prompt_token_ids=ids))
    pending = list(requests)
    engine.submit(pending.pop(0))
    steps = 0
    started = perf_counter()
    while engine.has_unfinished_requests or pending:
        # A new request every few steps, while the others are mid-generation.
        if pending and steps in (3, 9, 15):
            engine.submit(pending.pop(0))
        engine.step()
        steps += 1
        if steps > 5000:
            break
    wall = perf_counter() - started
    delivered = sum(len(r.output_token_ids) for r in requests)
    print(f"  {len(requests)} requests, {delivered} tokens, {steps} steps, {human(wall)} "
          f"= {delivered / wall:.0f} tok/s")
    print(f"  {'request':>8} {'state':>10} {'prompt':>7} {'out':>5} {'ttft':>9} {'itl':>9}")
    per_request = []
    for request in requests:
        report = request.latency_report()
        print(f"  {request.request_id:>8} {request.state.name:>10} "
              f"{request.prompt_token_count:>7} {len(request.output_token_ids):>5} "
              f"{(report['ttft_ms'] or 0):>8.1f}m {(report['mean_itl_ms'] or 0):>8.1f}m")
        per_request.append({"id": request.request_id, "state": request.state.name, **report})
    stats = engine.stats_snapshot()
    print(f"  steps: {stats['prefill_steps']} carried prefill, "
          f"{stats['decode_only_steps']} decode-only, {stats['fused_steps']} fused")
    print(f"  KV utilization {stats['kv_utilization']:.1%}, "
          f"preemptions {stats['preemptions_total']}")
    record["arrivals"] = {"requests": per_request, "steps": steps, "seconds": wall,
                          "tokens_per_second": delivered / wall, "stats": stats}

    every_finished = all(r.state is RequestState.FINISHED for r in requests)
    print(f"\nall requests reached FINISHED: {every_finished}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2, default=str))
        print(f"Saved -> {args.out}")
    return 0 if every_finished else 1


if __name__ == "__main__":
    raise SystemExit(main())
