"""Batch-1 speculative-decoding cost model: graphed verify cost x n-gram acceptance.

Verify cost: a CUDA-graphed stock forward of k tokens over a StaticCache at a fixed
context, relative to k=1 (weight-bound decode makes a few extra tokens nearly free).
Acceptance: prompt-lookup (n-gram) drafting simulated exactly on stock greedy
continuations - a draft token is accepted iff it equals the greedy token, so this is the
acceptance a lossless greedy verifier would see. No engine integration: this decides
whether one is worth building.

    python -m benchmarks.speculative.ngram_cost_model --out results/rtx4060/spec_ngram_cost_model.json
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch


def verify_cost(model, context: int, ks: list[int], iters: int = 50) -> dict[int, float]:
    from transformers import StaticCache

    costs = {}
    device = next(model.parameters()).device
    for k in ks:
        cache = StaticCache(config=model.config, max_cache_len=context + k + 8)
        prefix = torch.randint(100, 1000, (1, context), device=device)
        with torch.inference_mode():
            model(input_ids=prefix, past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(context, device=device))
            ids = torch.randint(100, 1000, (1, k), device=device)
            position = torch.arange(context, context + k, device=device)
            for _ in range(3):
                for layer in cache.layers:
                    layer.cumulative_length.fill_(context)
                model(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=position)
            torch.cuda.synchronize()
            # StaticLayer advances a device-side write counter in place on every update;
            # reset it inside the graph so each replay verifies at the same context (28
            # one-element fills per replay, identical for every k).
            def reset():
                for layer in cache.layers:
                    layer.cumulative_length.fill_(context)
            reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                reset()
                model(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=position)
        for _ in range(5):
            graph.replay()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        end.synchronize()
        costs[k] = start.elapsed_time(end) / iters
        del graph, cache
        torch.cuda.empty_cache()
    return costs


def ngram_draft(history: list[int], k: int, max_n: int = 3, min_n: int = 1) -> list[int]:
    """Prompt lookup: find the latest earlier occurrence of the last n tokens, copy what followed."""
    for n in range(max_n, min_n - 1, -1):
        if len(history) <= n:
            continue
        suffix = history[-n:]
        for start in range(len(history) - n - 1, -1, -1):
            if history[start:start + n] == suffix:
                return history[start + n:start + n + k]
    return []


def simulate(prompt: list[int], continuation: list[int], k: int) -> tuple[int, int]:
    """Verify steps needed to emit `continuation` greedily with k-token n-gram drafts."""
    history = list(prompt)
    emitted = steps = 0
    while emitted < len(continuation):
        draft = ngram_draft(history, k)
        accepted = 0
        for token in draft:
            if emitted + accepted < len(continuation) and token == continuation[emitted + accepted]:
                accepted += 1
            else:
                break
        produced = min(accepted + 1, len(continuation) - emitted)  # +1: the verifier's own token
        history.extend(continuation[emitted:emitted + produced])
        emitted += produced
        steps += 1
    return emitted, steps


# Free-generation prompts with no source text to copy: the low-acceptance case. Any
# n-gram hit must come from the model repeating its own output.
INSTRUCTIONS = [
    "Write a short story about a lighthouse keeper.", "Explain how vaccines train the immune system.",
    "Describe the water cycle to a ten-year-old.", "What are the causes of the French Revolution?",
    "Give advice for a first job interview.", "Compare cats and dogs as pets.",
    "Explain why the sky is blue.", "Write a poem about the ocean at night.",
    "Summarize the plot of a heist movie you invent.", "How does a bicycle stay upright?",
    "Describe a perfect weekend in the mountains.", "What makes a good teacher?",
    "Explain compound interest with an example.", "Write a letter thanking a friend for their help.",
    "Describe how bread is made from wheat.", "What would a city on Mars need to survive?",
]


def draft_lookup_cost_us(history_len: int = 2048, k: int = 4, trials: int = 200) -> float:
    """Host time of one n-gram lookup over a realistic history (worst case: no match)."""
    import random
    import time

    rng = random.Random(0)
    history = [rng.randrange(150000) for _ in range(history_len)]
    started = time.perf_counter()
    for _ in range(trials):
        ngram_draft(history, k)
    return (time.perf_counter() - started) / trials * 1e6


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"])
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--prompt-set", choices=("corpus", "instructions"), default="corpus")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from benchmarks.batching.int8_kv_pressure_ab import PROFILES, make_schedule
    from engine.kernels.rope import stock_rope
    from engine.model import load_model

    ks = [1, 2, 3, 4, 5, 6, 8]
    payload = {"args": vars(args), "gpu": torch.cuda.get_device_name(), "models": {},
               "host_draft_lookup_us_history2048_nomatch": draft_lookup_cost_us()}
    print(f"host n-gram lookup (2048-token history, no match): "
          f"{payload['host_draft_lookup_us_history2048_nomatch']:.0f} us")
    for name in args.models:
        loaded = load_model(name, dtype="float16")
        model = loaded.model
        model.config._attn_implementation = "sdpa"
        costs = verify_cost(model, args.context, ks)
        if args.prompt_set == "corpus":
            schedule = make_schedule(loaded.tokenizer, dict(PROFILES["chat"], requests=args.requests), 11)
            prompts = [row["prompt_ids"] for row in schedule]
        else:
            prompts = [
                loaded.tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}], add_generation_prompt=True,
                    enable_thinking=False, tokenize=True, return_dict=False)
                for text in INSTRUCTIONS[:args.requests]
            ]
            prompts = [list(p["input_ids"]) if isinstance(p, dict) else list(p) for p in prompts]
        continuations = []
        with stock_rope(), torch.inference_mode():
            for prompt in prompts:
                ids = torch.tensor([prompt], device=loaded.device)
                out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.new_tokens,
                                     do_sample=False, temperature=None, top_p=None, top_k=None,
                                     pad_token_id=loaded.tokenizer.pad_token_id)
                continuations.append((prompt, out[0, ids.shape[1]:].tolist()))
        rows = {}
        for k in [d for d in ks if d > 1]:
            draft_len = k - 1
            emitted = steps = 0
            per_request = []
            for prompt, continuation in continuations:
                e, s = simulate(prompt, continuation, draft_len)
                emitted += e
                steps += s
                per_request.append(e / s)
            tokens_per_step = emitted / steps
            relative_cost = costs[k] / costs[1]
            rows[k] = {
                "draft_tokens": draft_len, "tokens_per_verify_step": tokens_per_step,
                "verify_cost_ms": costs[k], "relative_cost": relative_cost,
                "modelled_speedup": tokens_per_step / relative_cost,
                "per_request_tokens_per_step": {"min": min(per_request), "median": statistics.median(per_request),
                                                "max": max(per_request)},
            }
            print(f"{name} k={k} draft={draft_len}: {tokens_per_step:.2f} tok/step, cost x{relative_cost:.3f} "
                  f"-> speedup {tokens_per_step / relative_cost:.2f}x (per-request median "
                  f"{statistics.median(per_request):.2f}, min {min(per_request):.2f})", flush=True)
        payload["models"][name] = {"decode_cost_ms_k1": costs[1], "verify_cost_ms": costs, "rows": rows}
        del loaded, model
        torch.cuda.empty_cache()
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
