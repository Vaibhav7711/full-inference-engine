"""Narrow batch-1 probe: graphed n-gram speculative greedy decoding versus graphed greedy.

Both loops use the same stock model, StaticCache and CUDA-graph discipline, so the
comparison isolates speculation. Rollback is free: the cache's write counter and the
query positions are reset to the accepted length before each verify; stale slots past
it are masked causally and overwritten later. The speculative output must equal plain
greedy exactly (lossless verification); any mismatch is reported.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from benchmarks.speculative.ngram_cost_model import INSTRUCTIONS, ngram_draft


class GraphedStep:
    """One captured forward of `width` tokens at device-resident positions."""

    def __init__(self, model, cache, width: int, max_len: int):
        device = next(model.parameters()).device
        self.cache = cache
        self.ids = torch.zeros((1, width), dtype=torch.long, device=device)
        self.start = torch.zeros((), dtype=torch.long, device=device)
        self.offsets = torch.arange(width, device=device)
        self.position = torch.zeros(width, dtype=torch.long, device=device)

        def forward():
            torch.add(self.offsets, self.start, out=self.position)
            for layer in cache.layers:
                layer.cumulative_length.copy_(self.start)
            return model(input_ids=self.ids, past_key_values=cache, use_cache=True,
                         cache_position=self.position).logits[0].argmax(-1)

        with torch.inference_mode():
            for _ in range(2):
                forward()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.tokens = forward()

    @torch.inference_mode()
    def run(self, ids: list[int], start: int) -> list[int]:
        self.ids.copy_(torch.tensor([ids], dtype=torch.long), non_blocking=True)
        self.start.fill_(start)
        self.graph.replay()
        return self.tokens.tolist()


def prefill(model, cache, prompt):
    device = next(model.parameters()).device
    for layer in cache.layers:
        if layer.is_initialized:
            layer.cumulative_length.zero_()
    with torch.inference_mode():
        ids = torch.tensor([prompt], device=device)
        return int(model(input_ids=ids, past_key_values=cache, use_cache=True,
                         cache_position=torch.arange(len(prompt), device=device)).logits[0, -1].argmax())


def greedy(model, cache, step1, prompt, new_tokens):
    token = prefill(model, cache, prompt)
    out, position = [token], len(prompt)
    while len(out) < new_tokens:
        token = step1.run([token], position)[0]
        out.append(token)
        position += 1
    return out


def speculative(model, cache, stepk, width, prompt, new_tokens):
    token = prefill(model, cache, prompt)
    out, history, position, steps = [token], list(prompt) + [token], len(prompt), 0
    while len(out) < new_tokens:
        draft = ngram_draft(history, width - 1)
        draft = (draft + [0] * (width - 1))[:width - 1]
        predicted = stepk.run([token] + draft, position)
        steps += 1
        accepted = 0
        while accepted < width - 1 and draft[accepted] == predicted[accepted]:
            accepted += 1
        new = predicted[:accepted + 1]
        new = new[:new_tokens - len(out)]
        out.extend(new)
        history.extend(new)
        position += len(new)
        token = out[-1]
    return out, steps


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--width", type=int, default=5, help="verify width = draft tokens + 1")
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import StaticCache

    from benchmarks.batching.int8_kv_pressure_ab import PROFILES, make_schedule
    from engine.model import load_model

    loaded = load_model(args.model, dtype="float16")
    model, tok = loaded.model, loaded.tokenizer
    model.config._attn_implementation = "sdpa"
    sets = {
        "corpus": [row["prompt_ids"] for row in make_schedule(tok, dict(PROFILES["chat"], requests=args.requests), 11)],
        "instructions": [list(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True,
                                                     enable_thinking=False, tokenize=True, return_dict=False))
                         for t in INSTRUCTIONS[:args.requests]],
    }
    max_len = max(len(p) for prompts in sets.values() for p in prompts) + args.new_tokens + args.width + 8
    cache = StaticCache(config=model.config, max_cache_len=max_len)
    prefill(model, cache, sets["corpus"][0][:16])
    step1 = GraphedStep(model, cache, 1, max_len)
    stepk = GraphedStep(model, cache, args.width, max_len)
    result = {"args": vars(args), "gpu": torch.cuda.get_device_name(), "sets": {}}
    for name, prompts in sets.items():
        rows = []
        for prompt in prompts:
            times = {"greedy": [], "spec": []}
            for _ in range(args.repeats):  # interleaved per request
                torch.cuda.synchronize(); t = time.perf_counter()
                reference = greedy(model, cache, step1, prompt, args.new_tokens)
                torch.cuda.synchronize(); times["greedy"].append(time.perf_counter() - t)
                t = time.perf_counter()
                drafted, steps = speculative(model, cache, stepk, args.width, prompt, args.new_tokens)
                torch.cuda.synchronize(); times["spec"].append(time.perf_counter() - t)
            first = next((i for i, (a, b) in enumerate(zip(drafted, reference)) if a != b), None)
            rows.append({"prompt_tokens": len(prompt), "identical": drafted == reference, "first_mismatch": first,
                         "verify_steps": steps, "tokens_per_step": len(drafted) / steps,
                         "greedy_s": statistics.median(times["greedy"]), "spec_s": statistics.median(times["spec"]),
                         "spread": max((max(v) - min(v)) / statistics.median(v) for v in times.values())})
        speedups = [r["greedy_s"] / r["spec_s"] for r in rows]
        summary = {"identical_requests": sum(r["identical"] for r in rows), "requests": len(rows),
                   "speedup_median": statistics.median(speedups), "speedup_min": min(speedups),
                   "speedup_max": max(speedups),
                   "aggregate_speedup": sum(r["greedy_s"] for r in rows) / sum(r["spec_s"] for r in rows),
                   "tokens_per_step_median": statistics.median(r["tokens_per_step"] for r in rows)}
        result["sets"][name] = {"rows": rows, "summary": summary}
        print(name, summary, flush=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
