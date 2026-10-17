"""Single-stream TTFT on Qwen3-4B-AWQ by prefill projection path and prompt length.

One model, three engines that differ only in how AWQ projections run for prefill rows:
tinygemm everywhere; the fused Triton int4pack kernel from 64 rows; the fused kernel from
64 rows and dequant+cuBLAS from 192 rows. Each round submits one request per (engine,
length) in rotated order, so clocks drift onto every arm equally. The first 32 generated
tokens are compared across arms: the paths differ only in rounding, so a difference is
reported as the first divergent position rather than a failure.

    python -m benchmarks.inference.awq_prefill_path_ab --out results/rtx4060/awq_prefill_path_ab_4b.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

ARMS = {
    "tinygemm": (None, None),
    "fused_from_64": (64, None),
    "fused_64_dequant_192": (64, 192),
}


def first_divergence(left, right):
    return next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen3-4B-AWQ")
    parser.add_argument("--prompts", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from benchmarks.inference.single_request_compare import prompt_of
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_awq_tinygemm_model
    from engine.quantization.awq_tinygemm import install_int4pack_prefill
    from engine.runtime import GenerationRequest

    loaded = load_awq_tinygemm_model(args.model)
    prompts = {n: loaded.tokenizer(prompt_of(loaded.tokenizer, n)).input_ids for n in args.prompts}
    blocks = (max(args.prompts) + args.new_tokens) // 16 + 24
    engine = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, loaded.device, num_blocks=blocks,
                                      block_size=16, max_active=1, prefix_cache_blocks=0,
                                      cuda_graph_batch_sizes=(1,), packed_fused_prefill=True,
                                      packed_prefill_width_buckets=True)
    engine.warmup()
    samples = {arm: {n: [] for n in args.prompts} for arm in ARMS}
    tokens = {arm: {} for arm in ARMS}

    def run(arm, n):
        fused, dequant = ARMS[arm]
        install_int4pack_prefill(loaded.model, fused, dequant)
        engine.reset()
        engine.eos_ids.clear()
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        request = GenerationRequest(f"{arm}-{n}", len(prompts[n]), args.new_tokens,
                                    prompt_token_ids=list(prompts[n]))
        request.created_ns = started
        assert engine.submit(request)
        while engine.has_unfinished_requests:
            engine.step()
        torch.cuda.synchronize()
        tokens[arm][n] = request.output_token_ids[:]
        return request.time_to_first_token_ms()

    for arm in ARMS:            # compile every path once before timing
        for n in args.prompts:
            run(arm, n)
    for round_index in range(args.rounds):
        order = list(ARMS)
        order = order[round_index % len(order):] + order[:round_index % len(order)]
        for arm in order:
            for n in args.prompts:
                samples[arm][n].append(run(arm, n))
    install_int4pack_prefill(loaded.model, 64, 192)   # leave the model on the default path

    report = {}
    for n in args.prompts:
        row = {}
        for arm in ARMS:
            values = samples[arm][n]
            row[arm] = {"ttft_p50_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values),
                        "spread": (max(values) - min(values)) / statistics.median(values)}
        base = row["tinygemm"]["ttft_p50_ms"]
        row["speedup_vs_tinygemm"] = {arm: base / row[arm]["ttft_p50_ms"] for arm in ARMS}
        row["first_divergence_vs_tinygemm"] = {arm: first_divergence(tokens[arm][n], tokens["tinygemm"][n])
                                               for arm in ARMS if arm != "tinygemm"}
        report[str(len(prompts[n]))] = row
        print(f"prompt {len(prompts[n]):5d}: " + "  ".join(
            f"{arm}={row[arm]['ttft_p50_ms']:7.0f}ms({row['speedup_vs_tinygemm'][arm]:.2f}x)" for arm in ARMS)
            + f"  divergence {row['first_divergence_vs_tinygemm']}", flush=True)
    Path(args.out).write_text(json.dumps({"benchmark": "awq_prefill_path_ab", "model": args.model,
                                          "gpu": torch.cuda.get_device_name(), "args": vars(args),
                                          "arms": {k: {"fused_min_rows": v[0], "dequant_min_rows": v[1]} for k, v in ARMS.items()},
                                          "rows": report}, indent=2))
    del engine
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
