"""Equal-VRAM A/B: FP16 Qwen3-1.7B against the stacked low-precision configuration.

Stacked = W8A16 MLP (searched dispatch) + W8A16 lm_head + INT8 KV with calibrated key
smoothing + tiled INT8 prefill + packed fused prefill rows. Both arms get the same
budget for weights plus KV pool, so the stacked arm's smaller weights and half-size KV
turn into ~2.7x the KV blocks. Each arm reloads its model per run (two 1.7B instances
plus pools do not fit 8 GB together); arm order alternates per repeat. Workloads, page
audit, token gate and summaries reuse `int8_kv_pressure_ab`.

    python -m benchmarks.batching.stacked_vs_fp16_ab --profile chat --out results/rtx4060/stacked_vs_fp16_chat_1.7b.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from benchmarks.batching.int8_kv_pressure_ab import (
    PROFILES, calibration_batches, kv_bytes, make_schedule, page_audit, run_workload,
    stock_reference, summarize, token_gate,
)

PROFILES = dict(PROFILES)
# Long prompts at eight concurrent requests: FP16 cannot hold them in the budget.
PROFILES["long_c8"] = {"requests": 12, "prompt": (3072, 4096), "new": (96, 160), "max_active": 8,
                       "arrival_rate": 1.5}


def build(arm: str, model_name: str, budget_bytes: int, max_active: int, *,
          w8a16: bool = True, int8_decode_tuned: bool = False):
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_awq_tinygemm_model, load_model
    from engine.quantization.int8 import model_storage_bytes

    awq = "awq" in model_name.lower()
    loaded = load_awq_tinygemm_model(model_name) if awq else load_model(model_name, dtype="float16")
    # Both AWQ arms take the accepted 4B scheduling settings; only the KV dtype differs.
    kwargs = dict(packed_fused_prefill=True, packed_prefill_width_buckets=True) if awq else {}
    if arm == "stacked" and awq:
        from engine.quantization.kv_smoothing import calibrate_key_smoothing

        kwargs.update(kv_cache_dtype="int8", int8_prefill_attention="tiled",
                      int8_decode_tuned=int8_decode_tuned,
                      kv_key_smoothing=calibrate_key_smoothing(loaded.model, calibration_batches(loaded.tokenizer)))
    elif arm == "stacked":
        from engine.kernels.w8a16_linear import quantize_mlp_w8a16
        from engine.kernels.w8a16_tuned import quantize_lm_head_w8a16
        from engine.quantization.kv_smoothing import calibrate_key_smoothing

        if w8a16:
            quantize_mlp_w8a16(loaded.model)
            quantize_lm_head_w8a16(loaded.model)
        torch.cuda.empty_cache()
        kwargs = dict(kv_cache_dtype="int8", int8_prefill_attention="tiled",
                      int8_decode_tuned=int8_decode_tuned,
                      kv_key_smoothing=calibrate_key_smoothing(loaded.model, calibration_batches(loaded.tokenizer)),
                      packed_fused_prefill=True)
    weights = model_storage_bytes(loaded.model)
    config = loaded.model.config
    if awq and arm == "fp16":
        kwargs["packed_fused_prefill"] = True
    per_block = kv_bytes(config, 1, 16, arm == "stacked")
    blocks = int((budget_bytes - weights) // per_block)
    buckets = tuple(size for size in (1, 2, 4, 8) if size <= max_active)
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, loaded.device, num_blocks=blocks, block_size=16,
        max_active=max_active, prefix_cache_blocks=0, cuda_graph_batch_sizes=buckets, **kwargs,
    )
    return loaded, engine, weights, blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--profile", default="chat", choices=sorted(PROFILES))
    parser.add_argument("--budget-gib", type=float, default=5.0, help="weights + KV pool per arm")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-w8a16", action="store_true",
                        help="stacked arm keeps FP16 weights (INT8 KV pipeline only)")
    parser.add_argument("--int8-decode-tuned", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from engine.model import load_model

    profile = PROFILES[args.profile]
    budget = int(args.budget_gib * 2**30)
    reference_model = load_model(args.model, dtype="float16")
    schedule = make_schedule(reference_model.tokenizer, profile, args.seed)
    reference = stock_reference(reference_model, schedule, Path(args.out).with_suffix(".stock_reference.json"))
    del reference_model
    gc.collect()
    torch.cuda.empty_cache()

    arms = ("fp16", "stacked")
    runs = {arm: [] for arm in arms}
    audits = {arm: [] for arm in arms}
    outputs, meta = {}, {}
    for repeat in range(args.repeats):
        for arm in (arms if repeat % 2 == 0 else arms[::-1]):
            loaded, engine, weights, blocks = build(arm, args.model, budget, profile["max_active"],
                                                    w8a16=not args.no_w8a16,
                                                    int8_decode_tuned=args.int8_decode_tuned)
            started = time.perf_counter()
            engine.warmup()
            warm_s = time.perf_counter() - started
            raw = run_workload(engine, schedule)
            audit = page_audit(engine, raw["requests"])
            summary = summarize(raw, engine)
            summary.update(weights_bytes=weights, num_blocks=blocks, warmup_s=warm_s,
                           kv_pool_bytes=kv_bytes(loaded.model.config, blocks, 16, arm == "stacked"))
            runs[arm].append(summary)
            audits[arm].append(audit)
            if repeat == 0:
                outputs[arm] = [r.output_token_ids[:] for r in raw["requests"]]
                meta[arm] = {"weights_bytes": weights, "num_blocks": blocks,
                             "kv_tokens": blocks * 16}
            print(f"[{arm} r{repeat}] blocks={blocks} tok/s={summary['throughput_tok_s']:.0f} "
                  f"itl p50/p99={summary['itl_ms']['p50']:.1f}/{summary['itl_ms']['p99']:.1f} "
                  f"ttft p50/p99={summary['ttft_ms']['p50']:.0f}/{summary['ttft_ms']['p99']:.0f} "
                  f"preempt={summary['preemptions']} spills={summary['spill_events']} "
                  f"audit={'ok' if not audit else audit[:2]}", flush=True)
            del loaded, engine, raw
            gc.collect()
            torch.cuda.empty_cache()

    def stat(arm, path):
        values = []
        for run in runs[arm]:
            value = run
            for part in path.split("."):
                value = value[part]
            values.append(value)
        median = statistics.median(values)
        return {"median": median, "min": min(values), "max": max(values),
                "spread": (max(values) - min(values)) / median if median else 0.0}

    metrics = ("throughput_tok_s", "itl_ms.p50", "itl_ms.p95", "itl_ms.p99", "ttft_ms.p50",
               "ttft_ms.p95", "ttft_ms.p99", "queue_total_ms.p95", "preemptions", "recomputed_tokens",
               "spill_events", "plain_decode_step_ms.p50", "prefill_step_ms.p50", "peak_allocated_bytes")
    medians = {arm: {m: stat(arm, m) for m in metrics} for arm in arms}
    for m in metrics:
        base, var = medians["fp16"][m], medians["stacked"][m]
        change = (var["median"] - base["median"]) / base["median"] if base["median"] else float("nan")
        noise = max(base["spread"], var["spread"])
        verdict = "resolved" if abs(change) > noise else "unresolved"
        print(f"  {m:28s} fp16={base['median']:.4g} stacked={var['median']:.4g} change={change:+.1%} ({verdict}, spread {noise:.1%})")
    gates = {arm: token_gate(outputs[arm], reference, 8) for arm in arms}
    for arm in arms:
        print(f"  token gate {arm}: prefix agreement {gates[arm]['prefix_agreement']:.1%}, "
              f"early non-tie {gates[arm]['early_non_tie_divergence']}")
    payload = {"benchmark": "stacked_vs_fp16_ab", "model": args.model, "profile": args.profile,
               "profile_config": profile, "budget_bytes": budget, "gpu": torch.cuda.get_device_name(),
               "arms_meta": meta, "runs": runs, "medians": medians, "audits": audits,
               "token_gate_vs_stock": gates, "order": "alternating per repeat, model reloaded per run",
               "stacked_w8a16": not args.no_w8a16, "int8_decode_tuned": args.int8_decode_tuned}
    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(f"Saved -> {args.out}")
    return 1 if any(any(a) for a in audits.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
