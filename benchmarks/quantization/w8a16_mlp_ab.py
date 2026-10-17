"""End-to-end FP16 versus selective MLP W8A16 on a real checkpoint.

The projection gate excludes attention matrices because W8A16 lost there on RTX 4060.
Both arms use separate model instances, decode CUDA graphs, eager prefill, identical
closed-loop workloads, and alternating execution order.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch


PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "Explain paged attention in one paragraph.",
    "Write a short story about a lighthouse keeper.",
]


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    return next((index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
                None if len(left) == len(right) else min(len(left), len(right)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-active", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=831)
    parser.add_argument("--prompt-profile", default=None,
                        help="a benchmarks.reliability.ab PROMPT_PROFILES name (default: the "
                             "original short 16-224-token mix)")
    parser.add_argument("--prefill-graphs", action="store_true",
                        help="capture prefill/fused graphs too (the current engine default)")
    parser.add_argument("--lm-head", action="store_true", help="W8 arm also quantizes lm_head")
    parser.add_argument("--attention", action="store_true", help="W8 arm also quantizes q/k/v/o")
    parser.add_argument("--packed", action="store_true", help="packed fused prefill rows in both arms")
    parser.add_argument("--output", default="results/rtx4060/w8a16_mlp_full_engine_b1.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if args.concurrency > args.max_active:
        parser.error("concurrency must not exceed max-active")

    from benchmarks.common import device_clock_record, environment_record, git_record
    from benchmarks.reliability.soak import SoakConfig, run_interleaved
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.kernels.w8a16_linear import quantize_mlp_w8a16
    from engine.model import load_model
    from engine.quantization.int8 import model_storage_bytes

    print("Loading independent FP16 and selective-W8 model instances...")
    fp16 = load_model(args.model, dtype="float16")
    w8 = load_model(args.model, dtype="float16")
    fp16_bytes = model_storage_bytes(fp16.model)
    replaced = quantize_mlp_w8a16(w8.model)
    if args.lm_head:
        from engine.kernels.w8a16_tuned import quantize_lm_head_w8a16
        quantize_lm_head_w8a16(w8.model)
    if args.attention:
        from engine.kernels.w8a16_tuned import quantize_attention_w8a16
        replaced += len(quantize_attention_w8a16(w8.model))
    torch.cuda.empty_cache()
    w8_bytes = model_storage_bytes(w8.model)

    def reference(model, tokenizer):
        outputs = []
        with torch.inference_mode():
            for prompt in PROMPTS:
                ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
                generated = model.generate(
                    ids, max_new_tokens=24, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                outputs.append(generated[0, ids.shape[1]:].tolist())
        return outputs

    fp16_tokens = reference(fp16.model, fp16.tokenizer)
    w8_tokens = reference(w8.model, w8.tokenizer)
    divergence = [_first_divergence(a, b) for a, b in zip(fp16_tokens, w8_tokens)]
    print(f"W8 replacements={replaced}; first divergence vs FP16={divergence}")
    if any(index is not None and index < 8 for index in divergence):
        raise RuntimeError(f"W8A16 diverged from FP16 before the 8-token gate: {divergence}")

    buckets = tuple(size for size in (1, 2, 4, 8, 16) if size <= args.max_active)
    common = dict(
        num_blocks=args.num_blocks, max_active=args.max_active, prefix_cache_blocks=0,
        cuda_graph_batch_sizes=buckets, prefill_cuda_graphs=args.prefill_graphs,
        packed_fused_prefill=args.packed,
    )
    loaded_by_arm = {"fp16": fp16, "w8a16_mlp": w8}

    def maker(label: str):
        loaded = loaded_by_arm[label]
        return ContinuousBatchingEngine(
            loaded.model, loaded.tokenizer, loaded.device, **common,
        )

    makers = {label: (lambda label=label: maker(label)) for label in loaded_by_arm}
    workload = dict(prefix_tokens=(16, 128), suffix_tokens=(4, 96), max_new_tokens=(8, 64))
    if args.prompt_profile:
        from benchmarks.reliability.ab import PROMPT_PROFILES
        workload = dict(PROMPT_PROFILES[args.prompt_profile])
    config = SoakConfig(
        duration_s=args.duration, concurrency=args.concurrency, seed=args.seed,
        cancel_probability=0.0, instrument=True, **workload,
    )
    clocks_before = device_clock_record()

    def report(label, offset, run):
        timing = run.step_timing
        print(
            f"[{label} {offset + 1}/{args.repeats}] "
            f"ITL={run.latency.get('itl_p50', 0):.2f} ms "
            f"decode_gpu={timing.get('decode_gpu_ms_p50', 0):.2f} ms "
            f"TTFT={run.latency.get('ttft_p50', 0):.1f} ms"
        )

    results = run_interleaved(makers, config, repeats=args.repeats, on_run=report)
    clocks_after = device_clock_record()
    metrics = ("latency.itl_p50", "latency.ttft_p50", "step_timing.decode_gpu_ms_p50",
               "step_timing.decode_step_p50_ms", "step_timing.prefill_gpu_ms_p50",
               "step_timing.expected_gap_ms", "latency.itl_p99", "latency.itl_p999",
               "latency.ttft_p99", "step_timing.prefill_step_p50_ms",
               "step_timing.fused_gpu_ms_p50", "delivered_tokens",
               "mean_decode_batch", "mean_context_tokens")
    summaries = {
        label: {metric: repeated.summary(metric) for metric in metrics}
        for label, repeated in results.items()
    }
    for metric in metrics[:-2]:
        base = summaries["fp16"][metric].get("median", 0)
        variant = summaries["w8a16_mlp"][metric].get("median", 0)
        change = (variant / base - 1) if base else float("nan")
        noise = max(summaries["fp16"][metric].get("spread", 0), summaries["w8a16_mlp"][metric].get("spread", 0))
        print(f"{metric}: FP16={base:.3f}, W8={variant:.3f}, change={change:+.1%} "
              f"({'resolved' if abs(change) > noise else 'unresolved'}; spread {noise:.1%})")

    payload = {
        "config": vars(args), "model_storage_bytes": {"fp16": fp16_bytes, "w8a16_mlp": w8_bytes},
        "replaced_projections": replaced, "first_divergence_vs_fp16": divergence,
        "token_gate_passed": not any(i is not None and i < 8 for i in divergence),
        "environment": environment_record(fp16.device), "git": git_record(),
        "clocks_before": clocks_before, "clocks_after": clocks_after,
        "summaries": summaries,
        "arms": {label: repeated.to_dict() for label, repeated in results.items()},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"Saved -> {output}")
    del results, fp16, w8
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
