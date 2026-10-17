"""One active request: naive explicit decode vs the old engine vs the current engine.

Run once per (arm, model) in its own process. `engine` is imported from whatever tree is
first on PYTHONPATH, so the same script measures the committed HEAD engine (exported with
`git archive` into a scratch directory) and the working tree. Prompts and calibration text
come from committed files (`git show HEAD:`), so every arm sees identical inputs.

Arms:
  naive              ExplicitDecodeRunner: stock HF forward + DynamicCache, one token per call
  engine_default     ContinuousBatchingEngine defaults, max_active=1, decode graph bucket 1
  engine_optimized   the per-model optimized configuration (current tree only)

Fixed work: EOS ignored, 128 new tokens, greedy. Reports TTFT, ITL (p50 of per-token gaps),
decode tokens/s, end-to-end time and peak memory, per prompt length.

    PYTHONPATH=<tree> python benchmarks/inference/single_request_compare.py \
        --arm engine_default --model Qwen/Qwen3-0.6B --out results/rtx4060/single_request/<file>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def committed_text(path: str) -> str:
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout


def prompt_of(tokenizer, tokens: int) -> str:
    ids = tokenizer(committed_text("docs/rtx4060-final-evaluation.md") * 4).input_ids
    return tokenizer.decode(ids[200:200 + tokens])


def load(model_name: str, awq_triton: bool):
    if "AWQ" in model_name:
        from engine.model import load_awq_tinygemm_model
        try:
            return load_awq_tinygemm_model(model_name, triton_prefill_rows=64 if awq_triton else None)
        except TypeError:  # an engine version without the Triton prefill option
            return load_awq_tinygemm_model(model_name)
    from engine.model import load_model
    return load_model(model_name, dtype="float16")


def optimized_engine_kwargs(loaded, model_name: str) -> dict:
    """Per-model optimized configuration from docs/rtx4060-rnd-priorities.md."""
    kwargs = dict(packed_fused_prefill=True)
    if "AWQ" in model_name:
        kwargs["packed_prefill_width_buckets"] = True  # enables the measured AWQ prefill graphs
        return kwargs
    if "1.7B" in model_name:
        from engine.kernels.w8a16_linear import quantize_mlp_w8a16
        from engine.kernels.w8a16_tuned import quantize_lm_head_w8a16
        quantize_mlp_w8a16(loaded.model)
        quantize_lm_head_w8a16(loaded.model)
        torch.cuda.empty_cache()
    from engine.quantization.kv_smoothing import calibrate_key_smoothing
    text = committed_text("tests/batching/test_continuous_batching.py") + committed_text("README.md")
    ids = loaded.tokenizer(text).input_ids
    batches = [ids[i:i + 2048] for i in range(0, min(len(ids) - 2048, 4 * 2048), 2048)]
    kwargs.update(kv_cache_dtype="int8", int8_prefill_attention="tiled", int8_decode_tuned=True,
                  kv_key_smoothing=calibrate_key_smoothing(loaded.model, batches))
    return kwargs


def run_naive(loaded, prompt: str, new_tokens: int, runs: int) -> list[dict]:
    from engine.model import ExplicitDecodeRunner

    runner = ExplicitDecodeRunner(loaded.model, loaded.tokenizer, loaded.device)
    runner.generate(prompt, max_new_tokens=8, eos_token_id=-1)
    results = []
    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()
        result = runner.generate(prompt, max_new_tokens=new_tokens, eos_token_id=-1)
        results.append({"ttft_ms": result.metrics.ttft_ms, "itl_ms": list(result.metrics.decode_ms),
                        "end_to_end_ms": result.metrics.total_ms, "tokens": len(result.token_ids),
                        "peak_allocated": torch.cuda.max_memory_allocated()})
    return results


def run_engine(engine, tokenizer, prompt: str, new_tokens: int, runs: int) -> list[dict]:
    from engine.runtime import GenerationRequest

    results = []
    for run in range(runs + 1):  # first run warms the prompt-length-specific paths
        engine.reset()
        engine.eos_ids.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        ids = tokenizer(prompt).input_ids
        request = GenerationRequest(request_id=f"single{run}", prompt_token_count=len(ids),
                                    max_new_tokens=new_tokens, prompt_token_ids=ids)
        tokenized = time.perf_counter()
        request.created_ns = time.perf_counter_ns() - int((tokenized - started) * 1e9)
        assert engine.submit(request)
        while engine.has_unfinished_requests:
            engine.step()
        torch.cuda.synchronize()
        total = (time.perf_counter() - started) * 1000
        stamps = request.token_timestamps_ns
        if run:
            results.append({"ttft_ms": request.time_to_first_token_ms(),
                            "itl_ms": [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])],
                            "end_to_end_ms": total, "tokens": len(request.output_token_ids),
                            "peak_allocated": torch.cuda.max_memory_allocated()})
    return results


def summarize(results: list[dict], prompt_tokens: int) -> dict:
    gaps = [g for r in results for g in r["itl_ms"]]
    itl = statistics.median(gaps)
    return {"prompt_tokens": prompt_tokens, "runs": len(results),
            "ttft_ms": statistics.median(r["ttft_ms"] for r in results),
            "itl_p50_ms": itl, "itl_mean_ms": statistics.fmean(gaps),
            "decode_tok_s": 1000 / itl,
            "end_to_end_ms": statistics.median(r["end_to_end_ms"] for r in results),
            "end_to_end_tok_s": statistics.median(r["tokens"] / (r["end_to_end_ms"] / 1000) for r in results),
            "peak_allocated_gib": max(r["peak_allocated"] for r in results) / 2**30}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("naive", "engine_default", "engine_optimized"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", default=None, help="name recorded for this tree (e.g. old_head)")
    parser.add_argument("--prompts", type=int, nargs="+", default=[128, 2048])
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import engine as engine_package

    loaded = load(args.model, awq_triton=args.arm != "naive")
    prompts = {n: prompt_of(loaded.tokenizer, n) for n in args.prompts}
    rows = {}
    if args.arm == "naive":
        for n, text in prompts.items():
            rows[str(n)] = summarize(run_naive(loaded, text, args.new_tokens, args.runs),
                                     len(loaded.tokenizer(text).input_ids))
    else:
        from engine.batching.continuous_batching import ContinuousBatchingEngine

        kwargs = optimized_engine_kwargs(loaded, args.model) if args.arm == "engine_optimized" else {}
        blocks = (max(args.prompts) + args.new_tokens) // 16 + 24
        engine = ContinuousBatchingEngine(loaded.model, loaded.tokenizer, loaded.device,
                                          num_blocks=blocks, block_size=16, max_active=1,
                                          prefix_cache_blocks=0, cuda_graph_batch_sizes=(1,), **kwargs)
        warm_started = time.perf_counter()
        engine.warmup()
        warmup_s = time.perf_counter() - warm_started
        for n, text in prompts.items():
            rows[str(n)] = summarize(run_engine(engine, loaded.tokenizer, text, args.new_tokens, args.runs),
                                     len(loaded.tokenizer(text).input_ids))
        rows["warmup_s"] = warmup_s
        rows["lazy_graph_captures"] = engine.lazy_graph_captures
        rows["engine_settings"] = {k: (str(v) if isinstance(v, torch.Tensor) else v)
                                   for k, v in kwargs.items() if k != "kv_key_smoothing"}
        rows["engine_settings"]["prefill_attention"] = engine.prefill_attention
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm": args.arm, "label": args.label or args.arm, "model": args.model,
                               "engine_tree": str(Path(engine_package.__file__).parents[1]),
                               "gpu": torch.cuda.get_device_name(), "new_tokens": args.new_tokens,
                               "results": rows}, indent=2))
    for n in args.prompts:
        r = rows[str(n)]
        print(f"{args.label or args.arm:18s} {args.model:22s} prompt {r['prompt_tokens']:5d}: "
              f"TTFT {r['ttft_ms']:8.1f} ms  ITL {r['itl_p50_ms']:6.2f} ms  "
              f"decode {r['decode_tok_s']:6.1f} tok/s  e2e {r['end_to_end_ms']:7.0f} ms  "
              f"peak {r['peak_allocated_gib']:.2f} GiB", flush=True)
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
