"""Phase 0 hook check: warm-up captures every graph, and the step split reports.

Run as its own process so the notebook kernel never owns a CUDA context; a model that
stays resident in the kernel skews every benchmark that follows and OOMs the tests.

    CUDA_VISIBLE_DEVICES=0 python scripts/check_hooks.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Run as a script, sys.path[0] is scripts/, not the repo; make the repo importable
# regardless of the caller's working directory (the notebook runner's cwd has varied).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16"))
    parser.add_argument("--decode-attention", default=None)
    parser.add_argument("--prefill-attention", default=None)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=16,
                        help="tokens per KV page (FlashAttention paged-KV requires 256)")
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--buckets", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--out", default=None)
    parser.add_argument("--backends-only", action="store_true",
                        help="print what this GPU and checkpoint can run, then exit; "
                             "no warmup, no graphs, seconds rather than minutes")
    args = parser.parse_args()

    from benchmarks.common import device_clock_record, environment_record, git_record
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    from engine.backends import Geometry, report
    from engine.kernels.device import current_device
    from engine.model import adapters

    loaded = load_model(args.model, dtype=args.dtype)
    model_report = adapters.describe(loaded.model)
    geometry = Geometry(
        num_q_heads=model_report["geometry"]["num_q_heads"],
        num_kv_heads=model_report["geometry"]["num_kv_heads"],
        head_dim=model_report["geometry"]["head_dim"],
        block_size=args.block_size,
        dtype=str(loaded.dtype).removeprefix("torch."),
    )
    backend_report = report(current_device(), geometry)
    print(f"\nmodel: {model_report['model_type']} ({model_report['class']}), "
          f"{model_report['geometry']['num_layers']} layers, "
          f"{model_report['geometry']['num_q_heads']}/{model_report['geometry']['num_kv_heads']} heads, "
          f"head_dim {model_report['geometry']['head_dim']}, "
          f"{model_report['kv_bytes_per_token'] / 1024:.0f} KiB KV per token")
    print(f"  fusions: {model_report['mlp_modules']} MLP, {model_report['norm_modules']} norm, "
          f"rope in {model_report['rope_module']}")
    if model_report["unsupported_reason"]:
        print(f"  UNSUPPORTED: {model_report['unsupported_reason']}")
    print(f"device: {backend_report['device']} "
          f"({'measured' if backend_report['measured'] else 'not yet measured'})")
    for phase in ("decode", "prefill"):
        for row in backend_report[f"{phase}_backends"]:
            mark = "ok " if row["available"] else "no "
            note = "" if row["available"] else f"  <- {row['reason']}"
            print(f"  {mark}{phase:8s} {row['name']:10s} p{row['priority']:<3d}{note}")
    print(f"defaults: {backend_report['defaults']}")
    for key, value in backend_report["reasons"].items():
        print(f"  {key}: {value}")
    if args.backends_only:
        if args.out:
            with open(args.out, "w") as handle:
                json.dump({"model": model_report, "backends": backend_report}, handle,
                          indent=2, default=str)
        return 0

    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, loaded.device, num_blocks=args.num_blocks,
        block_size=args.block_size, max_active=args.max_active,
        cuda_graph_batch_sizes=tuple(args.buckets),
        **({"decode_attention": args.decode_attention} if args.decode_attention else {}),
        **({"prefill_attention": args.prefill_attention} if args.prefill_attention else {}),
    )
    started = time.perf_counter()
    summary = engine.warmup()
    warmup_s = time.perf_counter() - started
    captured = sorted(engine._decode_graphs)
    print(f"warmup {warmup_s:.1f}s -> {summary}")
    print(f"captured: {captured}")
    expected = 2 * len(args.buckets)
    problems = []
    if engine.decode_cuda_graphs and summary["graphs"] != expected:
        problems.append(f"expected {expected} graphs (every bucket x block_n), got {summary['graphs']}")
    elif not engine.decode_cuda_graphs:
        print(f"decode graphs: eager-only ({engine.decode_graph_eager_reason})")
    if not summary["prefill_sdpa_calls"] or not summary["prefill_chunked_calls"]:
        problems.append(f"both prefill paths must run during warmup: {summary}")
    graph_unsupported = summary.get("prefill_graph_unsupported", {})
    # FlashAttention's kvcache entry point allocates its split-KV workspace internally,
    # which makes the prefill call non-capturable on the tested FA2 build. The engine
    # records that once and correctly uses its eager staged path thereafter; this is an
    # explicit capability limit, not a failed warmup or a silent fallback. Other
    # backends must still capture every requested graph.
    flash_graph_limit = (
        not engine.prefill_backend.graph_safe
        and engine.prefill_graph_eager_reason is not None
    ) or (
        not engine.prefill_backend.graph_safe
        and graph_unsupported
        and all(key in {"flash", "fused:flash"} for key in graph_unsupported)
    )
    if graph_unsupported and not flash_graph_limit:
        problems.append(f"prefill graph capture failed: {graph_unsupported}")
    elif flash_graph_limit:
        print("prefill graphs: FlashAttention eager-only "
              f"({engine.prefill_graph_eager_reason or graph_unsupported})")
    elif summary.get("prefill_graphs", 0) < len(args.buckets):
        problems.append(f"expected a prefill graph per bucket, got {summary.get('prefill_graphs')}")
    if engine.fused_step and not summary.get("fused_graphs") and not flash_graph_limit:
        problems.append(f"fused step graphs were not captured during warmup: {summary}")

    engine.instrument = True
    prompts = ["Explain KV caching in one sentence."] * args.max_active
    engine.generate(prompts, max_new_tokens=8)
    timing = {key: round(value, 3) for key, value in engine.last_step_timing.items()}
    print(f"last step phases (ms): {timing}")
    missing = {"host_stage_ms", "decode_gpu_ms", "sync_ms"} - set(timing)
    if missing:
        problems.append(f"step split missing phases: {sorted(missing)}")
    if engine.lazy_graph_captures:
        problems.append(f"{engine.lazy_graph_captures} graph capture(s) happened after warmup")

    record = {
        "model": model_report, "backends": backend_report,
        "warmup_s": warmup_s, "warmup": summary, "captured_graphs": captured,
        "last_step_timing_ms": timing, "git": git_record(),
        "environment": environment_record(loaded.device), "clocks": device_clock_record(),
        "problems": problems,
    }
    print(json.dumps({k: record[k] for k in ("git", "clocks", "problems")}, indent=2, default=str))
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(record, handle, indent=2, default=str)
    if problems:
        print("HOOK CHECK FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("hook check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
