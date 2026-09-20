"""Phase 0 hook check: warm-up captures every graph, and the step split reports.

Run as its own process so the notebook kernel never owns a CUDA context; a model that
stays resident in the kernel skews every benchmark that follows and OOMs the tests.

    CUDA_VISIBLE_DEVICES=0 python scripts/check_hooks.py
"""

from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--max-active", type=int, default=8)
    parser.add_argument("--buckets", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from benchmarks.common import device_clock_record, environment_record, git_record
    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    loaded = load_model(args.model)
    engine = ContinuousBatchingEngine(
        loaded.model, loaded.tokenizer, loaded.device, num_blocks=args.num_blocks,
        max_active=args.max_active, cuda_graph_batch_sizes=tuple(args.buckets),
    )
    started = time.perf_counter()
    summary = engine.warmup()
    warmup_s = time.perf_counter() - started
    captured = sorted(engine._decode_graphs)
    print(f"warmup {warmup_s:.1f}s -> {summary}")
    print(f"captured: {captured}")
    expected = 2 * len(args.buckets)
    problems = []
    if summary["graphs"] != expected:
        problems.append(f"expected {expected} graphs (every bucket x block_n), got {summary['graphs']}")
    if not summary["prefill_sdpa_calls"] or not summary["prefill_chunked_calls"]:
        problems.append(f"both prefill paths must run during warmup: {summary}")

    engine.instrument = True
    prompts = ["Explain KV caching in one sentence."] * args.max_active
    engine.generate(prompts, max_new_tokens=8)
    timing = {key: round(value, 3) for key, value in engine.last_step_timing.items()}
    print(f"last step phases (ms): {timing}")
    missing = {"host_stage_ms", "decode_gpu_ms", "sync_ms"} - set(timing)
    if missing:
        problems.append(f"step split missing phases: {sorted(missing)}")

    record = {
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
