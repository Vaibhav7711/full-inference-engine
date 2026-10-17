"""Mixed-arrival KV-pressure A/B: FP16 recompute versus INT8 exact host spill.

INT8 KV is only deployable if its preemption path is. A generated INT8 request that is
preempted spills its authoritative pages and scales to host and restores them on
readmission; an FP16 request recomputes its prompt plus generated prefix instead. This
benchmark drives both through the same seeded, wall-clock arrival schedule with a pool
deliberately smaller than peak demand, so preemption really happens, and reports:

- ITL p50/p95/p99/p999/max (all requests, and preempted requests alone), TTFT, total
  queue time (including post-yield waits) and stall time;
- preemptions, spill events/bytes/write/restore milliseconds, recomputed tokens;
- the steps that carried a spill or restore, so their wall time is attributed directly;
- throughput, KV-pool bytes and peak allocated device memory;
- the post-drain page audit and greedy-token agreement against a stock-Transformers
  reference, with top-2 margins so fp16 ties are not miscounted as divergence.

Arms (interleaved, order rotated each repeat):
  fp16_roomy / int8_roomy   pool covers peak demand; no preemption (the control)
  fp16_pressure             pool = --pressure x peak demand
  int8_equal_blocks         same block count as fp16_pressure (same preemption pressure)
  int8_equal_bytes          same KV bytes as fp16_pressure (~1.97x blocks: the capacity case)
  int8s_*                   the INT8 arms with calibrated static key smoothing
  int8st_*                  smoothing plus the tensor-core tiled INT8 prefill kernel
  int8stD_*                 int8st plus the offline-searched INT8 decode launch table

    python -m benchmarks.batching.int8_kv_pressure_ab --profile chat --repeats 3
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch

from engine.runtime import GenerationRequest, RequestState

PROFILES: dict[str, dict] = {
    # Chat-sized prompts with enough generation that requests are mid-decode when the
    # pool runs out, which is the only state in which INT8 spills rather than recomputes.
    "chat": {"requests": 32, "prompt": (256, 1024), "new": (64, 192), "max_active": 8,
             "arrival_rate": 4.0},
    # Every decode batch here exceeds the 24 MB L2 per layer in FP16 (2 rows x 4k x 4 KB).
    "long": {"requests": 10, "prompt": (3072, 4096), "new": (96, 160), "max_active": 4,
             "arrival_rate": 1.0},
    # Long generations from a near-simultaneous burst: every admitted request is decoding
    # when the pool runs out, so the LIFO victim has generated tokens and INT8 spills.
    # In "long" the victim is almost always still prefilling and simply recomputes.
    # Single-stream long context: every decode step is the batch-1 latency-bound regime.
    "b1_long": {"requests": 6, "prompt": (2048, 4096), "new": (192, 256), "max_active": 1,
                "arrival_rate": 0.5},
    "long_decode": {"requests": 8, "prompt": (1536, 3072), "new": (384, 768), "max_active": 4,
                    "arrival_rate": 4.0},
}

ARMS = ("fp16_roomy", "fp16_roomy_chunked_idle", "fp16_pressure", "int8_equal_blocks", "int8_equal_bytes", "int8_roomy",
        "int8s_equal_blocks", "int8s_equal_bytes", "int8s_roomy",
        "int8st_equal_blocks", "int8st_equal_bytes", "int8st_roomy",
        "int8stD_equal_bytes", "int8stD_roomy")
TIE_MARGIN = 0.02


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _dist(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "p50": _percentile(values, 0.50), "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99), "p999": _percentile(values, 0.999),
        "max": max(values, default=0.0),
        "mean": statistics.fmean(values) if values else 0.0,
    }


def _committed_text(patterns: list[str]) -> str:
    """Concatenated `git HEAD` contents of tracked files matching the patterns.

    Reading HEAD rather than the working tree keeps the workload fixed while the engine
    sources under test are being edited; results record the commit.
    """
    import subprocess

    root = Path(__file__).resolve().parents[2]
    names = subprocess.run(["git", "ls-files", "--", *patterns], cwd=root, capture_output=True,
                           text=True, check=True).stdout.split()
    texts = []
    for name in sorted(names):
        shown = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=root, capture_output=True,
                               text=True, errors="ignore")
        if shown.returncode == 0:
            texts.append(shown.stdout)
    return "\n\n".join(texts)


def _corpus_ids(tokenizer) -> list[int]:
    """Natural text from this repository's committed docs and engine sources.

    Real prose and code give the reference ordinary top-2 margins; random token ids
    produce near-uniform logits and turn the token gate into a tie detector.
    """
    text = _committed_text(["docs/*.md", "engine/*.py", "engine/**/*.py"])
    return tokenizer(text, return_tensors="pt").input_ids[0].tolist()


def calibration_batches(tokenizer, sequences: int = 4, length: int = 2048) -> list[list[int]]:
    """Held-out calibration text: tests, benchmarks, scripts and README.

    Disjoint from `_corpus_ids` (docs and engine sources), which the timed prompts use,
    so the token gate measures smoothing on text it was not calibrated on.
    """
    text = _committed_text(["tests/*.py", "tests/**/*.py", "benchmarks/*.py",
                            "benchmarks/**/*.py", "scripts/*.py", "README.md"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0].tolist()
    stride = max(length, (len(ids) - length) // max(1, sequences - 1))
    return [ids[start:start + length] for start in range(0, len(ids) - length + 1, stride)][:sequences]


def make_schedule(tokenizer, profile: dict, seed: int) -> list[dict]:
    """Seeded arrivals: Poisson inter-arrival times, prompt windows from the corpus."""
    corpus = _corpus_ids(tokenizer)
    rng = random.Random(seed)
    arrival = 0.0
    rows = []
    for index in range(profile["requests"]):
        length = rng.randint(*profile["prompt"])
        offset = rng.randrange(0, len(corpus) - length)
        rows.append({
            "index": index, "arrival_s": arrival,
            "prompt_ids": corpus[offset:offset + length],
            "max_new_tokens": rng.randint(*profile["new"]),
        })
        arrival += rng.expovariate(profile["arrival_rate"])
    return rows


def stock_reference(loaded, schedule: list[dict], cache_path: Path) -> dict:
    """Greedy continuations and top-2 margins from unpatched Transformers."""
    key = hashlib.sha256(json.dumps(
        [[row["prompt_ids"], row["max_new_tokens"]] for row in schedule]
    ).encode()).hexdigest()[:16]
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("key") == key:
            return cached
    from engine.kernels.rope import stock_rope

    model = loaded.model
    model.config._attn_implementation = "sdpa"
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = "sdpa"
    outputs, margins = [], []
    from transformers import DynamicCache

    # A manual greedy loop rather than `generate`: the engine arms ignore EOS, and
    # `min_new_tokens` would suppress the EOS logit and change the argmax instead.
    with stock_rope(), torch.inference_mode():
        for row in schedule:
            ids = torch.tensor([row["prompt_ids"]], device=loaded.device)
            cache = DynamicCache()
            logits = model(input_ids=ids, past_key_values=cache, use_cache=True).logits[0, -1]
            tokens, gaps = [], []
            for _ in range(row["max_new_tokens"]):
                top = torch.topk(logits.float(), 2).values
                gaps.append(float(top[0] - top[1]))
                token = int(torch.argmax(logits))
                tokens.append(token)
                logits = model(input_ids=torch.tensor([[token]], device=loaded.device),
                               past_key_values=cache, use_cache=True).logits[0, -1]
            outputs.append(tokens)
            margins.append(gaps)
    payload = {"key": key, "outputs": outputs, "margins": margins}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload))
    return payload


def _first_divergence(produced: list[int], expected: list[int]) -> int | None:
    index = next((i for i, (a, b) in enumerate(zip(produced, expected)) if a != b), None)
    if index is None and len(produced) != len(expected):
        index = min(len(produced), len(expected))
    return index


def token_gate(outputs: list[list[int]], reference: dict, min_identical: int) -> dict:
    """Per request: first divergence from stock, and whether it lands on a tie."""
    rows, early, ties, exact = [], [], [], 0
    agree = compared = 0
    for index, (produced, expected) in enumerate(zip(outputs, reference["outputs"])):
        first = _first_divergence(produced, expected)
        margin = (reference["margins"][index][first]
                  if first is not None and first < len(reference["margins"][index]) else None)
        tie = margin is not None and margin < TIE_MARGIN
        rows.append({"first_divergence": first, "margin": margin, "tie": tie})
        if first is None:
            exact += 1
        elif tie:
            ties.append(index)
        elif first < min_identical:
            early.append(index)
        # Agreement up to the first divergence: positions after it continue a different
        # greedy path and say nothing more about this arm.
        agree += first if first is not None else len(expected)
        compared += len(expected)
    return {"requests": rows, "exact_requests": exact, "tie_first_divergence": ties,
            "early_non_tie_divergence": early,
            "prefix_agreement": agree / max(compared, 1)}


def run_workload(engine, schedule: list[dict]) -> dict:
    """Drive one engine through the schedule in wall-clock time; return raw records."""
    engine.reset()
    engine.eos_ids.clear()  # fixed work: every request runs to its own length bound
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pending = list(schedule)
    submitted: list[GenerationRequest] = []
    steps: list[dict] = []
    peak_spill_live = 0
    started = time.perf_counter()
    while pending or engine.has_unfinished_requests:
        now = time.perf_counter() - started
        while pending and pending[0]["arrival_s"] <= now:
            row = pending.pop(0)
            request = GenerationRequest(
                request_id=f"r{row['index']}", prompt_token_count=len(row["prompt_ids"]),
                max_new_tokens=row["max_new_tokens"], prompt_token_ids=list(row["prompt_ids"]),
            )
            if not engine.submit(request):
                raise RuntimeError(f"scheduler rejected {request.request_id}")
            submitted.append(request)
        if not engine.has_unfinished_requests:
            time.sleep(max(0.0, pending[0]["arrival_s"] - (time.perf_counter() - started)))
            continue
        before = engine.recompute_report()
        preempt_before = engine.scheduler.preemption_count
        step_started = time.perf_counter()
        engine.step()
        step_ms = (time.perf_counter() - step_started) * 1000
        after = engine.recompute_report()
        spill_live = sum(
            tensor.numel() * tensor.element_size()
            for _, payload in engine._int8_preemption_spills.values() for tensor in payload
        )
        peak_spill_live = max(peak_spill_live, spill_live)
        steps.append({
            "ms": step_ms, "decode_rows": engine.last_step_decode_rows,
            "prefill_tokens": engine.last_step_prefill_tokens,
            "preemptions": engine.scheduler.preemption_count - preempt_before,
            "spill_events": after["int8_spill_events"] - before["int8_spill_events"],
            "spill_write_ms": after["int8_spill_write_ms"] - before["int8_spill_write_ms"],
            "spill_restore_ms": after["int8_spill_restore_ms"] - before["int8_spill_restore_ms"],
        })
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - started
    return {"requests": submitted, "steps": steps, "wall_s": wall_s,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_spill_live_bytes": peak_spill_live}


def page_audit(engine, requests: list[GenerationRequest]) -> list[str]:
    """Everything a drained engine must satisfy; any entry is a correctness failure."""
    problems = []
    reserved = len(engine._graph_dummy_blocks)
    snapshot = engine.block_manager.snapshot()
    if int(snapshot["used_blocks"]) != reserved:
        problems.append(f"page leak: {snapshot['used_blocks']} used blocks, {reserved} reserved")
    if int(snapshot["free_blocks"]) != int(snapshot["num_blocks"]) - reserved:
        problems.append(f"free list mismatch: {snapshot}")
    if engine.scheduler.active or engine.scheduler.waiting:
        problems.append("scheduler not drained")
    if engine._int8_preemption_spills:
        problems.append(f"{len(engine._int8_preemption_spills)} spill payloads left on host")
    for request in requests:
        if request.state is not RequestState.FINISHED:
            problems.append(f"{request.request_id} ended {request.state.name}: {request.finish_reason}")
        elif len(request.output_token_ids) != request.max_new_tokens:
            problems.append(f"{request.request_id} produced {len(request.output_token_ids)} "
                            f"of {request.max_new_tokens} tokens")
    return problems


def summarize(raw: dict, engine) -> dict:
    requests: list[GenerationRequest] = raw["requests"]
    gaps_all, gaps_preempted = [], []
    for request in requests:
        stamps = request.token_timestamps_ns
        gaps = [(b - a) / 1e6 for a, b in zip(stamps, stamps[1:])]
        gaps_all += gaps
        if request.preempted_count:
            gaps_preempted += gaps
    steps = raw["steps"]
    spill_steps = [s["ms"] for s in steps if s["spill_events"]]
    restore_steps = [s["ms"] for s in steps if s["spill_restore_ms"] > 0]
    preempt_steps = [s["ms"] for s in steps if s["preemptions"]]
    plain_decode = [s["ms"] for s in steps if not s["prefill_tokens"] and not s["preemptions"]
                    and not s["spill_restore_ms"]]
    report = engine.recompute_report()
    tokens = sum(len(request.output_token_ids) for request in requests)
    events = report["int8_spill_events"]
    return {
        "wall_s": raw["wall_s"], "output_tokens": tokens,
        "throughput_tok_s": tokens / raw["wall_s"],
        "steps": len(steps),
        "itl_ms": _dist(gaps_all), "itl_preempted_requests_ms": _dist(gaps_preempted),
        "ttft_ms": _dist([r.time_to_first_token_ms() or 0.0 for r in requests]),
        "queue_total_ms": _dist([r.total_queue_time_ms() or 0.0 for r in requests]),
        "stall_ms": _dist([r.stall_time_ms() for r in requests if r.preempted_count]),
        "stall_ms_total": sum(r.stall_time_ms() for r in requests),
        "preempted_requests": sum(1 for r in requests if r.preempted_count),
        "preemptions": report["preemptions"],
        "recomputed_tokens": report["recomputed_tokens"],
        "recompute_ms": report["recompute_ms"],
        "spill_events": events, "spill_bytes": report["int8_spill_bytes"],
        "spill_write_ms_total": report["int8_spill_write_ms"],
        "spill_restore_ms_total": report["int8_spill_restore_ms"],
        "spill_write_ms_per_event": report["int8_spill_write_ms"] / events if events else 0.0,
        "spill_restore_ms_per_event": report["int8_spill_restore_ms"] / events if events else 0.0,
        "spill_write_gb_s": (report["int8_spill_bytes"] / 1e9) / (report["int8_spill_write_ms"] / 1e3)
        if report["int8_spill_write_ms"] else 0.0,
        "peak_spill_live_bytes": raw["peak_spill_live_bytes"],
        "step_ms": _dist([s["ms"] for s in steps]),
        "plain_decode_step_ms": _dist(plain_decode),
        "prefill_step_ms": _dist([s["ms"] for s in steps if s["prefill_tokens"]]),
        "spill_step_ms": _dist(spill_steps), "restore_step_ms": _dist(restore_steps),
        "preempting_step_ms": _dist(preempt_steps),
        "peak_allocated_bytes": raw["peak_allocated_bytes"],
        "lazy_graph_captures": engine.lazy_graph_captures,
        "mean_decode_rows": statistics.fmean([s["decode_rows"] for s in steps if s["decode_rows"]])
        if any(s["decode_rows"] for s in steps) else 0.0,
    }


def kv_bytes(config, num_blocks: int, block_size: int, int8: bool) -> int:
    layers = int(config.num_hidden_layers)
    heads = int(config.num_key_value_heads)
    dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    per_token = layers * heads * 2 * ((dim + 2) if int8 else dim * 2)
    return num_blocks * block_size * per_token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--profile", default="chat", choices=sorted(PROFILES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pressure", type=float, default=0.5,
                        help="pressured FP16 pool as a fraction of peak token demand")
    parser.add_argument("--arrival-rate", type=float, default=None)
    parser.add_argument("--requests", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--min-identical-tokens", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from engine.batching.continuous_batching import ContinuousBatchingEngine
    from engine.model import load_model

    profile = dict(PROFILES[args.profile])
    if args.arrival_rate is not None:
        profile["arrival_rate"] = args.arrival_rate
    if args.requests is not None:
        profile["requests"] = args.requests
    arms = [arm for arm in args.arms.split(",") if arm]
    if any(arm not in ARMS for arm in arms):
        parser.error(f"arms must be drawn from {ARMS}")

    loaded = load_model(args.model, dtype="float16")
    config = loaded.model.config
    schedule = make_schedule(loaded.tokenizer, profile, args.seed)
    reference = stock_reference(
        loaded, schedule,
        Path(args.out).with_suffix(".stock_reference.json"),
    )

    smoothing = None
    if any(arm.startswith(("int8s_", "int8st")) for arm in arms):
        from engine.quantization.kv_smoothing import calibrate_key_smoothing

        calibration_started = time.perf_counter()
        smoothing = calibrate_key_smoothing(loaded.model, calibration_batches(loaded.tokenizer))
        calibration_s = time.perf_counter() - calibration_started
        print(f"key smoothing calibrated in {calibration_s:.2f}s; per-layer max factor "
              f"{[round(float(layer.max()), 1) for layer in smoothing[:4]]}...")

    block_size = 16
    max_active = profile["max_active"]
    buckets = tuple(size for size in (1, 2, 4, 8, 16) if size <= max_active)
    dummy = max(buckets) - 1
    longest = max(len(row["prompt_ids"]) + row["max_new_tokens"] for row in schedule)
    peak_tokens = sum(sorted(
        (len(row["prompt_ids"]) + row["max_new_tokens"] for row in schedule), reverse=True,
    )[:max_active])
    roomy_blocks = -(-peak_tokens // block_size) + 8
    pressure_blocks = max(-(-longest // block_size) + 1, int(args.pressure * peak_tokens / block_size))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    bytes_blocks = int(pressure_blocks * (2 * head_dim) / (head_dim + 2))
    arm_specs = {
        "fp16_roomy": ("fp16", roomy_blocks),
        "fp16_roomy_chunked_idle": ("fp16", roomy_blocks),
        "fp16_pressure": ("fp16", pressure_blocks),
        "int8_equal_blocks": ("int8", pressure_blocks),
        "int8_equal_bytes": ("int8", bytes_blocks),
        "int8_roomy": ("int8", roomy_blocks),
        "int8s_equal_blocks": ("int8", pressure_blocks),
        "int8s_equal_bytes": ("int8", bytes_blocks),
        "int8s_roomy": ("int8", roomy_blocks),
        "int8st_equal_blocks": ("int8", pressure_blocks),
        "int8st_equal_bytes": ("int8", bytes_blocks),
        "int8st_roomy": ("int8", roomy_blocks),
        "int8stD_equal_bytes": ("int8", bytes_blocks),
        "int8stD_roomy": ("int8", roomy_blocks),
    }
    print(f"profile {args.profile}: {len(schedule)} requests, peak demand {peak_tokens} tokens, "
          f"pressured pool {pressure_blocks} blocks, equal-bytes INT8 {bytes_blocks} blocks")

    results: dict[str, list[dict]] = {arm: [] for arm in arms}
    outputs: dict[str, list[list[int]]] = {}
    audits: dict[str, list[list[str]]] = {arm: [] for arm in arms}
    warmups: dict[str, list[float]] = {arm: [] for arm in arms}
    for repeat in range(args.repeats):
        order = arms[repeat % len(arms):] + arms[:repeat % len(arms)]
        for arm in order:
            kv, blocks = arm_specs[arm]
            engine = ContinuousBatchingEngine(
                loaded.model, loaded.tokenizer, loaded.device,
                num_blocks=blocks + dummy, block_size=block_size, max_active=max_active,
                prefix_cache_blocks=0, kv_cache_dtype=kv, decode_attention="per_head",
                cuda_graph_batch_sizes=buckets,
                kv_key_smoothing=smoothing if arm.startswith(("int8s_", "int8st")) else None,
                int8_prefill_attention="tiled" if arm.startswith("int8st") else "per_token",
                int8_decode_tuned=arm.startswith("int8stD_"),
                idle_prefill_tokens=None if arm.endswith("_chunked_idle") else 4096,
            )
            warm_started = time.perf_counter()
            engine.warmup()
            warmups[arm].append(time.perf_counter() - warm_started)
            raw = run_workload(engine, schedule)
            audit = page_audit(engine, raw["requests"])
            summary = summarize(raw, engine)
            summary["kv_pool_bytes"] = kv_bytes(config, blocks + dummy, block_size, kv == "int8")
            summary["num_blocks"] = blocks + dummy
            results[arm].append(summary)
            audits[arm].append(audit)
            if repeat == 0:
                outputs[arm] = [request.output_token_ids[:] for request in raw["requests"]]
            print(f"[{arm} r{repeat}] wall={summary['wall_s']:.2f}s "
                  f"tok/s={summary['throughput_tok_s']:.0f} "
                  f"itl p50/p99/max={summary['itl_ms']['p50']:.1f}/"
                  f"{summary['itl_ms']['p99']:.1f}/{summary['itl_ms']['max']:.1f} "
                  f"ttft p50/p99={summary['ttft_ms']['p50']:.0f}/{summary['ttft_ms']['p99']:.0f} "
                  f"preempt={summary['preemptions']} spills={summary['spill_events']} "
                  f"({summary['spill_bytes'] / 1e6:.1f} MB, w {summary['spill_write_ms_total']:.1f} "
                  f"r {summary['spill_restore_ms_total']:.1f} ms) recomputed={summary['recomputed_tokens']} "
                  f"lazy={summary['lazy_graph_captures']} audit={'ok' if not audit else audit[:2]}")
            del engine, raw
            gc.collect()
            torch.cuda.empty_cache()

    gates = {arm: token_gate(outputs[arm], reference, args.min_identical_tokens) for arm in arms}
    cross = {}
    for left, right in (("fp16_pressure", "fp16_roomy"), ("int8_equal_blocks", "int8_roomy"),
                        ("int8_equal_bytes", "int8_roomy"), ("int8_roomy", "fp16_roomy"),
                        ("int8s_equal_blocks", "int8s_roomy"), ("int8s_equal_bytes", "int8s_roomy"),
                        ("int8s_roomy", "fp16_roomy"), ("int8st_equal_blocks", "int8st_roomy"),
                        ("int8st_equal_bytes", "int8st_roomy"), ("int8st_roomy", "int8s_roomy"),
                        ("int8st_roomy", "fp16_roomy")):
        if left in outputs and right in outputs:
            firsts = [_first_divergence(a, b) for a, b in zip(outputs[left], outputs[right])]
            cross[f"{left}_vs_{right}"] = {
                "identical_requests": sum(1 for f in firsts if f is None),
                "first_divergence": firsts,
            }

    def med(arm, path):
        values = []
        for run in results[arm]:
            value = run
            for part in path.split("."):
                value = value[part]
            values.append(value)
        return {"median": statistics.median(values), "min": min(values), "max": max(values),
                "spread": (max(values) - min(values)) / statistics.median(values)
                if statistics.median(values) else 0.0}

    headline = ("throughput_tok_s", "itl_ms.p50", "itl_ms.p95", "itl_ms.p99", "itl_ms.max",
                "itl_preempted_requests_ms.p99", "itl_preempted_requests_ms.max",
                "ttft_ms.p50", "ttft_ms.p95", "ttft_ms.p99", "queue_total_ms.p95",
                "stall_ms.p50", "stall_ms.max", "stall_ms_total", "preemptions",
                "recomputed_tokens", "spill_events", "spill_bytes", "spill_write_ms_per_event",
                "spill_restore_ms_per_event", "spill_step_ms.max", "restore_step_ms.max",
                "plain_decode_step_ms.p50", "prefill_step_ms.p50", "prefill_step_ms.p99",
                "peak_allocated_bytes", "kv_pool_bytes")
    medians = {arm: {metric: med(arm, metric) for metric in headline} for arm in arms}
    print("\nmedian across repeats [min, max]")
    for metric in headline:
        cells = "  ".join(f"{arm}={medians[arm][metric]['median']:.4g}" for arm in arms)
        print(f"  {metric:32s} {cells}")
    for arm in arms:
        gate = gates[arm]
        print(f"  token gate {arm}: exact {gate['exact_requests']}/{len(schedule)}, ties "
              f"{gate['tie_first_divergence']}, early non-tie {gate['early_non_tie_divergence']}, "
              f"prefix agreement {gate['prefix_agreement']:.1%}")
    import subprocess
    payload = {
        "benchmark": "int8_kv_pressure_ab", "model": args.model, "dtype": "float16",
        "gpu": torch.cuda.get_device_name(), "profile": args.profile, "profile_config": profile,
        "args": vars(args), "block_size": block_size, "graph_buckets": buckets,
        "decode_attention": "per_head", "prefill_chunk_size": 128,
        "max_prefill_tokens_per_iteration": 128, "prefix_cache_blocks": 0,
        "arm_specs": arm_specs, "peak_demand_tokens": peak_tokens,
        "key_smoothing": None if smoothing is None else {
            "calibration": "4 x 2048 tokens of tests/benchmarks/scripts/README (held out)",
            "calibration_s": calibration_s,
            "max_factor_per_layer": [float(layer.max()) for layer in smoothing],
        },
        "schedule": [{"index": r["index"], "arrival_s": r["arrival_s"],
                      "prompt_tokens": len(r["prompt_ids"]), "max_new_tokens": r["max_new_tokens"]}
                     for r in schedule],
        "order": "interleaved, rotated per repeat", "warmup_s": warmups,
        "runs": results, "medians": medians, "audits": audits, "token_gate_vs_stock": gates,
        "cross_arm_tokens": cross,
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Saved -> {out}")
    failed = [arm for arm in arms if any(audits[arm])]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
