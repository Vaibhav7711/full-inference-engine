"""Same-session commit ladder for optimizations that have no runtime toggle.

The Phase 2 chain (block ownership, batched KV writes, in-kernel length offset,
persistent metadata, batched prefill, ...) changed the engine structurally, so the only
way to attribute each step is to run the *same* benchmark from each commit in one
session. Each rung is a git worktree; every round runs every rung in order, so thermal
drift is spread across all rungs rather than landing on whichever ran last.

    python scripts/commit_ladder.py --rounds 3 --widths 1,8,16 --out $RUN/ladder.json

Rungs follow the journal's *measurement* order, which differs from commit date in one
place: `90485ba` and `5126ade` were measured before `9eda221` (171.2 -> 167.5 -> 172.6
-> 178.3 tok/s). `HEAD` runs twice, without and with the width-16 graph bucket, so the
ladder closes at Phase 7.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

RUNGS: list[tuple[str, str, str]] = [
    # (sha, label, journal claim at width 16)
    ("79d8671", "B1_pre_refactor_k4", "170.3 tok/s"),
    ("acc719d", "D4_block_ownership", "171.2 (+0.5%)"),
    ("90485ba", "D6_scheduler_unification", "167.5 (-2.2%)"),
    ("5126ade", "D5_persistent_metadata", "172.6 (+3.0%)"),
    ("9eda221", "D7_direct_prefill_kv_write", "178.3 (+3.3%)"),
    ("a8469ab", "D8_batched_decode_kv_write", "247.8 (+39.0%)"),
    ("291ccf8", "D9_in_kernel_length_offset", "263.5 (+6.3%)"),
    ("f74c6ce", "R2_batched_token_materialization", "227.2 (reverted)"),
    ("9b4b428", "R2_reverted", "263.5 restored"),
    ("5282517", "D10_triton_rmsnorm", "253.1 (-4.0%, in noise)"),
    ("35a08f6", "D11_triton_rope_swiglu", "261.8 (+3.4%)"),
    ("b728915", "P1_batched_prefill", "413.8 (+58.1%)"),
    ("24e7621", "P2_chunked_prefill", "407.4 (-1.5%)"),
    ("6428f81", "M1_prefix_cache", "406.0 (-1.9%)"),
    ("55ebb7a", "pre_structural_overhead", "-"),
    ("5a9514c", "D16_D17_P6_structural_overhead", "unmeasured (3.4 -> 0.18 ms host stage)"),
    ("HEAD", "HEAD_dynamic", "-"),
]
GRAPH_RUNG = ("HEAD", "HEAD_graph16", "962.4 (Phase 7)")

BENCH = "benchmarks.batching.continuous_throughput"


def sh(cmd: list[str], **kw) -> str:
    return subprocess.check_output(cmd, text=True, **kw).strip()


def ensure_worktree(repo: Path, root: Path, sha: str) -> Path:
    path = root / sha
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        sh(["git", "worktree", "add", "--detach", str(path), sha], cwd=repo)
    return path


def run_rung(tree: Path, out: Path, widths: str, args, graph: bool, log: Path) -> dict:
    cmd = [sys.executable, "-m", BENCH,
           "--num-requests", str(args.num_requests),
           "--max-new-tokens", str(args.max_new_tokens),
           "--concurrencies", widths, "--num-blocks", str(args.num_blocks),
           "--warmup", "--output", str(out)]
    if graph:
        cmd += ["--cuda-graph-batch-size", "16"]
    env = {**os.environ, "PYTHONPATH": str(tree), "TOKENIZERS_PARALLELISM": "false"}
    started = time.time()
    with log.open("a") as fh:
        proc = subprocess.run(cmd, cwd=tree, env=env, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        return {"error": f"exit {proc.returncode}, see {log}", "elapsed_s": time.time() - started}
    data = json.loads(out.read_text())
    tps = {1: data["sequential"]["throughput_tok_s"]}
    for row in data["sweep"]:
        tps[int(row["concurrency"])] = row["throughput_tok_s"]
    return {"tok_s": tps, "elapsed_s": round(time.time() - started, 1)}


def summarize(runs: list[dict], width: int) -> dict:
    values = [r["tok_s"][width] for r in runs if "tok_s" in r and width in r["tok_s"]]
    if not values:
        return {"n": 0}
    med = statistics.median(values)
    return {"n": len(values), "median": med, "min": min(values), "max": max(values),
            "spread": (max(values) - min(values)) / med if med else 0.0}


def verdict(prev: dict, cur: dict) -> str:
    if not prev.get("n") or not cur.get("n"):
        return "no data"
    change = (cur["median"] - prev["median"]) / prev["median"]
    noise = max(prev["spread"], cur["spread"])
    if abs(change) <= noise:
        return f"unresolved: {change:+.1%} within {noise:.1%} spread"
    return f"{change:+.1%} (spread {noise:.1%})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--widths", default="1,8,16")
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--worktrees", default="../ladder_worktrees")
    parser.add_argument("--only", default=None,
                        help="comma-separated rung labels or shas to run (default: all)")
    parser.add_argument("--skip-graph-rung", action="store_true")
    parser.add_argument("--out", default="results/ladder.json")
    args = parser.parse_args()

    repo = Path(sh(["git", "rev-parse", "--show-toplevel"]))
    rungs = list(RUNGS) + ([] if args.skip_graph_rung else [GRAPH_RUNG])
    if args.only:
        keep = set(args.only.split(","))
        rungs = [r for r in rungs if r[0] in keep or r[1] in keep]
    root = (repo / args.worktrees).resolve()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = out.parent / "ladder_logs"
    logs.mkdir(exist_ok=True)
    widths = [int(w) for w in args.widths.split(",")]

    trees = {}
    for sha, label, _ in rungs:
        trees[label] = repo if sha == "HEAD" else ensure_worktree(repo, root, sha)
        print(f"rung {label:36s} {sha:8s} -> {trees[label]}")

    results = {label: {"sha": sha, "claim": claim, "runs": []} for sha, label, claim in rungs}
    payload = {"config": vars(args), "widths": widths, "order": [r[1] for r in rungs],
               "head": sh(["git", "rev-parse", "--short", "HEAD"], cwd=repo), "rungs": results}
    for rnd in range(args.rounds):
        for sha, label, _ in rungs:
            tmp = out.parent / f"ladder_{label}_r{rnd}.json"
            run = run_rung(trees[label], tmp, args.widths, args, graph=(label == GRAPH_RUNG[1]),
                           log=logs / f"{label}.log")
            results[label]["runs"].append(run)
            shown = run.get("tok_s", run.get("error"))
            print(f"[round {rnd + 1}/{args.rounds}] {label:36s} {shown}")
            out.write_text(json.dumps(payload, indent=2))  # checkpoint after every run

    print(f"\n{'rung':36s} {'sha':8s} " + " ".join(f"{'w' + str(w) + ' median':>14s}" for w in widths)
          + f"  {'vs previous @' + str(widths[-1]):>28s}  journal")
    prev = None
    for sha, label, claim in rungs:
        sums = {w: summarize(results[label]["runs"], w) for w in widths}
        results[label]["summary"] = sums
        cells = " ".join(f"{s['median']:>9.1f} ±{s['spread']:>4.0%}" if s.get("n") else f"{'n/a':>14s}"
                         for s in sums.values())
        step = verdict(prev, sums[widths[-1]]) if prev else "baseline"
        results[label]["vs_previous"] = step
        print(f"{label:36s} {sha:8s} {cells}  {step:>28s}  {claim}")
        prev = sums[widths[-1]]
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
