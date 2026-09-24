"""Select a draft/depth from Kaggle pair-screen artifacts using explicit gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


def summarize(artifact: dict) -> list[dict]:
    by_depth: dict[int, list[dict]] = defaultdict(list)
    for row in artifact["rows"]:
        by_depth[int(row["depth"])].append(row)
    summaries = []
    for depth, rows in sorted(by_depth.items()):
        baseline = sum(float(row["baseline_ms"]) for row in rows)
        speculative = sum(float(row["speculative_ms"]) for row in rows)
        proposed = sum(int(row["proposed"]) for row in rows)
        accepted = sum(int(row["accepted"]) for row in rows)
        strata: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            strata[str(row["stratum"])].append(float(row["speedup"]))
        stratum_speedups = {
            name: statistics.median(values) for name, values in sorted(strata.items())
        }
        speedups = [float(row["speedup"]) for row in rows]
        summaries.append({
            "draft_model": artifact["draft_model"],
            "draft_revision": artifact.get("draft_revision"),
            "depth": depth,
            "aggregate_speedup": baseline / speculative,
            "median_prompt_speedup": statistics.median(speedups),
            "worst_prompt_speedup": min(speedups),
            "worst_stratum_speedup": min(stratum_speedups.values()),
            "stratum_speedups": stratum_speedups,
            "acceptance_rate": accepted / proposed if proposed else 0.0,
            "draft_wall_fraction": (
                sum(float(row["draft_ms"]) for row in rows) / speculative
                if speculative else 0.0
            ),
            "tokens_match": all(bool(row["tokens_match"]) for row in rows),
            "rows": len(rows),
        })
    return summaries


def choose(
    artifacts: list[dict], *, min_speedup: float, min_stratum_speedup: float,
) -> tuple[dict | None, list[dict]]:
    candidates = [summary for artifact in artifacts for summary in summarize(artifact)]
    eligible = [
        row for row in candidates
        if row["tokens_match"]
        and row["aggregate_speedup"] >= min_speedup
        and row["worst_stratum_speedup"] >= min_stratum_speedup
    ]
    winner = max(
        eligible,
        key=lambda row: (row["aggregate_speedup"], row["worst_stratum_speedup"],
                         -row["draft_wall_fraction"]),
        default=None,
    )
    return winner, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--min-stratum-speedup", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.min_speedup <= 1 or args.min_stratum_speedup <= 0:
        parser.error("min-speedup must exceed 1 and min-stratum-speedup must be positive")

    artifacts = [json.loads(path.read_text()) for path in args.artifacts]
    targets = {(item["target_model"], item.get("target_revision")) for item in artifacts}
    if len(targets) != 1:
        raise SystemExit("all pair screens must use the same target model and revision")
    winner, candidates = choose(
        artifacts, min_speedup=args.min_speedup,
        min_stratum_speedup=args.min_stratum_speedup,
    )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_model": artifacts[0]["target_model"],
        "target_revision": artifacts[0].get("target_revision"),
        "gates": {"min_speedup": args.min_speedup,
                  "min_stratum_speedup": args.min_stratum_speedup},
        "winner": winner,
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if winner is None:
        raise SystemExit("no model/depth pair passed the speed and correctness gates")


if __name__ == "__main__":
    main()
