"""Prediction vs measurement for the Qwen3-8B A/B plan, from a directory of `ab_*.json`.

    python scripts/qwen3_8b_results_table.py results/sm75/qwen3-8b/plan --md

Same reading as the notebook's results cell, runnable on the CLI-driven kernel's output.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

PREDICTED = {
    ("fused_step", "chat"):     "large win (saves a weight sweep per prefill-carrying step)",
    ("fused_step", "long"):     "large win",
    ("prefill_chunk", "long"):  "TTFT/ITL trade far sharper than at 0.6B",
    ("kv_dtype", "long"):       "INT8 KV resolves for the first time (KV ~16% of step bytes)",
    ("cuda_graphs", "chat"):    "~1.3x, down from 3.5x",
    ("prefix_cache", "chat"):   "grows (13.9 GFLOP per reused token)",
    ("triton_rmsnorm", "chat"): "shrinks ~3x in share",
    ("triton_rope", "chat"):    "unresolved",
    ("triton_swiglu", "chat"):  "unresolved",
    ("decode_split_k", "long"): "batch-1/long-context only",
}
KEY = ["step_timing.expected_gap_ms", "latency.itl_p50", "latency.itl_p99", "latency.ttft_p50",
       "step_timing.decode_step_p50_ms", "step_timing.prefill_step_p50_ms", "delivered_tokens"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--md", action="store_true", help="print a Markdown table")
    args = parser.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.directory, "ab_*.json"))):
        r = json.load(open(path))
        setting, profile = r["setting"], r["prompt_profile"]
        identity = r.get("token_identity", {})
        gate = "ok" if not identity.get("early_divergence_arms") else "FAILED STOCK GATE"
        row = {"setting": setting, "profile": profile, "predicted": PREDICTED.get((setting, profile), ""),
               "arms": " vs ".join(r["arm_settings"]), "skipped": ",".join(r.get("skipped_arms", {})) or "-",
               "gate": gate, "repeats": r.get("repeats")}
        for m in KEY:
            row[m.split(".")[-1]] = r["comparison"].get(m, "")
        base = r["arm_settings"][0]
        summary = r["arms"][base]["summary"]
        row["baseline_itl_p50_ms"] = round(summary["latency.itl_p50"]["median"], 1)
        row["baseline_decode_step_ms"] = round(summary["step_timing.decode_step_p50_ms"]["median"], 1)
        rows.append(row)
    if not rows:
        print("no ab_*.json under", args.directory)
        return 1
    columns = list(rows[0])
    if args.md:
        print("| " + " | ".join(columns) + " |")
        print("|" + "---|" * len(columns))
        for row in rows:
            print("| " + " | ".join(str(row[c]) for c in columns) + " |")
    else:
        for row in rows:
            print(json.dumps(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
