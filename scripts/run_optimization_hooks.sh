#!/usr/bin/env bash
# Run one controlled A/B at a time.  Each comparison changes one engine feature only;
# this is more informative than attributing an arbitrary all-features permutation to
# its last flag.  Pass a setting name, or `all` to queue the independent comparisons.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

setting="${1:-all}"
repeats="${REPEATS:-5}"
duration="${DURATION_S:-8}"
concurrency="${CONCURRENCY:-8}"

run_setting() {
  local name="$1"
  echo "=== $name ==="
  .venv/bin/python -m benchmarks.reliability.ab \
    --setting "$name" --repeats "$repeats" --duration "$duration" \
    --concurrency "$concurrency" --max-active "$concurrency" \
    --cuda-graphs --out "results/ab_${name}.json"
}

if [[ "$setting" == "naive" ]]; then
  .venv/bin/python -m benchmarks.inference.naive_generate \
    --prompt "Explain KV caching in one sentence." --max-new-tokens 32 \
    --warmup-runs 2 --runs "$repeats" --output results/ab_naive_explicit.json
elif [[ "$setting" == "all" ]]; then
  # These are the serving features with end-to-end A/B definitions in this repository.
  for item in cuda_graphs prefix_cache kv_dtype prefill_kernel prefill_chunk; do
    run_setting "$item"
  done
else
  run_setting "$setting"
fi
