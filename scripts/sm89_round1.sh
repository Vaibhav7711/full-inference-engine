#!/usr/bin/env bash
# First boot of the sm_89 round-1 work on the RTX 4060. Every step is a gate; the script
# stops at the first failure and nothing below a failed gate is a result.
#
#   bash scripts/sm89_round1.sh            # gates + A/Bs
#   bash scripts/sm89_round1.sh --gates    # gates only, no timing
#
# Everything here was written without a GPU (docs/sm89-round1.md). The order is the
# argument: static + CPU, then CUDA kernels against their fp16 shadows, then the backend
# table, then a live token gate, and only then anything timed.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="results/sm89/round1-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
# 8 GB card: the T4 configuration (1024 blocks) does not fit beside graph capture.
BLOCKS="${BLOCKS:-512}"
GATES_ONLY="${1:-}"

step() { printf '\n==== %s ====\n' "$*"; }

step "provenance"
git rev-parse HEAD > "$RUN/git_sha.txt"
git status --short > "$RUN/git_status.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv | tee "$RUN/gpu.txt"
python -c "import torch, triton; print('torch', torch.__version__, '| triton', triton.__version__, '| cuda', torch.version.cuda)" | tee "$RUN/versions.txt"

step "gate 1: static checks and CPU suite"
python -m pytest -q -m "not cuda" -p no:cacheprovider

step "gate 2: CUDA suite (kernels against their fp16 shadows; verify path via speculative tests)"
python -m pytest -q -m cuda -p no:cacheprovider

step "gate 3: backend table on this device"
python scripts/check_hooks.py --backends-only --out "$RUN/backends.json"

step "gate 4: live token gate at the sm_89 defaults"
python scripts/check_hooks.py --model "$MODEL" --num-blocks "$BLOCKS" --out "$RUN/check_hooks.json"
python - "$RUN/check_hooks.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
problems = payload.get("problems", [])
print("problems:", problems)
sys.exit(1 if problems else 0)
PY

if [[ "$GATES_ONLY" == "--gates" ]]; then
  step "gates green; stopping before timing (--gates)"
  exit 0
fi

ab() {
  local setting="$1" profile="$2"; shift 2
  step "A/B: $setting ($profile)"
  python -m benchmarks.reliability.ab --setting "$setting" --model "$MODEL" --dtype float16 \
    --prompt-profile "$profile" --concurrency 8 --max-active 8 --num-blocks "$BLOCKS" \
    --repeats 3 --duration 30 --out "$RUN/ab_${setting}_${profile}.json" "$@"
}

# The oldest open question on this card: tiled prefill on the tensor cores.
ab prefill_kernel long
# Verification on an in-place kernel vs the prefill path's prefix gather.
ab verify_kernel chat
# Storage type, both profiles; the FP8 effect should grow with context.
ab kv_dtype_all chat
ab kv_dtype_all long
# T4 rejections re-measured with 6x the L2.
ab decode_kernel chat
ab decode_split_k chat

step "roofline, for the record"
python -m benchmarks.kernels.roofline --model "$MODEL" --out "$RUN/roofline.json"

step "done -> $RUN"
