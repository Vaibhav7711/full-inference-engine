#!/usr/bin/env bash
# The RTX machine's half of ci/rtx: pull main, run every request without a result, commit
# the result, push. Protocol in ci/rtx/README.md; operating rules in ci/rtx/AGENT.md.
#
#   bash scripts/rtx_ci_agent.sh           # loop, POLL_SECONDS (default 120) between passes
#   bash scripts/rtx_ci_agent.sh --once    # one pass
#
# Writes only under ci/rtx/results/ and results/sm89/. Never force-pushes. Runs only
# commands from request files that match the allowlist in `allowed`.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
BRANCH="${BRANCH:-main}"
POLL="${POLL_SECONDS:-120}"
GPU_IDLE_MIB="${GPU_IDLE_MIB:-500}"
ONCE="${1:-}"

say() { printf '[rtx-ci %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }

sync_main() {
  git fetch -q origin || return 1
  git checkout -q "$BRANCH" || return 1
  git pull -q --rebase origin "$BRANCH" || return 1
}

# JSON helpers via python: jq is not assumed.
jget() {  # jget <file> <key> [default]
  python3 - "$1" "$2" "${3-}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
value = data.get(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
if isinstance(value, bool): print("true" if value else "false")
elif value is None: print("")
elif isinstance(value, list): print("\n".join(str(v) for v in value))
else: print(value)
PY
}

allowed() {  # the only command shapes a request may contain
  case "$1" in
    "bash scripts/"*|"python scripts/"*|"python3 scripts/"*|"python -m "*|"python3 -m "*|"pytest"*|"pytest "*) return 0 ;;
    *) return 1 ;;
  esac
}

gpu_used_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0; }

env_json() {
  python3 - <<'PY'
import json, subprocess
out = {}
try:
    import torch; out["torch"] = torch.__version__; out["cuda"] = torch.version.cuda
    out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
except Exception as e: out["torch_error"] = repr(e)
try:
    import triton; out["triton"] = triton.__version__
except Exception as e: out["triton_error"] = repr(e)
try:
    out["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                            "--format=csv,noheader"], text=True).strip()
except Exception: pass
print(json.dumps(out))
PY
}

write_status() {  # write_status <dir> <id> <state> <ok> <sha> <started> <finished> <steps_json> <note>
  python3 - "$@" <<'PY'
import json, sys, time
d, rid, state, ok, sha, started, finished, steps, note = sys.argv[1:10]
env = json.loads(open(f"{d}/.env.json").read()) if __import__("os").path.exists(f"{d}/.env.json") else {}
status = {"id": rid, "state": state, "ok": ok == "true", "request_sha": sha,
          "started": started, "finished": finished or None,
          "seconds": (time.mktime(time.strptime(finished, "%Y-%m-%dT%H:%M:%SZ"))
                      - time.mktime(time.strptime(started, "%Y-%m-%dT%H:%M:%SZ"))) if finished else None,
          **env, "steps": json.loads(steps), "note": note or None}
json.dump(status, open(f"{d}/status.json", "w"), indent=2)
PY
}

commit_push() {  # commit_push <message> <paths...>
  local message="$1"; shift
  git add -A -- "$@" 2>/dev/null
  git diff --cached --quiet && return 0
  git commit -q -m "$message" -m "Co-Authored-By: rtx4060-agent <rtx4060-agent@local>" || return 1
  for attempt in 1 2 3 4; do
    git pull -q --rebase origin "$BRANCH" && git push -q origin "$BRANCH" && return 0
    say "push rejected (attempt $attempt); retrying"; sleep $((attempt * 15))
  done
  say "push failed after retries; stopping. Resolve by hand, never force."
  return 1
}

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

run_request() {
  local req="$1" rid; rid="$(jget "$req" id)"
  local dir="ci/rtx/results/$rid"
  local sha; sha="$(git rev-parse HEAD)"
  local dep; dep="$(jget "$req" depends_on)"
  if [[ -n "$dep" ]]; then
    if [[ ! -f "ci/rtx/results/$dep/status.json" ]] || [[ "$(jget "ci/rtx/results/$dep/status.json" ok)" != "true" ]]; then
      say "$rid waits on $dep"; return 0
    fi
  fi
  mkdir -p "$dir"
  env_json > "$dir/.env.json"
  local started; started="$(now)"

  if [[ "$(jget "$req" requires_gpu_idle true)" == "true" ]] && (( $(gpu_used_mib) > GPU_IDLE_MIB )); then
    say "$rid blocked: GPU holds $(gpu_used_mib) MiB"
    write_status "$dir" "$rid" blocked false "$sha" "$started" "$(now)" "[]" "GPU busy: $(gpu_used_mib) MiB in use"
    rm -f "$dir/status.json"      # blocked is transient: leave no result so it retries
    return 0
  fi

  # Validate every command before running any.
  local cmds=() c
  while IFS= read -r c; do [[ -n "$c" ]] && cmds+=("$c"); done < <(jget "$req" commands)
  for c in "${cmds[@]}"; do
    if ! allowed "$c"; then
      say "$rid rejected: command not on allowlist: $c"
      write_status "$dir" "$rid" rejected false "$sha" "$started" "$(now)" "[]" "not on allowlist: $c"
      rm -f "$dir/.env.json"
      commit_push "ci/rtx: $rid rejected" "$dir"; return 0
    fi
  done

  # Announce pickup so the other side sees it is running.
  write_status "$dir" "$rid" running false "$sha" "$started" "" "[]" ""
  commit_push "ci/rtx: $rid running on $(hostname)" "$dir" || return 1

  local timeout_min; timeout_min="$(jget "$req" timeout_minutes 60)"
  local cont; cont="$(jget "$req" continue_on_error false)"
  local steps="[]" state=ok ok=true n=0
  for c in "${cmds[@]}"; do
    n=$((n + 1)); local log="step-$n.log" t0 t1 code
    say "$rid step $n: $c"
    t0=$(date +%s)
    timeout "${timeout_min}m" bash -c "$c" > "$dir/$log" 2>&1; code=$?
    t1=$(date +%s)
    steps="$(python3 -c "import json,sys; s=json.loads(sys.argv[1]); s.append({'command': sys.argv[2], 'exit': int(sys.argv[3]), 'seconds': int(sys.argv[4]), 'log': sys.argv[5]}); print(json.dumps(s))" "$steps" "$c" "$code" "$((t1 - t0))" "$log")"
    if (( code != 0 )); then
      ok=false; state=failed; (( code == 124 )) && state=timeout
      say "$rid step $n exit $code ($state)"
      [[ "$cont" == "true" ]] || break
    fi
  done

  write_status "$dir" "$rid" "$state" "$ok" "$sha" "$started" "$(now)" "$steps" ""
  rm -f "$dir/.env.json"
  local paths=("$dir") a
  while IFS= read -r a; do [[ -n "$a" ]] && paths+=("$a"); done < <(jget "$req" artifacts)
  commit_push "ci/rtx: $rid $state ($(hostname), main @ ${sha:0:7})" "${paths[@]}"
}

pass() {
  sync_main || { say "cannot sync $BRANCH; will retry"; return 0; }
  shopt -s nullglob
  local req
  for req in ci/rtx/requests/*.json; do
    local rid; rid="$(jget "$req" id)"
    [[ "$(basename "$req" .json)" == "$rid" ]] || { say "skipping $req: id does not match filename"; continue; }
    # A finished result is final. A `running` one means a previous pass died mid-request
    # (power, OOM-kill); run it again rather than leaving it stuck forever.
    if [[ -f "ci/rtx/results/$rid/status.json" ]] && \
       [[ "$(jget "ci/rtx/results/$rid/status.json" state)" != "running" ]]; then
      continue
    fi
    run_request "$req" || return 1
  done
}

if [[ "$ONCE" == "--once" ]]; then pass; exit $?; fi
say "polling $BRANCH every ${POLL}s from $REPO"
while true; do pass || exit 1; sleep "$POLL"; done
