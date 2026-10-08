# `ci/rtx` — the request/result bus between the writing machine and the RTX 4060

Two machines, one repository, no shared shell.

- The **writing machine** (no GPU) commits code and **requests**: `ci/rtx/requests/<id>.json`.
- The **RTX agent** (GPU, no judgement about code) polls `main`, runs each request that has
  no result yet, and commits **results**: `ci/rtx/results/<id>/status.json` plus logs, and
  any artifacts the request names under `results/sm89/runs/`.

The two never write to the same path, so a plain `git pull --rebase` on either side never
conflicts. A round trip is one commit each way.

## Request

```json
{
  "id": "0001-sm89-round1-gates",
  "description": "First boot: static, CPU, CUDA kernels, backend table, live token gate.",
  "commands": ["bash scripts/sm89_round1.sh --gates"],
  "timeout_minutes": 60,
  "artifacts": ["results/sm89/round1-*"],
  "requires_gpu_idle": true,
  "depends_on": null,
  "continue_on_error": false
}
```

- `id` must equal the filename stem. Requests run in filename order.
- `commands` are run from the repo root, in order, each under `timeout`. Only these forms
  are accepted: `bash scripts/…`, `python scripts/…`, `python -m …` (and `python3`),
  `pytest …`. Anything else marks the request `rejected` without running it.
- `depends_on` names a request whose `status.json` must say `"ok": true` first; until
  then the request is left alone.
- `requires_gpu_idle`: if another process holds more than 500 MiB on the GPU the request is
  marked `blocked` and retried next poll rather than contaminating a measurement.
- `continue_on_error: false` stops at the first failing command. That is the default and the
  gate scripts rely on it.

## Result

`ci/rtx/results/<id>/status.json`:

```json
{
  "id": "…", "state": "ok | failed | rejected | blocked | timeout",
  "ok": true,
  "request_sha": "<commit of main the agent ran>",
  "started": "…Z", "finished": "…Z", "seconds": 812.4,
  "gpu": "NVIDIA GeForce RTX 4060", "driver": "…", "torch": "…", "triton": "…",
  "steps": [{"command": "…", "exit": 0, "seconds": 12.3, "log": "step-1.log"}]
}
```

Logs are `step-N.log` beside it, full stdout+stderr. Artifacts named by the request are
committed as the request's own commit, so `git log -- results/sm89/` is the run history.

## What the agent never does

Edit anything outside `ci/rtx/results/` and `results/sm89/`. Fix a failing test. Install or
upgrade a package. Force-push, rebase shared history, or delete. Run a command that is not
in a request file. Start a request while the GPU is busy. These are not guidelines for the
agent to weigh; they are the contract that makes a result from that machine trustworthy.

## Running it

On the RTX machine: `bash scripts/rtx_ci_agent.sh` (loops; `POLL_SECONDS=120` default), or
`bash scripts/rtx_ci_agent.sh --once` for a single pass. Operating instructions for an agent
driving that machine: [`AGENT.md`](AGENT.md).
