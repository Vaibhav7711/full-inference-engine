#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Install the project dependencies first." >&2
  exit 1
fi

echo "Loading Qwen3-0.6B on the local GPU."
echo "Chat UI: http://127.0.0.1:8000"
echo "Chat API: POST http://127.0.0.1:8000/v1/chat/completions"
exec .venv/bin/python -m uvicorn engine.server.api:create_app --factory \
  --host 127.0.0.1 --port 8000
