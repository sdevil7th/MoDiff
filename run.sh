#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -x ./.venv/bin/python ]]; then
  exec ./.venv/bin/python main.py "$@"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run main.py "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 main.py "$@"
fi

exec python main.py "$@"
