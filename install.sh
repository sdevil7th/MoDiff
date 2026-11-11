#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Compatibility wrapper for the managed system-preparation flow. Ordinary
# application setup uses uv sync directly; uv itself belongs to the operator.
if [[ -n "${PYTHON:-}" ]]; then
  exec "$PYTHON" -m modiff.install "$@"
fi
if command -v uv >/dev/null 2>&1; then
  modiff_uv=uv
  # Reuse the canonical version check when Python can already run it.
  if command -v python3 >/dev/null 2>&1 && python3 -c "import sys" >/dev/null 2>&1; then
    modiff_uv="$(python3 -c "from pathlib import Path; from modiff.tool_locks import resolve_uv; print(resolve_uv(Path('.modiff')))" )"
  fi
  exec "$modiff_uv" run --no-project --python 3.12 python -m modiff.install "$@"
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 -m modiff.install "$@"
fi
echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and add it to PATH." >&2
exit 2
