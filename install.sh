#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# Compatibility wrapper for the managed system-preparation flow. Ordinary
# application setup uses uv sync directly; uv itself belongs to the operator.
if [[ -n "${PYTHON:-}" ]]; then
  exec "$PYTHON" -m modiff.install "$@"
fi
if command -v uv >/dev/null 2>&1; then
  exec uv run --no-project --python 3.12 python -m modiff.install "$@"
fi
if command -v python3 >/dev/null 2>&1; then
  exec python3 -m modiff.install "$@"
fi
echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/ and add it to PATH." >&2
exit 2
