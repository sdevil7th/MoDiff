#!/usr/bin/env bash
# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ -x ./.venv/bin/python ]]; then
  ./scripts/with-runtime-env.sh ./.venv/bin/python -m modiff.preflight --fail-on-error
  exec ./scripts/with-runtime-env.sh ./.venv/bin/python main.py "$@"
fi

if [[ -f ./.modiff/install-state.json ]]; then
  echo "The managed MoDiff environment is missing or unusable. Run ./install.sh --repair before starting." >&2
else
  echo "No managed MoDiff environment was found. Run ./install.sh before starting." >&2
fi
exit 2
