#!/usr/bin/env bash
# Apply the installed MoDiff runtime profile environment, then run one command.
set -euo pipefail

if (( $# == 0 )); then
  echo "Usage: ./scripts/with-runtime-env.sh <command> [args...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MANAGED_PYTHON="$PROJECT_ROOT/.venv/bin/python"
INSTALL_STATE="$PROJECT_ROOT/.modiff/install-state.json"
RUNTIME_PROFILE="${MODIFF_RUNTIME_PROFILE:-}"

if [[ -z "$RUNTIME_PROFILE" && -x "$MANAGED_PYTHON" && -r "$INSTALL_STATE" ]]; then
  RUNTIME_PROFILE="$("$MANAGED_PYTHON" - "$INSTALL_STATE" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        value = json.load(handle)
    print(value.get("profile") or "")
except (OSError, UnicodeError, ValueError, TypeError):
    print("")
PY
)"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "$RUNTIME_PROFILE" == "amd-rocm-linux" ]]; then
  ROCM_LIBRARY_PATHS=()
  [[ -d /opt/rocm/lib ]] && ROCM_LIBRARY_PATHS+=(/opt/rocm/lib)
  for directory in /opt/rocm/core-*/lib; do
    [[ -d "$directory" ]] && ROCM_LIBRARY_PATHS+=("$directory")
  done
  if (( ${#ROCM_LIBRARY_PATHS[@]} > 0 )); then
    ROCM_JOINED="$(IFS=:; echo "${ROCM_LIBRARY_PATHS[*]}")"
    export LD_LIBRARY_PATH="$ROCM_JOINED${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export HIP_PATH="${HIP_PATH:-/opt/rocm}"
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
  fi
fi

exec "$@"
