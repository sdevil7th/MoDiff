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

# Use the same stdlib-only pre-import allocator policy as main.py. Both modern
# and legacy operator variables take precedence, including an explicit empty
# setting. Git Bash on Windows must not inject the Linux allocator default.
if [[ -x "$MANAGED_PYTHON" ]]; then
  ALLOCATOR_DEFAULT="$(cd -- "$PROJECT_ROOT" && "$MANAGED_PYTHON" - <<'PY'
from modiff.runtime_environment import configure_allocator
value = configure_allocator()
print(f'{value["variable"]}={value["setting"]}' if value["source"] == "platform_default" else "")
PY
)"
  if [[ -n "$ALLOCATOR_DEFAULT" ]]; then
    export "$ALLOCATOR_DEFAULT"
  fi
fi

if [[ "$RUNTIME_PROFILE" == "amd-rocm-linux" ]]; then
  if [[ ! -x "$MANAGED_PYTHON" ]]; then
    MANAGED_PYTHON="$(command -v python3)"
  fi
  exec "$MANAGED_PYTHON" -c '
import os
import sys

sys.path.insert(0, sys.argv[1])
from modiff.runtime_environment import configure_rocm_library_path

environment = os.environ.copy()
configure_rocm_library_path(environment)
os.execvpe(sys.argv[2], sys.argv[2:], environment)
' "$PROJECT_ROOT" "$@"
fi

exec "$@"
