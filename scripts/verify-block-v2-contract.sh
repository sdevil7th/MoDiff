#!/usr/bin/env bash

set -euo pipefail

backend_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
client_root="${MODIFF_CLIENT_ROOT:-${backend_root}/../MoDiff-client}"

if [[ ! -f "${client_root}/package.json" ]]; then
  echo "MoDiff client not found at ${client_root}. Set MODIFF_CLIENT_ROOT to its checkout." >&2
  exit 2
fi

if [[ ! -x "${backend_root}/.venv/bin/python" ]]; then
  echo "MoDiff backend runtime is missing at ${backend_root}/.venv." >&2
  exit 2
fi

if [[ -x "${backend_root}/.venv/bin/pytest" ]]; then
  backend_pytest=("${backend_root}/.venv/bin/pytest")
elif command -v uv >/dev/null 2>&1; then
  # Production/runtime repairs intentionally do not install test-only
  # packages. Keep the runtime immutable and layer pytest into a disposable uv
  # environment while retaining the managed interpreter and its dependencies.
  backend_pytest=(
    uv run
    --python "${backend_root}/.venv/bin/python"
    --no-project
    --with pytest
    python -m pytest
  )
else
  echo "MoDiff backend tests require pytest in .venv or the uv runner." >&2
  exit 2
fi

(
  cd "${client_root}"
  npm run test:block-schema-v2
  npm run test:registered-block-v2-adapter
  npm run test:block-runtime-v2
  npm run test:block-renderer-v2
  npm run test:block-persistence-v2
  npm run test:composite-migration-ui
  npm run test:composite-block-capabilities-v2
)

(
  cd "${backend_root}"
  ./scripts/with-runtime-env.sh "${backend_pytest[@]}" -q \
    tests/test_studio_blocks.py \
    tests/test_workflow_store.py \
    tests/test_composite_migration_inventory.py \
    tests/test_legacy_cluster_semantic_equivalence.py \
    tests/test_composite_migration_recovery_audit.py \
    tests/test_composite_migration.py \
    tests/test_composite_migration_server.py
)
