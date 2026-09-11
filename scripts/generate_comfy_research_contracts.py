#!/usr/bin/env python3
"""Generate or validate fail-closed contracts for pinned Comfy research candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from modiff.comfy_research_contracts import (
    build_comfy_research_contract_ledger,
    validate_comfy_research_contract_ledger,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_LEDGER = REPOSITORY_ROOT / "data" / "research" / "comfy-workflow-catalog.v1.json"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "research" / "comfy-research-contracts.v1.json"
DEFAULT_WORKFLOW_MANIFEST = REPOSITORY_ROOT / "data" / "workflow-library-manifest.json"


def _load_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object.")
    return value, payload


def _supported_workflow_ids(path: Path) -> set[str]:
    manifest, _payload = _load_object(path, label="MoDiff workflow manifest")
    workflows = manifest.get("workflows")
    if not isinstance(workflows, list) or any(not isinstance(item, dict) for item in workflows):
        raise RuntimeError("MoDiff workflow manifest workflows must be a list of objects.")
    ids = [item.get("id") for item in workflows]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("MoDiff workflow ids must be unique non-empty strings.")
    return set(ids)


def _render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-ledger", type=Path, default=DEFAULT_CATALOG_LEDGER)
    parser.add_argument("--workflow-manifest", type=Path, default=DEFAULT_WORKFLOW_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Fail when the checked-in contracts are stale.")
    parser.add_argument("--validate", action="store_true", help="Validate without rewriting the contracts.")
    args = parser.parse_args()

    catalog, catalog_payload = _load_object(args.catalog_ledger, label="Pinned Comfy catalog ledger")
    catalog_sha256 = hashlib.sha256(catalog_payload).hexdigest()
    workflow_ids = _supported_workflow_ids(args.workflow_manifest)
    generated = build_comfy_research_contract_ledger(
        catalog_ledger=catalog,
        catalog_ledger_sha256=catalog_sha256,
        supported_workflow_ids=workflow_ids,
    )
    rendered = _render(generated)

    if args.validate:
        current, _payload = _load_object(args.output, label="Comfy research-contract ledger")
        validate_comfy_research_contract_ledger(
            current,
            catalog_ledger=catalog,
            catalog_ledger_sha256=catalog_sha256,
            supported_workflow_ids=workflow_ids,
        )
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != rendered:
            raise SystemExit(f"{args.output} is stale; regenerate it with {Path(__file__).name}.")
        return 0
    if args.validate:
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
