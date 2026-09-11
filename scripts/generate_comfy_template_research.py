#!/usr/bin/env python3
"""Generate or validate the pinned official Comfy catalog research ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from modiff.comfy_template_research import (
    COMFY_SOURCE_FILES,
    PINNED_COMFY_WORKFLOW_TEMPLATES_COMMITTED_AT,
    PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION,
    build_comfy_template_research_ledger,
    validate_comfy_template_research_ledger,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "research" / "comfy-workflow-catalog.v1.json"
DEFAULT_WORKFLOW_MANIFEST = REPOSITORY_ROOT / "data" / "workflow-library-manifest.json"


def _git(source_repository: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source_repository), *arguments],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {message}")
    return result.stdout


def _source_blob(source_repository: Path, path: str) -> bytes:
    return _git(source_repository, "show", f"{PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION}:{path}")


def _json_blob(source_repository: Path, path: str) -> Any:
    try:
        return json.loads(_source_blob(source_repository, path))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Pinned Comfy source {path!r} is not valid JSON.") from exc


def _object_list(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise RuntimeError(f"{label} must be a list of objects.")
    return value


def _manifest_ids(value: Any, *, field: str) -> set[str]:
    if not isinstance(value, dict):
        raise RuntimeError("Pinned Comfy source manifest must be an object.")
    rows = _object_list(value.get(field), label=f"Comfy manifest {field}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"Comfy manifest {field} ids must be unique non-empty strings.")
    return set(ids)


def _supported_workflow_ids(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    workflows = _object_list(value.get("workflows") if isinstance(value, dict) else None, label="MoDiff workflows")
    ids = [workflow.get("id") for workflow in workflows]
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("MoDiff canonical workflow ids must be unique non-empty strings.")
    return set(ids)


def _catalog_ids(index: list[dict[str, Any]], *, field: str) -> set[str]:
    ids = []
    for category in index:
        rows = _object_list(category.get(field), label=f"Comfy catalog {field}")
        ids.extend(row.get("name") for row in rows)
    if any(not isinstance(item, str) or not item for item in ids) or len(ids) != len(set(ids)):
        raise RuntimeError(f"Comfy catalog {field} ids must be unique non-empty strings.")
    return set(ids)


def _verify_catalog_files(
    source_repository: Path,
    *,
    template_ids: set[str],
    blueprint_ids: set[str],
) -> None:
    tree = set(
        _git(
            source_repository,
            "ls-tree",
            "-r",
            "--name-only",
            PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION,
            "--",
            "templates",
            "blueprints",
        )
        .decode("utf-8")
        .splitlines()
    )
    expected = {f"templates/{item}.json" for item in template_ids} | {
        f"blueprints/{item}.json" for item in blueprint_ids
    }
    missing = sorted(expected - tree)
    if missing:
        raise RuntimeError(f"Pinned Comfy catalog references missing JSON files: {missing!r}")


def generate(source_repository: Path, *, workflow_manifest: Path) -> dict[str, Any]:
    _git(source_repository, "cat-file", "-e", f"{PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION}^{{commit}}")
    committed_at = (
        _git(
            source_repository,
            "show",
            "-s",
            "--format=%aI",
            PINNED_COMFY_WORKFLOW_TEMPLATES_REVISION,
        )
        .decode("utf-8")
        .strip()
    )
    if committed_at != PINNED_COMFY_WORKFLOW_TEMPLATES_COMMITTED_AT:
        raise RuntimeError("Pinned Comfy commit timestamp does not match the reviewed provenance contract.")

    source_blobs = {path: _source_blob(source_repository, path) for path in COMFY_SOURCE_FILES}
    license_text = source_blobs["LICENSE"].decode("utf-8")
    if "MIT License" not in license_text or "Copyright (c) 2023-present Comfy Org" not in license_text:
        raise RuntimeError("Pinned Comfy source license does not match the reviewed MIT notice.")

    template_index = _object_list(json.loads(source_blobs["templates/index.json"]), label="Comfy template index")
    blueprint_index = _object_list(
        json.loads(source_blobs["blueprints/index.json"]),
        label="Comfy blueprint index",
    )
    template_manifest = json.loads(source_blobs["packages/core/src/comfyui_workflow_templates_core/manifest.json"])
    blueprint_manifest = json.loads(
        source_blobs["packages/core/src/comfyui_workflow_templates_core/blueprints_manifest.json"]
    )
    template_ids = _catalog_ids(template_index, field="templates")
    blueprint_ids = _catalog_ids(blueprint_index, field="blueprints")
    _verify_catalog_files(
        source_repository,
        template_ids=template_ids,
        blueprint_ids=blueprint_ids,
    )
    return build_comfy_template_research_ledger(
        template_index=template_index,
        blueprint_index=blueprint_index,
        template_manifest_ids=_manifest_ids(template_manifest, field="templates"),
        blueprint_manifest_ids=_manifest_ids(blueprint_manifest, field="blueprints"),
        supported_workflow_ids=_supported_workflow_ids(workflow_manifest),
        source_hashes={path: hashlib.sha256(blob).hexdigest() for path, blob in source_blobs.items()},
    )


def _render(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repository",
        type=Path,
        help="Local Git object store containing the pinned official Comfy revision.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workflow-manifest", type=Path, default=DEFAULT_WORKFLOW_MANIFEST)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate from --source-repository and fail if the checked-in ledger differs.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the checked-in ledger without requiring the upstream Git object store.",
    )
    args = parser.parse_args()

    supported_workflow_ids = _supported_workflow_ids(args.workflow_manifest)
    if args.validate or args.source_repository is None:
        ledger = json.loads(args.output.read_text(encoding="utf-8"))
        validate_comfy_template_research_ledger(ledger, supported_workflow_ids=supported_workflow_ids)
        if args.source_repository is None:
            if args.check:
                parser.error("--check requires --source-repository")
            return 0

    generated = generate(args.source_repository, workflow_manifest=args.workflow_manifest)
    validate_comfy_template_research_ledger(generated, supported_workflow_ids=supported_workflow_ids)
    rendered = _render(generated)
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != rendered:
            raise SystemExit(f"{args.output} is stale; regenerate it with {Path(__file__).name}.")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
