"""Fail-closed planning for destructive Hugging Face cache turnover."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from modiff.workflow_store import list_workflows


_REVISION = re.compile(r"^[0-9a-f]{40}$")


class HfCacheDeletionPlanError(ValueError):
    pass


def parse_revision_hashes(value: str) -> list[str]:
    hashes = str(value or "").split(",")
    if not hashes or any(not _REVISION.fullmatch(item) for item in hashes):
        raise HfCacheDeletionPlanError(
            "Cache deletion requires one or more exact lowercase 40-character revision hashes."
        )
    if len(hashes) > 32 or len(set(hashes)) != len(hashes):
        raise HfCacheDeletionPlanError("Cache deletion revision hashes must be unique and bounded to 32 entries.")
    return hashes


def _artifact_uses_repo(artifact: Any, repo_id: str) -> bool:
    return isinstance(artifact, str) and (artifact == repo_id or artifact.startswith(f"{repo_id}/"))


def _resolve_targets(models: Iterable[dict], hashes: list[str]) -> tuple[list[dict], list[str]]:
    by_hash: dict[str, list[dict]] = {}
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        for revision in model.get("revisions") or []:
            if not isinstance(revision, dict) or not isinstance(revision.get("hash"), str):
                continue
            by_hash.setdefault(revision["hash"], []).append(
                {
                    "repoId": model["id"],
                    "revision": revision["hash"],
                    "sizeBytes": revision.get("size"),
                }
            )
    targets = []
    unresolved = []
    for revision_hash in hashes:
        matches = by_hash.get(revision_hash) or []
        if len(matches) != 1:
            unresolved.append(revision_hash)
        else:
            targets.append(matches[0])
    return targets, unresolved


def _receipt_closes_dependency(receipt: dict, workflow_id: str, repo_id: str, revision: str) -> bool:
    if (
        receipt.get("workflowId") != workflow_id
        or receipt.get("turnoverEligible") is not True
        or receipt.get("userApproval") not in {"approved", True}
        or receipt.get("rightsApproval") not in {"approved", True}
    ):
        return False
    execution = receipt.get("execution")
    if not isinstance(execution, dict):
        return False
    if execution.get("modelRepo") == repo_id and execution.get("modelRevision") == revision:
        return True
    artifact = execution.get("modelArtifact")
    return (
        isinstance(artifact, dict)
        and _artifact_uses_repo(artifact.get("value"), repo_id)
        and artifact.get("revision") == revision
    )


def _load_receipts(project_root: Path) -> list[dict]:
    receipts = []
    for directory in (project_root / "review-approved", project_root / "review-pending"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.generation-*-receipt.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError):
                continue
            if isinstance(value, dict):
                receipts.append(value)
    return receipts


def _contains_repo(value: Any, repo_id: str) -> bool:
    if _artifact_uses_repo(value, repo_id):
        return True
    if isinstance(value, dict):
        return any(_contains_repo(item, repo_id) for item in value.values())
    if isinstance(value, list):
        return any(_contains_repo(item, repo_id) for item in value)
    return False


def build_hf_cache_deletion_plan(
    *,
    project_root: Path,
    data_dir: Path,
    revision_hashes: list[str],
    models: Iterable[dict],
    current_task: dict | None = None,
    queued_tasks: dict | None = None,
    download_repos: Iterable[str] = (),
    template_gallery_active: bool = False,
    allow_redownload: bool = False,
) -> dict:
    """Build a deterministic plan for deletion or explicit local-cache eviction.

    ``allow_redownload`` preserves saved/canonical workflows while allowing
    their exact local revisions to be evicted.  Those dependencies remain
    visible as warnings and will require a new download before the workflow can
    run again.  Active work, ambiguous revisions, and mutable cache state always
    remain hard blockers.
    """

    targets, unresolved = _resolve_targets(models, revision_hashes)
    repo_targets = {(item["repoId"], item["revision"]) for item in targets}
    blockers = []
    warnings = []
    if unresolved:
        blockers.append({"code": "unknown_or_ambiguous_revision", "revisionHashes": unresolved})
    if current_task:
        blockers.append({"code": "graph_execution_active", "taskId": current_task.get("task_id")})
    if queued_tasks:
        blockers.append({"code": "graph_execution_queued", "taskIds": sorted(map(str, queued_tasks))})
    download_repos = sorted({str(item) for item in download_repos})
    if download_repos:
        blockers.append({"code": "model_download_active", "repoIds": download_repos})
    if template_gallery_active:
        blockers.append({"code": "template_gallery_install_active"})

    manifest_path = data_dir / "workflow-library-manifest.json"
    workflows = []
    if manifest_path.is_file():
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
            workflows = [
                item
                for item in [*(document.get("workflows") or []), *(document.get("experimentalWorkflows") or [])]
                if isinstance(item, dict) and isinstance(item.get("id"), str)
            ]
        except (OSError, TypeError, ValueError):
            blockers.append({"code": "dependency_catalog_invalid", "path": str(manifest_path)})
    else:
        blockers.append({"code": "dependency_catalog_missing", "path": str(manifest_path)})

    receipts = _load_receipts(project_root)
    canonical_dependencies = []
    for repo_id, revision in sorted(repo_targets):
        for workflow in workflows:
            required = workflow.get("requiredArtifacts") or []
            if not any(_artifact_uses_repo(artifact, repo_id) for artifact in required):
                continue
            workflow_id = workflow["id"]
            closed = any(
                _receipt_closes_dependency(receipt, workflow_id, repo_id, revision) for receipt in receipts
            )
            dependency = {
                "workflowId": workflow_id,
                "repoId": repo_id,
                "revision": revision,
                "turnoverClosed": closed,
            }
            canonical_dependencies.append(dependency)
            if not closed:
                if allow_redownload:
                    warnings.append({"code": "canonical_dependency_requires_redownload", **dependency})
                else:
                    blockers.append({"code": "canonical_dependency_open", **dependency})

    saved_dependencies = []
    try:
        saved_workflows = list_workflows(data_dir)
    except (OSError, TypeError, ValueError):
        blockers.append({"code": "saved_workflow_catalog_invalid"})
        saved_workflows = []
    for workflow in saved_workflows:
        used_repos = sorted(repo_id for repo_id, _revision in repo_targets if _contains_repo(workflow, repo_id))
        if not used_repos:
            continue
        dependency = {
            "workflowId": str(workflow.get("id") or ""),
            "title": str(workflow.get("title") or "Workflow"),
            "repoIds": used_repos,
        }
        saved_dependencies.append(dependency)
        if allow_redownload:
            warnings.append({"code": "saved_workflow_requires_redownload", **dependency})
        else:
            blockers.append({"code": "saved_workflow_dependency", **dependency})

    plan = {
        "schemaVersion": 1,
        "kind": "hf_cache_deletion_plan",
        "revisionHashes": revision_hashes,
        "targets": targets,
        "canonicalDependencies": canonical_dependencies,
        "savedWorkflowDependencies": saved_dependencies,
        "dependencyPolicy": "allow_redownload" if allow_redownload else "protect_dependencies",
        "warnings": warnings,
        "blockers": blockers,
        "canDelete": not blockers and len(targets) == len(revision_hashes),
    }
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    plan["planHash"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    return plan
