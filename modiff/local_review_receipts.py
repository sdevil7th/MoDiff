"""Capture fail-closed local technical-review receipts from the running app."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import ipaddress
import json
from pathlib import Path, PurePosixPath
import re
from typing import Callable
from urllib.parse import urlparse


SCHEMA_VERSION = 1
MAX_RECEIPTS = 512
MAX_OUTPUTS_PER_RECEIPT = 32
MAX_OUTPUT_BYTES = 1 << 34
_WORKFLOW_ID = re.compile(r"^[A-Za-z0-9_.-]+:[a-z0-9_]+$")
_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
_ALLOWED_OUTPUT_ROOTS = frozenset({"audio", "exports", "images", "videos"})
_MEDIA_SUFFIXES = {
    ".flac": "audio",
    ".jpeg": "image",
    ".jpg": "image",
    ".json": "json",
    ".m4a": "audio",
    ".mov": "video",
    ".mp3": "audio",
    ".mp4": "video",
    ".png": "image",
    ".webm": "video",
    ".webp": "image",
    ".wav": "audio",
}


def canonical_content_hash(document: dict) -> str:
    payload = deepcopy(document)
    payload.pop("contentHash", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def loopback_base_url(value: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    host = str(parsed.hostname or "").strip().strip("[]").lower()
    if parsed.scheme not in {"http", "https"} or not host or parsed.path not in {"", "/"}:
        raise ValueError("Review receipt server must be an HTTP(S) loopback origin without a path.")
    if host != "localhost":
        try:
            if not ipaddress.ip_address(host).is_loopback:
                raise ValueError
        except ValueError as error:
            raise ValueError("Review receipts may be captured only from a loopback MoDiff app.") from error
    return normalized


def _relative_output_path(value: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or len(path.parts) < 2
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[0] not in _ALLOWED_OUTPUT_ROOTS
    ):
        raise ValueError("Review output paths must stay under an approved app data subdirectory.")
    if path.suffix.lower() not in _MEDIA_SUFFIXES:
        raise ValueError("Review output path has an unsupported media suffix.")
    return path.as_posix()


def parse_record(value: str) -> dict:
    parts = str(value or "").split("|")
    if len(parts) != 3:
        raise ValueError("Review records must use WORKFLOW_ID|TASK_ID|OUTPUT[,OUTPUT...] syntax.")
    workflow_id, task_id, output_list = (part.strip() for part in parts)
    if not _WORKFLOW_ID.fullmatch(workflow_id):
        raise ValueError("Review record has an invalid canonical workflow ID.")
    if not _TASK_ID.fullmatch(task_id):
        raise ValueError("Review record has an invalid app task ID.")
    outputs = [_relative_output_path(item) for item in output_list.split(",")]
    if not outputs or len(outputs) > MAX_OUTPUTS_PER_RECEIPT or len(set(outputs)) != len(outputs):
        raise ValueError("Review record output paths must be unique and bounded.")
    return {"workflowId": workflow_id, "taskId": task_id, "outputs": outputs}


def load_candidate_contracts(root: Path) -> dict[str, dict]:
    path = root / "data" / "template-candidate-contracts.v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    contracts = payload.get("contracts")
    if not isinstance(contracts, list):
        raise ValueError("Template candidate contract ledger is malformed.")
    by_workflow = {}
    for contract in contracts:
        workflow_id = contract.get("canonicalWorkflowId") if isinstance(contract, dict) else None
        if not isinstance(workflow_id, str) or workflow_id in by_workflow:
            raise ValueError("Template candidate contract workflow identities must be unique.")
        by_workflow[workflow_id] = contract
    return by_workflow


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_receipt(data_root: Path, relative_path: str, expected_kind: str) -> dict:
    relative_path = _relative_output_path(relative_path)
    destination = data_root.joinpath(*PurePosixPath(relative_path).parts)
    if destination.is_symlink() or not destination.is_file():
        raise ValueError(f"Review output is missing or linked: {relative_path}")
    resolved = destination.resolve()
    try:
        resolved.relative_to(data_root.resolve())
    except ValueError as error:
        raise ValueError(f"Review output escapes the app data root: {relative_path}") from error
    byte_size = destination.stat().st_size
    if byte_size <= 0 or byte_size > MAX_OUTPUT_BYTES:
        raise ValueError(f"Review output size is outside the bounded receipt envelope: {relative_path}")
    media_kind = _MEDIA_SUFFIXES[destination.suffix.lower()]
    if media_kind != expected_kind:
        raise ValueError(
            f"Review output kind {media_kind!r} does not match workflow kind {expected_kind!r}: {relative_path}"
        )
    return {
        "byteSize": byte_size,
        "mediaKind": media_kind,
        "path": relative_path,
        "sha256": _sha256_file(destination),
        "storage": "local_app_data_ignored",
    }


def _task_receipt(run: dict, expected_task_id: str) -> dict:
    task = run.get("task") if isinstance(run, dict) else None
    if not isinstance(task, dict) or task.get("task_id") != expected_task_id:
        raise ValueError("App run response does not match the requested task identity.")
    if task.get("status") != "completed":
        raise ValueError(f"App task {expected_task_id} is not completed.")
    runtime = task.get("runtimeFingerprint")
    measurement = task.get("runtimeMeasurement")
    packages = runtime.get("packages") if isinstance(runtime, dict) else None
    if not isinstance(runtime, dict) or not isinstance(packages, dict) or not isinstance(measurement, dict):
        raise ValueError("Completed app task lacks bounded runtime evidence.")
    elapsed = measurement.get("elapsedSeconds")
    rss = measurement.get("processRssBytes")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ValueError("Completed app task has an invalid elapsed-time measurement.")
    if isinstance(rss, bool) or not isinstance(rss, int) or rss <= 0:
        raise ValueError("Completed app task has an invalid RSS measurement.")
    fingerprints = (runtime.get("fingerprint"), runtime.get("resourceFingerprint"))
    if any(not isinstance(value, str) or not value.startswith("sha256:") for value in fingerprints):
        raise ValueError("Completed app task lacks exact runtime fingerprints.")
    return {
        "clientRunId": task.get("client_run_id"),
        "completedAt": task.get("completed_at"),
        "elapsedSeconds": elapsed,
        "packages": {key: packages.get(key) for key in sorted(packages)},
        "processRssBytes": rss,
        "resourceFingerprint": runtime["resourceFingerprint"],
        "runtimeFingerprint": runtime["fingerprint"],
        "sid": task.get("sid"),
        "status": "completed",
        "taskId": expected_task_id,
        "workflowTitle": task.get("workflow_title"),
    }


def _candidate_contract_id(workflow_id: str) -> str:
    return f"template-candidate:{workflow_id}"


def _ledger_document(receipts: list[dict]) -> dict:
    receipts = sorted(deepcopy(receipts), key=lambda item: item["workflowId"])
    output_kinds = Counter(output["mediaKind"] for item in receipts for output in item["outputs"])
    document = {
        "boundary": {
            "assetsAreLocalIgnoredFiles": True,
            "doesNotAlterTemplateCandidateAssetState": True,
            "galleryApprovalClaimed": False,
            "humanReviewRequired": True,
            "publicationClaimed": False,
            "releaseEligibilityClaimed": False,
            "rightsApprovalClaimed": False,
        },
        "captureKind": "local_technical_review_candidate_receipts",
        "schemaVersion": SCHEMA_VERSION,
        "receipts": receipts,
        "summary": {
            "outputCount": sum(output_kinds.values()),
            "outputKindCounts": dict(sorted(output_kinds.items())),
            "pendingHumanReviewCount": len(receipts),
            "receiptCount": len(receipts),
        },
    }
    document["contentHash"] = canonical_content_hash(document)
    return document


def build_local_review_ledger(
    *,
    root: Path,
    records: list[dict],
    fetch_run: Callable[[str], dict],
) -> dict:
    if not records or len(records) > MAX_RECEIPTS:
        raise ValueError("Local review receipt count must be non-zero and bounded.")
    candidates = load_candidate_contracts(root)
    data_root = (root / "data").resolve()
    receipts = []
    seen_workflows = set()
    seen_tasks = set()
    seen_outputs = set()
    for record in records:
        workflow_id = record["workflowId"]
        task_id = record["taskId"]
        if workflow_id in seen_workflows or task_id in seen_tasks:
            raise ValueError("Local review workflow and task identities must be unique.")
        seen_workflows.add(workflow_id)
        seen_tasks.add(task_id)
        contract = candidates.get(workflow_id)
        if not isinstance(contract, dict):
            raise ValueError(f"Local review workflow is not a hidden candidate contract: {workflow_id}")
        expected_kind = contract.get("mediaKind")
        outputs = []
        for relative_path in record["outputs"]:
            if relative_path in seen_outputs:
                raise ValueError("Local review output paths must be globally unique.")
            seen_outputs.add(relative_path)
            outputs.append(_output_receipt(data_root, relative_path, expected_kind))
        receipts.append(
            {
                "candidateContractId": _candidate_contract_id(workflow_id),
                "captureState": "app_execution_completed_output_hashes_captured",
                "claims": {
                    "galleryApproved": False,
                    "humanQualityReviewed": False,
                    "published": False,
                    "releaseEligible": False,
                    "rightsReviewed": False,
                },
                "graph": {
                    "path": contract["graphPath"],
                    "sha256": contract["graphHash"],
                },
                "outputs": outputs,
                "reviewState": "pending_human_review",
                "task": _task_receipt(fetch_run(task_id), task_id),
                "workflowId": workflow_id,
            }
        )
    return _ledger_document(receipts)


def merge_local_review_ledger(
    *,
    root: Path,
    existing: dict,
    records: list[dict],
    fetch_run: Callable[[str], dict],
) -> dict:
    """Append newly fetched receipts after validating all retained local evidence."""

    validate_local_review_ledger(existing, root=root)
    added = build_local_review_ledger(root=root, records=records, fetch_run=fetch_run)
    receipts = [*existing["receipts"], *added["receipts"]]
    if len(receipts) > MAX_RECEIPTS:
        raise ValueError("Merged local review receipt count exceeds its bounded envelope.")
    workflow_ids = [item["workflowId"] for item in receipts]
    task_ids = [item["task"]["taskId"] for item in receipts]
    output_paths = [output["path"] for item in receipts for output in item["outputs"]]
    if (
        len(workflow_ids) != len(set(workflow_ids))
        or len(task_ids) != len(set(task_ids))
        or len(output_paths) != len(set(output_paths))
    ):
        raise ValueError("Merged local review workflow, task, and output identities must be unique.")
    document = _ledger_document(receipts)
    validate_local_review_ledger(document, root=root)
    return document


def validate_local_review_ledger(document: dict, *, root: Path | None = None) -> dict:
    if not isinstance(document, dict) or document.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("Local review receipt ledger has an unsupported schema.")
    if document.get("contentHash") != canonical_content_hash(document):
        raise ValueError("Local review receipt ledger content hash does not match.")
    receipts = document.get("receipts")
    if not isinstance(receipts, list) or not receipts or len(receipts) > MAX_RECEIPTS:
        raise ValueError("Local review receipt ledger has an invalid receipt collection.")
    expected_boundary = {
        "assetsAreLocalIgnoredFiles": True,
        "doesNotAlterTemplateCandidateAssetState": True,
        "galleryApprovalClaimed": False,
        "humanReviewRequired": True,
        "publicationClaimed": False,
        "releaseEligibilityClaimed": False,
        "rightsApprovalClaimed": False,
    }
    if document.get("boundary") != expected_boundary:
        raise ValueError("Local review receipt ledger weakens its review boundary.")
    expected_claims = {
        "galleryApproved": False,
        "humanQualityReviewed": False,
        "published": False,
        "releaseEligible": False,
        "rightsReviewed": False,
    }
    workflow_ids = set()
    task_ids = set()
    output_paths = set()
    output_kinds = Counter()
    if receipts != sorted(receipts, key=lambda item: item.get("workflowId", "")):
        raise ValueError("Local review receipts must remain in canonical workflow order.")
    if root is not None:
        candidates = load_candidate_contracts(root)
        data_root = (root / "data").resolve()
        for receipt in receipts:
            contract = candidates.get(receipt.get("workflowId"))
            if not isinstance(contract, dict):
                raise ValueError("Local review receipt no longer maps to a hidden candidate workflow.")
            if receipt.get("candidateContractId") != _candidate_contract_id(receipt["workflowId"]) or receipt.get(
                "graph"
            ) != {
                "path": contract.get("graphPath"),
                "sha256": contract.get("graphHash"),
            }:
                raise ValueError("Local review receipt graph binding is stale.")
            task = receipt.get("task")
            outputs = receipt.get("outputs")
            if (
                receipt.get("captureState") != "app_execution_completed_output_hashes_captured"
                or receipt.get("reviewState") != "pending_human_review"
                or receipt.get("claims") != expected_claims
                or not isinstance(task, dict)
                or task.get("status") != "completed"
                or not _TASK_ID.fullmatch(str(task.get("taskId") or ""))
                or not isinstance(outputs, list)
                or not outputs
                or len(outputs) > MAX_OUTPUTS_PER_RECEIPT
            ):
                raise ValueError("Local review receipt weakens its pending-review or completed-task contract.")
            workflow_id = receipt["workflowId"]
            task_id = task["taskId"]
            if workflow_id in workflow_ids or task_id in task_ids:
                raise ValueError("Local review receipt identities must remain unique.")
            workflow_ids.add(workflow_id)
            task_ids.add(task_id)
            for output in outputs:
                if not isinstance(output, dict) or output.get("path") in output_paths:
                    raise ValueError("Local review output receipts must remain unique objects.")
                output_paths.add(output["path"])
                current = _output_receipt(data_root, output["path"], contract["mediaKind"])
                if output != current:
                    raise ValueError(f"Local review output bytes drifted: {output['path']}")
                output_kinds[current["mediaKind"]] += 1
    expected_summary = {
        "outputCount": sum(output_kinds.values()),
        "outputKindCounts": dict(sorted(output_kinds.items())),
        "pendingHumanReviewCount": len(receipts),
        "receiptCount": len(receipts),
    }
    if root is not None and document.get("summary") != expected_summary:
        raise ValueError("Local review receipt summary does not match its exact outputs.")
    return document
