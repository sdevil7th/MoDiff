"""Stage local technical outputs for human review, then bind only after approval.

Staging copies existing campaign bytes into ``review-pending/``. Approval copies
the same bytes into ``review-approved/`` and writes a candidate-workflow stitch
record. It never marks Gallery approval, mutates the sealed candidate ledger,
edits public Studio templates, or publishes Dataset media.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from modiff.local_review_receipts import (
    canonical_content_hash,
    load_candidate_contracts,
    validate_local_review_ledger,
)


SCHEMA_VERSION = 1
PENDING_DIRNAME = "review-pending"
APPROVED_DIRNAME = "review-approved"
INDEX_NAME = "index.json"
BINDINGS_NAME = "bindings.v1.json"
HTML_NAME = "index.html"
_WORKFLOW_ID = re.compile(r"^[A-Za-z0-9_.-]+:[a-z0-9_]+(?::[A-Za-z0-9_.-]+)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_LEDGERS = (
    Path("data/qualification/local-review/technical-candidates.v1.json"),
    Path("data/qualification/local-review/image-stitch-candidate.v1.json"),
)
_BOUNDARY = {
    "galleryRegistered": False,
    "datasetPublished": False,
    "publicTemplateMutated": False,
    "candidateLedgerMutated": False,
    "requiresExplicitApproveQualityAndRights": True,
}


class HumanReviewAssetError(ValueError):
    """Raised when the review queue or an approval stitch is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def workflow_slug(workflow_id: str) -> str:
    if not _WORKFLOW_ID.fullmatch(workflow_id):
        raise HumanReviewAssetError(f"Invalid canonical workflow ID: {workflow_id}")
    return workflow_id.replace(":", "__")


def pending_root(root: Path) -> Path:
    return root / PENDING_DIRNAME


def approved_root(root: Path) -> Path:
    return root / APPROVED_DIRNAME


def load_receipt_ledgers(root: Path, ledger_paths: tuple[Path, ...] = DEFAULT_LEDGERS) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for relative in ledger_paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        validate_local_review_ledger(document, root=root)
        for receipt in document["receipts"]:
            merged[receipt["workflowId"]] = receipt
    if not merged:
        raise HumanReviewAssetError("No validated local-review receipts were found to stage.")
    return merged


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> dict:
    if source.is_symlink() or not source.is_file():
        raise HumanReviewAssetError(f"Review source is missing or linked: {source}")
    if not _SHA256.fullmatch(expected_sha256):
        raise HumanReviewAssetError("Review source SHA-256 is malformed.")
    if _sha256_file(source) != expected_sha256:
        raise HumanReviewAssetError(f"Review source bytes drifted: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if destination.is_symlink() or _sha256_file(destination) != expected_sha256:
        raise HumanReviewAssetError(f"Staged review copy is unsafe or drifted: {destination}")
    return {
        "byteSize": destination.stat().st_size,
        "fileName": destination.name,
        "sha256": expected_sha256,
    }


def _blocked_generation(root: Path, staged_ids: set[str]) -> list[dict]:
    path = root / "data" / "template-authoring-specs.v1.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    blocked = []
    for specification in payload.get("specifications") or []:
        if specification.get("authoringState") != "draft_complete_execution_pending":
            continue
        workflow_id = specification.get("canonicalWorkflowId")
        if not isinstance(workflow_id, str) or workflow_id in staged_ids:
            continue
        artifacts = (specification.get("artifactPlan") or {}).get("requiredArtifacts") or []
        blocked.append(
            {
                "workflowId": workflow_id,
                "reason": "heavy_or_uncached_model_not_run_on_this_host",
                "requiredArtifacts": list(artifacts),
                "generated": False,
            }
        )
    return sorted(blocked, key=lambda item: item["workflowId"])


def review_kind_for_workflow(workflow_id: str) -> str:
    return "builtin_operation" if workflow_id.startswith("Builtin") else "model_generation"


def _item_card(item: dict) -> str:
    media = []
    for asset in item["assets"]:
        relative = f"{item['slug']}/{asset['fileName']}"
        kind = asset["mediaKind"]
        if kind == "image":
            media.append(f'<img src="{relative}" alt="{item["workflowId"]}" />')
        elif kind == "video":
            media.append(f'<video controls src="{relative}"></video>')
        elif kind == "audio":
            media.append(f'<audio controls src="{relative}"></audio>')
        else:
            media.append(f'<a href="{relative}">{asset["fileName"]}</a>')
    kind_label = (
        "Diffusers/Transformers graph output"
        if item.get("reviewKind") == "model_generation"
        else "Builtin media-op canary (not a diffusion template card)"
    )
    return (
        "<article>"
        f"<h2>{item['workflowId']}</h2>"
        f"<p>{kind_label}</p>"
        f"<p>{item['mediaKind']} · {item['elapsedSeconds']:.3f}s · RSS {item['processRssBytes']} bytes</p>"
        f"<p>Approve only after visual/listening review: "
        f"<code>python scripts/approve_review_asset.py --workflow {item['workflowId']} "
        "--reviewer YOUR_NAME --approve-quality --approve-rights</code></p>"
        f"{''.join(media)}"
        "</article>"
    )


def _html_index(items: list[dict]) -> str:
    model_items = [item for item in items if item.get("reviewKind") == "model_generation"]
    builtin_items = [item for item in items if item.get("reviewKind") == "builtin_operation"]
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\" />"
        "<title>MoDiff pending review assets</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px}"
        "img,video{max-width:100%;height:auto}article{margin:0 0 32px;padding:0 0 24px;"
        "border-bottom:1px solid #ccc}code{font-size:12px}"
        "nav a{margin-right:16px}</style></head><body>"
        "<h1>Pending human review</h1>"
        "<p>These copies are not Gallery-approved and are not the 77 public Flux/Qwen/Wan "
        "templates. Approving one only keeps it as a hidden-candidate example after quality "
        "and rights both pass.</p>"
        f"<nav><a href=\"#model-generations\">Model generations ({len(model_items)})</a>"
        f"<a href=\"#builtin-operations\">Builtin canaries ({len(builtin_items)})</a></nav>"
        "<h2 id=\"model-generations\">1. Diffusers / Transformers graph outputs</h2>"
        "<p>These came from running hidden candidate graphs (SD1.5, LCM, DreamLite, Sana, "
        "SDXL Turbo, Marigold, Shap-E, LongCat, SmolLM2, SmolVLM). Review these if you want "
        "hidden examples. They are still not public Gallery cards.</p>"
        + "".join(_item_card(item) for item in model_items)
        + "<h2 id=\"builtin-operations\">2. Builtin crop / stitch / trim canaries</h2>"
        "<p>These are deterministic media ops on campaign test stills, not text-to-image "
        "templates. A 2×2 stitch of yellow squares is the expected stitch output. Skip this "
        "section unless you want those ops as hidden examples.</p>"
        + "".join(_item_card(item) for item in builtin_items)
        + "</body></html>\n"
    )


def stage_review_assets(root: Path, *, ledger_paths: tuple[Path, ...] = DEFAULT_LEDGERS) -> dict:
    root = root.resolve()
    receipts = load_receipt_ledgers(root, ledger_paths)
    candidates = load_candidate_contracts(root)
    destination = pending_root(root)
    destination.mkdir(parents=True, exist_ok=True)
    items = []
    for workflow_id, receipt in sorted(receipts.items()):
        contract = candidates[workflow_id]
        slug = workflow_slug(workflow_id)
        item_dir = destination / slug
        if item_dir.exists():
            shutil.rmtree(item_dir)
        assets = []
        for output in receipt["outputs"]:
            copied = _copy_verified(
                root / "data" / output["path"],
                item_dir / Path(output["path"]).name,
                output["sha256"],
            )
            assets.append(
                {
                    **copied,
                    "mediaKind": output["mediaKind"],
                    "sourcePath": output["path"],
                }
            )
        items.append(
            {
                "workflowId": workflow_id,
                "candidateContractId": receipt["candidateContractId"],
                "slug": slug,
                "mediaKind": contract["mediaKind"],
                "graphPath": receipt["graph"]["path"],
                "graphHash": receipt["graph"]["sha256"],
                "reviewKind": review_kind_for_workflow(workflow_id),
                "reviewState": "pending_human_review",
                "elapsedSeconds": receipt["task"]["elapsedSeconds"],
                "processRssBytes": receipt["task"]["processRssBytes"],
                "taskId": receipt["task"]["taskId"],
                "assets": assets,
                "galleryRegistered": False,
                "publicTemplateIds": [],
            }
        )
    document = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "local_human_review_queue",
        "boundary": dict(_BOUNDARY),
        "howToApprove": (
            "python scripts/approve_review_asset.py --workflow WORKFLOW_ID "
            "--reviewer YOUR_NAME --approve-quality --approve-rights"
        ),
        "items": items,
        "blockedGeneration": _blocked_generation(root, {item["workflowId"] for item in items}),
        "stagedAt": _now(),
        "summary": {
            "blockedGenerationCount": 0,
            "pendingCount": len(items),
            "modelGenerationCount": sum(item["reviewKind"] == "model_generation" for item in items),
            "builtinOperationCount": sum(item["reviewKind"] == "builtin_operation" for item in items),
        },
    }
    document["summary"]["blockedGenerationCount"] = len(document["blockedGeneration"])
    document["contentHash"] = canonical_content_hash(document)
    _write_json(destination / INDEX_NAME, document)
    (destination / HTML_NAME).write_text(_html_index(items), encoding="utf-8")
    return document


def load_pending_index(root: Path) -> dict:
    path = pending_root(root) / INDEX_NAME
    if not path.is_file() or path.is_symlink():
        raise HumanReviewAssetError("Review-pending index is missing. Stage assets first.")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != SCHEMA_VERSION or document.get("kind") != "local_human_review_queue":
        raise HumanReviewAssetError("Review-pending index has an unsupported schema.")
    if document.get("boundary") != _BOUNDARY:
        raise HumanReviewAssetError("Review-pending index weakens the approval boundary.")
    if document.get("contentHash") != canonical_content_hash(document):
        raise HumanReviewAssetError("Review-pending index content hash does not match.")
    return document


def load_bindings(root: Path) -> dict:
    path = approved_root(root) / BINDINGS_NAME
    if not path.is_file():
        document = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "local_human_review_bindings",
            "boundary": dict(_BOUNDARY),
            "bindings": [],
        }
        document["contentHash"] = canonical_content_hash(document)
        return document
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("contentHash") != canonical_content_hash(document):
        raise HumanReviewAssetError("Approved binding ledger content hash does not match.")
    if document.get("boundary") != _BOUNDARY:
        raise HumanReviewAssetError("Approved binding ledger weakens the approval boundary.")
    return document


def _item_for_workflow(index: dict, workflow_id: str) -> dict:
    matches = [item for item in index.get("items") or [] if item.get("workflowId") == workflow_id]
    if len(matches) != 1:
        raise HumanReviewAssetError(f"Workflow is not in the pending review queue: {workflow_id}")
    return matches[0]


def reject_review_asset(root: Path, *, workflow_id: str, reviewer: str, reason: str) -> dict:
    reviewer = str(reviewer or "").strip()
    reason = str(reason or "").strip()
    if not reviewer or not reason:
        raise HumanReviewAssetError("Rejection requires a reviewer name and a reason.")
    index = load_pending_index(root)
    item = _item_for_workflow(index, workflow_id)
    item["reviewState"] = "rejected"
    item["rejection"] = {"reviewer": reviewer, "reason": reason, "rejectedAt": _now()}
    index["contentHash"] = canonical_content_hash(index)
    _write_json(pending_root(root) / INDEX_NAME, index)
    return item


def approve_review_asset(
    root: Path,
    *,
    workflow_id: str,
    reviewer: str,
    approve_quality: bool,
    approve_rights: bool,
) -> dict:
    reviewer = str(reviewer or "").strip()
    if not reviewer:
        raise HumanReviewAssetError("Approval requires an explicit reviewer name.")
    if not approve_quality or not approve_rights:
        raise HumanReviewAssetError("Approval requires both --approve-quality and --approve-rights.")
    index = load_pending_index(root)
    item = _item_for_workflow(index, workflow_id)
    if item.get("reviewState") == "rejected":
        raise HumanReviewAssetError("Rejected assets cannot be approved without a new staged copy.")
    bindings = load_bindings(root)
    if any(entry.get("workflowId") == workflow_id for entry in bindings["bindings"]):
        raise HumanReviewAssetError(f"Workflow is already bound after approval: {workflow_id}")

    slug = item["slug"]
    source_dir = pending_root(root) / slug
    destination_dir = approved_root(root) / slug
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    approved_assets = []
    for asset in item["assets"]:
        copied = _copy_verified(
            source_dir / asset["fileName"],
            destination_dir / asset["fileName"],
            asset["sha256"],
        )
        approved_assets.append(
            {
                **copied,
                "mediaKind": asset["mediaKind"],
                "approvedPath": f"{APPROVED_DIRNAME}/{slug}/{asset['fileName']}",
            }
        )

    binding = {
        "workflowId": workflow_id,
        "candidateContractId": item["candidateContractId"],
        "graphPath": item["graphPath"],
        "graphHash": item["graphHash"],
        "reviewer": reviewer,
        "approvedAt": _now(),
        "qualityApproved": True,
        "rightsApproved": True,
        "stitchState": "bound_to_candidate_awaiting_public_template",
        "galleryRegistered": False,
        "datasetPublished": False,
        "publicTemplateIds": list(item.get("publicTemplateIds") or []),
        "assets": approved_assets,
    }
    bindings["bindings"] = sorted([*bindings["bindings"], binding], key=lambda entry: entry["workflowId"])
    bindings["contentHash"] = canonical_content_hash(bindings)
    _write_json(approved_root(root) / BINDINGS_NAME, bindings)
    _write_json(destination_dir / "stitch.json", binding)

    item["reviewState"] = "approved_bound_to_candidate"
    item["approval"] = {
        "reviewer": reviewer,
        "approvedAt": binding["approvedAt"],
        "bindingPath": f"{APPROVED_DIRNAME}/{BINDINGS_NAME}",
    }
    index["contentHash"] = canonical_content_hash(index)
    _write_json(pending_root(root) / INDEX_NAME, index)
    return binding
