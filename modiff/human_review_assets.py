"""Stage local technical outputs for human review, then bind only after approval.

Staging copies existing campaign bytes into ``review-pending/``. Approval copies
the same bytes into ``review-approved/`` and writes a candidate-workflow stitch
record. It never marks Gallery approval, mutates the sealed candidate ledger,
edits public Studio templates, or publishes Dataset media.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from modiff.local_review_receipts import (
    canonical_graph_hash,
    canonical_content_hash,
    graph_execution_semantic_hash,
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
_CONTENT_TYPE_SUFFIXES = {
    "audio/wav": ".wav",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
}
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
    for material in item.get("reviewMaterials") or []:
        relative = f"{item['slug']}/{material['fileName']}"
        if material.get("mediaKind") == "image":
            label = material.get("label") or material["fileName"]
            media.append(f'<figure><img src="{relative}" alt="{label}" /><figcaption>{label}</figcaption></figure>')
    kind_label = (
        "Diffusers/Transformers graph output"
        if item.get("reviewKind") == "model_generation"
        else "Builtin media-op canary (not a diffusion template card)"
    )
    elapsed = item.get("elapsedSeconds")
    elapsed_label = f"{elapsed:.3f}s" if isinstance(elapsed, (int, float)) else "timing unavailable"
    process_rss = item.get("processRssBytes")
    rss_label = f"RSS {process_rss} bytes" if isinstance(process_rss, int) else "RSS unavailable"
    return (
        "<article>"
        f"<h2>{item['workflowId']}</h2>"
        f"<p>{kind_label}</p>"
        f"<p>Review state: <strong>{item.get('reviewState', 'unknown')}</strong></p>"
        f"<p>{item['mediaKind']} · {elapsed_label} · {rss_label}</p>"
        f"<p>Record quality only after visual/listening review: "
        f"<code>python scripts/approve_review_asset.py --workflow {item['workflowId']} "
        "--reviewer YOUR_NAME --approve-quality</code>. Rights approval is separate and is required before "
        "copying or binding.</p>"
        f"{''.join(media)}"
        "</article>"
    )


def _html_index(items: list[dict]) -> str:
    model_items = [item for item in items if item.get("reviewKind") == "model_generation"]
    builtin_items = [item for item in items if item.get("reviewKind") == "builtin_operation"]
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8" />'
        "<title>MoDiff pending review assets</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px}"
        "img,video{max-width:100%;height:auto}article{margin:0 0 32px;padding:0 0 24px;"
        "border-bottom:1px solid #ccc}code{font-size:12px}"
        "nav a{margin-right:16px}</style></head><body>"
        "<h1>Staged human-review evidence</h1>"
        "<p>These copies are not Gallery-approved and are not the 77 public Flux/Qwen/Wan "
        "templates. Raw campaign outputs are not eligible for this page. Quality approval alone never copies, "
        "binds, registers, or publishes an asset; rights approval is a separate gate.</p>"
        f'<nav><a href="#model-generations">Model generations ({len(model_items)})</a>'
        f'<a href="#builtin-operations">Builtin canaries ({len(builtin_items)})</a></nav>'
        '<h2 id="model-generations">1. Diffusers / Transformers graph outputs</h2>'
        "<p>These came from running hidden candidate graphs (SD1.5, LCM, DreamLite, Sana, "
        "SDXL Turbo, Marigold, Shap-E, LongCat, SmolLM2, SmolVLM). Review these if you want "
        "hidden examples. They are still not public Gallery cards.</p>"
        + "".join(_item_card(item) for item in model_items)
        + '<h2 id="builtin-operations">2. Builtin crop / stitch / trim canaries</h2>'
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


def _refresh_review_summary(index: dict) -> None:
    items = index.get("items") or []
    summary = index.setdefault("summary", {})
    summary["awaitingHumanReviewCount"] = sum(item.get("reviewState") == "pending_human_review" for item in items)
    summary["qualityApprovedRightsPendingCount"] = sum(
        item.get("reviewState") == "quality_approved_rights_pending" for item in items
    )
    summary["rejectedCount"] = sum(item.get("reviewState") == "rejected" for item in items)


def _persist_pending_index(root: Path, index: dict) -> None:
    index["contentHash"] = canonical_content_hash(index)
    destination = pending_root(root)
    _write_json(destination / INDEX_NAME, index)
    (destination / HTML_NAME).write_text(_html_index(index.get("items") or []), encoding="utf-8")


def record_removed_review_assets(
    root: Path,
    *,
    reason: str,
    recovery_path: str | None = None,
) -> dict:
    """Remove index cards whose staged bytes were explicitly moved or deleted.

    This is intentionally a reconciliation step, not an implicit rejection
    policy. The caller must first resolve the exact files and provide the human
    decision/reason. Removed card metadata stays in the index so cleanup is
    auditable and recoverable without presenting broken review controls.
    """

    reason = str(reason or "").strip()
    recovery_path = str(recovery_path or "").strip() or None
    if not reason:
        raise HumanReviewAssetError("Removed review assets require an explicit reason.")
    index = load_pending_index(root)
    retained = []
    removed = []
    removed_at = _now()
    for item in index.get("items") or []:
        item_dir = pending_root(root) / str(item.get("slug") or "")
        missing_assets = [
            asset.get("fileName")
            for asset in item.get("assets") or []
            if not isinstance(asset.get("fileName"), str) or not (item_dir / asset["fileName"]).is_file()
        ]
        if not missing_assets:
            retained.append(item)
            continue
        removed.append(
            {
                **item,
                "reviewState": "removed_after_review",
                "removedAt": removed_at,
                "removalReason": reason,
                "missingAssets": missing_assets,
                **({"recoveryPath": recovery_path} if recovery_path else {}),
            }
        )
    if not removed:
        return index
    previous_removed = {
        item.get("workflowId"): item
        for item in index.get("removedReviewItems") or []
        if isinstance(item, dict) and isinstance(item.get("workflowId"), str)
    }
    for item in removed:
        previous_removed[item["workflowId"]] = item
    index["items"] = retained
    index["removedReviewItems"] = sorted(previous_removed.values(), key=lambda item: item["workflowId"])
    index["blockedGeneration"] = _blocked_generation(root, {item["workflowId"] for item in retained})
    summary = index.setdefault("summary", {})
    summary["pendingCount"] = len(retained)
    summary["modelGenerationCount"] = sum(item.get("reviewKind") == "model_generation" for item in retained)
    summary["builtinOperationCount"] = sum(item.get("reviewKind") == "builtin_operation" for item in retained)
    summary["blockedGenerationCount"] = len(index["blockedGeneration"])
    summary["removedReviewItemCount"] = len(index["removedReviewItems"])
    _refresh_review_summary(index)
    _persist_pending_index(root, index)
    return index


def _load_json_document(path: Path, *, label: str) -> dict:
    if path.is_symlink() or not path.is_file():
        raise HumanReviewAssetError(f"{label} is missing or linked: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HumanReviewAssetError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(document, dict):
        raise HumanReviewAssetError(f"{label} must be a JSON object: {path}")
    return document


def _sniff_review_content_type(path: Path) -> str | None:
    with path.open("rb") as stream:
        prefix = stream.read(16)
    if prefix.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if prefix.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP":
        return "image/webp"
    if prefix.startswith(b"RIFF") and prefix[8:12] == b"WAVE":
        return "audio/wav"
    if len(prefix) >= 8 and prefix[4:8] == b"ftyp":
        return "video/mp4"
    return None


def _campaign_model_revision(output: dict) -> tuple[str | None, str | None]:
    artifact = _campaign_diffusers_model_artifact(output)
    if artifact is None:
        return None, None
    return artifact["value"], artifact["revision"]


def _campaign_diffusers_model_artifact(output: dict) -> dict | None:
    """Resolve one exact generic Diffusers loader selection from Studio evidence."""

    graph = output.get("graphSnapshot")
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, list):
        return None
    matches = []
    for node in nodes:
        data = node.get("data") if isinstance(node, dict) else None
        if not isinstance(data, dict) or data.get("action") != "LoadPipeline":
            continue
        params = data.get("params")
        if not isinstance(params, dict):
            continue
        model_field = next((params.get(key) for key in ("model_id", "model", "repo") if params.get(key)), None)
        model = model_field.get("value") if isinstance(model_field, dict) else None
        if isinstance(model, dict):
            selection = dict(model)
            model = selection.get("value")
        else:
            selection = {"source": "hub", "value": model} if isinstance(model, str) else None
        revision = params.get("revision")
        if isinstance(revision, dict):
            revision = revision.get("value", revision.get("default"))
        if isinstance(model, str) and isinstance(revision, str) and model.strip() and revision.strip():
            matches.append(
                {
                    **(selection or {}),
                    "source": (selection or {}).get("source") or "hub",
                    "value": model.strip(),
                    "revision": revision.strip(),
                }
            )
    return matches[0] if len(matches) == 1 else None


def _campaign_output_matches_workflow(output: dict, workflow_id: str) -> bool:
    parts = workflow_id.split(":")
    expected = ":".join(parts[:2])
    model_type = output.get("modelType")
    mode = output.get("mode")
    return isinstance(model_type, str) and isinstance(mode, str) and f"{model_type}:{mode}" == expected


def _load_generation_research(root: Path, workflow_id: str, fixture_provenance: dict) -> dict:
    """Require one hash-bound official-source dossier for campaign staging."""

    prompt_source = fixture_provenance.get("promptSource")
    if not isinstance(prompt_source, str) or not prompt_source.strip():
        raise HumanReviewAssetError("Campaign fixture provenance must name its generation research dossier.")
    research_root = (root / PENDING_DIRNAME / "research").resolve()
    research_path = (root / prompt_source).resolve()
    try:
        research_path.relative_to(research_root)
    except ValueError as error:
        raise HumanReviewAssetError("Generation research must come from review-pending/research.") from error
    if research_path.is_symlink() or not research_path.is_file():
        raise HumanReviewAssetError("Generation research dossier is missing or linked.")
    research = _load_json_document(research_path, label="Generation research dossier")
    sources = research.get("officialSources")
    recipe = research.get("recipe")
    if (
        research.get("kind") != "generation_recipe_research"
        or research.get("workflowId") != workflow_id
        or not isinstance(recipe, dict)
        or not recipe
        or not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(source, dict)
            or not isinstance(source.get("url"), str)
            or not source["url"].startswith(("https://", "http://"))
            or not isinstance(source.get("finding"), str)
            or not source["finding"].strip()
            for source in sources
        )
    ):
        raise HumanReviewAssetError(
            "Generation research must match the workflow and contain an exact recipe plus official source findings."
        )
    return {
        "path": str(research_path.relative_to(root)),
        "sha256": _sha256_file(research_path),
        "recipeContentHash": canonical_content_hash(recipe),
        "officialSources": sources,
    }


def _load_frozen_campaign_acceptances(receipts_root: Path | None) -> dict[tuple[str, str], list[dict]]:
    """Index immutable campaign receipts without treating historical runs as current evidence."""

    if receipts_root is None:
        return {}
    receipts_root = receipts_root.resolve()
    if not receipts_root.is_dir():
        raise HumanReviewAssetError(f"Frozen campaign receipt root is missing: {receipts_root}")
    indexed: dict[tuple[str, str], list[dict]] = {}
    for path in sorted(receipts_root.rglob("*.quality-acceptance.json")):
        if path.is_symlink() or not path.is_file():
            raise HumanReviewAssetError(f"Frozen campaign receipt is missing or linked: {path}")
        receipt = _load_json_document(path, label="Frozen campaign acceptance")
        workflow_id = receipt.get("workflowId")
        media_sha256 = receipt.get("mediaSha256")
        if (
            receipt.get("schemaVersion") != 1
            or receipt.get("kind") != "generation_quality_acceptance"
            or not isinstance(workflow_id, str)
            or not _WORKFLOW_ID.fullmatch(workflow_id)
            or not isinstance(media_sha256, str)
            or not _SHA256.fullmatch(media_sha256)
            or not isinstance(receipt.get("taskId"), str)
            or not receipt["taskId"]
            or not isinstance(receipt.get("runtimeFingerprint"), str)
            or not receipt["runtimeFingerprint"].startswith("sha256:")
        ):
            raise HumanReviewAssetError(f"Frozen campaign acceptance has an invalid execution contract: {path}")
        evidence = {
            "path": str(path.relative_to(receipts_root)),
            "sha256": _sha256_file(path),
            "receipt": receipt,
        }
        indexed.setdefault((workflow_id, media_sha256), []).append(evidence)
    return indexed


def stage_screened_campaign_asset(
    root: Path,
    *,
    workflow_id: str,
    task_id: str,
    media_path: Path,
    quality_report_path: Path,
    comparison_card_path: Path,
    fixture_provenance_path: Path,
    codex_screen_path: Path,
    studio_outputs_path: Path,
    run_document: dict,
) -> dict:
    """Stage an exact campaign output only after automated and manual screens pass.

    Passing this function never records quality approval. It creates a pending
    human-review item and keeps rights, Gallery registration, and publication
    closed.
    """

    root = root.resolve()
    contract = load_candidate_contracts(root).get(workflow_id)
    if not isinstance(contract, dict):
        raise HumanReviewAssetError(f"Candidate contract is missing for {workflow_id}.")
    slug = workflow_slug(workflow_id)

    def require_regular(path: Path, label: str) -> Path:
        resolved = path.resolve()
        if path.is_symlink() or not path.is_file():
            raise HumanReviewAssetError(f"{label} is missing or linked: {path}")
        return resolved

    media_path = require_regular(media_path, "Campaign media")
    quality_report_path = require_regular(quality_report_path, "Automated quality report")
    comparison_card_path = require_regular(comparison_card_path, "Comparison card")
    fixture_provenance_path = require_regular(fixture_provenance_path, "Fixture provenance")
    codex_screen_path = require_regular(codex_screen_path, "Codex quality screen")
    try:
        media_path.relative_to((pending_root(root) / "raw-campaign" / slug).resolve())
    except ValueError as error:
        raise HumanReviewAssetError("Campaign media must come from the workflow's raw-campaign directory.") from error

    media_sha256 = _sha256_file(media_path)
    fixture_provenance = _load_json_document(fixture_provenance_path, label="Fixture provenance")
    generation_research = _load_generation_research(root, workflow_id, fixture_provenance)
    quality = _load_json_document(quality_report_path, label="Automated quality report")
    expected_quality_kinds = {
        "edit_image": "image_edit_quality_screen",
        "image_upscale": "image_upscale_quality_screen",
        "video_upscale": "video_upscale_quality_screen",
        "text_to_image": "text_to_image_quality_screen",
    }
    expected_quality_kind = expected_quality_kinds.get(contract.get("mode"))
    if (
        expected_quality_kind is None
        or quality.get("kind") != expected_quality_kind
        or quality.get("automatedStatus") != "pass"
        or quality.get("approvalStatus") != "human_review_required"
        or not quality.get("checks")
        or not all(value is True for value in quality["checks"].values())
        or (quality.get("output") or {}).get("sha256") != media_sha256
        or (quality.get("recipe") or {}).get("recipeContentHash")
        != generation_research["recipeContentHash"]
    ):
        raise HumanReviewAssetError("Campaign output did not pass the exact hash-bound automated screen.")
    codex_screen = _load_json_document(codex_screen_path, label="Codex quality screen")
    if (
        codex_screen.get("kind") != "codex_manual_quality_screen"
        or codex_screen.get("workflowId") != workflow_id
        or codex_screen.get("mediaSha256") != media_sha256
        or codex_screen.get("outcome") != "shortlisted_for_user_review"
        or codex_screen.get("approvalStatus") != "human_review_required"
        or codex_screen.get("userApproval") != "pending"
        or codex_screen.get("rightsApproval") != "pending"
        or codex_screen.get("published") is not False
        or not codex_screen.get("checks")
        or not all(value == "pass" for value in codex_screen["checks"].values())
    ):
        raise HumanReviewAssetError("Campaign output lacks a valid hash-bound manual shortlist screen.")

    studio_document = _load_json_document(studio_outputs_path.resolve(), label="Studio output history")
    output_matches = []
    for output in studio_document.get("outputs") or []:
        if not isinstance(output, dict) or output.get("taskId") != task_id:
            continue
        if not _campaign_output_matches_workflow(output, workflow_id):
            continue
        for media_item in output.get("mediaItems") or []:
            if media_item.get("mediaHash") == f"sha256:bytes:{media_sha256}":
                output_matches.append((output, media_item))
    if len(output_matches) != 1:
        raise HumanReviewAssetError("Campaign media does not have one exact workflow/task/hash Studio output match.")
    output, media_item = output_matches[0]
    if media_item.get("byteSize") != media_path.stat().st_size:
        raise HumanReviewAssetError("Studio output byte size disagrees with the campaign media.")

    task = run_document.get("task") if isinstance(run_document, dict) else None
    runtime = task.get("runtimeFingerprint") if isinstance(task, dict) else None
    measurement = task.get("runtimeMeasurement") if isinstance(task, dict) else None
    if (
        not isinstance(task, dict)
        or task.get("task_id") != task_id
        or task.get("status") != "completed"
        or not isinstance(runtime, dict)
        or not isinstance(measurement, dict)
    ):
        raise HumanReviewAssetError("Completed task runtime evidence is missing or does not match.")

    index = load_pending_index(root)
    existing = [item for item in index.get("items") or [] if item.get("workflowId") == workflow_id]
    if existing:
        same_pending_run = (
            len(existing) == 1
            and existing[0].get("reviewState") == "pending_human_review"
            and existing[0].get("taskId") == task_id
            and len(existing[0].get("assets") or []) == 1
            and existing[0]["assets"][0].get("sha256") == media_sha256
        )
        if not same_pending_run:
            raise HumanReviewAssetError("A review item already exists for this workflow; resolve it before restaging.")

    destination = pending_root(root) / slug
    destination.mkdir(parents=True, exist_ok=True)
    staged_media = _copy_verified(media_path, destination / media_path.name, media_sha256)
    material_sources = (
        (quality_report_path, "automated_quality_report", None),
        (comparison_card_path, "comparison_card", "Matched before/after comparison"),
        (fixture_provenance_path, "fixture_provenance", None),
        (codex_screen_path, "codex_quality_screen", None),
    )
    review_materials = []
    for source, kind, label in material_sources:
        copied = _copy_verified(source, destination / source.name, _sha256_file(source))
        review_materials.append(
            {
                **copied,
                "kind": kind,
                "mediaKind": "image" if kind == "comparison_card" else "json",
                **({"label": label} if label else {}),
            }
        )

    executed_graph = output.get("graphSnapshot") or {}
    workflow_snapshot = task.get("workflow_snapshot") or {}
    graph_binding = workflow_snapshot.get("studioGraphBinding") or {}
    execution_spec = graph_binding.get("executionSpec") or {}
    model_artifact = None
    for node in executed_graph.get("nodes") or []:
        data = node.get("data") if isinstance(node, dict) else None
        if not isinstance(data, dict) or data.get("studioRole") not in {"imageUpscaler", "videoUpscaler"}:
            continue
        model_field = (data.get("params") or {}).get("model_id") or {}
        model_artifact = model_field.get("value", model_field.get("default"))
        break
    diffusers_model_artifact = _campaign_diffusers_model_artifact(output)
    if model_artifact is None:
        model_artifact = diffusers_model_artifact
    model_repo = (diffusers_model_artifact or {}).get("value") or output.get("repo")
    model_revision = (diffusers_model_artifact or {}).get("revision")
    receipt_name = f"{media_path.stem}.generation-shortlist-receipt.json"
    receipt = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "campaign_generation_shortlist_receipt",
        "workflowId": workflow_id,
        "candidateContractId": f"template-candidate:{workflow_id}",
        "canonicalGraph": {"path": contract["graphPath"], "sha256": contract["graphHash"]},
        "execution": {
            "taskId": task_id,
            "clientRunId": task.get("client_run_id"),
            "runInputHash": task.get("run_input_hash"),
            "startedAt": task.get("started_at"),
            "completedAt": task.get("completed_at"),
            "runtimeFingerprint": runtime.get("fingerprint"),
            "resourceFingerprint": runtime.get("resourceFingerprint"),
            "packages": runtime.get("packages"),
            "runtimeMeasurement": measurement,
            "executionSpecId": execution_spec.get("id"),
            "executionSpecContentHash": execution_spec.get("contentHash"),
            "executionProfileId": execution_spec.get("executionProfileId"),
            "graphFingerprint": graph_binding.get("fingerprint"),
            "executedGraphSemanticHash": graph_execution_semantic_hash(executed_graph),
            "modelType": output.get("modelType"),
            "mode": output.get("mode"),
            "modelRepo": model_repo,
            "modelRevision": model_revision,
            "modelArtifact": model_artifact,
            "device": measurement.get("device"),
            "backend": measurement.get("backend"),
        },
        "input": fixture_provenance,
        "research": generation_research,
        "output": {
            "path": f"{PENDING_DIRNAME}/{slug}/{media_path.name}",
            "sha256": media_sha256,
            "byteSize": media_path.stat().st_size,
            "contentType": media_item.get("contentType") or _sniff_review_content_type(media_path),
            "width": media_item.get("width") or (quality.get("outputSize") or {}).get("width"),
            "height": media_item.get("height") or (quality.get("outputSize") or {}).get("height"),
            "studioOutputId": output.get("id"),
            "studioBackendPath": media_item.get("backendPath"),
        },
        "quality": {
            "automatedScreen": quality,
            "codexScreen": codex_screen,
            "comparisonCard": next(
                f"{PENDING_DIRNAME}/{slug}/{item['fileName']}"
                for item in review_materials
                if item["kind"] == "comparison_card"
            ),
        },
        "outcome": "shortlisted_for_user_review",
        "userApproval": "pending",
        "rightsApproval": "pending",
        "galleryRegistered": False,
        "datasetPublished": False,
        "turnoverEligible": False,
    }
    receipt["contentHash"] = canonical_content_hash(receipt)
    _write_json(destination / receipt_name, receipt)

    item = {
        "workflowId": workflow_id,
        "candidateContractId": f"template-candidate:{workflow_id}",
        "slug": slug,
        "mediaKind": contract["mediaKind"],
        "graphPath": contract["graphPath"],
        "graphHash": contract["graphHash"],
        "reviewKind": review_kind_for_workflow(workflow_id),
        "reviewState": "pending_human_review",
        "elapsedSeconds": measurement.get("elapsedSeconds"),
        "processRssBytes": measurement.get("processRssBytes"),
        "taskId": task_id,
        "assets": [
            {
                **staged_media,
                "mediaKind": contract["mediaKind"],
                "contentType": media_item.get("contentType") or _sniff_review_content_type(media_path),
                "sourcePath": str(media_item.get("backendPath") or ""),
                "generationReceipt": f"{PENDING_DIRNAME}/{slug}/{receipt_name}",
            }
        ],
        "reviewMaterials": review_materials,
        "galleryRegistered": False,
        "publicTemplateIds": [],
    }
    index["items"] = [value for value in index.get("items") or [] if value.get("workflowId") != workflow_id]
    index["items"].append(item)
    index["items"].sort(key=lambda value: value["workflowId"])
    index["blockedGeneration"] = _blocked_generation(root, {value["workflowId"] for value in index["items"]})
    summary = index.setdefault("summary", {})
    summary["pendingCount"] = len(index["items"])
    summary["modelGenerationCount"] = sum(
        value.get("reviewKind") == "model_generation" for value in index["items"]
    )
    summary["builtinOperationCount"] = sum(
        value.get("reviewKind") == "builtin_operation" for value in index["items"]
    )
    summary["blockedGenerationCount"] = len(index["blockedGeneration"])
    _refresh_review_summary(index)
    _persist_pending_index(root, index)
    return {"item": item, "receipt": receipt, "index": index}


def reconcile_screened_campaign_graph(
    root: Path,
    *,
    workflow_id: str,
    prior_graph: dict,
) -> dict:
    """Rebind a pending shortlist after a non-execution canonical graph change."""

    root = root.resolve()
    index = load_pending_index(root)
    item = _item_for_workflow(index, workflow_id)
    if item.get("reviewState") != "pending_human_review":
        raise HumanReviewAssetError("Only a still-pending shortlist may reconcile its canonical graph.")
    assets = item.get("assets") or []
    if len(assets) != 1 or not isinstance(assets[0].get("generationReceipt"), str):
        raise HumanReviewAssetError("Pending shortlist does not have one generation receipt.")
    receipt_path = root / assets[0]["generationReceipt"]
    receipt = _load_json_document(receipt_path, label="Generation shortlist receipt")
    if receipt.get("kind") != "campaign_generation_shortlist_receipt":
        raise HumanReviewAssetError("Only a campaign generation shortlist receipt may be reconciled.")
    contracts = load_candidate_contracts(root)
    contract = contracts.get(workflow_id)
    if not isinstance(contract, dict) or contract.get("graphPath") != item.get("graphPath"):
        raise HumanReviewAssetError("Current candidate contract does not match the pending workflow graph path.")
    old_hash = (receipt.get("canonicalGraph") or {}).get("sha256")
    if canonical_graph_hash(prior_graph) != old_hash:
        raise HumanReviewAssetError("Prior graph bytes do not match the shortlist's recorded canonical hash.")
    current_path = root / "data" / "graphs" / contract["graphPath"]
    current_graph = _load_json_document(current_path, label="Current canonical graph")
    if canonical_graph_hash(current_graph) != contract.get("graphHash"):
        raise HumanReviewAssetError("Current canonical graph bytes do not match the candidate contract.")
    prior_semantic_hash = graph_execution_semantic_hash(prior_graph)
    current_semantic_hash = graph_execution_semantic_hash(current_graph)
    if prior_semantic_hash != current_semantic_hash:
        raise HumanReviewAssetError("Canonical graph execution semantics changed; a new live run is required.")
    new_hash = contract["graphHash"]
    receipt["canonicalGraph"] = {"path": contract["graphPath"], "sha256": new_hash}
    receipt["canonicalGraphReconciliation"] = {
        "priorSha256": old_hash,
        "currentSha256": new_hash,
        "executionSemanticHash": current_semantic_hash,
        "reason": "Canonical field availability/default metadata changed without changing executable node values or edges.",
    }
    receipt["contentHash"] = canonical_content_hash(receipt)
    _write_json(receipt_path, receipt)
    item["graphHash"] = new_hash
    _persist_pending_index(root, index)
    return {"item": item, "receipt": receipt, "index": index}


def reconcile_campaign_review_assets(
    root: Path,
    *,
    studio_outputs_path: Path,
    user_review_path: Path,
    workflow_ids: set[str] | None = None,
    frozen_receipts_root: Path | None = None,
) -> dict:
    """Bind exact campaign bytes and human quality decisions into the review index.

    Studio output history is accepted only when its byte hash and canonical
    model/mode identity both match the reviewed file. Missing task-level runtime
    measurements remain explicit receipt gaps; rights and publication are never
    inferred from a quality-only decision.
    """

    root = root.resolve()
    studio_outputs_path = studio_outputs_path.resolve()
    user_review_path = user_review_path.resolve()
    studio_document = _load_json_document(studio_outputs_path, label="Studio output history")
    review_document = _load_json_document(user_review_path, label="User quality review")
    outputs = studio_document.get("outputs")
    decisions = review_document.get("results")
    boundary = review_document.get("boundary")
    if not isinstance(outputs, list):
        raise HumanReviewAssetError("Studio output history must contain an outputs array.")
    if not isinstance(decisions, list) or not isinstance(boundary, dict):
        raise HumanReviewAssetError("User quality review must contain results and a boundary object.")
    if boundary.get("rightsApproved") is not False or boundary.get("datasetPublished") is not False:
        raise HumanReviewAssetError("Campaign reconciliation requires an explicit quality-only review boundary.")

    selected = {str(item) for item in workflow_ids} if workflow_ids else None
    frozen_acceptances = _load_frozen_campaign_acceptances(frozen_receipts_root)
    candidates = load_candidate_contracts(root)
    index = load_pending_index(root)
    items_by_workflow = {
        item.get("workflowId"): item
        for item in index.get("items") or []
        if isinstance(item, dict) and isinstance(item.get("workflowId"), str)
    }
    output_matches: dict[str, list[tuple[dict, dict]]] = {}
    for output in outputs:
        if not isinstance(output, dict):
            continue
        for media_item in output.get("mediaItems") or []:
            if not isinstance(media_item, dict):
                continue
            media_hash = str(media_item.get("mediaHash") or "")
            if media_hash.startswith("sha256:bytes:") and _SHA256.fullmatch(media_hash[13:]):
                output_matches.setdefault(media_hash[13:], []).append((output, media_item))

    reconciled = []
    skipped = []
    for decision in decisions:
        if not isinstance(decision, dict) or decision.get("outcome") != "quality_approved_rights_pending":
            continue
        workflow_id = decision.get("workflowId")
        if not isinstance(workflow_id, str) or (selected is not None and workflow_id not in selected):
            continue
        if workflow_id not in candidates:
            skipped.append({"workflowId": workflow_id, "reason": "candidate_contract_missing"})
            continue
        media_relative = decision.get("mediaPath")
        if not isinstance(media_relative, str):
            skipped.append({"workflowId": workflow_id, "reason": "review_media_path_missing"})
            continue
        media_parts = Path(media_relative).parts
        expected_slug = workflow_slug(workflow_id)
        if (
            Path(media_relative).is_absolute()
            or ".." in media_parts
            or len(media_parts) != 2
            or media_parts[0] != expected_slug
        ):
            raise HumanReviewAssetError(f"Review media path is outside its workflow directory: {media_relative}")
        media_path = pending_root(root) / media_relative
        if media_path.is_symlink() or not media_path.is_file():
            skipped.append({"workflowId": workflow_id, "reason": "review_media_missing", "mediaPath": media_relative})
            continue
        media_sha256 = _sha256_file(media_path)
        exact_matches = [
            match
            for match in output_matches.get(media_sha256, [])
            if _campaign_output_matches_workflow(match[0], workflow_id)
        ]
        if not exact_matches:
            skipped.append(
                {
                    "workflowId": workflow_id,
                    "reason": "exact_studio_output_not_found",
                    "mediaPath": media_relative,
                    "mediaSha256": media_sha256,
                }
            )
            continue
        exact_matches.sort(key=lambda match: int(match[0].get("createdAt") or 0))
        output, media_item = exact_matches[0]
        if media_item.get("byteSize") != media_path.stat().st_size:
            raise HumanReviewAssetError(f"Studio output byte size disagrees with reviewed media: {media_relative}")
        sniffed_content_type = _sniff_review_content_type(media_path)
        declared_content_type = media_item.get("contentType")
        if (
            isinstance(declared_content_type, str)
            and sniffed_content_type is not None
            and declared_content_type != sniffed_content_type
        ):
            raise HumanReviewAssetError(f"Studio output MIME disagrees with reviewed media bytes: {media_relative}")
        content_type = sniffed_content_type or declared_content_type
        expected_suffix = _CONTENT_TYPE_SUFFIXES.get(content_type)
        extension_matches = expected_suffix is None or media_path.suffix.lower() == expected_suffix
        model_repo, model_revision = _campaign_model_revision(output)
        graph_snapshot = output.get("graphSnapshot")
        try:
            executed_graph_hash = graph_execution_semantic_hash(graph_snapshot)
        except ValueError:
            executed_graph_hash = None
        evidence_gaps = [
            "task_runtime_measurement_missing",
            "runtime_fingerprint_missing",
            "rights_approval_pending",
        ]
        if not extension_matches:
            evidence_gaps.append("filename_extension_mime_mismatch")
        if not model_repo or not model_revision:
            evidence_gaps.append("exact_model_revision_missing")

        frozen_matches = frozen_acceptances.get((workflow_id, media_sha256), [])
        if len(frozen_matches) > 1:
            raise HumanReviewAssetError(
                f"Multiple frozen campaign acceptances claim the same workflow bytes: {workflow_id}"
            )
        frozen_evidence = frozen_matches[0] if frozen_matches else None
        if frozen_evidence is not None:
            frozen = frozen_evidence["receipt"]
            if frozen.get("taskId") != output.get("taskId"):
                raise HumanReviewAssetError(
                    f"Frozen campaign acceptance task disagrees with the exact Studio output: {workflow_id}"
                )
            if frozen.get("byteSize") != media_path.stat().st_size or frozen.get("mime") != content_type:
                raise HumanReviewAssetError(
                    f"Frozen campaign acceptance media metadata disagrees with reviewed bytes: {workflow_id}"
                )
            frozen_measurement = frozen.get("runtimeMeasurement")
            if (
                isinstance(frozen_measurement, dict)
                and isinstance(frozen_measurement.get("elapsedSeconds"), (int, float))
                and isinstance(frozen_measurement.get("processRssBytes"), int)
            ):
                evidence_gaps = [
                    gap
                    for gap in evidence_gaps
                    if gap not in {"task_runtime_measurement_missing", "runtime_fingerprint_missing"}
                ]

        contract = candidates[workflow_id]
        receipt_name = f"{media_path.stem}.generation-review-receipt.json"
        receipt_relative = f"{PENDING_DIRNAME}/{expected_slug}/{receipt_name}"
        receipt = {
            "schemaVersion": SCHEMA_VERSION,
            "kind": "campaign_generation_review_receipt",
            "workflowId": workflow_id,
            "candidateContractId": f"template-candidate:{workflow_id}",
            "canonicalGraph": {"path": contract["graphPath"], "sha256": contract["graphHash"]},
            "execution": {
                "taskId": output.get("taskId"),
                "clientRunId": output.get("clientRunId"),
                "runInputHash": output.get("runInputHash"),
                "createdAt": output.get("createdAt"),
                "modelType": output.get("modelType"),
                "mode": output.get("mode"),
                "modelRepo": model_repo or output.get("repo"),
                "modelRevision": model_revision,
                "executedGraphSemanticHash": executed_graph_hash,
                "prompt": output.get("prompt"),
                "negativePrompt": output.get("negativePrompt"),
                "seed": output.get("seed"),
                "width": output.get("width"),
                "height": output.get("height"),
                "steps": output.get("steps"),
                "guidanceScale": output.get("guidanceScale"),
                **(
                    {
                        "runtimeFingerprint": frozen_evidence["receipt"]["runtimeFingerprint"],
                        "runtimeMeasurement": frozen_evidence["receipt"].get("runtimeMeasurement"),
                        "executionSpecId": frozen_evidence["receipt"].get("executionSpecId"),
                        "executionSpecContentHash": frozen_evidence["receipt"].get("executionSpecHash"),
                        "executionProfileId": frozen_evidence["receipt"].get("executionProfileId"),
                    }
                    if frozen_evidence is not None
                    else {}
                ),
            },
            "output": {
                "path": f"{PENDING_DIRNAME}/{media_relative}",
                "sha256": media_sha256,
                "byteSize": media_path.stat().st_size,
                "contentType": content_type,
                "filenameExtensionMatchesMime": extension_matches,
                "studioOutputId": output.get("id"),
                "studioBackendPath": media_item.get("backendPath"),
            },
            "qualityReview": {
                "reviewer": review_document.get("reviewer"),
                "reviewedAt": review_document.get("reviewedAt"),
                "outcome": "approved",
                "reason": decision.get("reason"),
            },
            **(
                {
                    "frozenCampaignAcceptance": {
                        "path": frozen_evidence["path"],
                        "sha256": frozen_evidence["sha256"],
                        "sourceCommit": frozen_evidence["receipt"].get("sourceCommit"),
                        "clientCommit": frozen_evidence["receipt"].get("clientCommit"),
                    }
                }
                if frozen_evidence is not None
                else {}
            ),
            "rightsApproved": False,
            "galleryRegistered": False,
            "datasetPublished": False,
            "turnoverEligible": False,
            "evidenceGaps": evidence_gaps,
        }
        receipt["contentHash"] = canonical_content_hash(receipt)
        _write_json(media_path.parent / receipt_name, receipt)

        asset = {
            "byteSize": media_path.stat().st_size,
            "fileName": media_path.name,
            "sha256": media_sha256,
            "mediaKind": contract["mediaKind"],
            "sourcePath": f"studio/outputs/{Path(str(media_item.get('backendPath') or '')).name}",
            "contentType": content_type,
            "filenameExtensionMatchesMime": extension_matches,
            "generationReceipt": receipt_relative,
        }
        item = items_by_workflow.get(workflow_id)
        if item is None:
            item = {
                "workflowId": workflow_id,
                "candidateContractId": f"template-candidate:{workflow_id}",
                "slug": expected_slug,
                "mediaKind": contract["mediaKind"],
                "graphPath": contract["graphPath"],
                "graphHash": contract["graphHash"],
                "reviewKind": review_kind_for_workflow(workflow_id),
                "elapsedSeconds": None,
                "processRssBytes": None,
                "taskId": output.get("taskId"),
                "assets": [],
                "galleryRegistered": False,
                "publicTemplateIds": [],
            }
            items_by_workflow[workflow_id] = item
        existing_assets = {entry.get("sha256") for entry in item.get("assets") or [] if isinstance(entry, dict)}
        if media_sha256 not in existing_assets:
            item.setdefault("assets", []).append(asset)
        item["reviewState"] = "quality_approved_rights_pending"
        item["qualityReview"] = {
            "reviewer": review_document.get("reviewer"),
            "outcome": "approved",
            "reviewedAt": review_document.get("reviewedAt"),
            "rightsApproved": False,
            "galleryRegistered": False,
            "datasetPublished": False,
        }
        item["technicalEvidenceState"] = "incomplete"
        item["technicalEvidenceGaps"] = evidence_gaps
        if frozen_evidence is not None and isinstance(
            frozen_evidence["receipt"].get("runtimeMeasurement"), dict
        ):
            measurement = frozen_evidence["receipt"]["runtimeMeasurement"]
        else:
            measurement = None
        if (
            isinstance(measurement, dict)
            and isinstance(measurement.get("elapsedSeconds"), (int, float))
            and isinstance(measurement.get("processRssBytes"), int)
        ):
            item["elapsedSeconds"] = measurement["elapsedSeconds"]
            item["processRssBytes"] = measurement["processRssBytes"]
        reconciled.append({"workflowId": workflow_id, "taskId": output.get("taskId"), "mediaSha256": media_sha256})

    index["items"] = sorted(items_by_workflow.values(), key=lambda item: item["workflowId"])
    staged_ids = {item["workflowId"] for item in index["items"]}
    index["blockedGeneration"] = _blocked_generation(root, staged_ids)
    summary = index.setdefault("summary", {})
    summary["pendingCount"] = len(index["items"])
    summary["modelGenerationCount"] = sum(item.get("reviewKind") == "model_generation" for item in index["items"])
    summary["builtinOperationCount"] = sum(item.get("reviewKind") == "builtin_operation" for item in index["items"])
    summary["blockedGenerationCount"] = len(index["blockedGeneration"])
    _refresh_review_summary(index)
    _persist_pending_index(root, index)
    return {"reconciled": reconciled, "skipped": skipped, "index": index}


def reject_review_asset(root: Path, *, workflow_id: str, reviewer: str, reason: str) -> dict:
    reviewer = str(reviewer or "").strip()
    reason = str(reason or "").strip()
    if not reviewer or not reason:
        raise HumanReviewAssetError("Rejection requires a reviewer name and a reason.")
    index = load_pending_index(root)
    item = _item_for_workflow(index, workflow_id)
    item["reviewState"] = "rejected"
    item.pop("qualityReview", None)
    item["rejection"] = {"reviewer": reviewer, "reason": reason, "rejectedAt": _now()}
    _refresh_review_summary(index)
    _persist_pending_index(root, index)
    return item


def record_quality_review(
    root: Path,
    *,
    workflow_id: str,
    reviewer: str,
    approved: bool,
    reason: str = "",
) -> dict:
    """Record quality independently without claiming rights or publication approval."""

    reviewer = str(reviewer or "").strip()
    reason = str(reason or "").strip()
    if not reviewer:
        raise HumanReviewAssetError("Quality review requires an explicit reviewer name.")
    if not approved:
        return reject_review_asset(root, workflow_id=workflow_id, reviewer=reviewer, reason=reason)

    index = load_pending_index(root)
    item = _item_for_workflow(index, workflow_id)
    if item.get("reviewState") == "rejected":
        raise HumanReviewAssetError("Rejected assets require a new staged copy before quality approval.")
    reviewed_at = _now()
    item["reviewState"] = "quality_approved_rights_pending"
    item.pop("rejection", None)
    item["qualityReview"] = {
        "reviewer": reviewer,
        "outcome": "approved",
        "reviewedAt": reviewed_at,
        "rightsApproved": False,
        "galleryRegistered": False,
        "datasetPublished": False,
    }
    _refresh_review_summary(index)
    _persist_pending_index(root, index)
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
    _refresh_review_summary(index)
    _persist_pending_index(root, index)
    return binding
