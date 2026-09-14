"""Inventory remaining generic-task gaps that still need a bounded MoDiff contract.

Image stitch was the first model-free Comfy-derived task implemented as an
original builtin. This module lists what is left. It does not add model-named
nodes, admit artifacts, or claim algorithm parity with Comfy.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from modiff.local_review_receipts import canonical_content_hash


SCHEMA_VERSION = 1
RESOLUTION_PATH = Path("data/research/comfy-contract-resolution.v1.json")
IMPLEMENTED_WITHOUT_WEIGHTS = {
    "comfy-research:template:utility_image_stitch": {
        "canonicalWorkflowId": "BuiltinImageOperation:image_stitch",
        "status": "implemented_hidden_candidate_pending_human_review",
        "note": (
            "Original MoDiff image-stitch builtin exists. Local technical output is staged for "
            "human quality and rights review. Not claimed equivalent to the Comfy graph."
        ),
    }
}
_BOUNDARY = {
    "addsModelNamedNodes": False,
    "admitsArtifacts": False,
    "claimsComfyParity": False,
    "authorsNewGenericTasks": False,
    "requiresHumanTaskSelection": True,
}


class GenericTaskGapInventoryError(ValueError):
    """Raised when the remaining generic-task inventory cannot be classified."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_resolutions(root: Path, resolution_ledger: dict | None) -> list[dict]:
    if resolution_ledger is None:
        path = root / RESOLUTION_PATH
        if path.is_symlink() or not path.is_file():
            raise GenericTaskGapInventoryError("Comfy contract-resolution ledger is missing.")
        resolution_ledger = json.loads(path.read_text(encoding="utf-8"))
    resolutions = resolution_ledger.get("resolutions")
    if not isinstance(resolutions, list) or not resolutions:
        raise GenericTaskGapInventoryError("Comfy contract-resolution ledger has no resolutions.")
    return resolutions


def _gap_record(resolution: dict) -> dict:
    contract_id = resolution.get("contractId")
    state = resolution.get("resolutionState")
    mode = resolution.get("selectedCandidateMode")
    if not isinstance(contract_id, str) or not isinstance(state, str) or not isinstance(mode, str):
        raise GenericTaskGapInventoryError("Resolution record is missing contractId, mode, or state.")
    implemented = IMPLEMENTED_WITHOUT_WEIGHTS.get(contract_id)
    if implemented is not None:
        authorable = False
        next_gate = "human_quality_and_rights_review"
        needs_weights = False
    elif state == "existing_task_boundary_builtin_operation_review_required":
        authorable = False
        next_gate = "builtin_algorithm_parity_review_not_a_new_task"
        needs_weights = False
    elif state == "new_task_boundary_required":
        authorable = False
        next_gate = "author_generic_task_then_admit_model"
        needs_weights = True
    else:
        raise GenericTaskGapInventoryError(
            f"Resolution {contract_id} is not a remaining generic-task gap ({state})."
        )
    return {
        "contractId": contract_id,
        "catalogId": resolution.get("catalogId"),
        "title": resolution.get("title"),
        "selectedCandidateMode": mode,
        "resolutionState": state,
        "mappingMeaning": resolution.get("mappingMeaning"),
        "needsWeights": needs_weights,
        "authorableWithoutWeights": authorable,
        "implementedWithoutWeights": implemented is not None,
        "canonicalWorkflowId": None if implemented is None else implemented["canonicalWorkflowId"],
        "nextGate": next_gate,
        "note": None if implemented is None else implemented["note"],
        "blockers": list(resolution.get("blockers") or []),
    }


def build_generic_task_gap_inventory(root: Path, *, resolution_ledger: dict | None = None) -> dict:
    root = root.resolve()
    resolutions = _load_resolutions(root, resolution_ledger)
    gaps = []
    for resolution in resolutions:
        state = resolution.get("resolutionState")
        contract_id = resolution.get("contractId")
        if state == "new_task_boundary_required" or state == "existing_task_boundary_builtin_operation_review_required":
            gaps.append(_gap_record(resolution))
        elif contract_id in IMPLEMENTED_WITHOUT_WEIGHTS:
            gaps.append(_gap_record(resolution))
    if not gaps:
        raise GenericTaskGapInventoryError("No remaining generic-task gaps were found.")

    mode_counts = Counter(item["selectedCandidateMode"] for item in gaps if item["needsWeights"])
    next_without_weights = [item["contractId"] for item in gaps if item["authorableWithoutWeights"]]
    implemented = [item["contractId"] for item in gaps if item["implementedWithoutWeights"]]
    builtin_parity = [
        item["contractId"]
        for item in gaps
        if item["resolutionState"] == "existing_task_boundary_builtin_operation_review_required"
        and not item["implementedWithoutWeights"]
    ]
    ledger = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "generic_task_gap_inventory",
        "boundary": dict(_BOUNDARY),
        "policy": {
            "noModelNamedNodes": True,
            "noWeightAdmissionHere": True,
            "imageStitchIsTheImplementedModelFreePattern": True,
            "remainingNewTasksNeedModels": True,
        },
        "diffusersSideContractsStillNeedingGenericTasks": [
            {
                "family": "DiffusionGemma",
                "candidateTask": "diffusion_text",
                "needsWeights": True,
                "authorableWithoutWeights": False,
                "nextGate": "author_generic_diffusion_text_contract_then_heavy_remote_execution",
                "note": (
                    "Artifact review is sealed and remote-only. A future node must stay task-generic "
                    "and must not be named DiffusionGemma."
                ),
            }
        ],
        "gaps": gaps,
        "inventoriedAt": _now(),
        "summary": {
            "gapCount": len(gaps),
            "newTaskBoundaryCount": sum(item["resolutionState"] == "new_task_boundary_required" for item in gaps),
            "builtinParityReviewCount": len(builtin_parity),
            "implementedWithoutWeightsCount": len(implemented),
            "authorableWithoutWeightsCount": len(next_without_weights),
            "newTaskModeCounts": dict(sorted(mode_counts.items())),
            "nextAuthorableWithoutWeights": next_without_weights,
            "implementedWithoutWeights": implemented,
            "remainingBuiltinParityReviews": builtin_parity,
        },
    }
    if ledger["summary"]["authorableWithoutWeightsCount"] != 0:
        raise GenericTaskGapInventoryError(
            "Inventory claimed a weight-free generic task remains; classify it before authoring."
        )
    ledger["contentHash"] = canonical_content_hash(ledger)
    return ledger


def render_generic_task_gap_html(ledger: dict) -> str:
    groups: dict[str, list[dict]] = {}
    for item in ledger["gaps"]:
        groups.setdefault(item["selectedCandidateMode"], []).append(item)
    sections = []
    for mode, rows in sorted(groups.items()):
        cards = "".join(
            "<article>"
            f"<h3>{row['catalogId'] or row['contractId']}</h3>"
            f"<p>{row['title']}</p>"
            f"<p>Gate: {row['nextGate']}</p>"
            f"<p>Needs weights: {str(row['needsWeights']).lower()}</p>"
            "</article>"
            for row in rows
        )
        sections.append(f"<h2>{mode} — {len(rows)}</h2>{cards}")
    extra = "".join(
        f"<p>{item['family']}: {item['candidateTask']} — {item['note']}</p>"
        for item in ledger["diffusersSideContractsStillNeedingGenericTasks"]
    )
    summary = ledger["summary"]
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\" />"
        "<title>MoDiff remaining generic-task gaps</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px}"
        "article{margin:0 0 16px;padding:0 0 12px;border-bottom:1px solid #ccc}</style></head><body>"
        "<h1>Remaining generic-task gaps</h1>"
        "<p>No new generic task can be authored on this host without a model admission. "
        f"Image stitch is already implemented ({summary['implementedWithoutWeightsCount']}). "
        f"{summary['newTaskBoundaryCount']} Comfy-derived boundaries still need a new task plus a model. "
        "Do not add model-named nodes.</p>"
        "<h2>Diffusers-side</h2>"
        + extra
        + "".join(sections)
        + "</body></html>\n"
    )


def write_generic_task_gap_inventory(root: Path, destination: Path | None = None) -> dict:
    ledger = build_generic_task_gap_inventory(root)
    target = destination if destination is not None else root / "review-pending" / "generic-task-gaps"
    _write_json(target / "inventory.v1.json", ledger)
    (target / "index.html").write_text(render_generic_task_gap_html(ledger), encoding="utf-8")
    return ledger
