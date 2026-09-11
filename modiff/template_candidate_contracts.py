"""Deterministic authoring contracts for canonical workflows without public templates.

The ledger produced here is deliberately non-public and non-executable.  It
binds future template authoring to canonical graph truth while leaving prompts,
template defaults, input examples, generated assets, runtime evidence, and
quality review outside the claims made by the contract.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from modiff.comfy_template_research import (
    ComfyTemplateResearchError,
    validate_comfy_template_research_ledger,
)


TEMPLATE_CANDIDATE_CONTRACT_SCHEMA_VERSION = 1
TEMPLATE_CANDIDATE_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "template-candidate-contracts.v1.json"
)

_WORKFLOW_MANIFEST_PATH = "data/workflow-library-manifest.json"
_UPSTREAM_COVERAGE_PATH = "data/upstream-coverage.v1.json"
_COMFY_RESEARCH_PATH = "data/research/comfy-workflow-catalog.v1.json"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_BINDING_FIELDS = (
    "pipelineClasses",
    "requiredArtifacts",
    "requiredInputs",
    "qualificationStatus",
    "graphQualificationStatus",
    "runtimeQualificationStatus",
    "optimizationQualificationStatus",
    "qualifiedRuntimeProfiles",
    "qualificationScope",
)
_MEDIA_INPUT_FIELDS = ("requiredImages", "requiredVideos", "requiredAudio")
_NON_MEDIA_INPUT_FIELDS = {"minimumCounts", "modelRequirements", "note"}
_AUTHORING_STATE_WITH_INPUTS = {
    "status": "authoring_required",
    "prompt": "required_if_applicable",
    "defaults": "required",
    "inputExamples": "required",
}
_AUTHORING_STATE_WITHOUT_INPUTS = {
    **_AUTHORING_STATE_WITH_INPUTS,
    "inputExamples": "not_applicable",
}
_EVIDENCE_STATE = {
    "maximumClaim": "canonical_graph_binding_only",
    "execution": "not_claimed",
    "runtime": "not_claimed",
    "quality": "not_claimed",
}
_BOUNDARY = {
    "studioVisible": False,
    "publicTemplate": False,
    "importsComfyGraphs": False,
    "executesComfyNodes": False,
    "downloadsModelsOrMedia": False,
    "generatesMedia": False,
    "allowsZeroAssets": True,
    "comfyMappingMeaning": "semantic_comparison_candidate_only",
    "qualificationMetadataMeaning": "exact_manifest_binding_not_candidate_evidence",
    "maximumEvidenceClaim": "canonical_graph_binding_only",
}


class TemplateCandidateContractError(RuntimeError):
    """Raised when candidate sources or the checked-in ledger drift."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_hash(value: Mapping[str, Any]) -> str:
    semantic = dict(value)
    semantic.pop("contentHash", None)
    return "sha256:" + _sha256_bytes(_stable_json(semantic).encode("utf-8"))


def _read_json(root: Path, relative_path: str, *, label: str) -> tuple[dict[str, Any], str]:
    path = root / relative_path
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise TemplateCandidateContractError(f"{label} is unavailable or invalid: {relative_path}") from error
    if not isinstance(value, dict):
        raise TemplateCandidateContractError(f"{label} must be a JSON object.")
    return value, "sha256:" + _sha256_bytes(raw)


def _records(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise TemplateCandidateContractError(f"{label} must be a list of objects.")
    return value


def _strings(value: Any, *, label: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise TemplateCandidateContractError(f"{label} must contain non-empty strings.")
    if not allow_empty and not value:
        raise TemplateCandidateContractError(f"{label} must not be empty.")
    if len(value) != len(set(value)):
        raise TemplateCandidateContractError(f"{label} must not contain duplicates.")
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TemplateCandidateContractError(f"{label} must be a non-empty string.")
    return value


def _canonical_json_sha256(path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TemplateCandidateContractError(f"Canonical graph is unavailable or invalid: {path}") from error
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def _workflow_records(root: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], set[str]]:
    supported = _records(manifest.get("workflows"), label="supported canonical workflows")
    experimental = _records(
        manifest.get("experimentalWorkflows"),
        label="experimental canonical workflows",
    )
    records: dict[str, dict[str, Any]] = {}
    supported_ids: set[str] = set()
    graph_root = (root / "data" / "graphs").resolve(strict=True)
    for status, rows in (("supported", supported), ("experimental", experimental)):
        for row in rows:
            workflow_id = _string(row.get("id"), label="canonical workflow id")
            if workflow_id in records:
                raise TemplateCandidateContractError(f"Duplicate canonical workflow id: {workflow_id}")
            if status == "supported":
                supported_ids.add(workflow_id)
            for field in ("modelType", "mode", "mediaKind", "graphPath", "graphHash"):
                _string(row.get(field), label=f"canonical workflow {workflow_id} {field}")
            graph_hash = row["graphHash"]
            if _SHA256.fullmatch(graph_hash) is None:
                raise TemplateCandidateContractError(f"Canonical workflow {workflow_id} has an invalid graph hash.")
            try:
                graph_path = (graph_root / row["graphPath"]).resolve(strict=True)
            except OSError as error:
                raise TemplateCandidateContractError(
                    f"Canonical workflow {workflow_id} has no checked-in graph."
                ) from error
            if not graph_path.is_relative_to(graph_root) or not graph_path.is_file():
                raise TemplateCandidateContractError(
                    f"Canonical workflow {workflow_id} graph escapes the checked-in graph root."
                )
            if _canonical_json_sha256(graph_path) != graph_hash:
                raise TemplateCandidateContractError(f"Canonical workflow {workflow_id} graph hash is stale.")

            _strings(
                row.get("pipelineClasses"),
                label=f"canonical workflow {workflow_id} pipeline classes",
                allow_empty=False,
            )
            _strings(
                row.get("requiredArtifacts"),
                label=f"canonical workflow {workflow_id} required artifacts",
            )
            required_inputs = row.get("requiredInputs")
            if not isinstance(required_inputs, (list, dict)):
                raise TemplateCandidateContractError(
                    f"Canonical workflow {workflow_id} required inputs must be a list or object."
                )
            if isinstance(required_inputs, list):
                _strings(required_inputs, label=f"canonical workflow {workflow_id} required inputs")
            else:
                unknown_input_fields = set(required_inputs) - set(_MEDIA_INPUT_FIELDS) - _NON_MEDIA_INPUT_FIELDS
                if unknown_input_fields:
                    raise TemplateCandidateContractError(
                        f"Canonical workflow {workflow_id} has unclassified required-input fields: "
                        f"{sorted(unknown_input_fields)}"
                    )
                for field in _MEDIA_INPUT_FIELDS:
                    if field in required_inputs:
                        _strings(
                            required_inputs[field],
                            label=f"canonical workflow {workflow_id} {field}",
                            allow_empty=False,
                        )
                minimum_counts = required_inputs.get("minimumCounts", {})
                declared_media_fields = {
                    item
                    for media_field in _MEDIA_INPUT_FIELDS
                    for item in required_inputs.get(media_field, [])
                }
                if not isinstance(minimum_counts, Mapping) or any(
                    not isinstance(field, str)
                    or field not in declared_media_fields
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or not 1 <= count <= 64
                    for field, count in minimum_counts.items()
                ):
                    raise TemplateCandidateContractError(
                        f"Canonical workflow {workflow_id} minimumCounts must bind declared media fields to integers from 1 to 64."
                    )
            for field in (
                "qualificationStatus",
                "graphQualificationStatus",
                "runtimeQualificationStatus",
                "optimizationQualificationStatus",
                "qualificationScope",
            ):
                _string(row.get(field), label=f"canonical workflow {workflow_id} {field}")
            _strings(
                row.get("qualifiedRuntimeProfiles"),
                label=f"canonical workflow {workflow_id} qualified runtime profiles",
            )
            records[workflow_id] = row
    return records, supported_ids


def _coverage_partition(
    coverage: Mapping[str, Any],
    workflow_records: Mapping[str, Mapping[str, Any]],
) -> tuple[set[str], set[str], int]:
    if coverage.get("schemaVersion") != 1:
        raise TemplateCandidateContractError("The upstream coverage schema is unsupported.")
    if coverage.get("contentHash") != _content_hash(coverage):
        raise TemplateCandidateContractError("The upstream coverage content hash is invalid.")
    coverage_rows = _records(coverage.get("canonicalWorkflows"), label="coverage canonical workflows")
    coverage_by_id: dict[str, dict[str, Any]] = {}
    public_ids_by_workflow: dict[str, list[str]] = {}
    for row in coverage_rows:
        workflow_id = _string(row.get("id"), label="coverage canonical workflow id")
        if workflow_id in coverage_by_id:
            raise TemplateCandidateContractError(f"Duplicate coverage workflow id: {workflow_id}")
        coverage_by_id[workflow_id] = row
        manifest_row = workflow_records.get(workflow_id)
        if manifest_row is None:
            raise TemplateCandidateContractError(f"Coverage names unknown canonical workflow {workflow_id}.")
        for field in ("modelType", "mode", "graphPath", "graphHash", "qualificationStatus"):
            if row.get(field) != manifest_row.get(field):
                raise TemplateCandidateContractError(
                    f"Coverage field {field!r} drifted for canonical workflow {workflow_id}."
                )
        public_ids_by_workflow[workflow_id] = sorted(
            _strings(
                row.get("publicTemplateIds"),
                label=f"coverage public template ids for {workflow_id}",
            )
        )
    if set(coverage_by_id) != set(workflow_records):
        missing = sorted(set(workflow_records) - set(coverage_by_id))
        raise TemplateCandidateContractError(f"Coverage omits canonical workflows: {missing}")

    templates = _records(coverage.get("publicTemplates"), label="coverage public templates")
    template_ids: set[str] = set()
    expected_ids_by_workflow: dict[str, list[str]] = defaultdict(list)
    for template in templates:
        template_id = _string(template.get("id"), label="public template id")
        if template_id in template_ids:
            raise TemplateCandidateContractError(f"Duplicate public template id: {template_id}")
        template_ids.add(template_id)
        workflow_id = _string(template.get("canonicalWorkflowId"), label=f"public template {template_id} workflow")
        manifest_row = workflow_records.get(workflow_id)
        if manifest_row is None:
            raise TemplateCandidateContractError(f"Public template {template_id} names an unknown workflow.")
        if template.get("modelType") != manifest_row.get("modelType") or template.get("mode") != manifest_row.get(
            "mode"
        ):
            raise TemplateCandidateContractError(f"Public template {template_id} identity drifted from its workflow.")
        expected_ids_by_workflow[workflow_id].append(template_id)
    for workflow_id in workflow_records:
        if public_ids_by_workflow[workflow_id] != sorted(expected_ids_by_workflow[workflow_id]):
            raise TemplateCandidateContractError(
                f"Public template partition is inconsistent for canonical workflow {workflow_id}."
            )

    public_workflows = {workflow_id for workflow_id, ids in public_ids_by_workflow.items() if ids}
    missing_workflows = set(workflow_records) - public_workflows
    summary = coverage.get("summary")
    if not isinstance(summary, Mapping):
        raise TemplateCandidateContractError("The upstream coverage summary is missing.")
    expected_summary = {
        "canonicalWorkflowCount": len(workflow_records),
        "publicTemplateCount": len(templates),
        "canonicalWorkflowsWithPublicTemplates": len(public_workflows),
        "canonicalWorkflowsWithoutPublicTemplates": len(missing_workflows),
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise TemplateCandidateContractError(f"Upstream coverage summary field {field!r} is stale.")
    return public_workflows, missing_workflows, len(templates)


def _input_examples_required(required_inputs: list[Any] | Mapping[str, Any]) -> bool:
    if isinstance(required_inputs, list):
        return bool(required_inputs)
    return any(bool(required_inputs.get(field)) for field in _MEDIA_INPUT_FIELDS)


def _comfy_mappings(
    catalog: Mapping[str, Any],
    *,
    supported_workflow_ids: set[str],
) -> dict[str, list[dict[str, str]]]:
    try:
        validate_comfy_template_research_ledger(
            catalog,
            supported_workflow_ids=supported_workflow_ids,
        )
    except ComfyTemplateResearchError as error:
        raise TemplateCandidateContractError("The checked-in Comfy research catalog is invalid.") from error
    mappings: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record_type, field in (("template", "templates"), ("blueprint", "blueprints")):
        for row in _records(catalog.get(field), label=f"Comfy research {field}"):
            mapping = row.get("candidateMapping")
            if not isinstance(mapping, Mapping) or not isinstance(mapping.get("moDiffWorkflowId"), str):
                continue
            mappings[mapping["moDiffWorkflowId"]].append(
                {
                    "recordType": record_type,
                    "id": _string(row.get("id"), label=f"Comfy research {record_type} id"),
                    "mappingStatus": _string(
                        row.get("mappingStatus"),
                        label=f"Comfy research {record_type} mapping status",
                    ),
                }
            )
    for rows in mappings.values():
        rows.sort(key=lambda item: (item["recordType"], item["id"]))
    return mappings


def build_template_candidate_contract_ledger(root: Path) -> dict[str, Any]:
    """Build candidate contracts from checked-in canonical and research truth."""

    root = root.resolve(strict=True)
    manifest, manifest_hash = _read_json(root, _WORKFLOW_MANIFEST_PATH, label="workflow manifest")
    coverage, coverage_file_hash = _read_json(root, _UPSTREAM_COVERAGE_PATH, label="upstream coverage")
    comfy, comfy_hash = _read_json(root, _COMFY_RESEARCH_PATH, label="Comfy research catalog")
    workflow_records, supported_workflow_ids = _workflow_records(root, manifest)
    public_workflows, missing_workflows, public_template_count = _coverage_partition(
        coverage,
        workflow_records,
    )
    if public_workflows & missing_workflows or public_workflows | missing_workflows != set(workflow_records):
        raise TemplateCandidateContractError("Canonical workflows do not form an exact public/candidate partition.")
    comfy_mappings = _comfy_mappings(comfy, supported_workflow_ids=supported_workflow_ids)

    contracts = []
    for workflow_id in sorted(missing_workflows):
        row = workflow_records[workflow_id]
        required_inputs = row["requiredInputs"]
        contract = {
            "canonicalWorkflowId": workflow_id,
            "modelType": row["modelType"],
            "mode": row["mode"],
            "mediaKind": row["mediaKind"],
            "graphPath": row["graphPath"],
            "graphHash": row["graphHash"],
            **{field: deepcopy(row[field]) for field in _MANIFEST_BINDING_FIELDS},
            "authoringState": deepcopy(
                _AUTHORING_STATE_WITH_INPUTS
                if _input_examples_required(required_inputs)
                else _AUTHORING_STATE_WITHOUT_INPUTS
            ),
            "assetState": "not_generated",
            "assets": [],
            "publicationState": "hidden_candidate",
            "evidenceState": deepcopy(_EVIDENCE_STATE),
            "comfyResearchRecords": deepcopy(comfy_mappings.get(workflow_id, [])),
        }
        contracts.append(contract)

    media_counts = Counter(contract["mediaKind"] for contract in contracts)
    comfy_status_counts = Counter(
        record["mappingStatus"] for contract in contracts for record in contract["comfyResearchRecords"]
    )
    ledger: dict[str, Any] = {
        "schemaVersion": TEMPLATE_CANDIDATE_CONTRACT_SCHEMA_VERSION,
        "contractKind": "non_public_template_candidate_contract_ledger",
        "boundary": deepcopy(_BOUNDARY),
        "sources": {
            "workflowManifest": {"path": _WORKFLOW_MANIFEST_PATH, "sha256": manifest_hash},
            "upstreamCoverage": {
                "path": _UPSTREAM_COVERAGE_PATH,
                "sha256": coverage_file_hash,
                "contentHash": coverage["contentHash"],
            },
            "comfyResearchCatalog": {
                "path": _COMFY_RESEARCH_PATH,
                "sha256": comfy_hash,
                "revision": comfy.get("source", {}).get("revision"),
            },
        },
        "summary": {
            "canonicalWorkflowCount": len(workflow_records),
            "publicTemplateCount": public_template_count,
            "canonicalWorkflowsWithPublicTemplates": len(public_workflows),
            "candidateContractCount": len(contracts),
            "mediaKindCounts": {key: media_counts[key] for key in sorted(media_counts)},
            "contractsRequiringInputExamples": sum(
                contract["authoringState"]["inputExamples"] == "required" for contract in contracts
            ),
            "contractsWithComfyResearchRecords": sum(bool(contract["comfyResearchRecords"]) for contract in contracts),
            "comfyResearchRecordCount": sum(len(contract["comfyResearchRecords"]) for contract in contracts),
            "comfyMappingStatusCounts": {
                key: comfy_status_counts[key] for key in sorted(comfy_status_counts)
            },
        },
        "contracts": contracts,
    }
    ledger["contentHash"] = _content_hash(ledger)
    return ledger


def validate_template_candidate_contract_ledger(
    ledger: Mapping[str, Any],
    *,
    root: Path,
) -> dict[str, Any]:
    """Validate ledger integrity and rebuild it to detect every source drift."""

    if ledger.get("schemaVersion") != TEMPLATE_CANDIDATE_CONTRACT_SCHEMA_VERSION:
        raise TemplateCandidateContractError("The template candidate contract schema is unsupported.")
    if ledger.get("contractKind") != "non_public_template_candidate_contract_ledger":
        raise TemplateCandidateContractError("Template candidate contracts must remain explicitly non-public.")
    if ledger.get("contentHash") != _content_hash(ledger):
        raise TemplateCandidateContractError("The template candidate contract content hash is invalid.")
    expected = build_template_candidate_contract_ledger(root)
    if dict(ledger) != expected:
        raise TemplateCandidateContractError(
            "The template candidate contract ledger is stale, incomplete, or internally inconsistent."
        )
    return expected


def render_template_candidate_contract_ledger(ledger: Mapping[str, Any]) -> str:
    return json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def load_template_candidate_contract_ledger(
    path: Path = TEMPLATE_CANDIDATE_CONTRACT_PATH,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    try:
        ledger = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TemplateCandidateContractError(
            "The checked-in template candidate contract ledger is unavailable or invalid."
        ) from error
    if not isinstance(ledger, dict):
        raise TemplateCandidateContractError("The template candidate contract ledger must be a JSON object.")
    return validate_template_candidate_contract_ledger(
        ledger,
        root=root if root is not None else Path(__file__).resolve().parents[1],
    )
