"""Report exact evidence availability for legacy registered Clusters.

This audit deliberately does not compile or mutate a workflow.  It separates
legacy instances that still match a current registered catalog manifest from
historical manifest identities that require a reviewed archived definition or
equivalence receipt.  Validated partial Studio-spec bodies may be returned for
manual review, but prompt text, parameter values, workflow paths, and instance
identifiers never leave the inventory in the returned report.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


RECOVERY_AUDIT_SCHEMA_VERSION = 4
_IDENTITY_REFERENCE_FIELDS = {"id", "libraryRevision", "contentHash"}
_STUDIO_EXECUTION_SPEC_FIELDS = {"id", "contentHash", "executionProfileId"}
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,511}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_STUDIO_HASH = re.compile(r"^studio-spec-v1-[0-9a-f]{8}$")
_ARCHIVED_DEFINITION_FIELDS = {
    "id",
    "libraryRevision",
    "contentHash",
    "executionAdmissions",
    "provider",
    "publisher",
    "surface",
    "ownership",
    "mutable",
    "pipelineClass",
    "workflowId",
    "graphAdapterContracts",
    "inputs",
    "outputs",
    "components",
    "steps",
    "blockContractHash",
    "rootBlockDefinitionId",
    "blockPlacements",
}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _public_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _PUBLIC_ID.fullmatch(value) is not None else None


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _definition_identity(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, Mapping):
        return None
    fields = (value.get("id"), value.get("libraryRevision"), value.get("contentHash"))
    if (
        _public_id(fields[0]) is None
        or not isinstance(fields[1], str)
        or _COMMIT.fullmatch(fields[1]) is None
        or not isinstance(fields[2], str)
        or _SHA256.fullmatch(fields[2]) is None
    ):
        return None
    return fields  # type: ignore[return-value]


def _studio_execution_spec_identity(value: Any) -> tuple[str, str, str] | None:
    if not isinstance(value, Mapping) or set(value) != _STUDIO_EXECUTION_SPEC_FIELDS:
        return None
    fields = (value.get("id"), value.get("contentHash"), value.get("executionProfileId"))
    if (
        _public_id(fields[0]) is None
        or not isinstance(fields[1], str)
        or _STUDIO_HASH.fullmatch(fields[1]) is None
        or _public_id(fields[2]) is None
    ):
        return None
    return fields  # type: ignore[return-value]


def _embedded_definition_status(value: Any) -> str:
    """Classify saved manifest material without granting it authority."""

    if not isinstance(value, Mapping) or _definition_identity(value) is None:
        return "malformed_identity_reference"
    if set(value) == _IDENTITY_REFERENCE_FIELDS:
        return "identity_reference_only"
    if not _ARCHIVED_DEFINITION_FIELDS.issubset(value):
        return "incomplete_definition_body"
    body = {key: item for key, item in value.items() if key != "contentHash"}
    if value.get("contentHash") != _canonical_hash(body):
        return "definition_body_hash_mismatch"
    return "complete_definition_body_unreviewed"


def _execution_reference_status(admission_id: Any, studio_spec: Any) -> str:
    admission = _text(admission_id)
    spec = (
        studio_spec
        if isinstance(studio_spec, tuple)
        and len(studio_spec) == 3
        and all(isinstance(item, str) and item for item in studio_spec)
        else _studio_execution_spec_identity(studio_spec)
    )
    if admission and spec:
        return "complete"
    if admission is None and studio_spec is None:
        return "missing"
    return "incomplete"


def _migration_storage_evidence(data_dir: str | Path) -> dict[str, Any]:
    """Count known migration journals/backups without returning private paths."""

    root = Path(data_dir) / "studio" / "composite-migrations"
    if not root.exists():
        return {"status": "absent", "journalManifestCount": 0, "backupFileCount": 0}
    if root.is_symlink() or not root.is_dir():
        return {"status": "unsafe", "journalManifestCount": 0, "backupFileCount": 0}
    manifest_count = 0
    backup_count = 0
    for migration in sorted(root.iterdir(), key=lambda item: item.name):
        if migration.is_symlink() or not migration.is_dir():
            continue
        manifest = migration / "manifest.json"
        if manifest.is_file() and not manifest.is_symlink():
            manifest_count += 1
        backup_root = migration / "backups"
        if backup_root.is_symlink() or not backup_root.is_dir():
            continue
        for current_root, directory_names, file_names in os.walk(backup_root, followlinks=False):
            current = Path(current_root)
            directory_names[:] = [
                name for name in sorted(directory_names) if not (current / name).is_symlink()
            ]
            backup_count += sum(
                (current / name).is_file() and not (current / name).is_symlink()
                for name in file_names
            )
    return {
        "status": "available",
        "journalManifestCount": manifest_count,
        "backupFileCount": backup_count,
    }


def _normalized_migration_storage_evidence(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {"status": "not_scanned", "journalManifestCount": 0, "backupFileCount": 0}
    if set(value) != {"status", "journalManifestCount", "backupFileCount"}:
        raise ValueError("Migration storage evidence is malformed.")
    status = value.get("status")
    counts = (value.get("journalManifestCount"), value.get("backupFileCount"))
    if status not in {"absent", "available", "unsafe"} or any(
        not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts
    ):
        raise ValueError("Migration storage evidence is malformed.")
    return dict(value)


def _library_index(
    definitions: Sequence[Mapping[str, Any]],
    *,
    require_complete_archive: bool = False,
) -> dict[tuple[str, str, str], dict[str, tuple[str, str, str] | None]]:
    result: dict[tuple[str, str, str], dict[str, tuple[str, str, str] | None]] = {}
    for definition in definitions:
        identity = _definition_identity(definition)
        if identity is None:
            continue
        if require_complete_archive:
            if not _ARCHIVED_DEFINITION_FIELDS.issubset(definition):
                raise ValueError("Archived Cluster definition is incomplete.")
            body = {key: value for key, value in definition.items() if key != "contentHash"}
            if definition.get("contentHash") != _canonical_hash(body):
                raise ValueError("Archived Cluster definition content hash does not match its complete body.")
        admissions = definition.get("executionAdmissions")
        if not isinstance(admissions, list):
            continue
        admission_specs: dict[str, tuple[str, str, str] | None] = {}
        for admission in admissions:
            if not isinstance(admission, Mapping):
                if require_complete_archive:
                    raise ValueError("Archived Cluster definition has a malformed execution admission.")
                continue
            admission_id = _public_id(admission.get("id"))
            studio_spec = _studio_execution_spec_identity(admission.get("studioExecutionSpec"))
            if admission_id is None or studio_spec is None:
                if require_complete_archive:
                    raise ValueError("Archived Cluster definition has a malformed execution admission.")
                continue
            if admission_id in admission_specs:
                raise ValueError("Cluster definition repeats an execution admission.")
            admission_specs[admission_id] = studio_spec
        previous = result.setdefault(identity, {})
        for admission_id, spec in admission_specs.items():
            if admission_id in previous and previous[admission_id] != spec:
                raise ValueError("Cluster definition repeats an execution admission with conflicting Studio identity.")
            previous[admission_id] = spec
    return result


def build_registered_cluster_recovery_audit(
    inventory: Mapping[str, Any],
    library: Mapping[str, Any],
    *,
    archived_definitions: Sequence[Mapping[str, Any]] = (),
    historical_compiler_mapping_ledger: Mapping[str, Any] | None = None,
    semantic_equivalence_ledger: Mapping[str, Any] | None = None,
    studio_spec_evidence_ledger: Mapping[str, Any] | None = None,
    studio_spec_partial_review_ledger: Mapping[str, Any] | None = None,
    migration_storage_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return aggregate exact-manifest recovery counts from detached inputs.

    ``archived_definitions`` must contain complete node-library definitions,
    including the original identity and execution admission.  Merely supplying
    an old content hash never changes recoverability. Recovered Studio-spec
    evidence and its partial reviews are reported only; neither can change a
    disposition or satisfy compiler/conversion authority.
    """

    definitions = library.get("definitions")
    if not isinstance(definitions, list):
        raise ValueError("Hugging Face node library definitions must be a list.")
    current = _library_index(definitions)
    archived = _library_index(archived_definitions, require_complete_archive=True)
    if set(current).intersection(archived):
        raise ValueError("Archived Cluster definitions must not duplicate a current manifest identity.")
    current_by_id_revision: dict[tuple[str, str], set[tuple[str, str, str]]] = defaultdict(set)
    current_by_id: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for identity in current:
        current_by_id_revision[(identity[0], identity[1])].add(identity)
        current_by_id[identity[0]].add(identity)

    compiler_mapping_keys: set[tuple[str, str, str, str, str, str, str]] = set()
    normalized_compiler_mappings: dict[str, Any] | None = None
    if historical_compiler_mapping_ledger is not None:
        from modiff.legacy_cluster_compiler_mappings import (
            validate_historical_compiler_mappings,
        )

        normalized_compiler_mappings = validate_historical_compiler_mappings(
            historical_compiler_mapping_ledger
        )
        compiler_mapping_keys = {
            (
                mapping["historical"]["manifestDefinitionId"],
                mapping["historical"]["libraryRevision"],
                mapping["historical"]["manifestContentHash"],
                mapping["historical"]["executionAdmissionId"],
                mapping["historical"]["studioExecutionSpec"]["id"],
                mapping["historical"]["studioExecutionSpec"]["contentHash"],
                mapping["historical"]["studioExecutionSpec"]["executionProfileId"],
            )
            for mapping in normalized_compiler_mappings["mappings"]
        }

    evidence_by_spec: dict[tuple[str, str, str], dict[str, Any]] = {}
    normalized_evidence: dict[str, Any] | None = None
    if studio_spec_evidence_ledger is not None:
        from modiff.legacy_cluster_studio_spec_evidence import (
            validate_studio_spec_evidence_ledger,
        )

        normalized_evidence = validate_studio_spec_evidence_ledger(studio_spec_evidence_ledger)
        evidence_by_spec = {
            (
                entry["identity"]["id"],
                entry["identity"]["contentHash"],
                entry["identity"]["executionProfileId"],
            ): entry
            for entry in normalized_evidence["specifications"]
        }

    reviews_by_execution_tuple: dict[
        tuple[str, str, str, str, str, str, str], dict[str, Any]
    ] = {}
    normalized_partial_reviews: dict[str, Any] | None = None
    if studio_spec_partial_review_ledger is not None:
        if normalized_evidence is None:
            raise ValueError("Historical Studio-spec partial reviews require their exact evidence ledger.")
        from modiff.legacy_cluster_studio_spec_evidence import (
            validate_studio_spec_partial_review_ledger,
        )

        normalized_partial_reviews = validate_studio_spec_partial_review_ledger(
            studio_spec_partial_review_ledger,
            evidence_ledger=normalized_evidence,
        )
        reviews_by_execution_tuple = {
            (
                review["historical"]["manifestDefinitionId"],
                review["historical"]["libraryRevision"],
                review["historical"]["manifestContentHash"],
                review["historical"]["executionAdmissionId"],
                review["historical"]["studioExecutionSpec"]["id"],
                review["historical"]["studioExecutionSpec"]["contentHash"],
                review["historical"]["studioExecutionSpec"]["executionProfileId"],
            ): review
            for review in normalized_partial_reviews["reviews"]
        }

    equivalence_keys: set[tuple[str, str, str, str, str, str, str]] = set()
    reviewed_receipt_count = 0
    if semantic_equivalence_ledger is not None:
        from modiff.legacy_cluster_semantic_equivalence import (
            validate_semantic_equivalence_receipts,
        )

        reviewed = validate_semantic_equivalence_receipts(semantic_equivalence_ledger)
        reviewed_receipt_count = len(reviewed["receipts"])
        equivalence_keys = {
            (
                receipt["historical"]["manifestDefinitionId"],
                receipt["historical"]["libraryRevision"],
                receipt["historical"]["manifestContentHash"],
                receipt["historical"]["executionAdmissionId"],
                receipt["historical"]["studioExecutionSpec"]["id"],
                receipt["historical"]["studioExecutionSpec"]["contentHash"],
                receipt["historical"]["studioExecutionSpec"]["executionProfileId"],
            )
            for receipt in reviewed["receipts"]
        }

    instances: list[dict[str, Any]] = []
    workflows = inventory.get("workflows")
    if not isinstance(workflows, list):
        raise ValueError("Composite migration inventory workflows must be a list.")
    for workflow in workflows:
        if not isinstance(workflow, Mapping):
            continue
        composites = workflow.get("composites")
        if not isinstance(composites, list):
            continue
        for composite in composites:
            if not isinstance(composite, Mapping) or composite.get("classification") != "legacy_cluster_root":
                continue
            refs = composite.get("sourceRefs")
            cluster_ref = refs.get("legacyCluster") if isinstance(refs, Mapping) else None
            if not isinstance(cluster_ref, Mapping):
                instances.append(
                    {
                        "identity": None,
                        "admissionId": None,
                        "studioExecutionSpec": None,
                        "embeddedDefinitionStatus": "malformed_identity_reference",
                        "projectedNodeCount": 0,
                    }
                )
                continue
            definition = cluster_ref.get("definition")
            identity = _definition_identity(definition)
            derived_child_ids = composite.get("derivedChildIds")
            instances.append(
                {
                    "identity": identity,
                    # Do not echo subordinate receipt values from a malformed
                    # manifest identity into this deliberately redacted
                    # report. Valid identities use only public identifier
                    # grammars; arbitrary workflow strings fail closed.
                    "admissionId": (
                        _public_id(cluster_ref.get("admissionId")) if identity is not None else None
                    ),
                    "studioExecutionSpec": (
                        _studio_execution_spec_identity(cluster_ref.get("studioExecutionSpec"))
                        if identity is not None
                        else None
                    ),
                    "embeddedDefinitionStatus": _embedded_definition_status(definition),
                    "projectedNodeCount": len(derived_child_ids) if isinstance(derived_child_ids, list) else 0,
                }
            )

    identity_instances: Counter[tuple[str, str, str] | None] = Counter(
        instance["identity"] for instance in instances
    )
    instances_by_identity: dict[tuple[str, str, str] | None, list[dict[str, Any]]] = defaultdict(list)
    admissions_by_identity: dict[tuple[str, str, str] | None, set[str]] = defaultdict(set)
    for instance in instances:
        identity = instance["identity"]
        admission_id = instance["admissionId"]
        instances_by_identity[identity].append(instance)
        if admission_id:
            admissions_by_identity[identity].add(admission_id)

    def exact_library_match(
        identity: tuple[str, str, str] | None,
        library_index: Mapping[tuple[str, str, str], Mapping[str, tuple[str, str, str] | None]],
    ) -> bool:
        if identity is None or identity not in library_index:
            return False
        admissions = library_index[identity]
        return all(
            instance["admissionId"] is not None
            and instance["studioExecutionSpec"] is not None
            and admissions.get(instance["admissionId"]) == instance["studioExecutionSpec"]
            for instance in instances_by_identity[identity]
        )

    def manifest_admission_match(
        identity: tuple[str, str, str] | None,
        library_index: Mapping[tuple[str, str, str], Mapping[str, tuple[str, str, str] | None]],
    ) -> bool:
        """Preserve the manifest/admission axis independently of Studio identity."""

        if identity is None or identity not in library_index:
            return False
        admissions = library_index[identity]
        return all(
            instance["admissionId"] is not None and instance["admissionId"] in admissions
            for instance in instances_by_identity[identity]
        )

    def reviewed_equivalence_match(identity: tuple[str, str, str] | None) -> bool:
        if identity is None:
            return False
        return all(
            instance["admissionId"] is not None
            and instance["studioExecutionSpec"] is not None
            and (*identity, instance["admissionId"], *instance["studioExecutionSpec"]) in equivalence_keys
            for instance in instances_by_identity[identity]
        )

    def reviewed_compiler_mapping_match(instance: Mapping[str, Any]) -> bool:
        identity = instance.get("identity")
        admission_id = instance.get("admissionId")
        studio_spec = instance.get("studioExecutionSpec")
        return (
            isinstance(identity, tuple)
            and isinstance(admission_id, str)
            and isinstance(studio_spec, tuple)
            and (*identity, admission_id, *studio_spec) in compiler_mapping_keys
        )

    def recovered_spec_entry(instance: Mapping[str, Any]) -> dict[str, Any] | None:
        studio_spec = instance.get("studioExecutionSpec")
        return evidence_by_spec.get(studio_spec) if isinstance(studio_spec, tuple) else None

    def execution_tuple_key(
        identity: tuple[str, str, str] | None,
        instance: Mapping[str, Any],
    ) -> tuple[str, str, str, str, str, str, str] | None:
        admission_id = instance.get("admissionId")
        studio_spec = instance.get("studioExecutionSpec")
        if identity is None or not isinstance(admission_id, str) or not isinstance(studio_spec, tuple):
            return None
        return (*identity, admission_id, *studio_spec)

    def recovered_execution_tuples(
        identity: tuple[str, str, str] | None,
    ) -> list[dict[str, Any]]:
        grouped: Counter[tuple[str | None, tuple[str, str, str] | None]] = Counter(
            (instance["admissionId"], instance["studioExecutionSpec"])
            for instance in instances_by_identity[identity]
        )
        rows = []
        for (admission_id, studio_spec), instance_count in sorted(
            grouped.items(),
            key=lambda item: (
                item[0][0] or "",
                "" if item[0][1] is None else "\0".join(item[0][1]),
            ),
        ):
            evidence = evidence_by_spec.get(studio_spec) if studio_spec is not None else None
            key = (
                (*identity, admission_id, *studio_spec)
                if identity is not None and admission_id is not None and studio_spec is not None
                else None
            )
            review = reviews_by_execution_tuple.get(key) if key is not None else None
            if evidence is None:
                recovered = {
                    "status": "missing",
                    "authorizesConversion": False,
                }
                manual_review = {
                    "status": "evidence_missing",
                    "authorizesConversion": False,
                }
            else:
                recovered = {
                    "status": "self_hash_valid_partial_evidence",
                    "canonicalBodySha256": evidence["canonicalBodySha256"],
                    "sourceIds": list(evidence["sourceIds"]),
                    "authorizesConversion": False,
                }
                if review is None:
                    manual_review = {
                        "status": "unreviewed",
                        "authorizesConversion": False,
                    }
                else:
                    manual_review = {
                        "status": review["decision"],
                        "reviewId": review["id"],
                        "reviewer": review["reviewer"],
                        "reviewedAt": review["reviewedAt"],
                        "notes": review["notes"],
                        "reviewHash": review["reviewHash"],
                        "authorizesConversion": False,
                    }
            rows.append(
                {
                    "admissionId": admission_id,
                    "studioExecutionSpec": (
                        dict(
                            zip(
                                ("id", "contentHash", "executionProfileId"),
                                studio_spec,
                                strict=True,
                            )
                        )
                        if studio_spec is not None
                        else None
                    ),
                    "instanceCount": instance_count,
                    "recoveredStudioSpec": recovered,
                    "manualReview": manual_review,
                }
            )
        return rows

    def disposition(identity: tuple[str, str, str] | None) -> str:
        if identity is None:
            return "identity_malformed"
        execution_states = {
            _execution_reference_status(instance["admissionId"], instance["studioExecutionSpec"])
            for instance in instances_by_identity[identity]
        }
        if execution_states == {"missing"}:
            return "execution_receipt_missing"
        if execution_states != {"complete"}:
            return "execution_receipt_incomplete"
        if exact_library_match(identity, current):
            return "current_manifest_exact"
        if exact_library_match(identity, archived):
            return "archived_manifest_exact"
        if reviewed_equivalence_match(identity):
            return "semantic_equivalence_reviewed"
        return "historical_manifest_evidence_required"

    def current_catalog_comparison(identity: tuple[str, str, str] | None) -> dict[str, Any]:
        if identity is None:
            return {"status": "identity_malformed", "candidateDefinitionCount": 0}
        if identity in current:
            return {
                "status": (
                    "exact_manifest_and_execution_tuple"
                    if exact_library_match(identity, current)
                    else "exact_manifest_execution_tuple_missing_or_mismatched"
                ),
                "candidateDefinitionCount": 1,
            }
        same_revision = current_by_id_revision.get((identity[0], identity[1]), set())
        if same_revision:
            return {
                "status": "same_definition_and_revision_different_manifest",
                "candidateDefinitionCount": len(same_revision),
            }
        same_id = current_by_id.get(identity[0], set())
        if same_id:
            return {
                "status": "same_definition_id_different_revision",
                "candidateDefinitionCount": len(same_id),
            }
        return {"status": "definition_id_absent", "candidateDefinitionCount": 0}

    def evidence_gaps(identity: tuple[str, str, str] | None, state: str) -> list[dict[str, str]]:
        if state in {
            "current_manifest_exact",
            "archived_manifest_exact",
            "semantic_equivalence_reviewed",
        }:
            return []
        gaps: list[dict[str, str]] = []
        if identity is None:
            return [
                {
                    "code": "manifest_identity_malformed",
                    "message": "The saved Cluster does not retain a complete immutable manifest identity.",
                }
            ]
        group = instances_by_identity[identity]
        execution_states = {
            _execution_reference_status(instance["admissionId"], instance["studioExecutionSpec"])
            for instance in group
        }
        if "missing" in execution_states:
            gaps.append(
                {
                    "code": "execution_receipt_missing",
                    "message": "At least one saved instance lacks both its execution admission and Studio execution-spec identity.",
                }
            )
        if "incomplete" in execution_states:
            gaps.append(
                {
                    "code": "execution_receipt_incomplete",
                    "message": "At least one saved instance retains only part of its execution admission and Studio execution-spec identity.",
                }
            )
        embedded_states = {instance["embeddedDefinitionStatus"] for instance in group}
        if "identity_reference_only" in embedded_states:
            gaps.append(
                {
                    "code": "archived_definition_body_not_embedded",
                    "message": "The workflow payload retains only id, library revision, and content hash; it does not embed the canonical manifest body.",
                }
            )
        if embedded_states.intersection({"incomplete_definition_body", "definition_body_hash_mismatch"}):
            gaps.append(
                {
                    "code": "embedded_definition_body_unusable",
                    "message": "An embedded definition fragment is incomplete or does not match its declared canonical content hash.",
                }
            )
        if any(instance["projectedNodeCount"] for instance in group):
            gaps.append(
                {
                    "code": "presentation_projection_is_not_definition_authority",
                    "message": "Expanded presentation children exist, but they do not authenticate the original manifest boundary, conditional hierarchy, or execution graph.",
                }
            )
        if any(recovered_spec_entry(instance) is not None for instance in group):
            gaps.extend(
                [
                    {
                        "code": "historical_manifest_definition_body_missing",
                        "message": "The recovered Studio specification is not the historical node-library manifest definition body.",
                    },
                    {
                        "code": "historical_public_interface_missing",
                        "message": "The recovered Studio specification does not authenticate the historical public input/output interface.",
                    },
                    {
                        "code": "historical_component_and_block_hierarchy_missing",
                        "message": "The recovered Studio specification does not authenticate the historical component and BlockDefinitionV2 hierarchy.",
                    },
                    {
                        "code": "historical_artifact_authority_missing",
                        "message": "The recovered Studio specification does not authenticate historical artifact revisions or admission authority.",
                    },
                    {
                        "code": "historical_compiler_mapping_missing",
                        "message": "No reviewed compiler mapping from this historical specification to a complete graph is registered.",
                    },
                    {
                        "code": "semantic_equivalence_not_reviewed",
                        "message": "Partial Studio-spec recovery is not a reviewed semantic-equivalence receipt.",
                    },
                ]
            )
        if identity not in archived and not reviewed_equivalence_match(identity):
            gaps.append(
                {
                    "code": "historical_manifest_authority_not_registered",
                    "message": "Neither a complete reviewed archived definition nor an exact checked-in semantic-equivalence receipt is registered.",
                }
            )
        return gaps

    def safe_next_actions(identity: tuple[str, str, str] | None, state: str) -> list[dict[str, str]]:
        if state in {
            "current_manifest_exact",
            "archived_manifest_exact",
            "semantic_equivalence_reviewed",
        }:
            return []
        if identity is None:
            return [
                {
                    "code": "restore_exact_source_record",
                    "message": "Recover a byte-exact saved workflow or migration backup containing the immutable manifest identity; do not infer it from labels or child nodes.",
                }
            ]
        actions: list[dict[str, str]] = []
        execution_states = {
            _execution_reference_status(instance["admissionId"], instance["studioExecutionSpec"])
            for instance in instances_by_identity[identity]
        }
        if execution_states != {"complete"}:
            actions.append(
                {
                    "code": "restore_exact_execution_receipt",
                    "message": "Recover the admission and Studio execution-spec identity from a byte-exact backup or original export; never select a current admission by name.",
                }
            )
        if any(
            recovered_spec_entry(instance) is not None
            for instance in instances_by_identity[identity]
        ):
            actions.append(
                {
                    "code": "review_recovered_studio_spec_partial_evidence",
                    "message": "Review the recovered Studio body and record only whether this partial evidence is internally credible. This review does not authorize conversion, execution, compiler supplementation, or semantic equivalence.",
                }
            )
        embedded_states = {instance["embeddedDefinitionStatus"] for instance in instances_by_identity[identity]}
        if "complete_definition_body_unreviewed" in embedded_states:
            actions.append(
                {
                    "code": "review_and_register_embedded_archive",
                    "message": "Validate the complete embedded body, its content hash, admissions, graph, and interface, then register it as a reviewed archived definition.",
                }
            )
        else:
            actions.append(
                {
                    "code": "recover_canonical_archived_definition",
                    "message": "Locate an exact historical node-library export or build artifact containing the complete canonical definition and register it only after hash review.",
                }
            )
        actions.append(
            {
                "code": "review_exact_semantic_equivalence",
                "message": "Alternatively compare authenticated historical graph/interface evidence to one exact V2 destination and issue a checked-in semantic-equivalence receipt after human review.",
            }
        )
        return actions

    identities = []
    for identity, instance_count in sorted(
        identity_instances.items(),
        key=lambda item: ("" if item[0] is None else "\0".join(item[0])),
    ):
        state = disposition(identity)
        group = instances_by_identity[identity]
        embedded_counts = Counter(instance["embeddedDefinitionStatus"] for instance in group)
        execution_counts = Counter(
            _execution_reference_status(instance["admissionId"], instance["studioExecutionSpec"])
            for instance in group
        )
        projection_instance_count = sum(bool(instance["projectedNodeCount"]) for instance in group)
        execution_tuples = recovered_execution_tuples(identity)
        recovered_tuples = [
            row
            for row in execution_tuples
            if row["recoveredStudioSpec"]["status"] == "self_hash_valid_partial_evidence"
        ]
        recovered_spec_hashes = {
            row["recoveredStudioSpec"]["canonicalBodySha256"] for row in recovered_tuples
        }
        identities.append(
            {
                "definitionId": identity[0] if identity else None,
                "libraryRevision": identity[1] if identity else None,
                "manifestContentHash": identity[2] if identity else None,
                "admissionIds": sorted(admissions_by_identity[identity]),
                "instanceCount": instance_count,
                "disposition": state,
                "sourceEvidence": {
                    "embeddedManifest": {
                        "identityReferenceOnlyInstanceCount": embedded_counts["identity_reference_only"],
                        "completeDefinitionBodyInstanceCount": embedded_counts[
                            "complete_definition_body_unreviewed"
                        ],
                        "incompleteOrInvalidInstanceCount": sum(
                            embedded_counts[item]
                            for item in {
                                "malformed_identity_reference",
                                "incomplete_definition_body",
                                "definition_body_hash_mismatch",
                            }
                        ),
                    },
                    "executionReceipt": {
                        "completeInstanceCount": execution_counts["complete"],
                        "missingInstanceCount": execution_counts["missing"],
                        "incompleteInstanceCount": execution_counts["incomplete"],
                    },
                    "presentationProjection": {
                        "instanceCount": projection_instance_count,
                        "nodeCount": sum(instance["projectedNodeCount"] for instance in group),
                        "isDefinitionOrExecutionAuthority": False,
                    },
                    "currentCatalog": current_catalog_comparison(identity),
                    "registeredArchive": {
                        "status": "exact" if exact_library_match(identity, archived) else "missing"
                    },
                    "reviewedSemanticEquivalence": {
                        "status": "exact" if reviewed_equivalence_match(identity) else "missing"
                    },
                    "recoveredStudioSpecs": {
                        "status": "partial_evidence_available" if recovered_tuples else "missing",
                        "specificationBodyCount": len(recovered_spec_hashes),
                        "executionTupleCount": len(recovered_tuples),
                        "instanceCount": sum(row["instanceCount"] for row in recovered_tuples),
                        "verifiedPartialReviewExecutionTupleCount": sum(
                            row["manualReview"]["status"] == "verified_partial_evidence"
                            for row in recovered_tuples
                        ),
                        "rejectedPartialReviewExecutionTupleCount": sum(
                            row["manualReview"]["status"] == "rejected_evidence"
                            for row in recovered_tuples
                        ),
                        "isManifestDefinitionOrConversionAuthority": False,
                    },
                },
                "executionTuples": execution_tuples,
                "missingEvidence": evidence_gaps(identity, state),
                "safeNextActions": safe_next_actions(identity, state),
            }
        )

    disposition_instance_counts = Counter()
    for identity, instance_count in identity_instances.items():
        disposition_instance_counts[disposition(identity)] += instance_count
    disposition_identity_counts = Counter(disposition(identity) for identity in identity_instances)
    current_manifest_exact_identities = {
        identity for identity in identity_instances if manifest_admission_match(identity, current)
    }
    current_execution_tuple_exact_identities = {
        identity for identity in identity_instances if exact_library_match(identity, current)
    }
    historical_states = {
        "historical_manifest_evidence_required",
        "execution_receipt_missing",
        "execution_receipt_incomplete",
    }
    compiler_mapping_eligible_instances = [
        instance for instance in instances if reviewed_compiler_mapping_match(instance)
    ]
    compiler_mapping_eligible_identities = {
        identity
        for identity, group in instances_by_identity.items()
        if identity is not None
        and bool(group)
        and all(reviewed_compiler_mapping_match(instance) for instance in group)
    }
    remaining_blocked_historical_instances = [
        instance
        for instance in instances
        if disposition(instance["identity"])
        not in {"current_manifest_exact", "semantic_equivalence_reviewed"}
        and not reviewed_compiler_mapping_match(instance)
    ]
    remaining_blocked_historical_identities = {
        instance["identity"] for instance in remaining_blocked_historical_instances
    }
    embedded_status_counts = Counter(instance["embeddedDefinitionStatus"] for instance in instances)
    execution_status_counts = Counter(
        _execution_reference_status(instance["admissionId"], instance["studioExecutionSpec"])
        for instance in instances
    )
    matched_recovered_instances = [
        instance for instance in instances if recovered_spec_entry(instance) is not None
    ]
    matched_recovered_spec_keys = {
        instance["studioExecutionSpec"] for instance in matched_recovered_instances
    }
    matched_recovered_identity_keys = {
        instance["identity"]
        for instance in matched_recovered_instances
        if instance["identity"] is not None
    }
    matched_recovered_execution_keys = {
        key
        for instance in matched_recovered_instances
        if (key := execution_tuple_key(instance["identity"], instance)) is not None
    }
    matched_review_decisions = {
        key: reviews_by_execution_tuple[key]
        for key in matched_recovered_execution_keys
        if key in reviews_by_execution_tuple
    }
    summary = {
        "legacyClusterInstanceCount": len(instances),
        "definitionIdentityCount": len(identity_instances),
        "currentManifestExactInstanceCount": sum(
            identity_instances[identity] for identity in current_manifest_exact_identities
        ),
        "currentManifestExactIdentityCount": len(current_manifest_exact_identities),
        "currentExecutionTupleExactInstanceCount": sum(
            identity_instances[identity] for identity in current_execution_tuple_exact_identities
        ),
        "currentExecutionTupleExactIdentityCount": len(current_execution_tuple_exact_identities),
        "archivedManifestExactInstanceCount": disposition_instance_counts["archived_manifest_exact"],
        "archivedManifestExactIdentityCount": disposition_identity_counts["archived_manifest_exact"],
        "semanticEquivalenceReviewedInstanceCount": disposition_instance_counts["semantic_equivalence_reviewed"],
        "semanticEquivalenceReviewedIdentityCount": disposition_identity_counts["semantic_equivalence_reviewed"],
        "compilerMappingEligibleInstanceCount": len(compiler_mapping_eligible_instances),
        "compilerMappingEligibleIdentityCount": len(compiler_mapping_eligible_identities),
        "remainingBlockedHistoricalInstanceCount": len(remaining_blocked_historical_instances),
        "remainingBlockedHistoricalIdentityCount": len(remaining_blocked_historical_identities),
        "historicalEvidenceRequiredInstanceCount": sum(
            disposition_instance_counts[state] for state in historical_states
        ),
        "historicalEvidenceRequiredIdentityCount": sum(
            disposition_identity_counts[state] for state in historical_states
        ),
        "completeExecutionReceiptInstanceCount": execution_status_counts["complete"],
        "missingExecutionReceiptInstanceCount": execution_status_counts["missing"],
        "missingExecutionReceiptIdentityCount": disposition_identity_counts["execution_receipt_missing"],
        "incompleteExecutionReceiptInstanceCount": execution_status_counts["incomplete"],
        "incompleteExecutionReceiptIdentityCount": disposition_identity_counts["execution_receipt_incomplete"],
        "identityReferenceOnlyInstanceCount": embedded_status_counts["identity_reference_only"],
        "embeddedCompleteDefinitionInstanceCount": embedded_status_counts[
            "complete_definition_body_unreviewed"
        ],
        "presentationProjectionInstanceCount": sum(
            bool(instance["projectedNodeCount"]) for instance in instances
        ),
        "presentationProjectionNodeCount": sum(instance["projectedNodeCount"] for instance in instances),
        "malformedIdentityInstanceCount": disposition_instance_counts["identity_malformed"],
        "malformedIdentityCount": disposition_identity_counts["identity_malformed"],
        "recoveredStudioSpecBodyCount": len(matched_recovered_spec_keys),
        "recoveredStudioSpecManifestIdentityCount": len(matched_recovered_identity_keys),
        "recoveredStudioSpecExecutionTupleCount": len(matched_recovered_execution_keys),
        "recoveredStudioSpecInstanceCount": len(matched_recovered_instances),
        "partialReviewVerifiedExecutionTupleCount": sum(
            review["decision"] == "verified_partial_evidence"
            for review in matched_review_decisions.values()
        ),
        "partialReviewRejectedExecutionTupleCount": sum(
            review["decision"] == "rejected_evidence"
            for review in matched_review_decisions.values()
        ),
    }
    expected_historical_instance_count = (
        summary["legacyClusterInstanceCount"]
        - summary["currentExecutionTupleExactInstanceCount"]
        - summary["semanticEquivalenceReviewedInstanceCount"]
    )
    expected_historical_identity_count = (
        summary["definitionIdentityCount"]
        - summary["currentExecutionTupleExactIdentityCount"]
        - summary["semanticEquivalenceReviewedIdentityCount"]
    )
    if (
        summary["compilerMappingEligibleInstanceCount"]
        + summary["remainingBlockedHistoricalInstanceCount"]
        != expected_historical_instance_count
        or summary["compilerMappingEligibleIdentityCount"]
        + summary["remainingBlockedHistoricalIdentityCount"]
        != expected_historical_identity_count
    ):
        raise ValueError("Historical compiler-mapping recovery counts are inconsistent.")
    matched_specifications = [
        {**deepcopy(evidence_by_spec[key]), "authorizesConversion": False}
        for key in sorted(matched_recovered_spec_keys)
    ]
    matched_source_ids = {
        source_id
        for entry in matched_specifications
        for source_id in entry["sourceIds"]
    }
    matched_sources = (
        [
            deepcopy(source)
            for source in normalized_evidence["sources"]
            if source["sourceId"] in matched_source_ids
        ]
        if normalized_evidence is not None
        else []
    )
    matched_reviews = [
        {**deepcopy(matched_review_decisions[key]), "authorizesConversion": False}
        for key in sorted(matched_review_decisions)
    ]
    body = {
        "schemaVersion": RECOVERY_AUDIT_SCHEMA_VERSION,
        "kind": "registered_cluster_manifest_recovery_audit",
        "boundary": {
            "readOnly": True,
            "containsWorkflowPaths": False,
            "containsInstanceIds": False,
            "containsPromptOrParameterValues": False,
            "historicalHashAloneIsNotExecutionAuthority": True,
            "presentationProjectionIsNotDefinitionAuthority": True,
            "recoveredStudioSpecEvidenceDoesNotAuthorizeConversion": True,
            "compilerMappingPresenceAloneDoesNotAuthorizeConversion": True,
        },
        "localEvidence": {
            "registeredArchivedDefinitionCount": len(archived),
            "reviewedSemanticEquivalenceReceiptCount": reviewed_receipt_count,
            "checkedInReviewedCompilerMappingCount": (
                len(normalized_compiler_mappings["mappings"])
                if normalized_compiler_mappings is not None
                else 0
            ),
            "checkedInReviewedCompilerMappingLedgerHash": (
                normalized_compiler_mappings["contentHash"]
                if normalized_compiler_mappings is not None
                else None
            ),
            "workflowEmbeddedCompleteDefinitionInstanceCount": embedded_status_counts[
                "complete_definition_body_unreviewed"
            ],
            "checkedInRecoveredStudioSpecSourceCount": (
                len(normalized_evidence["sources"]) if normalized_evidence is not None else 0
            ),
            "checkedInRecoveredStudioSpecBodyCount": (
                len(normalized_evidence["specifications"])
                if normalized_evidence is not None
                else 0
            ),
            "checkedInRecoveredStudioSpecEvidenceHash": (
                normalized_evidence["contentHash"] if normalized_evidence is not None else None
            ),
            "checkedInPartialReviewCount": (
                len(normalized_partial_reviews["reviews"])
                if normalized_partial_reviews is not None
                else 0
            ),
            "checkedInPartialReviewLedgerHash": (
                normalized_partial_reviews["contentHash"]
                if normalized_partial_reviews is not None
                else None
            ),
            "migrationStorage": _normalized_migration_storage_evidence(migration_storage_evidence),
        },
        "summary": summary,
        "partialStudioSpecEvidence": {
            "status": "matched" if matched_specifications else "no_matching_evidence",
            "authorizesConversion": False,
            "sources": matched_sources,
            "specifications": matched_specifications,
            "manualReviews": matched_reviews,
        },
        "identities": identities,
    }
    return {**deepcopy(body), "contentHash": _canonical_hash(body)}


def scan_registered_cluster_recovery_audit(
    data_dir: str,
    *,
    library: Mapping[str, Any] | None = None,
    archived_definitions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scan the bounded migration inventory and return only aggregate evidence."""

    from modiff.composite_migration_inventory import scan_composite_migration_inventory
    from modiff.huggingface_node_library import build_huggingface_node_library
    from modiff.legacy_cluster_archived_definitions import (
        reviewed_archived_cluster_definitions,
    )
    from modiff.legacy_cluster_semantic_equivalence import reviewed_semantic_equivalence_receipts
    from modiff.legacy_cluster_compiler_mappings import reviewed_historical_compiler_mappings
    from modiff.legacy_cluster_studio_spec_evidence import (
        reviewed_studio_spec_evidence,
        reviewed_studio_spec_partial_reviews,
    )

    inventory = scan_composite_migration_inventory(data_dir)
    return build_registered_cluster_recovery_audit(
        inventory,
        library if library is not None else build_huggingface_node_library(),
        archived_definitions=(
            reviewed_archived_cluster_definitions()
            if archived_definitions is None
            else archived_definitions
        ),
        historical_compiler_mapping_ledger=reviewed_historical_compiler_mappings(),
        semantic_equivalence_ledger=reviewed_semantic_equivalence_receipts(),
        studio_spec_evidence_ledger=reviewed_studio_spec_evidence(),
        studio_spec_partial_review_ledger=reviewed_studio_spec_partial_reviews(),
        migration_storage_evidence=_migration_storage_evidence(data_dir),
    )
