import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modiff.block_definition_v2 import (
    block_definition_canonical_sha256_v2,
    block_definition_content_hash_v2,
    block_graph_hash_v2,
    block_interface_hash_v2,
    validate_block_definition_v2,
    validate_block_instance_v2,
)
from modiff.composite_migration import (
    APPLY_CONFIRMATION,
    ROLLBACK_CONFIRMATION,
    CompositeMigrationAuthorizationError,
    CompositeMigrationConflictError,
    apply_composite_migration,
    build_composite_migration_preview,
    composite_migration_status,
    legacy_registered_cluster_composite_hash,
    migrate_user_block_definition_v1,
    migrate_workflow_document_v1,
    rollback_composite_migration,
    scan_composite_migration_preview,
)
from modiff.legacy_cluster_semantic_equivalence import (
    canonical_content_hash as semantic_equivalence_content_hash,
    validate_semantic_equivalence_receipt,
)
from tests.test_composite_migration_inventory import param, v1_definition, v2_definition, workflow_fixture


FAKE_REGISTERED_ADMISSION = "qwen-admission"


def registered_cluster_workflow():
    workflow = workflow_fixture()
    root = next(node for node in workflow["snapshot"]["nodes"] if node["id"] == "cluster-root")
    child = next(node for node in workflow["snapshot"]["nodes"] if node["id"] == "cluster-generate")
    prompt = root["data"]["params"]["prompt"]["value"]
    child["data"]["params"]["prompt"]["value"] = prompt
    return workflow


def registered_cluster_compiler_fixture(workflow, source_sha256: str):
    root = next(node for node in workflow["snapshot"]["nodes"] if node["id"] == "cluster-root")
    prompt = root["data"]["params"]["prompt"]["value"]

    definition = v2_definition()
    definition["definitionId"] = "qwen:text-to-image:v2"
    definition["displayName"] = "Qwen Image — Text To Image"
    definition["source"] = {
        "kind": "diffusers_catalog",
        "catalogCategory": "diffusers",
        "provider": "huggingface",
        "library": "diffusers",
        "libraryRevision": "a" * 40,
        "pipelineClass": "QwenImageModularPipeline",
        "blocksClass": "QwenImageAutoBlocks",
        "workflow": "text_to_image",
        "manifestDefinitionId": "qwen:text-to-image",
        "manifestContentHash": "sha256:" + "b" * 64,
        "executionAdmissionId": FAKE_REGISTERED_ADMISSION,
    }
    definition["ownership"] = {"kind": "registered", "definitionMutable": False}
    graph_node = definition["graph"]["nodes"][0]
    graph_node["data"]["params"]["steps"] = param(18, value_type="int", label="Steps")
    definition["graph"]["graphHash"] = block_graph_hash_v2(definition["graph"])
    definition["controls"].append(
        {
            "controlId": "steps",
            "label": "Steps",
            "binding": {"nodeId": "generate", "fieldId": "steps"},
            "valueType": "int",
            "defaultValue": 20,
            "order": 1,
        }
    )
    definition["contentHash"] = block_definition_content_hash_v2(definition)
    validate_block_definition_v2(definition)
    interface_hash = block_interface_hash_v2(definition)
    instance = {
        "schemaVersion": 2,
        "instanceId": "cluster-root",
        "definitionRef": {
            "definitionId": definition["definitionId"],
            "contentHash": definition["contentHash"],
        },
        "definitionSnapshot": definition,
        "effectiveGraph": copy.deepcopy(definition["graph"]),
        "effectiveInterface": {
            "boundary": copy.deepcopy(definition["boundary"]),
            "controls": copy.deepcopy(definition["controls"]),
            "baseInterfaceHash": interface_hash,
            "effectiveInterfaceHash": interface_hash,
        },
        "values": {"prompt-in": prompt, "prompt": prompt, "steps": 18},
        "customization": {
            "state": "parameters_changed",
            "baseGraphHash": definition["graph"]["graphHash"],
            "effectiveGraphHash": definition["graph"]["graphHash"],
        },
        "presentation": {
            "expanded": True,
            "position": {"x": 400, "y": 20},
            "size": {"width": 480, "height": 560},
            "internalLayout": {
                "generate": {"x": 32, "y": 96, "width": 320, "height": 240}
            },
        },
        "previewStates": [
            {
                "binding": copy.deepcopy(definition["previews"][0]),
                "status": "idle",
                **(
                    {"mediaReference": root["data"]["params"]["image"]["value"]}
                    if root["data"]["params"]["image"]["value"] is not None
                    else {}
                ),
            }
        ],
        "authorities": [],
    }
    validate_block_instance_v2(instance)
    conversion = {
        "legacyInstanceId": "cluster-root",
        "legacyCompositeHash": legacy_registered_cluster_composite_hash(workflow, "cluster-root"),
        "admissionId": FAKE_REGISTERED_ADMISSION,
        "compiledDefinitionContentHash": definition["contentHash"],
        "compiledDefinitionCanonicalSha256": block_definition_canonical_sha256_v2(definition),
        "blockInstanceV2": instance,
        "ownedNodeMappings": [
            {"legacyNodeId": "cluster-generate", "semanticNodeId": "generate"}
        ],
        "portMappings": [
            {
                "direction": "input",
                "legacyNodeId": "cluster-root",
                "legacyPortId": "prompt",
                "v2PortId": "prompt-in",
            },
            {
                "direction": "output",
                "legacyNodeId": "cluster-root",
                "legacyPortId": "image",
                "v2PortId": "image-out",
            },
        ],
        "valueMappings": [
            {
                "sourceKind": "execution_parameter_override",
                "sourceNodeId": "cluster-root",
                "sourceFieldId": "steps",
                "targetKind": "instance_value",
                "targetValueId": "steps",
            },
            {
                "sourceKind": "instance_parameter_override",
                "sourceNodeId": "cluster-root",
                "sourceFieldId": "prompt",
                "targetKind": "instance_value",
                "targetValueId": "prompt-in",
            },
            {
                "sourceKind": "node_param",
                "sourceNodeId": "cluster-generate",
                "sourceFieldId": "prompt",
                "targetKind": "instance_value",
                "targetValueId": "prompt",
            },
            {
                "sourceKind": "node_param",
                "sourceNodeId": "cluster-root",
                "sourceFieldId": "prompt",
                "targetKind": "instance_value",
                "targetValueId": "prompt-in",
            },
        ],
        "previewMappings": (
            [
                {
                    "sourceNodeId": "cluster-root",
                    "sourceFieldId": "image",
                    "targetNodeId": "generate",
                    "targetOutputPortId": "image",
                }
            ]
            if root["data"]["params"]["image"]["value"] is not None
            else []
        ),
        "absorbedInternalEdgeIds": [],
    }
    supplement = {
        "schemaVersion": 1,
        "kind": "registered_cluster_v2_compiler_supplement",
        "compilerOutputs": [
            {
                "sourcePath": "user-workflows/mixed-workflow.json",
                "sourceSha256": source_sha256,
                "conversions": [conversion],
            }
        ],
    }
    pin = (
        definition["contentHash"],
        block_definition_canonical_sha256_v2(definition),
    )
    return supplement, pin


def historical_semantic_equivalence_fixture(workflow, supplement):
    root = next(node for node in workflow["snapshot"]["nodes"] if node["id"] == "cluster-root")
    legacy_instance = root["data"]["huggingFaceClusterInstance"]
    definition = supplement["compilerOutputs"][0]["conversions"][0]["blockInstanceV2"]["definitionSnapshot"]
    source = definition["source"]
    receipt = {
        "id": "legacy-cluster-equivalence:qwen:text2image:test-review",
        "historical": {
            "manifestDefinitionId": legacy_instance["definition"]["id"],
            "libraryRevision": legacy_instance["definition"]["libraryRevision"],
            "manifestContentHash": legacy_instance["definition"]["contentHash"],
            "executionAdmissionId": legacy_instance["execution"]["admissionId"],
            "studioExecutionSpec": copy.deepcopy(legacy_instance["execution"]["studioExecutionSpec"]),
            "executionGraphHash": "sha256:" + "3" * 64,
            "interfaceHash": "sha256:" + "4" * 64,
        },
        "destination": {
            "manifestDefinitionId": source["manifestDefinitionId"],
            "libraryRevision": source["libraryRevision"],
            "manifestContentHash": source["manifestContentHash"],
            "executionAdmissionId": source["executionAdmissionId"],
            "blockDefinitionId": definition["definitionId"],
            "blockDefinitionContentHash": definition["contentHash"],
            "blockDefinitionCanonicalSha256": block_definition_canonical_sha256_v2(definition),
            "executionGraphHash": definition["graph"]["graphHash"],
            "interfaceHash": block_interface_hash_v2(definition),
        },
        "review": {
            "decision": "semantic_equivalent",
            "issuer": "workspace_owner:test-reviewer",
            "reviewedAt": "2026-09-02T12:00:00+05:30",
            "notes": "Fixture review of the exact historical and destination graph/interface contracts.",
        },
    }
    receipt["receiptHash"] = semantic_equivalence_content_hash(receipt, omit="receiptHash")
    receipt = validate_semantic_equivalence_receipt(receipt)
    supplement["compilerOutputs"][0]["conversions"][0]["semanticEquivalenceReceipt"] = {
        "receiptId": receipt["id"],
        "receiptHash": receipt["receiptHash"],
    }
    return receipt


def add_second_registered_cluster(workflow):
    nodes = workflow["snapshot"]["nodes"]
    edges = workflow["snapshot"]["edges"]
    root = copy.deepcopy(next(node for node in nodes if node["id"] == "cluster-root"))
    child = copy.deepcopy(next(node for node in nodes if node["id"] == "cluster-generate"))
    root["id"] = "cluster-root-b"
    root["position"] = {"x": 1000, "y": 700}
    root["data"]["huggingFaceClusterInstance"]["instanceId"] = "cluster-root-b"
    for param_value in root["data"]["params"].values():
        options = param_value.get("fieldOptions") or {}
        if options.get("huggingFaceClusterPortNodeId") == "cluster-generate":
            options["huggingFaceClusterPortNodeId"] = "cluster-generate-b"
    child["id"] = "cluster-generate-b"
    child["parentId"] = "cluster-root-b"
    child["data"]["huggingFaceClusterInstanceId"] = "cluster-root-b"
    nodes.extend([root, child])
    edges.extend(
        [
            {
                "id": "cluster-in-b",
                "source": "source",
                "sourceHandle": "prompt",
                "target": "cluster-root-b",
                "targetHandle": "prompt",
            },
            {
                "id": "cluster-out-b",
                "source": "cluster-root-b",
                "sourceHandle": "image",
                "target": "sink",
                "targetHandle": "image",
            },
        ]
    )


def append_second_compiler_conversion(workflow, supplement):
    first = supplement["compilerOutputs"][0]["conversions"][0]
    second = copy.deepcopy(first)
    second["legacyInstanceId"] = "cluster-root-b"
    second["legacyCompositeHash"] = legacy_registered_cluster_composite_hash(
        workflow,
        "cluster-root-b",
    )
    second["blockInstanceV2"]["instanceId"] = "cluster-root-b"
    second["blockInstanceV2"]["presentation"]["position"] = {"x": 1000, "y": 700}
    second["ownedNodeMappings"][0]["legacyNodeId"] = "cluster-generate-b"
    for mapping in second["portMappings"]:
        mapping["legacyNodeId"] = "cluster-root-b"
    for mapping in second["valueMappings"]:
        if mapping["sourceNodeId"] == "cluster-root":
            mapping["sourceNodeId"] = "cluster-root-b"
        elif mapping["sourceNodeId"] == "cluster-generate":
            mapping["sourceNodeId"] = "cluster-generate-b"
    for mapping in second["previewMappings"]:
        mapping["sourceNodeId"] = "cluster-root-b"
    supplement["compilerOutputs"][0]["conversions"].append(second)


def write_fixture(data_dir: Path):
    workflow = workflow_fixture()
    definition = v1_definition()
    workflow_path = data_dir / "user-workflows" / "mixed-workflow.json"
    block_path = data_dir / "studio" / "blocks" / "portrait-user-node.json"
    workflow_path.parent.mkdir(parents=True)
    block_path.parent.mkdir(parents=True)
    workflow_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    block_path.write_text(json.dumps(definition, ensure_ascii=False, indent=2), encoding="utf-8")
    return workflow_path, block_path


class CompositeMigrationConversionTests(unittest.TestCase):
    def test_v1_definition_reader_matches_v2_contract_without_inference(self):
        legacy = v1_definition()
        before = copy.deepcopy(legacy)

        definition = migrate_user_block_definition_v1(legacy)

        self.assertEqual(validate_block_definition_v2(definition), definition)
        self.assertEqual(legacy, before)
        self.assertEqual(definition["definitionId"], "portrait-user-node")
        self.assertEqual(definition["displayName"], "Portrait User Node")
        self.assertEqual(definition["boundary"]["mode"], "explicit")
        self.assertEqual(definition["boundary"]["inputs"][0]["portId"], "prompt-in")
        self.assertEqual(definition["boundary"]["outputs"][0]["portId"], "image-out")
        self.assertEqual(
            definition["boundary"]["inputs"][0]["binding"], {"nodeId": "generate", "fieldOrPortId": "prompt"}
        )
        self.assertEqual(definition["source"]["parent"]["definitionId"], "qwen:text-to-image")
        self.assertEqual(definition["ownership"], {"kind": "user", "definitionMutable": True})
        # Same fixture is asserted by the client V1 reader. These hashes make
        # converter drift fail the combined Block V2 contract gate.
        self.assertEqual(definition["graph"]["graphHash"], "block-graph-v2-02237c60")
        self.assertEqual(definition["contentHash"], "block-definition-v2-f1da4518")

    def test_workflow_conversion_preserves_root_identity_values_layout_ports_and_external_edges(self):
        source = workflow_fixture()
        before = copy.deepcopy(source)

        converted, disposition = migrate_workflow_document_v1(source)

        self.assertEqual(source, before)
        self.assertEqual(disposition["convertedV1InstanceIds"], ["user-root"])
        self.assertEqual(disposition["blockedLegacyClusterIds"], ["cluster-root"])
        nodes = {node["id"]: node for node in converted["snapshot"]["nodes"]}
        self.assertNotIn("user-generate", nodes)
        root = nodes["user-root"]
        self.assertEqual(root["type"], "block")
        self.assertEqual(root["position"], {"x": 800, "y": 20})
        self.assertEqual((root["width"], root["height"]), (440, 500))
        self.assertEqual(root["data"]["params"], {})
        self.assertNotIn("userBlockSnapshot", root["data"])
        instance = root["data"]["blockInstanceV2"]
        self.assertEqual(instance["instanceId"], "user-root")
        expected_interface_hash = "block-interface-v2-63fe84a5"
        self.assertEqual(block_interface_hash_v2(instance["definitionSnapshot"]), expected_interface_hash)
        self.assertEqual(
            instance["effectiveInterface"],
            {
                "boundary": instance["definitionSnapshot"]["boundary"],
                "controls": instance["definitionSnapshot"]["controls"],
                "baseInterfaceHash": expected_interface_hash,
                "effectiveInterfaceHash": expected_interface_hash,
            },
        )
        self.assertEqual(instance["values"], {"prompt-in": "workflow V1 prompt"})
        self.assertEqual(instance["presentation"]["position"], {"x": 800, "y": 20})
        self.assertEqual(instance["presentation"]["size"], {"width": 440, "height": 500})
        self.assertEqual(
            instance["presentation"]["internalLayout"]["generate"], {"x": 32, "y": 96, "width": 320, "height": 240}
        )
        self.assertEqual(
            instance["effectiveGraph"]["nodes"][0]["data"]["params"]["prompt"]["value"],
            "creator default",
        )
        self.assertEqual(
            [
                (edge["id"], edge["source"], edge.get("sourceHandle"), edge["target"], edge.get("targetHandle"))
                for edge in converted["snapshot"]["edges"]
                if edge["id"].startswith("v1")
            ],
            [
                ("v1-child-in", "source", "prompt", "user-root", "prompt-in"),
                ("v1-out", "user-root", "image-out", "sink", "image"),
            ],
        )
        self.assertEqual(nodes["cluster-root"], before["snapshot"]["nodes"][1])

    def test_preview_is_deterministic_read_only_and_marks_clusters_blocked(self):
        workflows = {"user-workflows/mixed-workflow.json": workflow_fixture()}
        blocks = {"studio/blocks/portrait-user-node.json": v1_definition()}
        before = copy.deepcopy((workflows, blocks))

        first = build_composite_migration_preview(workflows, blocks)
        second = build_composite_migration_preview(dict(reversed(list(workflows.items()))), blocks)

        self.assertEqual(first, second)
        self.assertEqual((workflows, blocks), before)
        self.assertEqual(first["mode"], "read_only_preview")
        self.assertFalse(first["boundary"]["writesFiles"])
        self.assertFalse(first["boundary"]["deletesRecords"])
        self.assertFalse(first["boundary"]["mergesRecords"])
        self.assertEqual(first["summary"]["targetFileCount"], 2)
        self.assertEqual(first["summary"]["convertibleCandidateCount"], 2)
        self.assertEqual(first["summary"]["legacyClusterBlockedCount"], 1)
        self.assertRegex(first["migrationId"], r"^block-v2-migration-[0-9a-f]{24}$")
        cluster = next(item for item in first["blocked"] if item["id"] == "cluster-root")
        self.assertIn("registered-catalog BlockDefinitionV2 compiler", cluster["reason"])

    def test_registered_compiler_supplement_is_exact_read_only_and_fail_closed(self):
        workflow = registered_cluster_workflow()
        workflows = {"user-workflows/mixed-workflow.json": workflow}
        default = build_composite_migration_preview(workflows, {})
        default_cluster = next(
            item
            for item in default["blocked"]
            if item["kind"] == "legacy_registered_cluster_instance"
        )
        source_sha = next(
            target["beforeSha256"]
            for target in default["targets"]
            if target["sourcePath"] == "user-workflows/mixed-workflow.json"
        )
        self.assertEqual(default_cluster["sourceSha256"], source_sha)
        self.assertEqual(
            default_cluster["legacyCompositeHash"],
            legacy_registered_cluster_composite_hash(workflow, "cluster-root"),
        )
        supplement, pin = registered_cluster_compiler_fixture(
            workflow,
            source_sha,
        )
        before = copy.deepcopy((workflows, supplement))
        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            preview = build_composite_migration_preview(
                workflows,
                {},
                compiler_supplement=supplement,
            )
            repeated = build_composite_migration_preview(
                workflows,
                {},
                compiler_supplement=copy.deepcopy(supplement),
            )

        self.assertEqual(preview, repeated)
        self.assertEqual((workflows, supplement), before)
        self.assertEqual(preview["summary"]["registeredClusterConvertibleCount"], 1)
        self.assertEqual(preview["summary"]["legacyClusterBlockedCount"], 0)
        self.assertTrue(preview["compilerSupplement"]["provided"])
        candidate = next(
            item
            for item in preview["candidates"]
            if item["kind"] == "legacy_registered_cluster_instance"
        )
        self.assertEqual(candidate["status"], "convertible")
        self.assertEqual(candidate["compilerReceipt"]["absorbedProjectionNodeIds"], ["cluster-generate"])
        self.assertNotIn("blockInstanceV2", candidate)
        self.assertFalse(preview["boundary"]["containsPromptAndParameterValues"])

        tampered = copy.deepcopy(supplement)
        tampered["compilerOutputs"][0]["conversions"][0]["blockInstanceV2"]["values"]["prompt-in"] = "changed"
        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            rejected = build_composite_migration_preview(
                workflows,
                {},
                compiler_supplement=tampered,
            )
        rejected_cluster = next(
            item
            for item in rejected["blocked"]
            if item["kind"] == "legacy_registered_cluster_instance"
        )
        self.assertIn("changes a persisted prompt or parameter value", rejected_cluster["reason"])

    def test_registered_compiler_supplement_rejects_unknown_fields_and_stale_sources(self):
        workflow = registered_cluster_workflow()
        workflows = {"user-workflows/mixed-workflow.json": workflow}
        default = build_composite_migration_preview(workflows, {})
        source_sha = next(
            target["beforeSha256"]
            for target in default["targets"]
            if target["sourcePath"] == "user-workflows/mixed-workflow.json"
        )
        supplement, pin = registered_cluster_compiler_fixture(workflow, source_sha)
        unknown = copy.deepcopy(supplement)
        unknown["unsafe"] = True
        with self.assertRaisesRegex(ValueError, "unknown field"):
            build_composite_migration_preview(workflows, {}, compiler_supplement=unknown)

        stale = copy.deepcopy(supplement)
        stale["compilerOutputs"][0]["sourceSha256"] = "sha256:" + "0" * 64
        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            preview = build_composite_migration_preview(workflows, {}, compiler_supplement=stale)
        cluster = next(item for item in preview["blocked"] if item["id"] == "cluster-root")
        self.assertIn("sourceSha256", cluster["reason"])

    def test_historical_manifest_requires_checked_in_semantic_equivalence_and_previews_exact_diff(self):
        workflow = registered_cluster_workflow()
        root = next(node for node in workflow["snapshot"]["nodes"] if node["id"] == "cluster-root")
        legacy_instance = root["data"]["huggingFaceClusterInstance"]
        legacy_instance["definition"]["libraryRevision"] = "1" * 40
        legacy_instance["definition"]["contentHash"] = "sha256:" + "2" * 64
        legacy_instance["execution"]["admissionId"] = "qwen-historical-admission"
        legacy_instance["execution"]["studioExecutionSpec"] = {
            "id": "qwen-historical-spec",
            "contentHash": "studio-spec-v1-87654321",
            "executionProfileId": "qwen-historical-profile",
        }
        workflows = {"user-workflows/mixed-workflow.json": workflow}
        default = build_composite_migration_preview(workflows, {})
        candidate = next(item for item in default["blocked"] if item["id"] == "cluster-root")
        self.assertNotIn("semanticEquivalenceAuthority", candidate)
        supplement, pin = registered_cluster_compiler_fixture(workflow, candidate["sourceSha256"])
        receipt = historical_semantic_equivalence_fixture(workflow, supplement)

        wrong_studio_spec = copy.deepcopy(receipt)
        wrong_studio_spec["historical"]["studioExecutionSpec"]["contentHash"] = (
            "studio-spec-v1-wrong-review"
        )
        wrong_studio_spec["receiptHash"] = semantic_equivalence_content_hash(
            wrong_studio_spec, omit="receiptHash"
        )
        wrong_studio_spec = validate_semantic_equivalence_receipt(wrong_studio_spec)
        with patch(
            "modiff.composite_migration._semantic_equivalence_receipt_for_historical",
            return_value=wrong_studio_spec,
        ):
            mismatched_preview = build_composite_migration_preview(workflows, {})
        mismatched_candidate = next(
            item for item in mismatched_preview["blocked"] if item["id"] == "cluster-root"
        )
        self.assertNotIn("semanticEquivalenceAuthority", mismatched_candidate)

        def receipt_by_reference(receipt_id, receipt_hash):
            return copy.deepcopy(receipt) if (receipt_id, receipt_hash) == (receipt["id"], receipt["receiptHash"]) else None

        with (
            patch.dict(
                "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
                {FAKE_REGISTERED_ADMISSION: pin},
            ),
            patch(
                "modiff.composite_migration._semantic_equivalence_receipt_for_historical",
                return_value=copy.deepcopy(receipt),
            ),
            patch(
                "modiff.composite_migration._semantic_equivalence_receipt_by_reference",
                side_effect=receipt_by_reference,
            ),
        ):
            preview = build_composite_migration_preview(
                workflows,
                {},
                compiler_supplement=supplement,
            )

        converted = next(item for item in preview["candidates"] if item["id"] == "cluster-root")
        self.assertEqual(converted["status"], "convertible")
        authority = converted["compilerReceipt"]["semanticEquivalenceAuthority"]
        self.assertEqual(authority["receiptId"], receipt["id"])
        self.assertEqual(authority["historical"], receipt["historical"])
        self.assertEqual(authority["destination"], receipt["destination"])
        self.assertEqual(authority["review"]["issuer"], "workspace_owner:test-reviewer")
        self.assertFalse(preview["boundary"]["containsPromptAndParameterValues"])
        self.assertNotIn("a copper observatory", json.dumps(preview))

        without_receipt = copy.deepcopy(supplement)
        without_receipt["compilerOutputs"][0]["conversions"][0].pop("semanticEquivalenceReceipt")
        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            rejected = build_composite_migration_preview(
                workflows,
                {},
                compiler_supplement=without_receipt,
            )
        blocked = next(item for item in rejected["blocked"] if item["id"] == "cluster-root")
        self.assertIn("historical Cluster conversion requires", blocked["reason"])

    def test_collapsed_registered_cluster_maps_root_ports_through_deterministic_semantic_roles(self):
        workflow = registered_cluster_workflow()
        root = next(node for node in workflow["snapshot"]["nodes"] if node["id"] == "cluster-root")
        root["data"]["params"]["image"]["value"] = "data/generated/collapsed-preview.png"
        virtual_generate = "cluster-root__diffusers.cluster-execution:generate"
        for param_value in root["data"]["params"].values():
            options = param_value.get("fieldOptions") or {}
            if options.get("huggingFaceClusterPortNodeId") == "cluster-generate":
                options["huggingFaceClusterPortNodeId"] = virtual_generate
        root["data"]["huggingFaceClusterInstance"]["presentation"]["expanded"] = False
        workflow["snapshot"]["nodes"] = [
            node for node in workflow["snapshot"]["nodes"] if node["id"] != "cluster-generate"
        ]
        default = build_composite_migration_preview(
            {"user-workflows/mixed-workflow.json": workflow},
            {},
        )
        source_sha = next(
            target["beforeSha256"]
            for target in default["targets"]
            if target["sourcePath"] == "user-workflows/mixed-workflow.json"
        )
        supplement, pin = registered_cluster_compiler_fixture(workflow, source_sha)
        conversion = supplement["compilerOutputs"][0]["conversions"][0]
        conversion["ownedNodeMappings"] = []
        conversion["valueMappings"] = [
            mapping
            for mapping in conversion["valueMappings"]
            if mapping["sourceNodeId"] != "cluster-generate"
        ]
        conversion["blockInstanceV2"]["presentation"]["expanded"] = False

        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            preview = build_composite_migration_preview(
                {"user-workflows/mixed-workflow.json": workflow},
                {},
                compiler_supplement=supplement,
            )

        self.assertEqual(preview["summary"]["registeredClusterConvertibleCount"], 1)
        candidate = next(
            item
            for item in preview["candidates"]
            if item["kind"] == "legacy_registered_cluster_instance"
        )
        self.assertEqual(candidate["status"], "convertible")
        self.assertEqual(candidate["compilerReceipt"]["absorbedProjectionNodeIds"], [])
        self.assertEqual(
            conversion["previewMappings"],
            [
                {
                    "sourceNodeId": "cluster-root",
                    "sourceFieldId": "image",
                    "targetNodeId": "generate",
                    "targetOutputPortId": "image",
                }
            ],
        )
        self.assertEqual(
            conversion["blockInstanceV2"]["previewStates"][0]["mediaReference"],
            "data/generated/collapsed-preview.png",
        )

    def test_registered_compiler_maps_non_null_outputs_to_preview_state_not_durable_values(self):
        workflow = registered_cluster_workflow()
        root = next(node for node in workflow["snapshot"]["nodes"] if node["id"] == "cluster-root")
        root["data"]["params"]["image"]["value"] = "data/generated/legacy-preview.png"
        default = build_composite_migration_preview(
            {"user-workflows/mixed-workflow.json": workflow},
            {},
        )
        source_sha = next(
            target["beforeSha256"]
            for target in default["targets"]
            if target["sourcePath"] == "user-workflows/mixed-workflow.json"
        )
        supplement, pin = registered_cluster_compiler_fixture(workflow, source_sha)
        conversion = supplement["compilerOutputs"][0]["conversions"][0]
        self.assertFalse(
            any(
                mapping["sourceFieldId"] == "image"
                for mapping in conversion["valueMappings"]
            )
        )
        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            preview = build_composite_migration_preview(
                {"user-workflows/mixed-workflow.json": workflow},
                {},
                compiler_supplement=supplement,
            )
        self.assertEqual(preview["summary"]["registeredClusterConvertibleCount"], 1)
        instance = conversion["blockInstanceV2"]
        self.assertNotIn("image", instance["values"])
        self.assertEqual(
            instance["previewStates"][0]["mediaReference"],
            "data/generated/legacy-preview.png",
        )

        missing = copy.deepcopy(supplement)
        missing["compilerOutputs"][0]["conversions"][0]["previewMappings"] = []
        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            rejected = build_composite_migration_preview(
                {"user-workflows/mixed-workflow.json": workflow},
                {},
                compiler_supplement=missing,
            )
        blocked = next(item for item in rejected["blocked"] if item["id"] == "cluster-root")
        self.assertIn("preview reference", blocked["reason"])

    def test_collapsed_instance_retains_definition_layout_with_legacy_frame_offset(self):
        workflow = workflow_fixture()
        snapshot = workflow["snapshot"]
        snapshot["nodes"] = [node for node in snapshot["nodes"] if node["id"] != "user-generate"]
        root = next(node for node in snapshot["nodes"] if node["id"] == "user-root")
        root["data"]["uiState"]["blockExpanded"] = False
        edge = next(edge for edge in snapshot["edges"] if edge["id"] == "v1-child-in")
        edge.update({"target": "user-root", "targetHandle": "prompt-in"})

        converted, _ = migrate_workflow_document_v1(workflow)

        root = next(node for node in converted["snapshot"]["nodes"] if node["id"] == "user-root")
        instance = root["data"]["blockInstanceV2"]
        self.assertFalse(instance["presentation"]["expanded"])
        self.assertEqual(
            instance["presentation"]["internalLayout"]["generate"],
            {"x": 28, "y": 76, "width": 320, "height": 240},
        )
        self.assertEqual(instance["effectiveGraph"]["graphHash"], instance["definitionSnapshot"]["graph"]["graphHash"])


class CompositeMigrationTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary.name)
        self.workflow_path, self.block_path = write_fixture(self.data_dir)
        self.original = {
            self.workflow_path: self.workflow_path.read_bytes(),
            self.block_path: self.block_path.read_bytes(),
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_apply_requires_exact_authority_creates_exact_backups_and_is_idempotent(self):
        preview = scan_composite_migration_preview(self.data_dir)
        with self.assertRaises(CompositeMigrationAuthorizationError):
            apply_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                plan_hash=preview["planHash"],
                confirmation="yes",
                allow_blocked_candidates=True,
            )
        with self.assertRaises(CompositeMigrationAuthorizationError):
            apply_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                plan_hash=preview["planHash"],
                confirmation=APPLY_CONFIRMATION,
            )
        self.assertEqual(self.workflow_path.read_bytes(), self.original[self.workflow_path])
        self.assertEqual(self.block_path.read_bytes(), self.original[self.block_path])

        applied = apply_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            plan_hash=preview["planHash"],
            confirmation=APPLY_CONFIRMATION,
            allow_blocked_candidates=True,
        )
        self.assertFalse(applied["idempotent"])
        self.assertEqual(applied["status"]["effectiveState"], "applied")
        converted_block = json.loads(self.block_path.read_text(encoding="utf-8"))
        converted_workflow = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        self.assertEqual(converted_block["schemaVersion"], 2)
        user_root = next(node for node in converted_workflow["snapshot"]["nodes"] if node["id"] == "user-root")
        self.assertIn("blockInstanceV2", user_root["data"])

        journal = self.data_dir / "studio" / "composite-migrations" / preview["migrationId"]
        self.assertEqual(
            (journal / "backups" / "user-workflows" / "mixed-workflow.json").read_bytes(),
            self.original[self.workflow_path],
        )
        self.assertEqual(
            (journal / "backups" / "studio" / "blocks" / "portrait-user-node.json").read_bytes(),
            self.original[self.block_path],
        )
        again = apply_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            plan_hash=preview["planHash"],
            confirmation=APPLY_CONFIRMATION,
            allow_blocked_candidates=True,
        )
        self.assertTrue(again["idempotent"])

    def test_registered_compiler_apply_preserves_values_edges_and_exact_rollback(self):
        workflow = registered_cluster_workflow()
        self.workflow_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        original_workflow = self.workflow_path.read_bytes()
        default = scan_composite_migration_preview(self.data_dir)
        source_sha = next(
            target["beforeSha256"]
            for target in default["targets"]
            if target["sourcePath"] == "user-workflows/mixed-workflow.json"
        )
        supplement, pin = registered_cluster_compiler_fixture(workflow, source_sha)

        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            preview = scan_composite_migration_preview(
                self.data_dir,
                compiler_supplement=supplement,
            )
            self.assertEqual(preview["summary"]["registeredClusterConvertibleCount"], 1)
            self.assertEqual(preview["summary"]["blockedCandidateCount"], 0)
            with self.assertRaises(CompositeMigrationConflictError):
                apply_composite_migration(
                    self.data_dir,
                    migration_id=preview["migrationId"],
                    plan_hash=preview["planHash"],
                    confirmation=APPLY_CONFIRMATION,
                )
            applied = apply_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                plan_hash=preview["planHash"],
                confirmation=APPLY_CONFIRMATION,
                compiler_supplement=supplement,
            )

        self.assertEqual(applied["status"]["effectiveState"], "applied")
        converted = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in converted["snapshot"]["nodes"]}
        self.assertNotIn("cluster-generate", nodes)
        root = nodes["cluster-root"]
        self.assertEqual(root["type"], "block")
        instance = root["data"]["blockInstanceV2"]
        self.assertEqual(instance["instanceId"], "cluster-root")
        self.assertEqual(instance["values"], {"prompt-in": "a copper observatory", "prompt": "a copper observatory", "steps": 18})
        self.assertEqual(instance["presentation"]["position"], {"x": 400, "y": 20})
        self.assertEqual(instance["presentation"]["size"], {"width": 480, "height": 560})
        edges = {edge["id"]: edge for edge in converted["snapshot"]["edges"]}
        self.assertEqual(
            (edges["cluster-in"]["target"], edges["cluster-in"]["targetHandle"]),
            ("cluster-root", "prompt-in"),
        )
        self.assertEqual(
            (edges["cluster-out"]["source"], edges["cluster-out"]["sourceHandle"]),
            ("cluster-root", "image-out"),
        )
        backup = (
            self.data_dir
            / "studio"
            / "composite-migrations"
            / preview["migrationId"]
            / "backups"
            / "user-workflows"
            / "mixed-workflow.json"
        )
        self.assertEqual(backup.read_bytes(), original_workflow)
        manifest = json.loads((backup.parents[2] / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["compilerSupplement"], preview["compilerSupplement"])
        self.assertEqual(manifest["compilerReceipts"][0]["instanceId"], "cluster-root")
        self.assertNotIn("blockInstanceV2", manifest["compilerReceipts"][0])
        with self.assertRaisesRegex(CompositeMigrationConflictError, "exact reviewed compiler supplement"):
            apply_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                plan_hash=preview["planHash"],
                confirmation=APPLY_CONFIRMATION,
            )
        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            idempotent = apply_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                plan_hash=preview["planHash"],
                confirmation=APPLY_CONFIRMATION,
                compiler_supplement=supplement,
            )
        self.assertTrue(idempotent["idempotent"])

        rolled_back = rollback_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            confirmation=ROLLBACK_CONFIRMATION,
        )
        self.assertEqual(rolled_back["status"]["effectiveState"], "rolled_back")
        self.assertEqual(self.workflow_path.read_bytes(), original_workflow)
        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            reapplied = apply_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                plan_hash=preview["planHash"],
                confirmation=APPLY_CONFIRMATION,
                compiler_supplement=supplement,
            )
        self.assertEqual(reapplied["status"]["effectiveState"], "applied")

    def test_historical_equivalence_apply_requires_exact_review_and_retains_rollback_journal(self):
        workflow = registered_cluster_workflow()
        root = next(node for node in workflow["snapshot"]["nodes"] if node["id"] == "cluster-root")
        legacy_instance = root["data"]["huggingFaceClusterInstance"]
        legacy_instance["definition"]["libraryRevision"] = "1" * 40
        legacy_instance["definition"]["contentHash"] = "sha256:" + "2" * 64
        legacy_instance["execution"]["admissionId"] = "qwen-historical-admission"
        legacy_instance["execution"]["studioExecutionSpec"] = {
            "id": "qwen-historical-spec",
            "contentHash": "studio-spec-v1-87654321",
            "executionProfileId": "qwen-historical-profile",
        }
        self.workflow_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        original = self.workflow_path.read_bytes()
        default = scan_composite_migration_preview(self.data_dir)
        candidate = next(item for item in default["blocked"] if item["id"] == "cluster-root")
        supplement, pin = registered_cluster_compiler_fixture(workflow, candidate["sourceSha256"])
        receipt = historical_semantic_equivalence_fixture(workflow, supplement)

        def receipt_by_reference(receipt_id, receipt_hash):
            return copy.deepcopy(receipt) if (receipt_id, receipt_hash) == (receipt["id"], receipt["receiptHash"]) else None

        with (
            patch.dict(
                "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
                {FAKE_REGISTERED_ADMISSION: pin},
            ),
            patch(
                "modiff.composite_migration._semantic_equivalence_receipt_for_historical",
                return_value=copy.deepcopy(receipt),
            ),
            patch(
                "modiff.composite_migration._semantic_equivalence_receipt_by_reference",
                side_effect=receipt_by_reference,
            ),
        ):
            preview = scan_composite_migration_preview(
                self.data_dir,
                compiler_supplement=supplement,
            )
            with self.assertRaises(CompositeMigrationAuthorizationError):
                apply_composite_migration(
                    self.data_dir,
                    migration_id=preview["migrationId"],
                    plan_hash=preview["planHash"],
                    confirmation="yes",
                    compiler_supplement=supplement,
                )
            self.assertEqual(self.workflow_path.read_bytes(), original)
            applied = apply_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                plan_hash=preview["planHash"],
                confirmation=APPLY_CONFIRMATION,
                compiler_supplement=supplement,
            )

        self.assertEqual(applied["status"]["effectiveState"], "applied")
        journal = self.data_dir / "studio" / "composite-migrations" / preview["migrationId"]
        manifest = json.loads((journal / "manifest.json").read_text(encoding="utf-8"))
        authority = manifest["compilerReceipts"][0]["semanticEquivalenceAuthority"]
        self.assertEqual(authority["receiptHash"], receipt["receiptHash"])
        self.assertEqual(
            (journal / "backups" / "user-workflows" / "mixed-workflow.json").read_bytes(),
            original,
        )
        rolled_back = rollback_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            confirmation=ROLLBACK_CONFIRMATION,
        )
        self.assertEqual(rolled_back["status"]["effectiveState"], "rolled_back")
        self.assertEqual(self.workflow_path.read_bytes(), original)

    def test_two_registered_clusters_share_one_atomic_workflow_target(self):
        workflow = registered_cluster_workflow()
        add_second_registered_cluster(workflow)
        self.workflow_path.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        original = self.workflow_path.read_bytes()
        default = scan_composite_migration_preview(self.data_dir)
        source_sha = next(
            target["beforeSha256"]
            for target in default["targets"]
            if target["sourcePath"] == "user-workflows/mixed-workflow.json"
        )
        supplement, pin = registered_cluster_compiler_fixture(workflow, source_sha)
        append_second_compiler_conversion(workflow, supplement)

        with patch.dict(
            "modiff.huggingface_cluster_runtime.REGISTERED_BLOCK_V2_DEFINITION_PINS",
            {FAKE_REGISTERED_ADMISSION: pin},
        ):
            preview = scan_composite_migration_preview(
                self.data_dir,
                compiler_supplement=supplement,
            )
            self.assertEqual(preview["summary"]["registeredClusterConvertibleCount"], 2)
            workflow_targets = [
                target
                for target in preview["targets"]
                if target["sourcePath"] == "user-workflows/mixed-workflow.json"
            ]
            self.assertEqual(len(workflow_targets), 1)
            self.assertEqual(
                workflow_targets[0]["convertedIds"],
                ["user-root", "cluster-root", "cluster-root-b"],
            )
            applied = apply_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                plan_hash=preview["planHash"],
                confirmation=APPLY_CONFIRMATION,
                compiler_supplement=supplement,
            )

        self.assertEqual(applied["changedPaths"].count("user-workflows/mixed-workflow.json"), 1)
        converted = json.loads(self.workflow_path.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in converted["snapshot"]["nodes"]}
        self.assertEqual(nodes["cluster-root"]["type"], "block")
        self.assertEqual(nodes["cluster-root-b"]["type"], "block")
        self.assertNotIn("cluster-generate", nodes)
        self.assertNotIn("cluster-generate-b", nodes)
        journal = self.data_dir / "studio" / "composite-migrations" / preview["migrationId"]
        self.assertEqual(
            (journal / "backups" / "user-workflows" / "mixed-workflow.json").read_bytes(),
            original,
        )
        rollback_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            confirmation=ROLLBACK_CONFIRMATION,
        )
        self.assertEqual(self.workflow_path.read_bytes(), original)

    def test_rollback_is_exact_idempotent_and_reapply_is_supported(self):
        preview = scan_composite_migration_preview(self.data_dir)
        apply_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            plan_hash=preview["planHash"],
            confirmation=APPLY_CONFIRMATION,
            allow_blocked_candidates=True,
        )

        rolled_back = rollback_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            confirmation=ROLLBACK_CONFIRMATION,
        )
        self.assertFalse(rolled_back["idempotent"])
        self.assertEqual(rolled_back["status"]["effectiveState"], "rolled_back")
        self.assertEqual(self.workflow_path.read_bytes(), self.original[self.workflow_path])
        self.assertEqual(self.block_path.read_bytes(), self.original[self.block_path])
        again = rollback_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            confirmation=ROLLBACK_CONFIRMATION,
        )
        self.assertTrue(again["idempotent"])

        reapplied = apply_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            plan_hash=preview["planHash"],
            confirmation=APPLY_CONFIRMATION,
            allow_blocked_candidates=True,
        )
        self.assertFalse(reapplied["idempotent"])
        self.assertEqual(reapplied["status"]["effectiveState"], "applied")

    def test_rollback_refuses_independent_post_migration_change(self):
        preview = scan_composite_migration_preview(self.data_dir)
        apply_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            plan_hash=preview["planHash"],
            confirmation=APPLY_CONFIRMATION,
            allow_blocked_candidates=True,
        )
        self.workflow_path.write_text('{"independent":true}', encoding="utf-8")

        with self.assertRaises(CompositeMigrationConflictError):
            rollback_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                confirmation=ROLLBACK_CONFIRMATION,
            )
        status = composite_migration_status(self.data_dir, preview["migrationId"])
        self.assertEqual(status["effectiveState"], "conflict")
        self.assertEqual(self.workflow_path.read_text(encoding="utf-8"), '{"independent":true}')

    def test_interrupted_status_rolls_back_only_applied_targets(self):
        preview = scan_composite_migration_preview(self.data_dir)
        apply_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            plan_hash=preview["planHash"],
            confirmation=APPLY_CONFIRMATION,
            allow_blocked_candidates=True,
        )
        self.block_path.write_bytes(self.original[self.block_path])

        interrupted = composite_migration_status(self.data_dir, preview["migrationId"])
        self.assertEqual(interrupted["effectiveState"], "interrupted")
        self.assertFalse(interrupted["resumeAvailable"])

        recovered = rollback_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            confirmation=ROLLBACK_CONFIRMATION,
        )
        self.assertEqual(recovered["status"]["effectiveState"], "rolled_back")
        self.assertEqual(self.workflow_path.read_bytes(), self.original[self.workflow_path])
        self.assertEqual(self.block_path.read_bytes(), self.original[self.block_path])

    def test_rollback_prevalidates_every_backup_before_changing_a_target(self):
        preview = scan_composite_migration_preview(self.data_dir)
        apply_composite_migration(
            self.data_dir,
            migration_id=preview["migrationId"],
            plan_hash=preview["planHash"],
            confirmation=APPLY_CONFIRMATION,
            allow_blocked_candidates=True,
        )
        applied = {
            self.workflow_path: self.workflow_path.read_bytes(),
            self.block_path: self.block_path.read_bytes(),
        }
        backup = (
            self.data_dir
            / "studio"
            / "composite-migrations"
            / preview["migrationId"]
            / "backups"
            / "studio"
            / "blocks"
            / "portrait-user-node.json"
        )
        backup.write_bytes(b"tampered")

        with self.assertRaises(CompositeMigrationConflictError):
            rollback_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                confirmation=ROLLBACK_CONFIRMATION,
            )
        self.assertEqual(self.workflow_path.read_bytes(), applied[self.workflow_path])
        self.assertEqual(self.block_path.read_bytes(), applied[self.block_path])
        self.assertEqual(
            composite_migration_status(self.data_dir, preview["migrationId"])["effectiveState"], "applied"
        )

    def test_apply_rejects_stale_preview_before_any_backup_or_write(self):
        preview = scan_composite_migration_preview(self.data_dir)
        original_block = json.loads(self.block_path.read_text(encoding="utf-8"))
        original_block["name"] = "Changed after preview"
        self.block_path.write_text(json.dumps(original_block), encoding="utf-8")

        with self.assertRaises(CompositeMigrationConflictError):
            apply_composite_migration(
                self.data_dir,
                migration_id=preview["migrationId"],
                plan_hash=preview["planHash"],
                confirmation=APPLY_CONFIRMATION,
                allow_blocked_candidates=True,
            )
        self.assertFalse((self.data_dir / "studio" / "composite-migrations" / preview["migrationId"]).exists())
        self.assertEqual(json.loads(self.block_path.read_text(encoding="utf-8"))["name"], "Changed after preview")


if __name__ == "__main__":
    unittest.main()
