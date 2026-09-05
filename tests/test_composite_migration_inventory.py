import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from modiff.block_definition_v2 import block_definition_content_hash_v2, block_graph_hash_v2
from modiff.composite_migration_inventory import (
    build_composite_migration_inventory,
    scan_composite_migration_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def param(value, *, display=None, value_type="string", label=None, field_options=None):
    result = {
        "label": label or "Value",
        "type": value_type,
        "value": value,
    }
    if display is not None:
        result["display"] = display
    if field_options is not None:
        result["fieldOptions"] = field_options
    return result


def ordinary_node(node_id, x, y, *, params=None):
    return {
        "id": node_id,
        "type": "custom",
        "position": {"x": x, "y": y},
        "width": 320,
        "height": 240,
        "data": {
            "type": "custom",
            "module": "modules.Test",
            "action": "Pass",
            "label": node_id,
            "params": params or {},
        },
    }


def v1_definition():
    return {
        "id": "portrait-user-node",
        "name": "Portrait User Node",
        "version": 1,
        "nodes": [
            ordinary_node(
                "generate",
                0,
                0,
                params={
                    "prompt": param("creator default", label="Prompt"),
                    "image": param(None, display="output", value_type="image", label="Image"),
                },
            )
        ],
        "edges": [],
        "inputs": [
            {
                "id": "prompt-in",
                "label": "Prompt",
                "nodeId": "generate",
                "paramKey": "prompt",
                "type": "string",
            }
        ],
        "outputs": [
            {
                "id": "image-out",
                "label": "Image",
                "nodeId": "generate",
                "paramKey": "image",
                "type": "image",
            }
        ],
        "exposedParams": [],
        "createdAt": 1,
        "updatedAt": 2,
        "origin": {
            "schemaVersion": 1,
            "kind": "hugging_face_cluster_fork",
            "provider": "diffusers",
            "definitionId": "qwen:text-to-image",
            "libraryRevision": "a" * 40,
            "contentHash": "sha256:" + "b" * 64,
            "pipelineClass": "QwenImageModularPipeline",
            "workflowId": "text_to_image",
            "importedAt": 1,
        },
    }


def v2_definition():
    graph = {
        "nodes": [
            {
                "nodeId": "generate",
                "nodeType": "custom",
                "semanticRole": "generate",
                "data": {
                    "type": "custom",
                    "module": "modules.Test",
                    "action": "Generate",
                    "label": "Generate",
                    "params": {
                        "prompt": param("creator V2 default", label="Prompt"),
                        "image": param(None, display="output", value_type="image", label="Image"),
                    },
                },
            }
        ],
        "edges": [],
        "executionOrder": ["generate"],
        "graphHash": "pending",
    }
    graph["graphHash"] = block_graph_hash_v2(graph)
    definition = {
        "schemaVersion": 2,
        "definitionId": "portrait-v2",
        "displayName": "Portrait V2",
        "contentHash": "pending",
        "source": {"kind": "user"},
        "graph": graph,
        "boundary": {
            "mode": "explicit",
            "inputs": [
                {
                    "portId": "prompt-in",
                    "label": "Prompt",
                    "valueType": "string",
                    "required": True,
                    "binding": {"nodeId": "generate", "fieldOrPortId": "prompt"},
                }
            ],
            "outputs": [
                {
                    "portId": "image-out",
                    "label": "Image",
                    "valueType": "image",
                    "required": True,
                    "binding": {"nodeId": "generate", "fieldOrPortId": "image"},
                }
            ],
        },
        "controls": [
            {
                "controlId": "prompt",
                "label": "Prompt",
                "binding": {"nodeId": "generate", "fieldId": "prompt"},
                "valueType": "string",
                "defaultValue": "creator V2 default",
                "order": 0,
            }
        ],
        "suggestedInputs": [
            {
                "suggestionId": "creator-example",
                "label": "Creator example",
                "values": {"prompt": "suggested portrait"},
            }
        ],
        "previews": [{"nodeId": "generate", "outputPortId": "image", "mediaType": "image", "primary": True}],
        "ownership": {"kind": "user", "definitionMutable": True},
    }
    definition["contentHash"] = block_definition_content_hash_v2(definition)
    return definition


def workflow_fixture():
    v1 = v1_definition()
    v2 = v2_definition()
    source = ordinary_node("source", 0, 0, params={"prompt": param("source prompt", label="Prompt")})
    sink = ordinary_node("sink", 1800, 0, params={"image": param(None, display="input", value_type="image")})
    cluster_root = {
        "id": "cluster-root",
        "type": "cluster",
        "position": {"x": 400, "y": 20},
        "width": 480,
        "height": 560,
        "data": {
            "type": "cluster",
            "module": "",
            "action": "",
            "label": "Qwen Image — Text To Image",
            "params": {
                "prompt": param(
                    "a copper observatory",
                    display="input",
                    label="Prompt",
                    field_options={
                        "huggingFaceClusterPortDirection": "input",
                        "huggingFaceClusterPortNodeId": "cluster-generate",
                        "huggingFaceClusterPortField": "prompt",
                    },
                ),
                "image": param(
                    None,
                    display="output",
                    value_type="image",
                    label="Image",
                    field_options={
                        "huggingFaceClusterPortDirection": "output",
                        "huggingFaceClusterPortNodeId": "cluster-generate",
                        "huggingFaceClusterPortField": "image",
                    },
                ),
            },
            "huggingFaceClusterRole": "root",
            "huggingFaceClusterInstance": {
                "schemaVersion": 2,
                "instanceId": "cluster-root",
                "definition": {
                    "id": "qwen:text-to-image",
                    "libraryRevision": "a" * 40,
                    "contentHash": "sha256:" + "b" * 64,
                },
                "parameterOverrides": {"prompt": "a copper observatory"},
                "execution": {
                    "admissionId": "qwen-admission",
                    "studioExecutionSpec": {
                        "id": "qwen-spec",
                        "contentHash": "sha256:" + "c" * 64,
                        "executionProfileId": "qwen-profile",
                    },
                    "parameterOverrides": {"steps": 18},
                    "explicitParameterSources": ["steps"],
                },
                "structuralFork": None,
                "presentation": {"expanded": True, "expandedPaths": [], "executionLayout": {}},
            },
            "uiState": {"clusterCollapsedWidth": 420, "clusterCollapsedHeight": 480},
        },
    }
    cluster_child = ordinary_node("cluster-generate", 32, 96, params={"prompt": param("child prompt")})
    cluster_child.update({"parentId": "cluster-root"})
    cluster_child["data"].update(
        {
            "huggingFaceClusterRole": "execution",
            "huggingFaceClusterInstanceId": "cluster-root",
            "huggingFaceClusterSemanticId": "generate",
        }
    )
    v1_root = {
        "id": "user-root",
        "type": "block",
        "position": {"x": 800, "y": 20},
        "width": 440,
        "height": 500,
        "data": {
            "type": "block",
            "module": "modiff.user_blocks",
            "action": "portrait-user-node",
            "label": "Portrait User Node",
            "params": {"prompt-in": param("workflow V1 prompt", display="input", label="Prompt")},
            "userBlockId": "portrait-user-node",
            "userBlockSnapshot": v1,
            "uiState": {"blockExpanded": True, "blockCollapsedWidth": 420, "blockCollapsedHeight": 480},
        },
    }
    v1_child = ordinary_node("user-generate", 32, 96, params={"prompt": param("workflow V1 prompt")})
    v1_child.update({"parentId": "user-root"})
    v1_child["data"].update({"userBlockInstanceId": "user-root", "userBlockSourceNodeId": "generate"})
    v2_instance = {
        "schemaVersion": 2,
        "instanceId": "v2-root",
        "definitionRef": {"definitionId": v2["definitionId"], "contentHash": v2["contentHash"]},
        "definitionSnapshot": v2,
        "effectiveGraph": v2["graph"],
        "values": {"prompt": "workflow V2 prompt"},
        "customization": {
            "state": "parameters_changed",
            "baseGraphHash": v2["graph"]["graphHash"],
            "effectiveGraphHash": v2["graph"]["graphHash"],
        },
        "presentation": {
            "expanded": True,
            "position": {"x": 1200, "y": 20},
            "size": {"width": 460, "height": 520},
            "internalLayout": {"generate": {"x": 32, "y": 96}},
        },
        "previewStates": [{"binding": v2["previews"][0], "status": "idle"}],
        "authorities": [],
    }
    v2_root = {
        "id": "v2-root",
        "type": "block",
        "position": {"x": 1200, "y": 20},
        "width": 460,
        "height": 520,
        "data": {"type": "block", "blockInstanceV2": v2_instance, "params": {}},
    }
    v2_child = ordinary_node("block-v2:v2-root:generate", 32, 96)
    v2_child.update({"parentId": "v2-root"})
    v2_child["data"].update(
        {"blockProjectionOwnerId": "v2-root", "blockProjectionNodeId": "generate", "blockProjectionKind": "node"}
    )
    edges = [
        {
            "id": "cluster-in",
            "source": "source",
            "sourceHandle": "prompt",
            "target": "cluster-root",
            "targetHandle": "prompt",
        },
        {
            "id": "cluster-out",
            "source": "cluster-root",
            "sourceHandle": "image",
            "target": "sink",
            "targetHandle": "image",
        },
        {
            "id": "v1-child-in",
            "source": "source",
            "sourceHandle": "prompt",
            "target": "user-generate",
            "targetHandle": "prompt",
        },
        {
            "id": "v1-out",
            "source": "user-root",
            "sourceHandle": "image-out",
            "target": "sink",
            "targetHandle": "image",
        },
        {
            "id": "v2-in",
            "source": "source",
            "sourceHandle": "prompt",
            "target": "v2-root",
            "targetHandle": "prompt-in",
        },
        {"id": "v2-out", "source": "v2-root", "sourceHandle": "image-out", "target": "sink", "targetHandle": "image"},
    ]
    return {
        "id": "mixed-workflow",
        "title": "Mixed composites",
        "revision": 7,
        "snapshot": {
            "nodes": [source, cluster_root, cluster_child, v1_root, v1_child, v2_root, v2_child, sink],
            "edges": edges,
            "viewport": {"x": 0, "y": 0, "zoom": 1},
        },
    }


class CompositeMigrationInventoryTests(unittest.TestCase):
    def test_inventory_captures_each_composite_generation_without_conversion(self):
        report = build_composite_migration_inventory(
            {"user-workflows/mixed.json": workflow_fixture()},
            {
                "studio/blocks/portrait-user-node.json": v1_definition(),
                "studio/blocks/portrait-v2.json": v2_definition(),
            },
        )

        self.assertEqual(report["mode"], "read_only_dry_run")
        self.assertFalse(report["boundary"]["writesFiles"])
        self.assertFalse(report["boundary"]["convertsRecords"])
        self.assertFalse(report["boundary"]["deletesRecords"])
        self.assertEqual(report["summary"]["legacyClusterRootCount"], 1)
        self.assertEqual(report["summary"]["legacyUserBlockV1RootCount"], 1)
        self.assertEqual(report["summary"]["blockV2RootCount"], 1)
        self.assertEqual(report["summary"]["legacyClusterDerivedChildCount"], 1)
        self.assertEqual(report["summary"]["legacyUserBlockDerivedChildCount"], 1)
        self.assertEqual(report["summary"]["blockV2ProjectionChildCount"], 1)
        self.assertEqual(report["summary"]["legacyV1ReusableDefinitionCount"], 1)
        self.assertEqual(report["summary"]["blockV2ReusableDefinitionCount"], 1)

        composites = {item["nodeId"]: item for item in report["workflows"][0]["composites"]}
        cluster = composites["cluster-root"]
        self.assertEqual(cluster["canvas"]["position"], {"x": 400, "y": 20})
        self.assertEqual(cluster["canvas"]["size"], {"width": 480, "height": 560})
        self.assertEqual(cluster["prompts"][0]["value"], "a copper observatory")
        self.assertEqual(cluster["instanceValues"]["clusterExecutionParameterOverrides"], {"steps": 18})
        self.assertEqual(cluster["sourceRefs"]["legacyCluster"]["definition"]["id"], "qwen:text-to-image")
        self.assertEqual(
            cluster["sourceRefs"]["legacyCluster"]["studioExecutionSpec"],
            {
                "id": "qwen-spec",
                "contentHash": "sha256:" + "c" * 64,
                "executionProfileId": "qwen-profile",
            },
        )
        self.assertEqual(cluster["ports"]["inputs"][0]["portId"], "prompt")
        self.assertEqual([edge["edgeId"] for edge in cluster["externalEdges"]], ["cluster-in", "cluster-out"])
        self.assertEqual(cluster["definitionResolution"]["status"], "external_registered_catalog_not_scanned")

        legacy_user = composites["user-root"]
        self.assertEqual(legacy_user["sourceRefs"]["legacyUserBlock"]["origin"]["definitionId"], "qwen:text-to-image")
        self.assertEqual(legacy_user["ports"]["inputs"][0]["portId"], "prompt-in")
        self.assertEqual(
            legacy_user["definitionResolution"]["matchedSourcePaths"], ["studio/blocks/portrait-user-node.json"]
        )
        self.assertIn("v1-child-in", [edge["edgeId"] for edge in legacy_user["externalEdges"]])

        v2 = composites["v2-root"]
        self.assertEqual(v2["instanceValues"]["blockV2Values"], {"prompt": "workflow V2 prompt"})
        self.assertEqual(v2["sourceRefs"]["blockV2"]["definitionRef"]["definitionId"], "portrait-v2")
        self.assertEqual(v2["definitionResolution"]["matchedSourcePaths"], ["studio/blocks/portrait-v2.json"])
        self.assertEqual(v2["ports"]["outputs"][0]["portId"], "image-out")

    def test_ambiguity_and_recovery_errors_are_reported_without_guessing(self):
        hybrid = workflow_fixture()["snapshot"]["nodes"][1]
        hybrid = copy.deepcopy(hybrid)
        hybrid["id"] = "hybrid"
        hybrid["data"]["blockInstanceV2"] = {
            "instanceId": "different-id",
            "definitionRef": {"definitionId": "missing-v2", "contentHash": "missing"},
        }
        orphan = ordinary_node("orphan", 0, 0)
        orphan["data"]["huggingFaceClusterInstanceId"] = "missing-owner"
        missing_snapshot = {
            "id": "missing-snapshot",
            "type": "block",
            "position": {"x": 0, "y": 0},
            "data": {"type": "block", "params": {}, "userBlockId": "not-in-store"},
        }
        document = {
            "id": "ambiguous",
            "snapshot": {
                "nodes": [hybrid, orphan, missing_snapshot, copy.deepcopy(missing_snapshot)],
                "edges": [{"id": "dangling", "source": "hybrid", "target": "absent"}, "bad-edge"],
            },
        }
        report = build_composite_migration_inventory(
            {"user-workflows/ambiguous.json": document},
            {"studio/blocks/unknown.json": {"id": "unknown", "version": 99}},
        )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertIn("composite_authority_ambiguous", codes)
        self.assertIn("derived_child_owner_missing", codes)
        self.assertIn("legacy_user_snapshot_missing", codes)
        self.assertIn("workflow_node_id_duplicate", codes)
        self.assertIn("workflow_edge_endpoint_missing", codes)
        self.assertIn("workflow_edge_not_object", codes)
        self.assertIn("reusable_block_schema_unknown", codes)
        self.assertEqual(report["summary"]["ambiguousCompositeRootCount"], 1)
        self.assertEqual(report["summary"]["orphanDerivedChildCount"], 1)
        self.assertTrue(report["summary"]["requiresManualReview"])
        self.assertTrue(
            all(
                item["migrationDisposition"] == "inventory_only_no_conversion"
                for item in report["workflows"][0]["composites"]
            )
        )

    def test_report_is_deterministic_and_does_not_mutate_caller_values(self):
        workflows = {"user-workflows/mixed.json": workflow_fixture()}
        blocks = {
            "studio/blocks/portrait-v2.json": v2_definition(),
            "studio/blocks/portrait-user-node.json": v1_definition(),
        }
        before = copy.deepcopy((workflows, blocks))
        first = build_composite_migration_inventory(workflows, blocks)
        second = build_composite_migration_inventory(
            dict(reversed(list(workflows.items()))),
            dict(reversed(list(blocks.items()))),
        )

        self.assertEqual(first, second)
        self.assertEqual((workflows, blocks), before)
        self.assertRegex(first["reportHash"], r"^sha256:[0-9a-f]{64}$")
        self.assertNotIn("generatedAt", first)

    def test_filesystem_scan_and_cli_are_read_only_and_report_parse_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            workflows_dir = data_dir / "user-workflows"
            blocks_dir = data_dir / "studio" / "blocks"
            workflows_dir.mkdir(parents=True)
            blocks_dir.mkdir(parents=True)
            (workflows_dir / "valid.json").write_text(json.dumps(workflow_fixture()), encoding="utf-8")
            (workflows_dir / "broken.json").write_text("{broken", encoding="utf-8")
            (blocks_dir / "v1.json").write_text(json.dumps(v1_definition()), encoding="utf-8")
            before = {
                path.relative_to(data_dir).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in sorted(data_dir.rglob("*"))
                if path.is_file()
            }

            report = scan_composite_migration_inventory(data_dir)
            self.assertEqual(report["summary"]["workflowDocumentCount"], 2)
            self.assertEqual(report["summary"]["workflowReadErrorCount"], 1)
            self.assertIn("source_json_invalid", {issue["code"] for issue in report["issues"]})

            command = [
                sys.executable,
                str(ROOT / "scripts" / "inventory_legacy_composites.py"),
                "--data-dir",
                str(data_dir),
                "--compact",
                "--fail-on-errors",
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(json.loads(completed.stdout), report)
            self.assertEqual(completed.stderr, "")

            after = {
                path.relative_to(data_dir).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in sorted(data_dir.rglob("*"))
                if path.is_file()
            }
            self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
