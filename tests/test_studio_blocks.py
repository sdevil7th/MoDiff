import asyncio
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault(
    "aiohttp_cors",
    types.SimpleNamespace(
        setup=lambda *args, **kwargs: None,
        ResourceOptions=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault(
    "nanoid",
    types.SimpleNamespace(generate=lambda size=12: "test-id"),
)

from modiff.block_definition_v2 import (  # noqa: E402
    block_definition_content_hash_v2,
    block_graph_hash_v2,
    validate_block_definition_v2,
)
from modiff.server import WebServer  # noqa: E402


class FakeRequest:
    def __init__(self, payload=None, match_info=None, query=None):
        self._payload = payload
        self.match_info = match_info or {}
        self.query = query or {}

    async def json(self):
        return self._payload


def response_json(response):
    return json.loads(response.text)


def block_definition_v2(*, source_kind="user", definition_id="user-block-v2"):
    source = {"kind": source_kind}
    if source_kind == "hub_import":
        source.update(
            {
                "provider": "huggingface",
                "library": "diffusers",
                "repository": "owner/repository",
                "repositoryRevision": "a" * 40,
                "pipelineClass": "ExamplePipeline",
            }
        )
    elif source_kind in {"diffusers_catalog", "transformers_catalog"}:
        library = source_kind.removesuffix("_catalog")
        source.update(
            {
                "catalogCategory": library,
                "library": library,
                "libraryRevision": "b" * 40,
                "pipelineClass": "ExamplePipeline",
                "workflow": "text_to_image",
                "manifestDefinitionId": "manifest-example",
                "manifestContentHash": "sha256:" + "c" * 64,
                "executionAdmissionId": "catalog-admission-example",
            }
        )

    ownership = (
        {"kind": "registered", "definitionMutable": False}
        if source_kind in {"diffusers_catalog", "transformers_catalog"}
        else {"kind": "user", "definitionMutable": True}
    )
    definition = {
        "schemaVersion": 2,
        "definitionId": definition_id,
        "displayName": "Unit Block V2",
        "description": "A persisted common composite-node definition.",
        "contentHash": "pending",
        "source": source,
        "graph": {
            "nodes": [
                {
                    "nodeId": "load",
                    "nodeType": "Diffusers.LoadPipeline",
                    "data": {"params": {"enabled": False, "count": 0, "optional": None}},
                    "semanticRole": "loader",
                },
                {
                    "nodeId": "generate",
                    "nodeType": "Diffusers.Generate",
                    "data": {"params": {}},
                    "semanticRole": "denoise",
                },
            ],
            "edges": [
                {
                    "edgeId": "pipeline-edge",
                    "sourceNodeId": "load",
                    "sourcePortId": "pipeline",
                    "targetNodeId": "generate",
                    "targetPortId": "pipeline",
                }
            ],
            "executionOrder": ["load", "generate"],
            "graphHash": "pending",
        },
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
                "defaultValue": "",
                "required": True,
                "sealed": False,
                "order": 0,
                "group": "Generation",
                "help": "Prompt sent to the model.",
            },
            {
                "controlId": "seed",
                "label": "Seed",
                "binding": {"nodeId": "generate", "fieldId": "seed"},
                "valueType": "integer",
                "defaultValue": 0,
                "order": 1,
            },
        ],
        "suggestedInputs": [
            {
                "suggestionId": "creator-example",
                "label": "Creator example",
                "source": "Model card",
                "values": {"prompt": "A detailed test image", "seed": 0},
            }
        ],
        "previews": [
            {
                "nodeId": "generate",
                "outputPortId": "image",
                "mediaType": "image",
                "primary": True,
            }
        ],
        "ownership": ownership,
    }
    definition["graph"]["graphHash"] = block_graph_hash_v2(definition["graph"])
    definition["contentHash"] = block_definition_content_hash_v2(definition)
    return definition


def with_modular_source_provenance(definition, *, complete=True):
    """Attach hash-covered catalog identity to one ordinary V2 graph node."""

    result = copy.deepcopy(definition)
    metadata = {
        "kind": "upstream_block",
        "pipelineClass": "ExamplePipeline",
        "blocksClass": "ExampleBlocks",
        "workflowId": "text_to_image",
        "libraryRevision": "d" * 40,
        "runtimeRole": "prompt",
        "blockDefinitionId": "diffusers.modular:example-block",
        "blockClass": "ExampleBlock",
        "blockKind": "block",
        "blockContractHash": "sha256:" + "e" * 64,
        "placementPath": ["text_encoder"],
        "sourceDefinitionId": "diffusers.modular:example-block",
        "sourcePlacementPath": ["text_encoder"],
        "sourceExecutionScope": "unpruned_pipeline",
        "componentNames": [],
    }
    if not complete:
        del metadata["sourceExecutionScope"]
    result["graph"]["nodes"][0]["modularDiffusers"] = metadata
    result["graph"]["graphHash"] = block_graph_hash_v2(result["graph"])
    result["contentHash"] = block_definition_content_hash_v2(result)
    return result


def mirrored_control_definition_v2():
    definition = {
        "schemaVersion": 2,
        "definitionId": "mirrored-control-v2",
        "displayName": "Mirrored Control V2",
        "contentHash": "pending",
        "source": {"kind": "user"},
        "graph": {
            "nodes": [
                {
                    "nodeId": "generate",
                    "nodeType": "Diffusers.Generate",
                    "data": {
                        "params": {
                            "prompt": {"type": "string"},
                            "image": {"type": "image", "display": "output"},
                        }
                    },
                    "semanticRole": "denoise",
                },
                {
                    "nodeId": "prepare",
                    "nodeType": "Diffusers.Prepare",
                    "data": {
                        "params": {
                            "prompt": {"type": "text"},
                            "image": {"type": "image", "display": "output"},
                        }
                    },
                    "semanticRole": "prepare",
                },
            ],
            "edges": [],
            "executionOrder": ["prepare", "generate"],
            "graphHash": "pending",
        },
        "boundary": {
            "mode": "explicit",
            "inputs": [
                {
                    "portId": "prompt",
                    "label": "Prompt",
                    "valueType": "string",
                    "required": True,
                    "binding": {"nodeId": "generate", "fieldOrPortId": "prompt"},
                    "mirrorBindings": [
                        {"nodeId": "prepare", "fieldOrPortId": "prompt"}
                    ],
                }
            ],
            "outputs": [],
        },
        "controls": [
            {
                "controlId": "prompt",
                "label": "Prompt",
                "binding": {"nodeId": "generate", "fieldId": "prompt"},
                "mirrorBindings": [{"nodeId": "prepare", "fieldId": "prompt"}],
                "valueType": "string",
                "defaultValue": "",
                "order": 0,
            }
        ],
        "suggestedInputs": [],
        "previews": [],
        "ownership": {"kind": "user", "definitionMutable": True},
    }
    definition["graph"]["graphHash"] = block_graph_hash_v2(definition["graph"])
    definition["contentHash"] = block_definition_content_hash_v2(definition)
    return definition


class StudioBlockPersistenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    async def make_server(self):
        server = WebServer(work_dir=self.temp_dir.name, data_dir=self.temp_dir.name)
        await asyncio.sleep(0)
        return server

    async def test_block_save_list_get_and_delete(self):
        server = await self.make_server()
        payload = {
            "id": "unit-block",
            "name": "Unit Block",
            "version": 1,
            "nodes": [{"id": "node-1", "data": {"params": {}}, "position": {"x": 0, "y": 0}}],
            "edges": [],
            "inputs": [],
            "outputs": [],
            "exposedParams": [],
            "createdAt": 1,
            "updatedAt": 2,
        }

        post_response = await server.studio_blocks_post(FakeRequest(payload))
        self.assertEqual(post_response.status, 200)
        self.assertEqual(response_json(post_response)["block"]["id"], "unit-block")

        list_response = await server.studio_blocks_get(FakeRequest(query={}))
        list_payload = response_json(list_response)
        self.assertEqual(list_payload["count"], 1)
        self.assertEqual(list_payload["blocks"][0]["name"], "Unit Block")

        get_response = await server.studio_block_get(FakeRequest(match_info={"block_id": "unit-block"}))
        self.assertEqual(response_json(get_response)["block"]["id"], "unit-block")

        delete_response = await server.studio_block_delete(FakeRequest(match_info={"block_id": "unit-block"}))
        self.assertEqual(response_json(delete_response)["id"], "unit-block")

        empty_response = await server.studio_blocks_get(FakeRequest(query={}))
        self.assertEqual(response_json(empty_response)["count"], 0)

    async def test_invalid_block_is_rejected(self):
        server = await self.make_server()
        response = await server.studio_blocks_post(FakeRequest({"id": "broken", "name": "Broken"}))

        self.assertEqual(response.status, 400)
        self.assertTrue(response_json(response)["error"])

    async def test_nested_block_definition_is_rejected(self):
        server = await self.make_server()
        response = await server.studio_blocks_post(
            FakeRequest(
                {
                    "id": "nested",
                    "name": "Nested",
                    "version": 1,
                    "nodes": [
                        {
                            "id": "child-block",
                            "type": "block",
                            "data": {"type": "block", "params": {}, "userBlockId": "existing"},
                            "position": {"x": 0, "y": 0},
                        }
                    ],
                    "edges": [],
                    "inputs": [],
                    "outputs": [],
                    "exposedParams": [],
                }
            )
        )

        self.assertEqual(response.status, 400)
        self.assertIn("Nested user blocks", response_json(response)["message"])

    async def test_v2_user_and_hub_import_definitions_round_trip_without_rewriting(self):
        server = await self.make_server()

        for source_kind, definition_id in (
            ("user", "user-block-v2"),
            ("hub_import", "hub-block-v2"),
        ):
            with self.subTest(source_kind=source_kind):
                payload = block_definition_v2(
                    source_kind=source_kind,
                    definition_id=definition_id,
                )
                # The user fixture hashes are also asserted from the identical
                # semantic payload in MoDiff-client/scripts/block-schema-v2.test.mjs.
                # A canonicalizer change must update both runtimes together.
                self.assertEqual(payload["graph"]["graphHash"], "block-graph-v2-d23b144c")
                self.assertEqual(
                    payload["contentHash"],
                    {
                        "user": "block-definition-v2-dde5d4d9",
                        "hub_import": "block-definition-v2-613810bb",
                    }[source_kind],
                )
                post_response = await server.studio_blocks_post(FakeRequest(payload))
                self.assertEqual(post_response.status, 200)
                post_payload = response_json(post_response)
                self.assertEqual(post_payload["block"], payload)

                persisted = json.loads(Path(post_payload["path"]).read_text(encoding="utf-8"))
                self.assertEqual(persisted, payload)

                get_response = await server.studio_block_get(FakeRequest(match_info={"block_id": definition_id}))
                self.assertEqual(get_response.status, 200)
                self.assertEqual(response_json(get_response)["block"], payload)

        list_response = await server.studio_blocks_get(FakeRequest(query={}))
        listed = response_json(list_response)
        self.assertEqual(listed["count"], 2)
        self.assertCountEqual(
            listed["blocks"],
            [
                block_definition_v2(source_kind="user", definition_id="user-block-v2"),
                block_definition_v2(source_kind="hub_import", definition_id="hub-block-v2"),
            ],
        )

        delete_response = await server.studio_block_delete(FakeRequest(match_info={"block_id": "user-block-v2"}))
        self.assertEqual(delete_response.status, 200)
        self.assertEqual(response_json(delete_response)["id"], "user-block-v2")

    async def test_v2_modular_source_provenance_round_trips_and_is_all_or_none(self):
        server = await self.make_server()
        payload = with_modular_source_provenance(block_definition_v2())

        self.assertEqual(validate_block_definition_v2(payload), payload)
        response = await server.studio_blocks_post(FakeRequest(payload))
        self.assertEqual(response.status, 200)
        persisted_metadata = response_json(response)["block"]["graph"]["nodes"][0][
            "modularDiffusers"
        ]
        self.assertEqual(
            persisted_metadata,
            payload["graph"]["nodes"][0]["modularDiffusers"],
        )

        incomplete = with_modular_source_provenance(
            block_definition_v2(definition_id="incomplete-modular-source"),
            complete=False,
        )
        rejected = await server.studio_blocks_post(FakeRequest(incomplete))
        self.assertEqual(rejected.status, 400)
        self.assertIn("must be declared together", response_json(rejected)["message"])

    async def test_v2_bindings_preserve_private_style_backend_field_ids(self):
        server = await self.make_server()
        payload = block_definition_v2(definition_id="private-field-binding-v2")
        payload["graph"]["nodes"][1]["data"]["params"]["_auto_resize"] = {
            "type": "boolean",
            "default": True,
        }
        payload["controls"][0].update(
            {
                "controlId": "auto_resize",
                "label": "Auto resize",
                "binding": {"nodeId": "generate", "fieldId": "_auto_resize"},
                "valueType": "boolean",
                "defaultValue": True,
            }
        )
        payload["suggestedInputs"][0]["values"] = {"auto_resize": True}
        payload["graph"]["graphHash"] = block_graph_hash_v2(payload["graph"])
        payload["contentHash"] = block_definition_content_hash_v2(payload)

        self.assertEqual(validate_block_definition_v2(payload), payload)
        response = await server.studio_blocks_post(FakeRequest(payload))
        self.assertEqual(response.status, 200)
        self.assertEqual(
            response_json(response)["block"]["controls"][0]["binding"]["fieldId"],
            "_auto_resize",
        )

    async def test_v2_mirrored_controls_match_client_hashes_and_fail_closed(self):
        server = await self.make_server()
        payload = mirrored_control_definition_v2()
        self.assertEqual(payload["graph"]["graphHash"], "block-graph-v2-79cf79b4")
        self.assertEqual(payload["contentHash"], "block-definition-v2-20917ee9")
        response = await server.studio_blocks_post(FakeRequest(payload))
        self.assertEqual(response.status, 200)
        self.assertEqual(response_json(response)["block"], payload)

        malformed_mirrors = (
            [],
            [{"nodeId": "generate", "fieldId": "prompt"}],
            [{"nodeId": "missing", "fieldId": "prompt"}],
            [{"nodeId": "prepare", "fieldId": "image"}],
            [
                {"nodeId": "prepare", "fieldId": "prompt"},
                {"nodeId": "generate", "fieldId": "image"},
            ],
        )
        for index, mirrors in enumerate(malformed_mirrors):
            candidate = copy.deepcopy(payload)
            candidate["definitionId"] = f"malformed-mirror-{index}"
            candidate["controls"][0]["mirrorBindings"] = mirrors
            candidate["contentHash"] = block_definition_content_hash_v2(candidate)
            with self.subTest(index=index):
                rejected = await server.studio_blocks_post(FakeRequest(candidate))
                self.assertEqual(rejected.status, 400)
                self.assertIn("mirrorBindings", response_json(rejected)["message"])

        malformed_input_mirrors = (
            [],
            [{"nodeId": "generate", "fieldOrPortId": "prompt"}],
            [{"nodeId": "prepare", "fieldOrPortId": "image"}],
        )
        for index, mirrors in enumerate(malformed_input_mirrors):
            candidate = copy.deepcopy(payload)
            candidate["definitionId"] = f"malformed-input-mirror-{index}"
            candidate["boundary"]["inputs"][0]["mirrorBindings"] = mirrors
            candidate["contentHash"] = block_definition_content_hash_v2(candidate)
            with self.subTest(input_index=index):
                rejected = await server.studio_blocks_post(FakeRequest(candidate))
                self.assertEqual(rejected.status, 400)
                self.assertIn("mirrorBindings", response_json(rejected)["message"])

        output_mirror = copy.deepcopy(payload)
        output_mirror["definitionId"] = "malformed-output-mirror"
        output_mirror["boundary"]["outputs"] = [
            {
                "portId": "image",
                "label": "Image",
                "valueType": "image",
                "required": False,
                "binding": {"nodeId": "generate", "fieldOrPortId": "image"},
                "mirrorBindings": [
                    {"nodeId": "prepare", "fieldOrPortId": "image"}
                ],
            }
        ]
        output_mirror["contentHash"] = block_definition_content_hash_v2(output_mirror)
        rejected = await server.studio_blocks_post(FakeRequest(output_mirror))
        self.assertEqual(rejected.status, 400)
        self.assertIn("supported only for public inputs", response_json(rejected)["message"])

    async def test_v2_user_store_rejects_definition_ids_that_would_be_rewritten(self):
        server = await self.make_server()
        for definition_id in ("unsafe:id", "unsafe/id", "x" * 81):
            with self.subTest(definition_id=definition_id):
                response = await server.studio_blocks_post(
                    FakeRequest(block_definition_v2(definition_id=definition_id))
                )
                self.assertEqual(response.status, 400)
                self.assertIn("definitionId", response_json(response)["message"])

        list_response = await server.studio_blocks_get(FakeRequest(query={}))
        self.assertEqual(response_json(list_response)["count"], 0)

    async def test_malformed_v2_definitions_fail_closed(self):
        server = await self.make_server()
        valid = block_definition_v2()
        cases = []

        missing_graph_hash = copy.deepcopy(valid)
        del missing_graph_hash["graph"]["graphHash"]
        cases.append(("missing graph hash", missing_graph_hash, "graphHash"))

        unknown_root_field = copy.deepcopy(valid)
        unknown_root_field["futureMeaning"] = True
        cases.append(("unknown root field", unknown_root_field, "unsupported field"))

        invalid_binding = copy.deepcopy(valid)
        invalid_binding["boundary"]["inputs"][0]["binding"]["nodeId"] = "missing"
        cases.append(("unknown binding node", invalid_binding, "declared graph node"))

        nested = copy.deepcopy(valid)
        nested["graph"]["nodes"][0]["nodeType"] = "block"
        cases.append(("nested composite", nested, "Nested composite nodes"))

        invalid_enum_type = copy.deepcopy(valid)
        invalid_enum_type["ownership"]["kind"] = {}
        cases.append(("invalid enum type", invalid_enum_type, "ownership.kind"))

        stale_graph_hash = copy.deepcopy(valid)
        stale_graph_hash["graph"]["nodes"][0]["data"]["params"]["count"] = 1
        cases.append(("stale graph hash", stale_graph_hash, "graph.graphHash must be"))

        stale_content_hash = copy.deepcopy(valid)
        stale_content_hash["boundary"]["inputs"][0]["label"] = "Changed prompt"
        cases.append(("stale definition hash", stale_content_hash, "contentHash must be"))

        ambiguous_shared_value_id = copy.deepcopy(valid)
        ambiguous_shared_value_id["boundary"]["inputs"][0]["portId"] = "prompt"
        ambiguous_shared_value_id["boundary"]["inputs"][0]["binding"] = {
            "nodeId": "load",
            "fieldOrPortId": "different-prompt",
        }
        cases.append(
            (
                "shared input and control id with different bindings",
                ambiguous_shared_value_id,
                "share an id but bind different graph fields",
            )
        )

        colliding_public_port_id = copy.deepcopy(valid)
        colliding_public_port_id["boundary"]["outputs"][0]["portId"] = (
            colliding_public_port_id["boundary"]["inputs"][0]["portId"]
        )
        cases.append(
            (
                "input and output share one public handle id",
                colliding_public_port_id,
                "public portIds must be unique across inputs and outputs",
            )
        )

        invalid_node_id = copy.deepcopy(valid)
        invalid_node_id["graph"]["nodes"][0]["nodeId"] = "invalid node id"
        cases.append(("invalid node id", invalid_node_id, "nodeId is malformed"))

        immutable_user_definition = copy.deepcopy(valid)
        immutable_user_definition["ownership"]["definitionMutable"] = False
        cases.append(
            (
                "immutable user definition",
                immutable_user_definition,
                "must be mutable and user-owned",
            )
        )

        for label, payload, message in cases:
            with self.subTest(label=label):
                response = await server.studio_blocks_post(FakeRequest(payload))
                self.assertEqual(response.status, 400)
                self.assertIn(message, response_json(response)["message"])

        list_response = await server.studio_blocks_get(FakeRequest(query={}))
        self.assertEqual(response_json(list_response)["count"], 0)

    async def test_v2_round_trip_preserves_well_formed_json_escaping(self):
        server = await self.make_server()
        payload = block_definition_v2(definition_id="unicode-block-v2")
        payload["graph"]["nodes"][0]["data"]["params"]["unpaired"] = "\ud800"
        payload["graph"]["graphHash"] = block_graph_hash_v2(payload["graph"])
        payload["contentHash"] = block_definition_content_hash_v2(payload)

        response = await server.studio_blocks_post(FakeRequest(payload))
        self.assertEqual(response.status, 200)
        response_payload = response_json(response)
        self.assertEqual(response_payload["block"], payload)
        persisted = json.loads(Path(response_payload["path"]).read_text(encoding="utf-8"))
        self.assertEqual(persisted, payload)

    async def test_registered_v2_definition_cannot_overwrite_user_store_record(self):
        server = await self.make_server()
        original = block_definition_v2(definition_id="protected-definition")
        saved = await server.studio_blocks_post(FakeRequest(original))
        self.assertEqual(saved.status, 200)

        for source_kind in ("diffusers_catalog", "transformers_catalog"):
            with self.subTest(source_kind=source_kind):
                registered = block_definition_v2(
                    source_kind=source_kind,
                    definition_id="protected-definition",
                )
                registered["displayName"] = "Attempted registered overwrite"
                response = await server.studio_blocks_post(FakeRequest(registered))
                self.assertEqual(response.status, 400)
                self.assertIn("Registered catalog definitions", response_json(response)["message"])

        get_response = await server.studio_block_get(FakeRequest(match_info={"block_id": "protected-definition"}))
        self.assertEqual(get_response.status, 200)
        self.assertEqual(response_json(get_response)["block"], original)

    async def test_v1_block_contract_remains_compatible(self):
        server = await self.make_server()
        payload = {
            "id": "legacy:unit-block",
            "name": " Legacy Unit Block ",
            "version": 1,
            "schemaVersion": 1,
            "nodes": [{"id": "node-1", "data": {"params": {}}, "position": {"x": 0, "y": 0}}],
            "edges": [],
            "inputs": [],
            "outputs": [],
            "exposedParams": [],
            "origin": {"kind": "hugging_face_cluster_fork", "definitionId": "registered-parent"},
            "createdAt": 1,
            "updatedAt": 2,
        }

        response = await server.studio_blocks_post(FakeRequest(payload))
        self.assertEqual(response.status, 200)
        saved = response_json(response)["block"]
        self.assertEqual(saved["id"], "legacy_unit-block")
        self.assertEqual(saved["name"], "Legacy Unit Block")
        self.assertEqual(saved["version"], 1)
        self.assertEqual(saved["schemaVersion"], 1)
        self.assertEqual(saved["nodes"], payload["nodes"])
        self.assertEqual(saved["origin"], payload["origin"])
        self.assertEqual(saved["createdAt"], 1)
        self.assertGreaterEqual(saved["updatedAt"], 2)


if __name__ == "__main__":
    unittest.main()
