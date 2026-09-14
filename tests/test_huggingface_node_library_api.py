import json
import unittest
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer

from modiff.server import WebServer


class HuggingFaceNodeLibraryApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_only_endpoint_publishes_detached_reviewed_library(self):
        server = WebServer({})
        routes = {resource.canonical for resource in server.app.router.resources()}
        self.assertIn("/huggingface/node-library", routes)
        self.assertIn("/huggingface/registered-block-v2", routes)
        self.assertIn("/huggingface/modular-conditionals", routes)
        self.assertIn("/huggingface/cluster/runtime-qualification", routes)
        self.assertIn("/huggingface/cluster/auto-authority", routes)
        self.assertIn("/huggingface/modular-composition/rebuild", routes)
        self.assertIn("/custom_modular/inspect", routes)

        first_response = await server.huggingface_node_library(None)
        first = json.loads(first_response.text)
        first["definitions"][0]["label"] = "Changed by caller"

        second_response = await server.huggingface_node_library(None)
        second = json.loads(second_response.text)

        self.assertEqual(first_response.status, 200)
        self.assertEqual(second_response.status, 200)
        self.assertEqual(second["schemaVersion"], 6)
        self.assertEqual(second["providers"], ["diffusers", "transformers"])
        self.assertEqual(len(second["definitions"]), 135)
        self.assertEqual(len(second["blockDefinitions"]), 559)
        self.assertEqual(len(second["blockRoleAdapters"]), 321)
        self.assertEqual(len(second["containerStateAdapters"]), 9)
        self.assertNotEqual(second["definitions"][0]["label"], "Changed by caller")
        self.assertTrue(all(definition["ownership"] == "library" for definition in second["definitions"]))
        self.assertTrue(all(definition["mutable"] is False for definition in second["definitions"]))
        admissions = [
            admission for definition in second["definitions"] for admission in definition["executionAdmissions"]
        ]
        self.assertEqual(len(admissions), 122)
        self.assertEqual(sum(admission["status"] == "admitted" for admission in admissions), 122)
        self.assertTrue(all(admission["executable"] is False for admission in admissions))

    async def test_registered_block_v2_endpoint_serves_one_detached_hash_pinned_definition(self):
        server = WebServer({})
        client = TestClient(TestServer(server.app))
        await client.start_server()
        query = {
            "definition_id": "diffusers.modular:QwenImageModularPipeline:text2image",
            "admission_id": (
                "diffusers.cluster-admission:QwenImageModularPipeline:"
                "text2image:mode:text_to_image"
            ),
        }
        try:
            first = await client.get("/huggingface/registered-block-v2", params=query)
            first_payload = await first.json()
            self.assertEqual(first.status, 200)
            self.assertFalse(first_payload["error"])
            entry = first_payload["entry"]
            self.assertEqual(entry["catalogDefinitionId"], query["definition_id"])
            self.assertEqual(entry["admissionId"], query["admission_id"])
            self.assertEqual(entry["internalLayoutMode"], "hierarchical")
            # The active text-to-image graph excludes the two optional VAE
            # encoder branches selected only by image/control-image routes.
            # They remain present in the unpruned Modular Diffusers catalog.
            self.assertEqual(len(entry["definition"]["graph"]["nodes"]), 19)
            self.assertEqual(
                sum(node["nodeType"] == "group" for node in entry["definition"]["graph"]["nodes"]),
                5,
            )

            entry["definition"]["displayName"] = "Changed by caller"
            second = await client.get("/huggingface/registered-block-v2", params=query)
            second_payload = await second.json()
            self.assertNotEqual(second_payload["entry"]["definition"]["displayName"], "Changed by caller")

            missing = await client.get(
                "/huggingface/registered-block-v2",
                params={"definition_id": query["definition_id"], "admission_id": "missing"},
            )
            self.assertEqual(missing.status, 404)
        finally:
            await client.close()

    async def test_read_only_conditional_endpoint_publishes_a_detached_unpruned_snapshot(self):
        server = WebServer({})
        first_response = await server.huggingface_modular_conditionals(None)
        first = json.loads(first_response.text)
        first["pipelines"][0]["pipelineClass"] = "ChangedByCaller"

        second_response = await server.huggingface_modular_conditionals(None)
        second = json.loads(second_response.text)

        self.assertEqual(first_response.status, 200)
        self.assertEqual(second_response.status, 200)
        self.assertEqual(second["schemaVersion"], 1)
        self.assertEqual(len(second["pipelines"]), 34)
        self.assertEqual(sum(len(item["conditionals"]) for item in second["pipelines"]), 73)
        self.assertNotEqual(second["pipelines"][0]["pipelineClass"], "ChangedByCaller")

    async def test_expert_runtime_qualification_endpoint_returns_backend_receipt(self):
        class Request:
            async def json(self):
                return {
                    "schemaVersion": 1,
                    "definitionId": "definition",
                    "admissionId": "admission",
                    "resourceMode": "expert",
                    "recipe": {},
                }

        server = WebServer({})
        receipt = {"schemaVersion": 1, "claim": "expert_cluster_runtime_qualified"}
        with (
            patch("modiff.server.reviewed_huggingface_node_library", return_value={"definitions": []}),
            patch("modiff.server.get_local_models", return_value=[]),
            patch(
                "modiff.server.qualify_huggingface_cluster_expert_runtime",
                return_value=receipt,
            ) as qualify,
            patch.object(
                server,
                "_runtime_fingerprint_for_control_request",
                return_value={"fingerprint": f"sha256:{'a' * 64}"},
            ),
        ):
            response = await server.huggingface_cluster_runtime_qualification(Request())

        payload = json.loads(response.text)
        self.assertEqual(response.status, 200)
        self.assertFalse(payload["error"])
        self.assertEqual(payload["receipt"], receipt)
        qualify.assert_called_once()

    async def test_auto_authority_post_route_returns_bounded_success_and_error_envelopes(self):
        server = WebServer({})
        methods = {(route.method, route.resource.canonical) for route in server.app.router.routes()}
        self.assertIn(("POST", "/huggingface/cluster/auto-authority"), methods)
        self.assertNotIn(("GET", "/huggingface/cluster/auto-authority"), methods)
        client = TestClient(TestServer(server.app))
        await client.start_server()
        payload = {"schemaVersion": 1, "instance": {"instanceId": "registered-v2"}, "form": {}}
        receipt = {"kind": "auto", "definitionId": "admission"}
        try:
            with (
                patch("modiff.server.reviewed_huggingface_node_library", return_value={"definitions": []}),
                patch("modiff.server.get_local_models", return_value=[]),
                patch(
                    "modiff.server.qualify_huggingface_cluster_auto_authority",
                    return_value=receipt,
                ) as qualify,
                patch.object(
                    server,
                    "_runtime_fingerprint_for_control_request",
                    return_value={"fingerprint": f"sha256:{'a' * 64}"},
                ),
            ):
                success = await client.post("/huggingface/cluster/auto-authority", json=payload)
                success_payload = await success.json()

            self.assertEqual(success.status, 200)
            self.assertEqual(success_payload, {"error": False, "receipt": receipt})
            qualify.assert_called_once()
            self.assertEqual(qualify.call_args.args, (payload,))
            self.assertEqual(qualify.call_args.kwargs["data_dir"], server.data_dir)

            with patch(
                "modiff.server.qualify_huggingface_cluster_auto_authority",
                side_effect=ValueError(
                    "Cannot qualify Hugging Face Cluster Node Auto execution: bounded planner rejection."
                ),
            ):
                rejected = await client.post("/huggingface/cluster/auto-authority", json=payload)
                rejected_payload = await rejected.json()

            self.assertEqual(rejected.status, 400)
            self.assertEqual(
                rejected_payload,
                {
                    "error": True,
                    "message": (
                        "Cannot qualify Hugging Face Cluster Node Auto execution: "
                        "bounded planner rejection."
                    ),
                },
            )
        finally:
            await client.close()

    async def test_modular_composition_endpoint_returns_an_init_pipeline_rebuild_receipt(self):
        class Request:
            async def json(self):
                return {"schemaVersion": 1, "operations": []}

        server = WebServer({})
        receipt = {
            "schemaVersion": 1,
            "claim": "reviewed_modular_composition_rebuilt",
            "executable": False,
        }
        with patch(
            "modiff.server.rebuild_reviewed_modular_composition",
            return_value=receipt,
        ) as rebuild:
            response = await server.huggingface_modular_composition_rebuild(Request())

        payload = json.loads(response.text)
        self.assertEqual(response.status, 200)
        self.assertFalse(payload["error"])
        self.assertEqual(payload["receipt"], receipt)
        rebuild.assert_called_once_with({"schemaVersion": 1, "operations": []})

    async def test_custom_modular_inspection_is_posted_to_the_bounded_backend_inspector(self):
        class Request:
            async def json(self):
                return {"repo_id": "owner/pipeline", "revision": "a" * 40}

        server = WebServer({})
        inspection = {
            "schemaVersion": 1,
            "repository": "owner/pipeline",
            "revision": "a" * 40,
            "sidecar": {"format": "mellon"},
        }
        with patch(
            "modiff.custom_modular_inspection.inspect_installed_custom_modular_contract",
            return_value=inspection,
        ) as inspect_contract:
            response = await server.custom_modular_inspect(Request())

        payload = json.loads(response.text)
        self.assertEqual(response.status, 200)
        self.assertFalse(payload["error"])
        self.assertEqual(payload["sidecar"]["format"], "mellon")
        inspect_contract.assert_called_once_with("owner/pipeline", "a" * 40)

    async def test_custom_modular_inspection_rejects_extra_request_fields(self):
        class Request:
            async def json(self):
                return {"repo_id": "owner/pipeline", "revision": "a" * 40, "trust_remote_code": True}

        response = await WebServer({}).custom_modular_inspect(Request())
        self.assertEqual(response.status, 400)
        self.assertEqual(
            json.loads(response.text)["code"],
            "invalid_custom_modular_inspection_request",
        )

    async def test_custom_modular_inspection_distinguishes_missing_and_invalid_sidecars(self):
        from modules.ModularDiffusers.pipeline_schema import HubPipelineSidecarNotInstalledError

        class Request:
            async def json(self):
                return {"repo_id": "owner/pipeline", "revision": "a" * 40}

        server = WebServer({})
        with patch(
            "modiff.custom_modular_inspection.inspect_installed_custom_modular_contract",
            side_effect=HubPipelineSidecarNotInstalledError("exact snapshot is not installed"),
        ):
            response = await server.custom_modular_inspect(Request())
        self.assertEqual(response.status, 409)
        self.assertEqual(json.loads(response.text)["code"], "custom_modular_revision_not_installed")

        with patch(
            "modiff.custom_modular_inspection.inspect_installed_custom_modular_contract",
            side_effect=EnvironmentError("sidecar schema is invalid"),
        ):
            response = await server.custom_modular_inspect(Request())
        self.assertEqual(response.status, 422)
        self.assertEqual(json.loads(response.text)["code"], "custom_modular_inspection_failed")


if __name__ == "__main__":
    unittest.main()
