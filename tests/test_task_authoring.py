"""Task selection uses reviewed starters without constructing or loading models."""

from copy import deepcopy
import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from modules import MODULE_MAP
from modiff.diffusers_profiles import public_execution_profiles
from modiff.operation_catalog import build_operation_catalog
from modiff.task_authoring import resolve_task_starter, starter_api_graph, unbind_starter


class TaskAuthoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        profiles = public_execution_profiles()
        contracts, support = build_operation_catalog(MODULE_MAP, profiles, catalog_resolver=lambda: {})
        # Readiness is an independent runtime probe; keep this selection fixture
        # deterministic in both base and activated optional-runtime environments.
        for pipeline in support:
            for task in pipeline["tasks"]:
                task["dependencies"] = "ready"
        cls.catalog = {"operationContracts": contracts, "pipelineSupport": support,
                       "diffusersExecutionProfiles": profiles}

    def resolve(self, selection=None, *, available=(), fits=lambda graph: {"canAutoRun": True}, catalog=None):
        with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed model")):
            return resolve_task_starter(
                MODULE_MAP, catalog or self.catalog, selection or {"task": "text_to_image"},
                installed=lambda profile: profile["id"] in available, inspect_resources=fits,
            )

    def test_only_installed_compatible_model_is_bound(self):
        result = self.resolve(available=("flux-schnell:direct",))
        self.assertFalse(result["unbound"])
        self.assertEqual(result["profileId"], "flux-schnell:direct")

    def test_remembered_profile_is_advisory_and_memory_is_still_checked(self):
        selection = {"task": "text_to_image", "preferredProfileId": "flux-krea:direct"}
        installed = ("flux-schnell:direct", "flux-krea:direct")
        self.assertEqual(self.resolve(selection, available=installed)["profileId"], "flux-krea:direct")
        checked = []

        def fits(graph):
            repo = graph["nodes"]["diffusion.load_models"]["params"]["model_id"]["value"]["value"]
            checked.append(repo)
            return {"canAutoRun": repo.endswith("schnell")}

        result = self.resolve(selection, available=installed, fits=fits)
        self.assertEqual(result["profileId"], "flux-schnell:direct")
        self.assertEqual(checked, ["black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.1-schnell"])
        self.assertEqual(self.resolve({**selection, "preferredProfileId": "unknown"}, available=(installed[0],))["profileId"], installed[0])

    def test_unavailable_or_memory_limited_selection_leaves_required_empty_model(self):
        for available in ((), ("flux-schnell:direct",)):
            with self.subTest(available=available):
                result = self.resolve(available=available, fits=lambda graph: {"canAutoRun": False})
                self.assertTrue(result["unbound"])
                self.assertIsNone(result["profileId"])
                loader = result["starter"]["nodes"][0]
                key = "repo_id" if loader["action"] == "ModelsLoader" else "model_id"
                self.assertEqual(loader["params"][key]["value"], {"source": "hub", "value": ""})
                self.assertEqual(loader["params"][key]["default"], loader["params"][key]["value"])
                self.assertTrue(loader["params"][key]["required"])
                self.assertIn({"operationId": loader["operation"]["operationId"], "field": key}, result["starter"]["requiredInputs"])

    def test_custom_memory_policy_skips_auto_recipe_but_not_runtime_readiness(self):
        catalog = deepcopy(self.catalog)
        for pipeline in catalog["pipelineSupport"]:
            if pipeline["pipelineClass"] == "FluxPipeline":
                for task in pipeline["tasks"]:
                    task["dependencies"] = "blocked"
        selection = {"task": "text_to_image", "resourceMode": "expert"}
        with patch("modiff.task_authoring.starter_api_graph", side_effect=AssertionError("Auto in custom policy")):
            self.assertFalse(self.resolve(selection, available=("flux-schnell:direct",))["unbound"])
            self.assertTrue(self.resolve(selection, available=("flux-schnell:direct",), catalog=catalog)["unbound"])

    def test_invalid_selections_do_not_touch_model_inventory(self):
        for selection in ({}, {"task": "text_to_image", "module": "remote"}, {"task": "../bad"},
                          {"task": "text_to_image", "preferredProfileId": []},
                          {"task": "text_to_image", "resourceMode": "other"}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                resolve_task_starter(MODULE_MAP, self.catalog, selection,
                                     installed=lambda profile: self.fail("unexpected inventory"), inspect_resources=None)

    def test_declared_task_without_published_profiles_still_creates_an_unbound_draft(self):
        catalog = deepcopy(self.catalog)
        catalog["diffusersExecutionProfiles"] = []
        result = self.resolve(catalog=catalog)
        self.assertTrue(result["unbound"])
        self.assertIsNone(result["profileId"])
        self.assertGreater(len(result["starter"]["nodes"]), 1)

    def test_initial_choice_prefers_smaller_known_memory_requirement_over_alphabetical_model(self):
        def inspect(graph):
            repo = graph["nodes"]["diffusion.load_models"]["params"]["model_id"]["value"]["value"]
            return {"canAutoRun": True, "requirements": {"systemRamBytes": 2 if repo.endswith("schnell") else 20, "vramBytes": 1}}
        result = self.resolve(available=("flux-krea:direct", "flux-schnell:direct"), fits=inspect)
        self.assertEqual(result["profileId"], "flux-schnell:direct")

    def test_native_modular_projection_retains_wires_and_unbinding_does_not_mutate_source(self):
        from modiff.operation_starters import resolve_operation_starter

        starter = resolve_operation_starter(MODULE_MAP, self.catalog["operationContracts"],
                                           {"pipelineClass": "ZImageModularPipeline", "task": "text_to_image"})
        original = deepcopy(starter)
        unbound = unbind_starter(starter)
        self.assertEqual(starter, original)
        graph = starter_api_graph(unbound)
        self.assertCountEqual(
            [node["action"] for node in graph["nodes"].values()],
            ["ModelsLoader", "Guider", "EncodePrompt", "Denoise", "DecodeLatents"],
        )
        for edge in unbound["edges"]:
            self.assertEqual(graph["nodes"][edge["target"]]["params"][edge["targetHandle"]],
                             {"sourceId": edge["source"], "sourceKey": edge["sourceHandle"]})
            self.assertLess(graph["paths"][0].index(edge["source"]), graph["paths"][0].index(edge["target"]))

    def task_starter_http(self, *, free_ram, reclaimable, active=False):
        """Exercise the real selector/planner with controlled recipe/storage bytes.

        No weights are allocated: this checks HTTP wiring and planning, while
        dispatch's actual release/recheck is covered by workflow executor tests.
        """
        from modiff.config import CONFIG
        from modiff.model_artifact_catalog import require_catalog_revision
        from modiff.workflow_auto_resource import build_workflow_auto_plan

        gib = 1024 ** 3
        fingerprint = {"hardware": {"system": {"ram_available": free_ram}}}
        snapshots, inspected = [], []
        cache = {"owners": {}, "reclaimable": {"systemRamBytes": reclaimable, "vramBytes": 0}}
        profile = next(item for item in self.catalog["diffusersExecutionProfiles"] if item["id"] == "flux-schnell:direct")

        def recipe(payload, **kwargs):
            form = payload["form"]
            return {"candidates": [{
                "id": "controlled-starter-memory-envelope", "modelRepo": form["modelRepo"],
                "artifactRevision": require_catalog_revision(form["modelRepo"], model_type=form["modelType"]),
                "modelType": form["modelType"], "mode": form["mode"],
                "executionProfileId": profile["id"],
                "loaderModule": profile["loader_module"], "loaderAction": profile["loader_action"],
                "executionPath": profile["execution_path"], "dtype": form["dtype"],
                "quantizationMode": form["quantizationMode"], "offloadMode": form["offloadMode"],
                "autoOffload": form["autoOffload"], "generation": {"batchSize": 1},
                "canAutoRun": True, "proof": {"status": "declared_safe"},
                "requirements": {"systemRamBytes": 8 * gib, "vramBytes": 0},
            }]}

        def inspect(graph, **kwargs):
            snapshots.append(kwargs.get("cache_snapshot"))
            hardware = {
                "accelerator": {"kind": "cpu", "totalBytes": None, "freeBytes": None},
                "systemMemory": {"totalBytes": 32 * gib,
                                 "availableBytes": kwargs["runtime_fingerprint"]["hardware"]["system"]["ram_available"]},
                "offloadDisk": {},
            }
            result = build_workflow_auto_plan(graph, **kwargs, hardware=hardware, plan_recipe=recipe)
            inspected.append(result)
            return result

        with tempfile.TemporaryDirectory(prefix="modiff-task-starter-test-") as directory:
            original_paths = dict(CONFIG.paths)
            try:
                for name in CONFIG.paths:
                    if name != "app_root":
                        CONFIG.paths[name] = str(Path(directory) / name)
                        Path(CONFIG.paths[name]).mkdir(parents=True, exist_ok=True)
                from modiff.server import WebServer

                server = SimpleNamespace(
                    modules=MODULE_MAP, data_dir=directory, current_task={"task_id": "active"} if active else None,
                    _model_capabilities_response=AsyncMock(return_value=json.dumps(self.catalog).encode()),
                    _runtime_fingerprint=Mock(return_value=fingerprint),
                    _runtime_fingerprint_for_control_request=Mock(return_value=fingerprint),
                    _auto_planning_runtime_fingerprint=Mock(return_value=fingerprint),
                    _workflow_auto_cache_snapshot=Mock(return_value=cache),
                    _describe_registered_node=Mock(return_value={}),
                )
                request = SimpleNamespace(json=AsyncMock(return_value={
                    "task": "text_to_image", "preferredProfileId": "flux-schnell:direct",
                }))
                with (
                    patch("modiff.server.get_local_models", return_value=[]),
                    patch("modiff.auto_resource.artifact_revision_cache_status",
                          side_effect=lambda repo, *_: {"complete": repo.endswith("schnell")}),
                    patch("modiff.workflow_auto_resource.build_workflow_auto_plan", side_effect=inspect),
                ):
                    response = asyncio.run(WebServer.resolve_task_starter(server, request))
            finally:
                CONFIG.paths.clear()
                CONFIG.paths.update(original_paths)
        self.assertEqual(response.status, 200)
        return json.loads(response.text), server, snapshots, inspected

    def test_task_starter_can_plan_release_of_measured_previous_cache(self):
        result, server, snapshots, inspected = self.task_starter_http(
            free_ram=5 * 1024 ** 3, reclaimable=4 * 1024 ** 3,
        )
        self.assertFalse(result["unbound"], [item["issues"] for item in inspected])
        self.assertEqual(result["profileId"], "flux-schnell:direct")
        self.assertTrue(inspected[0]["requiresCachePreparation"])
        self.assertEqual(inspected[0]["reusedOwnerIds"], [])
        self.assertEqual(inspected[0]["available"]["systemRamBytes"], 5 * 1024 ** 3)
        self.assertEqual(snapshots[0]["reclaimable"]["systemRamBytes"], 4 * 1024 ** 3)
        server._runtime_fingerprint.assert_called_once()
        server._workflow_auto_cache_snapshot.assert_called_once()
        server._auto_planning_runtime_fingerprint.assert_not_called()

    def test_task_starter_unknown_capacity_and_external_pressure_remain_unbound(self):
        for free_ram, reclaimable in ((None, 8 * 1024 ** 3), (2 * 1024 ** 3, 4 * 1024 ** 3),
                                      (5 * 1024 ** 3, 0)):
            with self.subTest(free_ram=free_ram, reclaimable=reclaimable):
                result, _, _, inspected = self.task_starter_http(free_ram=free_ram, reclaimable=reclaimable)
                self.assertTrue(result["unbound"])
                self.assertFalse(inspected[0]["canAutoRun"])
                self.assertFalse(inspected[0]["requiresCachePreparation"])

    def test_active_task_starter_uses_cached_capacity_and_no_live_owner_observations(self):
        result, server, snapshots, inspected = self.task_starter_http(
            free_ram=5 * 1024 ** 3, reclaimable=4 * 1024 ** 3, active=True,
        )
        self.assertTrue(result["unbound"])
        self.assertFalse(inspected[0]["canAutoRun"])
        self.assertTrue(snapshots)
        self.assertTrue(all(snapshot == {} for snapshot in snapshots))
        server._runtime_fingerprint.assert_not_called()
        server._workflow_auto_cache_snapshot.assert_not_called()
        server._runtime_fingerprint_for_control_request.assert_called_once()
