"""Task selection uses reviewed starters without constructing or loading models."""

from copy import deepcopy
import unittest
from unittest.mock import patch

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
        self.assertEqual(len(graph["nodes"]), 4)
        for edge in unbound["edges"]:
            self.assertEqual(graph["nodes"][edge["target"]]["params"][edge["targetHandle"]],
                             {"sourceId": edge["source"], "sourceKey": edge["sourceHandle"]})
            self.assertLess(graph["paths"][0].index(edge["source"]), graph["paths"][0].index(edge["target"]))
