"""Portable selected audio authoring and starter consistency regressions."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import modules
from modiff.config import CONFIG
from modiff.diffusers_profiles import AdapterAuthoringPolicy, DIFFUSERS_EXECUTION_PROFILES, public_execution_profiles
from modiff.model_artifact_catalog import catalog_profile_repository_revisions
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modiff.server import WebServer
from modules.DiffusersAudio.main import AUDIO_PIPELINE_ADAPTERS, _audio_contract_signal

IDENTITY = "ace-step-audio:direct"
TASKS = ("text_to_audio", "audio_variation", "audio_continuation", "audio_repaint")
PAIRS = (
    ("ACE-Step/acestep-v15-xl-turbo-diffusers", "200ba991ae448051e14b0183157e35c2d27c9fb0"),
    ("Runware/acestep-v15-turbo-diffusers", "be23effe449c5957947f3020fd63bee23c64abe4"),
)
EXPECTED = {"schemaVersion": 1, "mediaKind": "audio", "adapterNodeKey": "modules.DiffusersAudio.LoadAdapter",
            "tasks": list(TASKS), "maxOwnedAdapters": 1,
            "repositoryRevisions": [{"repo": repo, "revision": revision} for repo, revision in PAIRS]}


def policy(**changes):
    values = dict(schema_version=1, media_kind="audio", adapter_node_key="modules.DiffusersAudio.LoadAdapter",
                  tasks=TASKS, max_owned_adapters=1, repository_revisions=PAIRS)
    values.update(changes)
    return AdapterAuthoringPolicy(**values)


class AudioAuthoringPolicyTests(unittest.TestCase):
    def test_selected_ace_profile_has_exact_single_adapter_scope(self):
        profile = DIFFUSERS_EXECUTION_PROFILES[IDENTITY]
        self.assertIsNotNone(profile.adapter_authoring)
        self.assertEqual(profile.to_public_dict().get("adapterAuthoring"), EXPECTED)
        self.assertFalse(profile.live_proof)
        self.assertEqual(profile.lora_loadable_components, ())

    def test_audio_policy_is_immutable_and_public_copies_are_detached(self):
        declared = policy()
        self.assertEqual(declared.to_public_dict(), EXPECTED)
        with self.assertRaises(FrozenInstanceError):
            declared.max_owned_adapters = 2
        first = declared.to_public_dict()
        first["tasks"].clear()
        first["repositoryRevisions"][0]["repo"] = "foreign/repo"
        self.assertEqual(declared.to_public_dict(), EXPECTED)

    def test_policy_transport_media_and_audio_capacity_fail_closed(self):
        for changes in (
            {"media_kind": []}, {"adapter_node_key": []},
            {"media_kind": "video"}, {"media_kind": "image"},
            {"adapter_node_key": "modules.DiffusersImage.LoadAdapter"},
            {"adapter_node_key": "modules.ModularDiffusers.Lora"},
            {"max_owned_adapters": 2}, {"max_owned_adapters": True},
            {"tasks": ("text_to_audio", "text_to_audio")}, {"tasks": ("text_to_audio\n",)},
            {"repository_revisions": ((PAIRS[0][0], "main"),)},
            {"repository_revisions": (PAIRS[0], (PAIRS[0][0].lower(), PAIRS[0][1]))},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                policy(**changes)

    def test_profile_owner_task_and_repository_cannot_borrow_audio_permission(self):
        declared = policy()
        for identity, changes in (
            ("flux-dev:modular", {}),
            (IDENTITY, {"modes": ("text_to_audio",)}),
            (IDENTITY, {"compatible_repos": ()}),
            (IDENTITY, {"execution_path": "direct-diffusers-image", "loader_module": "modules.DiffusersImage"}),
        ):
            with self.subTest(identity=identity, changes=changes), self.assertRaises(ValueError):
                replace(DIFFUSERS_EXECUTION_PROFILES[identity], adapter_authoring=declared, **changes)

    def test_unrelated_audio_profiles_keep_absent_authoring(self):
        for profile in public_execution_profiles():
            if profile["id"] != IDENTITY and profile["loader_module"] in {"modules.DiffusersAudio", "modules.HuggingFaceSpeech"}:
                with self.subTest(profile=profile["id"]):
                    self.assertNotIn("adapterAuthoring", profile)


class AudioAuthoringPublicationTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="modiff-audio-authoring-"))
        paths = {key: str(Path(directory) / key) for key in CONFIG.paths if key != "app_root"}
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        self.enterContext(patch.dict(CONFIG.paths, paths))
        self.enterContext(patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed node")))
        self.enterContext(patch.object(WebServer, "_optimization_runtime_context", return_value=(None, {}, {})))
        self.enterContext(patch("modiff.server.public_optimization_catalog", return_value={"capabilities": []}))

    def payload(self, registry=None):
        return WebServer(modules.MODULE_MAP if registry is None else registry)._build_model_capabilities_payload()

    def test_actual_publication_copies_and_catalog_read_are_consistent(self):
        with patch("modiff.model_artifact_catalog.catalog_profile_repository_revisions", wraps=catalog_profile_repository_revisions) as catalog:
            payload = self.payload()
        self.assertEqual(catalog.call_count, 1)
        copies = [profile for rows in [payload["diffusersExecutionProfiles"], *[
            capability["executionProfiles"] for capability in payload["capabilities"] + payload["experimentalCapabilities"]
        ]] for profile in rows if profile["id"] == IDENTITY]
        self.assertGreaterEqual(len(copies), 2)
        for profile in copies:
            self.assertEqual(profile.get("adapterAuthoring"), EXPECTED)
            self.assertEqual(profile.get("repository_revisions"), EXPECTED["repositoryRevisions"])
            self.assertFalse(profile["live_proof"])
        capability = next(item for item in payload["capabilities"] if item["modelType"] == "AceStepAudioPipeline")
        from modiff.server import STUDIO_MODEL_CAPABILITIES
        declared = STUDIO_MODEL_CAPABILITIES["AceStepAudioPipeline"]
        self.assertFalse(capability["supportsLora"])
        for name in ("liveProof", "autoEligible", "galleryEligible", "qualifiedModes", "templateEligible"):
            self.assertEqual(capability.get(name), declared.get(name))

    def test_removing_only_the_new_policy_and_audio_pin_sidecar_preserves_entire_metadata(self):
        import json
        original = deepcopy(DIFFUSERS_EXECUTION_PROFILES)
        with_policy = json.loads(json.dumps(self.payload()))
        with patch.dict(DIFFUSERS_EXECUTION_PROFILES, {
            IDENTITY: replace(DIFFUSERS_EXECUTION_PROFILES[IDENTITY], adapter_authoring=None),
        }):
            without_policy = json.loads(json.dumps(self.payload()))
        for rows in [with_policy["diffusersExecutionProfiles"], *[
            item["executionProfiles"] for item in with_policy["capabilities"] + with_policy["experimentalCapabilities"]
        ]]:
            for profile in rows:
                if profile["id"] == IDENTITY:
                    self.assertEqual(profile.pop("adapterAuthoring"), EXPECTED)
                    self.assertEqual(profile.pop("repository_revisions"), EXPECTED["repositoryRevisions"])
        self.assertEqual(with_policy, without_policy)
        self.assertEqual(DIFFUSERS_EXECUTION_PROFILES, original)

    def test_catalog_mismatch_refuses_audio_grant(self):
        for pairs in ([], [{"repo": PAIRS[0][0], "revision": "a" * 40}],
                      [{"repo": PAIRS[0][0].lower(), "revision": PAIRS[0][1]}]):
            def changed(profiles):
                result = catalog_profile_repository_revisions(profiles)
                result[IDENTITY] = pairs
                return result
            with self.subTest(pairs=pairs), patch("modiff.model_artifact_catalog.catalog_profile_repository_revisions", side_effect=changed):
                with self.assertRaisesRegex(ValueError, "adapter authoring does not match"):
                    self.payload()

    def test_audio_task_output_kind_mismatch_refuses_grant(self):
        from modiff.server import STUDIO_MODEL_CAPABILITIES
        for kind in ("image", "video", "text"):
            capabilities = deepcopy(STUDIO_MODEL_CAPABILITIES)
            capabilities["AceStepAudioPipeline"]["modeOutputKinds"] = {TASKS[0]: kind}
            with self.subTest(kind=kind), patch("modiff.server.STUDIO_MODEL_CAPABILITIES", capabilities):
                with self.assertRaisesRegex(ValueError, "adapter authoring does not match"):
                    self.payload()

    def test_missing_or_foreign_registered_audio_adapter_refuses_grant(self):
        for mutation in ("missing", "foreign_input", "foreign_output", "missing_compatibility",
                         "metadata_nondict", "params_nondict", "input_nondict", "output_nondict",
                         "missing_forwarder", "foreign_forwarder", "missing_output_signal", "malformed_output_signal",
                         "foreign_signal_direction", "foreign_signal_origin"):
            registry = deepcopy(modules.MODULE_MAP)
            if mutation == "missing":
                registry["modules.DiffusersAudio"].pop("LoadAdapter")
            elif mutation == "metadata_nondict":
                registry["modules.DiffusersAudio"]["LoadAdapter"] = []
            elif mutation == "params_nondict":
                registry["modules.DiffusersAudio"]["LoadAdapter"]["params"] = []
            else:
                params = registry["modules.DiffusersAudio"]["LoadAdapter"]["params"]
                if mutation == "foreign_input":
                    params["pipeline"]["type"] = "image_diffusion_pipeline"
                elif mutation == "foreign_output":
                    params["output"]["type"] = "image_diffusion_pipeline"
                elif mutation == "missing_compatibility":
                    params["pipeline"].pop("signalCompatibility")
                elif mutation == "missing_forwarder":
                    params["pipeline"].pop("onSignal")
                elif mutation == "foreign_forwarder":
                    params["pipeline"]["onSignal"]["target"] = "foreign"
                elif mutation == "missing_output_signal":
                    params["output"].pop("signal")
                elif mutation == "malformed_output_signal":
                    params["output"]["signal"] = []
                elif mutation == "foreign_signal_direction":
                    params["output"]["signal"]["direction"] = "input"
                elif mutation == "foreign_signal_origin":
                    params["output"]["signal"]["origin"] = "foreign"
                else:
                    params["pipeline" if mutation == "input_nondict" else "output"] = []
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "registered audio adapter"):
                self.payload(registry)


class SelectedAudioStarterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts, _ = build_operation_catalog(modules.MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})

    def starter(self, task, repository):
        return resolve_operation_starter(modules.MODULE_MAP, self.contracts, {
            "pipelineClass": "AceStepPipeline", "task": task,
            "executionProfileId": IDENTITY, "repository": repository,
        })

    def test_all_ace_tasks_and_reviewed_repositories_project_exact_selected_contract(self):
        adapter = AUDIO_PIPELINE_ADAPTERS["AceStepPipeline"]
        for task in TASKS:
            for repository, revision in PAIRS:
                with self.subTest(task=task, repository=repository), patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed node")):
                    result = self.starter(task, repository)
                loader, generate = result["nodes"]
                expected = _audio_contract_signal(adapter, task, {"source": "hub", "value": repository})
                self.assertEqual(loader["values"]["model_id"], {"source": "hub", "value": repository})
                self.assertEqual(loader["values"]["revision"], revision)
                self.assertNotIn("execution_profile_id", loader["params"])
                self.assertEqual(loader["params"]["pipeline"]["signal"]["value"], expected)
                for node in (loader, generate):
                    self.assertEqual(node["params"]["audio_contract"]["value"], expected)
                    self.assertEqual(node["values"]["audio_contract"], expected)
                mode = adapter.contract_for_mode(task)
                self.assertEqual(generate["values"]["task_type"], mode.task_type)
                for field, overlay in mode.field_param_overlay().items():
                    for key, value in overlay.items():
                        self.assertEqual(generate["params"][field].get(key), value)
                required = {(row["operationId"], row["field"]) for row in result["requiredInputs"]}
                self.assertEqual(("diffusion.generate_audio", "source_audio") in required, task != "text_to_audio")

    def test_repository_change_preserves_creative_values_operation_identity_and_edges(self):
        for task in TASKS:
            with self.subTest(task=task):
                first, second = (self.starter(task, pair[0]) for pair in PAIRS)
                self.assertEqual(first["edges"], second["edges"])
                self.assertEqual(first["requiredInputs"], second["requiredInputs"])
                for a, b in zip(first["nodes"], second["nodes"]):
                    self.assertEqual(a["operation"], b["operation"])
                    for name in ("prompt", "negative_prompt", "lyrics", "seed", "audio_duration", "num_inference_steps", "guidance_scale", "shift", "bpm", "keyscale", "timesignature", "delivery_sample_rate"):
                        if name in a["params"]:
                            self.assertEqual(a["params"][name], b["params"][name])
                            self.assertEqual(a["values"].get(name), b["values"].get(name))

    def test_foreign_repository_or_task_refuses_before_projection(self):
        for repository in ("foreign/model", " " + PAIRS[0][0], "main", None):
            with self.subTest(repository=repository), self.assertRaises(ValueError):
                self.starter("text_to_audio", repository)
        with self.assertRaises(ValueError):
            self.starter("text_to_video", PAIRS[0][0])


if __name__ == "__main__":
    unittest.main()
