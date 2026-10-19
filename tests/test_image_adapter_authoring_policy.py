"""Selected image adapter authoring is separate from runtime qualification."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import ast
from importlib.metadata import distribution
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import modules as module_registry
from modiff.config import CONFIG
from modiff.diffusers_profiles import AdapterAuthoringPolicy, DIFFUSERS_EXECUTION_PROFILES, _execution_profile_from_definition
from modiff.model_artifact_catalog import catalog_profile_repository_revisions
from modiff.server import WebServer
from modiff.studio_execution_specs import studio_execution_profile_definitions


PROFILE_ID = "flux-dev:modular"
REPOSITORY = "black-forest-labs/FLUX.1-dev"
REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
EXPECTED_POLICY = {
    "schemaVersion": 1,
    "mediaKind": "image",
    "adapterNodeKey": "modules.ModularDiffusers.Lora",
    "tasks": ["text_to_image"],
    "maxOwnedAdapters": 32,
    "repositoryRevisions": [{"repo": REPOSITORY, "revision": REVISION}],
}
ALIAS_POLICIES = {
    "flux-schnell:modular": {
        **EXPECTED_POLICY,
        "repositoryRevisions": [{"repo": "black-forest-labs/FLUX.1-schnell",
                                 "revision": "741f7c3ce8b383c54771c7003378a50191e9efe9"}],
    },
    "flux-krea:modular": {
        **EXPECTED_POLICY,
        "repositoryRevisions": [{"repo": "black-forest-labs/FLUX.1-Krea-dev",
                                 "revision": "8162a9c7b05a641be098422bf2fcf335615c2f28"}],
    },
}
FLUX_POLICIES = {PROFILE_ID: EXPECTED_POLICY, **ALIAS_POLICIES}
EDIT_POLICIES = {
    "flux-kontext:modular": {
        **EXPECTED_POLICY,
        "tasks": ["edit_image"],
        "repositoryRevisions": [{"repo": "black-forest-labs/FLUX.1-Kontext-dev",
                                 "revision": "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d"}],
    },
    "flux2-klein:modular": {
        **EXPECTED_POLICY,
        "tasks": ["text_to_image", "edit_image"],
        "repositoryRevisions": [{"repo": "black-forest-labs/FLUX.2-klein-4B",
                                 "revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294"}],
    },
}
REVIEWED_POLICIES = {**FLUX_POLICIES, **EDIT_POLICIES}

# Conditional routes are separate; the five existing six-key policies stay exact.
CONDITIONAL_POLICIES = {'cosmos3-super-text-to-image:official-modular-workflow': {'schemaVersion': 1,
                                                           'mediaKind': 'image',
                                                           'adapterNodeKey': 'modules.ModularDiffusers.Lora',
                                                           'tasks': ['text_to_image'],
                                                           'maxOwnedAdapters': 32,
                                                           'repositoryRevisions': [{'repo': 'nvidia/Cosmos3-Super-Text2Image',
                                                                                    'revision': 'daf3d374804be4c512c2135568a7cb95d4341d79'}],
                                                           'requiredControls': {'offload_mode': 'none',
                                                                                'auto_offload': False,
                                                                                'weight_loading_mode': 'cpu',
                                                                                'quant_config': None,
                                                                                'unet': None,
                                                                                'vae': None,
                                                                                'controlnet': None,
                                                                                'trust_remote_code': False}},
 'flux-redux:direct': {'schemaVersion': 1,
                       'mediaKind': 'image',
                       'adapterNodeKey': 'modules.DiffusersImage.LoadAdapter',
                       'tasks': ['edit_image', 'multi_image_reference_edit'],
                       'maxOwnedAdapters': 32,
                       'repositoryRevisions': [{'repo': 'black-forest-labs/FLUX.1-Redux-dev',
                                                'revision': 'c95859fbf7703ca4d6824b4da4407d7cd0434f81'}],
                       'requiredControls': {'offload_mode': 'none',
                                            'auto_offload': False,
                                            'device_map': 'none',
                                            'quantization_mode': 'none',
                                            'prequantized_transformer': None}}}
CURRENT_POLICIES = {**REVIEWED_POLICIES, **CONDITIONAL_POLICIES}


class ImageAdapterAuthoringPublicationTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="modiff-image-adapter-policy-"))
        paths = {key: str(Path(directory) / key) for key in CONFIG.paths if key != "app_root"}
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        self.enterContext(patch.dict(CONFIG.paths, paths))
        self.enterContext(patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed node")))
        self.enterContext(patch.object(WebServer, "_optimization_runtime_context", return_value=(None, {}, {})))
        self.enterContext(patch("modiff.server.public_optimization_catalog", return_value={"capabilities": []}))

    def test_native_alias_policies_publish_only_their_own_reviewed_pair(self):
        payload = WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()
        for identity, expected in ALIAS_POLICIES.items():
            with self.subTest(profile=identity):
                copies = [profile for profiles in [payload["diffusersExecutionProfiles"], *[
                    capability["executionProfiles"]
                    for capability in payload["capabilities"] + payload["experimentalCapabilities"]
                ]] for profile in profiles if profile["id"] == identity]
                self.assertGreaterEqual(len(copies), 2)
                for published in copies:
                    self.assertEqual(published.get("adapterAuthoring"), expected)
                    self.assertEqual(published["repository_revisions"], expected["repositoryRevisions"])
                    self.assertFalse(published["live_proof"])
                    self.assertNotIn("lora_loadable_components", published)

    def test_reviewed_profile_publishes_the_exact_image_authoring_policy(self):
        profiles_before = deepcopy(DIFFUSERS_EXECUTION_PROFILES)
        published = DIFFUSERS_EXECUTION_PROFILES[PROFILE_ID].to_public_dict()
        self.assertEqual(published.get("adapterAuthoring"), EXPECTED_POLICY)
        self.assertNotIn("adapter_authoring", published)
        self.assertNotIn("lora_loadable_components", published)
        self.assertFalse(published["live_proof"])
        self.assertEqual(DIFFUSERS_EXECUTION_PROFILES, profiles_before)

    def test_current_metadata_copies_agree_without_promoting_capability_or_qualification(self):
        payload = WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()
        profiles = {profile["id"]: profile for profile in payload["diffusersExecutionProfiles"]}
        self.assertEqual(profiles[PROFILE_ID].get("adapterAuthoring"), EXPECTED_POLICY)
        found = []
        for capability in payload["capabilities"] + payload["experimentalCapabilities"]:
            for profile in capability["executionProfiles"]:
                if profile["id"] == PROFILE_ID:
                    found.append(capability)
                    self.assertEqual(profile["adapterAuthoring"], EXPECTED_POLICY)
                    self.assertEqual(profile["repository_revisions"], profiles[PROFILE_ID]["repository_revisions"])
        self.assertTrue(found)
        for capability in found:
            self.assertFalse(capability["supportsLora"])
            self.assertFalse(capability["liveProof"])
            self.assertFalse(capability["autoEligible"])
            self.assertFalse(capability["galleryEligible"])
            self.assertEqual(capability["qualifiedModes"], [])

    def test_edit_policies_keep_current_modes_and_exact_dense_pairs(self):
        payload = WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()
        profiles = {profile["id"]: profile for profile in payload["diffusersExecutionProfiles"]}
        for identity, expected in EDIT_POLICIES.items():
            with self.subTest(profile=identity):
                profile = profiles[identity]
                self.assertEqual(profile["modes"], ["text_to_image", "edit_image"])
                self.assertEqual(profile["adapterAuthoring"], expected)
                self.assertNotIn("multi_image_reference_edit", profile["adapterAuthoring"]["tasks"])
                self.assertFalse(profile["live_proof"])
                for capability in payload["capabilities"]:
                    for copy in capability["executionProfiles"]:
                        if copy["id"] == identity:
                            self.assertEqual(copy["adapterAuthoring"], expected)
                            self.assertFalse(capability["supportsLora"])
                            self.assertFalse(capability["autoEligible"])
                            self.assertEqual(capability["qualifiedModes"], [])
                pair = expected["repositoryRevisions"][0]
                for invalid in ([], [{**pair, "revision": "a" * 40}],
                                [{**pair, "repo": pair["repo"].lower()}], EXPECTED_POLICY["repositoryRevisions"]):
                    def catalog(profiles):
                        result = catalog_profile_repository_revisions(profiles)
                        result[identity] = invalid
                        return result
                    with patch("modiff.model_artifact_catalog.catalog_profile_repository_revisions", side_effect=catalog):
                        with self.assertRaisesRegex(ValueError, "adapter authoring does not match"):
                            WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()

    def test_only_reviewed_image_alias_owners_publish_the_existing_execution_mode(self):
        payload = WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()
        expected = {
            ("Flux2KleinModularPipeline", "multi_image_reference_edit", "diffusion.load_models"),
            ("Flux2ModularPipeline", "multi_image_reference_edit", "diffusion.load_models"),
            ("FluxKontextModularPipeline", "multi_image_reference_edit", "diffusion.load_models"),
        }
        observed = set()
        for contract in payload["operationContracts"]:
            binding = contract["binding"]
            if "executionMode" in binding:
                self.assertEqual(binding["executionMode"], "edit_image")
                self.assertEqual(contract["decomposition"], "loader")
                self.assertEqual(contract["nodeKey"], "modules.ModularDiffusers.ModelsLoader")
                self.assertEqual(binding["values"]["model_type"], contract["pipelineClass"])
                observed.add((contract["pipelineClass"], contract["task"], contract["operationId"]))
        self.assertEqual(observed, expected)

    def test_alias_projection_requires_an_explicit_image_output_declaration(self):
        from modiff.studio_execution_specs import studio_capability_definition

        for output in (None, "video", "audio"):
            def declared(model_type, *args, **kwargs):
                capability = studio_capability_definition(model_type, *args, **kwargs)
                if model_type == "FluxKontextModularPipeline":
                    capability.pop("modeOutputKinds", None)
                    if output is None:
                        capability.pop("outputKind", None)
                    else:
                        capability["outputKind"] = output
                return capability

            with self.subTest(output=output), patch(
                "modiff.studio_execution_specs.studio_capability_definition", side_effect=declared,
            ):
                payload = WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()
            owner = next(contract for contract in payload["operationContracts"]
                         if contract["pipelineClass"] == "FluxKontextModularPipeline"
                         and contract["task"] == "multi_image_reference_edit"
                         and contract["decomposition"] == "loader")
            self.assertNotIn("executionMode", owner["binding"])

    def test_other_profiles_have_no_new_permission_key(self):
        for identity, profile in DIFFUSERS_EXECUTION_PROFILES.items():
            if identity == "ace-step-audio:direct":
                self.assertEqual(profile.to_public_dict()["adapterAuthoring"], {
                    "schemaVersion": 1, "mediaKind": "audio",
                    "adapterNodeKey": "modules.DiffusersAudio.LoadAdapter",
                    "tasks": ["text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"],
                    "maxOwnedAdapters": 1,
                    "repositoryRevisions": [
                        {"repo": "ACE-Step/acestep-v15-xl-turbo-diffusers", "revision": "200ba991ae448051e14b0183157e35c2d27c9fb0"},
                        {"repo": "Runware/acestep-v15-turbo-diffusers", "revision": "be23effe449c5957947f3020fd63bee23c64abe4"},
                    ],
                })
            elif identity not in CURRENT_POLICIES:
                with self.subTest(profile=identity):
                    self.assertNotIn("adapterAuthoring", profile.to_public_dict())

    def test_catalog_binding_is_verified_once_for_every_public_copy(self):
        with patch("modiff.model_artifact_catalog.catalog_profile_repository_revisions",
                   wraps=catalog_profile_repository_revisions) as read:
            payload = WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()
        read.assert_called_once()
        for identity, expected in CURRENT_POLICIES.items():
            with self.subTest(profile=identity):
                policy = next(profile for profile in payload["diffusersExecutionProfiles"] if profile["id"] == identity)["adapterAuthoring"]
                self.assertEqual(policy, expected)
                self.assertIn(identity, {profile["id"] for profile in read.call_args.args[0]})

    def test_each_alias_requires_its_own_exact_catalog_pair(self):
        for identity, expected in ALIAS_POLICIES.items():
            pair = expected["repositoryRevisions"][0]
            for invalid in (None, [], [{**pair, "revision": "a" * 40}],
                            EXPECTED_POLICY["repositoryRevisions"],
                            [{**pair, "repo": pair["repo"].lower()}]):
                def altered_catalog(profiles):
                    result = catalog_profile_repository_revisions(profiles)
                    if invalid is None:
                        result.pop(identity, None)
                    else:
                        result[identity] = invalid
                    return result

                with self.subTest(profile=identity, pairs=invalid), patch(
                    "modiff.model_artifact_catalog.catalog_profile_repository_revisions",
                    side_effect=altered_catalog,
                ):
                    with self.assertRaisesRegex(ValueError, "adapter authoring does not match"):
                        WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()

    def test_missing_or_changed_reviewed_pin_cannot_publish_permission(self):
        for pairs in (None, [], [{"repo": REPOSITORY, "revision": "a" * 40}],
                      [{"repo": "black-forest-labs/FLUX.1-dev-FP8", "revision": REVISION}],
                      [{"repo": REPOSITORY.lower(), "revision": REVISION}]):
            with self.subTest(pairs=pairs), patch(
                "modiff.model_artifact_catalog.catalog_profile_repository_revisions",
                return_value={} if pairs is None else {PROFILE_ID: pairs},
            ):
                with self.assertRaisesRegex(ValueError, "adapter authoring does not match"):
                    WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()

    def test_actual_task_output_must_match_the_declared_authoring_media(self):
        from modiff.server import STUDIO_MODEL_CAPABILITIES

        capability = deepcopy(STUDIO_MODEL_CAPABILITIES["FluxModularPipeline"])
        capability["modeOutputKinds"] = {"text_to_image": "video", "image_to_image": "image"}
        with patch.dict(STUDIO_MODEL_CAPABILITIES, {"FluxModularPipeline": capability}):
            with self.assertRaisesRegex(ValueError, "reviewed image task"):
                WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()

    def test_incomplete_or_altered_serialized_policy_cannot_be_published(self):
        from modiff.server import public_execution_profiles

        for invalid in (None, {**EXPECTED_POLICY, "maxOwnedAdapters": 1},
                        {key: value for key, value in EXPECTED_POLICY.items() if key != "tasks"}):
            def profiles(*args, **kwargs):
                result = public_execution_profiles(*args, **kwargs)
                next(profile for profile in result if profile["id"] == PROFILE_ID)["adapterAuthoring"] = invalid
                return result

            with self.subTest(invalid=invalid), patch("modiff.server.public_execution_profiles", side_effect=profiles):
                with self.assertRaisesRegex(ValueError, "adapter authoring does not match"):
                    WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()

    def test_public_profile_copies_cannot_invent_policy_for_undeclared_profiles(self):
        from modiff.server import public_execution_profiles, public_experimental_pipelines

        baseline = WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()
        # Current experimental image records have no executable profiles. Use a
        # genuine published unsupported alias copy to exercise that second axis.
        capability_copy = next(capability for capability in baseline["capabilities"]
                               if any(profile["id"] == "flux2-klein-base:modular" for profile in capability["executionProfiles"]))
        for function_name, original in (("public_execution_profiles", public_execution_profiles),
                                        ("public_experimental_pipelines", public_experimental_pipelines)):
            for unexpected in (EXPECTED_POLICY, None):
                def profiles(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if function_name == "public_execution_profiles":
                        target = next(profile for profile in result if profile["id"] == "flux2-klein-base:modular")
                    else:
                        copy = deepcopy(capability_copy)
                        target = next(profile for profile in copy["executionProfiles"] if profile["id"] == "flux2-klein-base:modular")
                        result.append(copy)
                    target["adapterAuthoring"] = unexpected
                    return result

                with self.subTest(copy=function_name, policy=unexpected), patch(
                    f"modiff.server.{function_name}", side_effect=profiles,
                ):
                    with self.assertRaisesRegex(ValueError, "undeclared adapter authoring"):
                        WebServer(module_registry.MODULE_MAP)._build_model_capabilities_payload()

    def test_only_the_selected_public_policy_changes_and_registries_stay_immutable(self):
        original = deepcopy(DIFFUSERS_EXECUTION_PROFILES)
        server = WebServer(module_registry.MODULE_MAP)
        # The API serializes each profile occurrence independently; the in-memory
        # publisher may share the same profile object across metadata axes.
        for identity, expected in CURRENT_POLICIES.items():
            with self.subTest(profile=identity):
                selected = DIFFUSERS_EXECUTION_PROFILES[identity]
                with_policy = json.loads(json.dumps(server._build_model_capabilities_payload()))
                with patch.dict(DIFFUSERS_EXECUTION_PROFILES, {identity: replace(selected, adapter_authoring=None)}):
                    without_policy = json.loads(json.dumps(server._build_model_capabilities_payload()))
                copies = 0
                for profiles in [with_policy["diffusersExecutionProfiles"], *[
                    capability["executionProfiles"] for capability in with_policy["capabilities"] + with_policy["experimentalCapabilities"]
                ]]:
                    for profile in profiles:
                        if profile["id"] == identity:
                            copies += 1
                            self.assertEqual(profile.pop("adapterAuthoring"), expected)
                self.assertGreaterEqual(copies, 2)
                self.assertEqual(with_policy, without_policy, "All execution, qualification and other metadata must remain unchanged")
        self.assertEqual(DIFFUSERS_EXECUTION_PROFILES, original)

    def test_query_publication_retains_selected_policies_and_undeclared_aliases(self):
        server = WebServer(module_registry.MODULE_MAP)
        full = server._build_model_capabilities_payload()
        filtered = server._build_model_capabilities_payload("flux")
        self.assertEqual(full["diffusersExecutionProfiles"], filtered["diffusersExecutionProfiles"])
        for identity, expected in CURRENT_POLICIES.items():
            profile = next(profile for profile in filtered["diffusersExecutionProfiles"] if profile["id"] == identity)
            self.assertEqual(profile["adapterAuthoring"], expected)
        for identity in ("flux2-klein-base:modular", "flux2-klein-kv:t2i-modular"):
            profile = next(profile for profile in filtered["diffusersExecutionProfiles"] if profile["id"] == identity)
            self.assertNotIn("adapterAuthoring", profile)
        cosmos = next(profile for profile in filtered["diffusersExecutionProfiles"]
                      if profile["id"] == "cosmos3-super-text-to-image:official-modular-workflow")
        self.assertEqual(cosmos["adapterAuthoring"], CONDITIONAL_POLICIES[cosmos["id"]])
        self.assertFalse(cosmos["live_proof"])


class AdapterAuthoringPolicyValidationTests(unittest.TestCase):
    def policy(self, **changes):
        values = {
            "schema_version": 1,
            "media_kind": "image",
            "adapter_node_key": "modules.ModularDiffusers.Lora",
            "tasks": ("text_to_image",),
            "max_owned_adapters": 32,
            "repository_revisions": ((REPOSITORY, REVISION),),
        }
        return AdapterAuthoringPolicy(**{**values, **changes})

    def test_policy_is_immutable_and_public_values_are_independent(self):
        policy = self.policy()
        with self.assertRaises(FrozenInstanceError):
            policy.max_owned_adapters = 1
        published = policy.to_public_dict()
        published["repositoryRevisions"][0]["revision"] = "a" * 40
        published["tasks"].append("image_to_image")
        self.assertEqual(policy.to_public_dict(), EXPECTED_POLICY)

    def test_malformed_policy_fields_fail_closed(self):
        changes = (
            {"schema_version": True}, {"schema_version": 2}, {"media_kind": "audio"}, {"media_kind": "video"},
            {"adapter_node_key": "modules.DiffusersImage.LoadAdapter"}, {"adapter_node_key": "modules.Custom.Lora"},
            {"tasks": []}, {"tasks": ()}, {"tasks": ("text_to_image", "text_to_image")},
            {"tasks": ("text_to_image ",)}, {"tasks": tuple(f"task_{index}" for index in range(33))},
            {"max_owned_adapters": True}, {"max_owned_adapters": 0}, {"max_owned_adapters": 33}, {"max_owned_adapters": 1.5},
            {"repository_revisions": []}, {"repository_revisions": ()},
            {"repository_revisions": ((REPOSITORY, "main"),)},
            {"repository_revisions": ((REPOSITORY, REVISION.upper()),)},
            {"repository_revisions": ((REPOSITORY, REVISION), (REPOSITORY.lower(), REVISION))},
            {"repository_revisions": (("../model", REVISION),)},
            {"repository_revisions": (("publisher/model/extra", REVISION),)},
            {"repository_revisions": ((REPOSITORY, REVISION, "extra"),)},
            {"repository_revisions": tuple((f"Publisher/Model-{index}", REVISION) for index in range(33))},
        )
        for values in changes:
            with self.subTest(values=values):
                if values == {"adapter_node_key": "modules.DiffusersImage.LoadAdapter"}:
                    # This enum now has a reviewed direct route, but cannot be
                    # assigned to the original native loader/profile authority.
                    with self.assertRaisesRegex(ValueError, "incompatible adapter authoring policy"):
                        replace(DIFFUSERS_EXECUTION_PROFILES[PROFILE_ID], adapter_authoring=self.policy(**values))
                else:
                    with self.assertRaisesRegex(ValueError, "Invalid reviewed adapter authoring policy"):
                        self.policy(**values)

    def test_policy_cannot_escape_the_actual_profile_tasks_repository_or_loader(self):
        profile = DIFFUSERS_EXECUTION_PROFILES[PROFILE_ID]
        for values in (
            {"adapter_authoring": {}},
            {"adapter_authoring": self.policy(tasks=("edit_image",))},
            {"adapter_authoring": self.policy(repository_revisions=(("Publisher/Other", REVISION),))},
            {"execution_path": "direct-diffusers-image", "loader_module": "modules.DiffusersImage", "loader_action": "LoadPipeline"},
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "incompatible adapter authoring policy"):
                replace(profile, **values)

    def test_raw_declaration_has_exact_fields_and_is_converted_without_mutation(self):
        declaration = studio_execution_profile_definitions()[PROFILE_ID]
        original = deepcopy(declaration)
        self.assertEqual(_execution_profile_from_definition(declaration).adapter_authoring.to_public_dict(), EXPECTED_POLICY)
        self.assertEqual(declaration, original)
        for invalid in (None, (), {**declaration["adapter_authoring"], "unknown": True},
                        {key: value for key, value in declaration["adapter_authoring"].items() if key != "tasks"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _execution_profile_from_definition({**declaration, "adapter_authoring": invalid})

    def test_declarative_policy_import_does_not_import_model_libraries(self):
        command = "\n".join([
            "import sys",
            "from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES",
            f"assert DIFFUSERS_EXECUTION_PROFILES[{PROFILE_ID!r}].adapter_authoring.to_public_dict() == {EXPECTED_POLICY!r}",
            "assert DIFFUSERS_EXECUTION_PROFILES['flux-dev:modular'].adapter_authoring.media_kind == 'image'",
            "assert not {'torch', 'diffusers', 'transformers', 'peft', 'modules'} & set(sys.modules)",
        ])
        result = subprocess.run([sys.executable, "-B", "-c", command], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_optional_execution_mode_is_bounded_and_model_owner_only(self):
        from modiff.operation_contracts import with_operation_semantics

        owner = {"pipelineClass": "FixtureImagePipeline", "task": "multi_image_reference_edit",
                 "decomposition": "loader", "ports": []}
        plain = with_operation_semantics(owner)
        self.assertNotIn("executionMode", plain["binding"])
        self.assertEqual(with_operation_semantics(owner, execution_mode="edit_image")["binding"]["executionMode"], "edit_image")
        for invalid in (False, 0, {}, [], "", "edit_image\n", " edit_image", "EDIT_IMAGE", "a" * 65, "constructor", "__proto__"):
            with self.subTest(mode=invalid), self.assertRaisesRegex(ValueError, "Invalid operation execution mode"):
                with_operation_semantics(owner, execution_mode=invalid)
        for decomposition in ("block", "bundle", "pipeline"):
            with self.subTest(decomposition=decomposition), self.assertRaisesRegex(ValueError, "Invalid operation execution mode"):
                with_operation_semantics({**owner, "decomposition": decomposition}, execution_mode="edit_image")

    def test_contextual_installed_classes_have_their_actual_inherited_lora_targets(self):
        installed = distribution("diffusers")
        self.assertEqual(installed.version, "0.41.0")
        mixins = ast.parse(Path(installed.locate_file("diffusers/loaders/lora_pipeline.py")).read_text(encoding="utf-8"))
        for identity, family, child_name, parent_name, mixin_name in (
            ("flux-kontext:modular", "flux", "FluxKontextModularPipeline", "FluxModularPipeline", "FluxLoraLoaderMixin"),
            ("flux2-klein:modular", "flux2", "Flux2KleinModularPipeline", "Flux2ModularPipeline", "Flux2LoraLoaderMixin"),
        ):
            with self.subTest(profile=identity):
                source = Path(installed.locate_file(f"diffusers/modular_pipelines/{family}/modular_pipeline.py"))
                tree = ast.parse(source.read_text(encoding="utf-8"))
                child = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == child_name)
                parent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == parent_name)
                self.assertIn(parent_name, [node.id for node in child.bases if isinstance(node, ast.Name)])
                self.assertIn(mixin_name, [node.id for node in parent.bases if isinstance(node, ast.Name)])
                mixin = next(node for node in mixins.body if isinstance(node, ast.ClassDef) and node.name == mixin_name)
                targets = next(ast.literal_eval(node.value) for node in mixin.body if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == "_lora_loadable_modules" for target in node.targets))
                profile = DIFFUSERS_EXECUTION_PROFILES[identity]
                self.assertEqual(profile.pipeline_class, child_name)
                self.assertEqual(profile.lora_loadable_components, tuple(targets))
                self.assertIn("load_lora_weights", [node.name for node in mixin.body if isinstance(node, ast.FunctionDef)])

    def test_reviewed_installed_flux_class_declares_the_official_lora_api_and_targets(self):
        installed = distribution("diffusers")
        self.assertEqual(installed.version, "0.41.0")
        pipeline_source = Path(installed.locate_file("diffusers/modular_pipelines/flux/modular_pipeline.py"))
        mixin_source = Path(installed.locate_file("diffusers/loaders/lora_pipeline.py"))
        pipeline = next(node for node in ast.parse(pipeline_source.read_text(encoding="utf-8")).body
                        if isinstance(node, ast.ClassDef) and node.name == "FluxModularPipeline")
        self.assertIn("FluxLoraLoaderMixin", [node.id for node in pipeline.bases if isinstance(node, ast.Name)])
        mixin = next(node for node in ast.parse(mixin_source.read_text(encoding="utf-8")).body
                     if isinstance(node, ast.ClassDef) and node.name == "FluxLoraLoaderMixin")
        targets = next(ast.literal_eval(node.value) for node in mixin.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "_lora_loadable_modules" for target in node.targets))
        for identity in FLUX_POLICIES:
            with self.subTest(profile=identity):
                profile = DIFFUSERS_EXECUTION_PROFILES[identity]
                self.assertEqual(profile.pipeline_class, "FluxModularPipeline")
                self.assertEqual(tuple(targets), profile.lora_loadable_components)
        self.assertIn("load_lora_weights", [node.name for node in mixin.body if isinstance(node, ast.FunctionDef)])


if __name__ == "__main__":
    unittest.main()
