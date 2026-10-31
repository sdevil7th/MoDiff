"""Image authoring pins must bind the selected profile to its actual repository."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import modules as module_registry
from modiff.config import CONFIG
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, EXPERIMENTAL_DIFFUSERS_PIPELINES
import modiff.model_artifact_catalog as artifact_catalog
from modiff.server import STUDIO_MODEL_CAPABILITIES, WebServer


class ImageProfileRepositoryRevisionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="modiff-image-pins-"))
        paths = {key: str(Path(directory) / key) for key in CONFIG.paths if key != "app_root"}
        for path in paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        self.enterContext(patch.dict(CONFIG.paths, paths))

    async def test_exact_image_profile_pin_matches_the_selected_starter_loader(self):
        server = WebServer(module_registry.MODULE_MAP)
        with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed executable node")):
            response = await server.model_capabilities(SimpleNamespace(query={}))
            payload = json.loads(response.text)
            profiles = {profile["id"]: profile for profile in payload["diffusersExecutionProfiles"]}
            for pipeline, identity, task, repository, revision in (
                ("FluxPipeline", "flux-schnell:direct", "text_to_image", "black-forest-labs/FLUX.1-schnell",
                 "741f7c3ce8b383c54771c7003378a50191e9efe9"),
                ("FluxModularPipeline", "flux-schnell:modular", "text_to_image", "black-forest-labs/FLUX.1-schnell",
                 "741f7c3ce8b383c54771c7003378a50191e9efe9"),
                ("FluxModularPipeline", "flux-krea:modular", "text_to_image", "black-forest-labs/FLUX.1-Krea-dev",
                 "8162a9c7b05a641be098422bf2fcf335615c2f28"),
                ("FluxFillPipeline", "flux-fill:direct", "inpaint", "black-forest-labs/FLUX.1-Fill-dev",
                 "358293da0354175698b67ec8299acf928313a78a"),
                ("Cosmos3OmniModularPipeline", "cosmos3-nano:official-modular-workflow", "text_to_image",
                 "nvidia/Cosmos3-Nano", "7a312c868bcce8e40b3eb40861300a9d0ba3fde1"),
                ("Cosmos3OmniModularPipeline", "cosmos3-super-text-to-image:official-modular-workflow", "text_to_image",
                 "nvidia/Cosmos3-Super-Text2Image", "daf3d374804be4c512c2135568a7cb95d4341d79"),
            ):
                with self.subTest(profile=identity):
                    starter_response = await server.resolve_operation_starter(SimpleNamespace(json=AsyncMock(return_value={
                        "pipelineClass": pipeline, "task": task, "executionProfileId": identity,
                    })))
                    self.assertEqual(starter_response.status, 200, starter_response.text)
                    starter = json.loads(starter_response.text)
                    loader = next(entry["node"] for entry in starter["nodes"]
                                  if entry["node"]["action"] in {"LoadPipeline", "ModelsLoader"})
                    field = "repo_id" if loader["action"] == "ModelsLoader" else "model_id"
                    self.assertEqual(loader["params"][field]["value"], {"source": "hub", "value": repository})
                    self.assertEqual(loader["params"]["revision"]["value"], revision)
                    self.assertEqual(profiles[identity].get("repository_revisions"), [{"repo": repository, "revision": revision}])
                    owners = [capability for capability in payload["capabilities"] + payload["experimentalCapabilities"]
                              if any(profile["id"] == identity for profile in capability["executionProfiles"])]
                    self.assertTrue(owners)
                    for owner in owners:
                        published = next(profile for profile in owner["executionProfiles"] if profile["id"] == identity)
                        self.assertEqual(published["repository_revisions"], profiles[identity]["repository_revisions"])

    def test_only_adds_image_profile_pairs_and_preserves_all_existing_metadata_and_registries(self):
        from test_image_adapter_authoring_policy import FLUX3_PUBLIC_POLICIES
        from test_audio_adapter_authoring_policy import EXPECTED as ACE_POLICY, IDENTITY as ACE
        profiles_before = deepcopy(DIFFUSERS_EXECUTION_PROFILES)
        capabilities_before = deepcopy(STUDIO_MODEL_CAPABILITIES)
        experimental_before = deepcopy(EXPERIMENTAL_DIFFUSERS_PIPELINES)
        server = WebServer(module_registry.MODULE_MAP)
        with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed executable node")):
            enriched = server._build_model_capabilities_payload()
            with (
                patch("modiff.model_artifact_catalog.catalog_profile_repository_revisions", return_value={}),
                patch.dict(DIFFUSERS_EXECUTION_PROFILES, {
                    identity: replace(profile, adapter_authoring=None)
                    for identity, profile in DIFFUSERS_EXECUTION_PROFILES.items()
                    if profile.adapter_authoring is not None
                }),
            ):
                unchanged = server._build_model_capabilities_payload()
        image_ids = set()
        non_image_ids = set()
        for capability in unchanged["capabilities"] + unchanged["experimentalCapabilities"]:
            for profile in capability["executionProfiles"]:
                outputs = capability.get("modeOutputKinds") or {}
                target = image_ids if any(outputs.get(mode, capability["mediaKind"]) == "image"
                                         for mode in profile["modes"]) else non_image_ids
                target.add(profile["id"])
        self.assertTrue(image_ids)
        self.assertTrue(non_image_ids)
        self.assertIn("cosmos3-nano:official-modular-workflow", image_ids)
        added = set()
        for profiles in [enriched["diffusersExecutionProfiles"], *[
            capability["executionProfiles"] for capability in enriched["capabilities"] + enriched["experimentalCapabilities"]
        ]]:
            for profile in profiles:
                policy = profile.pop("adapterAuthoring", None)
                if policy is not None:
                    conditional_controls = {
                        **{identity: policy['requiredControls'] for identity, policy in FLUX3_PUBLIC_POLICIES.items()},
                        "cosmos3-super-text-to-image:official-modular-workflow": {
                            "offload_mode": "none", "auto_offload": False, "weight_loading_mode": "cpu",
                            "quant_config": None, "unet": None, "vae": None, "controlnet": None, "trust_remote_code": False,
                        },
                        "flux-redux:direct": {
                            "offload_mode": "none", "auto_offload": False, "device_map": "none",
                            "quantization_mode": "none", "prequantized_transformer": None,
                        },
                    }
                    self.assertIn(profile["id"], {"flux-dev:modular", "flux-schnell:modular", "flux-krea:modular",
                                                "flux-kontext:modular", "flux2-klein:modular", *conditional_controls, ACE})
                    if profile['id'] in FLUX3_PUBLIC_POLICIES:
                        self.assertEqual(policy, FLUX3_PUBLIC_POLICIES[profile['id']])
                        self.assertTrue(profile['public'])
                        self.assertEqual(profile['required_diffusers_provider'], 'huggingface-diffusers-flux3-7ecb1e1')
                    if profile["id"] == ACE:
                        self.assertEqual(policy, ACE_POLICY)
                    if profile["id"] in conditional_controls:
                        self.assertEqual(policy["requiredControls"], conditional_controls[profile["id"]])
                        self.assertEqual(policy["mediaKind"], "image")
                        self.assertFalse(profile["live_proof"])
                    self.assertEqual(policy, DIFFUSERS_EXECUTION_PROFILES[profile["id"]].adapter_authoring.to_public_dict())
                pairs = profile.pop("repository_revisions", None)
                if pairs is not None:
                    self.assertIn(profile["id"], image_ids | {ACE})
                    if profile["id"] == ACE:
                        self.assertEqual(pairs, ACE_POLICY["repositoryRevisions"])
                    added.add(profile["id"])
                    declared = {profile.get("default_repo"), profile.get("fallback_repo"), *profile.get("compatible_repos", [])}
                    self.assertLessEqual(len(pairs), 32)
                    self.assertEqual(len({pair["repo"] for pair in pairs}), len(pairs))
                    for pair in pairs:
                        self.assertEqual(set(pair), {"repo", "revision"})
                        self.assertIn(pair["repo"], declared)
                        self.assertRegex(pair["revision"], r"^[0-9a-f]{40}$")
                elif profile["id"] in non_image_ids - image_ids:
                    self.assertNotIn("repository_revisions", profile)
        self.assertTrue(added)
        self.assertEqual(added & (non_image_ids - image_ids), {ACE})
        self.assertEqual(enriched, unchanged, "No display, execution, runtime or qualification metadata may change")
        by_model = {capability["modelType"]: capability for capability in enriched["capabilities"]}
        self.assertFalse(by_model["FluxModularPipeline"]["supportsLora"])
        self.assertFalse(by_model["Cosmos3OmniModularPipeline"]["supportsLora"])
        for flag in ("autoEligible", "templateEligible", "galleryEligible", "liveProof"):
            self.assertFalse(by_model["Cosmos3OmniModularPipeline"][flag])
        self.assertEqual(DIFFUSERS_EXECUTION_PROFILES, profiles_before)
        self.assertEqual(STUDIO_MODEL_CAPABILITIES, capabilities_before)
        self.assertEqual(EXPERIMENTAL_DIFFUSERS_PIPELINES, experimental_before)

    def test_query_and_repeated_publication_keep_exact_profile_pins(self):
        server = WebServer(module_registry.MODULE_MAP)
        full = server._build_model_capabilities_payload()
        filtered = server._build_model_capabilities_payload("schnell")
        self.assertEqual(filtered["diffusersExecutionProfiles"], full["diffusersExecutionProfiles"])
        selected = next(capability for capability in filtered["capabilities"] if capability["modelType"] == "FluxSchnellPipeline")
        self.assertEqual(selected["executionProfiles"], next(
            capability["executionProfiles"] for capability in full["capabilities"] if capability["modelType"] == "FluxSchnellPipeline"
        ))
        self.assertTrue(all(profile.get("repository_revisions") for profile in selected["executionProfiles"]))

    def test_declared_repository_spelling_matches_the_exact_selected_starter(self):
        from modiff.operation_catalog import build_operation_catalog
        from modiff.operation_starters import resolve_operation_starter

        identity = "flux-schnell:direct"
        profile = replace(DIFFUSERS_EXECUTION_PROFILES[identity], default_repo="black-forest-labs/flux.1-schnell")
        with (
            patch.dict(DIFFUSERS_EXECUTION_PROFILES, {identity: profile}),
            patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed executable node")),
        ):
            published = artifact_catalog.catalog_profile_repository_revisions([profile.to_public_dict()])[identity]
            contracts, _ = build_operation_catalog(module_registry.MODULE_MAP, [], catalog_resolver=lambda: {})
            selected = resolve_operation_starter(module_registry.MODULE_MAP, contracts, {
                "pipelineClass": profile.pipeline_class, "task": "text_to_image", "executionProfileId": identity,
            })
        loader = selected["nodes"][0]
        self.assertEqual(published, [{
            "repo": loader["params"]["model_id"]["value"]["value"],
            "revision": loader["params"]["revision"]["value"],
        }])
        self.assertEqual(published[0]["repo"], profile.default_repo)


class CatalogProfileRevisionTests(unittest.TestCase):
    def test_batch_lookup_reads_once_and_preserves_alias_priority_and_declared_repo_spelling(self):
        catalog = {"models": [
            {"modelType": "OtherAlias", "baseRepo": "Publisher/Shared", "baseRevision": "a" * 40, "artifacts": []},
            {"modelType": "SelectedAlias", "baseRepo": "Publisher/Shared", "baseRevision": "B" * 40,
             "artifacts": [{"repo": "Publisher/Compatible", "revision": "c" * 40},
                           {"repo": "Publisher/Undeclared", "revision": "d" * 40}]},
        ], "repositoryPins": [{"repo": "Publisher/Fallback", "revision": "e" * 40}]}
        original = deepcopy(catalog)
        profiles = [{"id": "selected", "model_type": "SelectedAlias", "default_repo": "publisher/shared",
                     "fallback_repo": "Publisher/Fallback", "compatible_repos": ["Publisher/Compatible", "Publisher/SHARED", "Missing/Repo"]},
                    {"id": "other", "model_type": "OtherAlias", "default_repo": "Publisher/Shared"},
                    {"id": "unknown", "default_repo": "Missing/Repo"}]
        with patch.object(artifact_catalog, "read_model_artifact_catalog", return_value=catalog) as read:
            result = artifact_catalog.catalog_profile_repository_revisions(profiles)
        read.assert_called_once_with()
        self.assertEqual(result, {
            "selected": [{"repo": "publisher/shared", "revision": "b" * 40},
                         {"repo": "Publisher/Fallback", "revision": "e" * 40},
                         {"repo": "Publisher/Compatible", "revision": "c" * 40}],
            "other": [{"repo": "Publisher/Shared", "revision": "a" * 40}],
        })
        self.assertEqual(catalog, original)
        with patch.object(artifact_catalog, "read_model_artifact_catalog", return_value=catalog):
            self.assertEqual(artifact_catalog.catalog_repository_pin("publisher/shared", model_type="SelectedAlias")["repo"],
                             "Publisher/Shared", "Individual catalog lookup retains its canonical spelling contract")

    def test_invalid_catalog_revision_preserves_existing_public_lookup_error(self):
        for revision in (None, "main", "a" * 39, "g" * 40):
            with self.subTest(revision=revision), patch.object(artifact_catalog, "read_model_artifact_catalog", return_value={
                "models": [{"modelType": "Alias", "baseRepo": "Publisher/Model", "baseRevision": revision, "artifacts": []}],
            }):
                with self.assertRaises(ValueError) as existing:
                    artifact_catalog.catalog_revision("Publisher/Model", model_type="Alias")
                with self.assertRaises(ValueError) as published:
                    artifact_catalog.catalog_profile_repository_revisions([
                        {"id": "alias", "model_type": "Alias", "default_repo": "Publisher/Model"},
                    ])
                self.assertEqual(str(published.exception), str(existing.exception))

    def test_empty_public_lookup_keeps_its_no_read_behavior(self):
        with patch.object(artifact_catalog, "read_model_artifact_catalog", side_effect=AssertionError("Unnecessary read")):
            self.assertIsNone(artifact_catalog.catalog_repository_pin(" "))
            self.assertIsNone(artifact_catalog.catalog_revision(""))
            self.assertEqual(artifact_catalog.catalog_profile_repository_revisions([]), {})

    def test_declared_repository_pair_count_is_bounded_without_truncation(self):
        pins = [{"repo": f"Publisher/Variant-{index}", "revision": "a" * 40} for index in range(33)]
        with patch.object(artifact_catalog, "read_model_artifact_catalog", return_value={"models": [], "repositoryPins": pins}):
            selected = {"id": "bounded", "compatible_repos": [pin["repo"] for pin in pins[:32]]}
            self.assertEqual(len(artifact_catalog.catalog_profile_repository_revisions([selected])["bounded"]), 32)
            selected["compatible_repos"].append(pins[-1]["repo"])
            with self.assertRaisesRegex(ValueError, "more than 32 reviewed repository revisions"):
                artifact_catalog.catalog_profile_repository_revisions([selected])

    def test_batch_helper_imports_no_model_runtime(self):
        command = "\n".join([
            "import sys",
            "from modiff.model_artifact_catalog import catalog_profile_repository_revisions",
            "assert catalog_profile_repository_revisions([]) == {}",
            "assert not {'torch', 'diffusers', 'transformers', 'peft', 'modules'} & set(sys.modules)",
        ])
        result = subprocess.run([sys.executable, "-c", command], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
