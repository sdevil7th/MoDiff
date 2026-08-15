import builtins
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from modiff.auto_resource import build_auto_resource_plan
from modiff.diffusers_profiles import (
    DIFFUSERS_EXECUTION_PROFILES,
    OPTIONAL_RUNTIME_DELIVERY_BASE,
    optional_runtime_profile_ids_for_execution,
    public_execution_profiles,
)
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
    optional_runtime_requirements,
    public_optional_runtime_profiles,
)
from modiff.server import WebServer


GIB = 1024**3
OPTIONAL_STAGE_IMPORTS = {
    "transformers",
    "peft",
    "tokenizers",
    "typer",
    "annotated_doc",
    "rich",
    "markdown_it",
    "mdurl",
    "pygments",
    "shellingham",
}


def _version_resolver(versions):
    def resolve(distribution):
        if distribution not in versions:
            raise metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    return resolve


def _walk_graph_files(items):
    for item in items:
        if item.get("isDir"):
            yield from _walk_graph_files(item.get("children") or [])
        else:
            yield item


def _cpu_hardware():
    return {
        "runtimeFingerprint": "optional-runtime-test",
        "platform": "linux",
        "architecture": "x86_64",
        "accelerator": {
            "kind": "cpu",
            "name": "Mock CPU",
            "totalBytes": 0,
            "freeBytes": 0,
            "capability": None,
            "band": "cpu",
        },
        "systemMemory": {
            "totalBytes": 32 * GIB,
            "availableBytes": 28 * GIB,
            "pageFileTotalBytes": None,
            "pageFileAvailableBytes": None,
        },
        "offloadDisk": {
            "path": "unit-test",
            "totalBytes": 256 * GIB,
            "freeBytes": 128 * GIB,
        },
    }


class OptionalRuntimeContractTests(unittest.TestCase):
    def test_composite_contract_is_exact_hashed_and_platform_qualified(self):
        versions = {"transformers": "5.14.1", "peft": "0.20.0"}
        profile = public_optional_runtime_profiles(
            [TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID],
            version_resolver=_version_resolver(versions),
        )[0]

        self.assertEqual(profile["schemaVersion"], 1)
        self.assertEqual(profile["contractState"], "qualified")
        self.assertTrue(profile["cutoverReady"])
        self.assertTrue(profile["installActionAvailable"])
        self.assertTrue(profile["activationAvailable"])
        self.assertEqual((profile["platform"], profile["machine"]), ("linux", "x86_64"))
        self.assertEqual(len(profile["targetContracts"]), 6)
        self.assertEqual(profile["installPolicy"], "explicit_first_use")
        self.assertEqual(profile["status"], "present_unqualified")
        self.assertEqual(
            profile["requirements"],
            ["transformers==5.14.1", "peft==0.20.0"],
        )
        self.assertEqual(len(profile["artifactLocks"]), 60)
        canonical_spec = json.dumps(
            OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID].to_spec_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            profile["specDigest"],
            f"sha256:{hashlib.sha256(canonical_spec).hexdigest()}",
        )
        self.assertTrue(all(package["publisher"] == "Hugging Face" for package in profile["packages"]))
        self.assertEqual(
            [package["projectUrl"] for package in profile["packages"]],
            [
                "https://github.com/huggingface/transformers",
                "https://github.com/huggingface/peft",
            ],
        )
        self.assertTrue(all(package["license"] == "Apache-2.0" for package in profile["packages"]))
        self.assertTrue(
            all(package["distributionUrl"].startswith("https://pypi.org/project/") for package in profile["packages"])
        )
        self.assertTrue(all(package["status"] == "present_unqualified" for package in profile["packages"]))
        self.assertIn("StableAudioPipeline", profile["requiredDiffusersSymbols"])
        self.assertIn("ErnieImagePipeline", profile["requiredDiffusersSymbols"])
        self.assertIn("GlmImagePipeline", profile["requiredDiffusersSymbols"])
        self.assertTrue(
            {
                "StableDiffusionControlNetPAGPipeline",
                "StableDiffusionXLControlNetPAGPipeline",
                "StableDiffusionControlNetImg2ImgPipeline",
                "StableDiffusionXLControlNetImg2ImgPipeline",
                "StableDiffusionXLControlNetPAGImg2ImgPipeline",
                "FluxControlImg2ImgPipeline",
                "StableDiffusionControlNetInpaintPipeline",
                "StableDiffusionControlNetPAGInpaintPipeline",
                "StableDiffusionXLControlNetInpaintPipeline",
                "FluxControlInpaintPipeline",
                "QwenImageControlNetModel",
                "QwenImageControlNetPipeline",
                "QwenImageLayeredPipeline",
                "AnimateDiffPAGPipeline",
                "AnimateDiffVideoToVideoPipeline",
                "AnimateDiffControlNetPipeline",
                "AnimateDiffVideoToVideoControlNetPipeline",
                "CogVideoXVideoToVideoPipeline",
            }.issubset(profile["requiredDiffusersSymbols"])
        )
        self.assertTrue(
            {"QwenImageControlNetPipeline", "QwenImageLayeredPipeline"}.issubset(profile["pipelineAdapterSymbols"])
        )
        self.assertNotIn("StableAudioPipeline", profile["pipelineAdapterSymbols"])
        self.assertNotIn("GlmImagePipeline", profile["pipelineAdapterSymbols"])

        different_host = public_optional_runtime_profiles(
            [TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID],
            version_resolver=_version_resolver({"transformers": "0.0.0"}),
        )[0]
        self.assertEqual(different_host["specDigest"], profile["specDigest"])

        macos = public_optional_runtime_profiles(
            [TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID],
            version_resolver=_version_resolver(versions),
            platform_name="macos",
            machine="arm64",
        )[0]
        self.assertEqual(macos["contractState"], "candidate_unqualified")
        self.assertFalse(macos["cutoverReady"])
        self.assertFalse(macos["installActionAvailable"])
        self.assertFalse(macos["activationAvailable"])
        self.assertEqual(macos["specDigest"], profile["specDigest"])

    def test_status_distinguishes_missing_wrong_version_and_present_unqualified(self):
        missing_and_wrong = public_optional_runtime_profiles(
            [TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID],
            version_resolver=_version_resolver({"transformers": "4.49.0"}),
        )[0]
        self.assertEqual(missing_and_wrong["status"], "missing")
        package_status = {package["distribution"]: package["status"] for package in missing_and_wrong["packages"]}
        self.assertEqual(
            package_status,
            {"transformers": "wrong_version", "peft": "missing"},
        )

        wrong_version = public_optional_runtime_profiles(
            [TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID],
            version_resolver=_version_resolver({"transformers": "5.14.1", "peft": "0.19.0"}),
        )[0]
        self.assertEqual(wrong_version["status"], "wrong_version")

        present = public_optional_runtime_profiles(
            [TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID],
            version_resolver=_version_resolver({"transformers": "5.14.1", "peft": "0.20.0"}),
        )[0]
        self.assertEqual(present["status"], "present_unqualified")

    def test_observed_versions_are_bounded_before_publication(self):
        hostile_version = "9." + ("x" * 500) + "\nprivate-path"
        profile = public_optional_runtime_profiles(
            [TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID],
            version_resolver=_version_resolver({"transformers": hostile_version, "peft": "0.20.0"}),
        )[0]
        transformers = next(package for package in profile["packages"] if package["distribution"] == "transformers")
        self.assertEqual(transformers["status"], "wrong_version")
        self.assertLessEqual(len(transformers["installedVersion"]), 128)
        self.assertNotIn("\n", transformers["installedVersion"])
        self.assertNotIn("private-path", transformers["installedVersion"])

    def test_unreadable_distribution_metadata_fails_closed_without_echoing_error(self):
        def unreadable_metadata(distribution):
            if distribution == "transformers":
                raise ValueError("private-path/invalid.dist-info")
            return "0.20.0"

        profile = public_optional_runtime_profiles(
            [TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID],
            version_resolver=unreadable_metadata,
        )[0]
        transformers = next(package for package in profile["packages"] if package["distribution"] == "transformers")
        self.assertEqual(profile["status"], "wrong_version")
        self.assertEqual(transformers["status"], "wrong_version")
        self.assertEqual(transformers["metadataState"], "unreadable")
        self.assertNotIn("private-path", json.dumps(profile))

    def test_unexpected_metadata_assertion_propagates_to_purity_trap(self):
        def forbidden_import_sentinel(_distribution):
            raise AssertionError("optional package import attempted")

        with self.assertRaisesRegex(AssertionError, "optional package import attempted"):
            public_optional_runtime_profiles(
                [TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID],
                version_resolver=forbidden_import_sentinel,
            )

    def test_unknown_profile_id_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unknown optional runtime profile"):
            public_optional_runtime_profiles(["unknown-runtime"])
        with self.assertRaisesRegex(ValueError, "Unknown optional runtime profile"):
            optional_runtime_requirements(["unknown-runtime"])

    def test_execution_profiles_reference_the_central_composite(self):
        expected = (TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,)
        self.assertTrue(DIFFUSERS_EXECUTION_PROFILES)
        for profile in DIFFUSERS_EXECUTION_PROFILES.values():
            with self.subTest(profile=profile.id):
                if profile.optional_runtime_delivery == OPTIONAL_RUNTIME_DELIVERY_BASE:
                    self.assertIn(
                        profile.id,
                        {
                            "builtin-image-operations:direct",
                            "builtin-video-operations:direct",
                            "real-esrgan-x2-video-upscale:direct",
                        },
                    )
                    self.assertEqual(profile.optional_runtime_profiles, ())
                    for mode in profile.modes:
                        self.assertEqual(
                            optional_runtime_profile_ids_for_execution(
                                profile.model_type,
                                mode,
                                platform_name="linux",
                                machine="x86_64",
                            ),
                            (),
                        )
                        self.assertEqual(
                            optional_runtime_profile_ids_for_execution(
                                profile.model_type,
                                mode,
                                platform_name="windows",
                                machine="AMD64",
                            ),
                            (),
                        )
                    continue
                self.assertEqual(profile.optional_runtime_profiles, expected)
                for mode in profile.modes:
                    self.assertIn(
                        TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID,
                        optional_runtime_profile_ids_for_execution(
                            profile.model_type,
                            mode,
                            platform_name="linux",
                            machine="x86_64",
                        ),
                    )
                    self.assertEqual(
                        optional_runtime_profile_ids_for_execution(
                            profile.model_type,
                            mode,
                            platform_name="windows",
                            machine="AMD64",
                        ),
                        (TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,),
                    )

        host_profile_id = optional_runtime_profile_ids_for_execution(
            "ZImageModularPipeline",
            "text_to_image",
        )[0]
        public_profile = public_execution_profiles()[0]
        self.assertEqual(
            public_profile["optional_runtime_profiles"],
            [host_profile_id],
        )
        self.assertNotIn("optionalRuntimeProfiles", public_profile)

    def test_optional_metadata_state_does_not_change_auto_readiness(self):
        host_profile_id = optional_runtime_profile_ids_for_execution(
            "ZImageModularPipeline",
            "text_to_image",
        )[0]
        transformers_version = OPTIONAL_RUNTIME_PROFILES[host_profile_id].packages[0].version
        observations = {
            "present_unqualified": _version_resolver({"transformers": transformers_version, "peft": "0.20.0"}),
            "wrong_version": _version_resolver({"transformers": transformers_version, "peft": "0.19.0"}),
            "missing": _version_resolver({}),
        }
        outcomes = {}
        with tempfile.TemporaryDirectory() as temp_dir:
            for expected_status, resolver in observations.items():
                with patch("modiff.optional_runtimes.metadata.version", side_effect=resolver):
                    plan = build_auto_resource_plan(
                        {
                            "form": {
                                "modelType": "ZImageModularPipeline",
                                "mode": "text_to_image",
                            },
                            "hardwareOverride": _cpu_hardware(),
                        },
                        runtime_fingerprint=None,
                        local_models=["Tongyi-MAI/Z-Image-Turbo"],
                        data_dir=temp_dir,
                    )
                self.assertEqual(
                    plan["optionalRuntimeProfiles"][0]["status"],
                    expected_status,
                )
                outcomes[expected_status] = (
                    plan["status"],
                    plan["readiness"],
                    plan["canAutoRun"],
                    plan["selectedCandidate"]["id"],
                )

        self.assertEqual(len(set(outcomes.values())), 1)
        self.assertEqual(
            outcomes["missing"][:3],
            ("ready", "ready", True),
        )

    def test_cold_clean_base_registry_discovery_does_not_load_optional_packages(self):
        script = r"""
import builtins
import importlib.util
import sys

original_import = builtins.__import__
original_find_spec = importlib.util.find_spec
attempts = []

def clean_base_find_spec(name, *args, **kwargs):
    if name.split(".", 1)[0] in __OPTIONAL_IMPORTS__:
        return None
    return original_find_spec(name, *args, **kwargs)

def guarded_import(name, *args, **kwargs):
    if name.split(".", 1)[0] in __OPTIONAL_IMPORTS__:
        attempts.append(name)
        raise ImportError(f"forbidden optional import: {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
importlib.util.find_spec = clean_base_find_spec
import modules
assert modules.MODULE_MAP
assert not any(
    name.split(".", 1)[0] in __OPTIONAL_IMPORTS__
    for name in sys.modules
), attempts
""".replace("__OPTIONAL_IMPORTS__", repr(OPTIONAL_STAGE_IMPORTS))
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "DIFFUSERS_OFFLINE": "1",
            }
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class OptionalRuntimePublicationTests(unittest.IsolatedAsyncioTestCase):
    async def test_auto_capabilities_and_listgraphs_are_non_installing_and_import_free(self):
        host_profile_id = optional_runtime_profile_ids_for_execution(
            "ZImageModularPipeline",
            "text_to_image",
        )[0]
        host_target = OPTIONAL_RUNTIME_PROFILES[host_profile_id].contract_for_target()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            graph_dir = data_dir / "graphs" / "studio"
            graph_dir.mkdir(parents=True)
            (graph_dir / "sample.json").write_text("{}", encoding="utf-8")
            (data_dir / "workflow-library-manifest.json").write_text(
                json.dumps(
                    {
                        "workflows": [
                            {
                                "graphPath": "studio/sample.json",
                                "modelType": "ZImageModularPipeline",
                                "mode": "text_to_image",
                                "mediaKind": "image",
                                "supportTier": "supported",
                                "qualificationStatus": "graph-qualified",
                                "requiredArtifacts": ["Tongyi-MAI/Z-Image-Turbo"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            server = WebServer({}, work_dir=temp_dir, data_dir=temp_dir)

            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name.split(".", 1)[0] in OPTIONAL_STAGE_IMPORTS:
                    raise AssertionError(f"read-only metadata path imported {name}")
                return original_import(name, *args, **kwargs)

            def forbidden_install(*_args, **_kwargs):
                raise AssertionError("read-only metadata path called an installer")

            with (
                patch("builtins.__import__", side_effect=guarded_import),
                patch(
                    "modiff.optimization_packages.install_capability",
                    side_effect=forbidden_install,
                ),
                patch(
                    "modiff.server.install_optimization_capability",
                    side_effect=forbidden_install,
                ),
                patch(
                    "modiff.server.install_optional_runtime",
                    side_effect=forbidden_install,
                ),
                patch(
                    "modiff.server.activate_optional_runtime_environment",
                    side_effect=forbidden_install,
                ),
                patch(
                    "modiff.server.rollback_optional_runtime_environment",
                    side_effect=forbidden_install,
                ),
                patch(
                    "modiff.server.reserve_runtime_install",
                    side_effect=forbidden_install,
                ),
                patch(
                    "modiff.runtime_overlays.reserve_install",
                    side_effect=forbidden_install,
                ),
                patch(
                    "modiff.runtime_overlays.cache_locked_artifacts",
                    side_effect=forbidden_install,
                ),
                patch(
                    "modiff.optimization_packages._atomic_json",
                    side_effect=forbidden_install,
                ),
                patch("pathlib.Path.mkdir", side_effect=forbidden_install),
                patch("pathlib.Path.write_text", side_effect=forbidden_install),
                patch("modiff.server.validate_studio_execution_specs", return_value=[]),
                patch("urllib.request.urlopen", side_effect=forbidden_install),
                patch("subprocess.Popen", side_effect=forbidden_install),
                patch("subprocess.run", side_effect=forbidden_install),
            ):
                plan = build_auto_resource_plan(
                    {
                        "form": {
                            "modelType": "ZImageModularPipeline",
                            "mode": "text_to_image",
                        },
                        "hardwareOverride": _cpu_hardware(),
                    },
                    runtime_fingerprint=None,
                    local_models=[],
                    data_dir=temp_dir,
                )
                capabilities_response = await server.model_capabilities(type("Request", (), {"query": {}})())
                listgraphs_response = await server.listgraphs(object())
                template_open_response = await server.fileGet(
                    type(
                        "Request",
                        (),
                        {"query": {"file": "graphs/studio/sample.json"}},
                    )()
                )
                optional_runtime_response = await server.runtime_optional_runtimes(object())

            self.assertEqual(
                plan["optionalRuntimeProfileIds"],
                [host_profile_id],
            )
            self.assertEqual(
                plan["candidates"][0]["optionalRuntimeProfileIds"],
                [host_profile_id],
            )
            # This metadata-only slice must not affect the existing Auto result.
            self.assertEqual(plan["canAutoRun"], bool(plan["selectedCandidate"]))
            self.assertEqual(
                plan["optionalRuntimeProfiles"][0]["contractState"],
                host_target.contract_state,
            )
            self.assertEqual(
                plan["optionalRuntimeProfiles"][0]["cutoverReady"],
                host_target.cutover_ready,
            )
            self.assertEqual(template_open_response.status, 200)
            optional_runtime_catalog = json.loads(optional_runtime_response.text)
            selected_catalog_profile = next(
                profile for profile in optional_runtime_catalog["profiles"] if profile["id"] == host_profile_id
            )
            self.assertEqual(
                selected_catalog_profile["contractState"],
                host_target.contract_state,
            )
            self.assertEqual(
                selected_catalog_profile["cutoverReady"],
                host_target.cutover_ready,
            )

            capabilities = json.loads(capabilities_response.text)
            self.assertEqual(
                {profile["id"] for profile in capabilities["optionalRuntimeProfiles"]},
                {
                    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
                    TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID,
                },
            )
            z_image = next(
                capability
                for capability in capabilities["capabilities"]
                if capability["modelType"] == "ZImageModularPipeline"
            )
            self.assertEqual(
                z_image["optionalRuntimeProfileIds"],
                [host_profile_id],
            )

            graph_tree = json.loads(listgraphs_response.text)
            graph_file = next(_walk_graph_files(graph_tree))
            self.assertEqual(
                graph_file["optionalRuntimeProfileIds"],
                [host_profile_id],
            )
            self.assertEqual(
                graph_file["optionalRuntimeProfiles"][0]["contractState"],
                host_target.contract_state,
            )


if __name__ == "__main__":
    unittest.main()
