from dataclasses import replace
import unittest

from modiff.diffusers_profiles import (
    ACE_STEP_LORA_BASE_REPO,
    DIFFUSERS_EXECUTION_PROFILES,
    ExpertCudaPolicy,
    ExpertMpsPolicy,
    ExpertQuantizationPolicy,
    MPS_EXPERIMENTAL_POLICY,
    MPS_UNQUALIFIED_POLICY,
    MPS_UNQUALIFIED_WITH_Z_IMAGE_FALLBACK_POLICY,
    QWEN_EXPERT_CUDA_POLICY,
    QWEN_EXPERT_QUANTIZATION_POLICY,
)
from modiff.model_artifact_catalog import catalog_revision
from modules.DiffusersAudio.main import AUDIO_PIPELINE_ADAPTERS
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS
from modules.DiffusersVideo.main import VIDEO_PIPELINE_ADAPTERS


class DiffusersExecutionProfileTests(unittest.TestCase):
    def test_every_profile_declares_one_explicit_loader_and_execution_path(self):
        expected_targets = {
            "modular-diffusers": ("modules.ModularDiffusers", "ModelsLoader"),
            "direct-diffusers-image": ("modules.DiffusersImage", "LoadPipeline"),
            "direct-diffusers-video": ("modules.DiffusersVideo", "LoadPipeline"),
            "direct-wan-vace": ("modules.DiffusersVideo", "LoadPipeline"),
            "direct-diffusers-audio": ("modules.DiffusersAudio", "LoadPipeline"),
        }

        for profile in DIFFUSERS_EXECUTION_PROFILES.values():
            with self.subTest(profile=profile.id):
                self.assertEqual(
                    (profile.loader_module, profile.loader_action),
                    expected_targets[profile.execution_path],
                )
                self.assertEqual(
                    profile.backend_path,
                    f"{profile.loader_module}.{profile.loader_action}",
                )
                public = profile.to_public_dict()
                self.assertEqual(public["loader_module"], profile.loader_module)
                self.assertEqual(public["loader_action"], profile.loader_action)
                self.assertEqual(public["execution_path"], profile.execution_path)
                self.assertEqual(public["backend_path"], profile.backend_path)

    def test_every_supported_studio_model_has_an_execution_profile(self):
        expected = {
            "ZImageModularPipeline",
            "QwenImageModularPipeline",
            "QwenImageEditModularPipeline",
            "QwenImageEditPlusModularPipeline",
            "QwenImageLayeredModularPipeline",
            "WanVACEPipeline",
            "WanVideoPipeline",
            "WanImageToVideoPipeline",
            "WanTI2VPipeline",
            "LTXVideoPipeline",
            "AceStepAudioPipeline",
            "FluxSchnellPipeline",
            "FluxDevPipeline",
            "FluxKreaPipeline",
            "FluxKontextPipeline",
            "FluxFillPipeline",
            "FluxDepthPipeline",
            "FluxCannyPipeline",
            "FluxReduxPipeline",
            "Flux2KleinPipeline",
        }
        actual = {profile.model_type for profile in DIFFUSERS_EXECUTION_PROFILES.values()}
        self.assertEqual(expected, actual)

    def test_video_profile_uses_generic_facade(self):
        profile = DIFFUSERS_EXECUTION_PROFILES["wan-vace:direct"]
        self.assertEqual(profile.backend_path, "modules.DiffusersVideo.LoadPipeline")

        wan_video_profile = DIFFUSERS_EXECUTION_PROFILES["wan-video-to-video:direct"]
        self.assertEqual(wan_video_profile.backend_path, "modules.DiffusersVideo.LoadPipeline")
        self.assertEqual(wan_video_profile.pipeline_class, "WanVideoToVideoPipeline")

        wan_text_profile = DIFFUSERS_EXECUTION_PROFILES["wan-text-to-video:direct"]
        self.assertEqual(wan_text_profile.backend_path, "modules.DiffusersVideo.LoadPipeline")
        self.assertEqual(wan_text_profile.pipeline_class, "WanPipeline")

        ltx_profile = DIFFUSERS_EXECUTION_PROFILES["ltx-video:direct"]
        self.assertEqual(ltx_profile.backend_path, "modules.DiffusersVideo.LoadPipeline")
        self.assertEqual(ltx_profile.pipeline_class, "LTXConditionPipeline")

    def test_z_image_auto_profile_uses_the_registered_direct_image_facade(self):
        profile = DIFFUSERS_EXECUTION_PROFILES["z-image:auto"]

        self.assertEqual(profile.model_type, "ZImageModularPipeline")
        self.assertEqual(profile.backend_path, "modules.DiffusersImage.LoadPipeline")
        self.assertEqual(profile.execution_path, "direct-diffusers-image")
        self.assertEqual(profile.pipeline_class, "ZImagePipeline")

    def test_qwen_profiles_publish_reviewed_expert_resource_policies(self):
        qwen_profile_ids = {
            "qwen-image:t2i-direct",
            "qwen-image:modular",
            "qwen-edit:direct-inpaint",
            "qwen-edit:modular",
            "qwen-edit-plus:modular",
            "qwen-layered:modular",
        }

        for profile_id, profile in DIFFUSERS_EXECUTION_PROFILES.items():
            with self.subTest(profile=profile_id):
                self.assertIs(
                    profile.expert_cuda_policy,
                    QWEN_EXPERT_CUDA_POLICY if profile_id in qwen_profile_ids else None,
                )
                self.assertIs(
                    profile.expert_quantization_policy,
                    QWEN_EXPERT_QUANTIZATION_POLICY if profile_id in qwen_profile_ids else None,
                )
                self.assertEqual(
                    "expert_cuda_policy" in profile.to_public_dict(),
                    profile_id in qwen_profile_ids,
                )
                self.assertEqual(
                    "expert_quantization_policy" in profile.to_public_dict(),
                    profile_id in qwen_profile_ids,
                )

        public = DIFFUSERS_EXECUTION_PROFILES["qwen-image:t2i-direct"].to_public_dict()
        self.assertEqual(
            public["expert_cuda_policy"],
            {
                "schema_version": 1,
                "blocked_dtypes": ["float32"],
                "recommended_dtype": "bfloat16",
                "offloaded_vram_bytes": 10 * 1024**3,
                "resident_vram_bytes": 80 * 1024**3,
                "quantized_resident_vram_bytes": [["bnb_4bit", 24 * 1024**3]],
            },
        )
        self.assertEqual(
            public["expert_quantization_policy"],
            {
                "schema_version": 1,
                "quantization_mode": "bnb_4bit",
                "offload_mode": "model_cpu",
                "modular_node": "modules.ModularDiffusers.QuantizationConfigNode",
                "subfolder": "transformer",
                "component": "qwen_low_vram",
                "four_bit_quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "double_quant": True,
            },
        )
        self.assertEqual(public["expert_quantization_modes"], ["bnb_4bit"])
        self.assertEqual(
            public["expert_mps_policy"],
            {
                "schema_version": 1,
                "qualification": "unqualified",
                "fallback_action": "switch_to_z_image",
            },
        )

    def test_profiles_publish_only_the_reviewed_expert_mps_policies(self):
        unqualified = {
            "qwen-image:modular",
            "qwen-edit:direct-inpaint",
            "qwen-edit:modular",
            "qwen-edit-plus:modular",
            "qwen-layered:modular",
            "wan-vace:direct",
            "wan-22-image-to-video:direct",
            "wan-22-ti2v-5b:direct",
            "wan-text-to-video:direct",
            "wan-video-to-video:direct",
            "ltx-video:direct",
        }
        for profile_id, profile in DIFFUSERS_EXECUTION_PROFILES.items():
            with self.subTest(profile=profile_id):
                expected = (
                    MPS_UNQUALIFIED_WITH_Z_IMAGE_FALLBACK_POLICY
                    if profile_id == "qwen-image:t2i-direct"
                    else MPS_EXPERIMENTAL_POLICY
                    if profile_id == "z-image:auto"
                    else MPS_UNQUALIFIED_POLICY
                    if profile_id in unqualified
                    else None
                )
                self.assertIs(profile.expert_mps_policy, expected)
                self.assertEqual("expert_mps_policy" in profile.to_public_dict(), expected is not None)

    def test_only_reviewed_image_profiles_publish_expert_quantization_choices(self):
        flux_modes = ("bnb_4bit", "bnb_8bit", "quanto_float8", "torchao_float8")
        qwen_ids = {
            "qwen-image:t2i-direct",
            "qwen-image:modular",
            "qwen-edit:direct-inpaint",
            "qwen-edit:modular",
            "qwen-edit-plus:modular",
            "qwen-layered:modular",
        }
        for profile_id, profile in DIFFUSERS_EXECUTION_PROFILES.items():
            expected = ("bnb_4bit",) if profile_id in qwen_ids else flux_modes if profile.model_type.startswith("Flux") else ()
            with self.subTest(profile=profile_id):
                self.assertEqual(profile.expert_quantization_modes, expected)
                self.assertEqual(
                    profile.to_public_dict().get("expert_quantization_modes"),
                    list(expected) if expected else None,
                )

    def test_execution_profile_rejects_unreviewed_expert_quantization_choices(self):
        profile = DIFFUSERS_EXECUTION_PROFILES["z-image:auto"]
        for modes in (("unknown",), ("bnb_4bit", "bnb_4bit")):
            with self.subTest(modes=modes), self.assertRaisesRegex(ValueError, "invalid Expert quantization modes"):
                replace(profile, expert_quantization_modes=modes)

    def test_expert_cuda_policy_rejects_unreviewed_or_unbounded_values(self):
        cases = (
            {"schema_version": 2},
            {"blocked_dtypes": ("float32", "float32")},
            {"blocked_dtypes": ("unknown",)},
            {"recommended_dtype": "float32"},
            {"offloaded_vram_bytes": 0},
            {"resident_vram_bytes": 1025 * 1024**3},
            {"quantized_resident_vram_bytes": (("unknown", 24 * 1024**3),)},
        )
        values = {
            "schema_version": 1,
            "blocked_dtypes": ("float32",),
            "recommended_dtype": "bfloat16",
            "offloaded_vram_bytes": 10 * 1024**3,
            "resident_vram_bytes": 80 * 1024**3,
            "quantized_resident_vram_bytes": (("bnb_4bit", 24 * 1024**3),),
        }

        for update in cases:
            with self.subTest(update=update):
                with self.assertRaisesRegex(ValueError, "Invalid reviewed Expert CUDA policy"):
                    ExpertCudaPolicy(**{**values, **update})

    def test_expert_quantization_policy_rejects_unreviewed_values(self):
        cases = (
            {"schema_version": 2},
            {"quantization_mode": "bnb_8bit"},
            {"offload_mode": "none"},
            {"modular_node": "../../unsafe"},
            {"subfolder": "../transformer"},
            {"component": "qwen/unsafe"},
            {"four_bit_quant_type": "int4"},
            {"compute_dtype": "float64"},
            {"double_quant": 1},
        )
        values = {
            "schema_version": 1,
            "quantization_mode": "bnb_4bit",
            "offload_mode": "model_cpu",
            "modular_node": "modules.ModularDiffusers.QuantizationConfigNode",
            "subfolder": "transformer",
            "component": "qwen_low_vram",
            "four_bit_quant_type": "nf4",
            "compute_dtype": "bfloat16",
            "double_quant": True,
        }

        for update in cases:
            with self.subTest(update=update):
                with self.assertRaisesRegex(ValueError, "Invalid reviewed Expert quantization policy"):
                    ExpertQuantizationPolicy(**{**values, **update})

    def test_expert_mps_policy_rejects_unreviewed_values(self):
        values = {"schema_version": 1, "qualification": "unqualified", "fallback_action": "open_setup"}
        for update in (
            {"schema_version": 2},
            {"qualification": "certified"},
            {"fallback_action": "run_anyway"},
        ):
            with self.subTest(update=update):
                with self.assertRaisesRegex(ValueError, "Invalid reviewed Expert MPS policy"):
                    ExpertMpsPolicy(**{**values, **update})

    def test_ace_lora_template_base_is_an_exact_reviewed_compatible_artifact(self):
        profile = DIFFUSERS_EXECUTION_PROFILES["ace-step-audio:direct"]

        self.assertEqual(profile.compatible_repos, (ACE_STEP_LORA_BASE_REPO,))
        self.assertEqual(
            catalog_revision(ACE_STEP_LORA_BASE_REPO),
            "be23effe449c5957947f3020fd63bee23c64abe4",
        )

    def test_every_direct_profile_is_accepted_by_its_registered_adapter(self):
        """Profiles may be a supported subset; Expert-only adapters need no Auto profile."""

        registries = {
            "modules.DiffusersImage.LoadPipeline": IMAGE_PIPELINE_ADAPTERS,
            "modules.DiffusersVideo.LoadPipeline": VIDEO_PIPELINE_ADAPTERS,
            "modules.DiffusersAudio.LoadPipeline": AUDIO_PIPELINE_ADAPTERS,
        }
        direct_profiles = [
            profile
            for profile in DIFFUSERS_EXECUTION_PROFILES.values()
            if profile.backend_path in registries
        ]
        self.assertTrue(direct_profiles)

        for profile in direct_profiles:
            with self.subTest(profile=profile.id):
                adapter = registries[profile.backend_path].get(profile.pipeline_class)
                self.assertIsNotNone(
                    adapter,
                    f"{profile.id} names unregistered adapter {profile.pipeline_class}",
                )
                self.assertTrue(
                    set(profile.modes).issubset(adapter.modes),
                    f"{profile.id} modes {profile.modes} are outside {adapter.modes}",
                )

                if profile.backend_path == "modules.DiffusersImage.LoadPipeline":
                    managed_repos = {repo.casefold() for repo in adapter.managed_repos}
                    self.assertIn(profile.default_repo.casefold(), managed_repos)
                    if profile.fallback_repo:
                        self.assertIn(profile.fallback_repo.casefold(), managed_repos)
                else:
                    self.assertEqual(profile.default_repo.casefold(), adapter.default_repo.casefold())


if __name__ == "__main__":
    unittest.main()
