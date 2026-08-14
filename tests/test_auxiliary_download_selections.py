import unittest

from modiff.server import STUDIO_MODEL_CAPABILITIES, studio_download_files_for_repo
from modiff.studio_execution_specs import (
    ANIMATEDIFF_MOTION_FP16_FILES,
    ANIMATEDIFF_MOTION_REPO,
    ANIMATELCM_MOTION_FP16_FILES,
    ANIMATELCM_MOTION_REPO,
    HUNYUAN_DIT_CONTROLNET_CANNY_FILES,
    HUNYUAN_DIT_CONTROLNET_CANNY_REPO,
    QWEN_CONTROLNET_FILES,
    QWEN_CONTROLNET_REPO,
    SD15_CONTROLNET_CANNY_FILES,
    SD15_CONTROLNET_CANNY_REPO,
    SDXL_CONTROLNET_CANNY_FP16_FILES,
    SDXL_CONTROLNET_CANNY_REPO,
    SDXL_T2I_ADAPTER_CANNY_FP16_FILES,
    SDXL_T2I_ADAPTER_CANNY_REPO,
)


class AuxiliaryDownloadSelectionTests(unittest.TestCase):
    CASES = {
        ("AnimateDiffPipeline", "text_to_video"): (
            ANIMATEDIFF_MOTION_REPO,
            ANIMATEDIFF_MOTION_FP16_FILES,
            1_815_334_485,
        ),
        ("AnimateLCMPipeline", "text_to_video"): (
            ANIMATELCM_MOTION_REPO,
            ANIMATELCM_MOTION_FP16_FILES,
            1_042_327_684,
        ),
        ("StableDiffusionXLAdapterPipeline", "control_image"): (
            SDXL_T2I_ADAPTER_CANNY_REPO,
            SDXL_T2I_ADAPTER_CANNY_FP16_FILES,
            158_071_198,
        ),
        ("StableDiffusionXLControlNetPipeline", "control_image"): (
            SDXL_CONTROLNET_CANNY_REPO,
            SDXL_CONTROLNET_CANNY_FP16_FILES,
            2_502_145_849,
        ),
        ("HunyuanDiTControlNetPipeline", "control_image"): (
            HUNYUAN_DIT_CONTROLNET_CANNY_REPO,
            HUNYUAN_DIT_CONTROLNET_CANNY_FILES,
            2_976_971_742,
        ),
        ("StableDiffusionPipeline", "control_image"): (
            SD15_CONTROLNET_CANNY_REPO,
            SD15_CONTROLNET_CANNY_FILES,
            1_445_176_301,
        ),
        ("QwenImageModularPipeline", "control_image"): (
            QWEN_CONTROLNET_REPO,
            QWEN_CONTROLNET_FILES,
            3_536_036_150,
        ),
        ("QwenImageControlNetPipeline", "control_image"): (
            QWEN_CONTROLNET_REPO,
            QWEN_CONTROLNET_FILES,
            3_536_036_150,
        ),
    }

    def test_every_admitted_dependency_repository_resolves_a_bounded_selection(self):
        unbounded = set()
        for model_type, capability in STUDIO_MODEL_CAPABILITIES.items():
            requirements = list(capability.get("additionalRequirements") or [])
            for mode_requirement in (capability.get("modeRequirements") or {}).values():
                requirements.extend(mode_requirement.get("modelRequirements") or [])
            for requirement in requirements:
                repo = requirement.get("repo")
                if repo and not studio_download_files_for_repo(repo):
                    unbounded.add((model_type, repo))

        self.assertEqual(unbounded, set())

    def test_reviewed_auxiliary_selections_match_their_capability_requirements(self):
        total_bytes = 0
        for (model_type, mode), (repo, expected_files, selected_bytes) in self.CASES.items():
            with self.subTest(model_type=model_type, mode=mode):
                requirement = next(
                    item
                    for item in STUDIO_MODEL_CAPABILITIES[model_type]["modeRequirements"][mode][
                        "modelRequirements"
                    ]
                    if item["repo"] == repo
                )
                self.assertEqual(requirement["downloadFiles"], expected_files)
                self.assertCountEqual(studio_download_files_for_repo(repo), expected_files)
                self.assertGreater(selected_bytes, 0)
                total_bytes += selected_bytes

        self.assertEqual(total_bytes, 17_012_099_559)

    def test_auxiliary_selections_exclude_code_demos_and_unsafe_weights(self):
        for _pair, (_repo, files, _selected_bytes) in self.CASES.items():
            with self.subTest(files=files):
                self.assertFalse(any(path.endswith((".bin", ".ckpt", ".pt", ".pth", ".py")) for path in files))
                self.assertFalse(any(path.startswith(("conds/", "images/", "outputs/")) for path in files))
                self.assertFalse(any(path.endswith((".jpg", ".png", ".webp")) for path in files))

        self.assertIn("diffusion_pytorch_model.safetensors", SD15_CONTROLNET_CANNY_FILES)
        self.assertNotIn("diffusion_pytorch_model.fp16.safetensors", SD15_CONTROLNET_CANNY_FILES)
        self.assertIn("diffusion_pytorch_model.fp16.safetensors", ANIMATEDIFF_MOTION_FP16_FILES)
        self.assertIn("diffusion_pytorch_model.fp16.safetensors", ANIMATELCM_MOTION_FP16_FILES)
        self.assertIn("AnimateLCM_sd15_t2v_lora.safetensors", ANIMATELCM_MOTION_FP16_FILES)


if __name__ == "__main__":
    unittest.main()
