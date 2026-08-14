import json
from pathlib import Path
import unittest

from modiff.studio_execution_specs import (
    ALLEGRO_DIFFUSERS_FILES,
    COGVIDEOX_2B_DIFFUSERS_FILES,
    FRAMEPACK_BASE_COMPONENT_FILES,
    FRAMEPACK_TRANSFORMER_DIFFUSERS_FILES,
    FRAMEPACK_VISION_COMPONENT_FILES,
    LATTE_DIFFUSERS_FILES,
    LTX_VIDEO_DIFFUSERS_FILES,
    MOCHI_DIFFUSERS_FILES,
    SANA_VIDEO_DIFFUSERS_FILES,
    STABLE_VIDEO_DIFFUSION_FP16_FILES,
    WAN_22_T2V_A14B_DIFFUSERS_FILES,
    WAN_FLF_14B_DIFFUSERS_FILES,
    studio_capability_definitions,
)


ROOT = Path(__file__).resolve().parents[1]


class VideoDownloadSelectionTests(unittest.TestCase):
    def test_only_intentionally_gated_video_contracts_remain_without_download_manifests(self):
        capabilities = studio_capability_definitions()
        unbounded = {
            model_type
            for model_type, capability in capabilities.items()
            if capability.get("outputKind") == "video"
            and capability.get("defaultRepo")
            and not capability.get("downloadFiles")
        }

        self.assertEqual(unbounded, {"LTX2ConditionPipeline", "WanAnimatePipeline"})

    def test_remaining_admitted_wan_routes_publish_bounded_component_selections(self):
        capabilities = studio_capability_definitions()
        wan22_review = json.loads((ROOT / "data" / "wan22-a14b-modular-artifact-review.json").read_text())
        wan21_review = json.loads((ROOT / "data" / "wan21-14b-modular-artifact-review.json").read_text())
        receipts = {
            item["repository"]: item
            for item in [*wan22_review["repositories"], *wan21_review["repositories"]]
        }
        cases = {
            "Wan22Pipeline": (
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                WAN_22_T2V_A14B_DIFFUSERS_FILES,
                43,
                126_199_294_813,
            ),
            "WanImage2VideoModularPipeline": (
                "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
                WAN_FLF_14B_DIFFUSERS_FILES,
                41,
                90_104_408_960,
            ),
        }
        for model_type, (repo, files, file_count, byte_size) in cases.items():
            with self.subTest(model_type=model_type):
                selected = set(files)
                self.assertEqual(capabilities[model_type]["downloadFiles"], files)
                self.assertEqual(len(selected), file_count)
                self.assertEqual(receipts[repo]["selectedDownloadFileCount"], file_count)
                self.assertEqual(receipts[repo]["selectedDownloadBytes"], byte_size)
                self.assertFalse(any(path.startswith("assets/") for path in selected))
                self.assertFalse(any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected))

    def test_framepack_selection_matches_the_three_repository_loader_composition(self):
        review = json.loads((ROOT / "data" / "framepack-artifact-review.json").read_text())
        capability = studio_capability_definitions()["HunyuanVideoFramepackPipeline"]
        requirements = capability["modeRequirements"]["image_to_video"]["modelRequirements"]

        self.assertEqual(capability["downloadFiles"], FRAMEPACK_TRANSFORMER_DIFFUSERS_FILES)
        self.assertEqual(len(FRAMEPACK_TRANSFORMER_DIFFUSERS_FILES), 7)
        self.assertEqual(len(FRAMEPACK_BASE_COMPONENT_FILES), 21)
        self.assertEqual(len(FRAMEPACK_VISION_COMPONENT_FILES), 5)
        self.assertEqual(
            {requirement["repo"]: requirement["downloadFiles"] for requirement in requirements},
            {
                "hunyuanvideo-community/HunyuanVideo": FRAMEPACK_BASE_COMPONENT_FILES,
                "lllyasviel/flux_redux_bfl": FRAMEPACK_VISION_COMPONENT_FILES,
            },
        )
        self.assertFalse(
            any(
                path.endswith((".bin", ".ckpt", ".pt", ".pth"))
                for selection in (
                    FRAMEPACK_TRANSFORMER_DIFFUSERS_FILES,
                    FRAMEPACK_BASE_COMPONENT_FILES,
                    FRAMEPACK_VISION_COMPONENT_FILES,
                )
                for path in selection
            )
        )
        self.assertNotIn("transformer/config.json", FRAMEPACK_BASE_COMPONENT_FILES)
        self.assertNotIn("image_embedder/config.json", FRAMEPACK_VISION_COMPONENT_FILES)
        self.assertEqual(review["repositories"]["transformer"]["selectedDownloadBytes"], 25_748_917_665)
        self.assertEqual(review["repositories"]["base"]["selectedDownloadBytes"], 16_261_369_704)
        self.assertEqual(review["repositories"]["vision"]["selectedDownloadBytes"], 856_508_718)
        self.assertEqual(review["appDownloadQueue"]["aggregateSelectedBytes"], 42_866_796_087)

    def test_cogvideox_selection_is_complete_and_safe(self):
        selected = set(COGVIDEOX_2B_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["CogVideoXPipeline"]

        self.assertEqual(capability["downloadFiles"], COGVIDEOX_2B_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 17)
        self.assertIn("text_encoder/model.safetensors.index.json", selected)
        self.assertIn("transformer/diffusion_pytorch_model.safetensors", selected)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn(".gitignore", selected)
        self.assertNotIn("README_zh.md", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )

    def test_long_ltx_route_reuses_the_bounded_ltx_selection(self):
        capability = studio_capability_definitions()[
            "LTXI2VLongMultiPromptPipeline"
        ]

        self.assertEqual(capability["downloadFiles"], LTX_VIDEO_DIFFUSERS_FILES)

    def test_stable_video_selection_matches_loader_variant_and_excludes_duplicates(self):
        selected = set(STABLE_VIDEO_DIFFUSION_FP16_FILES)
        capability = studio_capability_definitions()["StableVideoDiffusionPipeline"]

        self.assertEqual(
            capability["downloadFiles"],
            STABLE_VIDEO_DIFFUSION_FP16_FILES,
        )
        self.assertEqual(len(selected), 12)
        self.assertIn("image_encoder/model.fp16.safetensors", selected)
        self.assertIn("unet/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertIn("vae/diffusion_pytorch_model.fp16.safetensors", selected)
        self.assertNotIn("image_encoder/model.safetensors", selected)
        self.assertNotIn("unet/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertNotIn("svd_xt_1_1.safetensors", selected)
        self.assertNotIn("svd11.webp", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )

    def test_sana_video_selection_covers_both_safe_routes(self):
        selected = set(SANA_VIDEO_DIFFUSERS_FILES)
        capabilities = studio_capability_definitions()

        self.assertEqual(len(selected), 20)
        for model_type in ("SanaVideoPipeline", "SanaImageToVideoPipeline"):
            with self.subTest(model_type=model_type):
                self.assertEqual(
                    capabilities[model_type]["downloadFiles"],
                    SANA_VIDEO_DIFFUSERS_FILES,
                )
        self.assertIn("text_encoder/model.safetensors.index.json", selected)
        self.assertIn(
            "transformer/diffusion_pytorch_model.safetensors.index.json",
            selected,
        )
        self.assertIn("vae/diffusion_pytorch_model.safetensors", selected)
        self.assertFalse(
            any(path.endswith((".bin", ".ckpt", ".pt", ".pth")) for path in selected)
        )

    def test_allegro_selection_excludes_unsafe_duplicate_encoder(self):
        review = json.loads((ROOT / "data" / "allegro-artifact-review.json").read_text())
        selected = set(ALLEGRO_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["AllegroPipeline"]

        self.assertEqual(capability["downloadFiles"], ALLEGRO_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 18)
        self.assertTrue({item["path"] for item in review["weightFiles"]}.issubset(selected))
        self.assertTrue(
            {item["path"] for item in review["excludedUnsafeWeightFiles"]}.isdisjoint(selected)
        )
        self.assertFalse(any(path.endswith(".bin") for path in selected))

    def test_latte_selection_excludes_legacy_and_unreferenced_components(self):
        review = json.loads((ROOT / "data" / "latte-artifact-review.json").read_text())
        selected = set(LATTE_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["LattePipeline"]

        self.assertEqual(capability["downloadFiles"], LATTE_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 18)
        self.assertTrue({item["path"] for item in review["weightFiles"]}.issubset(selected))
        self.assertTrue({item["path"] for item in review["excludedArtifacts"]}.isdisjoint(selected))
        self.assertNotIn("vae_temporal_decoder/config.json", selected)
        self.assertFalse(any(path.endswith((".bin", ".pt", ".pth")) for path in selected))

    def test_mochi_selection_excludes_every_duplicate_partition(self):
        review = json.loads((ROOT / "data" / "mochi-artifact-review.json").read_text())
        selected = set(MOCHI_DIFFUSERS_FILES)
        capability = studio_capability_definitions()["MochiPipeline"]
        excluded = {
            item["path"]
            for partition in review["excludedArtifactPartitions"]
            for item in partition["files"]
        }

        self.assertEqual(capability["downloadFiles"], MOCHI_DIFFUSERS_FILES)
        self.assertEqual(len(selected), 21)
        self.assertTrue({item["path"] for item in review["weightFiles"]}.issubset(selected))
        self.assertTrue(excluded.isdisjoint(selected))
        self.assertIn("transformer/diffusion_pytorch_model.safetensors.index.bf16.json", selected)
        self.assertNotIn("assets/grid.gif", selected)


if __name__ == "__main__":
    unittest.main()
