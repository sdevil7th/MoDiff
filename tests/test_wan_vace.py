import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from modiff.config import CONFIG
from modules.DiffusersVideo.main import Generate, LoadPipeline
from modules.DiffusersVideo.wan_vace import (
    WAN_VACE_DEFAULT_REPO,
    _neutralize_masked_region,
)


class WanVaceLoaderTests(unittest.TestCase):
    def test_loader_uses_the_app_configured_hugging_face_cache(self):
        node = object.__new__(LoadPipeline)
        node.node_id = "wan-loader-test"
        node.params = {}
        node.output = {}
        node.progress = Mock()
        node.mm_add = Mock()
        vae = Mock()
        pipeline = Mock()

        with (
            patch.dict(CONFIG.hf, {"cache_dir": "E:/MoDiff/huggingface/hub", "online_status": "Auto"}),
            patch("modules.DiffusersVideo.wan_vace.local_files_only", return_value=True),
            patch("diffusers.AutoencoderKLWan.from_pretrained", return_value=vae) as vae_from_pretrained,
            patch("diffusers.WanVACEPipeline.from_pretrained", return_value=pipeline) as from_pretrained,
            patch("modules.DiffusersVideo.wan_vace.apply_pipeline_offload"),
        ):
            result = node.execute(
                pipeline_class="WanVACEPipeline",
                model_id={"source": "hub", "value": WAN_VACE_DEFAULT_REPO},
                revision="ec4d2cb062b548996b179d493fdd05340de702a1",
                dtype="bfloat16",
                device="cuda:0",
                auto_offload=True,
                offload_mode="model_cpu",
                low_cpu_mem_usage=True,
            )

        self.assertIs(result["pipeline"], pipeline)
        vae_from_pretrained.assert_called_once_with(
            WAN_VACE_DEFAULT_REPO,
            subfolder="vae",
            torch_dtype=torch.float32,
            revision="ec4d2cb062b548996b179d493fdd05340de702a1",
            local_files_only=True,
            use_safetensors=True,
            cache_dir="E:/MoDiff/huggingface/hub",
        )
        from_pretrained.assert_called_once()
        load_args, load_kwargs = from_pretrained.call_args
        self.assertEqual(load_args, (WAN_VACE_DEFAULT_REPO,))
        self.assertEqual(load_kwargs["torch_dtype"], torch.bfloat16)
        self.assertIs(load_kwargs["vae"], vae)
        self.assertEqual(load_kwargs["revision"], "ec4d2cb062b548996b179d493fdd05340de702a1")
        self.assertEqual(load_kwargs["cache_dir"], "E:/MoDiff/huggingface/hub")
        self.assertTrue(load_kwargs["local_files_only"])
        self.assertTrue(load_kwargs["use_safetensors"])
        self.assertTrue(load_kwargs["low_cpu_mem_usage"])


class WanVaceLongVideoTests(unittest.TestCase):
    def test_generic_generate_preserves_canonical_numpy_and_torch_mask_layouts(self):
        class Output:
            frames = [["generated"]]

        class Pipeline:
            _modiff_video_pipeline_class = "WanVACEPipeline"
            _execution_device = "cpu"
            vae_scale_factor_temporal = 1
            vae_scale_factor_spatial = 8
            transformer = type("Transformer", (), {"config": type("Config", (), {"patch_size": (1, 2, 2)})()})()
            boundary_ratio = None

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return Output()

        cases = (
            (
                np.full((16, 16, 3), 23, dtype=np.uint8),
                np.pad(
                    np.full((16, 8), 255, dtype=np.uint8),
                    ((0, 0), (8, 0)),
                ),
                lambda conditioned: (
                    np.all(conditioned[:, :8] == 23),
                    np.all(conditioned[:, 8:] == 127),
                ),
            ),
            (
                torch.full((3, 16, 16), 23, dtype=torch.uint8),
                torch.cat(
                    [
                        torch.zeros((16, 8), dtype=torch.uint8),
                        torch.full((16, 8), 255, dtype=torch.uint8),
                    ],
                    dim=1,
                ),
                lambda conditioned: (
                    bool(torch.all(conditioned[:, :, :8] == 23)),
                    bool(torch.all(conditioned[:, :, 8:] == 127)),
                ),
            ),
        )
        for index, (frame, mask, assertions) in enumerate(cases):
            pipeline = Pipeline()
            result = Generate(f"canonical-vace-layout-{index}").execute(
                pipeline=pipeline,
                mode="video_inpaint",
                video=[frame],
                mask=[mask],
                width=16,
                height=16,
                num_frames=1,
                num_inference_steps=1,
            )

            conditioned = pipeline.calls[0]["video"][0]
            with self.subTest(container=type(frame).__name__):
                self.assertEqual(assertions(conditioned), (True, True))
                self.assertEqual(result["frames_out"], 1)

    def test_long_masked_video_uses_native_overlapping_segments_and_generated_anchor(self):
        class Output:
            def __init__(self, frames):
                self.frames = [frames]

        class Pipeline:
            _modiff_video_pipeline_class = "WanVACEPipeline"
            _execution_device = "cpu"
            vae_scale_factor_temporal = 4
            vae_scale_factor_spatial = 8

            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                call_index = len(self.calls)
                self.calls.append(kwargs)
                return Output([f"generated-{call_index}-{index}" for index in range(kwargs["num_frames"])])

        pipeline = Pipeline()
        video = [np.full((4, 4, 3), index % 255, dtype=np.uint8) for index in range(161)]
        mask = [np.full((4, 4), 255, dtype=np.uint8) for _ in range(161)]
        output = Generate().execute(
            pipeline=pipeline,
            mode="video_inpaint",
            video=video,
            mask=mask,
            prompt="Replace the masked vessel with one stable amber vessel.",
            width=832,
            height=480,
            num_frames=161,
            num_inference_steps=4,
            seed=17,
        )

        self.assertEqual(len(pipeline.calls), 2)
        self.assertEqual([call["num_frames"] for call in pipeline.calls], [81, 81])
        self.assertEqual(pipeline.calls[0]["generator"].initial_seed(), 17)
        self.assertEqual(pipeline.calls[1]["generator"].initial_seed(), 17)
        self.assertTrue(np.all(pipeline.calls[0]["video"][0] == 127))
        self.assertEqual(pipeline.calls[1]["video"][0], "generated-0-80")
        self.assertTrue(np.all(pipeline.calls[1]["video"][1] == 127))
        self.assertTrue(np.all(pipeline.calls[1]["mask"][0] == 0))
        self.assertTrue(np.all(pipeline.calls[1]["mask"][1] == 255))
        self.assertEqual(output["frames_out"], 161)
        self.assertEqual(output["video_out"][80], "generated-0-80")
        self.assertEqual(output["video_out"][81], "generated-1-1")

    def test_mask_neutralization_preserves_black_regions(self):
        frame = np.full((2, 2, 3), 23, dtype=np.uint8)
        mask = np.array([[0, 255], [0, 255]], dtype=np.uint8)

        result = _neutralize_masked_region(frame, mask)

        self.assertTrue(np.all(result[:, 0] == 23))
        self.assertTrue(np.all(result[:, 1] == 127))

    def test_torch_mask_neutralization_broadcasts_a_spatial_mask_across_channels(self):
        frame = torch.full((3, 2, 2), 23, dtype=torch.uint8)
        mask = torch.tensor([[0, 255], [0, 255]], dtype=torch.uint8)

        result = _neutralize_masked_region(frame, mask)

        self.assertTrue(torch.all(result[:, :, 0] == 23))
        self.assertTrue(torch.all(result[:, :, 1] == 127))

    def test_rgb_mask_neutralization_uses_gray_not_packed_red(self):
        frame = Image.new("RGB", (2, 1), (23, 41, 59))
        mask = Image.new("L", (2, 1), 0)
        mask.putpixel((1, 0), 255)

        result = _neutralize_masked_region(frame, mask)

        self.assertEqual(result.getpixel((0, 0)), (23, 41, 59))
        self.assertEqual(result.getpixel((1, 0)), (127, 127, 127))


if __name__ == "__main__":
    unittest.main()
