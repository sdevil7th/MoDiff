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
                model_id={"source": "hub", "value": WAN_VACE_DEFAULT_REPO},
                revision="pinned-revision",
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
            revision="pinned-revision",
            local_files_only=True,
            cache_dir="E:/MoDiff/huggingface/hub",
        )
        from_pretrained.assert_called_once()
        load_args, load_kwargs = from_pretrained.call_args
        self.assertEqual(load_args, (WAN_VACE_DEFAULT_REPO,))
        self.assertEqual(load_kwargs["torch_dtype"], torch.bfloat16)
        self.assertIs(load_kwargs["vae"], vae)
        self.assertEqual(load_kwargs["revision"], "pinned-revision")
        self.assertEqual(load_kwargs["cache_dir"], "E:/MoDiff/huggingface/hub")
        self.assertTrue(load_kwargs["local_files_only"])
        self.assertTrue(load_kwargs["low_cpu_mem_usage"])


class WanVaceLongVideoTests(unittest.TestCase):
    def test_long_masked_video_uses_native_overlapping_segments_and_generated_anchor(self):
        class Output:
            def __init__(self, frames):
                self.frames = [frames]

        class Pipeline:
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

    def test_rgb_mask_neutralization_uses_gray_not_packed_red(self):
        frame = Image.new("RGB", (2, 1), (23, 41, 59))
        mask = Image.new("L", (2, 1), 0)
        mask.putpixel((1, 0), 255)

        result = _neutralize_masked_region(frame, mask)

        self.assertEqual(result.getpixel((0, 0)), (23, 41, 59))
        self.assertEqual(result.getpixel((1, 0)), (127, 127, 127))


if __name__ == "__main__":
    unittest.main()
