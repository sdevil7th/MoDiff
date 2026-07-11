import unittest
from unittest.mock import Mock, patch

import torch

from modiff.config import CONFIG
from modules.WanVACE.main import LoadPipeline, WAN_VACE_DEFAULT_REPO


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
            patch("modules.WanVACE.main.local_files_only", return_value=True),
            patch("diffusers.AutoencoderKLWan.from_pretrained", return_value=vae) as vae_from_pretrained,
            patch("diffusers.WanVACEPipeline.from_pretrained", return_value=pipeline) as from_pretrained,
            patch("modules.WanVACE.main.apply_pipeline_offload"),
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


if __name__ == "__main__":
    unittest.main()
