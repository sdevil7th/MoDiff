"""Run the pinned Helios encoders with real offload hooks and tiny, local tensors."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

pytest.importorskip("transformers", reason="Helios encoders require the reviewed optional Transformers runtime")

from diffusers.modular_pipelines import PipelineState
from diffusers import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from diffusers.modular_pipelines.components_manager import custom_offload_with_hook
from diffusers.modular_pipelines.helios.encoders import HeliosImageVaeEncoderStep, HeliosVideoVaeEncoderStep
from diffusers.utils.accelerate_utils import apply_forward_hook

from modules.ModularDiffusers import reviewed_blocks


PIPELINES = ("HeliosModularPipeline", "HeliosPyramidModularPipeline", "HeliosPyramidDistilledModularPipeline")


class TinyVae(AutoencoderKLWan):
    def __init__(self):
        # Keep the real component type/config while making encode deterministic
        # for the offload-normalization test. This is not a full VAE quality test.
        super().__init__(base_dim=4, decoder_base_dim=4, z_dim=1, dim_mult=[1],
                         num_res_blocks=1, temperal_downsample=[],
                         latents_mean=[1.0], latents_std=[2.0])
        self.weight = torch.nn.Parameter(torch.ones(1))

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype

    @apply_forward_hook
    def encode(self, value):
        assert value.device == self.device
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda generator=None: value[:, :1]))


def reviewed_encoder(pipeline_class, workflow, *, device="meta", hooked=True, fail=False):
    # Meta is a second real tensor device available in CPU CI. Before the fix,
    # normalization fails with the same cross-device operation as CPU -> ROCm.
    target = torch.device(device)
    vae = TinyVae()
    strategy = Mock(return_value=[])
    user_hook = custom_offload_with_hook("test-vae", vae, target, offload_strategy=strategy) if hooked else None
    if user_hook:
        user_hook.hook.other_hooks = []
    processor = VideoProcessor(vae_scale_factor=1)
    processor.preprocess = Mock(return_value=torch.full((1, 3, 2, 2), 3.0))
    processor.preprocess_video = Mock(return_value=torch.full((1, 3, 5, 2, 2), 3.0))
    if fail:
        processor.preprocess.side_effect = ValueError("invalid test image")
    block = HeliosImageVaeEncoderStep() if workflow == "image2video" else HeliosVideoVaeEncoderStep()
    library = reviewed_blocks.reviewed_huggingface_node_library()
    definition = next(item for item in library["definitions"]
                      if item.get("pipelineClass") == pipeline_class and item.get("workflowId") == workflow)
    descriptor = next(item for item in library["blockDefinitions"] if item["className"] == type(block).__name__)
    placement = next(item for item in definition["blockPlacements"] if item["blockDefinitionId"] == descriptor["id"])
    root = SimpleNamespace(sub_blocks={})
    cursor = root
    for part in placement["path"][:-1]:
        cursor.sub_blocks[part] = SimpleNamespace(sub_blocks={})
        cursor = cursor.sub_blocks[part]
    cursor.sub_blocks[placement["path"][-1]] = block
    pipeline = SimpleNamespace(blocks=root, component_names=(), vae=vae, video_processor=processor,
                               _execution_device=target, vae_scale_factor_temporal=4)
    state = PipelineState()
    values = {"image" if workflow == "image2video" else "video": object(), "height": 2, "width": 2,
              "num_latent_frames_per_chunk": 2}
    with patch.object(reviewed_blocks, "_new_runtime", return_value=(object(), pipeline, state)):
        try:
            result = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                pipeline_class=pipeline_class, workflow_id=workflow, placement_path=placement["path"],
                block_definition_id=descriptor["id"], block_class=descriptor["className"],
                block_contract_hash=descriptor["contentHash"], execution_kind="step", **values,
            )
        except ValueError:
            if not fail:
                raise
            result = None
    return vae, user_hook, strategy, state, result


class ReviewedHeliosVaeOffloadTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MODIFF_TEST_HELIOS_VAE_ACCELERATOR") == "1", "opt-in accelerator probe")
    def test_accelerator_offload_with_real_tensors(self):
        for pipeline in PIPELINES:
            for workflow in ("image2video", "video2video"):
                with self.subTest(pipeline=pipeline, workflow=workflow):
                    vae, hook, strategy, state, result = reviewed_encoder(pipeline, workflow, device="cuda:0")
                    self.assertIsNotNone(result["state_out"])
                    self.assertTrue(torch.equal(state.get("image_latents").cpu(), torch.ones(1, 1, 1, 2, 2)))
                    self.assertIs(vae._hf_hook, hook.hook)
                    strategy.assert_called_once()
                    hook.offload()
                    self.assertEqual(vae.device.type, "cpu")

    def test_real_upstream_image_and_video_encoders_onload_before_normalization(self):
        for pipeline in PIPELINES:
            for workflow in ("image2video", "video2video"):
                with self.subTest(pipeline=pipeline, workflow=workflow):
                    vae, hook, strategy, state, result = reviewed_encoder(pipeline, workflow)
                    self.assertIsNotNone(result["state_out"])
                    self.assertEqual(state.get("image_latents").device.type, "meta")
                    extra = "fake_image_latents" if workflow == "image2video" else "video_latents"
                    self.assertEqual(state.get(extra).device.type, "meta")
                    self.assertIs(vae._hf_hook, hook.hook)
                    self.assertIs(hook.hook.offload_strategy, strategy)
                    strategy.assert_called_once()

    def test_cpu_and_unhooked_execution_preserve_normalized_values(self):
        for workflow in ("image2video", "video2video"):
            for hooked in (True, False):
                with self.subTest(workflow=workflow, hooked=hooked):
                    _vae, _hook, strategy, state, _result = reviewed_encoder(
                        PIPELINES[0], workflow, device="cpu", hooked=hooked)
                    self.assertTrue(torch.equal(state.get("image_latents"), torch.ones(1, 1, 1, 2, 2)))
                    strategy.assert_not_called()

    def test_encoder_error_preserves_the_installed_hook_and_strategy(self):
        vae, hook, strategy, _state, result = reviewed_encoder(PIPELINES[2], "image2video", fail=True)
        self.assertIsNone(result)
        self.assertIs(vae._hf_hook, hook.hook)
        self.assertIs(hook.hook.offload_strategy, strategy)
        strategy.assert_called_once()

    def test_other_pipeline_or_block_does_not_trigger_component_movement(self):
        vae = TinyVae()
        hook = custom_offload_with_hook("test-vae", vae, torch.device("meta"))
        pipeline = SimpleNamespace(vae=vae)
        for family, block in (("WanModularPipeline", "HeliosImageVaeEncoderStep"),
                              (PIPELINES[0], "HeliosTextEncoderStep")):
            reviewed_blocks._prepare_reviewed_block_components(family, block, pipeline)
        self.assertEqual(vae.device.type, "cpu")
        self.assertIs(vae._hf_hook, hook.hook)
