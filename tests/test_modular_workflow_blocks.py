import copy
import json
import pickle
import sys
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from modules.ModularDiffusers.route_state import bind_loader_outputs, issue_pipeline_instance_token
from modules.ModularDiffusers.workflow_blocks import (
    WorkflowCosmos3DistilledDecode,
    WorkflowCosmos3DistilledDenoise,
    WorkflowCosmos3DistilledTextEncode,
    WorkflowCosmos3DistilledVaeEncode,
    WorkflowCosmos3OmniAfterDecode,
    WorkflowCosmos3OmniDecode,
    WorkflowCosmos3OmniDenoise,
    WorkflowCosmos3OmniTextEncode,
    WorkflowCosmos3OmniVaeEncode,
    WorkflowDecodeAudio,
    WorkflowDecodeImage,
    WorkflowDecodeVideo,
    WorkflowDenoise,
    WorkflowHunyuanVideo15Decode,
    WorkflowHunyuanVideo15Denoise,
    WorkflowHunyuanVideo15ImageEncode,
    WorkflowHunyuanVideo15TextEncode,
    WorkflowHunyuanVideo15VaeEncode,
    WorkflowIdeogram4Decode,
    WorkflowIdeogram4Denoise,
    WorkflowIdeogram4PromptUpsample,
    WorkflowIdeogram4TextEncode,
    WorkflowImageDenoise,
    WorkflowImageEncode,
    WorkflowKrea2Decode,
    WorkflowKrea2Denoise,
    WorkflowKrea2TextEncode,
    WorkflowKrea2TurboDenoise,
    WorkflowKrea2TurboTextEncode,
    WorkflowLTX25ConditionEncode,
    WorkflowLTX25Decode,
    WorkflowLTX25Denoise,
    WorkflowLTX25ReferenceEncode,
    WorkflowLTX25TextEncode,
    WorkflowLTX25VaeEncode,
    WorkflowMiniMaxH3BeforeEncode,
    WorkflowMiniMaxH3Decode,
    WorkflowMiniMaxH3Denoise,
    WorkflowMiniMaxH3ReferenceAssembler,
    WorkflowMiniMaxH3TextEncode,
    WorkflowMiniMaxH3VaeEncode,
    WorkflowSemanticGeneration,
    WorkflowStableDiffusion3Decode,
    WorkflowStableDiffusion3Denoise,
    WorkflowStableDiffusion3TextEncode,
    WorkflowStableDiffusion3VaeEncode,
    WorkflowTextEncode,
    WorkflowVideoDenoise,
    WorkflowVideoEncode,
    WorkflowVideoImageEncode,
    WorkflowWanAnimateDecode,
    WorkflowWanAnimateDenoise,
    WorkflowWanAnimateImageEncode,
    WorkflowWanAnimateTextEncode,
    WorkflowWanAnimateVaeEncode,
    WorkflowWanAnimateVideoEncode,
    _issue_workflow_state,
    _require_workflow_state,
    _validated_minimax_h3_references,
)


PIPELINE_CLASS = "MiniMaxMusic3ModularPipeline"
WORKFLOW_ID = "default"


class FakeState:
    def __init__(self, **values):
        self.values = values

    def get(self, name):
        return self.values.get(name)


@dataclass
class FakeLTX2VideoCondition:
    frames: object
    index: int = 0
    strength: float = 1.0
    crf: int | None = None


@dataclass
class FakeLTX2ReferenceCondition:
    frames: object
    strength: float = 1.0


@dataclass
class FakeMiniMaxH3Reference:
    pass


@dataclass
class FakeMiniMaxH3ImageReference(FakeMiniMaxH3Reference):
    image: object

    kind = "image"


@dataclass
class FakeMiniMaxH3VideoReference(FakeMiniMaxH3Reference):
    frames: object
    fps: float | None = None
    audio: object | None = None
    sample_rate: int | None = None

    kind = "video"


@dataclass
class FakeMiniMaxH3AudioReference(FakeMiniMaxH3Reference):
    audio: object
    sample_rate: int | None = None

    kind = "audio"


class RecordingPipeline:
    def __init__(self, result, *, sampling_rate=None):
        self.result = result
        self.calls = []
        self._execution_device = torch.device("cpu")
        if sampling_rate is not None:
            self.sampling_rate = sampling_rate

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if type(self.result) is FakeState and "generator" in kwargs:
            self.result.values["generator"] = kwargs["generator"]
        return self.result

    def update_components(self, **components):
        for name, value in components.items():
            setattr(self, name, value)


class FakeGuider:
    def __init__(self, guidance_scale=4.0):
        self.guidance_scale = guidance_scale

    def new(self, **kwargs):
        return FakeGuider(kwargs.get("guidance_scale", self.guidance_scale))


class ModularWorkflowBlockTests(unittest.TestCase):
    def test_anima_text_to_image_runs_exact_official_stage_sequence(self):
        token = object()
        encoded_state = FakeState(prompt_embeds="encoded")
        text_pipeline = RecordingPipeline(encoded_state)
        text = WorkflowTextEncode("text")
        with mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)):
            text_output = text.execute(
                pipeline_components={},
                pipeline_class="AnimaModularPipeline",
                workflow_id="text2image",
                block_path="text_encoder",
                prompt="masterpiece, best quality, city lights",
                negative_prompt="low quality",
                max_sequence_length=512,
            )
        self.assertEqual(
            text_pipeline.calls,
            [
                {
                    "prompt": "masterpiece, best quality, city lights",
                    "negative_prompt": "low quality",
                    "max_sequence_length": 512,
                }
            ],
        )

        denoised_state = FakeState(latents="latents")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise_pipeline.guider = FakeGuider()
        denoise = WorkflowImageDenoise("denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ) as generator,
        ):
            denoise_output = denoise.execute(
                pipeline_components={},
                pipeline_class="AnimaModularPipeline",
                workflow_id="text2image",
                block_path="denoise",
                state_in=text_output["state_out"],
                width=1024,
                height=768,
                num_images_per_prompt=1,
                num_inference_steps=30,
                guidance_scale=4.5,
                seed=17,
            )
        generator.assert_called_once_with(17, denoise_pipeline)
        self.assertEqual(denoise_pipeline.guider.guidance_scale, 4.5)
        self.assertEqual(
            denoise_pipeline.calls,
            [
                {
                    "state": encoded_state,
                    "width": 1024,
                    "height": 768,
                    "num_images_per_prompt": 1,
                    "num_inference_steps": 30,
                    "generator": "generator",
                }
            ],
        )

        image = Image.new("RGB", (32, 32), "purple")
        decode_pipeline = RecordingPipeline(FakeState(images=[image]))
        decode = WorkflowDecodeImage("decode")
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
            output = decode.execute(
                pipeline_components={},
                pipeline_class="AnimaModularPipeline",
                workflow_id="text2image",
                block_path="decode",
                state_in=denoise_output["state_out"],
            )
        self.assertEqual(decode_pipeline.calls, [{"state": denoised_state, "output_type": "pil"}])
        self.assertEqual(output["images"], [image])

    def test_anima_image_to_image_inserts_only_official_vae_stage(self):
        token = object()
        encoded_state = FakeState(prompt_embeds="encoded")
        text_state = _issue_workflow_state(
            token=token,
            pipeline_class="AnimaModularPipeline",
            workflow_id="img2img",
            completed_stage="text_encoder",
            state=encoded_state,
        )
        vae_state = FakeState(image_latents="image-latents")
        vae_pipeline = RecordingPipeline(vae_state)
        source = Image.new("RGB", (512, 512), "orange")
        image_encode = WorkflowImageEncode("image-encode")
        with (
            mock.patch.object(image_encode, "_prepare_pipeline", return_value=(token, vae_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="image-generator",
            ),
        ):
            encoded = image_encode.execute(
                pipeline_components={},
                pipeline_class="AnimaModularPipeline",
                workflow_id="img2img",
                block_path="vae_encoder",
                state_in=text_state,
                image=source,
                width=512,
                height=512,
                seed=3,
            )
        self.assertEqual(
            vae_pipeline.calls,
            [
                {
                    "state": encoded_state,
                    "image": source,
                    "width": 512,
                    "height": 512,
                    "generator": "image-generator",
                }
            ],
        )

        denoise_pipeline = RecordingPipeline(FakeState(latents="edited"))
        denoise_pipeline.guider = FakeGuider()
        denoise = WorkflowImageDenoise("denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="denoise-generator",
            ),
        ):
            denoise.execute(
                pipeline_components={},
                pipeline_class="AnimaModularPipeline",
                workflow_id="img2img",
                block_path="denoise",
                state_in=encoded["state_out"],
                width=512,
                height=512,
                num_images_per_prompt=1,
                num_inference_steps=20,
                guidance_scale=4.0,
                strength=0.65,
                seed=3,
            )
        self.assertEqual(denoise_pipeline.calls[0]["state"], vae_state)
        self.assertEqual(denoise_pipeline.calls[0]["strength"], 0.65)

    def test_helios_base_text_to_video_runs_official_denoise_and_decode_blocks(self):
        token = object()
        encoded_state = FakeState(prompt_embeds="encoded")
        text_state = _issue_workflow_state(
            token=token,
            pipeline_class="HeliosModularPipeline",
            workflow_id="text2video",
            completed_stage="text_encoder",
            state=encoded_state,
        )
        denoised_state = FakeState(latent_chunks=["chunk"])
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise_pipeline.guider = FakeGuider(5.0)
        denoise = WorkflowVideoDenoise("helios-denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoised = denoise.execute(
                pipeline_components={},
                pipeline_class="HeliosModularPipeline",
                workflow_id="text2video",
                block_path="denoise",
                state_in=text_state,
                width=640,
                height=384,
                num_frames=132,
                num_latent_frames_per_chunk=9,
                history_size_long=16,
                history_size_mid=2,
                history_size_short=1,
                keep_first_frame=True,
                num_inference_steps=50,
                guidance_scale=5.0,
                seed=23,
            )
        self.assertEqual(
            denoise_pipeline.calls,
            [
                {
                    "state": encoded_state,
                    "width": 640,
                    "height": 384,
                    "num_frames": 132,
                    "num_latent_frames_per_chunk": 9,
                    "history_sizes": [16, 2, 1],
                    "keep_first_frame": True,
                    "generator": "generator",
                    "num_inference_steps": 50,
                }
            ],
        )

        frames = [Image.new("RGB", (16, 16), "navy") for _ in range(3)]
        decode_pipeline = RecordingPipeline(FakeState(videos=[frames]))
        decode = WorkflowDecodeVideo("helios-decode")
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
            output = decode.execute(
                pipeline_components={},
                pipeline_class="HeliosModularPipeline",
                workflow_id="text2video",
                block_path="decode",
                state_in=denoised["state_out"],
            )
        self.assertEqual(output["video"], frames)
        self.assertEqual(decode_pipeline.calls, [{"state": denoised_state, "output_type": "pil"}])

    def test_helios_conditional_vae_encoders_preserve_exact_media_route(self):
        token = object()
        source_state = FakeState(prompt_embeds="encoded")
        source_image = Image.new("RGB", (640, 384), "orange")
        image_pipeline = RecordingPipeline(FakeState(image_latents="image"))
        image_node = WorkflowVideoImageEncode("helios-image-encode")
        image_state = _issue_workflow_state(
            token=token,
            pipeline_class="HeliosPyramidModularPipeline",
            workflow_id="image2video",
            completed_stage="text_encoder",
            state=source_state,
        )
        with (
            mock.patch.object(image_node, "_prepare_pipeline", return_value=(token, image_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="image-generator",
            ),
        ):
            image_node.execute(
                pipeline_components={},
                pipeline_class="HeliosPyramidModularPipeline",
                workflow_id="image2video",
                block_path="vae_encoder",
                state_in=image_state,
                image=source_image,
                width=640,
                height=384,
                num_latent_frames_per_chunk=9,
                seed=5,
            )
        self.assertEqual(image_pipeline.calls[0]["image"], source_image)
        self.assertNotIn("video", image_pipeline.calls[0])

        frames = [source_image.copy() for _ in range(33)]
        video_pipeline = RecordingPipeline(FakeState(video_latents="video"))
        video_node = WorkflowVideoEncode("helios-video-encode")
        video_state = _issue_workflow_state(
            token=token,
            pipeline_class="HeliosPyramidModularPipeline",
            workflow_id="video2video",
            completed_stage="text_encoder",
            state=source_state,
        )
        with (
            mock.patch.object(video_node, "_prepare_pipeline", return_value=(token, video_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="video-generator",
            ),
        ):
            video_node.execute(
                pipeline_components={},
                pipeline_class="HeliosPyramidModularPipeline",
                workflow_id="video2video",
                block_path="vae_encoder",
                state_in=video_state,
                video=frames,
                width=640,
                height=384,
                num_latent_frames_per_chunk=9,
                seed=5,
            )
        self.assertEqual(video_pipeline.calls[0]["video"], frames)
        self.assertNotIn("image", video_pipeline.calls[0])

    def test_helios_distilled_uses_stage_steps_and_fixed_guidance(self):
        token = object()
        input_state = FakeState(prompt_embeds="encoded")
        sealed = _issue_workflow_state(
            token=token,
            pipeline_class="HeliosPyramidDistilledModularPipeline",
            workflow_id="text2video",
            completed_stage="text_encoder",
            state=input_state,
        )
        pipeline = RecordingPipeline(FakeState(latent_chunks=[]))
        pipeline.guider = FakeGuider(1.0)
        node = WorkflowVideoDenoise("helios-distilled")
        with (
            mock.patch.object(node, "_prepare_pipeline", return_value=(token, pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            node.execute(
                pipeline_components={},
                pipeline_class="HeliosPyramidDistilledModularPipeline",
                workflow_id="text2video",
                block_path="denoise",
                state_in=sealed,
                width=640,
                height=384,
                num_frames=264,
                num_latent_frames_per_chunk=9,
                history_size_long=16,
                history_size_mid=2,
                history_size_short=1,
                keep_first_frame=True,
                pyramid_stage_1_steps=2,
                pyramid_stage_2_steps=2,
                pyramid_stage_3_steps=2,
                guidance_scale=1.0,
                is_amplify_first_chunk=True,
                seed=9,
            )
        self.assertEqual(pipeline.calls[0]["pyramid_num_inference_steps_list"], [2, 2, 2])
        self.assertTrue(pipeline.calls[0]["is_amplify_first_chunk"])
        with mock.patch.object(node, "_prepare_pipeline", return_value=(token, pipeline)):
            with self.assertRaisesRegex(ValueError, "fixed guidance scale"):
                node.execute(
                    pipeline_components={},
                    pipeline_class="HeliosPyramidDistilledModularPipeline",
                    workflow_id="text2video",
                    block_path="denoise",
                    state_in=sealed,
                    width=640,
                    height=384,
                    num_frames=132,
                    guidance_scale=2.0,
                )

    def test_hunyuan_video_15_text_to_video_runs_exact_three_stage_route(self):
        pipeline_class = "HunyuanVideo15ModularPipeline"
        workflow_id = "text2video"
        token = object()

        encoded_state = FakeState(prompt_embeds="qwen", prompt_embeds_2="byt5")
        text_pipeline = RecordingPipeline(encoded_state)
        text = WorkflowHunyuanVideo15TextEncode("hunyuan-text")
        with mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)):
            encoded = text.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="text_encoder",
                prompt="A red fox crosses a snowy forest clearing",
                negative_prompt="blur",
                num_videos_per_prompt=1,
            )
        self.assertEqual(
            text_pipeline.calls,
            [
                {
                    "prompt": "A red fox crosses a snowy forest clearing",
                    "negative_prompt": "blur",
                    "num_videos_per_prompt": 1,
                }
            ],
        )

        denoised_state = FakeState(latents="video-latents")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise_pipeline.guider = FakeGuider(6.0)
        denoise = WorkflowHunyuanVideo15Denoise("hunyuan-denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoised = denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="denoise",
                state_in=encoded["state_out"],
                width=848,
                height=480,
                num_frames=121,
                num_inference_steps=50,
                guidance_scale=6.0,
                seed=31,
            )
        self.assertEqual(denoise_pipeline.guider.guidance_scale, 6.0)
        self.assertEqual(
            denoise_pipeline.calls,
            [
                {
                    "state": encoded_state,
                    "width": 848,
                    "height": 480,
                    "num_frames": 121,
                    "num_inference_steps": 50,
                    "generator": "generator",
                }
            ],
        )

        frames = [Image.new("RGB", (16, 16), "white") for _ in range(3)]
        decode_pipeline = RecordingPipeline(FakeState(videos=[frames]))
        decode = WorkflowHunyuanVideo15Decode("hunyuan-decode")
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
            output = decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="decode",
                state_in=denoised["state_out"],
            )
        self.assertEqual(decode_pipeline.calls, [{"state": denoised_state, "output_type": "pil"}])
        self.assertEqual(output["video"], frames)

    def test_hunyuan_video_15_image_to_video_preserves_vae_siglip_and_meanflow_route(self):
        pipeline_class = "HunyuanVideo15ModularPipeline"
        workflow_id = "image2video"
        token = object()
        text_state = FakeState(prompt_embeds="qwen", prompt_embeds_2="byt5")
        sealed_text = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage="text_encoder",
            state=text_state,
        )
        source = Image.new("RGB", (848, 480), "orange")

        vae_state = FakeState(image_latents="image-latents")
        vae_pipeline = RecordingPipeline(vae_state)
        vae = WorkflowHunyuanVideo15VaeEncode("hunyuan-vae")
        with mock.patch.object(vae, "_prepare_pipeline", return_value=(token, vae_pipeline)):
            encoded_vae = vae.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="vae_encoder",
                state_in=sealed_text,
                image=source,
            )
        self.assertEqual(
            vae_pipeline.calls,
            [{"state": text_state, "image": source}],
        )

        image_state = FakeState(image_embeds="siglip")
        image_pipeline = RecordingPipeline(image_state)
        image_encode = WorkflowHunyuanVideo15ImageEncode("hunyuan-siglip")
        with mock.patch.object(image_encode, "_prepare_pipeline", return_value=(token, image_pipeline)):
            encoded_image = image_encode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="image_encoder",
                state_in=encoded_vae["state_out"],
            )
        self.assertEqual(image_pipeline.calls, [{"state": vae_state}])

        denoise_pipeline = RecordingPipeline(FakeState(latents="meanflow-latents"))
        denoise_pipeline.guider = FakeGuider(1.0)
        denoise = WorkflowHunyuanVideo15Denoise("hunyuan-i2v-denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="denoise",
                state_in=encoded_image["state_out"],
                num_frames=121,
                num_inference_steps=12,
                guidance_scale=1.0,
                seed=37,
            )
        self.assertEqual(denoise_pipeline.calls[0]["state"], image_state)
        self.assertEqual(denoise_pipeline.calls[0]["num_inference_steps"], 12)
        self.assertNotIn("width", denoise_pipeline.calls[0])
        self.assertNotIn("height", denoise_pipeline.calls[0])
        self.assertEqual(denoise_pipeline.guider.guidance_scale, 1.0)
        with mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)):
            with self.assertRaisesRegex(ValueError, "fixed guidance scale"):
                denoise.execute(
                    pipeline_components={},
                    pipeline_class=pipeline_class,
                    workflow_id=workflow_id,
                    block_path="denoise",
                    state_in=encoded_image["state_out"],
                    num_frames=121,
                    num_inference_steps=12,
                    guidance_scale=2.0,
                )

    def test_hunyuan_video_15_enforces_single_video_and_i2v_geometry_ownership(self):
        pipeline_class = "HunyuanVideo15ModularPipeline"
        token = object()

        text = WorkflowHunyuanVideo15TextEncode("hunyuan-single-video")
        with mock.patch.object(text, "_prepare_pipeline", return_value=(token, RecordingPipeline(FakeState()))):
            with self.assertRaisesRegex(ValueError, "videos per prompt"):
                text.execute(
                    pipeline_components={},
                    pipeline_class=pipeline_class,
                    workflow_id="text2video",
                    block_path="text_encoder",
                    prompt="A single video",
                    num_videos_per_prompt=2,
                )

        text_state = FakeState(prompt_embeds="qwen", prompt_embeds_2="byt5")
        sealed_text = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id="image2video",
            completed_stage="text_encoder",
            state=text_state,
        )
        source = Image.new("RGB", (848, 480), "orange")
        vae = WorkflowHunyuanVideo15VaeEncode("hunyuan-partial-geometry")
        with mock.patch.object(vae, "_prepare_pipeline", return_value=(token, RecordingPipeline(FakeState()))):
            with self.assertRaisesRegex(ValueError, "both be omitted or both be set"):
                vae.execute(
                    pipeline_components={},
                    pipeline_class=pipeline_class,
                    workflow_id="image2video",
                    block_path="vae_encoder",
                    state_in=sealed_text,
                    image=source,
                    width=848,
                )

        image_state = FakeState(image_embeds="siglip")
        sealed_image = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id="image2video",
            completed_stage="image_encoder",
            state=image_state,
        )
        denoise_pipeline = RecordingPipeline(FakeState(latents="meanflow-latents"))
        denoise_pipeline.guider = FakeGuider(1.0)
        denoise = WorkflowHunyuanVideo15Denoise("hunyuan-denoise-geometry")
        with mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)):
            with self.assertRaisesRegex(ValueError, "owned by its VAE encoder"):
                denoise.execute(
                    pipeline_components={},
                    pipeline_class=pipeline_class,
                    workflow_id="image2video",
                    block_path="denoise",
                    state_in=sealed_image,
                    width=848,
                    height=480,
                )

    def test_stable_diffusion_3_text_to_image_runs_exact_three_encoder_and_denoise_route(self):
        pipeline_class = "StableDiffusion3ModularPipeline"
        workflow_id = "text2image"
        token = object()

        encoded_state = FakeState(prompt_embeds="clip-t5")
        text_pipeline = RecordingPipeline(encoded_state)
        text = WorkflowStableDiffusion3TextEncode("sd3-text")
        with mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)):
            encoded = text.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="text_encoder",
                prompt="A glass observatory beneath a violet aurora",
                prompt_2="",
                prompt_3="",
                negative_prompt="blur",
                negative_prompt_2="",
                negative_prompt_3="",
                clip_skip=0,
                max_sequence_length=256,
            )
        self.assertEqual(
            text_pipeline.calls,
            [
                {
                    "prompt": "A glass observatory beneath a violet aurora",
                    "prompt_2": None,
                    "prompt_3": None,
                    "negative_prompt": "blur",
                    "negative_prompt_2": None,
                    "negative_prompt_3": None,
                    "clip_skip": None,
                    "max_sequence_length": 256,
                }
            ],
        )

        denoised_state = FakeState(latents="sd3-latents")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise_pipeline.guider = FakeGuider(7.0)
        denoise = WorkflowStableDiffusion3Denoise("sd3-denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoised = denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="denoise",
                state_in=encoded["state_out"],
                width=1024,
                height=768,
                num_images_per_prompt=1,
                num_inference_steps=28,
                guidance_scale=7.0,
                seed=41,
            )
        self.assertEqual(denoise_pipeline.guider.guidance_scale, 7.0)
        self.assertEqual(
            denoise_pipeline.calls,
            [
                {
                    "state": encoded_state,
                    "width": 1024,
                    "height": 768,
                    "num_images_per_prompt": 1,
                    "num_inference_steps": 28,
                    "generator": "generator",
                }
            ],
        )

        image = Image.new("RGB", (32, 32), "teal")
        decode_pipeline = RecordingPipeline(FakeState(images=[image]))
        decode = WorkflowStableDiffusion3Decode("sd3-decode")
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
            output = decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="decode",
                state_in=denoised["state_out"],
            )
        self.assertEqual(decode_pipeline.calls, [{"state": denoised_state, "output_type": "pil"}])
        self.assertEqual(output["images"], [image])

    def test_stable_diffusion_3_image_to_image_preserves_vae_and_strength_route(self):
        pipeline_class = "StableDiffusion3ModularPipeline"
        workflow_id = "image2image"
        token = object()
        text_state = FakeState(prompt_embeds="clip-t5")
        sealed_text = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage="text_encoder",
            state=text_state,
        )
        source = Image.new("RGB", (1024, 768), "navy")

        vae_state = FakeState(image_latents="sd3-image-latents")
        vae_pipeline = RecordingPipeline(vae_state)
        vae = WorkflowStableDiffusion3VaeEncode("sd3-vae")
        with (
            mock.patch.object(vae, "_prepare_pipeline", return_value=(token, vae_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            encoded = vae.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="vae_encoder",
                state_in=sealed_text,
                image=source,
                width=1024,
                height=768,
                seed=43,
            )
        self.assertEqual(
            vae_pipeline.calls,
            [{"state": text_state, "image": source, "width": 1024, "height": 768, "generator": "generator"}],
        )

        denoise_pipeline = RecordingPipeline(FakeState(latents="sd3-i2i-latents"))
        denoise_pipeline.guider = FakeGuider(7.0)
        denoise = WorkflowStableDiffusion3Denoise("sd3-i2i-denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="denoise",
                state_in=encoded["state_out"],
                width=1024,
                height=768,
                num_images_per_prompt=1,
                num_inference_steps=50,
                guidance_scale=7.0,
                strength=0.6,
                seed=47,
            )
        self.assertEqual(denoise_pipeline.calls[0]["state"], vae_state)
        self.assertEqual(denoise_pipeline.calls[0]["strength"], 0.6)
        self.assertEqual(denoise_pipeline.calls[0]["num_inference_steps"], 50)

    def test_krea_2_base_and_turbo_preserve_distinct_cfg_and_step_contracts(self):
        token = object()
        base_text_state = FakeState(prompt_embeds="qwen3-vl-base")
        base_text_pipeline = RecordingPipeline(base_text_state)
        base_text = WorkflowKrea2TextEncode("krea2-text")
        with mock.patch.object(base_text, "_prepare_pipeline", return_value=(token, base_text_pipeline)):
            encoded_base = base_text.execute(
                pipeline_components={},
                pipeline_class="Krea2ModularPipeline",
                workflow_id="text2image",
                block_path="text_encoder",
                prompt="A chrome orchid in soft gallery light",
                negative_prompt="blur",
                max_sequence_length=512,
            )
        self.assertEqual(
            base_text_pipeline.calls,
            [
                {
                    "prompt": "A chrome orchid in soft gallery light",
                    "negative_prompt": "blur",
                    "max_sequence_length": 512,
                }
            ],
        )

        base_denoise_pipeline = RecordingPipeline(FakeState(latents="krea2-base-latents"))
        base_denoise_pipeline.guider = FakeGuider(4.5)
        base_denoise = WorkflowKrea2Denoise("krea2-denoise")
        with (
            mock.patch.object(base_denoise, "_prepare_pipeline", return_value=(token, base_denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoised_base = base_denoise.execute(
                pipeline_components={},
                pipeline_class="Krea2ModularPipeline",
                workflow_id="text2image",
                block_path="denoise",
                state_in=encoded_base["state_out"],
                width=1024,
                height=768,
                num_images_per_prompt=1,
                num_inference_steps=28,
                guidance_scale=4.5,
                seed=53,
            )
        self.assertEqual(base_denoise_pipeline.guider.guidance_scale, 4.5)
        self.assertEqual(base_denoise_pipeline.calls[0]["num_inference_steps"], 28)

        image = Image.new("RGB", (32, 32), "silver")
        decode_pipeline = RecordingPipeline(FakeState(images=[image]))
        decode = WorkflowKrea2Decode("krea2-decode")
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
            decoded = decode.execute(
                pipeline_components={},
                pipeline_class="Krea2ModularPipeline",
                workflow_id="text2image",
                block_path="decode",
                state_in=denoised_base["state_out"],
            )
        self.assertEqual(decoded["images"], [image])

        turbo_text_state = FakeState(prompt_embeds="qwen3-vl-turbo")
        turbo_text_pipeline = RecordingPipeline(turbo_text_state)
        turbo_text = WorkflowKrea2TurboTextEncode("krea2-turbo-text")
        with mock.patch.object(turbo_text, "_prepare_pipeline", return_value=(token, turbo_text_pipeline)):
            encoded_turbo = turbo_text.execute(
                pipeline_components={},
                pipeline_class="Krea2TurboModularPipeline",
                workflow_id="text2image",
                block_path="text_encoder",
                prompt="A chrome orchid in soft gallery light",
                max_sequence_length=512,
            )
        self.assertEqual(
            turbo_text_pipeline.calls,
            [{"prompt": "A chrome orchid in soft gallery light", "max_sequence_length": 512}],
        )

        turbo_denoise_pipeline = RecordingPipeline(FakeState(latents="krea2-turbo-latents"))
        turbo_denoise = WorkflowKrea2TurboDenoise("krea2-turbo-denoise")
        with (
            mock.patch.object(turbo_denoise, "_prepare_pipeline", return_value=(token, turbo_denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            turbo_denoise.execute(
                pipeline_components={},
                pipeline_class="Krea2TurboModularPipeline",
                workflow_id="text2image",
                block_path="denoise",
                state_in=encoded_turbo["state_out"],
                width=1024,
                height=768,
                num_images_per_prompt=1,
                num_inference_steps=8,
                seed=59,
            )
        self.assertEqual(turbo_denoise_pipeline.calls[0]["num_inference_steps"], 8)
        self.assertFalse(hasattr(turbo_denoise_pipeline, "guider"))

    def test_ideogram_4_preserves_optional_local_upsample_and_asymmetric_cfg_route(self):
        pipeline_class = "Ideogram4ModularPipeline"
        workflow_id = "text2image"
        token = object()

        upsampled_state = FakeState(prompt=["structured caption"])
        upsample_pipeline = RecordingPipeline(upsampled_state)
        upsample = WorkflowIdeogram4PromptUpsample("ideogram4-upsample")
        with (
            mock.patch.object(upsample, "_prepare_pipeline", return_value=(token, upsample_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            upsampled = upsample.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="prompt_upsample",
                prompt="A precise typographic poster",
                prompt_upsampling=False,
                prompt_upsampling_temperature=1.0,
                width=2048,
                height=2048,
                max_sequence_length=2048,
                seed=61,
            )
        self.assertEqual(
            upsample_pipeline.calls,
            [
                {
                    "prompt": "A precise typographic poster",
                    "prompt_upsampling": False,
                    "prompt_upsampling_temperature": 1.0,
                    "width": 2048,
                    "height": 2048,
                    "max_sequence_length": 2048,
                    "generator": "generator",
                }
            ],
        )

        encoded_state = FakeState(text_features="qwen3-vl")
        text_pipeline = RecordingPipeline(encoded_state)
        text = WorkflowIdeogram4TextEncode("ideogram4-text")
        with mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)):
            encoded = text.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="text_encoder",
                state_in=upsampled["state_out"],
            )
        self.assertEqual(text_pipeline.calls, [{"state": upsampled_state}])

        denoised_state = FakeState(latents="ideogram4-latents")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise = WorkflowIdeogram4Denoise("ideogram4-denoise")
        schedule = [7.0, 7.0, 3.0]
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoised = denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="denoise",
                state_in=encoded["state_out"],
                width=1024,
                height=768,
                num_images_per_prompt=1,
                num_inference_steps=3,
                mu=0.0,
                std=1.5,
                guidance_schedule_json=json.dumps(schedule),
                seed=67,
            )
        self.assertEqual(denoise_pipeline.calls[0]["guidance_schedule"], schedule)
        self.assertEqual(denoise_pipeline.calls[0]["num_inference_steps"], 3)

        image = Image.new("RGB", (32, 32), "ivory")
        decode_pipeline = RecordingPipeline(FakeState(images=[image]))
        decode = WorkflowIdeogram4Decode("ideogram4-decode")
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
            decoded = decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="decode",
                state_in=denoised["state_out"],
            )
        self.assertEqual(decoded["images"], [image])
        with mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)):
            with self.assertRaisesRegex(ValueError, "one finite numeric value"):
                denoise.execute(
                    pipeline_components={},
                    pipeline_class=pipeline_class,
                    workflow_id=workflow_id,
                    block_path="denoise",
                    state_in=encoded["state_out"],
                    width=1024,
                    height=768,
                    num_inference_steps=3,
                    guidance_schedule_json="[7.0, 3.0]",
                )

    def test_cosmos_3_distilled_runs_exact_conditioned_and_unconditioned_routes(self):
        pipeline_class = "Cosmos3DistilledModularPipeline"
        token = object()

        encoded_state = FakeState(cond_input_ids="tokens")
        text_pipeline = RecordingPipeline(encoded_state)
        text = WorkflowCosmos3DistilledTextEncode("cosmos3-text")
        with mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)):
            encoded = text.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="video2video",
                block_path="text_encoder",
                prompt="A camera tracks a running robot",
                width=1280,
                height=720,
                num_frames=189,
                fps=24.0,
                use_system_prompt=True,
                add_resolution_template=True,
                add_duration_template=True,
            )
        self.assertEqual(
            text_pipeline.calls,
            [
                {
                    "prompt": "A camera tracks a running robot",
                    "width": 1280,
                    "height": 720,
                    "num_frames": 189,
                    "fps": 24.0,
                    "use_system_prompt": True,
                    "add_resolution_template": True,
                    "add_duration_template": True,
                }
            ],
        )

        frames = [Image.new("RGB", (32, 32), color) for color in ("red", "green", "blue")]
        conditioned_state = FakeState(x0_tokens_vision="conditioned")
        vae_pipeline = RecordingPipeline(conditioned_state)
        vae = WorkflowCosmos3DistilledVaeEncode("cosmos3-vae")
        with mock.patch.object(vae, "_prepare_pipeline", return_value=(token, vae_pipeline)):
            conditioned = vae.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="video2video",
                block_path="vae_encoder",
                state_in=encoded["state_out"],
                video=frames,
                condition_frame_indexes_vision="[0, 1]",
                condition_video_keep="last",
            )
        self.assertEqual(
            vae_pipeline.calls,
            [
                {
                    "state": encoded_state,
                    "video": frames,
                    "condition_frame_indexes_vision": [0, 1],
                    "condition_video_keep": "last",
                }
            ],
        )

        denoised_state = FakeState(latents="vision-latents")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise = WorkflowCosmos3DistilledDenoise("cosmos3-denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoised = denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="video2video",
                block_path="denoise",
                state_in=conditioned["state_out"],
                num_inference_steps=4,
                guidance_scale=1.0,
                seed=71,
            )
        self.assertEqual(
            denoise_pipeline.calls,
            [
                {
                    "state": conditioned_state,
                    "num_inference_steps": 4,
                    "guidance_scale": 1.0,
                    "generator": "generator",
                }
            ],
        )

        decoded_frames = [Image.new("RGB", (32, 32), "purple"), Image.new("RGB", (32, 32), "orange")]
        decode_pipeline = RecordingPipeline(FakeState(videos=decoded_frames))
        decode = WorkflowCosmos3DistilledDecode("cosmos3-decode")
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
            output = decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="video2video",
                block_path="decode",
                state_in=denoised["state_out"],
            )
        self.assertEqual(decode_pipeline.calls, [{"state": denoised_state, "output_type": "pil"}])
        self.assertEqual(output["video"], decoded_frames)

        still_state = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id="text2image",
            completed_stage="denoise",
            state=denoised_state,
        )
        still = Image.new("RGB", (32, 32), "cyan")
        still_pipeline = RecordingPipeline(FakeState(videos=[still]))
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, still_pipeline)):
            output = decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="text2image",
                block_path="decode",
                state_in=still_state,
            )
        self.assertEqual(output["image"], still)

    def test_cosmos_3_omni_runs_exact_action_and_sound_branches(self):
        pipeline_class = "Cosmos3OmniModularPipeline"
        token = object()
        reference = Image.new("RGB", (640, 480), "teal")

        encoded_state = FakeState(action="condition")
        text_pipeline = RecordingPipeline(encoded_state)
        text = WorkflowCosmos3OmniTextEncode("cosmos3-omni-text")
        action_condition = SimpleNamespace(
            mode="policy",
            chunk_size=16,
            domain_name="droid_lerobot",
            image=reference,
        )
        with (
            mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)),
            mock.patch.object(text, "_action_condition", return_value=action_condition),
        ):
            encoded = text.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="action_policy",
                block_path="text_encoder",
                prompt="A robot stacks colored blocks",
                negative_prompt="blur",
                image=reference,
                fps=24.0,
                use_system_prompt=True,
                add_resolution_template=True,
                add_duration_template=True,
                action_chunk_size=16,
                action_domain_name="droid_lerobot",
                action_resolution_tier=480,
                action_view_point="ego_view",
            )
        action = text_pipeline.calls[0]["action"]
        self.assertEqual(action.mode, "policy")
        self.assertEqual(action.chunk_size, 16)
        self.assertEqual(action.domain_name, "droid_lerobot")
        self.assertIs(action.image, reference)
        self.assertNotIn("width", text_pipeline.calls[0])

        vae_state = FakeState(action_latents="conditioned")
        vae_pipeline = RecordingPipeline(vae_state)
        vae = WorkflowCosmos3OmniVaeEncode("cosmos3-omni-vae")
        with mock.patch.object(vae, "_prepare_pipeline", return_value=(token, vae_pipeline)):
            conditioned = vae.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="action_policy",
                block_path="vae_encoder",
                state_in=encoded["state_out"],
            )
        self.assertEqual(vae_pipeline.calls, [{"state": encoded_state}])

        denoised_state = FakeState(latents="vision", action_latents="actions")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise = WorkflowCosmos3OmniDenoise("cosmos3-omni-denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoised = denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="action_policy",
                block_path="denoise",
                state_in=conditioned["state_out"],
                num_inference_steps=50,
                guidance_scale=6.0,
                seed=73,
            )
        self.assertEqual(denoise_pipeline.calls[0]["enable_sound"], False)
        self.assertEqual(denoise_pipeline.calls[0]["generator"], "generator")

        frames = [Image.new("RGB", (32, 32), "teal"), Image.new("RGB", (32, 32), "navy")]
        decoded_state = FakeState(videos=frames, action_latents="actions")
        decode_pipeline = RecordingPipeline(decoded_state)
        decode = WorkflowCosmos3OmniDecode("cosmos3-omni-decode")
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
            decoded = decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="action_policy",
                block_path="decode",
                state_in=denoised["state_out"],
            )
        self.assertEqual(decoded["videos"], frames)

        action_state = FakeState(action=["predicted-actions"])
        after_pipeline = RecordingPipeline(action_state)
        after = WorkflowCosmos3OmniAfterDecode("cosmos3-omni-after")
        with mock.patch.object(after, "_prepare_pipeline", return_value=(token, after_pipeline)):
            output = after.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="action_policy",
                block_path="after_decode",
                state_in=decoded["state_out"],
            )
        self.assertEqual(after_pipeline.calls, [{"state": decoded_state}])
        self.assertEqual(output["action"], ["predicted-actions"])

        sound_text_state = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id="text2video_with_sound",
            completed_stage="text_encoder",
            state=encoded_state,
        )
        sound_pipeline = RecordingPipeline(denoised_state)
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, sound_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="sound-generator",
            ),
        ):
            denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="text2video_with_sound",
                block_path="denoise",
                state_in=sound_text_state,
                num_inference_steps=35,
                guidance_scale=6.0,
                seed=79,
            )
        self.assertEqual(sound_pipeline.calls[0]["enable_sound"], True)

    def test_cosmos_3_nano_runs_exact_plain_text_workflows_and_propagates_safety_failures(self):
        pipeline_class = "Cosmos3OmniModularPipeline"
        prompt_value = "A small warehouse robot moves a blue box across a clean floor."
        for workflow_id, num_frames in (("text2image", 1), ("text2video", 189)):
            with self.subTest(workflow_id=workflow_id):
                token = object()
                encoded_state = FakeState(cond_input_ids="safe-tokens")
                text_pipeline = RecordingPipeline(encoded_state)
                text = WorkflowCosmos3OmniTextEncode(f"cosmos3-nano-{workflow_id}-text")
                with mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)):
                    encoded = text.execute(
                        pipeline_components={},
                        pipeline_class=pipeline_class,
                        workflow_id=workflow_id,
                        block_path="text_encoder",
                        prompt=prompt_value,
                        negative_prompt="",
                        width=1280,
                        height=720,
                        num_frames=num_frames,
                        fps=24.0,
                        use_system_prompt=True,
                        add_resolution_template=True,
                        add_duration_template=True,
                    )
                self.assertEqual(
                    text_pipeline.calls,
                    [
                        {
                            "prompt": prompt_value,
                            "negative_prompt": "",
                            "fps": 24.0,
                            "use_system_prompt": True,
                            "add_resolution_template": True,
                            "add_duration_template": True,
                            "width": 1280,
                            "height": 720,
                            "num_frames": num_frames,
                        }
                    ],
                )

                denoised_state = FakeState(latents="safe-vision-latents")
                denoise_pipeline = RecordingPipeline(denoised_state)
                denoise = WorkflowCosmos3OmniDenoise(f"cosmos3-nano-{workflow_id}-denoise")
                with (
                    mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
                    mock.patch(
                        "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                        return_value="generator-123",
                    ),
                ):
                    denoised = denoise.execute(
                        pipeline_components={},
                        pipeline_class=pipeline_class,
                        workflow_id=workflow_id,
                        block_path="denoise",
                        state_in=encoded["state_out"],
                        num_inference_steps=35,
                        guidance_scale=6.0,
                        seed=123,
                    )
                self.assertEqual(
                    denoise_pipeline.calls,
                    [
                        {
                            "state": encoded_state,
                            "num_inference_steps": 35,
                            "guidance_scale": 6.0,
                            "enable_sound": False,
                            "generator": "generator-123",
                        }
                    ],
                )

                frames = [Image.new("RGB", (32, 32), "blue")]
                if workflow_id == "text2video":
                    frames.append(Image.new("RGB", (32, 32), "green"))
                decoded_state = FakeState(videos=frames)
                decode_pipeline = RecordingPipeline(decoded_state)
                decode = WorkflowCosmos3OmniDecode(f"cosmos3-nano-{workflow_id}-decode")
                with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
                    decoded = decode.execute(
                        pipeline_components={},
                        pipeline_class=pipeline_class,
                        workflow_id=workflow_id,
                        block_path="decode",
                        state_in=denoised["state_out"],
                    )
                self.assertEqual(decode_pipeline.calls, [{"state": denoised_state, "output_type": "pil"}])
                declared_decode_outputs = {
                    name
                    for name, parameter in WorkflowCosmos3OmniDecode.params.items()
                    if parameter.get("display") == "output"
                }
                self.assertLessEqual(set(decoded), declared_decode_outputs)
                if workflow_id == "text2image":
                    self.assertIs(decoded["image"], frames[0])
                else:
                    self.assertEqual(decoded["videos"], frames)

                after_pipeline = RecordingPipeline(FakeState(action=None))
                after = WorkflowCosmos3OmniAfterDecode(f"cosmos3-nano-{workflow_id}-after")
                with mock.patch.object(after, "_prepare_pipeline", return_value=(token, after_pipeline)):
                    after_result = after.execute(
                        pipeline_components={},
                        pipeline_class=pipeline_class,
                        workflow_id=workflow_id,
                        block_path="after_decode",
                        state_in=decoded["state_out"],
                    )
                self.assertEqual(after_pipeline.calls, [{"state": decoded_state}])
                self.assertIsNone(after_result["action"])

        text = WorkflowCosmos3OmniTextEncode("cosmos3-nano-safety-failure")
        unsafe_pipeline = mock.Mock(
            side_effect=ValueError("Cosmos3 requires a safety checker by default."),
        )
        with (
            mock.patch.object(text, "_prepare_pipeline", return_value=(object(), unsafe_pipeline)),
            self.assertRaisesRegex(ValueError, "requires a safety checker"),
        ):
            text.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="text2video",
                block_path="text_encoder",
                prompt=prompt_value,
                num_frames=189,
                width=1280,
                height=720,
                fps=24.0,
            )

    def test_minimax_h3_reference_assembler_preserves_typed_semantic_order(self):
        reference_types = (
            FakeMiniMaxH3Reference,
            FakeMiniMaxH3ImageReference,
            FakeMiniMaxH3VideoReference,
            FakeMiniMaxH3AudioReference,
        )
        assembler = WorkflowMiniMaxH3ReferenceAssembler("minimax-h3-reference")
        image = Image.new("RGB", (16, 16), "orange")
        frames = [Image.new("RGB", (16, 16), "navy") for _ in range(2)]
        waveform = torch.zeros((1, 48), dtype=torch.float32).numpy()
        with (
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks._minimax_h3_reference_types",
                return_value=reference_types,
            ),
            mock.patch(
                "modules.DiffusersAudio.main.audio_to_numpy",
                return_value=(waveform, 48_000),
            ),
        ):
            image_result = assembler.execute(reference_kind="image", image=image)
            video_result = assembler.execute(
                references_in=image_result["references"],
                reference_kind="video",
                video=frames,
                video_fps=30.0,
                audio={"decoded": True},
            )
            result = assembler.execute(
                references_in=video_result["references"],
                reference_kind="audio",
                audio={"decoded": True},
            )

            self.assertEqual([reference.kind for reference in result["references"]], ["image", "video", "audio"])
            self.assertIs(result["references"][0].image, image)
            self.assertEqual(result["references"][1].frames, frames)
            self.assertEqual(result["references"][1].fps, 30.0)
            self.assertEqual(result["references"][1].sample_rate, 48_000)
            self.assertEqual(result["references"][2].sample_rate, 48_000)
            with self.assertRaisesRegex(ValueError, "paired with an image or video"):
                _validated_minimax_h3_references(
                    [FakeMiniMaxH3AudioReference(audio=torch.zeros((1, 1)), sample_rate=48_000)],
                    allow_audio_only=False,
                )

    def test_minimax_h3_runs_exact_first_last_frame_joint_video_audio_route(self):
        pipeline_class = "MiniMaxH3ModularPipeline"
        workflow_id = "fl2va"
        token = object()
        first = Image.new("RGB", (640, 480), "red")
        last = Image.new("RGB", (640, 480), "blue")

        prepared_state = FakeState(keyframes=[first, last])
        before_pipeline = RecordingPipeline(prepared_state)
        before = WorkflowMiniMaxH3BeforeEncode("minimax-h3-before")
        with mock.patch.object(before, "_prepare_pipeline", return_value=(token, before_pipeline)):
            prepared = before.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="before_encode",
                image=first,
                last_image=last,
                width=1344,
                height=768,
                num_frames=124,
            )
        self.assertEqual(
            before_pipeline.calls,
            [{"image": first, "last_image": last, "width": 1344, "height": 768, "num_frames": 124}],
        )

        encoded_state = FakeState(prompt_embeds="qwen3-vl")
        text_pipeline = RecordingPipeline(encoded_state)
        text = WorkflowMiniMaxH3TextEncode("minimax-h3-text")
        with mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)):
            encoded = text.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="text_encoder",
                state_in=prepared["state_out"],
                prompt="A cinematic transition from sunrise to night",
            )
        self.assertEqual(
            text_pipeline.calls,
            [{"state": prepared_state, "prompt": "A cinematic transition from sunrise to night"}],
        )

        conditioned_state = FakeState(condition_latents="keyframes")
        vae_pipeline = RecordingPipeline(conditioned_state)
        vae = WorkflowMiniMaxH3VaeEncode("minimax-h3-vae")
        with mock.patch.object(vae, "_prepare_pipeline", return_value=(token, vae_pipeline)):
            conditioned = vae.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="vae_encoder",
                state_in=encoded["state_out"],
            )
        self.assertEqual(vae_pipeline.calls, [{"state": encoded_state}])

        denoised_state = FakeState(latents="video", audio_latents="audio")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise = WorkflowMiniMaxH3Denoise("minimax-h3-denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            denoised = denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="denoise",
                state_in=conditioned["state_out"],
                num_inference_steps=50,
                seed=83,
            )
        self.assertEqual(
            denoise_pipeline.calls,
            [{"state": conditioned_state, "num_inference_steps": 50, "generator": "generator"}],
        )

        frames = [Image.new("RGB", (32, 32), "red"), Image.new("RGB", (32, 32), "blue")]
        decode_pipeline = RecordingPipeline(FakeState(videos=[frames], audio="waveform", sampling_rate=48_000))
        decode = WorkflowMiniMaxH3Decode("minimax-h3-decode")
        with (
            mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)),
            mock.patch(
                "modules.DiffusersAudio.main.output_to_audio_object",
                return_value={"array": "waveform", "sampling_rate": 48_000},
            ),
        ):
            output = decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="decode",
                state_in=denoised["state_out"],
            )
        self.assertEqual(output["video"], frames)
        self.assertEqual(output["sample_rate"], 48_000)

    def test_ltx25_runs_exact_in_context_diffusion_decoder_route(self):
        pipeline_class = "LTX25ModularPipeline"
        workflow_id = "in_context"
        token = object()
        condition_frame = Image.new("RGB", (32, 32), "gold")
        reference_frames = [Image.new("RGB", (32, 32), "silver")]
        conditions = [FakeLTX2VideoCondition(frames=condition_frame, index=0, strength=0.8, crf=0)]
        references = [FakeLTX2ReferenceCondition(frames=reference_frames, strength=1.0)]
        attention_mask = torch.ones((1, 1, 1, 32, 32), dtype=torch.float32)
        condition_modules = {
            "diffusers.pipelines.ltx2.pipeline_ltx2_condition": SimpleNamespace(
                LTX2VideoCondition=FakeLTX2VideoCondition
            ),
            "diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora": SimpleNamespace(
                LTX2ReferenceCondition=FakeLTX2ReferenceCondition
            ),
        }

        encoded_state = FakeState(connector_prompt_embeds="text")
        text_pipeline = RecordingPipeline(encoded_state)
        text = WorkflowLTX25TextEncode("ltx25-text")
        with mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)):
            encoded = text.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="text_encoder",
                prompt="A paper kite circles above a coastal cliff",
                negative_prompt="blur",
                max_sequence_length=1024,
            )
        self.assertEqual(
            text_pipeline.calls,
            [
                {
                    "prompt": "A paper kite circles above a coastal cliff",
                    "negative_prompt": "blur",
                    "max_sequence_length": 1024,
                }
            ],
        )

        condition_state = FakeState(condition_latents=[])
        condition_pipeline = RecordingPipeline(condition_state)
        condition = WorkflowLTX25ConditionEncode("ltx25-condition")
        with (
            mock.patch.object(condition, "_prepare_pipeline", return_value=(token, condition_pipeline)),
            mock.patch.dict(sys.modules, condition_modules),
        ):
            conditioned = condition.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="condition_encoder",
                state_in=encoded["state_out"],
                conditions=conditions,
                reference_conditions=references,
                width=704,
                height=512,
                num_frames=121,
                seed=89,
            )
        self.assertIs(condition_pipeline.calls[0]["conditions"], conditions)
        self.assertEqual(condition_pipeline.calls[0]["reference_conditions"], references)
        self.assertEqual(condition_pipeline.calls[0]["num_frames"], 121)
        generator = condition_pipeline.calls[0]["generator"]
        self.assertIsInstance(generator, torch.Generator)
        self.assertEqual(generator.initial_seed(), 89)

        reference_state = FakeState(reference_latents="tokens")
        reference_pipeline = RecordingPipeline(reference_state)
        reference = WorkflowLTX25ReferenceEncode("ltx25-reference")
        with (
            mock.patch.object(reference, "_prepare_pipeline", return_value=(token, reference_pipeline)),
            mock.patch.dict(sys.modules, condition_modules),
        ):
            referenced = reference.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="reference_encoder",
                state_in=conditioned["state_out"],
                reference_conditions=references,
                reference_downscale_factor=2,
                conditioning_attention_strength=0.75,
                conditioning_attention_mask=attention_mask,
                frame_rate=24.0,
                seed=89,
            )
        self.assertEqual(reference_pipeline.calls[0]["reference_downscale_factor"], 2)
        self.assertEqual(reference_pipeline.calls[0]["conditioning_attention_strength"], 0.75)
        self.assertIs(reference_pipeline.calls[0]["conditioning_attention_mask"], attention_mask)
        self.assertIs(reference_pipeline.calls[0]["generator"], generator)

        denoised_state = FakeState(latents="video", audio_latents="audio")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise = WorkflowLTX25Denoise("ltx25-denoise")
        with mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)):
            denoised = denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="denoise",
                state_in=referenced["state_out"],
                width=704,
                height=512,
                frame_rate=24.0,
                num_videos_per_prompt=1,
                noise_scale=1.0,
                use_cross_timestep=True,
                seed=89,
            )
        self.assertNotIn("width", denoise_pipeline.calls[0])
        self.assertNotIn("num_frames", denoise_pipeline.calls[0])
        self.assertNotIn("num_inference_steps", denoise_pipeline.calls[0])
        self.assertEqual(
            denoise_pipeline.calls[0]["sigmas"],
            [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875],
        )
        self.assertIs(denoise_pipeline.calls[0]["generator"], generator)
        self.assertEqual(denoise_pipeline.guider.guidance_scale, 1.0)
        self.assertEqual(denoise_pipeline.guider.stg_scale, 0.0)
        self.assertEqual(denoise_pipeline.guider.modality_scale, 1.0)
        self.assertEqual(denoise_pipeline.audio_guider.guidance_scale, 1.0)

        frames = [Image.new("RGB", (32, 32), "skyblue"), Image.new("RGB", (32, 32), "navy")]
        decode_pipeline = RecordingPipeline(FakeState(videos=[frames], audio="waveform"))
        decode_pipeline.audio_sampling_rate = 16_000
        decode_pipeline.vocoder = SimpleNamespace(config={"output_sampling_rate": 48_000})
        decode = WorkflowLTX25Decode("ltx25-decode")
        with (
            mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)),
            mock.patch("modules.ModularDiffusers.workflow_blocks.configure_ltx25_diffusion_decoder") as configure,
            mock.patch(
                "modules.DiffusersAudio.main.output_to_audio_object",
                return_value={"array": "waveform", "sampling_rate": 48_000},
            ) as convert_audio,
        ):
            output = decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="decode",
                state_in=denoised["state_out"],
                seed=89,
            )
        self.assertEqual(decode_pipeline.calls[0]["output_type"], "pil")
        self.assertIs(decode_pipeline.calls[0]["generator"], generator)
        configure.assert_called_once_with(decode_pipeline)
        self.assertEqual(output["video"], frames)
        self.assertEqual(output["sample_rate"], 48_000)
        convert_audio.assert_called_once_with("waveform", sample_rate=48_000)

    def test_ltx25_denoise_preserves_null_noise_scale_for_text_and_image_workflows(self):
        pipeline_class = "LTX25ModularPipeline"
        self.assertIsNone(WorkflowLTX25Denoise.params["noise_scale"]["default"])

        for workflow_id, preceding_stage in (("text2video", "duration"), ("image2video", "vae_encoder")):
            with self.subTest(workflow_id=workflow_id):
                token = object()
                preceding_state = FakeState(source=workflow_id)
                state_in = _issue_workflow_state(
                    token=token,
                    pipeline_class=pipeline_class,
                    workflow_id=workflow_id,
                    completed_stage=preceding_stage,
                    state=preceding_state,
                )
                denoise_pipeline = RecordingPipeline(FakeState(latents="video", audio_latents="audio"))
                denoise = WorkflowLTX25Denoise(f"ltx25-{workflow_id}-denoise")
                with mock.patch.object(
                    denoise,
                    "_prepare_pipeline",
                    return_value=(token, denoise_pipeline),
                ):
                    denoise.execute(
                        pipeline_components={},
                        pipeline_class=pipeline_class,
                        workflow_id=workflow_id,
                        block_path="denoise",
                        state_in=state_in,
                        width=704,
                        height=512,
                        frame_rate=24.0,
                        num_videos_per_prompt=1,
                        use_cross_timestep=True,
                        seed=91,
                    )

                self.assertIsNone(denoise_pipeline.calls[0]["noise_scale"])
                self.assertNotIn("num_inference_steps", denoise_pipeline.calls[0])
                self.assertEqual(len(denoise_pipeline.calls[0]["sigmas"]), 8)

    def test_ltx25_image_encode_and_denoise_continue_one_generator(self):
        pipeline_class = "LTX25ModularPipeline"
        workflow_id = "image2video"
        token = object()
        duration_state = FakeState(num_frames=121)
        duration_output = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage="duration",
            state=duration_state,
        )
        vae_pipeline = RecordingPipeline(FakeState(image_latents="reference"))
        vae = WorkflowLTX25VaeEncode("ltx25-image-generator")
        with mock.patch.object(vae, "_prepare_pipeline", return_value=(token, vae_pipeline)):
            encoded = vae.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="vae_encoder",
                state_in=duration_output,
                image=Image.new("RGB", (32, 32), "green"),
                width=704,
                height=512,
                image_crf=18,
                seed=97,
            )
        generator = vae_pipeline.calls[0]["generator"]

        denoise_pipeline = RecordingPipeline(FakeState(latents="video", audio_latents="audio"))
        denoise = WorkflowLTX25Denoise("ltx25-image-denoise-generator")
        with mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)):
            denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="denoise",
                state_in=encoded["state_out"],
                width=704,
                height=512,
                frame_rate=24.0,
                num_videos_per_prompt=1,
                use_cross_timestep=True,
                seed=97,
            )

        self.assertIs(denoise_pipeline.calls[0]["generator"], generator)

    def test_ltx25_denoise_rejects_step_count_substitution(self):
        pipeline_class = "LTX25ModularPipeline"
        workflow_id = "text2video"
        token = object()
        state_in = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage="duration",
            state=FakeState(num_frames=121),
        )
        pipeline = RecordingPipeline(FakeState(latents="video", audio_latents="audio"))
        denoise = WorkflowLTX25Denoise("ltx25-no-step-substitution")

        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, pipeline)),
            self.assertRaisesRegex(ValueError, "does not accept a num_inference_steps substitute"),
        ):
            denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="denoise",
                state_in=state_in,
                width=704,
                height=512,
                num_inference_steps=8,
                seed=97,
            )

        self.assertEqual(pipeline.calls, [])

    def test_ltx25_condition_ports_reject_untyped_values_and_malformed_masks(self):
        pipeline_class = "LTX25ModularPipeline"
        workflow_id = "in_context"
        token = object()
        reference = FakeLTX2ReferenceCondition(
            frames=[Image.new("RGB", (16, 16), "purple")],
            strength=1.0,
        )
        condition_modules = {
            "diffusers.pipelines.ltx2.pipeline_ltx2_condition": SimpleNamespace(
                LTX2VideoCondition=FakeLTX2VideoCondition
            ),
            "diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora": SimpleNamespace(
                LTX2ReferenceCondition=FakeLTX2ReferenceCondition
            ),
        }
        encoded_state = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage="text_encoder",
            state=FakeState(connector_prompt_embeds="text"),
        )
        condition_pipeline = RecordingPipeline(FakeState(condition_latents=[]))
        condition = WorkflowLTX25ConditionEncode("ltx25-untyped-condition")
        with (
            mock.patch.object(condition, "_prepare_pipeline", return_value=(token, condition_pipeline)),
            mock.patch.dict(sys.modules, condition_modules),
            self.assertRaisesRegex(TypeError, "exact Diffusers LTX2VideoCondition"),
        ):
            condition.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="condition_encoder",
                state_in=encoded_state,
                conditions=[SimpleNamespace(frames=object())],
                reference_conditions=[reference],
                width=704,
                height=512,
                num_frames=121,
                seed=101,
            )
        self.assertEqual(condition_pipeline.calls, [])

        condition_state = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage="condition_encoder",
            state=FakeState(
                condition_latents=[],
                generator=torch.Generator(device="cpu").manual_seed(101),
            ),
        )
        reference_pipeline = RecordingPipeline(FakeState(reference_latents="tokens"))
        reference_node = WorkflowLTX25ReferenceEncode("ltx25-malformed-mask")
        with (
            mock.patch.object(reference_node, "_prepare_pipeline", return_value=(token, reference_pipeline)),
            mock.patch.dict(sys.modules, condition_modules),
            self.assertRaisesRegex(ValueError, r"shaped \(1, 1, F, H, W\)"),
        ):
            reference_node.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="reference_encoder",
                state_in=condition_state,
                reference_conditions=[reference],
                conditioning_attention_mask=torch.ones((2, 1, 1, 16, 16)),
                seed=101,
            )
        self.assertEqual(reference_pipeline.calls, [])

    def test_ltx25_reference_conditions_accept_only_one_official_batched_video(self):
        pipeline_class = "LTX25ModularPipeline"
        workflow_id = "in_context"
        token = object()
        condition_modules = {
            "diffusers.pipelines.ltx2.pipeline_ltx2_condition": SimpleNamespace(
                LTX2VideoCondition=FakeLTX2VideoCondition
            ),
            "diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora": SimpleNamespace(
                LTX2ReferenceCondition=FakeLTX2ReferenceCondition
            ),
        }
        generator = torch.Generator(device="cpu").manual_seed(103)
        condition_state = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage="condition_encoder",
            state=FakeState(condition_latents=[], generator=generator),
        )
        reference_pipeline = RecordingPipeline(FakeState(reference_latents="tokens"))
        reference_node = WorkflowLTX25ReferenceEncode("ltx25-batched-reference")
        reference = FakeLTX2ReferenceCondition(
            frames=torch.rand((1, 5, 3, 16, 16), generator=torch.Generator(device="cpu").manual_seed(7)),
            strength=1.0,
        )
        with (
            mock.patch.object(reference_node, "_prepare_pipeline", return_value=(token, reference_pipeline)),
            mock.patch.dict(sys.modules, condition_modules),
        ):
            reference_node.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="reference_encoder",
                state_in=condition_state,
                reference_conditions=[reference],
                seed=103,
            )
        self.assertIs(reference_pipeline.calls[0]["reference_conditions"][0], reference)
        self.assertIs(reference_pipeline.calls[0]["generator"], generator)

        reference_pipeline.calls.clear()
        reference.frames = torch.zeros((2, 5, 3, 16, 16))
        with (
            mock.patch.object(reference_node, "_prepare_pipeline", return_value=(token, reference_pipeline)),
            mock.patch.dict(sys.modules, condition_modules),
            self.assertRaisesRegex(ValueError, "exactly one batched video"),
        ):
            reference_node.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="reference_encoder",
                state_in=condition_state,
                reference_conditions=[reference],
                seed=103,
            )
        self.assertEqual(reference_pipeline.calls, [])

    def test_ltx25_decode_rejects_a_missing_vocoder_output_rate_instead_of_falling_back(self):
        pipeline_class = "LTX25ModularPipeline"
        workflow_id = "text2video"
        token = object()
        denoised_state = FakeState(latents="video", audio_latents="audio")
        state_in = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id=workflow_id,
            completed_stage="denoise",
            state=denoised_state,
        )
        frames = [Image.new("RGB", (32, 32), "skyblue")]
        decode_pipeline = RecordingPipeline(FakeState(videos=[frames], audio="waveform"))
        decode_pipeline.audio_sampling_rate = 16_000
        decode_pipeline.vocoder = SimpleNamespace(config={})
        decode = WorkflowLTX25Decode("ltx25-decode-missing-rate")

        with (
            mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)),
            mock.patch("modules.ModularDiffusers.workflow_blocks.configure_ltx25_diffusion_decoder"),
            mock.patch("modules.DiffusersAudio.main.output_to_audio_object") as convert_audio,
            self.assertRaisesRegex(ValueError, "output_sampling_rate"),
        ):
            decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id=workflow_id,
                block_path="decode",
                state_in=state_in,
                seed=91,
            )
        convert_audio.assert_not_called()

    def test_wan_animate_2_runs_the_exact_six_stage_media_and_state_route(self):
        pipeline_class = "WanAnimate2ModularPipeline"
        token = object()

        text_pipeline = RecordingPipeline(FakeState(prompt_embeds="text"))
        text = WorkflowWanAnimateTextEncode("wan-text")
        with mock.patch.object(text, "_prepare_pipeline", return_value=(token, text_pipeline)):
            encoded_text = text.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="default",
                block_path="text_encoder",
                prompt="A dancer in a red jacket",
                negative_prompt="blur",
                prompt_ref="人物动作的参考视频",
                max_sequence_length=512,
            )
        self.assertEqual(
            text_pipeline.calls,
            [
                {
                    "prompt": "A dancer in a red jacket",
                    "negative_prompt": "blur",
                    "prompt_ref": "人物动作的参考视频",
                    "max_sequence_length": 512,
                }
            ],
        )

        source_image = Image.new("RGB", (320, 480), "red")
        image_state = FakeState(image_pixels="reference")
        image_pipeline = RecordingPipeline(image_state)
        image_encode = WorkflowWanAnimateImageEncode("wan-image")
        with mock.patch.object(image_encode, "_prepare_pipeline", return_value=(token, image_pipeline)):
            encoded_image = image_encode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="default",
                block_path="image_encoder",
                state_in=encoded_text["state_out"],
                image=source_image,
                width=640,
                height=800,
            )
        self.assertEqual(image_pipeline.calls[0]["image"], source_image)
        self.assertEqual((image_pipeline.calls[0]["width"], image_pipeline.calls[0]["height"]), (640, 800))

        driving_video = [source_image.copy() for _ in range(5)]
        video_state = FakeState(driving_video_pixels="driving")
        video_pipeline = RecordingPipeline(video_state)
        video_encode = WorkflowWanAnimateVideoEncode("wan-video")
        with mock.patch.object(video_encode, "_prepare_pipeline", return_value=(token, video_pipeline)):
            encoded_video = video_encode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="default",
                block_path="video_encoder",
                state_in=encoded_image["state_out"],
                driving_video=driving_video,
                driving_video_fps=30.0,
                fps=24,
                segment_frame_length=81,
                prev_segment_conditioning_frames=1,
            )
        self.assertEqual(
            video_pipeline.calls,
            [
                {
                    "state": image_state,
                    "driving_video": driving_video,
                    "driving_video_fps": 30.0,
                    "fps": 24,
                    "segment_frame_length": 81,
                    "prev_segment_conditioning_frames": 1,
                }
            ],
        )

        vae_state = FakeState(reference_image_latents="latents")
        vae_pipeline = RecordingPipeline(vae_state)
        vae_encode = WorkflowWanAnimateVaeEncode("wan-vae")
        with mock.patch.object(vae_encode, "_prepare_pipeline", return_value=(token, vae_pipeline)):
            encoded_vae = vae_encode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="default",
                block_path="vae_encoder",
                state_in=encoded_video["state_out"],
            )
        self.assertEqual(vae_pipeline.calls, [{"state": video_state}])

        denoised_state = FakeState(segment_frames="segments")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise_pipeline.guider = FakeGuider(3.0)
        denoise = WorkflowWanAnimateDenoise("wan-denoise")
        with (
            mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ) as generator,
        ):
            denoised = denoise.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="default",
                block_path="denoise",
                state_in=encoded_vae["state_out"],
                num_inference_steps=40,
                guidance_scale=3.0,
                seed=29,
            )
        generator.assert_called_once_with(29, denoise_pipeline)
        self.assertEqual(denoise_pipeline.guider.guidance_scale, 3.0)
        self.assertEqual(
            denoise_pipeline.calls,
            [{"state": vae_state, "num_inference_steps": 40, "generator": "generator"}],
        )

        frames = [Image.new("RGB", (16, 16), "blue") for _ in range(3)]
        decode_pipeline = RecordingPipeline(FakeState(videos=[frames]))
        decode = WorkflowWanAnimateDecode("wan-decode")
        with mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)):
            output = decode.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="default",
                block_path="decode",
                state_in=denoised["state_out"],
            )
        self.assertEqual(output["video"], frames)
        self.assertEqual(decode_pipeline.calls, [{"state": denoised_state, "output_type": "pil"}])

    def test_wan_animate_2_distilled_enforces_official_ten_steps_and_guidance_one(self):
        pipeline_class = "WanAnimate2DistilledModularPipeline"
        token = object()
        source_state = FakeState(reference_image_latents="latents")
        sealed = _issue_workflow_state(
            token=token,
            pipeline_class=pipeline_class,
            workflow_id="default",
            completed_stage="vae_encoder",
            state=source_state,
        )
        pipeline = RecordingPipeline(FakeState(segment_frames=[]))
        pipeline.guider = FakeGuider(1.0)
        node = WorkflowWanAnimateDenoise("wan-distilled")
        with (
            mock.patch.object(node, "_prepare_pipeline", return_value=(token, pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ),
        ):
            node.execute(
                pipeline_components={},
                pipeline_class=pipeline_class,
                workflow_id="default",
                block_path="denoise",
                state_in=sealed,
                num_inference_steps=10,
                guidance_scale=1.0,
                seed=4,
            )
        self.assertEqual(pipeline.calls[0]["num_inference_steps"], 10)
        self.assertEqual(pipeline.guider.guidance_scale, 1.0)
        with mock.patch.object(node, "_prepare_pipeline", return_value=(token, pipeline)):
            with self.assertRaisesRegex(ValueError, "fixed guidance scale"):
                node.execute(
                    pipeline_components={},
                    pipeline_class=pipeline_class,
                    workflow_id="default",
                    block_path="denoise",
                    state_in=sealed,
                    num_inference_steps=10,
                    guidance_scale=2.0,
                )

    def test_semantic_generation_constructs_generator_and_issues_sealed_state(self):
        token = object()
        pipeline_state = FakeState(frame_hiddens="frames")
        pipeline = RecordingPipeline(pipeline_state)
        node = WorkflowSemanticGeneration("semantic")
        with (
            mock.patch.object(node, "_prepare_pipeline", return_value=(token, pipeline)),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.modular_generator_from_seed",
                return_value="generator",
            ) as generator,
        ):
            output = node.execute(
                pipeline_components={},
                pipeline_class=PIPELINE_CLASS,
                workflow_id=WORKFLOW_ID,
                block_path="semantic_generator",
                prompt="Genre: acoustic pop.",
                lyrics="[verse]\nMorning light",
                audio_duration=12.0,
                seed=7,
            )

        generator.assert_called_once_with(7, pipeline)
        self.assertEqual(
            pipeline.calls,
            [
                {
                    "prompt": "Genre: acoustic pop.",
                    "lyrics": "[verse]\nMorning light",
                    "audio_duration": 12.0,
                    "generator": "generator",
                }
            ],
        )
        state = output["state_out"]
        self.assertIs(copy.copy(state), state)
        self.assertIs(copy.deepcopy(state), state)
        with self.assertRaisesRegex(TypeError, "cannot be serialized"):
            pickle.dumps(state)
        self.assertIs(
            _require_workflow_state(
                state,
                token=token,
                pipeline_class=PIPELINE_CLASS,
                workflow_id=WORKFLOW_ID,
                completed_stage="semantic_generator",
            ),
            pipeline_state,
        )

    def test_denoise_requires_exact_preceding_stage_and_loader_execution(self):
        first_token = object()
        second_token = object()
        state = _issue_workflow_state(
            token=first_token,
            pipeline_class=PIPELINE_CLASS,
            workflow_id=WORKFLOW_ID,
            completed_stage="semantic_generator",
            state=FakeState(frame_hiddens="frames"),
        )
        pipeline = RecordingPipeline(FakeState(latent_chunks="chunks"))
        node = WorkflowDenoise("denoise")
        with mock.patch.object(node, "_prepare_pipeline", return_value=(second_token, pipeline)):
            with self.assertRaisesRegex(ValueError, "different Models Loader"):
                node.execute(
                    pipeline_components={},
                    pipeline_class=PIPELINE_CLASS,
                    workflow_id=WORKFLOW_ID,
                    block_path="denoise",
                    state_in=state,
                    num_inference_steps=30,
                )
        self.assertEqual(pipeline.calls, [])

        wrong_stage = _issue_workflow_state(
            token=first_token,
            pipeline_class=PIPELINE_CLASS,
            workflow_id=WORKFLOW_ID,
            completed_stage="denoise",
            state=FakeState(),
        )
        with mock.patch.object(node, "_prepare_pipeline", return_value=(first_token, pipeline)):
            with self.assertRaisesRegex(ValueError, "not required stage"):
                node.execute(
                    pipeline_components={},
                    pipeline_class=PIPELINE_CLASS,
                    workflow_id=WORKFLOW_ID,
                    block_path="denoise",
                    state_in=wrong_stage,
                    num_inference_steps=30,
                )

    def test_denoise_and_decode_call_only_their_official_block_pipeline(self):
        token = object()
        semantic_state = FakeState(frame_hiddens="frames", generator="generator")
        issued_semantic = _issue_workflow_state(
            token=token,
            pipeline_class=PIPELINE_CLASS,
            workflow_id=WORKFLOW_ID,
            completed_stage="semantic_generator",
            state=semantic_state,
        )
        denoised_state = FakeState(latent_chunks="chunks")
        denoise_pipeline = RecordingPipeline(denoised_state)
        denoise = WorkflowDenoise("denoise")
        with mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)):
            denoise_output = denoise.execute(
                pipeline_components={},
                pipeline_class=PIPELINE_CLASS,
                workflow_id=WORKFLOW_ID,
                block_path="denoise",
                state_in=issued_semantic,
                num_inference_steps=18,
            )
        self.assertEqual(denoise_pipeline.calls, [{"state": semantic_state, "num_inference_steps": 18}])

        # Browser NumberField commits its canonical editable transport as
        # text. The official workflow boundary parses that exact integer form
        # without weakening malformed/fractional/bool validation.
        denoise_pipeline.calls.clear()
        with mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)):
            denoise.execute(
                pipeline_components={},
                pipeline_class=PIPELINE_CLASS,
                workflow_id=WORKFLOW_ID,
                block_path="denoise",
                state_in=issued_semantic,
                num_inference_steps="18",
            )
        self.assertEqual(denoise_pipeline.calls, [{"state": semantic_state, "num_inference_steps": 18}])

        for invalid_steps in (True, "18.0", "018", " 18", "18 ", 0, 101):
            with (
                mock.patch.object(denoise, "_prepare_pipeline", return_value=(token, denoise_pipeline)),
                self.assertRaisesRegex(ValueError, "denoise steps must be an integer"),
            ):
                denoise.execute(
                    pipeline_components={},
                    pipeline_class=PIPELINE_CLASS,
                    workflow_id=WORKFLOW_ID,
                    block_path="denoise",
                    state_in=issued_semantic,
                    num_inference_steps=invalid_steps,
                )

        waveform = object()
        decoded_state = FakeState(audios=waveform)
        decode_pipeline = RecordingPipeline(decoded_state, sampling_rate=44_100)
        decode = WorkflowDecodeAudio("decode")
        audio = {
            "samples": "samples",
            "sample_rate": 44_100,
            "duration_seconds": 12.0,
        }
        with (
            mock.patch.object(decode, "_prepare_pipeline", return_value=(token, decode_pipeline)),
            mock.patch("modules.DiffusersAudio.main.output_to_audio_object", return_value=audio) as convert,
        ):
            output = decode.execute(
                pipeline_components={},
                pipeline_class=PIPELINE_CLASS,
                workflow_id=WORKFLOW_ID,
                block_path="decode",
                state_in=denoise_output["state_out"],
            )
        self.assertEqual(decode_pipeline.calls, [{"state": denoised_state, "output_type": "pt"}])
        convert.assert_called_once_with(waveform, sample_rate=44_100)
        self.assertEqual(output["audio"], audio)
        self.assertEqual(output["sample_rate"], 44_100)
        self.assertEqual(output["duration_seconds"], 12.0)

    def test_prepare_pipeline_requires_bound_complete_component_bundle(self):
        token = issue_pipeline_instance_token(
            model_type=PIPELINE_CLASS,
            repo_id="MiniMaxAI/MiniMax-Music3",
            repo_source="hub",
            revision="f" * 40,
        )
        bundle = {
            "vocoder": {"model_id": "vocoder-id"},
            "model_type": PIPELINE_CLASS,
            "repo_id": "MiniMaxAI/MiniMax-Music3",
            "repo_source": "hub",
            "revision": "f" * 40,
        }
        bind_loader_outputs({"pipeline_components": bundle}, token)

        runtime_pipeline = mock.Mock(pretrained_component_names=["vocoder"])
        block = mock.Mock()
        block.init_pipeline.return_value = runtime_pipeline
        blocks = mock.Mock(sub_blocks={"decode": block})
        definition = mock.Mock(blocks=blocks)
        pipeline_type = mock.Mock(return_value=definition)
        node = WorkflowDecodeAudio("decode")
        with (
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.pipeline_class_from_model_type",
                return_value=pipeline_type,
            ),
            mock.patch(
                "modules.ModularDiffusers.workflow_blocks.collect_model_ids",
                return_value=["vocoder-id"],
            ),
            mock.patch.object(
                __import__("modules.ModularDiffusers.workflow_blocks", fromlist=["components"]).components,
                "get_components_by_ids",
                return_value={"vocoder": "resident-vocoder"},
            ),
        ):
            returned_token, returned_pipeline = node._prepare_pipeline(
                pipeline_components=bundle,
                pipeline_class=PIPELINE_CLASS,
                workflow_id=WORKFLOW_ID,
                block_path="decode",
            )
        self.assertIs(returned_token, token)
        self.assertIs(returned_pipeline, runtime_pipeline)
        runtime_pipeline.update_components.assert_called_once_with(vocoder="resident-vocoder")


if __name__ == "__main__":
    unittest.main()
