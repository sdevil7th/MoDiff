"""No-weight regression against pinned GLM input validation and prompt encoding."""
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")
import torch
from diffusers import GlmImagePipeline as PinnedGlmImagePipeline
from PIL import Image

from modiff.model_artifact_catalog import catalog_revision
from modules.DiffusersImage.main import Generate, IMAGE_PIPELINE_ADAPTERS, _tag_image_pipeline


class GlmImagePipeline:
    check_inputs = PinnedGlmImagePipeline.check_inputs
    encode_prompt = PinnedGlmImagePipeline.encode_prompt
    _execution_device = torch.device("cpu")
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    vae_scale_factor = 8
    dtype = torch.float32

    def __init__(self, fail=False):
        self.transformer = torch.nn.Linear(4, 4).to(torch.bfloat16)
        self.transformer.config = SimpleNamespace(patch_size=2)
        self.text_encoder = torch.nn.Linear(4, 4).to(torch.float32)
        self.fail = fail
        self.prior_calls = []
        self.glyph_calls = []
        adapter = IMAGE_PIPELINE_ADAPTERS["GlmImagePipeline"]
        _tag_image_pipeline(self, adapter, "text_to_image", adapter.default_repo,
                            "hub", catalog_revision(adapter.default_repo))

    def generate_prior_tokens(self, prompt, image, height, width, device, generator):
        self.prior_calls.append((prompt, generator.initial_seed()))
        return torch.ones((1, 3), dtype=torch.long), None, None

    def _get_glyph_embeds(self, prompts, max_sequence_length, device, dtype):
        length = min(max_sequence_length, max(len(prompt.split()) for prompt in prompts) or 1)
        self.glyph_calls.append((list(prompts), length, dtype))
        encoded = self.text_encoder(torch.ones((len(prompts), length, 4), dtype=torch.float32))
        return encoded.to(device=device, dtype=dtype)

    def __call__(self, prompt=None, height=1024, width=1024, num_inference_steps=50,
                 guidance_scale=1.5, max_sequence_length=512, generator=None,
                 prompt_embeds=None, negative_prompt_embeds=None, prior_token_ids=None,
                 prior_token_image_ids=None, source_image_grid_thw=None,
                 output_type="pil", return_dict=True, callback_on_step_end=None,
                 callback_on_step_end_tensor_inputs=None):
        # These are the pinned pipeline's real checker and encoder, in native order.
        self.check_inputs(prompt, height, width, callback_on_step_end_tensor_inputs,
                          prompt_embeds, negative_prompt_embeds, prior_token_ids,
                          prior_token_image_ids, source_image_grid_thw)
        assert prompt == "A detailed recipe with several lines of text"
        if prior_token_ids is None:
            self.generate_prior_tokens(prompt, None, height, width, self._execution_device, generator)
        positive, negative = self.encode_prompt(
            prompt, guidance_scale > 1, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds, device=self._execution_device,
            dtype=self.dtype, max_sequence_length=max_sequence_length,
        )
        assert positive.shape == (1, 8, 4)
        assert self.transformer(positive).dtype == torch.bfloat16
        if guidance_scale > 1:
            assert negative.shape == (1, 1, 4)
            assert self.transformer(negative).dtype == torch.bfloat16
        else:
            assert negative is None
        if self.fail:
            raise RuntimeError("injected denoising failure")
        return SimpleNamespace(images=[Image.new("RGB", (width, height))])


@pytest.mark.parametrize("guidance", [1.0, 1.5])
@pytest.mark.parametrize("instance_method", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_native_variable_length_glyphs_keep_dtype_and_restore_encoder(guidance, instance_method, fail):
    pipeline = GlmImagePipeline(fail=fail)
    if instance_method:
        pipeline.encode_prompt = pipeline.encode_prompt
    original = pipeline.__dict__.get("encode_prompt")
    node = Generate("glm-native-glyph-regression")
    node.progress = lambda *args, **kwargs: None
    kwargs = dict(pipeline=pipeline, prompt="A detailed recipe with several lines of text",
                  negative_prompt="", width=1024, height=1024, num_inference_steps=50,
                  guidance_scale=guidance, max_sequence_length=512, seed=42)
    if fail:
        with pytest.raises(RuntimeError, match="injected denoising failure"):
            node.execute(**kwargs)
    else:
        assert node.execute(**kwargs)["width_out"] == 1024
    assert pipeline.__dict__.get("encode_prompt") is original
    assert ("encode_prompt" in pipeline.__dict__) == instance_method
    assert pipeline.text_encoder.weight.dtype == torch.float32
    assert pipeline.prior_calls == [(kwargs["prompt"], 42)]
    assert len(pipeline.glyph_calls) == (2 if guidance > 1 else 1)
