# Copyright 2026 The HuggingFace Team. All rights reserved.
# Modified by MoDiff: explicit native guidance and numerical compatibility.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""App-owned native numerical/guidance compatibility for reviewed block trees."""

import math

import torch
from diffusers import (
    ClassifierFreeGuidance, FluxKontextModularPipeline, FluxModularPipeline,
    QwenImageModularPipeline, ZImageModularPipeline,
)
from diffusers.configuration_utils import FrozenDict
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from diffusers.modular_pipelines.flux.denoise import (
    FluxDenoiseStep, FluxKontextDenoiseStep, FluxKontextLoopDenoiser, FluxLoopAfterDenoiser, FluxLoopDenoiser,
)
from diffusers.modular_pipelines.flux.encoders import FluxTextEncoderStep
from diffusers.modular_pipelines.flux.inputs import FluxTextInputStep
from diffusers.modular_pipelines.flux.modular_blocks_flux import (
    FluxAutoBeforeDenoiseStep,
    FluxAutoBlocks,
    FluxAutoInputStep,
    FluxBeforeDenoiseStep,
    FluxCoreDenoiseStep,
    FluxImg2ImgBeforeDenoiseStep,
    FluxImg2ImgInputStep,
)
from diffusers.modular_pipelines.flux.modular_blocks_flux_kontext import (
    FluxKontextAutoBeforeDenoiseStep, FluxKontextAutoBlocks, FluxKontextAutoInputStep,
    FluxKontextBeforeDenoiseStep, FluxKontextCoreDenoiseStep,
    FluxKontextImageConditionedBeforeDenoiseStep, FluxKontextInputStep,
)
from diffusers.modular_pipelines.qwenimage.before_denoise import QwenImageRoPEInputsStep
from diffusers.modular_pipelines.qwenimage.inputs import QwenImageTextInputsStep
from diffusers.modular_pipelines.qwenimage.modular_blocks_qwenimage import (
    QwenImageAutoBlocks, QwenImageAutoCoreDenoiseStep, QwenImageCoreDenoiseStep,
)
from diffusers.modular_pipelines.z_image.denoise import (
    ZImageDenoiseStep,
    ZImageLoopBeforeDenoiser,
    ZImageLoopDenoiser,
)
from diffusers.modular_pipelines.z_image.modular_blocks_z_image import (
    ZImageAutoBlocks,
    ZImageAutoDenoiseStep,
    ZImageCoreDenoiseStep,
    ZImageImage2ImageCoreDenoiseStep,
)


_Z_IMAGE_GUIDER_FIELDS = {"cap_feats": ("prompt_embeds", "negative_prompt_embeds")}
_FLUX_GUIDER_FIELDS = {
    "prompt_embeds": ("prompt_embeds", "negative_prompt_embeds"),
    "pooled_prompt_embeds": ("pooled_prompt_embeds", "negative_pooled_prompt_embeds"),
}


def _flux_guider_spec():
    # Ordinary SDK-style native graphs keep embedded guidance only. An actual
    # connected CFG component explicitly enables unconditional conditioning.
    return ComponentSpec("guider", ClassifierFreeGuidance, config=FrozenDict({
        "enabled": False, "guidance_scale": 1.0,
    }), default_creation_method="from_config")


def _flux_guider(components):
    guider = getattr(components, "guider", None)
    if guider is not None and type(guider) is not ClassifierFreeGuidance:
        raise ValueError("The reviewed Flux native conditioning requires exact ClassifierFreeGuidance.")
    return guider


def _flux_needs_negative(guider):
    # Do not consult num_conditions here: its last-run step window may still
    # be at the final step when the shared guider is reused by EncodePrompt.
    return guider is not None and guider.get_state()["enabled"] and not math.isclose(
        guider.guidance_scale, 0.0 if guider.use_original_formulation else 1.0,
    )


class FluxCFGTextEncoderStep(FluxTextEncoderStep):
    """Use official CLIP/T5 encoding for both explicit CFG conditions.

    Derived from Diffusers fbf49e7f's FluxTextEncoderStep/FluxPipeline under
    Apache-2.0, and modified by MoDiff to declare the shared guider and outputs.
    """

    @property
    def expected_components(self):
        return super().expected_components + [_flux_guider_spec()]

    @property
    def inputs(self):
        return super().inputs + [InputParam("negative_prompt", default=""), InputParam("negative_prompt_2")]

    @property
    def intermediate_outputs(self):
        return super().intermediate_outputs + [
            OutputParam(name, type_hint=torch.Tensor, kwargs_type="denoiser_input_fields")
            for name in ("negative_prompt_embeds", "negative_pooled_prompt_embeds")
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        guider = _flux_guider(components)
        block_state = self.get_block_state(state)
        self.check_inputs(block_state)
        negative = block_state.negative_prompt
        negative_2 = block_state.negative_prompt_2
        if _flux_needs_negative(guider):
            if not isinstance(block_state.prompt, (str, list)):
                raise ValueError("Flux CFG requires a positive prompt batch.")
            batch_size = 1 if isinstance(block_state.prompt, str) else len(block_state.prompt)
            negative = [negative or ""] * batch_size if isinstance(negative, str) or negative is None else negative
            if (
                not isinstance(negative, list) or len(negative) != batch_size
                or any(not isinstance(value, str) for value in negative)
                or (negative_2 is not None and not isinstance(negative_2, (str, list)))
                or (isinstance(negative_2, list) and (
                    len(negative_2) != batch_size or any(not isinstance(value, str) for value in negative_2)
                ))
            ):
                raise ValueError("Flux negative prompts must match the positive prompt batch.")
            if isinstance(negative_2, str):
                negative_2 = [negative_2] * batch_size
        attention = block_state.joint_attention_kwargs
        lora_scale = attention.get("scale") if attention is not None else None
        block_state.prompt_embeds, block_state.pooled_prompt_embeds = self.encode_prompt(
            components, prompt=block_state.prompt, prompt_2=None,
            device=components._execution_device, max_sequence_length=block_state.max_sequence_length,
            lora_scale=lora_scale,
        )
        block_state.negative_prompt_embeds = None
        block_state.negative_pooled_prompt_embeds = None
        if _flux_needs_negative(guider):
            block_state.negative_prompt_embeds, block_state.negative_pooled_prompt_embeds = self.encode_prompt(
                components, prompt=negative, prompt_2=negative_2,
                device=components._execution_device, max_sequence_length=block_state.max_sequence_length,
                lora_scale=lora_scale,
            )
        self.set_block_state(state, block_state)
        return components, state


class FluxCFGTextInputStep(FluxTextInputStep):
    """Repeat unconditional embeddings with the native positive batch order.

    Derived from Diffusers fbf49e7f's FluxTextInputStep under Apache-2.0,
    and modified by MoDiff to include declared unconditional tensors.
    """

    @property
    def inputs(self):
        return super().inputs + [
            InputParam(name, type_hint=torch.Tensor, kwargs_type="denoiser_input_fields")
            for name in ("negative_prompt_embeds", "negative_pooled_prompt_embeds")
        ]

    @property
    def intermediate_outputs(self):
        return super().intermediate_outputs + [
            OutputParam(name, type_hint=torch.Tensor, kwargs_type="denoiser_input_fields")
            for name in ("negative_prompt_embeds", "negative_pooled_prompt_embeds")
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        initial = self.get_block_state(state)
        negatives = (initial.negative_prompt_embeds, initial.negative_pooled_prompt_embeds)
        if (negatives[0] is None) != (negatives[1] is None):
            raise ValueError("Flux CFG requires both negative text and pooled embeddings.")
        if negatives[0] is not None and (
            any(not isinstance(value, torch.Tensor) for value in negatives)
            or negatives[0].ndim != 3 or negatives[1].ndim != 2
            or any(value.shape[0] != initial.prompt_embeds.shape[0] for value in negatives)
        ):
            raise ValueError("Flux CFG negative embeddings must match the positive prompt batch.")
        block_state = initial
        self.check_inputs(components, block_state)
        block_state.batch_size = block_state.prompt_embeds.shape[0]
        block_state.dtype = block_state.prompt_embeds.dtype
        batch = block_state.batch_size * block_state.num_images_per_prompt
        block_state.prompt_embeds = block_state.prompt_embeds.repeat(1, block_state.num_images_per_prompt, 1).view(
            batch, block_state.prompt_embeds.shape[1], -1,
        )
        block_state.pooled_prompt_embeds = block_state.pooled_prompt_embeds.repeat(1, block_state.num_images_per_prompt).view(batch, -1)
        if negatives[0] is not None:
            block_state.negative_prompt_embeds = negatives[0].repeat(1, block_state.num_images_per_prompt, 1).view(
                batch, negatives[0].shape[1], -1,
            )
            block_state.negative_pooled_prompt_embeds = negatives[1].repeat(1, block_state.num_images_per_prompt).view(batch, -1)
        self.set_block_state(state, block_state)
        return components, state


class _FluxCFGPrediction:
    """Official whole Flux/Kontext conditioning and cache-context convention.

    Derived from Diffusers fbf49e7f's FluxPipeline/FluxKontextPipeline and
    native loop predictors under Apache-2.0, and modified by MoDiff to consume
    the actual shared ClassifierFreeGuidance component in a native block.
    """
    @property
    def expected_components(self):
        return super().expected_components + [_flux_guider_spec()]

    @property
    def inputs(self):
        return super().inputs + [
            InputParam("num_inference_steps"),
            InputParam("negative_prompt_embeds", type_hint=torch.Tensor),
            InputParam("negative_pooled_prompt_embeds", type_hint=torch.Tensor),
        ]

    @torch.no_grad()
    def _predict_cfg(self, components, block_state, i, t, *, image_conditioned=False):
        guider = _flux_guider(components)
        if guider is None:
            return None
        guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        batches = guider.prepare_inputs_from_block_state(block_state, _FLUX_GUIDER_FIELDS)
        latents = block_state.latents
        for batch in batches:
            prompt = getattr(batch, "prompt_embeds", None)
            pooled = getattr(batch, "pooled_prompt_embeds", None)
            if (
                not isinstance(prompt, torch.Tensor) or not isinstance(pooled, torch.Tensor)
                or prompt.shape != block_state.prompt_embeds.shape
                or pooled.shape != block_state.pooled_prompt_embeds.shape
                or prompt.shape[0] != latents.shape[0]
            ):
                raise ValueError("Flux CFG requires matching positive and negative conditioning batches.")
        model_input = latents
        if image_conditioned and block_state.image_latents is not None:
            model_input = torch.cat([latents, block_state.image_latents], dim=1)
        timestep = t.expand(latents.shape[0]).to(latents.dtype) / 1000
        for index, batch in enumerate(batches):
            text_ids = block_state.txt_ids if index == 0 else torch.zeros(
                batch.prompt_embeds.shape[1], 3, dtype=batch.prompt_embeds.dtype, device=batch.prompt_embeds.device,
            )
            guider.prepare_models(components.transformer)
            try:
                with components.transformer.cache_context("cond" if index == 0 else "uncond"):
                    prediction = components.transformer(
                        hidden_states=model_input, timestep=timestep, guidance=block_state.guidance,
                        encoder_hidden_states=batch.prompt_embeds, pooled_projections=batch.pooled_prompt_embeds,
                        joint_attention_kwargs=block_state.joint_attention_kwargs,
                        txt_ids=text_ids, img_ids=block_state.img_ids, return_dict=False,
                    )[0]
                batch.noise_pred = prediction[:, :latents.size(1)] if image_conditioned else prediction
            finally:
                guider.cleanup_models(components.transformer)
        # Whole Flux/Kontext combines predictions in the actual model dtype,
        # unlike Z-Image's FP32 guidance arithmetic.
        block_state.noise_pred = guider(batches)[0]
        return components, block_state


class FluxWholePipelineTimeDenoiser(_FluxCFGPrediction, FluxLoopDenoiser):
    """Keep the whole FluxPipeline's cast-before-normalization convention.

    FluxTransformer casts normalized time to its model dtype again. Dividing
    before the latent-dtype cast changes effective model times for reviewed
    BF16 Dev/Krea schedules. All upstream model kwargs and embedded guidance
    remain unchanged; the loop's scheduler still receives the original time.
    """

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        result = self._predict_cfg(components, block_state, i, t)
        if result is not None:
            return result
        model_time = t.expand(block_state.latents.shape[0]).to(block_state.latents.dtype)
        return super().__call__(components, block_state, i=i, t=model_time)


class FluxKontextCFGDenoiser(_FluxCFGPrediction, FluxKontextLoopDenoiser):
    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        result = self._predict_cfg(components, block_state, i, t, image_conditioned=True)
        return result if result is not None else super().__call__(components, block_state, i=i, t=t)


def _replace_block(parent, name, replacement):
    parent.sub_blocks[name] = replacement
    classes = list(parent.block_classes)
    classes[list(parent.block_names).index(name)] = replacement
    parent.block_classes = classes


def _prepare_flux_blocks(blocks):
    if type(blocks) is not FluxAutoBlocks:
        raise ValueError("Flux native time compatibility requires its reviewed AutoBlocks blueprint.")
    core = blocks.sub_blocks.get("denoise")
    if type(core) is not FluxCoreDenoiseStep:
        raise ValueError("The reviewed Flux native denoise core has changed.")
    if type(blocks.sub_blocks.get("text_encoder")) is not FluxTextEncoderStep:
        raise ValueError("The reviewed Flux native text encoder has changed.")
    image_input = core.sub_blocks.get("input").sub_blocks.get("img2img") if core.sub_blocks.get("input") else None
    if (
        type(image_input) is not FluxImg2ImgInputStep
        or type(image_input.sub_blocks.get("text_inputs")) is not FluxTextInputStep
    ):
        raise ValueError("The reviewed Flux native image text inputs have changed.")
    for name, selector_type, branches in (
        ("input", FluxAutoInputStep, {"img2img": FluxImg2ImgInputStep, "text2image": FluxTextInputStep}),
        ("before_denoise", FluxAutoBeforeDenoiseStep,
         {"img2img": FluxImg2ImgBeforeDenoiseStep, "text2image": FluxBeforeDenoiseStep}),
    ):
        selector = core.sub_blocks.get(name)
        if type(selector) is not selector_type or any(
            type(selector.sub_blocks.get(key)) is not branch_type for key, branch_type in branches.items()
        ):
            raise ValueError("The reviewed Flux native input or preparation branches have changed.")
    loop = core.sub_blocks.get("denoise")
    if (
        type(loop) is not FluxDenoiseStep
        or type(loop.sub_blocks.get("denoiser")) is not FluxLoopDenoiser
        or type(loop.sub_blocks.get("after_denoiser")) is not FluxLoopAfterDenoiser
        or list(loop.block_names) != ["denoiser", "after_denoiser"]
        or len(loop.block_classes) != 2
    ):
        raise ValueError("The reviewed Flux native denoise loop has changed.")
    _replace_block(blocks, "text_encoder", FluxCFGTextEncoderStep())
    _replace_block(core.sub_blocks["input"], "text2image", FluxCFGTextInputStep())
    _replace_block(image_input, "text_inputs", FluxCFGTextInputStep())
    _replace_block(loop, "denoiser", FluxWholePipelineTimeDenoiser())
    return blocks


def _prepare_flux_kontext_blocks(blocks):
    if type(blocks) is not FluxKontextAutoBlocks:
        raise ValueError("Flux Kontext native CFG requires its reviewed AutoBlocks blueprint.")
    core = blocks.sub_blocks.get("denoise")
    if type(core) is not FluxKontextCoreDenoiseStep or type(blocks.sub_blocks.get("text_encoder")) is not FluxTextEncoderStep:
        raise ValueError("The reviewed Flux Kontext native core or text encoder has changed.")
    inputs = core.sub_blocks.get("input")
    preparations = core.sub_blocks.get("before_denoise")
    if (
        type(inputs) is not FluxKontextAutoInputStep
        or type(inputs.sub_blocks.get("text2image")) is not FluxTextInputStep
        or type(inputs.sub_blocks.get("image_conditioned")) is not FluxKontextInputStep
        or type(inputs.sub_blocks["image_conditioned"].sub_blocks.get("text_inputs")) is not FluxTextInputStep
        or type(preparations) is not FluxKontextAutoBeforeDenoiseStep
        or type(preparations.sub_blocks.get("text2image")) is not FluxKontextBeforeDenoiseStep
        or type(preparations.sub_blocks.get("image_conditioned")) is not FluxKontextImageConditionedBeforeDenoiseStep
    ):
        raise ValueError("The reviewed Flux Kontext native input or preparation branches have changed.")
    loop = core.sub_blocks.get("denoise")
    if (
        type(loop) is not FluxKontextDenoiseStep
        or type(loop.sub_blocks.get("denoiser")) is not FluxKontextLoopDenoiser
        or type(loop.sub_blocks.get("after_denoiser")) is not FluxLoopAfterDenoiser
        or list(loop.block_names) != ["denoiser", "after_denoiser"]
        or len(loop.block_classes) != 2
    ):
        raise ValueError("The reviewed Flux Kontext native denoise loop has changed.")
    _replace_block(blocks, "text_encoder", FluxCFGTextEncoderStep())
    _replace_block(inputs, "text2image", FluxCFGTextInputStep())
    _replace_block(inputs.sub_blocks["image_conditioned"], "text_inputs", FluxCFGTextInputStep())
    _replace_block(loop, "denoiser", FluxKontextCFGDenoiser())
    return blocks


class ZImageFloat32TimeInput(ZImageLoopBeforeDenoiser):
    """Keep the whole ZImagePipeline's FP32 normalized time convention.

    The pinned native block casts scheduler time to the latent/model dtype
    before subtraction. BF16 rounds six of the eight reviewed template steps.
    Model inputs retain the upstream latent shape/dtype; only time arithmetic
    follows the official whole-pipeline path. No weights or scheduler state are
    changed. Derived from Diffusers fbf49e7f's ZImageLoopBeforeDenoiser, under
    Apache-2.0, and modified by MoDiff.
    """

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        latents = block_state.latents.unsqueeze(2).to(block_state.dtype)
        block_state.latent_model_input = list(latents.unbind(dim=0))
        timestep = t.expand(latents.shape[0]).float()
        block_state.timestep = (1000 - timestep) / 1000
        return components, block_state


class ZImageFloat32CFGDenoiser(ZImageLoopDenoiser):
    """Match whole Z-Image's batched CFG and FP32 prediction arithmetic.

    The actual ClassifierFreeGuidance object retains its enabled, formulation,
    scale and step-window choices. Only that exact no-hook guider uses one
    conditional-then-unconditional model batch and FP32 predictions. Other
    guidance techniques keep their upstream per-condition model hooks/path.
    Derived from Diffusers fbf49e7f's ZImageLoopDenoiser and ZImagePipeline,
    under Apache-2.0, and modified by MoDiff.
    """

    @torch.no_grad()
    def __call__(self, components, block_state, i, t):
        guider = components.guider
        if type(guider) is not ClassifierFreeGuidance:
            return super().__call__(components, block_state, i=i, t=t)
        if self._guider_input_fields != _Z_IMAGE_GUIDER_FIELDS:
            raise ValueError("The reviewed Z-Image CFG conditioning fields have changed.")

        guider.set_state(step=i, num_inference_steps=block_state.num_inference_steps, timestep=t)
        batches = guider.prepare_inputs_from_block_state(block_state, self._guider_input_fields)
        if len(batches) not in (1, 2):
            raise ValueError("The reviewed Z-Image CFG condition count has changed.")
        model_inputs = block_state.latent_model_input
        if not isinstance(model_inputs, list) or not model_inputs:
            raise ValueError("The reviewed Z-Image CFG requires its native latent input batch.")
        batch_size = len(model_inputs)
        timestep = block_state.timestep
        if not isinstance(timestep, torch.Tensor) or timestep.shape != (batch_size,):
            raise ValueError("The reviewed Z-Image CFG timestep batch does not match its latents.")
        cap_feats = []
        for batch in batches:
            features = getattr(batch, "cap_feats", None)
            if (
                not isinstance(features, list) or len(features) != batch_size
                or any(not isinstance(value, torch.Tensor) for value in features)
            ):
                raise ValueError("The reviewed Z-Image CFG text batch does not match its latents.")
            cap_feats.extend(value.to(block_state.dtype) for value in features)

        prepared = 0
        try:
            # Exact ClassifierFreeGuidance inherits the no-hook model lifecycle;
            # retain its per-condition counters while combining model execution.
            for _ in batches:
                guider.prepare_models(components.transformer)
                prepared += 1
            predictions = components.transformer(
                x=model_inputs * len(batches), t=timestep.repeat(len(batches)),
                cap_feats=cap_feats, return_dict=False,
            )[0]
            if not isinstance(predictions, list) or len(predictions) != batch_size * len(batches):
                raise ValueError("The reviewed Z-Image CFG prediction batch has changed.")
            for index, batch in enumerate(batches):
                values = predictions[index * batch_size:(index + 1) * batch_size]
                batch.noise_pred = torch.stack([value.float() for value in values], dim=0).squeeze(2)
            # Whole Z-Image performs guidance on positive model predictions,
            # then changes the sign once before the FP32 scheduler update.
            block_state.noise_pred = -guider(batches)[0]
        finally:
            for _ in range(prepared):
                guider.cleanup_models(components.transformer)
        return components, block_state


class QwenImageWholePromptMaskRoPEInputsStep(QwenImageRoPEInputsStep):
    """Keep the whole Qwen T2I pipeline's all-valid attention-mask convention.

    Diffusers fbf49e7f's QwenImagePipeline.encode_prompt removes masks with no
    padding after truncation and batch expansion. The native encoder/input
    steps retain those masks, selecting a different SDPA execution path.
    Native text-input and RoPE preparation require those tensors. Finish both
    real stages first, then normalize only all-valid masks before denoising.
    Padded masks and the encoded tensors remain untouched.
    """

    @torch.no_grad()
    def __call__(self, components, state):
        components, state = super().__call__(components, state)
        for name in ("prompt_embeds_mask", "negative_prompt_embeds_mask"):
            mask = state.get(name)
            if mask is not None and mask.all():
                state.set(name, None, OutputParam.template(name).kwargs_type)
        return components, state


def _prepare_qwen_t2i_blocks(blocks):
    if type(blocks) is not QwenImageAutoBlocks:
        raise ValueError("Qwen T2I mask compatibility requires its reviewed AutoBlocks blueprint.")
    denoise = blocks.sub_blocks.get("denoise")
    if type(denoise) is not QwenImageAutoCoreDenoiseStep:
        raise ValueError("The reviewed Qwen native denoise selector has changed.")
    branch = denoise.sub_blocks.get("text2image")
    if type(branch) is not QwenImageCoreDenoiseStep:
        raise ValueError("The reviewed Qwen native text2image branch has changed.")
    if (
        type(branch.sub_blocks.get("input")) is not QwenImageTextInputsStep
        or type(branch.sub_blocks.get("prepare_rope_inputs")) is not QwenImageRoPEInputsStep
        or list(branch.block_names) != [
            "input", "prepare_latents", "set_timesteps", "prepare_rope_inputs", "denoise", "after_denoise",
        ]
        or len(branch.block_classes) != 6
        or type(list(branch.block_classes)[0]) is not QwenImageTextInputsStep
        or type(list(branch.block_classes)[3]) is not QwenImageRoPEInputsStep
    ):
        raise ValueError("The reviewed Qwen native text-input contract has changed.")
    # This fresh tree is caller-owned. Sibling image/control/mask routes and
    # the upstream encoder retain their existing native behavior.
    adapted = QwenImageWholePromptMaskRoPEInputsStep()
    classes = list(branch.block_classes)
    classes[3] = adapted
    branch.block_classes = classes
    branch.sub_blocks["prepare_rope_inputs"] = adapted
    return blocks


def prepare_native_pipeline_blocks(pipeline_class, blocks):
    """Adapt newly constructed exact Flux, Kontext, Qwen and Z-Image blueprints.

    Callers own this fresh tree. Neither upstream class-level block definitions
    nor another component owner's pipeline is modified. Both reviewed native
    branches must retain their exact denoiser contracts before any change.
    """

    if pipeline_class is FluxModularPipeline:
        return _prepare_flux_blocks(blocks)
    if pipeline_class is FluxKontextModularPipeline:
        return _prepare_flux_kontext_blocks(blocks)
    if pipeline_class is QwenImageModularPipeline:
        return _prepare_qwen_t2i_blocks(blocks)
    if pipeline_class is not ZImageModularPipeline:
        return blocks
    if type(blocks) is not ZImageAutoBlocks:
        raise ValueError("Z-Image native time compatibility requires its reviewed AutoBlocks blueprint.")
    denoise = blocks.sub_blocks.get("denoise")
    if type(denoise) is not ZImageAutoDenoiseStep:
        raise ValueError("The reviewed Z-Image native denoise selector has changed.")
    loops = []
    for name, branch_type in (
        ("text2image", ZImageCoreDenoiseStep),
        ("image2image", ZImageImage2ImageCoreDenoiseStep),
    ):
        branch = denoise.sub_blocks.get(name)
        if type(branch) is not branch_type:
            raise ValueError(f"The reviewed Z-Image {name} branch has changed.")
        loop = branch.sub_blocks.get("denoise")
        if type(loop) is not ZImageDenoiseStep:
            raise ValueError("The reviewed Z-Image native denoise loop has changed.")
        before = getattr(loop, "sub_blocks", {}).get("before_denoiser")
        if type(before) is not ZImageLoopBeforeDenoiser:
            raise ValueError("The reviewed Z-Image native before-denoiser block has changed.")
        prediction = loop.sub_blocks.get("denoiser")
        if (
            type(prediction) is not ZImageLoopDenoiser
            or prediction._guider_input_fields != _Z_IMAGE_GUIDER_FIELDS
            or list(loop.block_names) != ["before_denoiser", "denoiser", "after_denoiser"]
            or len(loop.block_classes) != 3
        ):
            raise ValueError("The reviewed Z-Image native CFG denoiser contract has changed.")
        loops.append(loop)
    for loop in loops:
        loop.sub_blocks["before_denoiser"] = ZImageFloat32TimeInput()
        loop.sub_blocks["denoiser"] = ZImageFloat32CFGDenoiser(guider_input_fields=dict(_Z_IMAGE_GUIDER_FIELDS))
        classes = list(loop.block_classes)
        classes[loop.block_names.index("before_denoiser")] = ZImageFloat32TimeInput
        classes[loop.block_names.index("denoiser")] = loop.sub_blocks["denoiser"]
        loop.block_classes = classes
    return blocks
