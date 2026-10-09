# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
# Modified by MoDiff: explicitly selected Qwen whole-template compatibility.
# SPDX-License-Identifier: Apache-2.0
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
"""Conditional adapters for the pinned, genuine Qwen Edit inpainting blocks.

The native default is unchanged. ``whole_v1`` preserves the existing app's
single-image whole-inpaint preprocessing, sampled source posterior and final
masked composite. It does not run an opaque pipeline or claim identical final
RNG state: the whole pipeline also samples unused masked-image latents after
initial noise, whereas the native loop consumes only the source latents/mask.

Derived from Diffusers fbf49e7f35857f76bc57b177e26f12b03687c668's
QwenImageEditInpaintPipeline and Qwen inpaint encoder/output blocks under
Apache-2.0, with MoDiff's existing mask-composite helper and sealed route state.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from PIL import Image
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam
from diffusers.modular_pipelines.qwenimage.encoders import (
    QwenImageEditInpaintProcessImagesInputStep,
    QwenImageVaeEncoderStep,
)
from diffusers.modular_pipelines.qwenimage.decoders import QwenImageInpaintProcessImagesOutputStep
from diffusers.modular_pipelines.qwenimage.before_denoise import QwenImageEditRoPEInputsStep
from diffusers.modular_pipelines.qwenimage.modular_blocks_qwenimage_edit import (
    QwenImageEditAutoBlocks,
    QwenImageEditAutoCoreDenoiseStep,
    QwenImageEditAutoDecodeStep,
    QwenImageEditAutoVaeEncoderStep,
    QwenImageEditInpaintDecodeStep,
    QwenImageEditInpaintVaeEncoderStep,
    QwenImageEditInpaintCoreDenoiseStep,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_inpaint import retrieve_latents


COMPATIBILITY_INPUT = "inpaint_compatibility"
COMPATIBILITY_STATE = "qwen_inpaint_compatibility_state"


def validate_compatibility(value):
    if type(value) is not str or value not in ("native", "whole_v1"):
        raise ValueError("Qwen inpaint compatibility must be 'native' or 'whole_v1'.")
    return value


def validate_owner_compatibility(value, *, model_type, workflow_id):
    """Validate the technical selection before owner lookup or weight loading."""
    value = validate_compatibility(value)
    if value == "whole_v1" and (
        model_type != "QwenImageEditModularPipeline"
        or workflow_id != "image_conditioned_inpainting"
    ):
        raise ValueError("Qwen whole_v1 model policy requires the reviewed Qwen Edit inpainting workflow.")
    return value


def require_compatible_vae_owner(vae, value):
    """An encoder cannot select whole compatibility on another owner's VAE."""
    value = validate_compatibility(value)
    actual = getattr(vae, "_modiff_modular_runtime_policy", {}).get(COMPATIBILITY_INPUT, "native")
    if actual != value:
        raise ValueError("Qwen Image Encode compatibility must match its connected model owner's policy.")


@dataclass(frozen=True, slots=True)
class QwenInpaintCompatibilityState:
    """Immutable media snapshot carried only by backend-issued route state."""

    image_size: tuple[int, int]
    image_bytes: bytes
    mask_size: tuple[int, int]
    mask_bytes: bytes

    @classmethod
    def from_images(cls, image, mask):
        if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
            raise ValueError("Qwen whole_v1 requires one PIL source image and one PIL mask.")
        image, mask = image.convert("RGB"), mask.convert("L")
        return cls(image.size, image.tobytes(), mask.size, mask.tobytes())

    def images(self):
        return (
            Image.frombytes("RGB", self.image_size, self.image_bytes),
            Image.frombytes("L", self.mask_size, self.mask_bytes),
        )


def _compatibility_param():
    return InputParam(COMPATIBILITY_INPUT, default="native", type_hint=str)


class QwenWholeInpaintPreprocessStep(QwenImageEditInpaintProcessImagesInputStep):
    @property
    def inputs(self):
        return super().inputs + [InputParam.template("image"), _compatibility_param()]

    @property
    def intermediate_outputs(self):
        return super().intermediate_outputs + [OutputParam(COMPATIBILITY_STATE)]

    @torch.no_grad()
    def __call__(self, components, state):
        block = self.get_block_state(state)
        if validate_compatibility(block.inpaint_compatibility) == "native":
            return super().__call__(components, state)
        snapshot = QwenInpaintCompatibilityState.from_images(block.image, block.mask_image)
        if not isinstance(block.resized_image, list) or len(block.resized_image) != 1:
            raise ValueError("Qwen whole_v1 requires exactly one resized source image.")
        image = block.resized_image[0]
        calculated_width, calculated_height = image.size
        multiple = components.vae_scale_factor * 2
        width, height = calculated_width // multiple * multiple, calculated_height // multiple * multiple
        processor = components.image_mask_processor
        crops = None
        resize_mode = "default"
        if block.padding_mask_crop is not None:
            crops = processor._mask_processor.get_crop_region(
                block.mask_image, width, height, pad=block.padding_mask_crop,
            )
            resize_mode = "fill"
        # Same two processor calls as the pinned whole EditInpaint pipeline;
        # the image retains calculated dimensions and the mask uses packed ones.
        block.processed_image = processor._image_processor.preprocess(
            image, height=calculated_height, width=calculated_width,
            crops_coords=crops, resize_mode=resize_mode,
        ).to(dtype=torch.float32)
        block.processed_mask_image = processor._mask_processor.preprocess(
            block.mask_image, height=height, width=width,
            crops_coords=crops, resize_mode=resize_mode,
        )
        block.mask_overlay_kwargs = {
            "crops_coords": crops,
            "original_image": image if crops is not None else None,
            "original_mask": block.mask_image if crops is not None else None,
        }
        block.qwen_inpaint_compatibility_state = snapshot
        self.set_block_state(state, block)
        return components, state


class QwenWholeInpaintVaeEncoderStep(QwenImageVaeEncoderStep):
    @property
    def inputs(self):
        return super().inputs + [_compatibility_param()]

    @torch.no_grad()
    def __call__(self, components, state):
        block = self.get_block_state(state)
        if validate_compatibility(block.inpaint_compatibility) == "native":
            return super().__call__(components, state)
        image = getattr(block, self._image_input_name)
        if not isinstance(image, torch.Tensor) or image.ndim not in (4, 5):
            raise ValueError("Qwen whole_v1 requires a 4D or 5D processed image tensor.")
        if image.ndim == 4:
            image = image.unsqueeze(2)
        image = image.to(device=components._execution_device, dtype=components.vae.dtype)
        # Sample with the caller's generator, and retain BF16 operation order
        # (subtract then multiply reciprocal), not the native mode/division.
        latents = retrieve_latents(components.vae.encode(image), generator=block.generator)
        mean = torch.tensor(components.vae.config.latents_mean).view(
            1, components.vae.config.z_dim, 1, 1, 1,
        ).to(latents.device, latents.dtype)
        reciprocal_std = 1.0 / torch.tensor(components.vae.config.latents_std).view(
            1, components.vae.config.z_dim, 1, 1, 1,
        ).to(latents.device, latents.dtype)
        setattr(block, self._image_latents_output_name, (latents - mean) * reciprocal_std)
        self.set_block_state(state, block)
        return components, state


class QwenWholeInpaintPostprocessStep(QwenImageInpaintProcessImagesOutputStep):
    @property
    def inputs(self):
        return super().inputs + [InputParam(COMPATIBILITY_STATE)]

    @torch.no_grad()
    def __call__(self, components, state):
        snapshot = state.get(COMPATIBILITY_STATE)
        if snapshot is None:
            return super().__call__(components, state)
        if type(snapshot) is not QwenInpaintCompatibilityState:
            raise ValueError("Qwen whole_v1 decode requires its backend-issued media snapshot.")
        block = self.get_block_state(state)
        if block.output_type != "pil":
            raise ValueError("Qwen whole_v1 final mask compositing requires PIL output.")
        components, state = super().__call__(components, state)
        # Shared app helper preserves soft mask pixels and source outside mask.
        from modules.DiffusersImage.main import composite_masked_pil_outputs

        source, mask = snapshot.images()
        block = self.get_block_state(state)
        block.images = composite_masked_pil_outputs(block.images, source, mask)
        self.set_block_state(state, block)
        return components, state


class QwenWholeInpaintRoPEInputsStep(QwenImageEditRoPEInputsStep):
    """Apply whole inpaint's mask convention after genuine input/RoPE work.

    The pinned whole encode_prompt removes an all-valid mask after expansion.
    Native input preparation still needs the tensor. Only a backend-issued
    whole_v1 media snapshot selects this convention; ordinary Edit and padded
    masks retain their original behavior.
    """

    @property
    def inputs(self):
        return super().inputs + [InputParam(COMPATIBILITY_STATE)]

    @torch.no_grad()
    def __call__(self, components, state):
        snapshot = state.get(COMPATIBILITY_STATE)
        if snapshot is not None and type(snapshot) is not QwenInpaintCompatibilityState:
            raise ValueError("Qwen whole_v1 denoise requires its backend-issued media snapshot.")
        components, state = super().__call__(components, state)
        if snapshot is not None:
            for name in ("prompt_embeds_mask", "negative_prompt_embeds_mask"):
                mask = state.get(name)
                if mask is not None and mask.all():
                    state.set(name, None, OutputParam.template(name).kwargs_type)
        return components, state


def prepare_qwen_edit_inpaint_blocks(blocks):
    """Adapt fresh exact inpaint leaves; sibling/default behavior is preserved."""
    if type(blocks) is not QwenImageEditAutoBlocks:
        raise ValueError("Qwen inpaint compatibility requires the reviewed AutoBlocks blueprint.")
    # Upstream SequentialPipelineBlocks definitions include instantiated
    # leaves. A newly constructed root can still share those child objects.
    blocks = deepcopy(blocks)
    encoder = blocks.sub_blocks.get("vae_encoder")
    decoder = blocks.sub_blocks.get("decode")
    denoiser = blocks.sub_blocks.get("denoise")
    if type(encoder) is not QwenImageEditAutoVaeEncoderStep or type(decoder) is not QwenImageEditAutoDecodeStep:
        raise ValueError("The reviewed Qwen Edit stage selectors have changed.")
    if type(denoiser) is not QwenImageEditAutoCoreDenoiseStep:
        raise ValueError("The reviewed Qwen Edit denoise selector has changed.")
    encode = encoder.sub_blocks.get("edit_inpaint")
    decode = decoder.sub_blocks.get("inpaint_decode")
    denoise = denoiser.sub_blocks.get("edit_inpaint")
    if (
        type(encode) is not QwenImageEditInpaintVaeEncoderStep
        or list(encode.block_names) != ["resize", "preprocess", "encode"]
        or type(encode.sub_blocks.get("preprocess")) is not QwenImageEditInpaintProcessImagesInputStep
        or type(encode.sub_blocks.get("encode")) is not QwenImageVaeEncoderStep
        or type(decode) is not QwenImageEditInpaintDecodeStep
        or list(decode.block_names) != ["decode", "postprocess"]
        or type(decode.sub_blocks.get("postprocess")) is not QwenImageInpaintProcessImagesOutputStep
        or type(denoise) is not QwenImageEditInpaintCoreDenoiseStep
        or type(denoise.sub_blocks.get("prepare_rope_inputs")) is not QwenImageEditRoPEInputsStep
    ):
        raise ValueError("The reviewed Qwen Edit inpainting leaf contracts have changed.")
    for branch, replacements in (
        (encode, {"preprocess": QwenWholeInpaintPreprocessStep(), "encode": QwenWholeInpaintVaeEncoderStep()}),
        (decode, {"postprocess": QwenWholeInpaintPostprocessStep()}),
        (denoise, {"prepare_rope_inputs": QwenWholeInpaintRoPEInputsStep()}),
    ):
        classes = list(branch.block_classes)
        for name, replacement in replacements.items():
            classes[list(branch.block_names).index(name)] = replacement
            branch.sub_blocks[name] = replacement
        branch.block_classes = classes
    return blocks
