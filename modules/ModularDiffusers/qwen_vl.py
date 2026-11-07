# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
# Modified by MoDiff: forward existing processor multimodal token types.
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
"""Owned fresh-leaf adapters for the two pinned Qwen image-edit encoders."""

from copy import deepcopy

import torch
from diffusers import QwenImageEditModularPipeline, QwenImageEditPlusModularPipeline
from diffusers.modular_pipelines.qwenimage.encoders import (
    QwenImageEditTextEncoderStep as SDKEditTextEncoderStep,
    QwenImageEditPlusTextEncoderStep as SDKEditPlusTextEncoderStep,
    _extract_masked_hidden,
)
from diffusers.modular_pipelines.qwenimage.modular_blocks_qwenimage_edit import (
    QwenImageEditAutoBlocks, QwenImageEditVLEncoderStep,
)
from diffusers.modular_pipelines.qwenimage.modular_blocks_qwenimage_edit_plus import (
    QwenImageEditPlusAutoBlocks, QwenImageEditPlusVLEncoderStep,
)
from modules.DiffusersRuntime.qwen_vl import qwen_edit_prompt_embeds


class _QwenEditVLCall:
    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        self.check_inputs(block_state.prompt, block_state.negative_prompt)
        image = getattr(block_state, self._image_input)
        kwargs = {
            "prompt_template": self.prompt_template_encode,
            "drop_idx": self.prompt_template_encode_start_idx,
            "extract_masked_hidden": _extract_masked_hidden,
            "device": components._execution_device,
            "image_template": getattr(self, "img_template_encode", None),
        }
        block_state.prompt_embeds, block_state.prompt_embeds_mask = qwen_edit_prompt_embeds(
            components.text_encoder, components.processor, block_state.prompt, image, **kwargs,
        )
        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        if components.requires_unconditional_embeds:
            block_state.negative_prompt_embeds, block_state.negative_prompt_embeds_mask = qwen_edit_prompt_embeds(
                components.text_encoder, components.processor, block_state.negative_prompt or " ", image, **kwargs,
            )
        self.set_block_state(state, block_state)
        return components, state


class QwenImageEditTextEncoderStep(_QwenEditVLCall, SDKEditTextEncoderStep):
    _image_input = "resized_image"


class QwenImageEditPlusTextEncoderStep(_QwenEditVLCall, SDKEditPlusTextEncoderStep):
    _image_input = "resized_cond_image"


def prepare_qwen_edit_vl_blocks(pipeline_class, blocks):
    """Replace only the exact encoder leaf of a newly owned reviewed blueprint."""
    declarations = {
        QwenImageEditModularPipeline: (
            QwenImageEditAutoBlocks, QwenImageEditVLEncoderStep, SDKEditTextEncoderStep, QwenImageEditTextEncoderStep,
        ),
        QwenImageEditPlusModularPipeline: (
            QwenImageEditPlusAutoBlocks, QwenImageEditPlusVLEncoderStep,
            SDKEditPlusTextEncoderStep, QwenImageEditPlusTextEncoderStep,
        ),
    }
    declaration = declarations.get(pipeline_class)
    if declaration is None:
        return blocks
    root_type, encoder_type, leaf_type, replacement_type = declaration
    if type(blocks) is not root_type:
        raise ValueError("The reviewed Qwen image-edit VL encoder blueprint has changed.")
    encoder = blocks.sub_blocks.get("text_encoder")
    if (
        type(blocks) is not root_type or type(encoder) is not encoder_type
        or list(encoder.block_names) != ["resize", "encode"]
        or type(encoder.sub_blocks.get("encode")) is not leaf_type
        or len(encoder.block_classes) != 2 or type(list(encoder.block_classes)[1]) is not leaf_type
    ):
        raise ValueError("The reviewed Qwen image-edit VL encoder blueprint has changed.")
    # SDK bare constructors can retain class-level child instances. Copy the
    # reviewed blueprint so neither those definitions nor sibling trees change.
    blocks = deepcopy(blocks)
    encoder = blocks.sub_blocks["text_encoder"]
    replacement = replacement_type()
    classes = list(encoder.block_classes)
    classes[1] = replacement
    encoder.block_classes = classes
    encoder.sub_blocks["encode"] = replacement
    return blocks
