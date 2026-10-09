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
"""Prompt-method-only adapters for the pinned whole Qwen editing pipelines.

Canonical class names intentionally match the SDK: model-index serialization,
admission, saved recipes, LoRA and offload identity remain the declared pipeline.
No encoder method is replaced, and all other pipeline methods are inherited.
"""

from diffusers import (
    QwenImageEditPipeline as SDKEditPipeline,
    QwenImageEditPlusPipeline as SDKEditPlusPipeline,
    QwenImageEditInpaintPipeline as SDKEditInpaintPipeline,
)
from modules.DiffusersRuntime.qwen_vl import qwen_edit_prompt_embeds


class _QwenEditVLPrompt:
    _modiff_image_template = None

    def _get_qwen_prompt_embeds(self, prompt=None, image=None, device=None, dtype=None):
        return qwen_edit_prompt_embeds(
            self.text_encoder, self.processor, prompt, image,
            prompt_template=self.prompt_template_encode,
            drop_idx=self.prompt_template_encode_start_idx,
            extract_masked_hidden=self._extract_masked_hidden,
            device=device or self._execution_device,
            dtype=dtype or self.text_encoder.dtype,
            image_template=self._modiff_image_template,
        )


class QwenImageEditPipeline(_QwenEditVLPrompt, SDKEditPipeline):
    pass


class QwenImageEditPlusPipeline(_QwenEditVLPrompt, SDKEditPlusPipeline):
    _modiff_image_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"


class QwenImageEditInpaintPipeline(_QwenEditVLPrompt, SDKEditInpaintPipeline):
    pass


QWEN_EDIT_PIPELINE_CLASSES = {
    "QwenImageEditPipeline": QwenImageEditPipeline,
    "QwenImageEditPlusPipeline": QwenImageEditPlusPipeline,
    "QwenImageEditInpaintPipeline": QwenImageEditInpaintPipeline,
}
