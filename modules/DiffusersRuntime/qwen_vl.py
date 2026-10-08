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
"""Exact Qwen Edit VL call and packing shared by native and whole adapters.

Derived from the pinned Diffusers Qwen Edit/Edit Plus prompt helpers. Only an
existing processor field is added, when the actual encoder explicitly accepts
it. Shared encoders, their forward methods, hooks and configuration are untouched.
"""

import inspect


def qwen_vl_encoder_outputs(text_encoder, model_inputs, *, image_conditioned):
    """Preserve the SDK call, adding supported processor multimodal token types."""
    fields = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "pixel_values": model_inputs.get("pixel_values"),
        "image_grid_thw": model_inputs.get("image_grid_thw"),
        "output_hidden_states": True,
    }
    mm = model_inputs.get("mm_token_type_ids")
    try:
        parameter = inspect.signature(text_encoder.forward).parameters.get("mm_token_type_ids")
    except (TypeError, ValueError):
        parameter = None
    if image_conditioned and mm is not None and parameter is not None and parameter.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    ):
        fields["mm_token_type_ids"] = mm
    return text_encoder(**fields)


def qwen_edit_prompt_embeds(
    text_encoder, processor, prompt, image, *, prompt_template, drop_idx,
    extract_masked_hidden, device, dtype=None, image_template=None,
):
    """Retain the pinned Edit/Plus formatting, processing, crop, padding and cast."""
    import torch

    prompt = [prompt] if isinstance(prompt, str) else prompt
    image_prefix = ""
    if image_template is not None:
        if isinstance(image, list):
            for i, _image in enumerate(image):
                image_prefix += image_template.format(i + 1)
        elif image is not None:
            image_prefix = image_template.format(1)
    text = [
        prompt_template.format(value if image_template is None else image_prefix + value) for value in prompt
    ]
    model_inputs = processor(text=text, images=image, padding=True, return_tensors="pt").to(device)
    outputs = qwen_vl_encoder_outputs(text_encoder, model_inputs, image_conditioned=image is not None)
    hidden = extract_masked_hidden(outputs.hidden_states[-1], model_inputs["attention_mask"])
    hidden = [value[drop_idx:] for value in hidden]
    masks = [torch.ones(value.size(0), dtype=torch.long, device=value.device) for value in hidden]
    max_seq_len = max(value.size(0) for value in hidden)
    embeds = torch.stack([
        torch.cat([value, value.new_zeros(max_seq_len - value.size(0), value.size(1))]) for value in hidden
    ])
    mask = torch.stack([torch.cat([value, value.new_zeros(max_seq_len - value.size(0))]) for value in masks])
    # Native helpers preserve the encoder dtype; whole helpers pass their SDK-resolved dtype.
    embeds = embeds.to(device=device) if dtype is None else embeds.to(dtype=dtype, device=device)
    return embeds, mask
