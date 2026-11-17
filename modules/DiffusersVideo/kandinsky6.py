"""Official Kandinsky 6 calls behind the shared video facade.

This module owns argument translation only. Loading, placement, cancellation and
execution continue through the existing DiffusersVideo nodes and resource owner.
"""

from math import isfinite

from .wan_vace import (
    callback_tensor_inputs,
    ensure_reference_images,
    ensure_single_prompt,
    ensure_video_list,
    none_if_blank,
)


FULL_REPO = "kandinskylab/Kandinsky-6.0-Pro-5s-Diffusers"
DISTILLED_REPO = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"


def _value(values, key, default):
    value = values.get(key)
    return default if value is None else value


def execute(node, pipeline, adapter, mode, values):
    import torch

    mode = mode.removesuffix('_with_audio')
    for field in ("encoded_inputs", "prompt_embeds", "negative_prompt_embeds", "attention_kwargs_json"):
        if none_if_blank(values.get(field)) is not None:
            raise ValueError(f"The selected video facade does not admit {field}; use its native prompt encoding.")
    for field in ("video", "mask", "control_video", "reference_video", "pose_video", "face_video", "background_video"):
        if ensure_video_list(values.get(field), field) is not None:
            raise ValueError(f"This Kandinsky 6 route does not accept {field}.")
    if none_if_blank(values.get("last_image")) is not None:
        raise ValueError("This Kandinsky 6 route does not accept an ending image.")
    images = ensure_reference_images(values.get("reference_images"))
    if (mode == "text_to_video" and images) or (mode == "image_to_video" and (not images or len(images) != 1)):
        raise ValueError("Kandinsky 6 text generation takes no images; image generation requires exactly one.")
    width, height = int(_value(values, "width", 864)), int(_value(values, "height", 480))
    frames = int(_value(values, "num_frames", 121))
    fps = float(_value(values, "frame_rate", 24.0))
    if min(width, height) <= 0 or width % 16 or height % 16:
        raise ValueError("Kandinsky 6 width and height must be positive multiples of 16.")
    temporal = int(pipeline.vae_scale_factor_temporal)
    if frames < 1 or (frames - 1) % temporal:
        raise ValueError(f"Kandinsky 6 frame count must be 1 plus a multiple of {temporal}.")
    if not isfinite(fps) or fps <= 0:
        raise ValueError("Frame rate must be finite and positive.")
    output_type = values.get("output_type", "pil")
    if output_type not in {"pil", "np", "pt"}:
        raise ValueError("A video output requires decoded frames, not latents.")
    distilled = adapter.default_repo == DISTILLED_REPO
    steps = int(_value(values, "num_inference_steps", 10 if distilled else 50))
    guidance = float(_value(values, "guidance_scale", 1.0 if distilled else 5.0))
    if steps < 1 or not isfinite(guidance):
        raise ValueError("Steps must be positive and guidance must be finite.")
    arguments = {
        "prompt": ensure_single_prompt(none_if_blank(values.get("prompt")), "prompt"),
        "negative_prompt": ensure_single_prompt(none_if_blank(values.get("negative_prompt")), "negative prompt"),
        "image": images[0] if images else None,
        "height": height,
        "width": width,
        "num_frames": frames,
        "frame_rate": fps,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
        "num_videos_per_prompt": 1,
        "generator": torch.Generator(device=pipeline._execution_device).manual_seed(int(values.get("seed", 0))),
        "sample_audio": bool(values.get("_sample_audio", False)),
        "expand_prompts": bool(values.get("expand_prompt", False)),
        "max_sequence_length": int(values.get("max_sequence_length") or 1024),
        "latents": none_if_blank(values.get("latents")),
        "output_type": output_type,
        "return_dict": True,
        "callback_on_step_end": node.pipe_callback,
        "callback_on_step_end_tensor_inputs": callback_tensor_inputs(values.get("callback_on_step_end_tensor_inputs")),
    }
    node._active_pipeline = pipeline
    try:
        output = node._invoke_pipeline(pipeline, arguments)
    finally:
        node._active_pipeline = None
    video = output.frames
    if output_type == "np":
        video = pipeline.video_processor.numpy_to_pil(video[0])
    elif output_type == "pt":
        # Official postprocessing already normalized this decoded tensor to
        # [0, 1]. Do not apply the VAE's denormalization a second time.
        video = pipeline.video_processor.postprocess(
            video[0], output_type="pil", do_denormalize=[False] * len(video[0])
        )
    elif isinstance(video, list) and len(video) == 1 and isinstance(video[0], list):
        video = video[0]
    return {
        "video_out": video,
        "width_out": width,
        "height_out": height,
        "frames_out": len(video) if isinstance(video, list) else frames,
        "_audio": output.audio,
    }
