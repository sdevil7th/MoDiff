"""Static storage admission for one pinned, sequential tiled video upscaler.

This is neither an activation peak nor permission for its generating model.
The existing owner schedule, actual free-memory resample and policy floors apply.
"""
from __future__ import annotations

REVIEWED_VIDEO_UPSCALER = {
    "repository": "nateraw/real-esrgan", "weightName": "RealESRGAN_x2plus.pth",
    "revision": "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094",
    "sha256": "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
    "byteSize": 67061725,
}
# 702 serialized FP32 RRDBNet storages in the exact immutable archive above.
# The x2 pixel-unshuffle input adds 20,736 bytes over the reviewed x4 network.
_WEIGHT_BYTES = 66812684
_REPO = "Lightricks/LTX-Video-0.9.8-13B-distilled"
_REVISION = "7c64400e1861cc0d7b98d570a1926d5408ec60cd"
_FRAMES = {"text_to_video": 121, "image_to_video": 161, "reference_to_video": 81, "video_to_video": 81}


def video_upscaler_owner(nodes, node_id, planned, hardware, *, literal, number, resolve_artifact):
    """Inspect exact ordinary stage edges and retained-frame storage, never run them."""
    params = nodes[node_id]["params"]
    controls = {key: literal(nodes, node_id, key) for key in ("downscale", "tile_size", "tile_overlap", "device")}
    if (type(controls["downscale"]) not in (int, float) or controls["downscale"] != 1
            or type(controls["tile_size"]) is not int or controls["tile_size"] != 256
            or type(controls["tile_overlap"]) is not int or controls["tile_overlap"] != 32):
        raise ValueError("The reviewed video upscaler requires downscale 1, tile size 256 and overlap 32.")
    device = controls["device"]
    kind = hardware.get("accelerator", {}).get("kind")
    if device != "cpu" and not (kind in {"cuda", "xpu", "mps"} and device in {kind, kind + ":0"}):
        raise ValueError("The video upscaler requires CPU or the current primary accelerator.")

    image = params.get("image", {})
    decode_id = image.get("sourceId")
    decode = nodes.get(decode_id, {})
    dp = decode.get("params", {})
    if ((decode.get("module"), decode.get("action"), image.get("sourceKey"))
            != ("modules.DiffusersVideo", "DecodeLatents", "video_out")
            or set(dp) != {"pipeline", "output_type", "latents", "latent_binding"}
            or literal(nodes, decode_id, "output_type") != "pil"):
        raise ValueError("The video upscaler needs the unchanged native PIL frame decoder and latent binding.")
    denoise_id = dp["latents"].get("sourceId")
    denoise = nodes.get(denoise_id, {})
    owners = [owner for owner in planned if owner["modelType"] == "LTXVideoPipeline"
              and owner["repository"] == _REPO and owner["mode"] in _FRAMES
              and {decode_id, denoise_id} <= set(owner["consumers"])]
    if ((denoise.get("module"), denoise.get("action")) != ("modules.DiffusersVideo", "Denoise")
            or dp["latents"].get("sourceKey") != "latents_out"
            or (dp["latent_binding"].get("sourceId"), dp["latent_binding"].get("sourceKey")) != (denoise_id, "latent_binding")
            or len(owners) != 1):
        raise ValueError("The video upscaler requires one accepted native video stage owner.")
    owner = owners[0]
    loader_id, mode = owner["nodeId"], owner["mode"]
    loader = nodes.get(loader_id, {})
    if ((loader.get("module"), loader.get("action")) != ("modules.DiffusersVideo", "LoadPipeline")
            or literal(nodes, loader_id, "revision") != _REVISION
            or literal(nodes, loader_id, "pipeline_class") != "LTXConditionPipeline"
            or literal(nodes, loader_id, "model_id") != {"source": "hub", "value": _REPO}):
        raise ValueError("The generating pipeline differs from the reviewed video upscaler route.")
    np = denoise["params"]
    encoded = np.get("encoded_inputs", {})
    encode_id = encoded.get("sourceId")
    encode = nodes.get(encode_id, {})
    if ((encode.get("module"), encode.get("action"), encoded.get("sourceKey"))
            != ("modules.DiffusersVideo", "EncodeInputs", "encoded_inputs")
            or encode_id not in owner["consumers"]
            or not isinstance(literal(nodes, encode_id, "prompt"), str)
            or any(literal(nodes, key, "mode") != mode for key in (encode_id, denoise_id))):
        raise ValueError("The video upscaler needs one matching single-prompt encode/denoise route.")
    for stage_id in (encode_id, denoise_id, decode_id):
        pipeline = nodes[stage_id]["params"].get("pipeline", {})
        if (pipeline.get("sourceId"), pipeline.get("sourceKey")) != (loader_id, "pipeline"):
            raise ValueError("Video upscaler stages must share their exact pipeline owner.")
    for field, expected in {"width": 768, "height": 512, "num_frames": _FRAMES[mode]}.items():
        if number(literal(nodes, denoise_id, field)) != expected:
            raise ValueError("The video upscaler storage contract requires its exact declared geometry and frame count.")
    for field in ("batch_size", "num_videos_per_prompt", "num_images_per_prompt"):
        value = literal(nodes, denoise_id, field)
        if value is not None and (type(value) not in (int, float) or value != 1):
            raise ValueError("The video upscaler has no reviewed multi-video batch bound.")
    for field in ("latents", "latents_in", "latent_binding"):
        param = np.get(field, {})
        if param.get("sourceId") or param.get("value") is not None:
            raise ValueError("Externally supplied video latents have no reviewed upscaler geometry bound.")
    # Reuse normal path containment, installed-file and complete SHA checks.
    resolve_artifact(literal(nodes, node_id, "model_id"))
    width, height, frames = 768, 512, _FRAMES[mode]
    # ImageToTensor eagerly retains every input frame; inference is sequential.
    # Include all input float32 tensors, all output frames at float32 size and
    # one complete tiled CPU assembly. PIL RGB inputs/outputs fit inside this
    # conservative storage allowance. Activations/copies still use policy floors.
    host = _WEIGHT_BYTES + width * height * 3 * 4 * (frames * (1 + 2 ** 2) + 2 ** 2)
    return {"nodeId": node_id, "resourceOwnerKind": "auxiliary_model", "modelType": "SpandrelImageUpscale",
            "mode": "image_upscale", "repository": REVIEWED_VIDEO_UPSCALER["repository"],
            "candidateId": "reviewed-spandrel-x2-video-tiled-storage-v1", "proofStatus": "declared_safe",
            "consumers": [], "capacityRequirements": {}, "workingMemoryPolicy": "runtime_headroom_policy",
            "requirements": {"systemRamBytes": host, "vramBytes": 0 if device == "cpu" else _WEIGHT_BYTES, "diskFreeBytes": 0},
            "artifact": dict(REVIEWED_VIDEO_UPSCALER),
            "workingDemandEvidence": {"kind": "reviewed_static_storage", "weightStorageBytes": _WEIGHT_BYTES,
                "inputWidth": width, "inputHeight": height, "frameCount": frames, "scale": 2, "dtype": "float32",
                "inferencePeakMeasured": False},
            "settings": {"device": device, "dtype": "float32", "offloadMode": "none", "autoOffload": False,
                         "quantizationMode": "none", **controls}}
