"""Static storage admission for one pinned, sequential tiled video upscaler.

This is neither an activation peak nor permission for its generating model.
The existing owner schedule, actual free-memory resample and policy floors apply.
"""
from __future__ import annotations

import json
import math

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
_WAN_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
_WAN_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
# Reviewed Generate/GenerateSequence inputs. An added execution input requires
# reviewing this storage contract; disconnected optional media inputs are inert.
_SEQUENCE_FIELDS = set("""pipeline prompt negative_prompt video mask reference_images conditioning_scale width height
num_frames num_inference_steps guidance_scale guidance_scale_2 use_guidance_scale_2 num_videos_per_prompt seed latents
prompt_embeds negative_prompt_embeds output_type attention_kwargs_json callback_on_step_end_tensor_inputs max_sequence_length
video_contract mode control_video frame_rate strength denoise_strength last_image reference_video reference_strength
reference_downscale_factor conditioning_attention_strength framepack_sampling latent_window_size true_cfg_scale
secondary_guidance_scale scheduler_flow_shift pose_video face_video background_video segment_frame_length
previous_conditioning_frames motion_encode_batch_size temporal_tile_size temporal_overlap temporal_overlap_condition_strength
adain_factor prompt_segments_json pag_scale pag_adaptive_scale prompts_json""".split())


def _sequence_video_upscaler_owner(nodes, node_id, compose_id, planned, controls, *, literal, number, resolve_artifact):
    compose = nodes[compose_id]
    cp = compose.get("params", {})
    if (set(cp) - {*(f"clip_{i}" for i in range(1, 7)), "transition_seconds", "fps"}
            or any(cp.get(f"clip_{i}", {}).get("sourceId") or literal(nodes, compose_id, f"clip_{i}") not in (None, "")
                   for i in range(2, 7))):
        raise ValueError("Sequence upscaler storage requires one complete generated collection and no additional clips.")
    for field, lower, upper in (("fps", 1, 120), ("transition_seconds", 0, 2)):
        if cp.get(field, {}).get("sourceId"):
            raise ValueError("Sequence composition needs literal finite FPS and crossfade controls.")
        value = literal(nodes, compose_id, field)
        if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError("Sequence composition needs literal finite FPS and crossfade controls.")
    clip = cp.get("clip_1", {})
    sequence_id = clip.get("sourceId")
    sequence = nodes.get(sequence_id, {})
    sp = sequence.get("params", {})
    if ((sequence.get("module"), sequence.get("action"), clip.get("sourceKey"))
            != ("modules.DiffusersVideo", "GenerateSequence", "clips")
            or set(sp) - _SEQUENCE_FIELDS
            or any(value.get("sourceId") for key, value in sp.items() if key != "pipeline")
            or literal(nodes, sequence_id, "mode") != "text_to_video"
            or literal(nodes, sequence_id, "output_type") != "pil"):
        raise ValueError("Sequence upscaler storage needs the unchanged native text-to-video PIL collection.")
    for field in ("video", "mask", "reference_images", "latents", "prompt_embeds", "negative_prompt_embeds",
                  "control_video", "last_image", "reference_video", "pose_video", "face_video", "background_video"):
        if literal(nodes, sequence_id, field) not in (None, ""):
            raise ValueError("Conditioned sequence inputs need a separate reviewed storage bound.")
    for field in ("batch_size", "num_videos_per_prompt", "num_images_per_prompt"):
        value = literal(nodes, sequence_id, field)
        if value is not None and (type(value) is not int or value != 1):
            raise ValueError("Sequence upscaler storage has no multi-video batch bound.")
    dimensions = [literal(nodes, sequence_id, key) for key in ("width", "height", "num_frames")]
    if any(type(value) is not int or value <= 0 or number(value) is None for value in dimensions):
        raise ValueError("Sequence upscaler storage needs literal positive native geometry and frame count.")
    width, height, per_shot = dimensions
    raw = literal(nodes, sequence_id, "prompts_json")
    try:
        bounded_shots = isinstance(raw, str) and len(raw.encode("utf-8")) <= 65536
    except UnicodeError:
        bounded_shots = False
    if not bounded_shots:
        raise ValueError("Sequence storage needs a finite literal shot list.")
    try:
        shots = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Sequence storage needs a valid JSON shot list.") from exc
    if not isinstance(shots, list) or not 2 <= len(shots) <= 6:
        raise ValueError("Sequence storage follows the native two-to-six-shot contract.")
    for shot in shots:
        if isinstance(shot, str) and shot.strip():
            continue
        if (not isinstance(shot, dict) or set(shot) - {"prompt", "seed"}
                or not isinstance(shot.get("prompt"), str) or not shot["prompt"].strip()
                or shot.get("seed") is not None and type(shot["seed"]) is not int):
            raise ValueError("Sequence storage requires prompt strings or reviewed prompt/seed objects.")
    owners = [owner for owner in planned if owner.get("modelType") == "WanVideoPipeline"
              and owner.get("repository") == _WAN_REPO and owner.get("mode") == "text_to_video"
              and owner.get("settings", {}).get("executionProfileId") == "wan-text-to-video:direct"
              and owner.get("consumers") == [sequence_id]]
    pipeline = sp.get("pipeline", {})
    if len(owners) != 1 or (pipeline.get("sourceId"), pipeline.get("sourceKey")) != (owners[0]["nodeId"], "pipeline"):
        raise ValueError("Sequence storage requires one accepted generating owner with one sequence consumer.")
    loader_id = owners[0]["nodeId"]
    loader = nodes[loader_id]
    from modiff.model_artifact_catalog import catalog_revision
    if ((loader.get("module"), loader.get("action")) != ("modules.DiffusersVideo", "LoadPipeline")
            or any(loader.get("params", {}).get(key, {}).get("sourceId")
                   for key in ("pipeline_class", "model_id", "revision"))
            or literal(nodes, loader_id, "pipeline_class") != "WanPipeline"
            or literal(nodes, loader_id, "model_id") != {"source": "hub", "value": _WAN_REPO}
            or catalog_revision(_WAN_REPO) != _WAN_REVISION
            or literal(nodes, loader_id, "revision") not in (None, "", _WAN_REVISION)):
        raise ValueError("The generating pipeline differs from the reviewed immutable sequence route.")
    resolve_artifact(literal(nodes, node_id, "model_id"))
    frames = per_shot * len(shots)
    # Retain the original RGB shots and composed RGB frames, float32 input and
    # x2 output tensors, and converted x2 PIL outputs conservatively together.
    # Add a complete float32 x2 assembly. This is known storage, not an
    # activation/loading peak; owner eviction, policy floors and Run resampling
    # remain necessary. Composition cannot increase the sum of source frames.
    host = _WEIGHT_BYTES + width * height * 3 * (26 * frames + 4 * 4)
    return _storage_owner(node_id, controls, width, height, frames, host,
                          candidate_id="reviewed-spandrel-x2-video-sequence-storage-v1",
                          extra_evidence={"shotCount": len(shots), "framesPerShot": per_shot,
                                          "compositionSource": "native-pil-sequence-collection"})


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
    from modiff.native_video_storage import native_video_storage_geometry
    staged = native_video_storage_geometry(nodes, decode_id, image.get('sourceKey'), planned,
                                           literal=literal, number=number)
    if staged is not None:
        width, height, frames = (staged[name] for name in ('width', 'height', 'frames'))
        resolve_artifact(literal(nodes, node_id, 'model_id'))
        # Retain original shots, composed RGB frames, input/output FP32 tensors,
        # converted x2 frames and a complete extra tiled assembly together.
        host = _WEIGHT_BYTES + width * height * 3 * 4 * (9 * frames + 16)
        return _storage_owner(node_id, controls, width, height, frames, host,
            candidate_id='reviewed-spandrel-x2-native-video-storage-v1',
            extra_evidence={'shotCount': staged['shotCount'], 'frameCountUpperBound': True,
                            'compositionSource': staged.get('sourceKind', 'explicit-native-video-stages')})
    if ((decode.get("module"), decode.get("action"), image.get("sourceKey"))
            == ("modules.Video", "Compose", "video")):
        return _sequence_video_upscaler_owner(nodes, node_id, decode_id, planned, controls,
                                             literal=literal, number=number, resolve_artifact=resolve_artifact)
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
    return _storage_owner(node_id, controls, width, height, frames, host,
                          candidate_id="reviewed-spandrel-x2-video-tiled-storage-v1")


def _storage_owner(node_id, controls, width, height, frames, host, *, candidate_id, extra_evidence=None):
    device = controls["device"]
    return {"nodeId": node_id, "resourceOwnerKind": "auxiliary_model", "modelType": "SpandrelImageUpscale",
            "mode": "image_upscale", "repository": REVIEWED_VIDEO_UPSCALER["repository"],
            "candidateId": candidate_id, "proofStatus": "declared_safe",
            "consumers": [], "capacityRequirements": {}, "workingMemoryPolicy": "runtime_headroom_policy",
            "requirements": {"systemRamBytes": host, "vramBytes": 0 if device == "cpu" else _WEIGHT_BYTES, "diskFreeBytes": 0},
            "artifact": dict(REVIEWED_VIDEO_UPSCALER),
            "workingDemandEvidence": {"kind": "reviewed_static_storage", "weightStorageBytes": _WEIGHT_BYTES,
                "inputWidth": width, "inputHeight": height, "frameCount": frames, "scale": 2, "dtype": "float32",
                "inferencePeakMeasured": False, **(extra_evidence or {})},
            "settings": {"device": device, "dtype": "float32", "offloadMode": "none", "autoOffload": False,
                         "quantizationMode": "none", **controls}}
