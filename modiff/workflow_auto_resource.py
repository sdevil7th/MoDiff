"""Resource planning for the existing concrete graph executor.

No model execution, installation, alternate graph format or source admission.
Machine capacity classes are checked against totals. Explicit working budgets
include every independent owner, with credit only for compatible live weight
storage. Unmeasured working demand uses the existing runtime headroom policy;
this is not a measured model-fit qualification or combined-fit authority for
independent models. Such models require nonoverlapping owner lifetimes and the
existing release/recheck schedule. Cache preparation is rechecked at Run.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
import time
from typing import Any, Callable

from modiff.auto_resource import READY_PROOF_STATUSES, build_auto_resource_plan, _hardware_snapshot, _requirements_missing_for_dict
from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
from modiff.huggingface_cluster_admission import REVIEWED_CLUSTER_EXECUTION_CANDIDATES
from modiff.workflow_task_identity import resource_consumers as _consumers

_WORKLOAD = {
    "width": "width", "height": "height", "num_inference_steps": "steps", "steps": "steps",
    "guidance_scale": "guidanceScale", "num_frames": "numFrames", "batch_size": "batchSize",
    "num_images_per_prompt": "batchSize", "max_sequence_length": "maxSequenceLength",
    "duration": "duration", "audio_duration": "audioDuration", "strength": "strength",
}
_SETTINGS = {"dtype": "dtype", "device": "device", "offload_mode": "offloadMode", "auto_offload": "autoOffload", "quantization_mode": "quantizationMode"}
_DATA_MODULES = {"modules.Primitive", "modules.Text", "modules.Image", "modules.ImageOperations", "modules.Audio", "modules.Video"}
from modiff.workflow_auto_values import DATA_ACTIONS as _DATA_ACTIONS


def graph_material(graph: dict) -> dict:
    nodes = graph.get("nodes")
    paths = graph.get("paths")
    if not isinstance(nodes, dict) or not nodes or len(nodes) > 2048 or not isinstance(paths, list):
        raise ValueError("Auto needs a nonempty executable graph with at most 2048 nodes.")
    if len(paths) > 2048 or any(not isinstance(path, list) or len(path) > 2048 or any(not isinstance(item, str) for item in path) for path in paths):
        raise ValueError("The executable paths exceed the Auto planning limit.")
    order = list(dict.fromkeys(item for path in paths for item in path if isinstance(item, str)))
    if set(order) != set(nodes) or sum(map(len, paths)) > 32768:
        raise ValueError("Auto paths must cover every executable node exactly by identity.")
    for node_id, node in nodes.items():
        if not isinstance(node_id, str) or not isinstance(node, dict) or not isinstance(node.get("params"), dict):
            raise ValueError("Auto received an invalid executable node.")
        if not isinstance(node.get("module"), str) or not isinstance(node.get("action"), str):
            raise ValueError("Auto received an invalid executable node identity.")
        for param in node["params"].values():
            if not isinstance(param, dict):
                raise ValueError("Auto requires the existing API parameter contract.")
            if param.get("sourceId") is not None and param["sourceId"] not in nodes:
                raise ValueError(f"{node_id} refers to a missing input source.")
    # Validate dependency ordering independently of client paths. Repeated path
    # prefixes are normal exports; dependencies must still precede consumers.
    seen: set[str] = set()
    for node_id in order:
        dependencies = {p["sourceId"] for p in nodes[node_id]["params"].values() if p.get("sourceId")}
        if not dependencies.issubset(seen):
            raise ValueError(f"{node_id} has a cycle or an input ordered after its consumer.")
        seen.add(node_id)
    material = {"nodes": nodes, "paths": paths}
    if "loops" in graph:
        material["loops"] = graph["loops"]
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode()) > 8 * 1024 * 1024:
        raise ValueError("The graph exceeds the Auto planning size limit.")
    return material


def workflow_graph_hash(graph: dict) -> str:
    return "sha256:workflow-auto-v1:" + hashlib.sha256(json.dumps(graph_material(graph), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def workflow_owner_cache_key(graph: dict, loader_id: str, resolved_fields: dict | None = None) -> str:
    """Identify a loader and upstream controls/adapters, excluding consumers.

    Consumer prompts/seeds may change without replacing model weights. Connected
    resource values are included after data preparation without executing their
    suppliers during inspection.
    """
    nodes = graph["nodes"]
    pending, selected = [loader_id], {}
    while pending:
        node_id = pending.pop()
        if node_id in selected:
            continue
        selected[node_id] = deepcopy(nodes[node_id])
        for field, param in selected[node_id]["params"].items():
            if param.get("sourceId"):
                pending.append(param["sourceId"])
                values = (resolved_fields or {}).get(node_id, {})
                if field in values:
                    param["resolvedResourceValue"] = values[field]
    encoded = json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:workflow-owner-v1:" + hashlib.sha256(encoded.encode()).hexdigest()


from modiff.workflow_auto_values import inspect_resource_value, inspect_execution_recipe_controls, inspect_native_runtime_controls, DeferredResourceValue
from contextvars import ContextVar

_RESOLVED_FIELDS = ContextVar("workflow_auto_resolved_fields", default=None)

def _literal(nodes, node_id, field, visited=frozenset()):
    value = inspect_resource_value(nodes, node_id, field, visited)
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"{node_id}.{field} did not resolve to a finite data value.") from error
    if len(encoded.encode()) > 1_048_576:
        raise ValueError(f"{node_id}.{field} exceeds the resource control size limit.")
    captured = _RESOLVED_FIELDS.get()
    if captured is not None and field in nodes[node_id]["params"]:
        captured.setdefault(node_id, {})[field] = value
    return value


def _values(nodes: dict, node_id: str) -> dict:
    return {key: param.get("value") for key, param in nodes[node_id]["params"].items() if not param.get("sourceId")}


def _repository(value: Any) -> str:
    if isinstance(value, dict):
        return value.get("value", "") if value.get("source") == "hub" else ""
    return value if isinstance(value, str) else ""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def select_memory_requirement(envelope: dict, offload_mode: str) -> dict:
    """Select the same exact offload tier for capacity or working demands."""
    if not isinstance(envelope, dict):
        return {}
    if "minimum" not in envelope:
        return envelope
    by_offload = envelope.get("offloadRequirements") or {}
    return (by_offload.get(offload_mode)
            or (envelope.get("fullResidency") if offload_mode == "none" else None)
            or envelope["minimum"])


def _native_lora_owners(graph, loaders):
    """Resolve only the native descriptor chain consumed by a reviewed loader."""
    from modiff.auxiliary_lora import native_lora_chain_ids

    nodes = graph['nodes']
    owners = {}
    for loader_id in loaders:
        loader = nodes[loader_id]
        if (loader['module'], loader['action']) != ('modules.ModularDiffusers', 'ModelsLoader'):
            continue
        param = loader['params'].get('lora_list') or {}
        terminal = param.get('sourceId')
        if terminal is None:
            if param.get('value') not in (None, []):
                raise ValueError(f'{loader_id}: inline LoRA descriptors have no inspected workflow Auto memory envelope.')
            continue
        if param.get('sourceKey') != 'lora':
            raise ValueError(f'{loader_id}: the native LoRA chain must connect its exact lora output to lora_list.')
        for descriptor_id in native_lora_chain_ids(graph, terminal):
            owners.setdefault(descriptor_id, set()).add(loader_id)
    return owners


def _workload_values(nodes: dict, node_id: str):
    """Inspect the fields actually bound by the reviewed dispatch adapters.

    Keep the original API field name in resolvedFields so dispatch can compare
    connected values without rewriting the user's persisted graph. A carried
    loop value has no static upper bound; inspecting it must never execute the
    loop or substitute its initial value for all subsequent iterations.
    """
    node = nodes[node_id]
    reviewed_step = (node["module"], node["action"]) == (
        "modules.ModularDiffusers", "ReviewedModularWorkflowStep",
    )
    prefixes = ("", "state_input__", "iteration_input__") if reviewed_step else ("",)
    for name, target in _WORKLOAD.items():
        for prefix in prefixes:
            field = prefix + name
            if field in node["params"]:
                yield field, target, _literal(nodes, node_id, field)
    if reviewed_step and "iteration_bindings" in node["params"]:
        bindings = _literal(nodes, node_id, "iteration_bindings")
        if bindings is None:
            return
        if not isinstance(bindings, dict):
            raise ValueError(f"{node_id}.iteration_bindings must be an object.")
        for field, binding in bindings.items():
            if field not in _WORKLOAD:
                continue
            if not isinstance(binding, dict) or set(binding) != {"kind", "value"} or binding["kind"] != "constant":
                raise ValueError(
                    f"{node_id}.{field} changes inside an iteration and has no reviewed Auto resource bound. "
                    "Use a constant resource input or select Expert."
                )
            yield f"iteration_bindings.{field}", _WORKLOAD[field], binding["value"]


def _candidate_covers_workload(candidate: dict, form: dict) -> bool:
    generation = candidate.get("generation") or {}
    for key in set(_WORKLOAD.values()):
        if key not in form:
            continue
        actual = _number(generation.get(key))
        expected = _number(form[key])
        if key == "batchSize":
            # Historical recipes cover one item. Missing batch evidence is
            # never permission to reuse that envelope for a larger batch.
            if actual is None:
                if generation.get(key) is not None or expected != 1:
                    return False
            elif actual != expected:
                return False
        elif actual not in {None, expected}:
            return False
    return True


_REVIEWED_UPSCALER = {
    "repository": "amd/realesrgan-x4plus", "weightName": "RealESRGAN_x4plus.pth",
    "revision": "bda69abcaf525425b371622349e975245ae090c2",
    "sha256": "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    "byteSize": 67040989,
}
# Reviewed RRDBNet x4 float32 checkpoint: sum of the 702 serialized tensor
# storages in the exact file above. This is known weight storage, not a measured
# peak inference budget or qualification of a machine. Tiled inference retains
# the existing runtime headroom policy for its unmeasured activation demand.
_REVIEWED_UPSCALER_WEIGHT_BYTES = 66791948


_REVIEWED_QWEN_CONTROLNET = {
    "repository": "InstantX/Qwen-Image-ControlNet-Union",
    "revision": "b13036f066d6dee7c20513e263d3d673055e9de8",
    "weightName": "diffusion_pytorch_model.safetensors",
    "sha256": "d51dca0073366a675108d5b83c3b7ef941cf2214c9a1c95c23f1e9a228ddbdb0",
    "byteSize": 3536027816,
    "configSha256": "6d80a5e2ab3abeb773b06f917b54a72f60c35ce91091f59f79243fb0f5d3d4bd",
    "headerSha256": "82eea2a342aeb3d99dc32f9c084458d5d63d85f1c33b04943199f151affaa8ae",
}
_REVIEWED_QWEN_CONTROLNET_WEIGHT_BYTES = 3536007168


def _inspect_reviewed_controlnet_artifact():
    """Verify one immutable cached component, without importing/loading it."""
    from modiff.controlled_artifacts import _managed_hub_file, _sha256_file

    artifact = _REVIEWED_QWEN_CONTROLNET
    config = _managed_hub_file(artifact["repository"], artifact["revision"], "config.json")
    if config.stat().st_size != 491 or _sha256_file(config) != artifact["configSha256"]:
        raise ValueError("The reviewed Qwen ControlNet config bytes changed.")
    path = _managed_hub_file(artifact["repository"], artifact["revision"], artifact["weightName"])
    if path.stat().st_size != artifact["byteSize"] or _sha256_file(path) != artifact["sha256"]:
        raise ValueError("The reviewed Qwen ControlNet weight bytes changed.")
    # The exact SHA binds all bytes; additionally verify the bounded reviewed
    # header that establishes the 181 BF16 tensor storages. No tensor/model is
    # deserialized. Loading copies/activations remain unmeasured policy demand.
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        if length != 20640 or hashlib.sha256(handle.read(length)).hexdigest() != artifact["headerSha256"]:
            raise ValueError("The reviewed Qwen ControlNet tensor storage header changed.")
    return {"weightStorageBytes": _REVIEWED_QWEN_CONTROLNET_WEIGHT_BYTES,
            "tensorCount": 181, "dtype": "bfloat16", "inferencePeakMeasured": False}


def _controlnet_owner(nodes, node_id, planned, hardware):
    """Bound the published Qwen control route and its separate weight owner."""
    params = nodes[node_id]["params"]
    expected = {"model_type": "controlnet", "model_id": {"source": "hub", "value": _REVIEWED_QWEN_CONTROLNET["repository"]},
                "revision": _REVIEWED_QWEN_CONTROLNET["revision"], "dtype": "bfloat16", "subfolder": "",
                "variant": "", "component_class": "", "trust_remote_code": False, "auto_offload": False,
                "offload_mode": "none"}
    if set(params) != {*expected, "device"} or any(
        _literal(nodes, node_id, key) != value
        or isinstance(value, bool) and type(_literal(nodes, node_id, key)) is not bool
        for key, value in expected.items()
    ):
        raise ValueError("Auto covers only the exact reviewed BF16 Qwen ControlNet Hub component and unchanged placement.")
    if any(params[key].get("sourceId") for key in params):
        raise ValueError("The reviewed auxiliary ControlNet selector and placement must be literal values.")
    device = _literal(nodes, node_id, "device")
    if hardware.get("accelerator", {}).get("kind") != "cuda" or device not in {"cuda", "cuda:0"}:
        raise ValueError("Auto covers this Qwen ControlNet only on the current primary CUDA device.")
    # API parameters also carry UI display metadata. Match executable edges,
    # rather than requiring their dictionaries to have no presentation keys.
    owners = [owner for owner in planned if owner["modelType"] == "QwenImageModularPipeline"
              and owner["mode"] == "control_image" and owner["repository"] == "Qwen/Qwen-Image-2512"
              and (nodes[owner["nodeId"]]["params"].get("controlnet", {}).get("sourceId"),
                   nodes[owner["nodeId"]]["params"].get("controlnet", {}).get("sourceKey")) == (node_id, "model")]
    if len(owners) != 1:
        raise ValueError("The auxiliary ControlNet needs exactly one accepted Qwen 2512 control-image model owner.")
    owner = owners[0]; loader_id = owner["nodeId"]
    from modiff.workflow_task_identity import modular_graph_tasks
    if modular_graph_tasks(nodes, loader_id, owner["consumers"], owner["modelType"]) != {"control_image"}:
        raise ValueError("The auxiliary ControlNet requires the complete reviewed control-image executable task.")
    if (_literal(nodes, loader_id, "revision") != "25468b98e3276ca6700de15c6628e51b7de54a26"
            or _literal(nodes, loader_id, "dtype") != "bfloat16"
            or _literal(nodes, loader_id, "device") != device
            or _literal(nodes, loader_id, "workflow_id") != "controlnet_text2image"):
        raise ValueError("The ControlNet owner identity, precision, device or workflow differs from the reviewed route.")
    members = set(owner["consumers"])
    def exact_action(action):
        ids = [key for key in members if (nodes[key]["module"], nodes[key]["action"]) == ("modules.ModularDiffusers", action)]
        if len(ids) != 1:
            raise ValueError("The reviewed auxiliary ControlNet needs one complete control, denoise and decode branch.")
        return ids[0]
    control_id, denoise_id, decode_id = (exact_action(action) for action in ("Controlnet", "Denoise", "DecodeLatents"))
    required_edges = [(loader_id, "controlnet", node_id, "model"), (control_id, "controlnet", node_id, "model"),
                      (control_id, "vae", loader_id, "vae_out"), (denoise_id, "unet", loader_id, "unet_out"),
                      (denoise_id, "scheduler", loader_id, "scheduler"),
                      (denoise_id, "controlnet_bundle", control_id, "controlnet_bundle"),
                      (denoise_id, "route_state_in", control_id, "route_state_out"),
                      (decode_id, "vae", loader_id, "vae_out"), (decode_id, "latents", denoise_id, "latents"),
                      (decode_id, "route_state_in", denoise_id, "route_state_out")]
    if any((nodes[target]["params"].get(field, {}).get("sourceId"),
            nodes[target]["params"].get(field, {}).get("sourceKey")) != (source, output)
           for target, field, source, output in required_edges):
        raise ValueError("The auxiliary ControlNet component, VAE or sealed route edges differ from the reviewed branch.")
    fanout = {(target, field, p.get("sourceKey")) for target, node in nodes.items()
              for field, p in node["params"].items() if p.get("sourceId") == node_id}
    if fanout != {(loader_id, "controlnet", "model"), (control_id, "controlnet", "model")}:
        raise ValueError("The auxiliary ControlNet has unreviewed consumers or output ports.")
    control = nodes[control_id]["params"]
    if set(control) - {"model_type", "control_image", "controlnet_conditioning_scale", "control_guidance_start",
                       "control_guidance_end", "height", "width", "seed", "route_state_in", "controlnet", "vae"}:
        raise ValueError("The ControlNet action has unreviewed additional inputs.")
    image = control.get("control_image", {})
    source = nodes.get(image.get("sourceId"), {})
    if ((source.get("module"), source.get("action"), image.get("sourceKey")) != ("modules.Image", "Load", "image")
            or _literal(nodes, control_id, "model_type") != "QwenImageModularPipeline"
            or control.get("route_state_in", {}).get("sourceId")
            or control.get("route_state_in", {}).get("value") is not None):
        raise ValueError("Auto needs the reviewed built-in control image and initial ControlNet route.")
    denoise = nodes[denoise_id]["params"]
    if (denoise.get("image_latents", {}).get("sourceId") or denoise.get("image_latents", {}).get("value") is not None
            or any(_literal(nodes, denoise_id, field) not in (None, 1) for field in ("batch_size", "num_images_per_prompt"))):
        raise ValueError("The auxiliary ControlNet storage contract covers only the reviewed single-image control branch.")
    for field in ("width", "height"):
        value = _number(_literal(nodes, control_id, field))
        if value is None or value != _number(_literal(nodes, denoise_id, field)) or value != 768:
            raise ValueError("The auxiliary ControlNet geometry or originating seed differs from its denoising route.")
    def exact_seed(target):
        value = _literal(nodes, target, "seed")
        if type(value) is float and math.isfinite(value) and value.is_integer():
            value = int(value)
        if type(value) is not int or not 0 <= value < 2 ** 64:
            raise ValueError("The reviewed ControlNet seed must be an exact unsigned 64-bit integer.")
        return value
    # Seeds are identities, not approximate resource amounts: converting
    # integer seeds to float would collapse adjacent values above 2**53.
    if exact_seed(control_id) != exact_seed(denoise_id):
        raise ValueError("The ControlNet originating seed differs from its denoising route.")
    scale, start, end = (_number(_literal(nodes, control_id, field)) for field in (
        "controlnet_conditioning_scale", "control_guidance_start", "control_guidance_end"))
    if scale is None or scale > 2 or start is None or end is None or not 0 <= start <= end <= 1:
        raise ValueError("The ControlNet strength or guidance window is outside its reviewed finite range.")
    evidence = _inspect_reviewed_controlnet_artifact()
    storage = evidence["weightStorageBytes"]
    return {"nodeId": node_id, "resourceOwnerKind": "auxiliary_model", "sharedWithModelOwnerId": loader_id,
            "modelType": "QwenImageControlNetModel", "mode": "control_image", "repository": _REVIEWED_QWEN_CONTROLNET["repository"],
            "candidateId": "reviewed-qwen-union-bf16-storage-v1", "proofStatus": "declared_safe",
            "consumers": sorted({loader_id, *members}), "capacityRequirements": {},
            "requirements": {"systemRamBytes": storage, "vramBytes": storage, "diskFreeBytes": 0},
            "workingMemoryPolicy": "runtime_headroom_policy", "artifact": dict(_REVIEWED_QWEN_CONTROLNET),
            "workingDemandEvidence": {"kind": "reviewed_static_storage", **evidence},
            "settings": {"device": device, "dtype": "bfloat16", "offloadMode": "none", "autoOffload": False,
                         "quantizationMode": "none"}}


def _upscaler_owner(nodes, node_id, planned, hardware):
    """Inspect the bounded built-in image branch without loading any model."""
    from modiff.controlled_artifacts import portable_upscaler_selection, resolve_upscaler_artifact

    params = nodes[node_id]["params"]
    if set(params) - {"image", "model_id", "downscale", "tile_size", "tile_overlap", "device"}:
        raise ValueError("Auto has no reviewed upscaler bound for these additional inputs; use Custom memory policy.")
    selection = _literal(nodes, node_id, "model_id")
    if not isinstance(selection, dict) or set(selection) - {"source", "value", "revision", "sha256", "byteSize", "license"}:
        raise ValueError("Auto needs the exact reviewed Hub upscaler selector.")
    if portable_upscaler_selection(selection) != _REVIEWED_UPSCALER:
        raise ValueError("This upscaler artifact has no reviewed workflow Auto storage contract; use Custom memory policy.")
    controls = {field: _literal(nodes, node_id, field) for field in ("downscale", "tile_size", "tile_overlap", "device")}
    if (controls["downscale"] != .5 or type(controls["tile_size"]) is not int or controls["tile_size"] != 256
            or type(controls["tile_overlap"]) is not int or controls["tile_overlap"] != 32):
        raise ValueError("Auto covers this upscaler only with tile size 256, overlap 32 and downscale 0.5; use Custom for other settings.")
    device = controls["device"]
    kind = hardware.get("accelerator", {}).get("kind")
    if device != "cpu" and not (kind in {"cuda", "xpu", "mps"} and device in {kind, kind + ":0"}):
        raise ValueError("Auto needs an explicit CPU or the current primary accelerator for the reviewed upscaler.")

    image = params.get("image", {})
    decode_id = image.get("sourceId")
    decode = nodes.get(decode_id, {})
    if (image.get("sourceKey") != "images" or (decode.get("module"), decode.get("action"))
            != ("modules.ModularDiffusers", "DecodeLatents")):
        raise ValueError("Auto can bound this upscaler only from a reviewed native text-to-image DecodeLatents output.")
    latents = decode.get("params", {}).get("latents", {})
    denoise_id = latents.get("sourceId")
    denoise = nodes.get(denoise_id, {})
    from modiff.workflow_task_identity import modular_graph_tasks
    owners = [owner for owner in planned if {decode_id, denoise_id} <= set(owner["consumers"])
              and modular_graph_tasks(nodes, owner["nodeId"], owner["consumers"], owner["modelType"]) == {"text_to_image"}]
    if (latents.get("sourceKey") != "latents" or (denoise.get("module"), denoise.get("action"))
            != ("modules.ModularDiffusers", "Denoise") or len(owners) != 1):
        raise ValueError("The upscaler input needs one accepted native text-to-image model owner.")
    width, height = (_number(_literal(nodes, denoise_id, field)) for field in ("width", "height"))
    if any(value is None or value <= 0 or value > 1024 or value % 64 for value in (width, height)):
        raise ValueError("Auto bounds this upscaler to a single generated image of at most 1024 by 1024, with dimensions divisible by 64.")
    if any(field in decode["params"] and _number(_literal(nodes, decode_id, field)) not in (None, expected)
           for field, expected in (("width", width), ("height", height))):
        raise ValueError("The decoded image dimensions differ from the reviewed upscaler input bound.")
    denoise_params = denoise["params"]
    if any(_literal(nodes, denoise_id, field) not in (None, 1) for field in ("batch_size", "num_images_per_prompt")):
        raise ValueError("Auto has no reviewed upscaler bound for multiple generated images.")
    if denoise_params.get("image_latents", {}).get("sourceId") or denoise_params.get("image_latents", {}).get("value") is not None:
        raise ValueError("Conditioned image geometry needs its own reviewed upscaler bound.")
    embeddings = denoise_params.get("embeddings", {})
    encode_id = embeddings.get("sourceId")
    encode = nodes.get(encode_id, {})
    if (embeddings.get("sourceKey") != "embeddings" or (encode.get("module"), encode.get("action"))
            != ("modules.ModularDiffusers", "EncodePrompt") or encode_id not in owners[0]["consumers"]
            or not isinstance(_literal(nodes, encode_id, "prompt"), str)):
        raise ValueError("The upscaler input needs the reviewed single-prompt text-to-image encoding branch.")
    # Containment, size and SHA validation reuse the normal execution resolver.
    # A complete pin never resolves mutable refs or downloads missing files.
    resolve_upscaler_artifact(selection)
    width, height = int(width), int(height)
    weight_bytes = _REVIEWED_UPSCALER_WEIGHT_BYTES
    # Known float32 input and assembled CPU output storage. Activations, loading
    # transients and conversion copies are unmeasured and covered by the policy
    # floor, not falsely presented as a measured upper bound.
    host_storage = weight_bytes + width * height * 3 * 4 * (1 + 4 ** 2)
    return {"nodeId": node_id, "resourceOwnerKind": "auxiliary_model", "modelType": "SpandrelImageUpscale",
            "mode": "image_upscale", "repository": _REVIEWED_UPSCALER["repository"],
            "candidateId": "reviewed-spandrel-x4-tiled-storage-v1", "proofStatus": "declared_safe",
            "consumers": [], "requirements": {"systemRamBytes": host_storage,
                "vramBytes": 0 if device == "cpu" else weight_bytes, "diskFreeBytes": 0},
            "capacityRequirements": {}, "workingMemoryPolicy": "runtime_headroom_policy",
            "artifact": dict(_REVIEWED_UPSCALER),
            "workingDemandEvidence": {"kind": "reviewed_static_storage", "weightStorageBytes": weight_bytes,
                "inputWidth": width, "inputHeight": height, "scale": 4, "dtype": "float32",
                "inferencePeakMeasured": False},
            "settings": {"device": device, "dtype": "float32", "offloadMode": "none", "autoOffload": False,
                         "quantizationMode": "none", **controls}}


def _build_workflow_auto_plan(
    graph: dict, *, runtime_fingerprint: dict, local_models: list | None, data_dir: str,
    plan_recipe: Callable = build_auto_resource_plan, hardware: dict | None = None,
    cache_snapshot: dict | None = None,
) -> dict:
    material = graph_material(graph)
    nodes = material["nodes"]
    graph_hash = workflow_graph_hash(graph)
    hardware = hardware if hardware is not None else _hardware_snapshot(runtime_fingerprint, data_dir)
    issues: list[str] = []
    loaders: dict[str, Any] = {}
    data_nodes: set[str] = set()
    custom_nodes: set[str] = set()
    preparation_nodes: set[str] = set()
    auxiliary_owners: set[str] = set()
    deferred_fields: list[dict] = []
    for node_id, node in nodes.items():
        if node['module'].startswith('custom.'):
            from modiff.custom_extensions import ExtensionStore
            try:
                extension = ExtensionStore().require_enabled(node['module'].removeprefix('custom.'))
                if extension['preview']['kind'] == 'modular' and node['action'] == 'LoadModels':
                    raise ValueError('loading additional custom block models requires Custom memory policy; this supplier is not Auto-qualified.')
                if extension['runtimeRole'] == 'manual':
                    raise ValueError('this custom source declares manual resource management; use Expert or review its data/connected-components contract.')
                custom_nodes.add(node_id)
                if extension['runtimeRole'] == 'data':
                    data_nodes.add(node_id)
            except (ValueError, OSError, SyntaxError) as error:
                issues.append(f'{node_id}: {error}')
            continue
        if (node["module"], node["action"]) in _DATA_ACTIONS:
            data_nodes.add(node_id)
            continue
        if (node["module"], node["action"]) in {("modules.Spandrel", "Upscaler"),
                                                 ("modules.ModularDiffusers", "AutoModelLoader")}:
            # These concrete nodes own separate weights. Their exact auxiliary
            # contracts must not borrow unrelated pipeline/Expert profiles.
            auxiliary_owners.add(node_id)
            loaders[node_id] = None
            continue
        values = _values(nodes, node_id)
        try:
            for field in ("model_type", "pipeline_class", "execution_profile_id", "repo_id", "model_id"):
                if node["params"].get(field, {}).get("sourceId"):
                    values[field] = _literal(nodes, node_id, field)
        except DeferredResourceValue as error:
            preparation_nodes.update(error.node_ids)
            deferred_fields.append(error.target)
            loaders[node_id] = None
            continue
        except (ValueError, TypeError) as error:
            issues.append(f"{node_id}: {error}")
            continue
        profiles, reason = resolve_execution_profiles_for_loader(node["module"], node["action"], values)
        if profiles and all(profile.execution_path.startswith("builtin-") for profile in profiles):
            data_nodes.add(node_id)
            continue  # Reviewed deterministic operations do not own model weights.
        if profiles:
            if reason or len(profiles) != 1:
                issues.append(f"{node_id}: the loader's exact execution profile is unresolved ({reason or 'ambiguous profile'}).")
            else:
                loaders[node_id] = profiles[0]
        elif re.search(r"load.*(?:model|pipeline|encoder)|modelsloader|automodelloader", node["action"], re.I):
            issues.append(f"{node_id}: no reviewed resource recipe exists for this model loader.")
        elif node["module"] not in _DATA_MODULES and not node["module"].startswith("modules.Diffusers") and node["module"] != "modules.ModularDiffusers":
            issues.append(f"{node_id}: {node['module']}.{node['action']} has no workflow Auto resource contract.")

    owned = set(loaders)
    for loader_id in loaders:
        owned.update(_consumers(nodes, loader_id, set(loaders)))
    for node_id, node in nodes.items():
        if node_id not in owned | data_nodes and node["module"] not in _DATA_MODULES and node["action"] != "Lora":
            issues.append(f"{node_id}: this resource operation has no connected, reviewed model owner.")
    planned: list[dict] = []
    recipe_cache: dict[str, dict] = {}
    patches: list[dict] = []
    total = {"systemRamBytes": 0, "vramBytes": 0, "diskFreeBytes": 0}
    capacity_requirements = {"systemRamBytes": 0, "vramBytes": 0}
    for loader_id, profile in loaders.items():
        if profile is None:
            continue
        values = _values(nodes, loader_id)
        consumers = _consumers(nodes, loader_id, set(loaders))
        try:
            for field in ("repo_id", "model_id", "model_type", "pipeline_class", "execution_profile_id", "revision", *_SETTINGS):
                if field in nodes[loader_id]["params"]:
                    values[field] = _literal(nodes, loader_id, field)
            native_runtime_controls = inspect_native_runtime_controls(nodes, loader_id, _literal)
            values.update(native_runtime_controls)
            recipe_controls, recipe_bindings = inspect_execution_recipe_controls(nodes, loader_id, _literal)
            values.update(recipe_controls)
            modes = {str(_literal(nodes, id, "mode")) for id in [loader_id, *consumers] if _literal(nodes, id, "mode")}
            if not modes:
                workflow = values.get("workflow_id")
                modes = {item["studioMode"] for item in REVIEWED_CLUSTER_EXECUTION_CANDIDATES if item["pipelineClass"] == profile.model_type and item["workflowId"] == workflow and item["studioMode"] in profile.modes}
                if modes == {"edit_image", "multi_image_reference_edit"}:
                    modes = {"multi_image_reference_edit"}
            if not modes and profile.execution_path == "modular-diffusers":
                from modiff.workflow_task_identity import modular_graph_tasks

                modes = modular_graph_tasks(nodes, loader_id, consumers, profile.model_type)
                if len(modes) == 1 and not modes.issubset(profile.modes):
                    from modiff.modular_workflow_contracts import PINNED_MODULAR_WORKFLOW_TRUTH

                    # Operation tasks and historical resource modes can name the
                    # same reviewed upstream workflow differently. Use its existing
                    # mapping rather than a frontend/model-specific alias table.
                    truth = PINNED_MODULAR_WORKFLOW_TRUTH.get(profile.model_type)
                    route = truth.mode(next(iter(modes))) if truth else None
                    if route:
                        aliases = {item["studioMode"] for item in REVIEWED_CLUSTER_EXECUTION_CANDIDATES
                                   if item["pipelineClass"] == profile.model_type
                                   and item["workflowId"] == (route.upstream_workflow or "default")
                                   and item["studioMode"] in profile.modes}
                        modes = aliases or modes
            mode = next(iter(modes)) if len(modes) == 1 else profile.modes[0] if not modes and len(profile.modes) == 1 else None
            if mode not in profile.modes:
                raise ValueError("The model's task is ambiguous; select an explicit mode on its loader or consumer.")
            repo = _repository(values.get("repo_id") or values.get("model_id"))
            if not repo:
                raise ValueError("Auto requires an exact installed Hub model repository.")
            form = {"modelType": profile.model_type, "modelRepo": repo, "mode": mode, "resourceMode": "auto",
                    "executionProfileId": profile.id,
                    "device": values.get("device", "cpu"), "dtype": values.get("dtype", "float32"),
                    "offloadMode": values.get("offload_mode", "none"), "autoOffload": values.get("auto_offload", False),
                    "quantizationMode": values.get("quantization_mode", "none")}
            device = str(form["device"])
            kind = hardware.get("accelerator", {}).get("kind")
            if device != "cpu" and kind and (device.split(":", 1)[0] != kind or (":" in device and device.rsplit(":", 1)[1] != "0")):
                raise ValueError("This device differs from the accelerator covered by the resource snapshot.")
            for id in [loader_id, *consumers]:
                for field, target, raw in _workload_values(nodes, id):
                    if raw is None:
                        continue
                    value = _number(raw)
                    if value is None:
                        raise ValueError(f"{id}.{field} is not a finite resource value.")
                    form[target] = max(form.get(target, 0), value)
            form_key = json.dumps(form, sort_keys=True)
            if form_key not in recipe_cache:
                recipe_cache[form_key] = plan_recipe({"form": form}, runtime_fingerprint=runtime_fingerprint, local_models=local_models, data_dir=data_dir)
            plan = recipe_cache[form_key]
            candidates = [candidate for candidate in plan.get("candidates", []) if isinstance(candidate, dict)
                          and candidate.get("canAutoRun") is True and candidate.get("proof", {}).get("status") in READY_PROOF_STATUSES
                          and candidate.get("modelRepo") == repo and candidate.get("modelType") == profile.model_type
                          and candidate.get("mode") == mode and candidate.get("loaderModule") == profile.loader_module
                          and candidate.get("executionProfileId") == profile.id
                          and candidate.get("loaderAction") == profile.loader_action and candidate.get("executionPath") == profile.execution_path
                          and candidate.get("dtype") == form["dtype"] and candidate.get("quantizationMode", "none") == form["quantizationMode"]]
            revision = values.get("revision")
            if revision:
                candidates = [candidate for candidate in candidates if (candidate.get("artifactResolution", {}).get("resolved", {}).get("revision") or candidate.get("artifactRevision")) == revision]
            candidates = [candidate for candidate in candidates if _candidate_covers_workload(candidate, form)]
            candidates.sort(key=lambda candidate: candidate.get("offloadMode") != form["offloadMode"])
            if not candidates:
                batch_reason = (
                    f"No accepted recipe covers batchSize={form['batchSize']:g}. "
                    "Use a covered batch size or select Expert."
                    if form.get("batchSize", 1) > 1 else None
                )
                reason = plan.get("blockingReason") or batch_reason or (
                    "No accepted recipe preserves this graph's model, precision and requested workload. "
                    f"Requested {repo}, {form['dtype']}, {form['quantizationMode']} quantization, "
                    f"{form.get('width', 'default')}x{form.get('height', 'default')}, "
                    f"{form.get('steps', 'default')} steps. "
                    "Installed model files do not establish Auto resource qualification. "
                    "Review the model's resource recipes or select Custom memory to keep your settings."
                )
                raise ValueError(reason)
            candidate = candidates[0]
            envelope = candidate.get("requirements", {})
            declared = select_memory_requirement(envelope, candidate.get("offloadMode"))
            if not isinstance(declared, dict) or any(_number(declared.get(key)) is None for key in capacity_requirements):
                raise ValueError("The accepted recipe has no complete memory requirement contract.")
            capacity = {}
            working_policy = "explicit_working_demand"
            if envelope.get("memorySemantics") == "machine_capacity":
                capacity = {**declared, **{key: int(_number(declared[key])) for key in capacity_requirements}}
                for key in capacity_requirements:
                    capacity_requirements[key] = max(capacity_requirements[key], capacity[key])
                issues.extend(f"{loader_id}: {missing}" for missing in _requirements_missing_for_dict(
                    hardware, declared, offload_mode=candidate.get("offloadMode", "none")))
                working = candidate.get("workingMemoryRequirements")
                if working is None:
                    working = envelope.get("workingMemoryRequirements")
                if working is None:
                    requirements = {key: 0 for key in capacity_requirements}
                    working_policy = "runtime_headroom_policy"
                else:
                    requirements = select_memory_requirement(working, candidate.get("offloadMode"))
                    if not isinstance(requirements, dict) or any(_number(requirements.get(key)) is None for key in capacity_requirements):
                        raise ValueError("The explicit working-memory demand is incomplete or invalid.")
                    requirements = {key: int(_number(requirements[key])) for key in capacity_requirements}
                # Disk remains a free-space demand, never a machine-memory tier.
                requirements["diskFreeBytes"] = int(_number(declared.get("diskFreeBytes")) or 0)
            else:
                # Historical flat contracts and explicit extension/test budgets
                # keep their strict additive working-demand meaning.
                requirements = declared
            for key in total:
                total[key] += int(_number(requirements.get(key)) or 0)
            settings = {"offloadMode": candidate.get("offloadMode", "none"), "autoOffload": candidate.get("autoOffload", False)}
            if recipe_controls and values['offload_mode'] != settings['offloadMode']:
                recipe_id, recipe_field = recipe_bindings['offload_mode']
                if nodes[recipe_id]['params'].get(recipe_field, {}).get('sourceId'):
                    raise ValueError(f"Connected {recipe_id}.{recipe_field} must select the planned value {settings['offloadMode']}.")
                if recipe_field not in nodes[recipe_id]['params']:
                    raise ValueError('Auto needs an existing offload control on the connected execution recipe.')
                patches.append({"nodeId": recipe_id, "field": recipe_field, "value": settings['offloadMode']})
            for field, key in _SETTINGS.items():
                if key in settings and field in nodes[loader_id]["params"] and (
                    values.get(field) != settings[key] or recipe_controls and field in {'offload_mode', 'auto_offload'}
                    and nodes[loader_id]['params'][field].get('value') != settings[key]
                ):
                    if nodes[loader_id]["params"][field].get("sourceId"):
                        raise ValueError(f"Connected {field} must select the planned value {settings[key]}.")
                    patches.append({"nodeId": loader_id, "field": field, "value": settings[key]})
            planned.append({"nodeId": loader_id, "modelType": profile.model_type, "mode": mode, "repository": repo,
                            "candidateId": candidate["id"], "proofStatus": candidate["proof"]["status"],
                            "consumers": consumers, "requirements": requirements,
                            "capacityRequirements": capacity, "workingMemoryPolicy": working_policy,
                            "settings": {**form, **settings, **native_runtime_controls}})
        except DeferredResourceValue as error:
            preparation_nodes.update(error.node_ids)
            deferred_fields.append(error.target)
        except (ValueError, TypeError) as error:
            issues.append(f"{loader_id}: {error}")
    for node_id in nodes:
        if node_id not in auxiliary_owners:
            continue
        try:
            owner = (_upscaler_owner if nodes[node_id]["module"] == "modules.Spandrel" else _controlnet_owner)(
                nodes, node_id, planned, hardware)
            planned.append(owner)
            for key in total:
                total[key] += owner["requirements"][key]
        except (ValueError, TypeError, OSError) as error:
            issues.append(f"{node_id}: {error}")
    adapters = []
    adapter_names = {}
    try:
        native_lora_owners = _native_lora_owners(material, loaders)
    except (ValueError, TypeError) as error:
        issues.append(str(error))
        native_lora_owners = {}
    execution_order = dict.fromkeys(node_id for path in material['paths'] for node_id in path)
    for node_id in execution_order:
        node = nodes[node_id]
        if re.search("lora|adapter|quantconfig", node["action"], re.I) and node_id not in loaders:
            try:
                from modiff.auxiliary_lora import lora_resource_requirement
                native = (node['module'], node['action']) == ('modules.ModularDiffusers', 'Lora')
                # One ordered descriptor may reach several independent owners.
                # Arbitrary data/list suppliers and unrelated input ports cannot
                # establish model ownership or hide unbudgeted adapter storage.
                owners = native_lora_owners.get(node_id, set()) if native else {
                    target for target, loader in nodes.items() if target in loaders
                    and any(p.get("sourceId") == node_id for p in loader["params"].values())}
                if not owners:
                    raise ValueError("The adapter's model owners cannot be resolved from its reviewed loader connections.")
                requirements = lora_resource_requirement(node_id, node, graph=material) if native else lora_resource_requirement(node_id, node)
                for owner_id in owners:
                    names = adapter_names.setdefault(owner_id, set())
                    name = requirements.get('adapterName')
                    if name in names:
                        raise ValueError(f'The native LoRA chain for {owner_id} contains a duplicate adapter name {name}.')
                    names.add(name)
                for key in ("systemRamBytes", "vramBytes"):
                    total[key] += requirements[key] * len(owners)
                adapters.append({"nodeId": node_id, "loaderIds": sorted(owners), **requirements})
            except (ValueError, TypeError, OSError) as error:
                issues.append(f"{node_id}: {error}")
    accelerator = hardware.get("accelerator", {})
    system = hardware.get("systemMemory", {})
    disk = hardware.get("offloadDisk", {})
    available = {"systemRamBytes": system.get("availableBytes"), "vramBytes": accelerator.get("freeBytes"), "diskFreeBytes": disk.get("freeBytes")}
    # Shared accelerator bytes come from system RAM, never an additional pool.
    topology = str(accelerator.get("memoryKind", ""))
    shared = topology in {"shared", "unified", "shared_system", "system_shared"} or accelerator.get("sharedMemory") is True
    if shared:
        # The standard planner intentionally publishes dedicated VRAM in
        # freeBytes. This combined envelope also needs the actual shared pool,
        # which is already present in the same runtime hardware snapshot.
        devices = hardware.get("runtime", {}).get("hardware", {}).get("devices", [])
        device = next((item for item in devices if item.get("memory_kind") in {"shared", "unified"} and item.get("type") == accelerator.get("kind")), {})
        shared_free = _number(device.get("shared_memory_free"))
        accessible_free = _number(device.get("torch_vram_free"))
        if shared_free is not None or accessible_free is not None:
            available["vramBytes"] = min(value for value in (shared_free, accessible_free, _number(system.get("availableBytes"))) if value is not None)
    # Unknown model working demand is not silently equated to a machine tier.
    # Retain the established minimum free headroom once for this workflow.
    floors = {"systemRamBytes": max(4 * 1024 ** 3, int((_number(system.get("totalBytes")) or 0) * .1)),
              "vramBytes": max(2 * 1024 ** 3, int((_number(accelerator.get("totalBytes")) or 0) * .1))}
    policy_owners = [owner for owner in planned if owner["capacityRequirements"]
                     or owner["workingMemoryPolicy"] == "runtime_headroom_policy"]
    policy_floors = {"systemRamBytes": floors["systemRamBytes"] if policy_owners else 0,
                     "vramBytes": floors["vramBytes"] if any(
                         not str(owner["settings"].get("device", "")).startswith("cpu")
                         and (str(owner["settings"].get("device", "")).startswith(("cuda", "xpu", "mps"))
                              or accelerator.get("kind") in {"cuda", "xpu", "mps"})
                         for owner in policy_owners) else 0}
    for key, floor in policy_floors.items():
        total[key] = max(total[key], floor)
    retained_total = dict(total)
    cache_snapshot = cache_snapshot if isinstance(cache_snapshot, dict) else {}
    cached_owners = cache_snapshot.get("owners") or {}
    candidate_graph = deepcopy(material)
    for patch in patches:
        candidate_graph["nodes"][patch["nodeId"]]["params"][patch["field"]]["value"] = patch["value"]
    captured = _RESOLVED_FIELDS.get() or {}
    resident_credit = {"systemRamBytes": 0, "vramBytes": 0}
    reused = []
    for owner in planned:
        owner["cacheKey"] = workflow_owner_cache_key(candidate_graph, owner["nodeId"], captured)
        cached = cached_owners.get(owner["nodeId"], {})
        if custom_nodes or preparation_nodes or cached.get("cacheKey") != owner["cacheKey"]:
            continue
        reused.append(owner["nodeId"])
        for key in resident_credit:
            budget = owner["requirements"][key] + sum(
                adapter[key] for adapter in adapters if owner["nodeId"] in adapter["loaderIds"]
            )
            resident_credit[key] += min(budget, int(_number(cached.get(key)) or 0))
    # Only an explicit working budget can receive weight-storage credit.
    # A capacity-only recipe contributes no presumed cold-load bytes.
    for key, credit in resident_credit.items():
        if credit:
            total[key] = max(min(retained_total[key], floors[key]), total[key] - credit)
    from modiff.workflow_auto_lifecycle import plan_owner_lifetimes
    schedule = plan_owner_lifetimes(material, planned, adapters)
    for key, floor in policy_floors.items():
        schedule["peak"][key] = max(schedule["peak"][key], floor)
    schedule["sharedPeakBytes"] = max(schedule["sharedPeakBytes"], sum(policy_floors.values()))
    # A machine tier and one free-memory floor do not bound the sum of distinct
    # primary model weights/activations. Never retain several such owners on the
    # assumption that their zero, unmeasured working demands establish fit.
    # Reviewed auxiliary storage remains additive under its existing contract.
    primary_ids = {owner["nodeId"] for owner in planned
                   if owner.get("resourceOwnerKind") != "auxiliary_model"}
    unknown_primary_ids = {owner["nodeId"] for owner in planned
                           if owner["nodeId"] in primary_ids
                           and owner["workingMemoryPolicy"] == "runtime_headroom_policy"}
    unknown_combined_demand = len(primary_ids) > 1 and bool(unknown_primary_ids)
    unknown_overlap = any(
        first["ownerId"] in unknown_primary_ids and second["ownerId"] in primary_ids
        and first["ownerId"] != second["ownerId"]
        and first["first"] <= second["last"] and second["first"] <= first["last"]
        for first in schedule["owners"] for second in schedule["owners"]
    ) if unknown_combined_demand else False
    # The established executor starts scheduled runs cold, detaches retained
    # material outputs, destroys expired ownership and checks real free memory
    # before the next load. Custom Python cannot grant that release guarantee.
    unknown_release_required = (unknown_combined_demand and not custom_nodes
                                and not unknown_overlap and bool(schedule["releases"]))
    if unknown_combined_demand and not unknown_release_required:
        issues.append(
            "One or more independent model owners lack a complete working-memory demand, "
            "and cannot be safely released before another owner is needed. "
            "Auto cannot establish combined model fit from machine capacity tiers. "
            "Use separate sequential image stages, a reviewed working-memory recipe, "
            "or Custom memory to keep this workflow."
        )
    # Retain independent owners only when their complete combined working
    # envelope fits. Under pressure use the existing release/recheck schedule.
    # Unknown primary demands always need safe release, even when headroom fits.
    retained_demand = {**total, "systemRamBytes": total["systemRamBytes"] + (total["vramBytes"] if shared else 0)}
    retention_fits = all(
        not required or required <= (_number(available[key]) or 0)
        for key, required in retained_demand.items()
    )
    # Custom Python can retain references outside the graph. Its approval grants
    # execution, not a proof that early model eviction is safe.
    use_schedule = unknown_release_required or (
        not custom_nodes and not retention_fits and len(planned) > 1 and bool(schedule["releases"]) and any(
            schedule["peak"][key] < total[key] for key in ("systemRamBytes", "vramBytes")
        )
    )
    if use_schedule:
        total = {**total, **schedule["peak"]}
        reused = []  # The scheduled executor starts from an empty model cache.
    memory_required = (schedule["sharedPeakBytes"] if shared and use_schedule else total["systemRamBytes"] + (total["vramBytes"] if shared else 0))
    demand = {**total, "systemRamBytes": memory_required}
    memory_issues = [key for key, required in demand.items()
                     if required and (_number(available[key]) is None or required > _number(available[key]))]
    # Only measured app-owned weight storage can make cleanup a plausible plan.
    # Dispatch destroys ownership and rechecks real free memory: external
    # references or OS arenas may prevent the projected reclamation.
    reclaimable = cache_snapshot.get("reclaimable") or {}
    projected = dict(available)
    for key in resident_credit:
        free = _number(available[key])
        if free is not None:
            credit = _number(reclaimable.get(key)) or 0
            if key == "systemRamBytes" and shared:
                credit += _number(reclaimable.get("vramBytes")) or 0
            capacity = _number((system if key == "systemRamBytes" else accelerator).get("totalBytes"))
            if key == 'vramBytes' and shared:
                # freeBytes above was resolved from the shared/accessible pool,
                # so its forecast cannot be capped by dedicated VRAM alone.
                limits = [_number(value) for value in (
                    system.get('totalBytes'), accelerator.get('accessibleTotalBytes'),
                    accelerator.get('sharedTotalBytes'), device.get('torch_vram_total'),
                    device.get('shared_memory_total'),
                )]
                limits = [value for value in limits if value is not None and value > 0]
                capacity = min(limits) if limits else None
            projected[key] = min(capacity, free + credit) if capacity is not None else free + credit
    cold_total = dict(total) if use_schedule else dict(retained_total)
    cold_demand = {**cold_total, "systemRamBytes": cold_total["systemRamBytes"] + (cold_total["vramBytes"] if shared else 0)}
    requires_cache_preparation = bool(
        memory_issues and not custom_nodes and not preparation_nodes and not issues
        and any((_number(reclaimable.get(key)) or 0) > 0 for key in resident_credit)
        and all(not required or (_number(projected[key]) is not None and required <= _number(projected[key]))
                for key, required in cold_demand.items())
    )
    if requires_cache_preparation:
        total, demand, reused = cold_total, cold_demand, []
        memory_issues = []
    labels = {"systemRamBytes": "System RAM", "vramBytes": "Accelerator memory", "diskFreeBytes": "Offload disk space"}
    def format_bytes(value):
        return f"{int(value)} bytes" if value < 1024 ** 2 else f"{value / 1024 ** 3:.2f} GiB"
    for key in memory_issues:
        required = demand[key]
        free = _number(available[key])
        available_label = format_bytes(free) if free is not None else "unknown capacity"
        issues.append(f"Combined {labels[key]} ({key}): needs {format_bytes(required)}; {available_label} available.")
    # The hash must describe exactly the patches returned to the client. Any
    # deferred supplier postpones all mutations until dispatch-time replanning.
    patches = patches if not issues and not preparation_nodes else []
    patched_graph = deepcopy(material)
    for patch in patches:
        patched_graph["nodes"][patch["nodeId"]]["params"][patch["field"]]["value"] = patch["value"]
    return {"schemaVersion": 1, "graphHash": graph_hash, "plannedGraphHash": workflow_graph_hash(patched_graph),
            "canAutoRun": not issues, "issues": issues, "loaders": planned, "adapters": adapters, "patches": patches,
            "requirements": total, "retainedRequirements": retained_total, "available": available, "sharedMemory": shared,
            "capacityRequirements": capacity_requirements,
            "workingMemoryPolicy": ("runtime_headroom_policy" if any(
                owner["workingMemoryPolicy"] == "runtime_headroom_policy" for owner in planned)
                else "explicit_working_demand"),
            "reusedOwnerIds": reused, "requiresCachePreparation": requires_cache_preparation,
            "schedule": schedule if use_schedule else None,
            "requiresPreparation": bool(preparation_nodes), "preparationNodeIds": sorted(preparation_nodes), "deferredFields": deferred_fields,
            "strategy": "dependency_order_release_owners" if use_schedule else "dependency_order_retained_owners", "checkedAt": int(time.time() * 1000),
            "message": ("Run will release previous model cache and recheck actual available memory before loading models. " if requires_cache_preparation else "") + ("Auto reuses compatible loaded weights and reserves additional inference headroom. " if reused else "") + ("Run will first execute the data suppliers and validate their actual resource values before loading models. " if preparation_nodes else "") + ("Auto finishes dependent work and releases completed model owners before loading the next owner. Downstream material outputs are retained." if use_schedule else "Auto uses the existing dependency order and budgets all retained model owners.") + " Shared loader nodes are counted once. Model, precision and creative controls are preserved."}


def build_workflow_auto_plan(graph, **kwargs):
    captured = {}
    token = _RESOLVED_FIELDS.set(captured)
    try:
        result = _build_workflow_auto_plan(graph, **kwargs)
        result["resolvedFields"] = captured
        return result
    finally:
        _RESOLVED_FIELDS.reset(token)
