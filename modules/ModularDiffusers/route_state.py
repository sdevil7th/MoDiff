"""Opaque, process-local state for multi-action Modular Diffusers routes.

Large graph values such as latents remain on their existing typed edges.  This
module carries only the small pieces of upstream routing state that cannot be
reconstructed safely between the VAE, denoise, and decode actions.  Neither
loader bindings nor route states are serializable graph data.
"""

from __future__ import annotations

import hashlib
import math
import threading
import weakref
from collections.abc import Mapping
from enum import Enum

import torch
from PIL import Image

from modiff.model_artifact_catalog import require_catalog_revision
from modiff.modular_workflow_contracts import WAN_WORKFLOW_REPOSITORIES


_QWEN_ROUTE_CONTRACT = "qwen"
_SDXL_ROUTE_CONTRACT = "sdxl"
_WAN_ROUTE_CONTRACT = "wan_i2v"
_ROUTE_CONTRACT_BY_MODEL_TYPE = {
    "QwenImageModularPipeline": _QWEN_ROUTE_CONTRACT,
    "QwenImageEditModularPipeline": _QWEN_ROUTE_CONTRACT,
    "QwenImageEditPlusModularPipeline": _QWEN_ROUTE_CONTRACT,
    "StableDiffusionXLModularPipeline": _SDXL_ROUTE_CONTRACT,
    "WanImage2VideoModularPipeline": _WAN_ROUTE_CONTRACT,
}
SUPPORTED_ROUTE_MODEL_TYPES = frozenset(_ROUTE_CONTRACT_BY_MODEL_TYPE)
ROUTE_STATE_INPUT = "route_state_in"
ROUTE_STATE_OUTPUT = "route_state_out"
SDXL_UNION_CONTROL_MODE_LIMIT = 32
ROUTE_RESERVED_PIPELINE_INPUTS = frozenset(
    {
        "generator",
        "processed_mask_image",
        "mask_overlay_kwargs",
        "mask",
    }
)

_ENCODE_TO_DENOISE = "encode_to_denoise"
_IMAGE_EMBED_TO_VAE = "image_embed_to_vae"
_CONTROLNET_TO_DENOISE = "controlnet_to_denoise"
_DENOISE_TO_DECODE = "denoise_to_decode"
_MAX_PAIRED_LATENT_TENSORS = 64
_MAX_MASK_CROP_PADDING = 8192
_MAX_OVERLAY_EDGE_PIXELS = 8192
_MAX_OVERLAY_AGGREGATE_PIXELS = 16 * 1024 * 1024
_MAX_WAN_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_WAN_REQUEST_AREA = 16 * 1024 * 1024
_MAX_WAN_VIDEO_TENSOR_BYTES = 512 * 1024 * 1024
_WAN_VIDEO_PIXEL_BYTES = 3 * 4
_MAX_WAN_DIMENSION = 8192
_MAX_WAN_FRAMES = 480
_MAX_WAN_PROCESSOR_NORMALIZATION = 16.0
_MAX_WAN_VAE_CONFIG_MAGNITUDE = 1024.0
_MIN_WAN_POSITIVE_SCALE = 1e-6
_WAN_SPATIAL_SCALE = 8
_WAN_TEMPORAL_SCALE = 4
_WAN_LATENT_CHANNELS = 16
_WAN_TRANSFORMER_PATCH_SIZE = (1, 2, 2)
_WAN_PATCH_SIZE_SPATIAL = 2
_WAN_TRANSFORMER_INPUT_CHANNELS = 36
_WAN_TRANSFORMER_OUTPUT_CHANNELS = 16
_WAN_IMAGE_SIZE = 224
_WAN_IMAGE_EMBED_DIM = 1280
_WAN_IMAGE_EMBED_TOKENS = 257
_WAN_IMAGE_ENCODER_PATCH_SIZE = 14
_WAN_IMAGE_ENCODER_PROJECTION_DIM = 1024
_WAN_IMAGE_ENCODER_LAYERS = 32
_WAN_IMAGE_ENCODER_HEADS = 16
_WAN_CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
_WAN_CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
_WAN_VAE_LATENTS_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
_WAN_VAE_LATENTS_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.916,
)
_WAN_I2V_WORKFLOW = "image2video"
_WAN_FLF_WORKFLOW = "flf2v"
_WAN_CLIP_PROCESSOR_FIELDS = (
    "do_resize",
    "size",
    "resample",
    "do_center_crop",
    "crop_size",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "do_convert_rgb",
    "do_pad",
    "pad_size",
    "disable_grouping",
)
_WAN_VIDEO_PROCESSOR_FIELDS = (
    "do_resize",
    "vae_scale_factor",
    "vae_latent_channels",
    "resample",
    "reducing_gap",
    "do_normalize",
    "do_binarize",
    "do_convert_rgb",
    "do_convert_grayscale",
)
_WAN_VIDEO_PROCESSOR_DEFAULTS = (
    ("do_resize", True),
    ("vae_scale_factor", 8),
    ("vae_latent_channels", 4),
    ("resample", "lanczos"),
    ("reducing_gap", None),
    ("do_normalize", True),
    ("do_binarize", False),
    ("do_convert_rgb", False),
    ("do_convert_grayscale", False),
)
_ISSUER_SEAL = object()
_NOT_PROVIDED = object()
_REGISTRY_LOCK = threading.RLock()


class _PipelineBindingKey:
    """Identity-only, non-string component-dictionary key."""

    __slots__ = ()

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Modular pipeline bindings cannot be serialized.")


_PIPELINE_BINDING_KEY = _PipelineBindingKey()


class _StandaloneComponentBindingKey:
    """Identity-only, non-string standalone-component dictionary key."""

    __slots__ = ()

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Standalone component bindings cannot be serialized.")


_STANDALONE_COMPONENT_BINDING_KEY = _StandaloneComponentBindingKey()


class _PipelineInstanceToken:
    """One successful ModelsLoader execution, compared only by identity."""

    __slots__ = (
        "_model_type",
        "_repo_id",
        "_repo_source",
        "_revision",
        "_sealed",
        "__weakref__",
    )

    def __init__(self, seal, *, model_type, repo_id, repo_source, revision):
        if seal is not _ISSUER_SEAL:
            raise TypeError("Pipeline instance tokens are issued only by ModelsLoader.")
        object.__setattr__(self, "_model_type", model_type)
        object.__setattr__(self, "_repo_id", repo_id)
        object.__setattr__(self, "_repo_source", repo_source)
        object.__setattr__(self, "_revision", revision)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name, _value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Pipeline instance tokens are immutable.")
        object.__setattr__(self, _name, _value)

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Modular pipeline bindings cannot be serialized.")


class _ComponentBinding:
    """Sealed role and component-id inventory for one loader output port."""

    __slots__ = ("_pipeline_token", "_role", "_component_ids", "_sealed", "__weakref__")

    def __init__(self, seal, *, pipeline_token, role, component_ids):
        if seal is not _ISSUER_SEAL:
            raise TypeError("Component bindings are issued only by ModelsLoader.")
        object.__setattr__(self, "_pipeline_token", pipeline_token)
        object.__setattr__(self, "_role", role)
        object.__setattr__(self, "_component_ids", component_ids)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name, _value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Component bindings are immutable.")
        object.__setattr__(self, _name, _value)

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Modular component bindings cannot be serialized.")


class _StandaloneComponentIssuer:
    """One AutoModelLoader instance, compared only by identity."""

    __slots__ = ("_sealed", "__weakref__")

    def __init__(self, seal):
        if seal is not _ISSUER_SEAL:
            raise TypeError("Standalone component issuers are created only by AutoModelLoader.")
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name, _value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Standalone component issuers are immutable.")
        object.__setattr__(self, _name, _value)

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Standalone component issuers cannot be serialized.")


class _StandaloneComponentBinding:
    """Sealed provenance and publication identity for one standalone component."""

    __slots__ = (
        "_issuer",
        "_component_kind",
        "_manager_model_id",
        "_repo_source",
        "_repo_id",
        "_revision",
        "_subfolder",
        "_class_name",
        "_config_fingerprint",
        "_sealed",
        "__weakref__",
    )

    def __init__(
        self,
        seal,
        *,
        issuer,
        component_kind,
        manager_model_id,
        repo_source,
        repo_id,
        revision,
        subfolder,
        class_name,
        config_fingerprint,
    ):
        if seal is not _ISSUER_SEAL:
            raise TypeError("Standalone component bindings are issued only by AutoModelLoader.")
        object.__setattr__(self, "_issuer", issuer)
        object.__setattr__(self, "_component_kind", component_kind)
        object.__setattr__(self, "_manager_model_id", manager_model_id)
        object.__setattr__(self, "_repo_source", repo_source)
        object.__setattr__(self, "_repo_id", repo_id)
        object.__setattr__(self, "_revision", revision)
        object.__setattr__(self, "_subfolder", subfolder)
        object.__setattr__(self, "_class_name", class_name)
        object.__setattr__(self, "_config_fingerprint", config_fingerprint)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name, _value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Standalone component bindings are immutable.")
        object.__setattr__(self, _name, _value)

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Standalone component bindings cannot be serialized.")


class _SealedRoutePayload:
    """Immutable base for one reviewed pipeline-family route payload."""

    __slots__ = ("_sealed",)

    def __setattr__(self, _name, _value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Modular route payloads are immutable.")
        object.__setattr__(self, _name, _value)

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Modular route payloads cannot be serialized.")


class _QwenRoutePayload(_SealedRoutePayload):
    """Pinned Qwen generator, mask/overlay, and tensor-pairing state."""

    __slots__ = (
        "_generator_snapshot",
        "_processed_mask_image",
        "_mask_overlay_kwargs",
        "_inpaint",
        "_paired_latents_ref",
        "_paired_control_latents_ref",
        "_standalone_controlnet_binding",
    )

    def __init__(
        self,
        *,
        generator_snapshot,
        processed_mask_image,
        mask_overlay_kwargs,
        inpaint,
        paired_latents_ref,
        paired_control_latents_ref=None,
        standalone_controlnet_binding=None,
    ):
        object.__setattr__(self, "_generator_snapshot", generator_snapshot)
        object.__setattr__(self, "_processed_mask_image", processed_mask_image)
        object.__setattr__(self, "_mask_overlay_kwargs", mask_overlay_kwargs)
        object.__setattr__(self, "_inpaint", inpaint)
        object.__setattr__(self, "_paired_latents_ref", paired_latents_ref)
        object.__setattr__(self, "_paired_control_latents_ref", paired_control_latents_ref)
        object.__setattr__(self, "_standalone_controlnet_binding", standalone_controlnet_binding)
        object.__setattr__(self, "_sealed", True)


class _SdxlRoutePayload(_SealedRoutePayload):
    """Pinned SDXL generator, typed latent, and optional crop-overlay state."""

    __slots__ = (
        "_generator_snapshot",
        "_inpaint",
        "_paired_latents_ref",
        "_mask_ref",
        "_masked_image_latents_ref",
        "_padding_mask_crop",
        "_crops_coords",
        "_original_image_snapshot",
        "_original_mask_snapshot",
        "_vae_ref",
        "_vae_latent_channels",
        "_vae_scale_factor",
    )

    def __init__(
        self,
        *,
        generator_snapshot,
        inpaint,
        paired_latents_ref,
        mask_ref=None,
        masked_image_latents_ref=None,
        padding_mask_crop=None,
        crops_coords=None,
        original_image_snapshot=None,
        original_mask_snapshot=None,
        vae_ref=None,
        vae_latent_channels=None,
        vae_scale_factor=None,
    ):
        object.__setattr__(self, "_generator_snapshot", generator_snapshot)
        object.__setattr__(self, "_inpaint", inpaint)
        object.__setattr__(self, "_paired_latents_ref", paired_latents_ref)
        object.__setattr__(self, "_mask_ref", mask_ref)
        object.__setattr__(self, "_masked_image_latents_ref", masked_image_latents_ref)
        object.__setattr__(self, "_padding_mask_crop", padding_mask_crop)
        object.__setattr__(self, "_crops_coords", crops_coords)
        object.__setattr__(self, "_original_image_snapshot", original_image_snapshot)
        object.__setattr__(self, "_original_mask_snapshot", original_mask_snapshot)
        object.__setattr__(self, "_vae_ref", vae_ref)
        object.__setattr__(self, "_vae_latent_channels", vae_latent_channels)
        object.__setattr__(self, "_vae_scale_factor", vae_scale_factor)
        object.__setattr__(self, "_sealed", True)


class _WanRoutePayload(_SealedRoutePayload):
    """Pinned Wan split-route media, geometry, tensor, and component state."""

    __slots__ = (
        "_generator_snapshot",
        "_inpaint",
        "_paired_latents_ref",
        "_image_embeds_ref",
        "_image_condition_latents_ref",
        "_source_image_ref",
        "_source_image_seal",
        "_last_image_ref",
        "_last_image_seal",
        "_workflow",
        "_requested_height",
        "_requested_width",
        "_first_height",
        "_first_width",
        "_second_height",
        "_second_width",
        "_num_frames",
        "_image_encoder_ref",
        "_image_encoder_config_seal",
        "_image_processor_ref",
        "_image_processor_config_seal",
        "_image_encoder_execution_device",
        "_vae_ref",
        "_video_processor_ref",
        "_video_processor_config_seal",
        "_vae_config_seal",
        "_vae_execution_device",
        "_transformer_ref",
        "_transformer_config_seal",
    )

    def __init__(
        self,
        *,
        generator_snapshot,
        paired_latents_ref,
        image_embeds_ref,
        image_condition_latents_ref=None,
        source_image_ref,
        source_image_seal,
        last_image_ref,
        last_image_seal,
        workflow,
        requested_height,
        requested_width,
        first_height,
        first_width,
        second_height=None,
        second_width=None,
        num_frames=None,
        image_encoder_ref=None,
        image_encoder_config_seal=None,
        image_processor_ref=None,
        image_processor_config_seal=None,
        image_encoder_execution_device=None,
        vae_ref=None,
        video_processor_ref=None,
        video_processor_config_seal=None,
        vae_config_seal=None,
        vae_execution_device=None,
        transformer_ref=None,
        transformer_config_seal=None,
    ):
        object.__setattr__(self, "_generator_snapshot", generator_snapshot)
        object.__setattr__(self, "_inpaint", False)
        object.__setattr__(self, "_paired_latents_ref", paired_latents_ref)
        object.__setattr__(self, "_image_embeds_ref", image_embeds_ref)
        object.__setattr__(self, "_image_condition_latents_ref", image_condition_latents_ref)
        object.__setattr__(self, "_source_image_ref", source_image_ref)
        object.__setattr__(self, "_source_image_seal", source_image_seal)
        object.__setattr__(self, "_last_image_ref", last_image_ref)
        object.__setattr__(self, "_last_image_seal", last_image_seal)
        object.__setattr__(self, "_workflow", workflow)
        object.__setattr__(self, "_requested_height", requested_height)
        object.__setattr__(self, "_requested_width", requested_width)
        object.__setattr__(self, "_first_height", first_height)
        object.__setattr__(self, "_first_width", first_width)
        object.__setattr__(self, "_second_height", second_height)
        object.__setattr__(self, "_second_width", second_width)
        object.__setattr__(self, "_num_frames", num_frames)
        object.__setattr__(self, "_image_encoder_ref", image_encoder_ref)
        object.__setattr__(self, "_image_encoder_config_seal", image_encoder_config_seal)
        object.__setattr__(self, "_image_processor_ref", image_processor_ref)
        object.__setattr__(self, "_image_processor_config_seal", image_processor_config_seal)
        object.__setattr__(self, "_image_encoder_execution_device", image_encoder_execution_device)
        object.__setattr__(self, "_vae_ref", vae_ref)
        object.__setattr__(self, "_video_processor_ref", video_processor_ref)
        object.__setattr__(self, "_video_processor_config_seal", video_processor_config_seal)
        object.__setattr__(self, "_vae_config_seal", vae_config_seal)
        object.__setattr__(self, "_vae_execution_device", vae_execution_device)
        object.__setattr__(self, "_transformer_ref", transformer_ref)
        object.__setattr__(self, "_transformer_config_seal", transformer_config_seal)
        object.__setattr__(self, "_sealed", True)


class _ModularRouteState:
    """Sealed route envelope with a contract-dispatched opaque payload."""

    __slots__ = ("_stage", "_binding", "_seed", "_contract", "_payload", "_sealed", "__weakref__")

    def __init__(self, seal, *, stage, binding, seed, contract, payload):
        if seal is not _ISSUER_SEAL:
            raise TypeError("Modular route states are issued only by the backend runtime.")
        expected_payload_type = {
            _QWEN_ROUTE_CONTRACT: _QwenRoutePayload,
            _SDXL_ROUTE_CONTRACT: _SdxlRoutePayload,
            _WAN_ROUTE_CONTRACT: _WanRoutePayload,
        }.get(contract)
        if expected_payload_type is None or type(payload) is not expected_payload_type:
            raise TypeError("Modular route state payload does not match its reviewed contract.")
        object.__setattr__(self, "_stage", stage)
        object.__setattr__(self, "_binding", binding)
        object.__setattr__(self, "_seed", seed)
        object.__setattr__(self, "_contract", contract)
        object.__setattr__(self, "_payload", payload)
        object.__setattr__(self, "_sealed", True)

    @property
    def _generator_snapshot(self):
        return self._payload._generator_snapshot

    @property
    def _processed_mask_image(self):
        return getattr(self._payload, "_processed_mask_image", None)

    @property
    def _mask_overlay_kwargs(self):
        return getattr(self._payload, "_mask_overlay_kwargs", None)

    @property
    def _inpaint(self):
        return self._payload._inpaint

    @property
    def _paired_latents_ref(self):
        return self._payload._paired_latents_ref

    @property
    def _paired_control_latents_ref(self):
        return getattr(self._payload, "_paired_control_latents_ref", None)

    @property
    def _standalone_controlnet_binding(self):
        return getattr(self._payload, "_standalone_controlnet_binding", None)

    def __setattr__(self, _name, _value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Modular route states are immutable.")
        object.__setattr__(self, _name, _value)

    def __copy__(self):
        return self

    def __deepcopy__(self, _memo):
        return self

    def __reduce_ex__(self, _protocol):
        raise TypeError("Modular route states cannot be serialized.")


_ISSUED_BINDINGS = weakref.WeakSet()
_ISSUED_COMPONENT_BINDINGS = weakref.WeakSet()
_ISSUED_ROUTE_STATES = weakref.WeakSet()
_ISSUED_STANDALONE_COMPONENT_ISSUERS = weakref.WeakSet()
_ISSUED_STANDALONE_COMPONENT_BINDINGS = weakref.WeakSet()
_CURRENT_STANDALONE_COMPONENT_PUBLICATIONS = weakref.WeakKeyDictionary()
_CURRENT_STANDALONE_MANAGER_PUBLICATIONS = {}

_STANDALONE_COMPONENT_KINDS = frozenset({"unet", "transformer", "vae", "controlnet"})

_LOADER_OUTPUT_ROLES = {
    "unet_out": "denoiser",
    "vae_out": "vae",
    "text_encoders": "text_encoders",
    "scheduler": "scheduler",
    "image_encoder": "image_encoder",
}


def route_contract_for_model_type(model_type):
    """Return the reviewed opaque-state contract for one pipeline class name."""

    return _ROUTE_CONTRACT_BY_MODEL_TYPE.get(model_type)


def route_requires_controlnet_state(model_type):
    """Whether generic ControlNet inputs require a preceding opaque Control route."""

    return route_contract_for_model_type(model_type) == _QWEN_ROUTE_CONTRACT


def route_uses_hidden_denoise_mask(model_type):
    """Whether Denoise must request Qwen's hidden mask result for route advancement."""

    return route_contract_for_model_type(model_type) == _QWEN_ROUTE_CONTRACT


def _is_issued_binding(value):
    with _REGISTRY_LOCK:
        return type(value) is _PipelineInstanceToken and value in _ISSUED_BINDINGS


def _is_issued_route_state(value):
    with _REGISTRY_LOCK:
        return type(value) is _ModularRouteState and value in _ISSUED_ROUTE_STATES


def _is_issued_component_binding(value):
    with _REGISTRY_LOCK:
        return type(value) is _ComponentBinding and value in _ISSUED_COMPONENT_BINDINGS


def _is_issued_standalone_component_issuer(value):
    with _REGISTRY_LOCK:
        return type(value) is _StandaloneComponentIssuer and value in _ISSUED_STANDALONE_COMPONENT_ISSUERS


def _is_current_standalone_component_binding(value):
    with _REGISTRY_LOCK:
        if (
            type(value) is not _StandaloneComponentBinding
            or value not in _ISSUED_STANDALONE_COMPONENT_BINDINGS
            or value._issuer not in _ISSUED_STANDALONE_COMPONENT_ISSUERS
        ):
            return False
        current_ref = _CURRENT_STANDALONE_COMPONENT_PUBLICATIONS.get(value._issuer)
        return current_ref is not None and current_ref() is value


def _is_lower_hex(value, length):
    return type(value) is str and len(value) == length and all(character in "0123456789abcdef" for character in value)


def _normalize_standalone_reviewed_identity(reviewed_identity):
    if type(reviewed_identity) is not tuple or len(reviewed_identity) != 6:
        raise ValueError("A standalone component binding requires one exact reviewed identity tuple.")
    repo_source, repo_id, revision, subfolder, class_name, config_fingerprint = reviewed_identity
    if type(repo_source) is not str or repo_source not in {"hub", "local"}:
        raise ValueError("A standalone component binding requires a reviewed repository source.")
    if (
        type(repo_id) is not str
        or not repo_id
        or repo_id != repo_id.strip()
        or len(repo_id) > 4096
        or "\x00" in repo_id
    ):
        raise ValueError("A standalone component binding requires a reviewed repository identity.")
    if repo_source == "hub":
        if not _is_lower_hex(revision, 40):
            raise ValueError("A standalone Hub component binding requires an immutable revision.")
    elif revision is not None:
        raise ValueError("A standalone local component binding cannot carry a Hub revision.")
    if subfolder is not None:
        if type(subfolder) is not str or len(subfolder) > 512 or "\\" in subfolder or "\x00" in subfolder:
            raise ValueError("A standalone component binding requires a normalized subfolder.")
        parts = subfolder.split("/")
        if not subfolder or subfolder.startswith("/") or any(part in ("", ".", "..") or ":" in part for part in parts):
            raise ValueError("A standalone component binding requires a normalized subfolder.")
    if type(class_name) is not str or not class_name or not class_name.isascii() or not class_name.isidentifier():
        raise ValueError("A standalone component binding requires a reviewed Diffusers class name.")
    if not _is_lower_hex(config_fingerprint, 64):
        raise ValueError("A standalone component binding requires a reviewed config fingerprint.")
    return reviewed_identity


def _standalone_binding_reviewed_identity(binding):
    return (
        binding._repo_source,
        binding._repo_id,
        binding._revision,
        binding._subfolder,
        binding._class_name,
        binding._config_fingerprint,
    )


def _forget_standalone_manager_publication(manager_model_id, binding_ref):
    with _REGISTRY_LOCK:
        if _CURRENT_STANDALONE_MANAGER_PUBLICATIONS.get(manager_model_id) is binding_ref:
            _CURRENT_STANDALONE_MANAGER_PUBLICATIONS.pop(manager_model_id, None)


def _validate_standalone_component_metadata(component, binding, *, label):
    expected = {
        "model_id": binding._manager_model_id,
        "repo_source": binding._repo_source,
        "repo_id": binding._repo_id,
        "revision": binding._revision,
        "class_name": binding._class_name,
    }
    for field, expected_value in expected.items():
        if (
            field not in component
            or component.get(field) != expected_value
            or type(component.get(field)) is not type(expected_value)
        ):
            raise ValueError(
                f"Connected standalone Diffusers {label} does not match its backend-issued provenance binding."
            )
    if component.get("trust_remote_code") is not False:
        raise ValueError(
            f"Connected standalone Diffusers {label} does not match its backend-issued provenance binding."
        )


def issue_standalone_component_issuer():
    """Mint one process-local publication issuer for an AutoModelLoader instance."""

    issuer = _StandaloneComponentIssuer(_ISSUER_SEAL)
    with _REGISTRY_LOCK:
        _ISSUED_STANDALONE_COMPONENT_ISSUERS.add(issuer)
    return issuer


def standalone_component_reuse_is_bound(*, manager_model_id, component_kind, reviewed_identity):
    """Return whether a resident manager entry has this exact reviewed provenance."""

    if type(manager_model_id) is not str or not manager_model_id:
        return False
    if type(component_kind) is not str or component_kind not in _STANDALONE_COMPONENT_KINDS:
        return False
    reviewed_identity = _normalize_standalone_reviewed_identity(reviewed_identity)
    with _REGISTRY_LOCK:
        binding_ref = _CURRENT_STANDALONE_MANAGER_PUBLICATIONS.get(manager_model_id)
        binding = binding_ref() if binding_ref is not None else None
        if type(binding) is not _StandaloneComponentBinding or binding not in _ISSUED_STANDALONE_COMPONENT_BINDINGS:
            if binding_ref is not None:
                _CURRENT_STANDALONE_MANAGER_PUBLICATIONS.pop(manager_model_id, None)
            return False
        return (
            binding._manager_model_id == manager_model_id
            and binding._component_kind == component_kind
            and _standalone_binding_reviewed_identity(binding) == reviewed_identity
        )


def bind_standalone_component_output(
    component,
    *,
    issuer,
    component_kind,
    reviewed_identity,
):
    """Publish and seal one reviewed standalone ComponentsManager payload."""

    if not _is_issued_standalone_component_issuer(issuer):
        raise ValueError("AutoModelLoader received an invalid standalone component issuer.")
    if type(component) is not dict:
        raise ValueError("AutoModelLoader can bind only an exact ComponentsManager metadata dictionary.")
    if _STANDALONE_COMPONENT_BINDING_KEY in component:
        raise ValueError("AutoModelLoader cannot republish an already-bound component payload.")
    if type(component_kind) is not str or component_kind not in _STANDALONE_COMPONENT_KINDS:
        raise ValueError("AutoModelLoader received an invalid standalone component kind.")
    reviewed_identity = _normalize_standalone_reviewed_identity(reviewed_identity)
    repo_source, repo_id, revision, subfolder, class_name, config_fingerprint = reviewed_identity
    manager_model_id = component.get("model_id")
    if (
        type(manager_model_id) is not str
        or not manager_model_id
        or len(manager_model_id) > 4096
        or "\x00" in manager_model_id
    ):
        raise ValueError("AutoModelLoader publication is missing its managed component identity.")

    expected_metadata = {
        "repo_source": repo_source,
        "repo_id": repo_id,
        "revision": revision,
        "class_name": class_name,
    }
    for field, expected_value in expected_metadata.items():
        if (
            field not in component
            or component.get(field) != expected_value
            or type(component.get(field)) is not type(expected_value)
        ):
            raise ValueError("AutoModelLoader publication does not match its reviewed component identity.")
    if component.get("trust_remote_code") is not False:
        raise ValueError("AutoModelLoader publication must keep repository code disabled.")

    with _REGISTRY_LOCK:
        if issuer not in _ISSUED_STANDALONE_COMPONENT_ISSUERS:
            raise ValueError("AutoModelLoader standalone component issuer is no longer valid.")
        resident_ref = _CURRENT_STANDALONE_MANAGER_PUBLICATIONS.get(manager_model_id)
        resident_binding = resident_ref() if resident_ref is not None else None
        if resident_binding is not None and (
            resident_binding._component_kind != component_kind
            or _standalone_binding_reviewed_identity(resident_binding) != reviewed_identity
        ):
            raise ValueError(
                "AutoModelLoader cannot republish a resident component under a different reviewed identity."
            )
        binding = _StandaloneComponentBinding(
            _ISSUER_SEAL,
            issuer=issuer,
            component_kind=component_kind,
            manager_model_id=manager_model_id,
            repo_source=repo_source,
            repo_id=repo_id,
            revision=revision,
            subfolder=subfolder,
            class_name=class_name,
            config_fingerprint=config_fingerprint,
        )
        _ISSUED_STANDALONE_COMPONENT_BINDINGS.add(binding)
        component[_STANDALONE_COMPONENT_BINDING_KEY] = binding
        _CURRENT_STANDALONE_COMPONENT_PUBLICATIONS[issuer] = weakref.ref(binding)
        binding_ref = weakref.ref(
            binding,
            lambda dead_ref, model_id=manager_model_id: _forget_standalone_manager_publication(
                model_id,
                dead_ref,
            ),
        )
        _CURRENT_STANDALONE_MANAGER_PUBLICATIONS[manager_model_id] = binding_ref
    return component


def require_standalone_component_binding(
    component,
    *,
    label,
    expected_kind=None,
    expected_binding=None,
    expected_issuer=None,
    expected_reviewed_identity=None,
):
    """Require one current, exact AutoModelLoader component publication."""

    if type(component) is not dict:
        raise ValueError(f"Connected standalone Diffusers {label} is not a managed component payload.")
    binding = component.get(_STANDALONE_COMPONENT_BINDING_KEY)
    if type(binding) is not _StandaloneComponentBinding:
        raise ValueError(
            f"Connected standalone Diffusers {label} is missing its process-local AutoModelLoader binding. "
            "Rerun the Load Model node and reconnect the component."
        )
    with _REGISTRY_LOCK:
        issued = (
            binding in _ISSUED_STANDALONE_COMPONENT_BINDINGS
            and binding._issuer in _ISSUED_STANDALONE_COMPONENT_ISSUERS
        )
    if not issued:
        raise ValueError(
            f"Connected standalone Diffusers {label} is missing its process-local AutoModelLoader binding. "
            "Rerun the Load Model node and reconnect the component."
        )
    if not _is_current_standalone_component_binding(binding):
        raise ValueError(
            f"Connected standalone Diffusers {label} is no longer the current AutoModelLoader publication. "
            "Rerun the Load Model node and reconnect the component."
        )

    _validate_standalone_component_metadata(component, binding, label=label)
    if expected_kind is not None and binding._component_kind != expected_kind:
        raise ValueError(
            f"Connected standalone Diffusers {label} is a '{binding._component_kind}' component, "
            f"not '{expected_kind}'."
        )
    if expected_binding is not None and binding is not expected_binding:
        raise ValueError(f"Connected standalone Diffusers {label} comes from a different component publication.")
    if expected_issuer is not None and binding._issuer is not expected_issuer:
        raise ValueError(f"Connected standalone Diffusers {label} comes from a different Load Model node.")
    if expected_reviewed_identity is not None:
        expected_reviewed_identity = _normalize_standalone_reviewed_identity(expected_reviewed_identity)
        if _standalone_binding_reviewed_identity(binding) != expected_reviewed_identity:
            raise ValueError(
                f"Connected standalone Diffusers {label} does not match the current reviewed component identity."
            )
    return binding


def require_sdxl_controlnet_component_binding(component, *, union=False, expected_binding=None):
    """Require one exact ordinary or Union SDXL ControlNet contract.

    ControlNet Union uses the same generic graph port, so the process-local
    reviewed class identity is the authority that keeps the still-dormant
    Union route from being admitted through the ordinary ControlNet path.
    """

    binding = require_standalone_component_binding(
        component,
        label="ControlNet model",
        expected_kind="controlnet",
        expected_binding=expected_binding,
    )
    if type(union) is not bool:
        raise TypeError("SDXL ControlNet variant selection must be a boolean.")
    expected_class = "ControlNetUnionModel" if union else "ControlNetModel"
    if binding._class_name != expected_class:
        variant = "Union" if union else "ordinary"
        raise ValueError(f"SDXL {variant} ControlNet execution requires an exact {expected_class} component.")
    return binding


def _component_id_inventory(component):
    pending = [(component, "", 0)]
    visited = set()
    values = []
    value_count = 0
    while pending:
        value, path, depth = pending.pop()
        value_count += 1
        if value_count > 2048 or depth > 32:
            raise ValueError("A ModelsLoader component payload exceeds the safe nested-value limit.")
        if isinstance(value, dict):
            value_id = id(value)
            if value_id in visited:
                continue
            visited.add(value_id)
            if len(value) > 512:
                raise ValueError("A ModelsLoader component payload exceeds the safe container-size limit.")
            for key, nested in value.items():
                if key is _PIPELINE_BINDING_KEY or key is _STANDALONE_COMPONENT_BINDING_KEY:
                    continue
                nested_path = f"{path}.{key}" if path else str(key)
                if key == "model_id":
                    if not isinstance(nested, str) or not nested:
                        raise ValueError("A ModelsLoader component payload contains an invalid model_id.")
                    values.append((nested_path, nested))
                elif isinstance(nested, (dict, list, tuple)):
                    pending.append((nested, nested_path, depth + 1))
        elif isinstance(value, (list, tuple)):
            if len(value) > 512:
                raise ValueError("A ModelsLoader component payload exceeds the safe container-size limit.")
            pending.extend((nested, f"{path}[{index}]", depth + 1) for index, nested in enumerate(value))
    if not values:
        raise ValueError("A ModelsLoader component payload is missing its managed model_id.")
    return tuple(sorted(values))


def issue_pipeline_instance_token(*, model_type, repo_id, repo_source, revision):
    """Mint the identity token for one successful ModelsLoader execution."""

    if not isinstance(model_type, str) or not model_type:
        raise ValueError("A pipeline instance binding requires a model type.")
    if not isinstance(repo_id, str) or not repo_id:
        raise ValueError("A pipeline instance binding requires a repository identity.")
    if not isinstance(repo_source, str) or not repo_source:
        raise ValueError("A pipeline instance binding requires a repository source.")
    if revision is not None and not isinstance(revision, str):
        raise ValueError("A pipeline instance binding revision must be a string or null.")
    token = _PipelineInstanceToken(
        _ISSUER_SEAL,
        model_type=model_type,
        repo_id=repo_id,
        repo_source=repo_source,
        revision=revision,
    )
    with _REGISTRY_LOCK:
        _ISSUED_BINDINGS.add(token)
    return token


def bind_loader_outputs(loaded_components, token):
    """Attach one private binding to every top-level ModelsLoader output."""

    if not _is_issued_binding(token):
        raise ValueError("ModelsLoader received an invalid pipeline instance binding.")
    for output_name, value in loaded_components.items():
        if isinstance(value, dict):
            role = _LOADER_OUTPUT_ROLES.get(output_name)
            if role is None:
                raise ValueError(f"ModelsLoader cannot bind unknown component output '{output_name}'.")
            component_binding = _ComponentBinding(
                _ISSUER_SEAL,
                pipeline_token=token,
                role=role,
                component_ids=_component_id_inventory(value),
            )
            with _REGISTRY_LOCK:
                _ISSUED_COMPONENT_BINDINGS.add(component_binding)
            value[_PIPELINE_BINDING_KEY] = component_binding
    return loaded_components


def _validate_component_metadata(component, token, *, label):
    expected = {
        "model_type": token._model_type,
        "repo_id": token._repo_id,
        "repo_source": token._repo_source,
        "revision": token._revision,
    }
    for field, expected_value in expected.items():
        if component.get(field) != expected_value:
            raise ValueError(f"Connected Modular Diffusers {label} does not match its backend-issued loader binding.")


def require_component_binding(
    component,
    *,
    label,
    expected_model_type=None,
    expected_token=None,
    expected_role=None,
):
    """Validate one authoritative component dictionary and return its token."""

    if not isinstance(component, dict):
        raise ValueError(f"Connected Modular Diffusers {label} is not a managed component payload.")
    component_binding = component.get(_PIPELINE_BINDING_KEY)
    if not _is_issued_component_binding(component_binding):
        raise ValueError(
            f"Connected Modular Diffusers {label} is missing its process-local ModelsLoader binding. "
            "Rerun the Models Loader and reconnect the component."
        )
    token = component_binding._pipeline_token
    if not _is_issued_binding(token):
        raise ValueError(f"Connected Modular Diffusers {label} has an invalid loader execution binding.")
    _validate_component_metadata(component, token, label=label)
    if _component_id_inventory(component) != component_binding._component_ids:
        raise ValueError(f"Connected Modular Diffusers {label} model identity changed after ModelsLoader publication.")
    if expected_role is not None and component_binding._role != expected_role:
        raise ValueError(
            f"Connected Modular Diffusers {label} came from loader role '{component_binding._role}', "
            f"not '{expected_role}'."
        )
    if expected_model_type is not None and token._model_type != expected_model_type:
        raise ValueError(
            f"Connected Modular Diffusers {label} belongs to '{token._model_type}', not '{expected_model_type}'."
        )
    if expected_token is not None and token is not expected_token:
        raise ValueError(f"Connected Modular Diffusers {label} comes from a different Models Loader execution.")
    return token


def require_matching_token_bearers(value, expected_token, *, label):
    """Reject every nested managed component issued by another loader run."""

    max_depth = 32
    max_values = 2048
    max_container_items = 512
    visited = set()
    pending = [(value, 0)]
    value_count = 0
    if not _is_issued_binding(expected_token):
        raise ValueError("The authoritative Modular Diffusers loader binding is invalid.")
    while pending:
        item, depth = pending.pop()
        value_count += 1
        if value_count > max_values or depth > max_depth:
            raise ValueError(f"Connected Modular Diffusers {label} exceeds the safe nested-value limit.")
        if isinstance(item, dict):
            item_id = id(item)
            if item_id in visited:
                continue
            visited.add(item_id)
            if len(item) > max_container_items:
                raise ValueError(f"Connected Modular Diffusers {label} exceeds the safe container-size limit.")
            if _PIPELINE_BINDING_KEY in item:
                require_component_binding(
                    item,
                    label=label,
                    expected_model_type=expected_token._model_type,
                    expected_token=expected_token,
                )
            for key, nested in item.items():
                if key is not _PIPELINE_BINDING_KEY and key is not _STANDALONE_COMPONENT_BINDING_KEY:
                    pending.append((nested, depth + 1))
        elif isinstance(item, (list, tuple)):
            if len(item) > max_container_items:
                raise ValueError(f"Connected Modular Diffusers {label} exceeds the safe container-size limit.")
            pending.extend((nested, depth + 1) for nested in item)


def require_route_state_shape_before_identity_resolution(kwargs):
    """Reject serialized or misplaced route values before model identity scanning."""

    route_value = kwargs.get(ROUTE_STATE_INPUT)
    if route_value is not None and type(route_value) is not _ModularRouteState:
        raise ValueError(
            "Modular route state must be the opaque value emitted by the preceding backend action; "
            "serialized mappings and user-provided values are not accepted."
        )

    pending = [(value, 0) for name, value in kwargs.items() if name != ROUTE_STATE_INPUT]
    visited = set()
    value_count = 0
    while pending:
        value, depth = pending.pop()
        value_count += 1
        if value_count > 2048 or depth > 32:
            raise ValueError("Connected Modular Diffusers inputs exceed the safe nested-value limit.")
        if type(value) is _ModularRouteState:
            raise ValueError("A Modular route state was connected to an undeclared graph field.")
        if isinstance(value, Mapping):
            value_id = id(value)
            if value_id in visited:
                continue
            visited.add(value_id)
            if len(value) > 512:
                raise ValueError("Connected Modular Diffusers inputs exceed the safe container-size limit.")
            pending.extend((nested, depth + 1) for nested in value.values())
        elif isinstance(value, (list, tuple)):
            if len(value) > 512:
                raise ValueError("Connected Modular Diffusers inputs exceed the safe container-size limit.")
            pending.extend((nested, depth + 1) for nested in value)


def reject_route_reserved_inputs_before_identity_resolution(kwargs, *, allowed_direct_inputs=()):
    """Bound and reject route-owned names before recursive model recovery."""

    allowed_direct_inputs = frozenset(allowed_direct_inputs)
    pending = [(kwargs, 0)]
    visited = set()
    value_count = 0
    while pending:
        value, depth = pending.pop()
        value_count += 1
        if value_count > 2048 or depth > 32:
            raise ValueError("Connected Modular Diffusers inputs exceed the safe nested-value limit.")
        if isinstance(value, Mapping):
            value_id = id(value)
            if value_id in visited:
                continue
            visited.add(value_id)
            if len(value) > 512:
                raise ValueError("Connected Modular Diffusers inputs exceed the safe container-size limit.")
            collision = ROUTE_RESERVED_PIPELINE_INPUTS.intersection(value)
            if depth == 0:
                collision -= allowed_direct_inputs
            if collision:
                raise ValueError(
                    "Modular route fields are backend-managed and cannot participate in model identity: "
                    + ", ".join(sorted(collision))
                )
            for nested in value.values():
                if type(nested) is not _ModularRouteState:
                    pending.append((nested, depth + 1))
        elif isinstance(value, (list, tuple)):
            if len(value) > 512:
                raise ValueError("Connected Modular Diffusers inputs exceed the safe container-size limit.")
            pending.extend((nested, depth + 1) for nested in value)


def validate_route_field_contract(kwargs, node_config):
    """Enforce the selected backend schema after identity recovery."""

    route_value = kwargs.get(ROUTE_STATE_INPUT)
    declared = ROUTE_STATE_INPUT in node_config.get("input_names", ())
    if route_value is not None and not declared:
        raise ValueError(
            "The selected Modular Diffusers action does not declare route state. "
            "Remove the stale route-state edge and reconnect the current model workflow."
        )
    if route_value is not None and not _is_issued_route_state(route_value):
        raise ValueError("The Modular route state was not issued by this backend process.")


def _route_reserved_inputs(*, model_type, action):
    contract = route_contract_for_model_type(model_type)
    if contract != _SDXL_ROUTE_CONTRACT:
        return ROUTE_RESERVED_PIPELINE_INPUTS, ROUTE_RESERVED_PIPELINE_INPUTS
    common = frozenset({"generator", "processed_mask_image", "mask_overlay_kwargs"})
    if action == "denoise":
        return common | {"crops_coords"}, common | {"crops_coords", "mask", "masked_image_latents"}
    if action == "decoder":
        decode_owned = {"image", "mask_image", "padding_mask_crop", "crops_coords"}
        return common | decode_owned, common | decode_owned
    return common, common


def reject_route_reserved_inputs(kwargs, *, bundle_names=(), model_type=None, action=None):
    """Prevent graph values or generic bundles from overwriting sealed route data."""

    direct_reserved, bundle_reserved = _route_reserved_inputs(model_type=model_type, action=action)
    for name in direct_reserved:
        if kwargs.get(name) is not None:
            raise ValueError(f"Modular route input '{name}' is backend-managed and cannot be supplied directly.")
    for bundle_name in bundle_names:
        bundle = kwargs.get(bundle_name)
        if not isinstance(bundle, Mapping):
            continue
        collision = bundle_reserved.intersection(bundle)
        if collision:
            raise ValueError(
                f"Modular input bundle '{bundle_name}' cannot overwrite backend-managed route fields: "
                + ", ".join(sorted(collision))
            )


def route_cache_params_equal(previous, current, *, fallback):
    """Compare route-bearing cache inputs without value-comparing tensors.

    A route authenticates exact tensor objects, while the default node cache
    deliberately treats equal-valued tensors as unchanged.  Check tensor
    identity first whenever either input set carries a route, then retain the
    established comparison for every other value.  Rewrapped builtin lists
    containing the same tensor objects remain cacheable.
    """

    if not isinstance(previous, dict) or not isinstance(current, dict):
        return fallback(previous, current)
    if ROUTE_STATE_INPUT not in previous and ROUTE_STATE_INPUT not in current:
        return fallback(previous, current)

    pending = [(previous, current, 0)]
    visited = set()
    value_count = 0
    while pending:
        left, right, depth = pending.pop()
        value_count += 1
        if value_count > 2048 or depth > 32:
            return False
        if type(left) is not type(right):
            return False
        if type(left) is torch.Tensor:
            if left is not right:
                return False
            continue
        if type(left) is dict:
            pair = (id(left), id(right))
            if pair in visited:
                continue
            visited.add(pair)
            if len(left) > 512 or len(right) > 512 or set(left) != set(right):
                return False
            pending.extend((left[key], right[key], depth + 1) for key in left)
        elif type(left) in (list, tuple):
            pair = (id(left), id(right))
            if pair in visited:
                continue
            visited.add(pair)
            if len(left) > 512 or len(left) != len(right):
                return False
            pending.extend((left_item, right_item, depth + 1) for left_item, right_item in zip(left, right))

    return fallback(previous, current)


def effective_modular_block_input(kwargs, *, node_input_names, block_input_names, target_name):
    """Resolve one effective input using the action's generic bundle-flattening rules."""

    block_input_names = set(block_input_names)
    effective_value = None
    for input_name in node_input_names:
        if input_name not in kwargs:
            continue
        value = kwargs.get(input_name)
        if isinstance(value, dict) and input_name not in block_input_names:
            if target_name in block_input_names and target_name in value:
                effective_value = value[target_name]
        elif input_name == target_name and input_name in block_input_names:
            effective_value = value
    return effective_value


def _clone_generator(generator):
    if not isinstance(generator, torch.Generator):
        raise TypeError("Modular route generation requires a Torch generator.")
    clone_state = getattr(generator, "clone_state", None)
    if callable(clone_state):
        return clone_state()
    clone = torch.Generator(device=generator.device)
    clone.set_state(generator.get_state().clone())
    return clone


def _devices_compatible(left, right):
    left_device = torch.device(left)
    right_device = torch.device(right)
    if left_device.type != right_device.type:
        return False
    if left_device.type == "cpu":
        return True
    return left_device.index is None or right_device.index is None or left_device.index == right_device.index


def _wan_execution_device(value, *, label):
    if value is None:
        raise ValueError(f"Wan {label} execution device is unavailable.")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Wan {label} execution device is invalid.") from error
    if device.type == "meta":
        raise ValueError(f"Wan {label} execution device must be resident.")
    return device


def _require_wan_tensor_device(tensor, execution_device, *, label):
    device = _wan_execution_device(execution_device, label=label)
    if type(tensor) is not torch.Tensor or not _devices_compatible(tensor.device, device):
        raise ValueError(f"{label} must be resident on the producing Wan execution device.")
    return device


def _require_wan_reference_device(reference, execution_device, *, label):
    device = _wan_execution_device(execution_device, label=label)
    _issued_kind, issued_refs = reference
    for issued_ref, issued_seal in issued_refs:
        issued_item = issued_ref()
        if issued_item is None:
            raise ValueError(f"The {label} paired with this Modular route state is no longer resident.")
        _require_tensor_seal(issued_item, issued_seal, label=label)
        if not _devices_compatible(issued_item.device, device):
            raise ValueError(f"{label} must remain on its producing Wan execution device.")
    return device


def _new_route_state(*, stage, binding, seed, contract, payload):
    state = _ModularRouteState(
        _ISSUER_SEAL,
        stage=stage,
        binding=binding,
        seed=seed,
        contract=contract,
        payload=payload,
    )
    with _REGISTRY_LOCK:
        _ISSUED_ROUTE_STATES.add(state)
    return state


def _validate_sdxl_overlay_media_pair(original_image, original_mask):
    for label, value in (("SDXL original image", original_image), ("SDXL original mask", original_mask)):
        if not isinstance(value, Image.Image):
            raise TypeError(f"{label} must be a PIL image when mask crop overlay is enabled.")
        width, height = value.size
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or width > _MAX_OVERLAY_EDGE_PIXELS
            or height > _MAX_OVERLAY_EDGE_PIXELS
        ):
            raise ValueError(
                f"{label} dimensions must be positive and no larger than {_MAX_OVERLAY_EDGE_PIXELS} pixels per edge."
            )
    if original_image.size != original_mask.size:
        raise ValueError("SDXL crop overlay image and mask dimensions must match exactly.")
    aggregate_pixels = original_image.width * original_image.height + original_mask.width * original_mask.height
    if aggregate_pixels > _MAX_OVERLAY_AGGREGATE_PIXELS:
        raise ValueError(
            "SDXL crop overlay image and mask exceed the cumulative 16-Mi-pixel snapshot limit."
        )


def validate_sdxl_crop_overlay_inputs(padding_mask_crop, original_image, original_mask):
    """Fail closed on untrusted crop media before any SDXL block is initialized."""

    if padding_mask_crop is None:
        return
    if (
        type(padding_mask_crop) is not int
        or padding_mask_crop < 0
        or padding_mask_crop > _MAX_MASK_CROP_PADDING
    ):
        raise ValueError(
            f"SDXL mask crop padding must be a canonical integer from 0 through {_MAX_MASK_CROP_PADDING}, or null."
        )
    if original_mask is None:
        raise ValueError("SDXL mask crop padding requires a mask image.")
    _validate_sdxl_overlay_media_pair(original_image, original_mask)


def _freeze_sdxl_overlay_media(original_image, original_mask, *, crops_coords):
    """Validate pinned PIL overlay geometry and copy within one cumulative limit."""

    _validate_sdxl_overlay_media_pair(original_image, original_mask)
    x1, y1, x2, y2 = crops_coords
    if not (0 <= x1 < x2 <= original_image.width and 0 <= y1 < y2 <= original_image.height):
        raise ValueError("SDXL crop coordinates must describe a nonempty region within the original image and mask.")
    return ("pil", original_image.copy()), ("pil", original_mask.copy())


def _materialize_overlay_media(snapshot):
    kind, value = snapshot
    if kind == "pil":
        return value.copy()
    raise ValueError("The sealed Modular overlay snapshot has an invalid media kind.")


def sdxl_vae_geometry_from_component(vae):
    """Read the bounded latent geometry used by pinned SDXL blocks."""

    config = getattr(vae, "config", None)
    latent_channels = getattr(config, "latent_channels", None)
    block_out_channels = getattr(config, "block_out_channels", None)
    if type(latent_channels) is not int or latent_channels != 4:
        raise ValueError("The connected SDXL VAE must declare the pinned four-channel latent contract.")
    if (
        type(block_out_channels) not in (list, tuple)
        or not block_out_channels
        or len(block_out_channels) > 16
    ):
        raise ValueError("The connected SDXL VAE has an invalid scale-factor contract.")
    scale_factor = 2 ** (len(block_out_channels) - 1)
    return latent_channels, scale_factor


def _canonical_wan_dimension(value, *, label):
    if type(value) is not int or not 1 <= value <= _MAX_WAN_DIMENSION:
        raise ValueError(
            f"Wan {label} must be a canonical integer from 1 through {_MAX_WAN_DIMENSION}."
        )
    return value


def _validate_wan_requested_dimensions(height, width):
    height = _canonical_wan_dimension(height, label="height")
    width = _canonical_wan_dimension(width, label="width")
    if height * width > _MAX_WAN_REQUEST_AREA:
        raise ValueError("Wan requested dimensions exceed the bounded 16-Mi-pixel area budget.")
    return height, width


def _validate_wan_source_pair(image, last_image):
    values = (("source image", image),) if last_image is None else (
        ("source image", image),
        ("last image", last_image),
    )
    aggregate_pixels = 0
    for label, value in values:
        if type(value) is not Image.Image:
            raise TypeError(f"Wan {label} must be one exact PIL Image value.")
        width, height = value.size
        if (
            type(width) is not int
            or type(height) is not int
            or width <= 0
            or height <= 0
            or width > _MAX_WAN_DIMENSION
            or height > _MAX_WAN_DIMENSION
        ):
            raise ValueError(
                f"Wan {label} dimensions must be positive and no larger than {_MAX_WAN_DIMENSION} pixels per edge."
            )
        bands = value.getbands()
        if type(bands) is not tuple or not 1 <= len(bands) <= 4:
            raise ValueError(f"Wan {label} has an unsupported pixel-band contract.")
        if width * height * 4 > _MAX_WAN_SOURCE_BYTES:
            raise ValueError(f"Wan {label} exceeds the bounded 64-MiB pixel snapshot budget.")
        aggregate_pixels += width * height
    if aggregate_pixels > _MAX_OVERLAY_AGGREGATE_PIXELS:
        raise ValueError("Wan source and last images exceed the cumulative 16-Mi-pixel route budget.")
    return _WAN_FLF_WORKFLOW if last_image is not None else _WAN_I2V_WORKFLOW


def _wan_image_seal(image, *, label):
    try:
        image.load()
        # Hash a bounded canonical rendering so palette/transparency changes
        # cannot preserve the same index bytes while changing decoded pixels.
        pixels = image.convert("RGBA").tobytes()
    except Exception as error:
        raise ValueError(f"Wan {label} pixels could not be sealed before execution.") from error
    if len(pixels) > _MAX_WAN_SOURCE_BYTES:
        raise ValueError(f"Wan {label} exceeds the bounded 64-MiB pixel snapshot budget.")
    return (image.mode, image.size, image.getbands(), hashlib.sha256(pixels).digest())


def snapshot_wan_source_media(image, last_image=None):
    """Seal bounded exact source objects before a split Wan image action."""

    workflow = _validate_wan_source_pair(image, last_image)
    source_ref = _component_identity_reference(image, label="Wan source image")
    last_ref = (
        _component_identity_reference(last_image, label="Wan last image")
        if last_image is not None
        else None
    )
    return (
        workflow,
        source_ref,
        _wan_image_seal(image, label="source image"),
        last_ref,
        _wan_image_seal(last_image, label="last image") if last_image is not None else None,
    )


def _require_wan_media_snapshot(snapshot, image, last_image):
    if type(snapshot) is not tuple or len(snapshot) != 5:
        raise ValueError("Wan source media is missing its bounded pre-execution snapshot.")
    workflow, source_ref, source_seal, last_ref, last_seal = snapshot
    if workflow != _validate_wan_source_pair(image, last_image):
        raise ValueError("Wan source media changed between I2V and FLF routing semantics.")
    issued_source = source_ref() if isinstance(source_ref, weakref.ReferenceType) else None
    if issued_source is not image:
        raise ValueError("Wan source image is not the exact object sealed for this route.")
    if _wan_image_seal(image, label="source image") != source_seal:
        raise ValueError("Wan source image pixels changed after route snapshotting.")
    if last_image is None:
        if last_ref is not None or last_seal is not None:
            raise ValueError("Wan I2V source state unexpectedly contains a last image.")
    else:
        issued_last = last_ref() if isinstance(last_ref, weakref.ReferenceType) else None
        if issued_last is not last_image:
            raise ValueError("Wan last image is not the exact object sealed for this route.")
        if _wan_image_seal(last_image, label="last image") != last_seal:
            raise ValueError("Wan last-image pixels changed after route snapshotting.")
    return snapshot


def _wan_payload_media_snapshot(payload):
    return (
        payload._workflow,
        payload._source_image_ref,
        payload._source_image_seal,
        payload._last_image_ref,
        payload._last_image_seal,
    )


def _validate_wan_resized_media(
    *,
    workflow,
    resized_image,
    resized_last_image,
    height,
    width,
    expected_last_size,
    stage,
):
    if type(resized_image) is not Image.Image or resized_image.size != (width, height):
        raise ValueError(f"Pinned Wan {stage} did not publish the exact resized source geometry.")
    if workflow == _WAN_FLF_WORKFLOW:
        if (
            type(expected_last_size) is not tuple
            or len(expected_last_size) != 2
            or type(resized_last_image) is not Image.Image
            or resized_last_image.size != expected_last_size
        ):
            raise ValueError(f"Pinned Wan FLF {stage} did not publish the exact resized last-image geometry.")
    elif expected_last_size is not None or resized_last_image is not None:
        raise ValueError(f"Pinned Wan I2V {stage} unexpectedly published a resized last image.")


def _validate_wan_intermediate_dimensions(height, width, *, label):
    if (
        type(height) is not int
        or type(width) is not int
        or height <= 0
        or width <= 0
        or height > _MAX_WAN_DIMENSION
        or width > _MAX_WAN_DIMENSION
    ):
        raise ValueError(
            f"Wan {label} must resolve to positive edges no larger than {_MAX_WAN_DIMENSION} pixels."
        )
    if height * width > _MAX_OVERLAY_AGGREGATE_PIXELS:
        raise ValueError(f"Wan {label} exceeds the bounded 16-Mi-pixel intermediate budget.")
    return height, width


def _wan_clip_resize_dimensions(height, width, *, workflow):
    _validate_wan_intermediate_dimensions(height, width, label="CLIP source intermediate")
    if workflow == _WAN_FLF_WORKFLOW:
        # The distinct official FLF artifact uses shortest_edge=224. Keep its
        # preparatory state contract exact even though that artifact is not an
        # executable/cataloged MoDiff dependency yet.
        short_edge = min(height, width)
        long_edge = max(height, width)
        resized_short = _WAN_IMAGE_SIZE
        resized_long = int(_WAN_IMAGE_SIZE * long_edge / short_edge)
        if height <= width:
            resized_height, resized_width = resized_short, resized_long
        else:
            resized_height, resized_width = resized_long, resized_short
        return _validate_wan_intermediate_dimensions(
            resized_height,
            resized_width,
            label="FLF CLIP resize intermediate",
        )
    if workflow != _WAN_I2V_WORKFLOW:
        raise ValueError("Wan CLIP resize received an unknown workflow contract.")
    # The cataloged I2V repository publishes explicit height/width dimensions,
    # so Transformers resizes directly to 224x224.
    return _validate_wan_intermediate_dimensions(
        _WAN_IMAGE_SIZE,
        _WAN_IMAGE_SIZE,
        label="CLIP resize intermediate",
    )


def _wan_last_image_intermediate_dimensions(last_image, *, height, width, stage):
    if last_image is None:
        return None
    resize_ratio = max(width / last_image.width, height / last_image.height)
    resized_width = round(last_image.width * resize_ratio)
    resized_height = round(last_image.height * resize_ratio)
    _validate_wan_intermediate_dimensions(
        resized_height,
        resized_width,
        label=f"FLF {stage} last-image intermediate",
    )
    return resized_height, resized_width


def wan_area_budget_dimensions(image, height, width):
    """Return the exact pinned Wan resize result for one area-budget pass."""

    _validate_wan_source_pair(image, None)
    height, width = _validate_wan_requested_dimensions(height, width)
    aspect_ratio = image.height / image.width
    mod_value = _WAN_SPATIAL_SCALE * _WAN_PATCH_SIZE_SPATIAL
    resolved_height = round(math.sqrt(height * width * aspect_ratio)) // mod_value * mod_value
    resolved_width = round(math.sqrt(height * width / aspect_ratio)) // mod_value * mod_value
    if resolved_height <= 0 or resolved_width <= 0:
        raise ValueError("Wan area-budget resize resolved to an empty dimension.")
    return _validate_wan_intermediate_dimensions(
        resolved_height,
        resolved_width,
        label="area-budget resize",
    )


def preflight_wan_image_encoder_inputs(*, image, last_image, height, width):
    """Bound both first-pass Wan/FLF resizes before pipeline initialization."""

    workflow = _validate_wan_source_pair(image, last_image)
    requested_height, requested_width = _validate_wan_requested_dimensions(height, width)
    first_height, first_width = wan_area_budget_dimensions(image, requested_height, requested_width)
    clip_height, clip_width = _wan_clip_resize_dimensions(
        first_height,
        first_width,
        workflow=workflow,
    )
    last_intermediate = _wan_last_image_intermediate_dimensions(
        last_image,
        height=first_height,
        width=first_width,
        stage="image-encoder",
    )
    return (
        workflow,
        requested_height,
        requested_width,
        first_height,
        first_width,
        clip_height,
        clip_width,
        last_intermediate,
    )


def require_cataloged_wan_action_source(*, image, last_image, binding):
    """Bind the selected Wan route to its exact reviewed loader artifact."""

    workflow = _validate_wan_source_pair(image, last_image)
    if not _is_issued_binding(binding) or binding._model_type != "WanImage2VideoModularPipeline":
        raise ValueError("Wan image execution requires the current Models Loader publication.")
    expected_repository = dict(WAN_WORKFLOW_REPOSITORIES).get(workflow)
    if expected_repository is None:
        raise ValueError("Wan image execution selected an unknown workflow contract.")
    expected_revision = require_catalog_revision(
        expected_repository,
        model_type="WanImage2VideoModularPipeline",
    )
    if (
        binding._repo_source != "hub"
        or binding._repo_id != expected_repository
        or binding._revision != expected_revision
    ):
        raise ValueError(
            "Wan image execution does not match the reviewed immutable artifact for the selected workflow."
        )
    return workflow


def wan_vae_geometry_from_component(vae):
    """Validate the exact VAE defaults assumed by the pinned split Wan blocks."""

    config = getattr(vae, "config", None)
    z_dim = getattr(config, "z_dim", None)
    temporal_downsample = getattr(vae, "temperal_downsample", None)
    if type(z_dim) is not int or z_dim != _WAN_LATENT_CHANNELS:
        raise ValueError("The connected Wan VAE must declare the pinned z_dim 16 contract.")
    if (
        type(temporal_downsample) not in (list, tuple)
        or tuple(temporal_downsample) != (False, True, True)
    ):
        raise ValueError("The connected Wan VAE must use temporal_downsample (False, True, True).")
    spatial_scale = 2 ** len(temporal_downsample)
    temporal_scale = 2 ** sum(temporal_downsample)
    if (spatial_scale, temporal_scale) != (_WAN_SPATIAL_SCALE, _WAN_TEMPORAL_SCALE):
        raise ValueError("The connected Wan VAE must use spatial scale 8 and temporal scale 4.")
    expected_config = {
        "in_channels": 3,
        "out_channels": 3,
        "patch_size": None,
        "scale_factor_spatial": _WAN_SPATIAL_SCALE,
        "scale_factor_temporal": _WAN_TEMPORAL_SCALE,
    }
    for name, expected in expected_config.items():
        value = getattr(config, name, _NOT_PROVIDED)
        if value != expected or (expected is not None and type(value) is not type(expected)):
            raise ValueError(f"The connected Wan VAE must declare pinned {name}={expected!r}.")
    return z_dim, spatial_scale, temporal_scale


def _wan_vae_config_seal(vae):
    z_dim, spatial_scale, temporal_scale = wan_vae_geometry_from_component(vae)
    config = getattr(vae, "config", None)

    def finite_vector(name, *, nonzero=False):
        value = getattr(config, name, None)
        if type(value) not in (list, tuple) or len(value) != z_dim:
            raise ValueError(f"The connected Wan VAE must declare exactly {z_dim} {name} values.")
        canonical = []
        for item in value:
            if (
                type(item) not in (int, float)
                or not math.isfinite(item)
                or abs(item) > _MAX_WAN_VAE_CONFIG_MAGNITUDE
            ):
                raise ValueError(f"The connected Wan VAE {name} values must be finite numbers.")
            if nonzero and not _MIN_WAN_POSITIVE_SCALE <= item <= _MAX_WAN_VAE_CONFIG_MAGNITUDE:
                raise ValueError(
                    f"The connected Wan VAE {name} values must be strictly positive bounded scales."
                )
            canonical.append(item)
        return tuple(canonical)

    latents_mean = finite_vector("latents_mean")
    latents_std = finite_vector("latents_std", nonzero=True)
    if latents_mean != _WAN_VAE_LATENTS_MEAN or latents_std != _WAN_VAE_LATENTS_STD:
        raise ValueError("The connected Wan VAE must retain the pinned latent mean and standard deviation.")
    return (
        z_dim,
        spatial_scale,
        temporal_scale,
        tuple(vae.temperal_downsample),
        latents_mean,
        latents_std,
    )


def wan_transformer_contract_from_component(transformer, *, workflow=None):
    """Validate the exact patch/channel contract assumed by split Wan Denoise."""

    config = getattr(transformer, "config", None)
    patch_size = getattr(config, "patch_size", None)
    in_channels = getattr(config, "in_channels", None)
    out_channels = getattr(config, "out_channels", None)
    image_dim = getattr(config, "image_dim", None)
    if (
        type(patch_size) not in (list, tuple)
        or len(patch_size) != len(_WAN_TRANSFORMER_PATCH_SIZE)
        or any(type(value) is not int for value in patch_size)
    ):
        raise ValueError("The connected Wan transformer has an invalid patch-size contract.")
    if tuple(patch_size) != _WAN_TRANSFORMER_PATCH_SIZE:
        raise ValueError("The connected Wan transformer must use pinned patch size (1, 2, 2).")
    if type(in_channels) is not int or in_channels != _WAN_TRANSFORMER_INPUT_CHANNELS:
        raise ValueError("The connected Wan transformer must use the pinned 36-channel I2V input contract.")
    if type(out_channels) is not int or out_channels != _WAN_TRANSFORMER_OUTPUT_CHANNELS:
        raise ValueError("The connected Wan transformer must use the pinned 16-channel output contract.")
    if type(image_dim) is not int or image_dim != _WAN_IMAGE_EMBED_DIM:
        raise ValueError("The connected Wan transformer must use the pinned 1280-wide image dimension.")
    pos_embed_seq_len = getattr(config, "pos_embed_seq_len", None)
    if workflow == _WAN_I2V_WORKFLOW:
        if pos_embed_seq_len is not None:
            raise ValueError("The cataloged Wan I2V transformer must not declare FLF positional embeddings.")
    elif workflow == _WAN_FLF_WORKFLOW:
        if type(pos_embed_seq_len) is not int or pos_embed_seq_len != 514:
            raise ValueError("The preparatory Wan FLF transformer contract requires pos_embed_seq_len 514.")
    elif workflow is not None:
        raise ValueError("Wan transformer validation received an unknown workflow contract.")
    elif pos_embed_seq_len is not None and (type(pos_embed_seq_len) is not int or pos_embed_seq_len != 514):
        raise ValueError("The connected Wan transformer has an unreviewed positional-embedding contract.")
    return tuple(patch_size), in_channels, out_channels, image_dim, pos_embed_seq_len


def _canonical_processor_scalar(value, *, label):
    if value is None or type(value) in (bool, int, str):
        if type(value) is str and len(value) > 256:
            raise ValueError(f"{label} exceeds the bounded processor string limit.")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must be finite.")
        return value
    if isinstance(value, Enum):
        return (
            "enum",
            type(value).__module__,
            type(value).__qualname__,
            value.name,
            _canonical_processor_scalar(value.value, label=label),
        )
    if type(value) in (list, tuple):
        if len(value) > 32:
            raise ValueError(f"{label} exceeds the bounded processor sequence limit.")
        return tuple(
            _canonical_processor_scalar(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise ValueError(f"{label} has an unsupported effective processor value.")


def _canonical_clip_size(value, *, label):
    if isinstance(value, Mapping):
        getter = value.get
    else:
        getter = lambda name: getattr(value, name, _NOT_PROVIDED)
    fields = ("height", "width", "longest_edge", "shortest_edge", "max_height", "max_width")
    canonical = []
    for name in fields:
        field_value = getter(name)
        if field_value is _NOT_PROVIDED:
            raise ValueError(f"{label} is missing its bounded '{name}' field.")
        if field_value is not None and (
            type(field_value) is not int or not 1 <= field_value <= _MAX_WAN_DIMENSION
        ):
            raise ValueError(
                f"{label}.{name} must be null or a canonical positive integer no larger than "
                f"{_MAX_WAN_DIMENSION}."
            )
        canonical.append((name, field_value))
    if not any(value is not None for _name, value in canonical):
        raise ValueError(f"{label} must declare at least one bounded effective dimension.")
    return tuple(canonical)


def wan_image_processor_config_seal(image_processor, *, workflow=_WAN_I2V_WORKFLOW):
    """Seal the bounded effective CLIP preprocessing fields used by Wan."""

    if image_processor is None:
        raise ValueError("The pinned Wan action is missing its CLIP image processor.")
    values = []
    for name in _WAN_CLIP_PROCESSOR_FIELDS:
        value = getattr(image_processor, name, _NOT_PROVIDED)
        if value is _NOT_PROVIDED:
            raise ValueError(f"The pinned Wan image processor is missing effective field '{name}'.")
        if name in {"size", "crop_size", "pad_size"}:
            if name == "pad_size" and value is None:
                pass
            else:
                value = _canonical_clip_size(value, label=f"Wan image processor {name}")
        elif name in {
            "do_resize",
            "do_center_crop",
            "do_rescale",
            "do_normalize",
            "do_convert_rgb",
        }:
            if type(value) is not bool:
                raise ValueError(f"Wan image processor {name} must be an exact boolean.")
            expected = workflow == _WAN_FLF_WORKFLOW if name == "do_center_crop" else True
            if value is not expected:
                raise ValueError(f"Wan image processor {name} does not match the reviewed setting.")
        elif name in {"do_pad", "disable_grouping"}:
            if value is not None and type(value) is not bool:
                raise ValueError(f"Wan image processor {name} must be null or an exact boolean.")
        elif name == "resample":
            if not (
                (type(value) is int and value == int(Image.Resampling.BICUBIC))
                or (type(value) is Image.Resampling and value is Image.Resampling.BICUBIC)
            ):
                raise ValueError("Wan image processor resample must be the reviewed PIL bicubic value.")
            value = int(Image.Resampling.BICUBIC)
        elif name == "rescale_factor":
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not _MIN_WAN_POSITIVE_SCALE <= value <= 1.0
            ):
                raise ValueError("Wan image processor rescale_factor must be a finite bounded positive scale.")
        elif name in {"image_mean", "image_std"}:
            if type(value) not in (list, tuple) or len(value) != 3:
                raise ValueError(f"Wan image processor {name} must contain exactly three numeric values.")
            canonical_vector = []
            for item in value:
                if (
                    type(item) not in (int, float)
                    or not math.isfinite(item)
                    or abs(item) > _MAX_WAN_PROCESSOR_NORMALIZATION
                ):
                    raise ValueError(f"Wan image processor {name} values must be finite and bounded.")
                if name == "image_std" and not _MIN_WAN_POSITIVE_SCALE <= item:
                    raise ValueError("Wan image processor image_std values must be strictly positive bounded scales.")
                canonical_vector.append(item)
            value = tuple(canonical_vector)
        else:
            value = _canonical_processor_scalar(value, label=f"Wan image processor {name}")
        values.append((name, value))
    values = tuple(values)
    effective = dict(values)
    if workflow == _WAN_I2V_WORKFLOW:
        expected_size = (
            ("height", _WAN_IMAGE_SIZE),
            ("width", _WAN_IMAGE_SIZE),
            ("longest_edge", None),
            ("shortest_edge", None),
            ("max_height", None),
            ("max_width", None),
        )
    elif workflow == _WAN_FLF_WORKFLOW:
        expected_size = (
            ("height", None),
            ("width", None),
            ("longest_edge", None),
            ("shortest_edge", _WAN_IMAGE_SIZE),
            ("max_height", None),
            ("max_width", None),
        )
    else:
        raise ValueError("Wan image processor validation received an unknown workflow contract.")
    expected_crop = (
        ("height", _WAN_IMAGE_SIZE),
        ("width", _WAN_IMAGE_SIZE),
        ("longest_edge", None),
        ("shortest_edge", None),
        ("max_height", None),
        ("max_width", None),
    )
    if effective["size"] != expected_size or effective["crop_size"] != expected_crop:
        raise ValueError("Wan image processor must retain the reviewed 224-pixel CLIP resize/crop contract.")
    if effective["rescale_factor"] != 1 / 255:
        raise ValueError("Wan image processor must retain the reviewed 1/255 rescale factor.")
    if effective["do_pad"] not in {None, False} or effective["pad_size"] is not None:
        raise ValueError("Wan image processor padding must remain disabled with no pad size.")
    if effective["image_mean"] != _WAN_CLIP_IMAGE_MEAN or effective["image_std"] != _WAN_CLIP_IMAGE_STD:
        raise ValueError("Wan image processor must retain the pinned CLIP image mean and standard deviation.")
    return (type(image_processor).__module__, type(image_processor).__qualname__, values)


def wan_image_encoder_contract_from_component(image_encoder):
    """Validate the CLIP vision geometry consumed by pinned Wan I2V blocks."""

    config = getattr(image_encoder, "config", None)
    image_size = getattr(config, "image_size", None)
    hidden_size = getattr(config, "hidden_size", None)
    patch_size = getattr(config, "patch_size", None)
    num_channels = getattr(config, "num_channels", None)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    projection_dim = getattr(config, "projection_dim", None)
    if type(image_size) is not int or image_size != _WAN_IMAGE_SIZE:
        raise ValueError("The connected Wan image encoder must use the pinned 224-pixel image size.")
    if type(hidden_size) is not int or hidden_size != _WAN_IMAGE_EMBED_DIM:
        raise ValueError("The connected Wan image encoder must use the pinned 1280-wide hidden state.")
    if type(patch_size) is not int or patch_size != _WAN_IMAGE_ENCODER_PATCH_SIZE:
        raise ValueError("The connected Wan image encoder must use the pinned patch size 14.")
    if type(num_channels) is not int or num_channels != 3:
        raise ValueError("The connected Wan image encoder must consume exactly three channels.")
    if type(num_hidden_layers) is not int or num_hidden_layers != _WAN_IMAGE_ENCODER_LAYERS:
        raise ValueError("The connected Wan image encoder must expose the pinned 32 hidden layers.")
    if type(num_attention_heads) is not int or num_attention_heads != _WAN_IMAGE_ENCODER_HEADS:
        raise ValueError("The connected Wan image encoder must expose the pinned 16 attention heads.")
    if type(projection_dim) is not int or projection_dim != _WAN_IMAGE_ENCODER_PROJECTION_DIM:
        raise ValueError("The connected Wan image encoder must use the pinned projection dimension 1024.")
    return (
        image_size,
        hidden_size,
        patch_size,
        num_channels,
        num_hidden_layers,
        num_attention_heads,
        projection_dim,
    )


def _effective_video_processor_value(video_processor, name):
    config = getattr(video_processor, "config", None)
    if isinstance(config, Mapping) and name in config:
        return config[name]
    return getattr(video_processor, name, _NOT_PROVIDED)


def wan_video_processor_config_seal(video_processor):
    """Validate and seal the pinned from-config Wan VideoProcessor defaults."""

    if video_processor is None:
        raise ValueError("The pinned Wan action is missing its video processor.")
    values = []
    for name in _WAN_VIDEO_PROCESSOR_FIELDS:
        value = _effective_video_processor_value(video_processor, name)
        if value is _NOT_PROVIDED:
            raise ValueError(f"The pinned Wan video processor is missing effective field '{name}'.")
        values.append((name, _canonical_processor_scalar(value, label=f"Wan video processor {name}")))
    values = tuple(values)
    if values != _WAN_VIDEO_PROCESSOR_DEFAULTS:
        raise ValueError("The pinned Wan video processor does not match its reviewed from-config defaults.")
    return (type(video_processor).__module__, type(video_processor).__qualname__, values)


def require_wan_video_processor(video_processor):
    """Validate the reviewed effective processor used by Wan VAE/decode steps."""

    wan_video_processor_config_seal(video_processor)
    return video_processor


def _require_component_reference(reference, component, *, label, require_connected=True):
    issued = reference() if isinstance(reference, weakref.ReferenceType) else None
    if issued is None:
        raise ValueError(f"The {label} paired with this Modular route state is no longer resident.")
    if require_connected and issued is not component:
        raise ValueError(f"The connected {label} is not the exact component paired with this route state.")
    return issued


def _validate_wan_image_embeds(image_embeds, *, workflow):
    if type(image_embeds) is not torch.Tensor:
        raise TypeError("Wan image embeddings must be an exact Torch tensor.")
    if image_embeds.layout != torch.strided or image_embeds.device.type == "meta" or image_embeds.ndim != 3:
        raise ValueError("Wan image embeddings must be a resident strided rank-3 tensor.")
    expected_batch = 2 if workflow == _WAN_FLF_WORKFLOW else 1
    if tuple(image_embeds.shape) != (expected_batch, _WAN_IMAGE_EMBED_TOKENS, _WAN_IMAGE_EMBED_DIM):
        raise ValueError("Wan image embeddings do not match the selected I2V/FLF image batch contract.")
    if not image_embeds.dtype.is_floating_point:
        raise ValueError("Wan image embeddings must use a floating-point dtype.")


def _validate_wan_image_embed_dimension(image_embeds, *, image_dim):
    if type(image_dim) is not int or image_dim <= 0 or image_embeds.shape[-1] != image_dim:
        raise ValueError("Wan image embeddings do not match the connected transformer's image dimension.")


def _validate_wan_frames(num_frames, *, workflow, temporal_scale=_WAN_TEMPORAL_SCALE):
    if type(num_frames) is not int or not 1 <= num_frames <= _MAX_WAN_FRAMES:
        raise ValueError(f"Wan num_frames must be a canonical integer from 1 through {_MAX_WAN_FRAMES}.")
    if (num_frames - 1) % temporal_scale != 0:
        raise ValueError(
            f"Wan num_frames must satisfy (num_frames - 1) % {temporal_scale} == 0."
        )
    if workflow == _WAN_FLF_WORKFLOW and num_frames < temporal_scale + 1:
        raise ValueError("Wan FLF requires at least 5 frames; one-frame FLF is invalid upstream.")


def _validate_wan_video_tensor(
    tensor,
    *,
    label,
    channels,
    num_frames,
    height,
    width,
    spatial_scale,
    temporal_scale,
):
    if type(tensor) is not torch.Tensor:
        raise TypeError(f"{label} must be an exact Torch tensor.")
    if tensor.layout != torch.strided or tensor.device.type == "meta" or tensor.ndim != 5:
        raise ValueError(f"{label} must be a resident strided rank-5 tensor.")
    expected_shape = (
        1,
        channels,
        (num_frames - 1) // temporal_scale + 1,
        height // spatial_scale,
        width // spatial_scale,
    )
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"{label} does not match the sealed Wan frame and spatial geometry.")
    if not tensor.dtype.is_floating_point:
        raise ValueError(f"{label} must use a floating-point dtype.")


def resolve_managed_component_by_id(component_manager, component_input, *, label):
    """Resolve exactly one connected managed component without initializing a block."""

    if not isinstance(component_input, Mapping):
        raise TypeError(f"{label} metadata must be a mapping.")
    model_id = component_input.get("model_id")
    if type(model_id) is not str or not model_id:
        raise ValueError(f"{label} metadata must contain one non-empty managed component ID.")
    try:
        resolved = component_manager.get_components_by_ids(
            ids=[model_id],
            return_dict_with_names=False,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} could not be resolved from its exact managed component ID.") from error
    if type(resolved) is not dict or set(resolved) != {model_id} or resolved[model_id] is None:
        raise ValueError(f"{label} could not be resolved from its exact managed component ID.")
    return resolved[model_id]


def _component_identity_reference(component, *, label):
    try:
        reference = weakref.ref(component)
    except TypeError as error:
        raise TypeError(f"{label} must support process-local weak identity provenance.") from error
    if reference() is not component:
        raise ValueError(f"{label} could not be sealed by exact process identity.")
    return reference


def _require_sdxl_vae_provenance(payload, vae_component):
    if vae_component is None:
        raise ValueError("The exact connected SDXL VAE component is required for route validation.")
    issued_vae = payload._vae_ref() if isinstance(payload._vae_ref, weakref.ReferenceType) else None
    if issued_vae is None:
        raise ValueError("The SDXL VAE paired with this Modular route state is no longer resident.")
    if issued_vae is not vae_component:
        raise ValueError("The connected SDXL VAE is not the exact component paired with this route state.")
    geometry = sdxl_vae_geometry_from_component(issued_vae)
    if geometry != (payload._vae_latent_channels, payload._vae_scale_factor):
        raise ValueError("The SDXL VAE geometry changed after this Modular route state was issued.")
    return geometry


def validate_sdxl_route_vae_provenance(route_state, *, binding, model_type, vae_component):
    """Validate a cached SDXL route against its exact loader and resident VAE."""

    if not _is_issued_route_state(route_state):
        raise ValueError("The Modular route state was not issued by this backend process.")
    if route_state._binding is not binding or binding._model_type != model_type:
        raise ValueError("The Modular route state comes from a different Models Loader execution.")
    if route_state._contract != _SDXL_ROUTE_CONTRACT or route_contract_for_model_type(model_type) != _SDXL_ROUTE_CONTRACT:
        raise ValueError("The Modular route state does not carry the SDXL VAE contract.")
    payload = route_state._payload
    geometry = _require_sdxl_vae_provenance(payload, vae_component)
    _require_paired_latents_resident(payload._paired_latents_ref, label="routed latents")
    _require_paired_latents_resident(payload._mask_ref, label="VAE latent mask")
    _require_paired_latents_resident(payload._masked_image_latents_ref, label="VAE masked-image latents")
    return geometry


def _validate_sdxl_latent_tensor(tensor, *, label, latent_channels, scale_factor):
    if type(tensor) is not torch.Tensor:
        raise TypeError(f"{label} must be an exact Torch tensor.")
    if tensor.layout != torch.strided or tensor.device.type == "meta" or tensor.ndim != 4:
        raise ValueError(f"{label} must be a resident strided rank-4 tensor.")
    batch, channels, height, width = tensor.shape
    if batch <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"{label} batch and spatial dimensions must be positive.")
    if channels != latent_channels:
        raise ValueError(f"{label} channels do not match the connected SDXL VAE latent contract.")
    if not tensor.dtype.is_floating_point:
        raise ValueError(f"{label} must use a floating-point dtype.")
    decoded_pixels = batch * height * width * scale_factor * scale_factor
    if decoded_pixels > _MAX_OVERLAY_AGGREGATE_PIXELS:
        raise ValueError(f"{label} exceeds the bounded SDXL decoded-pixel budget.")


def _validate_sdxl_encoder_tensors(
    image_latents,
    mask,
    masked_image_latents,
    *,
    latent_channels,
    scale_factor,
):
    _validate_sdxl_latent_tensor(
        image_latents,
        label="SDXL VAE image latents",
        latent_channels=latent_channels,
        scale_factor=scale_factor,
    )
    if mask is None and masked_image_latents is None:
        return
    _validate_sdxl_latent_tensor(
        masked_image_latents,
        label="SDXL VAE masked-image latents",
        latent_channels=latent_channels,
        scale_factor=scale_factor,
    )
    if type(mask) is not torch.Tensor:
        raise TypeError("SDXL VAE latent mask must be an exact Torch tensor.")
    if mask.layout != torch.strided or mask.device.type == "meta" or mask.ndim != 4:
        raise ValueError("SDXL VAE latent mask must be a resident strided rank-4 tensor.")
    if mask.shape[1] != 1:
        raise ValueError("SDXL VAE latent mask must have exactly one channel.")
    if mask.shape[0] <= 0 or mask.shape[2] <= 0 or mask.shape[3] <= 0:
        raise ValueError("SDXL VAE latent mask batch and spatial dimensions must be positive.")
    if (mask.shape[0], mask.shape[2], mask.shape[3]) != (
        image_latents.shape[0],
        image_latents.shape[2],
        image_latents.shape[3],
    ):
        raise ValueError("SDXL VAE latent mask batch and spatial dimensions must match image latents.")
    if (
        masked_image_latents.shape[0] != image_latents.shape[0]
        or masked_image_latents.shape[2:] != image_latents.shape[2:]
    ):
        raise ValueError("SDXL VAE masked-image latent batch and spatial dimensions must match image latents.")
    if mask.dtype != image_latents.dtype or masked_image_latents.dtype != image_latents.dtype:
        raise ValueError("SDXL VAE image, mask, and masked-image latents must use one exact dtype.")
    if mask.device != image_latents.device or masked_image_latents.device != image_latents.device:
        raise ValueError("SDXL VAE image, mask, and masked-image latents must use one exact device.")


def _paired_latent_items(latents, *, label, allow_list):
    if type(latents) is torch.Tensor:
        return "tensor", (latents,)
    if not allow_list or type(latents) is not list:
        suffix = " or a bounded nonempty list of exact Torch tensors" if allow_list else ""
        raise TypeError(f"{label} must be an exact Torch tensor{suffix}.")
    if not latents:
        raise ValueError(f"{label} latent lists must not be empty.")
    if len(latents) > _MAX_PAIRED_LATENT_TENSORS:
        raise ValueError(f"{label} latent lists cannot contain more than {_MAX_PAIRED_LATENT_TENSORS} tensors.")
    for index, item in enumerate(latents):
        if type(item) is not torch.Tensor:
            raise TypeError(f"{label}[{index}] must be an exact Torch tensor.")
    return "list", tuple(latents)


def _paired_latents_reference(latents, *, label, allow_list=False):
    kind, items = _paired_latent_items(latents, label=label, allow_list=allow_list)
    return kind, tuple((weakref.ref(item), _tensor_identity_seal(item)) for item in items)


def _tensor_identity_seal(tensor):
    try:
        version = tensor._version
    except RuntimeError as error:
        raise ValueError("Modular route tensors must expose a mutation version counter.") from error
    if tensor.layout != torch.strided or tensor.device.type == "meta":
        raise ValueError("Modular route tensors must be resident strided tensors.")
    try:
        data_ptr = tensor.data_ptr()
    except RuntimeError as error:
        raise ValueError("Modular route tensors must expose stable resident storage.") from error
    stride = tuple(tensor.stride())
    storage_offset = tensor.storage_offset()
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        stride,
        storage_offset,
        data_ptr,
        version,
    )


def _require_tensor_seal(tensor, seal, *, label):
    if _tensor_identity_seal(tensor) != seal:
        raise ValueError(f"The exact {label} paired with this Modular route state was mutated or rebound.")


def _require_paired_latents(route_state, latents, *, label):
    _require_paired_latents_reference(route_state._paired_latents_ref, latents, label=label)


def _require_paired_latents_reference(reference, latents, *, label):
    issued_kind, issued_refs = reference
    connected_kind, connected_items = _paired_latent_items(
        latents,
        label=f"Connected {label}",
        allow_list=issued_kind == "list",
    )
    if connected_kind != issued_kind or len(connected_items) != len(issued_refs):
        raise ValueError(f"Connected {label} does not match the exact latent output paired with this route state.")
    for (issued_ref, issued_seal), connected_item in zip(issued_refs, connected_items):
        issued_item = issued_ref()
        if issued_item is None:
            raise ValueError(f"The {label} paired with this Modular route state is no longer resident.")
        if issued_item is not connected_item:
            raise ValueError(f"Connected {label} does not match the exact latent output paired with this route state.")
        _require_tensor_seal(issued_item, issued_seal, label=label)


def _require_optional_paired_latents(reference, latents, *, label):
    if reference is None:
        if latents is not None:
            raise ValueError(f"Connected {label} was not part of this Modular route state.")
        return
    _require_paired_latents_reference(reference, latents, label=label)


def _require_paired_latents_resident(reference, *, label):
    """Require every exact tensor paired with a sealed route to remain alive."""

    if reference is None:
        return
    _issued_kind, issued_refs = reference
    for issued_ref, issued_seal in issued_refs:
        issued_item = issued_ref()
        if issued_item is None:
            raise ValueError(f"The {label} paired with this Modular route state is no longer resident.")
        _require_tensor_seal(issued_item, issued_seal, label=label)


def require_route_state_current_publication(route_state, *, label, controlnet_component=_NOT_PROVIDED):
    """Reject a cached route whose standalone component publication was superseded."""

    if route_state is None:
        return None
    if not _is_issued_route_state(route_state):
        raise ValueError(f"Connected Modular {label} was not issued by this backend process.")
    controlnet_binding = route_state._standalone_controlnet_binding
    if controlnet_binding is not None and not _is_current_standalone_component_binding(controlnet_binding):
        raise ValueError(
            f"Connected Modular {label} references a superseded AutoModelLoader ControlNet publication. "
            "Rerun Load Model and ControlNet, then reconnect the route."
        )
    if controlnet_binding is not None and controlnet_component is not _NOT_PROVIDED:
        require_standalone_component_binding(
            controlnet_component,
            label="ControlNet model",
            expected_kind="controlnet",
            expected_binding=controlnet_binding,
        )
    return route_state


def issue_wan_image_encoder_route_state(
    *,
    binding,
    image,
    last_image,
    height,
    width,
    image_embeds,
    image_encoder,
    image_processor,
    resized_image,
    resized_last_image,
    execution_device,
    source_snapshot,
    preflight_geometry,
):
    """Issue the first opaque Wan edge after the pinned image-encoder pass."""

    if not _is_issued_binding(binding) or binding._model_type != "WanImage2VideoModularPipeline":
        raise ValueError("Cannot issue a Wan image route for an invalid ModelsLoader binding.")
    _require_wan_media_snapshot(source_snapshot, image, last_image)
    workflow = source_snapshot[0]
    current_preflight = preflight_wan_image_encoder_inputs(
        image=image,
        last_image=last_image,
        height=height,
        width=width,
    )
    if current_preflight != preflight_geometry:
        raise ValueError("Wan image-encoder preflight geometry changed before route publication.")
    (
        _workflow,
        requested_height,
        requested_width,
        first_height,
        first_width,
        _clip_height,
        _clip_width,
        last_intermediate,
    ) = current_preflight
    _validate_wan_resized_media(
        workflow=workflow,
        resized_image=resized_image,
        resized_last_image=resized_last_image,
        height=first_height,
        width=first_width,
        expected_last_size=last_intermediate,
        stage="image encoding",
    )
    _validate_wan_image_embeds(image_embeds, workflow=workflow)
    image_encoder_execution_device = _require_wan_tensor_device(
        image_embeds,
        execution_device,
        label="Wan image embeddings",
    )
    image_encoder_config_seal = wan_image_encoder_contract_from_component(image_encoder)
    _validate_wan_image_embed_dimension(image_embeds, image_dim=image_encoder_config_seal[1])
    image_encoder_ref = _component_identity_reference(image_encoder, label="Connected Wan image encoder")
    image_processor_ref = _component_identity_reference(image_processor, label="Connected Wan image processor")
    return _new_route_state(
        stage=_IMAGE_EMBED_TO_VAE,
        binding=binding,
        seed=None,
        contract=_WAN_ROUTE_CONTRACT,
        payload=_WanRoutePayload(
            generator_snapshot=None,
            paired_latents_ref=_paired_latents_reference(image_embeds, label="Wan image embeddings"),
            image_embeds_ref=_paired_latents_reference(image_embeds, label="Wan image embeddings"),
            source_image_ref=source_snapshot[1],
            source_image_seal=source_snapshot[2],
            last_image_ref=source_snapshot[3],
            last_image_seal=source_snapshot[4],
            workflow=workflow,
            requested_height=requested_height,
            requested_width=requested_width,
            first_height=first_height,
            first_width=first_width,
            image_encoder_ref=image_encoder_ref,
            image_encoder_config_seal=image_encoder_config_seal,
            image_processor_ref=image_processor_ref,
            image_processor_config_seal=wan_image_processor_config_seal(
                image_processor,
                workflow=workflow,
            ),
            image_encoder_execution_device=image_encoder_execution_device,
        ),
    )


def validate_wan_image_encoder_route_state(
    route_state,
    *,
    binding,
    model_type,
    image,
    last_image,
    height,
    width,
    image_embeds=None,
    image_encoder=_NOT_PROVIDED,
    image_processor=_NOT_PROVIDED,
    execution_device=_NOT_PROVIDED,
):
    """Validate image-to-VAE state without trusting visible resize fields."""

    if not _is_issued_route_state(route_state) or route_state._stage != _IMAGE_EMBED_TO_VAE:
        raise ValueError("The Wan image route state is connected to the wrong action stage.")
    if route_state._binding is not binding or binding._model_type != model_type:
        raise ValueError("The Wan image route comes from a different Models Loader execution.")
    if route_state._contract != _WAN_ROUTE_CONTRACT or route_contract_for_model_type(model_type) != _WAN_ROUTE_CONTRACT:
        raise ValueError("The Wan image route belongs to a different pipeline state contract.")
    payload = route_state._payload
    preflight_geometry = preflight_wan_image_encoder_inputs(
        image=image,
        last_image=last_image,
        height=height,
        width=width,
    )
    requested = preflight_geometry[1:3]
    if requested != (payload._requested_height, payload._requested_width):
        raise ValueError("Wan requested dimensions changed after the image-embedding action.")
    _require_wan_media_snapshot(_wan_payload_media_snapshot(payload), image, last_image)
    expected_first = preflight_geometry[3:5]
    if expected_first != (payload._first_height, payload._first_width):
        raise ValueError("Wan first-pass image dimensions no longer match the sealed route.")
    if image_embeds is None:
        _require_paired_latents_resident(payload._image_embeds_ref, label="Wan image embeddings")
    else:
        _validate_wan_image_embeds(image_embeds, workflow=payload._workflow)
        _require_paired_latents_reference(payload._image_embeds_ref, image_embeds, label="Wan image embeddings")
    sealed_execution_device = _wan_execution_device(
        payload._image_encoder_execution_device,
        label="image-encoder producer",
    )
    if execution_device is not _NOT_PROVIDED and not _devices_compatible(
        sealed_execution_device,
        _wan_execution_device(execution_device, label="image-encoder producer"),
    ):
        raise ValueError("Wan Image Embeddings execution device changed after route publication.")
    _require_wan_reference_device(
        payload._image_embeds_ref,
        sealed_execution_device,
        label="Wan image embeddings",
    )
    image_encoder = _require_component_reference(
        payload._image_encoder_ref,
        None if image_encoder is _NOT_PROVIDED else image_encoder,
        label="Wan image encoder",
        require_connected=image_encoder is not _NOT_PROVIDED,
    )
    image_encoder_config_seal = wan_image_encoder_contract_from_component(image_encoder)
    if image_encoder_config_seal != payload._image_encoder_config_seal:
        raise ValueError("The Wan image encoder contract changed after route publication.")
    if image_embeds is not None:
        _validate_wan_image_embed_dimension(image_embeds, image_dim=image_encoder_config_seal[1])
    processor = _require_component_reference(
        payload._image_processor_ref,
        None if image_processor is _NOT_PROVIDED else image_processor,
        label="Wan image processor",
        require_connected=image_processor is not _NOT_PROVIDED,
    )
    if (
        wan_image_processor_config_seal(processor, workflow=payload._workflow)
        != payload._image_processor_config_seal
    ):
        raise ValueError("The Wan image processor configuration changed after route publication.")
    return route_state


def consume_wan_image_encoder_route_state(route_state, **kwargs):
    """Return only the backend-owned first-pass dimensions for Wan VAE."""

    validate_wan_image_encoder_route_state(route_state, **kwargs)
    payload = route_state._payload
    return {
        "workflow": payload._workflow,
        "height": payload._first_height,
        "width": payload._first_width,
    }


def preflight_wan_vae_route_state(
    route_state,
    *,
    binding,
    model_type,
    image,
    last_image,
    height,
    width,
    num_frames,
    vae_component,
):
    """Validate Wan VAE geometry/resource bounds before pipeline init."""

    validate_wan_image_encoder_route_state(
        route_state,
        binding=binding,
        model_type=model_type,
        image=image,
        last_image=last_image,
        height=height,
        width=width,
    )
    payload = route_state._payload
    z_dim, spatial_scale, temporal_scale = wan_vae_geometry_from_component(vae_component)
    _validate_wan_frames(num_frames, workflow=payload._workflow, temporal_scale=temporal_scale)
    second_height, second_width = wan_area_budget_dimensions(
        image,
        payload._first_height,
        payload._first_width,
    )
    second_last_intermediate = _wan_last_image_intermediate_dimensions(
        last_image,
        height=second_height,
        width=second_width,
        stage="VAE",
    )
    video_tensor_bytes = second_height * second_width * num_frames * _WAN_VIDEO_PIXEL_BYTES
    if video_tensor_bytes > _MAX_WAN_VIDEO_TENSOR_BYTES:
        raise ValueError("Wan resolved dimensions and frame count exceed the bounded 512-MiB video-tensor budget.")
    return (
        payload._workflow,
        payload._first_height,
        payload._first_width,
        second_height,
        second_width,
        num_frames,
        z_dim,
        spatial_scale,
        temporal_scale,
        _wan_vae_config_seal(vae_component),
        second_last_intermediate,
        video_tensor_bytes,
    )


def issue_wan_vae_route_state(
    route_state,
    *,
    binding,
    seed,
    generator,
    image,
    last_image,
    height,
    width,
    num_frames,
    image_condition_latents,
    raw_frame_latents,
    vae_component,
    video_processor,
    resized_image,
    resized_last_image,
    execution_device,
    preflight_geometry,
):
    """Advance the Wan route after the second resize and deterministic VAE."""

    validate_wan_image_encoder_route_state(
        route_state,
        binding=binding,
        model_type=binding._model_type,
        image=image,
        last_image=last_image,
        height=height,
        width=width,
    )
    if type(seed) is not int or not 0 <= seed <= 4294967295:
        raise ValueError("A Wan route seed must be a canonical integer from 0 through 4294967295.")
    if not isinstance(generator, torch.Generator) or generator.initial_seed() != seed:
        raise ValueError("The post-VAE Wan generator does not match the validated route seed.")
    input_payload = route_state._payload
    current_preflight = preflight_wan_vae_route_state(
        route_state,
        binding=binding,
        model_type=binding._model_type,
        image=image,
        last_image=last_image,
        height=height,
        width=width,
        num_frames=num_frames,
        vae_component=vae_component,
    )
    if current_preflight != preflight_geometry:
        raise ValueError("Wan VAE preflight geometry changed before route publication.")
    (
        _workflow,
        _first_height,
        _first_width,
        second_height,
        second_width,
        _preflight_num_frames,
        z_dim,
        spatial_scale,
        temporal_scale,
        vae_config_seal,
        second_last_intermediate,
        _video_tensor_bytes,
    ) = current_preflight
    video_processor_config_seal = wan_video_processor_config_seal(video_processor)
    _validate_wan_resized_media(
        workflow=input_payload._workflow,
        resized_image=resized_image,
        resized_last_image=resized_last_image,
        height=second_height,
        width=second_width,
        expected_last_size=second_last_intermediate,
        stage="VAE encoding",
    )
    _validate_wan_video_tensor(
        raw_frame_latents,
        label="Wan raw frame latents",
        channels=z_dim,
        num_frames=num_frames,
        height=second_height,
        width=second_width,
        spatial_scale=spatial_scale,
        temporal_scale=temporal_scale,
    )
    _validate_wan_video_tensor(
        image_condition_latents,
        label="Wan image condition latents",
        channels=z_dim + temporal_scale,
        num_frames=num_frames,
        height=second_height,
        width=second_width,
        spatial_scale=spatial_scale,
        temporal_scale=temporal_scale,
    )
    vae_execution_device = _require_wan_tensor_device(
        raw_frame_latents,
        execution_device,
        label="Wan raw frame latents",
    )
    _require_wan_tensor_device(
        image_condition_latents,
        vae_execution_device,
        label="Wan image condition latents",
    )
    if not _devices_compatible(generator.device, vae_execution_device):
        raise ValueError("The post-VAE Wan generator must remain on the producing execution device.")
    _require_wan_media_snapshot(_wan_payload_media_snapshot(input_payload), image, last_image)
    condition_ref = _paired_latents_reference(
        image_condition_latents,
        label="Wan image condition latents",
    )
    return _new_route_state(
        stage=_ENCODE_TO_DENOISE,
        binding=binding,
        seed=seed,
        contract=_WAN_ROUTE_CONTRACT,
        payload=_WanRoutePayload(
            generator_snapshot=_clone_generator(generator),
            paired_latents_ref=condition_ref,
            image_embeds_ref=input_payload._image_embeds_ref,
            image_condition_latents_ref=condition_ref,
            source_image_ref=input_payload._source_image_ref,
            source_image_seal=input_payload._source_image_seal,
            last_image_ref=input_payload._last_image_ref,
            last_image_seal=input_payload._last_image_seal,
            workflow=input_payload._workflow,
            requested_height=input_payload._requested_height,
            requested_width=input_payload._requested_width,
            first_height=input_payload._first_height,
            first_width=input_payload._first_width,
            second_height=second_height,
            second_width=second_width,
            num_frames=num_frames,
            image_encoder_ref=input_payload._image_encoder_ref,
            image_encoder_config_seal=input_payload._image_encoder_config_seal,
            image_processor_ref=input_payload._image_processor_ref,
            image_processor_config_seal=input_payload._image_processor_config_seal,
            image_encoder_execution_device=input_payload._image_encoder_execution_device,
            vae_ref=_component_identity_reference(vae_component, label="Connected Wan VAE"),
            video_processor_ref=_component_identity_reference(
                video_processor,
                label="Connected Wan video processor",
            ),
            video_processor_config_seal=video_processor_config_seal,
            vae_config_seal=vae_config_seal,
            vae_execution_device=vae_execution_device,
        ),
    )


def _require_wan_vae_provenance(payload, vae_component, video_processor=_NOT_PROVIDED):
    vae = _require_component_reference(payload._vae_ref, vae_component, label="Wan VAE")
    processor = _require_component_reference(
        payload._video_processor_ref,
        None if video_processor is _NOT_PROVIDED else video_processor,
        label="Wan VAE video processor",
        require_connected=video_processor is not _NOT_PROVIDED,
    )
    if wan_video_processor_config_seal(processor) != payload._video_processor_config_seal:
        raise ValueError("The Wan VAE video processor configuration changed after route publication.")
    if _wan_vae_config_seal(vae) != payload._vae_config_seal:
        raise ValueError("The Wan VAE geometry changed after this Modular route state was issued.")
    return wan_vae_geometry_from_component(vae)


def validate_wan_vae_route_state(
    route_state,
    *,
    binding,
    model_type,
    seed,
    image_embeds,
    image_condition_latents,
    height,
    width,
    num_frames,
    vae_component,
    video_processor=_NOT_PROVIDED,
    transformer_component=_NOT_PROVIDED,
    producer_execution_device=_NOT_PROVIDED,
):
    """Validate the exact Wan VAE-to-Denoise route and typed tensor edges."""

    if not _is_issued_route_state(route_state) or route_state._stage != _ENCODE_TO_DENOISE:
        raise ValueError("The Wan VAE route state is connected to the wrong action stage.")
    if route_state._binding is not binding or binding._model_type != model_type:
        raise ValueError("The Wan VAE route comes from a different Models Loader execution.")
    if route_state._contract != _WAN_ROUTE_CONTRACT or route_contract_for_model_type(model_type) != _WAN_ROUTE_CONTRACT:
        raise ValueError("The Wan VAE route belongs to a different pipeline state contract.")
    payload = route_state._payload
    if type(seed) is not int or seed != route_state._seed:
        raise ValueError("Denoise seed must match the seed used by the preceding Wan VAE route.")
    generator = payload._generator_snapshot
    if not isinstance(generator, torch.Generator) or generator.initial_seed() != seed:
        raise ValueError("The Wan route generator snapshot no longer matches its originating seed.")
    requested = _validate_wan_requested_dimensions(height, width)
    if requested != (payload._requested_height, payload._requested_width):
        raise ValueError("Wan requested dimensions changed after VAE encoding.")
    _validate_wan_frames(num_frames, workflow=payload._workflow)
    if num_frames != payload._num_frames:
        raise ValueError("Wan num_frames changed after VAE encoding.")
    _require_wan_media_snapshot(
        _wan_payload_media_snapshot(payload),
        payload._source_image_ref(),
        payload._last_image_ref() if payload._last_image_ref is not None else None,
    )
    if image_embeds is _NOT_PROVIDED:
        _require_paired_latents_resident(payload._image_embeds_ref, label="Wan image embeddings")
    else:
        _validate_wan_image_embeds(image_embeds, workflow=payload._workflow)
        _require_paired_latents_reference(payload._image_embeds_ref, image_embeds, label="Wan image embeddings")
    z_dim, spatial_scale, temporal_scale = _require_wan_vae_provenance(
        payload,
        vae_component,
        video_processor,
    )
    _validate_wan_video_tensor(
        image_condition_latents,
        label="Wan image condition latents",
        channels=z_dim + temporal_scale,
        num_frames=num_frames,
        height=payload._second_height,
        width=payload._second_width,
        spatial_scale=spatial_scale,
        temporal_scale=temporal_scale,
    )
    _require_paired_latents_reference(
        payload._image_condition_latents_ref,
        image_condition_latents,
        label="Wan image condition latents",
    )
    _require_wan_reference_device(
        payload._image_embeds_ref,
        payload._image_encoder_execution_device,
        label="Wan image embeddings",
    )
    sealed_vae_execution_device = _require_wan_reference_device(
        payload._image_condition_latents_ref,
        payload._vae_execution_device,
        label="Wan image condition latents",
    )
    if producer_execution_device is not _NOT_PROVIDED and not _devices_compatible(
        sealed_vae_execution_device,
        _wan_execution_device(producer_execution_device, label="VAE producer"),
    ):
        raise ValueError("Wan Image Encode execution device changed after route publication.")
    image_encoder = _require_component_reference(
        payload._image_encoder_ref,
        None,
        label="Wan image encoder",
        require_connected=False,
    )
    if wan_image_encoder_contract_from_component(image_encoder) != payload._image_encoder_config_seal:
        raise ValueError("The Wan image encoder contract changed after route publication.")
    image_processor = _require_component_reference(
        payload._image_processor_ref,
        None,
        label="Wan image processor",
        require_connected=False,
    )
    if (
        wan_image_processor_config_seal(image_processor, workflow=payload._workflow)
        != payload._image_processor_config_seal
    ):
        raise ValueError("The Wan image processor configuration changed after route publication.")
    if transformer_component is not _NOT_PROVIDED:
        transformer_seal = wan_transformer_contract_from_component(
            transformer_component,
            workflow=payload._workflow,
        )
        if image_embeds is _NOT_PROVIDED:
            raise ValueError("Wan transformer validation requires the exact connected image embeddings.")
        _validate_wan_image_embed_dimension(image_embeds, image_dim=transformer_seal[3])
        if payload._transformer_config_seal is not None and transformer_seal != payload._transformer_config_seal:
            raise ValueError("The Wan transformer contract changed after Denoise route publication.")
        if payload._transformer_ref is not None:
            _require_component_reference(payload._transformer_ref, transformer_component, label="Wan transformer")
    return route_state


def consume_wan_vae_route_state(route_state, *, execution_device, **kwargs):
    """Materialize retry-safe Wan generator state plus routed dimensions."""

    validate_wan_vae_route_state(route_state, **kwargs)
    payload = route_state._payload
    if not _devices_compatible(payload._generator_snapshot.device, execution_device):
        raise ValueError("The Wan route generator device is incompatible with Denoise execution.")
    for label, tensor in (
        ("image embeddings", kwargs.get("image_embeds")),
        ("image condition latents", kwargs.get("image_condition_latents")),
    ):
        if type(tensor) is not torch.Tensor or not _devices_compatible(tensor.device, execution_device):
            raise ValueError(f"Wan {label} must be resident on the Denoise execution device.")
    return {
        "generator": _clone_generator(payload._generator_snapshot),
        "height": payload._second_height,
        "width": payload._second_width,
        "num_frames": payload._num_frames,
        "processed_mask_image": None,
    }


def validate_wan_post_vae_route_state(route_state, **kwargs):
    """Validate a cached Wan VAE result before a transformer is in scope."""

    return validate_wan_vae_route_state(
        route_state,
        image_embeds=_NOT_PROVIDED,
        transformer_component=_NOT_PROVIDED,
        **kwargs,
    )


def issue_encoder_route_state(
    *,
    binding,
    seed,
    generator,
    image_latents,
    processed_mask_image=None,
    mask_overlay_kwargs=None,
    mask=None,
    masked_image_latents=None,
    padding_mask_crop=None,
    crops_coords=None,
    original_image=None,
    original_mask=None,
    vae_component=None,
    vae_latent_channels=None,
    vae_scale_factor=None,
):
    """Seal post-VAE generator state and mask routing values for Denoise."""

    if not _is_issued_binding(binding):
        raise ValueError("Cannot issue route state for an invalid ModelsLoader binding.")
    if binding._model_type not in SUPPORTED_ROUTE_MODEL_TYPES:
        raise ValueError(f"Pipeline '{binding._model_type}' does not declare opaque route-state support.")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise ValueError("A Modular route seed must be a canonical integer from 0 through 4294967295.")
    if not isinstance(generator, torch.Generator) or generator.initial_seed() != seed:
        raise ValueError("The post-VAE generator does not match the validated Modular route seed.")
    contract = route_contract_for_model_type(binding._model_type)
    if contract == _SDXL_ROUTE_CONTRACT:
        if processed_mask_image is not None or mask_overlay_kwargs is not None:
            raise ValueError("SDXL route state cannot contain Qwen processed-mask or overlay fields.")
        inpaint = mask is not None or masked_image_latents is not None
        if (mask is None) != (masked_image_latents is None):
            raise ValueError(
                "The SDXL VAE route must provide both mask and masked-image latents for inpainting, or neither."
            )
        if (
            type(vae_latent_channels) is not int
            or not 1 <= vae_latent_channels <= 64
            or type(vae_scale_factor) is not int
            or not 1 <= vae_scale_factor <= 32768
        ):
            raise ValueError("SDXL route state requires the exact connected VAE latent geometry.")
        if sdxl_vae_geometry_from_component(vae_component) != (
            vae_latent_channels,
            vae_scale_factor,
        ):
            raise ValueError("SDXL route state geometry does not match the exact connected VAE component.")
        _validate_sdxl_encoder_tensors(
            image_latents,
            mask,
            masked_image_latents,
            latent_channels=vae_latent_channels,
            scale_factor=vae_scale_factor,
        )
        if inpaint:
            mask_ref = _paired_latents_reference(mask, label="VAE latent mask")
            masked_image_latents_ref = _paired_latents_reference(
                masked_image_latents,
                label="VAE masked-image latents",
            )
        else:
            mask_ref = None
            masked_image_latents_ref = None
        if padding_mask_crop is not None and (
            type(padding_mask_crop) is not int
            or padding_mask_crop < 0
            or padding_mask_crop > _MAX_MASK_CROP_PADDING
        ):
            raise ValueError(
                f"SDXL mask crop padding must be a canonical integer from 0 through {_MAX_MASK_CROP_PADDING}, or null."
            )
        if padding_mask_crop is None:
            if crops_coords is not None:
                raise ValueError("SDXL crop coordinates require non-null mask crop padding.")
            original_image_snapshot = None
            original_mask_snapshot = None
        else:
            if not inpaint:
                raise ValueError("SDXL mask crop padding requires an inpaint mask and masked-image latents.")
            if (
                type(crops_coords) is not tuple
                or len(crops_coords) != 4
                or any(type(item) is not int for item in crops_coords)
            ):
                raise ValueError("SDXL crop coordinates must be an exact tuple of four canonical integers.")
            if original_image is None or original_mask is None:
                raise ValueError("SDXL crop overlay requires the original image and mask.")
            original_image_snapshot, original_mask_snapshot = _freeze_sdxl_overlay_media(
                original_image,
                original_mask,
                crops_coords=crops_coords,
            )
        return _new_route_state(
            stage=_ENCODE_TO_DENOISE,
            binding=binding,
            seed=seed,
            contract=contract,
            payload=_SdxlRoutePayload(
                generator_snapshot=_clone_generator(generator),
                inpaint=inpaint,
                paired_latents_ref=_paired_latents_reference(image_latents, label="VAE image latents"),
                mask_ref=mask_ref,
                masked_image_latents_ref=masked_image_latents_ref,
                padding_mask_crop=padding_mask_crop,
                crops_coords=crops_coords,
                original_image_snapshot=original_image_snapshot,
                original_mask_snapshot=original_mask_snapshot,
                vae_ref=_component_identity_reference(vae_component, label="Connected SDXL VAE"),
                vae_latent_channels=vae_latent_channels,
                vae_scale_factor=vae_scale_factor,
            ),
        )

    if any(
        value is not None
        for value in (
            mask,
            masked_image_latents,
            padding_mask_crop,
            crops_coords,
            original_image,
            original_mask,
            vae_component,
            vae_latent_channels,
            vae_scale_factor,
        )
    ):
        raise ValueError("Qwen route state cannot contain SDXL typed mask or crop fields.")
    if binding._model_type == "QwenImageEditPlusModularPipeline" and (
        processed_mask_image is not None or mask_overlay_kwargs is not None
    ):
        raise ValueError("Qwen Image Edit Plus supports generator-only route state; inpaint state is not enabled.")
    if mask_overlay_kwargs is not None and type(mask_overlay_kwargs) is not dict:
        raise TypeError("Modular mask overlay state must be an exact dictionary or null.")
    if (processed_mask_image is None) != (mask_overlay_kwargs is None):
        raise ValueError(
            "The Modular VAE route must provide both processed mask and overlay state for inpainting, "
            "or neither for a normal image route."
        )
    if processed_mask_image is not None:
        if type(processed_mask_image) is not torch.Tensor:
            raise TypeError("The processed Modular inpaint mask must be a Torch tensor.")
        expected_overlay_keys = {"crops_coords", "original_image", "original_mask"}
        if set(mask_overlay_kwargs) != expected_overlay_keys:
            raise ValueError("The Modular inpaint overlay state does not match the pinned Qwen contract.")
        crops_coords = mask_overlay_kwargs["crops_coords"]
        original_image = mask_overlay_kwargs["original_image"]
        original_mask = mask_overlay_kwargs["original_mask"]
        if crops_coords is None:
            if original_image is not None or original_mask is not None:
                raise ValueError("Original image and mask require non-null Modular inpaint crop coordinates.")
        else:
            if (
                type(crops_coords) is not tuple
                or len(crops_coords) != 4
                or any(isinstance(item, bool) or not isinstance(item, int) for item in crops_coords)
                or original_image is None
                or original_mask is None
            ):
                raise ValueError(
                    "Modular inpaint crop coordinates require four integers plus original image and mask."
                )
    return _new_route_state(
        stage=_ENCODE_TO_DENOISE,
        binding=binding,
        seed=seed,
        contract=contract,
        payload=_QwenRoutePayload(
            generator_snapshot=_clone_generator(generator),
            processed_mask_image=processed_mask_image,
            mask_overlay_kwargs=dict(mask_overlay_kwargs) if mask_overlay_kwargs is not None else None,
            inpaint=processed_mask_image is not None,
            paired_latents_ref=_paired_latents_reference(
                image_latents,
                label="VAE image latents",
                allow_list=binding._model_type == "QwenImageEditPlusModularPipeline",
            ),
        ),
    )


def validate_encoder_route_state(
    route_state,
    *,
    binding,
    model_type,
    seed,
    image_latents,
    mask=None,
    masked_image_latents=None,
    vae_component=None,
    vae_latent_channels=None,
    vae_scale_factor=None,
):
    """Reject wrong-stage, cross-loader, or seed-mismatched state before pipeline init."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise ValueError("A Modular route seed must be a canonical integer from 0 through 4294967295.")
    if not _is_issued_route_state(route_state):
        raise ValueError("The Modular route state was not issued by this backend process.")
    if route_state._stage != _ENCODE_TO_DENOISE:
        raise ValueError("The Modular route state is connected to the wrong action stage.")
    if route_state._binding is not binding:
        raise ValueError("The Modular route state comes from a different Models Loader execution.")
    if binding._model_type != model_type:
        raise ValueError("The Modular route state belongs to a different pipeline class.")
    if route_state._contract != route_contract_for_model_type(model_type):
        raise ValueError("The Modular route state belongs to a different pipeline state contract.")
    if seed != route_state._seed:
        raise ValueError("Denoise seed must match the seed used by the preceding VAE encoder route.")
    generator = route_state._generator_snapshot
    if not isinstance(generator, torch.Generator) or generator.initial_seed() != route_state._seed:
        raise ValueError("The Modular route generator snapshot no longer matches its originating seed.")
    _require_paired_latents(route_state, image_latents, label="VAE image latents")
    if route_state._contract == _SDXL_ROUTE_CONTRACT:
        payload = route_state._payload
        _require_sdxl_vae_provenance(payload, vae_component)
        _require_optional_paired_latents(route_state._payload._mask_ref, mask, label="VAE latent mask")
        _require_optional_paired_latents(
            route_state._payload._masked_image_latents_ref,
            masked_image_latents,
            label="VAE masked-image latents",
        )
        _validate_sdxl_encoder_tensors(
            image_latents,
            mask,
            masked_image_latents,
            latent_channels=payload._vae_latent_channels,
            scale_factor=payload._vae_scale_factor,
        )
        if vae_latent_channels is not None or vae_scale_factor is not None:
            if (vae_latent_channels, vae_scale_factor) != (
                payload._vae_latent_channels,
                payload._vae_scale_factor,
            ):
                raise ValueError("The connected Denoise VAE geometry does not match the originating VAE route.")


def consume_encoder_route_state(
    route_state,
    *,
    binding,
    model_type,
    seed,
    execution_device,
    image_latents,
    mask=None,
    masked_image_latents=None,
    vae_component=None,
    vae_latent_channels=None,
    vae_scale_factor=None,
):
    """Materialize a fresh post-VAE generator after pre-init route validation."""

    validate_encoder_route_state(
        route_state,
        binding=binding,
        model_type=model_type,
        seed=seed,
        image_latents=image_latents,
        mask=mask,
        masked_image_latents=masked_image_latents,
        vae_component=vae_component,
        vae_latent_channels=vae_latent_channels,
        vae_scale_factor=vae_scale_factor,
    )
    generator = route_state._generator_snapshot
    if not _devices_compatible(generator.device, execution_device):
        raise ValueError(
            "The Modular route generator device is incompatible with the Denoise execution device; "
            "rerun the connected Models Loader and VAE encoder on one execution path."
        )
    values = {
        "generator": _clone_generator(generator),
        "processed_mask_image": route_state._processed_mask_image,
    }
    if route_state._contract == _SDXL_ROUTE_CONTRACT:
        values.update(
            mask=mask,
            masked_image_latents=masked_image_latents,
            crops_coords=route_state._payload._crops_coords,
        )
    return values


def issue_controlnet_route_state(
    route_state,
    *,
    binding,
    controlnet_component,
    seed,
    generator,
    control_image_latents,
):
    """Seal the exact post-ControlNet VAE generator and latent pairing for Denoise."""

    if not _is_issued_binding(binding) or binding._model_type != "QwenImageModularPipeline":
        raise ValueError("Qwen ControlNet route state requires a matching ModelsLoader pipeline binding.")
    controlnet_binding = require_standalone_component_binding(
        controlnet_component,
        label="ControlNet model",
        expected_kind="controlnet",
    )
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise ValueError("A Modular ControlNet route seed must be a canonical integer from 0 through 4294967295.")
    if not isinstance(generator, torch.Generator) or generator.initial_seed() != seed:
        raise ValueError("The post-ControlNet generator does not match the validated Modular route seed.")

    if route_state is None:
        paired_latents_ref = None
        processed_mask_image = None
        mask_overlay_kwargs = None
        inpaint = False
    else:
        validate_controlnet_input_route_state(
            route_state,
            binding=binding,
            model_type=binding._model_type,
            seed=seed,
        )
        paired_latents_ref = route_state._paired_latents_ref
        processed_mask_image = route_state._processed_mask_image
        mask_overlay_kwargs = route_state._mask_overlay_kwargs
        inpaint = route_state._inpaint

    return _new_route_state(
        stage=_CONTROLNET_TO_DENOISE,
        binding=binding,
        seed=seed,
        contract=_QWEN_ROUTE_CONTRACT,
        payload=_QwenRoutePayload(
            generator_snapshot=_clone_generator(generator),
            processed_mask_image=processed_mask_image,
            mask_overlay_kwargs=dict(mask_overlay_kwargs) if mask_overlay_kwargs is not None else None,
            inpaint=inpaint,
            paired_latents_ref=paired_latents_ref,
            paired_control_latents_ref=_paired_latents_reference(
                control_image_latents,
                label="ControlNet image latents",
                allow_list=True,
            ),
            standalone_controlnet_binding=controlnet_binding,
        ),
    )


def validate_controlnet_input_route_state(route_state, *, binding, model_type, seed):
    """Validate an optional ImageEncode route without receiving its large latent edge.

    Image latents remain directly connected from ImageEncode to Denoise.  This
    stage authenticates the route provenance and keeps its exact weak pairing
    alive so Denoise can compare the typed edge after ControlNet has run.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise ValueError("A Modular route seed must be a canonical integer from 0 through 4294967295.")
    if not _is_issued_route_state(route_state):
        raise ValueError("The Modular route state was not issued by this backend process.")
    if route_state._stage != _ENCODE_TO_DENOISE:
        raise ValueError("The Modular route state is connected to the wrong action stage.")
    if route_state._binding is not binding:
        raise ValueError("The Modular route state comes from a different Models Loader execution.")
    if binding._model_type != model_type:
        raise ValueError("The Modular route state belongs to a different pipeline class.")
    if seed != route_state._seed:
        raise ValueError("ControlNet seed must match the seed used by the preceding VAE encoder route.")
    generator = route_state._generator_snapshot
    if not isinstance(generator, torch.Generator) or generator.initial_seed() != route_state._seed:
        raise ValueError("The Modular route generator snapshot no longer matches its originating seed.")
    _require_paired_latents_resident(route_state._paired_latents_ref, label="VAE image latents")


def consume_controlnet_input_route_state(
    route_state,
    *,
    binding,
    model_type,
    seed,
    execution_device,
):
    """Materialize a retry-safe post-ImageEncode generator for ControlNet."""

    validate_controlnet_input_route_state(
        route_state,
        binding=binding,
        model_type=model_type,
        seed=seed,
    )
    generator = route_state._generator_snapshot
    if not _devices_compatible(generator.device, execution_device):
        raise ValueError(
            "The Modular route generator device is incompatible with the ControlNet execution device; "
            "rerun the connected model actions on one execution path."
        )
    return _clone_generator(generator)


def validate_controlnet_route_state(
    route_state,
    *,
    binding,
    model_type,
    seed,
    image_latents,
    control_image_latents,
    controlnet_component,
):
    """Validate the ControlNet-to-Denoise stage before pipeline initialization."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 4294967295:
        raise ValueError("A Modular route seed must be a canonical integer from 0 through 4294967295.")
    if not _is_issued_route_state(route_state):
        raise ValueError("The Modular route state was not issued by this backend process.")
    if route_state._stage != _CONTROLNET_TO_DENOISE:
        raise ValueError("The Modular route state is connected to the wrong action stage.")
    if route_state._binding is not binding:
        raise ValueError("The Modular route state comes from a different Models Loader execution.")
    if binding._model_type != model_type:
        raise ValueError("The Modular route state belongs to a different pipeline class.")
    if seed != route_state._seed:
        raise ValueError("Denoise seed must match the seed used by the preceding ControlNet route.")
    generator = route_state._generator_snapshot
    if not isinstance(generator, torch.Generator) or generator.initial_seed() != route_state._seed:
        raise ValueError("The Modular route generator snapshot no longer matches its originating seed.")
    require_route_state_current_publication(route_state, label="ControlNet route state")
    require_standalone_component_binding(
        controlnet_component,
        label="ControlNet model",
        expected_kind="controlnet",
        expected_binding=route_state._standalone_controlnet_binding,
    )
    _require_optional_paired_latents(
        route_state._paired_latents_ref,
        image_latents,
        label="VAE image latents",
    )
    _require_paired_latents_reference(
        route_state._paired_control_latents_ref,
        control_image_latents,
        label="ControlNet image latents",
    )


def validate_denoise_route_state(
    route_state,
    *,
    binding,
    model_type,
    seed,
    image_latents,
    mask=None,
    masked_image_latents=None,
    vae_component=None,
    vae_latent_channels=None,
    vae_scale_factor=None,
    control_image_latents=None,
    controlnet_component=None,
    control_mode=None,
    controlnet_bundle_present=False,
    ip_adapter_present=False,
    image_embeds=None,
    image_condition_latents=None,
    height=None,
    width=None,
    num_frames=None,
    transformer_component=None,
):
    """Validate either the direct VAE or ControlNet route accepted by Denoise."""

    if not _is_issued_route_state(route_state):
        raise ValueError("The Modular route state was not issued by this backend process.")
    if route_state._contract == _WAN_ROUTE_CONTRACT:
        if any(
            value is not None
            for value in (
                image_latents,
                mask,
                masked_image_latents,
                control_image_latents,
                controlnet_component,
            )
        ) or controlnet_bundle_present or ip_adapter_present:
            raise ValueError("Combined Wan I2V route and image/inpaint/control adapter state is not enabled.")
        return validate_wan_vae_route_state(
            route_state,
            binding=binding,
            model_type=model_type,
            seed=seed,
            image_embeds=image_embeds,
            image_condition_latents=image_condition_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            vae_component=vae_component,
            transformer_component=transformer_component,
        )
    if route_state._contract == _SDXL_ROUTE_CONTRACT:
        if route_state._stage != _ENCODE_TO_DENOISE:
            raise ValueError("The SDXL route state is connected to the wrong action stage.")
        if ip_adapter_present:
            raise ValueError("Combined SDXL VAE-route and IP-Adapter execution is not enabled.")
        if control_image_latents is not None:
            raise ValueError("SDXL ControlNet does not accept prepared Qwen ControlNet latents.")
        if controlnet_bundle_present != (controlnet_component is not None):
            raise ValueError("SDXL ControlNet requires one exact connected component bundle.")
        if controlnet_component is not None:
            if control_mode is not None and (
                type(control_mode) is not int or not 0 <= control_mode < SDXL_UNION_CONTROL_MODE_LIMIT
            ):
                raise ValueError("SDXL ControlNet Union mode must be one bounded canonical integer.")
            require_sdxl_controlnet_component_binding(
                controlnet_component,
                union=control_mode is not None,
            )
        elif control_mode is not None:
            raise ValueError("SDXL ControlNet Union mode requires one exact connected component bundle.")
        return validate_encoder_route_state(
            route_state,
            binding=binding,
            model_type=model_type,
            seed=seed,
            image_latents=image_latents,
            mask=mask,
            masked_image_latents=masked_image_latents,
            vae_component=vae_component,
            vae_latent_channels=vae_latent_channels,
            vae_scale_factor=vae_scale_factor,
        )
    if route_state._stage == _CONTROLNET_TO_DENOISE:
        return validate_controlnet_route_state(
            route_state,
            binding=binding,
            model_type=model_type,
            seed=seed,
            image_latents=image_latents,
            control_image_latents=control_image_latents,
            controlnet_component=controlnet_component,
        )
    if controlnet_bundle_present or control_image_latents is not None or controlnet_component is not None:
        raise ValueError("A Qwen ControlNet bundle requires its matching ControlNet route state.")
    return validate_encoder_route_state(
        route_state,
        binding=binding,
        model_type=model_type,
        seed=seed,
        image_latents=image_latents,
        mask=mask,
        masked_image_latents=masked_image_latents,
        vae_component=vae_component,
        vae_latent_channels=vae_latent_channels,
        vae_scale_factor=vae_scale_factor,
    )


def consume_denoise_route_state(
    route_state,
    *,
    binding,
    model_type,
    seed,
    execution_device,
    image_latents,
    mask=None,
    masked_image_latents=None,
    vae_component=None,
    vae_latent_channels=None,
    vae_scale_factor=None,
    control_image_latents=None,
    controlnet_component=None,
    control_mode=None,
    controlnet_bundle_present=False,
    ip_adapter_present=False,
    image_embeds=None,
    image_condition_latents=None,
    height=None,
    width=None,
    num_frames=None,
    transformer_component=None,
):
    """Materialize a fresh generator from either accepted pre-Denoise route stage."""

    if _is_issued_route_state(route_state) and route_state._contract == _WAN_ROUTE_CONTRACT:
        return consume_wan_vae_route_state(
            route_state,
            binding=binding,
            model_type=model_type,
            seed=seed,
            execution_device=execution_device,
            image_embeds=image_embeds,
            image_condition_latents=image_condition_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            vae_component=vae_component,
            transformer_component=transformer_component,
        )

    validate_denoise_route_state(
        route_state,
        binding=binding,
        model_type=model_type,
        seed=seed,
        image_latents=image_latents,
        mask=mask,
        masked_image_latents=masked_image_latents,
        vae_component=vae_component,
        vae_latent_channels=vae_latent_channels,
        vae_scale_factor=vae_scale_factor,
        control_image_latents=control_image_latents,
        controlnet_component=controlnet_component,
        control_mode=control_mode,
        controlnet_bundle_present=controlnet_bundle_present,
        ip_adapter_present=ip_adapter_present,
    )
    generator = route_state._generator_snapshot
    if not _devices_compatible(generator.device, execution_device):
        raise ValueError(
            "The Modular route generator device is incompatible with the Denoise execution device; "
            "rerun the connected model actions on one execution path."
        )
    values = {
        "generator": _clone_generator(generator),
        "processed_mask_image": route_state._processed_mask_image,
    }
    if route_state._contract == _SDXL_ROUTE_CONTRACT:
        values.update(
            mask=mask,
            masked_image_latents=masked_image_latents,
            crops_coords=route_state._payload._crops_coords,
        )
    return values


def issue_decode_route_state(
    route_state,
    *,
    binding,
    actual_mask=None,
    latents,
    vae_component=None,
    transformer_component=None,
    execution_device=None,
):
    """Advance one validated encoder route using the mask returned by Denoise."""

    if not _is_issued_route_state(route_state) or route_state._stage not in {
        _ENCODE_TO_DENOISE,
        _CONTROLNET_TO_DENOISE,
    }:
        raise ValueError("Only a valid pre-Denoise route can advance to Decode.")
    if route_state._binding is not binding:
        raise ValueError("The Modular route state comes from a different Models Loader execution.")
    require_route_state_current_publication(route_state, label="Denoise input route")
    if route_state._contract == _QWEN_ROUTE_CONTRACT:
        if actual_mask is not None and type(actual_mask) is not torch.Tensor:
            raise TypeError("The Denoise inpaint mask must be a Torch tensor.")
        actual_inpaint = actual_mask is not None
        if actual_inpaint != route_state._inpaint:
            raise ValueError("The Denoise mask result does not match the VAE encoder route.")
        payload = _QwenRoutePayload(
            generator_snapshot=None,
            processed_mask_image=None,
            mask_overlay_kwargs=route_state._mask_overlay_kwargs,
            inpaint=actual_inpaint,
            paired_latents_ref=_paired_latents_reference(latents, label="Denoise latents"),
        )
    elif route_state._contract == _SDXL_ROUTE_CONTRACT:
        if actual_mask is not None:
            raise ValueError("SDXL Denoise must not publish hidden Qwen mask state.")
        _require_sdxl_vae_provenance(route_state._payload, vae_component)
        _validate_sdxl_latent_tensor(
            latents,
            label="SDXL Denoise latents",
            latent_channels=route_state._payload._vae_latent_channels,
            scale_factor=route_state._payload._vae_scale_factor,
        )
        payload = _SdxlRoutePayload(
            generator_snapshot=None,
            inpaint=route_state._inpaint,
            paired_latents_ref=_paired_latents_reference(latents, label="Denoise latents"),
            padding_mask_crop=route_state._payload._padding_mask_crop,
            crops_coords=route_state._payload._crops_coords,
            original_image_snapshot=route_state._payload._original_image_snapshot,
            original_mask_snapshot=route_state._payload._original_mask_snapshot,
            vae_ref=route_state._payload._vae_ref,
            vae_latent_channels=route_state._payload._vae_latent_channels,
            vae_scale_factor=route_state._payload._vae_scale_factor,
        )
    else:
        if route_state._contract != _WAN_ROUTE_CONTRACT:
            raise ValueError("The Denoise route carries an unknown state contract.")
        if actual_mask is not None:
            raise ValueError("Wan Denoise must not publish hidden mask state.")
        input_payload = route_state._payload
        z_dim, spatial_scale, temporal_scale = _require_wan_vae_provenance(input_payload, vae_component)
        transformer_seal = wan_transformer_contract_from_component(
            transformer_component,
            workflow=input_payload._workflow,
        )
        _validate_wan_video_tensor(
            latents,
            label="Wan Denoise latents",
            channels=z_dim,
            num_frames=input_payload._num_frames,
            height=input_payload._second_height,
            width=input_payload._second_width,
            spatial_scale=spatial_scale,
            temporal_scale=temporal_scale,
        )
        if execution_device is None or not _devices_compatible(latents.device, execution_device):
            raise ValueError("Wan Denoise output latents must be resident on the Denoise execution device.")
        source_image = input_payload._source_image_ref()
        last_image = input_payload._last_image_ref() if input_payload._last_image_ref is not None else None
        if source_image is None or (input_payload._last_image_ref is not None and last_image is None):
            raise ValueError("Wan source media paired with this route is no longer resident.")
        _require_wan_media_snapshot(_wan_payload_media_snapshot(input_payload), source_image, last_image)
        _require_paired_latents_resident(input_payload._image_embeds_ref, label="Wan image embeddings")
        _require_paired_latents_resident(
            input_payload._image_condition_latents_ref,
            label="Wan image condition latents",
        )
        payload = _WanRoutePayload(
            generator_snapshot=None,
            paired_latents_ref=_paired_latents_reference(latents, label="Wan Denoise latents"),
            image_embeds_ref=input_payload._image_embeds_ref,
            image_condition_latents_ref=input_payload._image_condition_latents_ref,
            source_image_ref=input_payload._source_image_ref,
            source_image_seal=input_payload._source_image_seal,
            last_image_ref=input_payload._last_image_ref,
            last_image_seal=input_payload._last_image_seal,
            workflow=input_payload._workflow,
            requested_height=input_payload._requested_height,
            requested_width=input_payload._requested_width,
            first_height=input_payload._first_height,
            first_width=input_payload._first_width,
            second_height=input_payload._second_height,
            second_width=input_payload._second_width,
            num_frames=input_payload._num_frames,
            image_encoder_ref=input_payload._image_encoder_ref,
            image_encoder_config_seal=input_payload._image_encoder_config_seal,
            image_processor_ref=input_payload._image_processor_ref,
            image_processor_config_seal=input_payload._image_processor_config_seal,
            image_encoder_execution_device=input_payload._image_encoder_execution_device,
            vae_ref=input_payload._vae_ref,
            video_processor_ref=input_payload._video_processor_ref,
            video_processor_config_seal=input_payload._video_processor_config_seal,
            vae_config_seal=input_payload._vae_config_seal,
            vae_execution_device=input_payload._vae_execution_device,
            transformer_ref=_component_identity_reference(
                transformer_component,
                label="Connected Wan transformer",
            ),
            transformer_config_seal=transformer_seal,
        )
    return _new_route_state(
        stage=_DENOISE_TO_DECODE,
        binding=binding,
        seed=route_state._seed,
        contract=route_state._contract,
        payload=payload,
    )


def issue_normal_decode_route_state(
    *,
    binding,
    latents,
    vae_component=None,
    vae_latent_channels=None,
    vae_scale_factor=None,
):
    """Bind a text/control Denoise result to normal Decode semantics."""

    if not _is_issued_binding(binding):
        raise ValueError("Cannot issue Decode route state for an invalid ModelsLoader binding.")
    if binding._model_type not in SUPPORTED_ROUTE_MODEL_TYPES:
        raise ValueError(f"Pipeline '{binding._model_type}' does not declare opaque route-state support.")
    contract = route_contract_for_model_type(binding._model_type)
    if contract == _WAN_ROUTE_CONTRACT:
        raise ValueError("Wan image-to-video Denoise requires its preceding image/VAE route state.")
    if contract == _SDXL_ROUTE_CONTRACT:
        if vae_latent_channels is None or vae_scale_factor is None:
            raise ValueError("A normal SDXL Decode route requires the exact connected VAE geometry.")
        if sdxl_vae_geometry_from_component(vae_component) != (
            vae_latent_channels,
            vae_scale_factor,
        ):
            raise ValueError("A normal SDXL Decode route must match the exact connected VAE component.")
        _validate_sdxl_latent_tensor(
            latents,
            label="SDXL Denoise latents",
            latent_channels=vae_latent_channels,
            scale_factor=vae_scale_factor,
        )
    return _new_route_state(
        stage=_DENOISE_TO_DECODE,
        binding=binding,
        seed=None,
        contract=contract,
        payload=(
            _SdxlRoutePayload(
                generator_snapshot=None,
                inpaint=False,
                paired_latents_ref=_paired_latents_reference(latents, label="Denoise latents"),
                vae_ref=_component_identity_reference(vae_component, label="Connected SDXL VAE"),
                vae_latent_channels=vae_latent_channels,
                vae_scale_factor=vae_scale_factor,
            )
            if contract == _SDXL_ROUTE_CONTRACT
            else _QwenRoutePayload(
                generator_snapshot=None,
                processed_mask_image=None,
                mask_overlay_kwargs=None,
                inpaint=False,
                paired_latents_ref=_paired_latents_reference(latents, label="Denoise latents"),
            )
        ),
    )


def consume_decode_route_state(
    route_state,
    *,
    binding,
    model_type,
    latents,
    vae_component=None,
    vae_latent_channels=None,
    vae_scale_factor=None,
    video_processor=None,
    execution_device=None,
    materialize_overlay=True,
):
    """Validate Denoise-to-Decode state and return only normal decode kwargs."""

    if not _is_issued_route_state(route_state):
        raise ValueError("The Modular route state was not issued by this backend process.")
    if route_state._stage != _DENOISE_TO_DECODE:
        raise ValueError("The Modular route state is connected to the wrong action stage.")
    if route_state._binding is not binding:
        raise ValueError("The Modular route state comes from a different Models Loader execution.")
    if binding._model_type != model_type:
        raise ValueError("The Modular route state belongs to a different pipeline class.")
    if route_state._contract != route_contract_for_model_type(model_type):
        raise ValueError("The Modular route state belongs to a different pipeline state contract.")
    _require_paired_latents(route_state, latents, label="Denoise latents")
    if route_state._contract == _SDXL_ROUTE_CONTRACT:
        payload = route_state._payload
        _require_sdxl_vae_provenance(payload, vae_component)
        _validate_sdxl_latent_tensor(
            latents,
            label="SDXL Denoise latents",
            latent_channels=payload._vae_latent_channels,
            scale_factor=payload._vae_scale_factor,
        )
        if vae_latent_channels is not None or vae_scale_factor is not None:
            if (vae_latent_channels, vae_scale_factor) != (
                payload._vae_latent_channels,
                payload._vae_scale_factor,
            ):
                raise ValueError("The connected Decode VAE geometry does not match the Denoise route.")
        decode_inputs = None
        if payload._padding_mask_crop is not None:
            if (
                payload._crops_coords is None
                or payload._original_image_snapshot is None
                or payload._original_mask_snapshot is None
            ):
                raise ValueError("The sealed SDXL crop-overlay route is incomplete.")
            if materialize_overlay:
                decode_inputs = {
                    "image": _materialize_overlay_media(payload._original_image_snapshot),
                    "mask_image": _materialize_overlay_media(payload._original_mask_snapshot),
                    "padding_mask_crop": payload._padding_mask_crop,
                    "crops_coords": payload._crops_coords,
                }
        return {
            "contract": _SDXL_ROUTE_CONTRACT,
            "inpaint": route_state._inpaint,
            "decode_inputs": decode_inputs,
            "mask_overlay_kwargs": None,
        }
    if route_state._contract == _WAN_ROUTE_CONTRACT:
        payload = route_state._payload
        z_dim, spatial_scale, temporal_scale = _require_wan_vae_provenance(payload, vae_component)
        if video_processor is not None:
            require_wan_video_processor(video_processor)
        if execution_device is not None and not _devices_compatible(latents.device, execution_device):
            raise ValueError("Wan Denoise latents must be resident on the Decode execution device.")
        source_image = payload._source_image_ref()
        last_image = payload._last_image_ref() if payload._last_image_ref is not None else None
        if source_image is None or (payload._last_image_ref is not None and last_image is None):
            raise ValueError("Wan source media paired with this route is no longer resident.")
        _require_wan_media_snapshot(_wan_payload_media_snapshot(payload), source_image, last_image)
        _require_paired_latents_resident(payload._image_embeds_ref, label="Wan image embeddings")
        _require_paired_latents_resident(
            payload._image_condition_latents_ref,
            label="Wan image condition latents",
        )
        transformer = _require_component_reference(
            payload._transformer_ref,
            None,
            label="Wan transformer",
            require_connected=False,
        )
        if (
            wan_transformer_contract_from_component(transformer, workflow=payload._workflow)
            != payload._transformer_config_seal
        ):
            raise ValueError("The Wan transformer contract changed after Denoise execution.")
        _validate_wan_video_tensor(
            latents,
            label="Wan Denoise latents",
            channels=z_dim,
            num_frames=payload._num_frames,
            height=payload._second_height,
            width=payload._second_width,
            spatial_scale=spatial_scale,
            temporal_scale=temporal_scale,
        )
        return {
            "contract": _WAN_ROUTE_CONTRACT,
            "inpaint": False,
            "decode_inputs": None,
            "mask_overlay_kwargs": None,
        }
    return {
        "contract": _QWEN_ROUTE_CONTRACT,
        "inpaint": route_state._inpaint,
        "decode_inputs": None,
        "mask_overlay_kwargs": (
            dict(route_state._mask_overlay_kwargs) if route_state._mask_overlay_kwargs is not None else None
        ),
    }
