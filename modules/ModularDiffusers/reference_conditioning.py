"""Process-local authority for ordered image conditioning and authored regions.

The existing graph carries the tensors. This receipt carries weak identities and
bounded fingerprints only; JSON, a reference count, or a saved workflow cannot
create it. Source boxes use the actual preprocessed reference frame.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import weakref

from PIL import Image
import torch

from .route_state import (
    _paired_latents_reference,
    _require_paired_latents_reference,
    require_component_binding,
    resolve_managed_component_by_id,
)


REFERENCE_BINDING = "reference_binding"


class _ReferenceBindingKey:
    __slots__ = ()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce_ex__(self, protocol):
        raise TypeError("Reference markers are process-local and cannot be serialized.")


_PROMPT_BINDING = _ReferenceBindingKey()
_ISSUER = object()
_MAX_SOURCE_RGB_BYTES = 256 * 1024 * 1024
_CONTRACT = {"schemaVersion": 1, "maxReferences": 10}


def declares_reference_binding(config, *, output=False):
    if output and REFERENCE_BINDING not in config["output_names"]:
        return False
    field = config["params"].get(REFERENCE_BINDING, {})
    return (field.get("type") == "reference_conditioning"
            and field.get("display") == ("output" if output else "input")
            and field.get("fieldOptions", {}).get("referenceBinding") == _CONTRACT)


def _images(value):
    images = value if type(value) is list else [value]
    if not 1 <= len(images) <= _CONTRACT["maxReferences"]:
        raise ValueError("Reference conditioning requires between 1 and 10 images.")
    for image in images:
        if not isinstance(image, Image.Image):
            raise ValueError("Reference conditioning requires actual PIL images.")
        if image.width <= 0 or image.height <= 0 or image.width * image.height * 3 > _MAX_SOURCE_RGB_BYTES:
            raise ValueError("Reference source exceeds the 256 MiB RGB fingerprint limit.")
    return images


def _source_seal(image):
    # Convert bounded strips, including palette colour interpretation, exactly
    # as RGB preprocessing does. Do not retain a second full-sized source.
    digest = hashlib.sha256()
    rows = max(1, (1024 * 1024) // max(1, image.width * 3))
    for top in range(0, image.height, rows):
        digest.update(image.crop((0, top, image.width, min(image.height, top + rows))).convert("RGB").tobytes())
    return image.size, digest.hexdigest()


def _processed_seal(tensor):
    if (type(tensor) is not torch.Tensor or tensor.ndim != 4 or tensor.shape[:2] != (1, 3)
            or tensor.layout != torch.strided or tensor.device.type == "meta"
            or not tensor.dtype.is_floating_point or tensor.shape[2] <= 0 or tensor.shape[3] <= 0
            or tensor.shape[2] * tensor.shape[3] > 4096 * 4096):
        raise ValueError("Reference preprocessing must expose its actual bounded RGB tensor geometry.")
    digest = hashlib.sha256()
    rows = max(1, (1024 * 1024) // max(1, tensor.shape[3] * tensor.element_size() * 3))
    for top in range(0, tensor.shape[2], rows):
        chunk = tensor[:, :, top:top + rows].detach().cpu().contiguous()
        digest.update(chunk.view(torch.uint8).numpy().tobytes())
    return tuple(tensor.shape), str(tensor.dtype), digest.hexdigest()


def snapshot_reference_sources(images):
    return tuple((weakref.ref(image), _source_seal(image)) for image in _images(images))


class _ReferenceBinding:
    __slots__ = ("_issuer", "_token", "_vae_ref", "_vae_id", "_sources", "_latents", "_ids",
                 "_summaries", "_fingerprint", "__weakref__")

    def __init__(self, issuer, *, token, vae, vae_id, sources, latents, identities, summaries):
        if issuer is not _ISSUER:
            raise ValueError("Reference bindings can only be issued by Encode Image.")
        for key, value in (("_issuer", issuer), ("_token", token), ("_vae_ref", weakref.ref(vae)),
                           ("_vae_id", vae_id), ("_sources", sources), ("_latents", latents), ("_ids", identities),
                           ("_summaries", summaries), ("_fingerprint", "sha256:" + secrets.token_hex(32))):
            object.__setattr__(self, key, value)

    def __setattr__(self, name, value):
        raise AttributeError("Reference bindings are immutable.")

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce_ex__(self, protocol):
        raise TypeError("Reference bindings are process-local and cannot be serialized.")


def issue_reference_binding(*, vae_payload, manager, images, processed_images, latents, source_snapshot):
    source_images = _images(images)
    if type(processed_images) is not list or len(processed_images) != len(source_images):
        raise ValueError("Reference preprocessing did not preserve the actual ordered image count.")
    if type(latents) is not list or len(latents) != len(source_images):
        raise ValueError("Reference encoding did not preserve the actual ordered latent count.")
    token = require_component_binding(vae_payload, label="Reference VAE", expected_role="vae")
    vae = resolve_managed_component_by_id(manager, vae_payload, label="Reference VAE")
    sources, identities, summaries = [], [], []
    for image, processed in zip(source_images, processed_images):
        seal = _source_seal(image)
        processed_seal = _processed_seal(processed)
        sources.append((weakref.ref(image), seal))
        encoded = json.dumps((seal, processed_seal), separators=(",", ":")).encode("ascii")
        identities.append("sha256:" + hashlib.sha256(encoded).hexdigest())
        summaries.append((image.size, (processed.shape[3], processed.shape[2])))
    if source_snapshot != tuple(sources):
        raise ValueError("Reference source content changed during encoding; rerun Encode Image.")
    return _ReferenceBinding(_ISSUER, token=token, vae=vae, vae_id=vae_payload["model_id"],
                             sources=tuple(sources), identities=tuple(identities), summaries=tuple(summaries),
                             latents=_paired_latents_reference(latents, label="Reference latents", allow_list=True))


def require_reference_binding(value, *, component, role, manager, latents=None, images=None):
    if type(value) is not _ReferenceBinding or value._issuer is not _ISSUER:
        raise ValueError("Reference conditioning requires the opaque binding emitted by Encode Image.")
    require_component_binding(component, label=role, expected_role=role, expected_token=value._token)
    vae = value._vae_ref()
    resident = manager.get_components_by_ids(ids=[value._vae_id], return_dict_with_names=False)
    if vae is None or type(resident) is not dict or resident.get(value._vae_id) is not vae:
        raise ValueError("The reference VAE owner is no longer resident; rerun Encode Image.")
    if images is not None:
        actual = _images(images)
        if len(actual) != len(value._sources) or any(ref() is not image for (ref, _), image in zip(value._sources, actual)):
            raise ValueError("Reference image identity or order changed; rerun Encode Image.")
    for reference, seal in value._sources:
        image = reference()
        if image is None or _source_seal(image) != seal:
            raise ValueError("Reference source content changed; rerun Encode Image and explicitly rebind source regions.")
    if latents is None:
        references = value._latents[1]
        latents = [reference() for reference, _seal in references]
    _require_paired_latents_reference(value._latents, latents, label="reference conditioning latents")
    return value._ids


def reference_summary(binding, *, component, role, manager, identity):
    identities = require_reference_binding(binding, component=component, role=role, manager=manager)
    owner = json.dumps((id(binding._token), binding._token._model_type, binding._vae_id), separators=(",", ":"))
    result = {
        "schemaVersion": 1, "bindingFingerprint": binding._fingerprint,
        "ownerFingerprint": "sha256:" + hashlib.sha256(owner.encode("utf-8")).hexdigest(),
        "references": [{"referenceId": identifier, "sourceSize": list(sizes[0]), "processedSize": list(sizes[1])}
                       for identifier, sizes in zip(identities, binding._summaries)],
    }
    task = identity.get("task_id")
    attempt = identity.get("attempt_index")
    if type(task) is str and 0 < len(task) <= 128 and type(attempt) is int and 0 <= attempt <= 1000:
        result.update(taskId=task, attemptIndex=attempt)
        run_hash = identity.get("run_input_hash")
        if type(run_hash) is str and len(run_hash) == 12 and run_hash.startswith("run_"):
            result["runInputHash"] = run_hash
    return json.dumps(result, separators=(",", ":"))


def mark_reference_embeddings(embeddings, binding):
    return {**embeddings, _PROMPT_BINDING: binding}


def validate_denoise_reference_inputs(values, *, manager):
    embeddings = values.get("embeddings")
    marker = embeddings.get(_PROMPT_BINDING) if type(embeddings) is dict else None
    binding = values.get(REFERENCE_BINDING)
    if marker is None and binding is None:
        return
    if marker is not binding or binding is None:
        raise ValueError("Prompt and Denoise must use the same Encode Image reference binding.")
    if values.get("image_cond_latents") is None:
        raise ValueError("Denoise requires the exact reference latents paired with its binding.")
    require_reference_binding(binding, component=values.get("unet"), role="denoiser", manager=manager,
                              latents=values.get("image_cond_latents"))


def unmark_reference_embeddings(embeddings):
    if type(embeddings) is not dict or _PROMPT_BINDING not in embeddings:
        return embeddings
    return {key: value for key, value in embeddings.items() if key is not _PROMPT_BINDING}
