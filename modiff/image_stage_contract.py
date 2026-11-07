"""Exact static backend image-stage declarations; no execution or qualification."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from modiff.model_artifact_catalog import catalog_repository_pin


def declared_backend_image_stages(
    profile: Mapping[str, Any], mode: str, artifact: Mapping[str, Any], spec: Mapping[str, Any] | None,
) -> bool:
    """Inspect the current exact image specification, never the node registry.

    This proves only a declared backend graph. Upstream Modular availability,
    actual runtime readiness, model output and fit remain independent evidence.
    """
    if (not isinstance(spec, Mapping)
            or profile.get("execution_path") != "direct-diffusers-image"
            or profile.get("loader_module") != "modules.DiffusersImage"
            or profile.get("loader_action") != "LoadPipeline"):
        return False
    identity = {
        "modelType": "model_type", "executionProfileId": "id", "pipelineClass": "pipeline_class",
        "executionPath": "execution_path", "loaderModule": "loader_module", "loaderAction": "loader_action",
        "defaultRepo": "default_repo",
    }
    if (any(spec.get(key) != profile.get(field) for key, field in identity.items())
            or spec.get("mode") != mode or mode not in profile.get("modes", ())
            or artifact.get("repository") != profile.get("default_repo")):
        return False
    revision = artifact.get("revision")
    if not isinstance(revision, str) or re.fullmatch(r"[a-f0-9]{40}", revision) is None:
        return False
    pin = catalog_repository_pin(artifact["repository"], model_type=profile["model_type"])
    if not pin or (pin.get("repo"), pin.get("revision")) != (artifact["repository"], revision):
        return False
    rows, raw_edges = spec.get("roles"), spec.get("edges")
    if (not isinstance(rows, (list, tuple)) or not isinstance(raw_edges, (list, tuple))
            or any(not isinstance(row, (list, tuple)) or len(row) != 4
                   or not isinstance(row[0], str) or not isinstance(row[1], str) for row in rows)
            or any(not isinstance(edge, (list, tuple)) or len(edge) != 4
                   or any(not isinstance(part, str) for part in edge) for edge in raw_edges)):
        return False
    keys = ("modules.DiffusersImage.LoadPipeline", "modules.DiffusersImage.EncodeInputs",
            "modules.DiffusersImage.Denoise", "modules.DiffusersImage.DecodeLatents")
    roles = {role: key for role, key, _x, _y in rows}
    if len(roles) != len(rows) or any(sum(key == target for key in roles.values()) != 1 for target in keys):
        return False
    owner, encode, denoise, decode = (next(role for role, key in roles.items() if key == target) for target in keys)
    edges = {tuple(edge) for edge in raw_edges}
    if len(edges) != len(raw_edges) or any(source not in roles or target not in roles for source, _out, target, _in in edges):
        return False
    required = {(owner, "pipeline", target, "pipeline") for target in (encode, denoise, decode)} | {
        (encode, "encoded_inputs", denoise, "encoded_inputs"),
        (denoise, "latents_out", decode, "latents"), (denoise, "latent_binding", decode, "latent_binding"),
        (denoise, "width_out", decode, "width"), (denoise, "height_out", decode, "height"),
    }
    core = {owner, encode, denoise, decode}
    if ({edge for edge in edges if edge[0] in core and edge[2] in core} != required
            or {edge for edge in edges if edge[0] == owner and edge[1] == "pipeline"}
            != {(owner, "pipeline", target, "pipeline") for target in (encode, denoise, decode)}
            or any(sum(edge[2:] == (target, field) for edge in edges) != 1
                   for _source, _output, target, field in required)):
        return False
    return any(source == decode and output == "images" and roles[target] == "modules.Image.Preview"
               and field == "image" for source, output, target, field in edges)
