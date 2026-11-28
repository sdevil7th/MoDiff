"""Compare retained image outputs under identical, explicit generation contracts.

This is local migration evidence, not model, Auto, Gallery, or aesthetic approval.
Graph node IDs may change during a migration; consumed recipe contracts may not.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image
from packaging.version import InvalidVersion, Version


CONTRACT_FORMAT = "modiff.image-template-comparison-contract.v1"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_HASH = re.compile(r"(?:sha256:)?[0-9a-f]{64}")


def validate_comparison_contract(contract: Mapping) -> None:
    """Reject guessed/random recipes and mutable model/input identities."""
    if not isinstance(contract, Mapping) or contract.get("format") != CONTRACT_FORMAT:
        raise ValueError(f"Comparison contracts must declare {CONTRACT_FORMAT}.")
    for name in ("templateId", "task"):
        if not isinstance(contract.get(name), str) or not contract[name]:
            raise ValueError(f"Comparison contract requires {name}.")
    models = contract.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Comparison contract requires the actual immutable model set.")
    for model in models:
        if not isinstance(model, dict) or not all(isinstance(model.get(key), str) and model[key] for key in ("role", "repoId")):
            raise ValueError("Every compared model requires a role and repository identity.")
        if not isinstance(model.get("revision"), str) or not _COMMIT.fullmatch(model["revision"]):
            raise ValueError("Every compared model requires an immutable 40-character revision.")
    settings = contract.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("Comparison contract requires actually consumed settings.")
    seed = settings.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Comparison requires the resolved nonnegative seed, not a random-seed request.")
    runtime = contract.get("runtime")
    if not isinstance(runtime, dict) or not all(
        isinstance(runtime.get(key), str) and runtime[key]
        for key in ("torch", "transformers", "peft", "platform", "device")
    ):
        raise ValueError("Comparison requires the actual runtime and accelerator identity.")
    if "diffusersCommit" in runtime:
        if not isinstance(runtime["diffusersCommit"], str) or not _COMMIT.fullmatch(runtime["diffusersCommit"]):
            raise ValueError("Comparison requires the immutable Diffusers source revision.")
        if "diffusersWheelSha256" in runtime:
            raise ValueError("Comparison must choose one actual Diffusers source or wheel identity.")
    else:
        version = runtime.get("diffusersVersion")
        digest = runtime.get("diffusersWheelSha256")
        if not isinstance(version, str) or not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("Comparison requires the actual Diffusers version and verified wheel SHA-256.")
        try:
            canonical_version = str(Version(version))
        except InvalidVersion as error:
            raise ValueError("Comparison requires an exact published Diffusers distribution version.") from error
        if canonical_version != version:
            raise ValueError("Comparison requires an exact published Diffusers distribution version.")
    inputs = contract.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError("Comparison requires input hashes, including an explicit empty list for text-only recipes.")
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) or not item["role"]:
            raise ValueError("Each compared input requires its consumed role.")
        if not isinstance(item.get("sha256"), str) or not _HASH.fullmatch(item["sha256"]):
            raise ValueError("Each compared input requires its actual SHA-256.")
    expected = contract.get("expectedOutputs")
    if not isinstance(expected, list) or not expected:
        raise ValueError("Comparison requires the complete ordered output dimension contract.")
    for size in expected:
        if not isinstance(size, dict) or any(
            isinstance(size.get(key), bool) or not isinstance(size.get(key), int) or size[key] <= 0
            for key in ("width", "height")
        ):
            raise ValueError("Each expected image requires positive integer width and height.")
    # Reject NaN/Infinity even in otherwise opaque consumed settings.
    json.dumps(contract, sort_keys=True, allow_nan=False)


def contract_hash(contract: Mapping) -> str:
    validate_comparison_contract(contract)
    material = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(material.encode()).hexdigest()


def _differences(before, after, path="") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        return [difference for key in sorted(before.keys() | after.keys())
                for difference in ([f"{path}.{key}".lstrip(".")] if key not in before or key not in after
                                   else _differences(before[key], after[key], f"{path}.{key}".lstrip(".")))]
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [path]
        return [difference for index, (left, right) in enumerate(zip(before, after))
                for difference in _differences(left, right, f"{path}[{index}]")]
    return [] if type(before) is type(after) and before == after else [path]


def compare_image_template_outputs(
    baseline: Sequence[Image.Image], candidate: Sequence[Image.Image], *,
    baseline_contract: Mapping, candidate_contract: Mapping, max_pixel_error: int = 0,
) -> dict:
    """Default to exact decoded RGBA pixels, independent of PNG metadata.

    A nonzero tolerance must be declared by the caller and remains numerical
    comparison evidence. It cannot establish better aesthetics or qualify Auto.
    """
    validate_comparison_contract(baseline_contract)
    validate_comparison_contract(candidate_contract)
    if isinstance(max_pixel_error, bool) or not isinstance(max_pixel_error, int) or not 0 <= max_pixel_error <= 255:
        raise ValueError("max_pixel_error must be an integer from 0 to 255.")
    mismatches = _differences(dict(baseline_contract), dict(candidate_contract))
    expected = baseline_contract["expectedOutputs"]
    count_matches = len(baseline) == len(candidate) == len(expected)
    outputs = []
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        size = expected[index] if index < len(expected) else {}
        dimensions_match = left.size == right.size == (size.get("width"), size.get("height"))
        record = {"index": index, "baselineSize": list(left.size), "candidateSize": list(right.size),
                  "dimensionsMatch": dimensions_match, "pixelIdentity": False, "withinTolerance": False}
        if dimensions_match:
            before = np.asarray(left.convert("RGBA"), dtype=np.uint8)
            after = np.asarray(right.convert("RGBA"), dtype=np.uint8)
            delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
            mse = float(np.square(delta.astype(np.float64)).mean())
            maximum = int(delta.max())
            prefix = json.dumps({"size": left.size, "mode": "RGBA"}, sort_keys=True).encode()
            record.update(
                baselinePixelHash="sha256:" + hashlib.sha256(prefix + before.tobytes()).hexdigest(),
                candidatePixelHash="sha256:" + hashlib.sha256(prefix + after.tobytes()).hexdigest(),
                pixelIdentity=maximum == 0, withinTolerance=maximum <= max_pixel_error,
                maxPixelError=maximum, meanAbsolutePixelError=float(delta.mean()),
                rootMeanSquarePixelError=math.sqrt(mse),
                changedPixelFraction=float(np.any(delta != 0, axis=-1).mean()),
                psnrDb=None if mse == 0 else 10 * math.log10(255 ** 2 / mse),
            )
        outputs.append(record)
    passed = not mismatches and count_matches and all(item["withinTolerance"] for item in outputs)
    return {
        "format": "modiff.image-template-comparison.v1",
        "templateId": baseline_contract["templateId"],
        "baselineContractHash": contract_hash(baseline_contract),
        "candidateContractHash": contract_hash(candidate_contract),
        "contractMismatchPaths": mismatches, "outputCountMatches": count_matches,
        "baselineCount": len(baseline), "candidateCount": len(candidate),
        "declaredMaxPixelError": max_pixel_error, "outputs": outputs,
        "passed": passed, "pixelIdentity": passed and all(item["pixelIdentity"] for item in outputs),
        "aestheticApproval": "human_review_required", "qualifiesAuto": False, "publishesGalleryAssets": False,
    }
