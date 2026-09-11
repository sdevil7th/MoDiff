"""Exact, local-only resolution for reviewed SDXL IP-Adapter artifacts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

from modiff.model_artifact_catalog import catalog_repository_pin


_EXACT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_EXACT_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PURPOSE = "sdxl-ip-adapter"
_IMAGE_ENCODER_CLASS = "CLIPVisionModelWithProjection"
_MAX_WEIGHT_BYTES = 2 * 1024 * 1024 * 1024
_MAX_IMAGE_ENCODER_FILE_BYTES = 8 * 1024 * 1024 * 1024


@dataclass(frozen=True)
class ResolvedSDXLIPAdapter:
    repository: str
    revision: str
    weight_name: str
    content_sha256: str
    byte_size: int
    image_encoder_subfolder: str
    image_encoder_class: str
    load_directory: Path

    @property
    def identity(self) -> tuple[str, str, str, str, int, str, str]:
        return (
            self.repository,
            self.revision,
            self.weight_name,
            self.content_sha256,
            self.byte_size,
            self.image_encoder_subfolder,
            self.image_encoder_class,
        )


def _exact_posix_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 512:
        raise ValueError(f"Reviewed IP-Adapter {label} must be a bounded relative POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") or ":" in part for part in path.parts):
        raise ValueError(f"Reviewed IP-Adapter {label} must be a traversal-free relative POSIX path.")
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_reviewed_sdxl_ip_adapter(
    *,
    selection: Any,
    revision: Any,
    weight_name: Any,
) -> ResolvedSDXLIPAdapter:
    """Resolve one exact reviewed adapter from the local Hub cache only."""

    if not isinstance(selection, Mapping):
        raise TypeError("SDXL IP-Adapter model selection must be a model-selector object.")
    if selection.get("source") != "hub":
        raise ValueError("SDXL IP-Adapter currently supports only reviewed immutable Hub artifacts.")
    repository = selection.get("value")
    if not isinstance(repository, str) or not repository or repository != repository.strip() or len(repository) > 512:
        raise ValueError("SDXL IP-Adapter requires one bounded reviewed repository ID.")
    pin = catalog_repository_pin(repository)
    if not isinstance(pin, dict) or pin.get("purpose") != _PURPOSE:
        raise ValueError("The selected repository is not a reviewed SDXL IP-Adapter artifact.")
    canonical_repository = pin.get("repo")
    if repository != canonical_repository:
        raise ValueError("The selected SDXL IP-Adapter repository spelling is not canonical.")

    selected_revision = str(revision or "").strip()
    reviewed_revision = pin.get("revision")
    if (
        not _EXACT_REVISION.fullmatch(selected_revision)
        or selected_revision != reviewed_revision
    ):
        raise ValueError("SDXL IP-Adapter requires its reviewed immutable repository revision.")
    selected_weight = _exact_posix_path(weight_name, label="weight name")
    reviewed_weight = _exact_posix_path(pin.get("weightName"), label="catalog weight name")
    if selected_weight != reviewed_weight:
        raise ValueError("SDXL IP-Adapter requires its reviewed single-adapter weight file.")
    image_encoder_subfolder = _exact_posix_path(
        pin.get("imageEncoderSubfolder"),
        label="image-encoder subfolder",
    )
    if pin.get("imageEncoderClass") != _IMAGE_ENCODER_CLASS:
        raise ValueError("The reviewed SDXL IP-Adapter image-encoder class is invalid.")
    encoder_files = pin.get("imageEncoderFiles")
    expected_encoder_files = {
        f"{image_encoder_subfolder}/config.json",
        f"{image_encoder_subfolder}/model.safetensors",
    }
    if not isinstance(encoder_files, list) or len(encoder_files) != len(expected_encoder_files):
        raise ValueError("The reviewed SDXL IP-Adapter image-encoder file inventory is invalid.")
    reviewed_encoder_files = set()
    for contract in encoder_files:
        if not isinstance(contract, Mapping):
            raise ValueError("The reviewed SDXL IP-Adapter image-encoder file contract is invalid.")
        filename = _exact_posix_path(contract.get("filename"), label="image-encoder filename")
        digest = contract.get("sha256")
        size = contract.get("byteSize")
        if (
            not isinstance(digest, str)
            or not _EXACT_SHA256.fullmatch(digest)
            or type(size) is not int
            or not 1 <= size <= _MAX_IMAGE_ENCODER_FILE_BYTES
        ):
            raise ValueError("The reviewed SDXL IP-Adapter image-encoder file contract is invalid.")
        reviewed_encoder_files.add(filename)
    if reviewed_encoder_files != expected_encoder_files:
        raise ValueError("The reviewed SDXL IP-Adapter image-encoder file inventory is invalid.")
    content_sha256 = pin.get("sha256")
    byte_size = pin.get("byteSize")
    if not isinstance(content_sha256, str) or not _EXACT_SHA256.fullmatch(content_sha256):
        raise ValueError("The reviewed SDXL IP-Adapter weight digest is invalid.")
    if type(byte_size) is not int or not 1 <= byte_size <= _MAX_WEIGHT_BYTES:
        raise ValueError("The reviewed SDXL IP-Adapter weight size is invalid.")

    try:
        cached_path = hf_hub_download(
            repo_id=canonical_repository,
            revision=selected_revision,
            filename=selected_weight,
            local_files_only=True,
        )
    except (EntryNotFoundError, LocalEntryNotFoundError) as error:
        raise FileNotFoundError(
            "The reviewed SDXL IP-Adapter weight is not installed locally; graph execution never downloads it."
        ) from error
    try:
        cached_file = Path(cached_path)
        resolved_path = cached_file.resolve(strict=True)
        stat = resolved_path.stat()
    except (OSError, RuntimeError, TypeError) as error:
        raise FileNotFoundError("The cached SDXL IP-Adapter weight could not be resolved safely.") from error
    if not resolved_path.is_file() or stat.st_size != byte_size:
        raise ValueError("The cached SDXL IP-Adapter weight does not match its reviewed byte size.")
    if _sha256_file(resolved_path) != content_sha256:
        raise ValueError("The cached SDXL IP-Adapter weight failed its reviewed SHA-256 check.")
    if cached_file.name != PurePosixPath(reviewed_weight).name:
        raise ValueError("The cached SDXL IP-Adapter weight lost its reviewed file identity.")

    return ResolvedSDXLIPAdapter(
        repository=canonical_repository,
        revision=selected_revision,
        # Keep the logical snapshot filename so Diffusers selects its
        # safetensors loader. Integrity is still checked against the resolved
        # content-addressed blob above.
        weight_name=cached_file.name,
        content_sha256=content_sha256,
        byte_size=byte_size,
        image_encoder_subfolder=image_encoder_subfolder,
        image_encoder_class=_IMAGE_ENCODER_CLASS,
        load_directory=cached_file.parent,
    )
