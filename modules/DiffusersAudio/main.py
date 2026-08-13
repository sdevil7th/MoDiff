import inspect
import hashlib
import logging
import os
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from modiff.NodeBase import NodeBase
from modiff.diffusers_offload import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    apply_pipeline_offload,
    offload_mode_param,
)
from modiff.config import CONFIG
from modiff.model_artifact_catalog import IMMUTABLE_HUB_REVISION, catalog_revision
from modiff.path_identifiers import resolve_managed_path_identifier, resolve_runtime_input_path
from utils.huggingface import local_files_only, validate_hf_repo_id
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

logger = logging.getLogger("modiff")

ACE_STEP_DEFAULT_REPO = "ACE-Step/acestep-v15-xl-turbo-diffusers"
STABLE_AUDIO_DEFAULT_REPO = "stabilityai/stable-audio-open-1.0"
LONGCAT_AUDIO_DIT_DEFAULT_REPO = "ruixiangma/LongCat-AudioDiT-1B-Diffusers"
AUDIO_LDM2_DEFAULT_REPO = "cvssp/audioldm2"
ACE_MAX_DURATION_SECONDS = 240.0
ACE_CONTINUATION_MAX_EXTENSION_SECONDS = 180.0
ACE_CONTINUATION_DEFAULT_EXTENSION_SECONDS = 15.0
DEVICE_OPTIONS = list(DEVICE_LIST.keys())
DIRECT_AUDIO_OFFLOAD_MODES = [
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
]
ACE_TASK_TYPES = ["text2music", "cover", "continuation", "repaint"]
STALE_ACE_TASK_TYPES = frozenset({"extract", "lego", "complete"})
AUDIO_SAMPLE_RATE_OPTIONS = {
    "16000": "16 kHz",
    "24000": "24 kHz",
    "44100": "44.1 kHz",
    "48000": "48 kHz",
    "88200": "88.2 kHz",
    "96000": "96 kHz",
}

_AUDIO_CONTRACT_VISIBILITY_FIELDS = (
    "negative_prompt",
    "stable_audio_steps",
    "stable_audio_guidance",
    "num_waveforms",
    "lyrics",
    "vocal_language",
    "num_inference_steps",
    "guidance_scale",
    "shift",
    "bpm",
    "keyscale",
    "timesignature",
    "audio_duration",
    "extension_duration",
    "return_continuation_tail",
    "repainting_start",
    "repainting_end",
    "audio_cover_strength",
)
_ACE_COMMON_VISIBLE_FIELDS = (
    "lyrics",
    "vocal_language",
    "num_inference_steps",
    "guidance_scale",
    "shift",
    "bpm",
    "keyscale",
    "timesignature",
)
_STABLE_AUDIO_VISIBLE_FIELDS = (
    "negative_prompt",
    "stable_audio_steps",
    "stable_audio_guidance",
    "num_waveforms",
    "audio_duration",
)
_LONGCAT_AUDIO_DIT_VISIBLE_FIELDS = (
    "negative_prompt",
    "stable_audio_steps",
    "stable_audio_guidance",
    "audio_duration",
)


@dataclass(frozen=True)
class AudioModeContract:
    mode: str
    task_type: str
    upstream_task_type: str
    source_audio: str
    reference_audio: str
    visible_fields: tuple[str, ...]
    validate_repaint_interval: bool = False
    max_duration_seconds: float | None = None
    max_extension_seconds: float | None = None

    def __post_init__(self) -> None:
        if len(set(self.visible_fields)) != len(self.visible_fields) or any(
            field not in _AUDIO_CONTRACT_VISIBILITY_FIELDS for field in self.visible_fields
        ):
            raise ValueError("Audio mode contracts must declare unique reviewed visibility fields.")

    def field_param_overlay(self) -> dict[str, dict[str, Any]]:
        overlay = {
            field: {"hidden": field not in self.visible_fields}
            for field in _AUDIO_CONTRACT_VISIBILITY_FIELDS
        }
        overlay["task_type"] = {
            "options": [self.task_type],
            "default": self.task_type,
            "value": self.task_type,
        }
        overlay["source_audio"] = {
            "required": self.source_audio == "required",
            "hidden": self.source_audio == "forbidden",
        }
        overlay["reference_audio"] = {
            "required": self.reference_audio == "required",
            "hidden": self.reference_audio == "forbidden",
        }
        overlay["audio_duration"]["max"] = self.max_duration_seconds or ACE_MAX_DURATION_SECONDS
        overlay["extension_duration"]["max"] = (
            self.max_extension_seconds or ACE_CONTINUATION_MAX_EXTENSION_SECONDS
        )
        return overlay

    def signal_value(self, pipeline_class: str, repository: str) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "library": "diffusers",
            "mediaKind": "audio",
            "pipelineClass": pipeline_class,
            "mode": self.mode,
            "repository": repository,
            "taskType": self.task_type,
            "upstreamTaskType": self.upstream_task_type,
            "sourceAudio": self.source_audio,
            "referenceAudio": self.reference_audio,
            "validateRepaintInterval": self.validate_repaint_interval,
            "maxDurationSeconds": self.max_duration_seconds,
            "maxExtensionSeconds": self.max_extension_seconds,
            "fieldParams": self.field_param_overlay(),
        }


@dataclass(frozen=True)
class AudioPipelineAdapter:
    pipeline_class: str
    default_repo: str
    mode_contracts: tuple[AudioModeContract, ...]
    generation_kind: str = "ace_step"
    safe_serialization_required: bool = False
    max_inference_steps: int = 100
    default_inference_steps: int = 8
    default_guidance_scale: float = 1.0
    supports_multiple_waveforms: bool = False
    source_audio_channels: int | None = None
    duplicate_mono_source: bool = False

    def __post_init__(self) -> None:
        if self.generation_kind not in {"ace_step", "stable_audio", "longcat_audio_dit", "audioldm2"}:
            raise ValueError("Audio pipeline adapters must select a reviewed generation kind.")
        if self.max_inference_steps < 1 or not 1 <= self.default_inference_steps <= self.max_inference_steps:
            raise ValueError("Audio pipeline adapters must declare bounded inference-step defaults.")
        if not isfinite(self.default_guidance_scale) or not 0 <= self.default_guidance_scale <= 20:
            raise ValueError("Audio pipeline adapters must declare guidance between 0 and 20.")

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(contract.mode for contract in self.mode_contracts)

    def contract_for_mode(self, mode: str) -> AudioModeContract:
        for contract in self.mode_contracts:
            if contract.mode == mode:
                return contract
        supported = ", ".join(self.modes) or "none"
        raise ValueError(f"{self.pipeline_class} does not support {mode}. Supported modes: {supported}.")

    def resolve_pipeline_class(self):
        import diffusers

        pipeline = getattr(diffusers, self.pipeline_class, None)
        if pipeline is None:
            raise ValueError(f"Diffusers does not expose audio pipeline class {self.pipeline_class}.")
        return pipeline


AUDIO_PIPELINE_ADAPTERS = {
    "AceStepPipeline": AudioPipelineAdapter(
        pipeline_class="AceStepPipeline",
        default_repo=ACE_STEP_DEFAULT_REPO,
        mode_contracts=(
            AudioModeContract(
                "text_to_audio",
                "text2music",
                "text2music",
                "forbidden",
                "forbidden",
                visible_fields=(*_ACE_COMMON_VISIBLE_FIELDS, "audio_duration"),
                max_duration_seconds=ACE_MAX_DURATION_SECONDS,
            ),
            AudioModeContract(
                "audio_variation",
                "cover",
                "cover",
                "required",
                "forbidden",
                visible_fields=(*_ACE_COMMON_VISIBLE_FIELDS, "audio_duration", "audio_cover_strength"),
                max_duration_seconds=ACE_MAX_DURATION_SECONDS,
            ),
            AudioModeContract(
                "audio_continuation",
                "continuation",
                "repaint",
                "required",
                "forbidden",
                visible_fields=(*_ACE_COMMON_VISIBLE_FIELDS, "extension_duration", "return_continuation_tail"),
                max_duration_seconds=ACE_MAX_DURATION_SECONDS,
                max_extension_seconds=ACE_CONTINUATION_MAX_EXTENSION_SECONDS,
            ),
            AudioModeContract(
                "audio_repaint",
                "repaint",
                "repaint",
                "required",
                "forbidden",
                visible_fields=(*_ACE_COMMON_VISIBLE_FIELDS, "repainting_start", "repainting_end"),
                validate_repaint_interval=True,
                max_duration_seconds=ACE_MAX_DURATION_SECONDS,
            ),
        ),
        # The reviewed ACE-Step artifact uses AutoencoderOobleck with
        # ``audio_channels=2``. A mono waveform has one unambiguous,
        # deterministic stereo representation; layouts with more than two
        # channels do not, so they fail closed below instead of being mixed
        # according to an undeclared speaker layout.
        source_audio_channels=2,
        duplicate_mono_source=True,
    ),
    "StableAudioPipeline": AudioPipelineAdapter(
        pipeline_class="StableAudioPipeline",
        default_repo=STABLE_AUDIO_DEFAULT_REPO,
        mode_contracts=(
            AudioModeContract(
                "text_to_audio",
                "text2audio",
                "text2audio",
                "forbidden",
                "forbidden",
                visible_fields=_STABLE_AUDIO_VISIBLE_FIELDS,
                max_duration_seconds=47,
            ),
        ),
        generation_kind="stable_audio",
        safe_serialization_required=True,
        max_inference_steps=300,
        default_inference_steps=100,
        default_guidance_scale=7,
        supports_multiple_waveforms=True,
    ),
    "LongCatAudioDiTPipeline": AudioPipelineAdapter(
        pipeline_class="LongCatAudioDiTPipeline",
        default_repo=LONGCAT_AUDIO_DIT_DEFAULT_REPO,
        mode_contracts=(
            AudioModeContract(
                "text_to_audio",
                "text2audio",
                "text2audio",
                "forbidden",
                "forbidden",
                visible_fields=_LONGCAT_AUDIO_DIT_VISIBLE_FIELDS,
                max_duration_seconds=30,
            ),
        ),
        generation_kind="longcat_audio_dit",
        safe_serialization_required=True,
        max_inference_steps=100,
        default_inference_steps=16,
        default_guidance_scale=4,
    ),
    "AudioLDM2Pipeline": AudioPipelineAdapter(
        pipeline_class="AudioLDM2Pipeline",
        default_repo=AUDIO_LDM2_DEFAULT_REPO,
        mode_contracts=(
            AudioModeContract(
                "text_to_audio",
                "text2audio",
                "text2audio",
                "forbidden",
                "forbidden",
                visible_fields=_STABLE_AUDIO_VISIBLE_FIELDS,
                max_duration_seconds=10,
            ),
        ),
        generation_kind="audioldm2",
        safe_serialization_required=True,
        max_inference_steps=300,
        default_inference_steps=200,
        default_guidance_scale=3.5,
        supports_multiple_waveforms=True,
    ),
}


def get_audio_pipeline_adapter(name: Any) -> AudioPipelineAdapter:
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError("A registered Diffusers audio pipeline class is required.")
    adapter = AUDIO_PIPELINE_ADAPTERS.get(name)
    if adapter is None:
        supported = ", ".join(AUDIO_PIPELINE_ADAPTERS)
        raise ValueError(f"Unsupported Diffusers audio pipeline class {name}. Supported classes: {supported}.")
    return adapter


def _loader_audio_pipeline_adapter(values: dict[str, Any]) -> AudioPipelineAdapter:
    """Require the exact class serialized by the generic loader field."""

    if not isinstance(values, dict):
        raise ValueError("Diffusers audio loader values must be an object.")
    return get_audio_pipeline_adapter(values.get("pipeline_class"))


def _loader_audio_mode(values: dict[str, Any], adapter: AudioPipelineAdapter) -> str:
    mode = values.get("mode") if isinstance(values, dict) else None
    if not isinstance(mode, str) or not mode or mode != mode.strip():
        raise ValueError("A registered Diffusers audio mode is required.")
    adapter.contract_for_mode(mode)
    return mode


def _canonical_model_source(source: Any, *, label: str) -> str:
    if not isinstance(source, str) or not source or source != source.strip():
        raise ValueError(f"{label} source must be exactly hub or local.")
    normalized = source.casefold()
    if normalized not in {"hub", "local"}:
        raise ValueError(f"{label} source must be exactly hub or local.")
    return normalized


def _validated_hub_repository(value: str, *, label: str) -> str:
    if value.count("/") != 1:
        raise ValueError(f"{label} must use an exact Hugging Face namespace/repository ID.")
    try:
        validate_hf_repo_id(value)
    except ValueError as error:
        raise ValueError(f"{label} must use an exact Hugging Face namespace/repository ID.") from error
    try:
        resolves_locally = Path(value).expanduser().exists()
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} could not be validated as a Hugging Face repository ID.") from error
    if resolves_locally:
        raise ValueError(
            f"{label} resolves to an existing local filesystem target. Select source=local for local models."
        )
    return value


def _validated_local_model_directory(value: str) -> str:
    try:
        resolved = Path(value).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Local audio model directory does not exist: {value}") from error
    if not resolved.is_dir():
        raise ValueError(f"Local audio model selection must be a directory: {value}")
    return str(resolved)


def _resolve_audio_model_selection(adapter: AudioPipelineAdapter, value: Any):
    if value is None or (isinstance(value, str) and not value.strip()):
        source = "hub"
        selected = adapter.default_repo
    elif isinstance(value, dict):
        source = _canonical_model_source(value.get("source"), label="Audio model")
        raw_selected = value.get("value")
        if not isinstance(raw_selected, str):
            raise ValueError("Audio model value must be a repository ID or local path string.")
        selected = raw_selected.strip()
        if not selected:
            if source == "hub":
                selected = adapter.default_repo
            else:
                raise ValueError("A local audio model path is required.")
    elif isinstance(value, str):
        source = "hub"
        selected = value.strip()
    else:
        raise ValueError("Audio model selection must be a repository ID or a hub/local selection object.")

    if source == "hub":
        selected = _validated_hub_repository(selected, label="Audio model repository")
    else:
        selected = _validated_local_model_directory(selected)
    managed_defaults = {candidate.default_repo.casefold() for candidate in AUDIO_PIPELINE_ADAPTERS.values()}
    if source != "local" and selected.casefold() in managed_defaults:
        return {"source": "hub", "value": adapter.default_repo}
    return {"source": source, "value": selected}


def _resolve_audio_loader_revision(model_selection: dict[str, str], model_id: str, revision: Any) -> str | None:
    """Resolve one immutable Hub identity while preserving local loading behavior."""

    source = model_selection["source"]
    if source == "local":
        return None

    catalog_pin = catalog_revision(model_id)
    if revision is None or revision == "":
        if catalog_pin is not None:
            return catalog_pin
        raise ValueError(
            f"Custom Hugging Face audio repository {model_id!r} requires an explicit immutable "
            "lowercase 40-character commit SHA revision."
        )
    if (
        not isinstance(revision, str)
        or revision != revision.strip()
        or revision != revision.lower()
        or not IMMUTABLE_HUB_REVISION.fullmatch(revision)
    ):
        raise ValueError("Hugging Face audio revision must be an exact lowercase 40-character commit SHA.")
    if catalog_pin is not None and revision != catalog_pin:
        raise ValueError(
            f"Cataloged Hugging Face audio repository {model_id!r} is pinned to {catalog_pin}; "
            f"the requested revision {revision} does not match."
        )
    return revision


def _required_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an exact 64-character SHA-256 digest.")
    normalized = value.strip().lower().removeprefix("sha256:")
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{label} must be an exact 64-character SHA-256 digest.")
    return normalized


def _validated_hub_filename(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise ValueError(f"{label} must be an exact relative Hub file path without traversal.")
    return value


def _require_lowercase_safetensors_filename(value: str, *, label: str) -> str:
    # The pinned Diffusers LoRA loader selects its deserializer with a
    # case-sensitive suffix check. ``use_safetensors=True`` does not override
    # an explicit ``.bin`` (or differently-cased) weight name.
    if not value.endswith(".safetensors"):
        raise ValueError(f"{label} must end with the literal lowercase .safetensors suffix.")
    return value


def _audio_contract_signal(adapter: AudioPipelineAdapter, mode: str, model_selection: Any) -> dict[str, Any]:
    # Static registry construction must not inspect the process working
    # directory. Runtime/action callers normalize the selection first.
    repository = repo_value(model_selection) or adapter.default_repo
    return adapter.contract_for_mode(mode).signal_value(adapter.pipeline_class, repository)


def repo_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or "")
    return str(value or "")


DEFAULT_AUDIO_CONTRACT = _audio_contract_signal(
    AUDIO_PIPELINE_ADAPTERS["AceStepPipeline"],
    "text_to_audio",
    ACE_STEP_DEFAULT_REPO,
)


def none_if_blank(value: Any):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def value_or_default(value: Any, default: Any):
    return default if value is None else value


def _bounded_float(
    value: Any,
    *,
    default: float,
    label: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number.")
    try:
        number = float(value_or_default(value, default))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number.") from error
    minimum_valid = number >= minimum if minimum_inclusive else number > minimum
    if not isfinite(number) or not minimum_valid or number > maximum:
        comparison = "at least" if minimum_inclusive else "greater than"
        raise ValueError(f"{label} must be finite, {comparison} {minimum:g}, and at most {maximum:g}.")
    return number


def _bounded_int(value: Any, *, default: int, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer from {minimum} through {maximum}.")
    try:
        number = float(value_or_default(value, default))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer from {minimum} through {maximum}.") from error
    if not isfinite(number) or not number.is_integer() or number < minimum or number > maximum:
        raise ValueError(f"{label} must be an integer from {minimum} through {maximum}.")
    return int(number)


def _pcm_to_float32(array: np.ndarray) -> np.ndarray:
    if array.dtype.kind == "u":
        midpoint = float(np.iinfo(array.dtype).max + 1) / 2.0
        return (array.astype(np.float32) - midpoint) / midpoint
    if array.dtype.kind == "i":
        limits = np.iinfo(array.dtype)
        scale = float(max(abs(int(limits.min)), abs(int(limits.max))))
        return array.astype(np.float32) / scale
    return array.astype(np.float32, copy=False)


def pipeline_class_from_name(name: str):
    return get_audio_pipeline_adapter(name).resolve_pipeline_class()


def supports_arg(pipeline: Any, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):
        return False


class AudioInputContractError(ValueError):
    """An audio input violates MoDiff's path or media-layout boundary."""


def _resolve_audio_source_path(value: str | os.PathLike[str]) -> Path:
    """Resolve one graph-controlled audio file inside a configured media root."""

    try:
        runtime_path = resolve_runtime_input_path(value)
        managed_path = resolve_managed_path_identifier(
            runtime_path,
            work_root=CONFIG.paths["work_dir"],
            data_root=CONFIG.paths["data"],
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise AudioInputContractError("The audio source path identifier is invalid.") from error
    if managed_path is None:
        raise AudioInputContractError(
            "The audio source path must stay inside the configured MoDiff work or data directory."
        )
    try:
        resolved = managed_path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise AudioInputContractError(f"The audio source file does not exist: {value}") from error
    if not resolved.is_file():
        raise AudioInputContractError(f"The audio source must be a file: {value}")
    return resolved


def _declared_audio_channels(audio: Any) -> int | None:
    if not isinstance(audio, dict) or "channels" not in audio:
        return None
    value = audio.get("channels")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise AudioInputContractError("Audio channel metadata must be a positive integer.") from error
    if isinstance(value, bool) or not isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise AudioInputContractError("Audio channel metadata must be a positive integer.")
    return int(numeric)


def _declared_audio_layout(audio: Any) -> str | None:
    if not isinstance(audio, dict) or "sample_layout" not in audio:
        return None
    value = audio.get("sample_layout")
    if value not in {"channels_first", "frames_first"}:
        raise AudioInputContractError(
            "Audio sample_layout metadata must be exactly channels_first or frames_first."
        )
    return value


def _is_torch_audio_tensor(value: Any) -> bool:
    value_type = type(value)
    return (
        str(getattr(value_type, "__module__", "")).startswith("torch")
        and callable(getattr(value, "detach", None))
        and callable(getattr(value, "cpu", None))
        and callable(getattr(value, "numpy", None))
        and hasattr(value, "shape")
    )


def _normalize_audio_layout(
    array: np.ndarray,
    *,
    declared_channels: int | None,
    declared_layout: str | None,
    decoded_file: bool,
) -> np.ndarray:
    if array.ndim == 1:
        if declared_channels not in {None, 1}:
            raise AudioInputContractError(
                f"Audio channel metadata declares {declared_channels} channels, but the waveform is one-dimensional."
            )
        return array[None, :]
    if array.ndim != 2:
        return array

    rows, columns = (int(value) for value in array.shape)
    if rows <= 0 or columns <= 0:
        return array
    if decoded_file:
        # scipy.io.wavfile returns [frames, channels]. Decoder provenance is
        # authoritative even for tiny files where shape heuristics cannot be.
        decoded_channels = columns
        if declared_channels is not None and declared_channels != decoded_channels:
            raise AudioInputContractError(
                f"Audio channel metadata declares {declared_channels} channels, but the decoded file has "
                f"{decoded_channels}."
            )
        return array.T

    if declared_layout is not None:
        layout_channels = rows if declared_layout == "channels_first" else columns
        if declared_channels is not None and declared_channels != layout_channels:
            raise AudioInputContractError(
                f"Audio channel metadata declares {declared_channels} channels, but sample_layout "
                f"{declared_layout} identifies {layout_channels}."
            )
        return array if declared_layout == "channels_first" else array.T

    if declared_channels is not None:
        rows_match = rows == declared_channels
        columns_match = columns == declared_channels
        if rows_match and columns_match:
            if declared_channels == 1:
                return array
            raise AudioInputContractError(
                "Audio sample layout is ambiguous because both axes match the declared channel count."
            )
        if rows_match:
            return array
        if columns_match:
            return array.T
        raise AudioInputContractError(
            f"Audio channel metadata declares {declared_channels} channels, but neither sample axis matches."
        )

    if rows <= 8 and columns <= 8 and (rows, columns) != (1, 1):
        raise AudioInputContractError(
            "Short two-dimensional audio has an ambiguous channel orientation; provide exact channels metadata."
        )
    return array.T if rows > columns else array


def audio_to_numpy(audio: Any, *, clip: bool = True) -> tuple[np.ndarray, int]:
    declared_channels = _declared_audio_channels(audio)
    declared_layout = _declared_audio_layout(audio)

    data = audio
    sample_rate = 48000
    if isinstance(audio, dict):
        for field in ("samples", "audio", "array", "path", "file"):
            if field in audio:
                data = audio[field]
                break
        if "sample_rate" in audio:
            sample_rate = int(audio.get("sample_rate"))

    decoded_file = isinstance(data, (str, os.PathLike))
    if decoded_file:
        if declared_layout not in {None, "frames_first"}:
            raise AudioInputContractError(
                "Decoded audio files use frames_first sample_layout; the supplied metadata conflicts."
            )
        audio_path = _resolve_audio_source_path(data)
        from scipy.io import wavfile

        sample_rate, wav_data = wavfile.read(audio_path)
        data = wav_data
    if _is_torch_audio_tensor(data):
        data = data.detach().float().cpu().numpy()
    array = np.asarray(data)
    array = _pcm_to_float32(array)
    array = _normalize_audio_layout(
        array,
        declared_channels=declared_channels,
        declared_layout=declared_layout,
        decoded_file=decoded_file,
    )
    if clip:
        array = np.clip(array, -1.0, 1.0)
    return array, sample_rate


def audio_to_tensor(audio: Any, device: Any, target_sample_rate: int | None = None):
    array, sample_rate = audio_to_numpy(audio)
    return _audio_array_to_tensor(array, sample_rate, device, target_sample_rate)


def _audio_array_to_tensor(
    array: np.ndarray,
    sample_rate: int,
    device: Any,
    target_sample_rate: int | None = None,
):
    import torch
    from scipy.signal import resample_poly
    from math import gcd

    if target_sample_rate and sample_rate != target_sample_rate:
        divisor = gcd(sample_rate, target_sample_rate)
        array = resample_poly(
            array,
            target_sample_rate // divisor,
            sample_rate // divisor,
            axis=-1,
        ).astype(np.float32, copy=False)
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


@dataclass(frozen=True)
class ValidatedAudioInput:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float


def _normalize_source_audio_channels(
    source: ValidatedAudioInput,
    *,
    adapter: AudioPipelineAdapter,
    label: str,
) -> ValidatedAudioInput:
    expected_channels = adapter.source_audio_channels
    if expected_channels is None:
        return source

    actual_channels = int(source.samples.shape[0])
    if actual_channels == expected_channels:
        return source
    if actual_channels == 1 and expected_channels == 2 and adapter.duplicate_mono_source:
        return ValidatedAudioInput(
            samples=np.repeat(source.samples, 2, axis=0),
            sample_rate=source.sample_rate,
            duration_seconds=source.duration_seconds,
        )
    raise ValueError(
        f"{label} must contain exactly {expected_channels} channels for {adapter.pipeline_class}; "
        f"received {actual_channels}. Mono is duplicated deterministically, but multichannel downmixing "
        "requires channel-layout metadata and is not implicit."
    )


def _validated_audio_input(audio: Any, *, label: str, maximum_duration: float) -> ValidatedAudioInput:
    if isinstance(audio, dict) and "sample_rate" in audio:
        raw_sample_rate = audio.get("sample_rate")
        try:
            numeric_sample_rate = float(raw_sample_rate)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{label} sample rate must be a positive integer.") from error
        if (
            isinstance(raw_sample_rate, bool)
            or not isfinite(numeric_sample_rate)
            or not numeric_sample_rate.is_integer()
            or numeric_sample_rate <= 0
        ):
            raise ValueError(f"{label} sample rate must be a positive integer.")
    try:
        # Validate the decoded values before clipping. Clipping first would
        # silently turn +/-infinity into valid-looking full-scale PCM.
        samples, sample_rate = audio_to_numpy(audio, clip=False)
    except AudioInputContractError:
        raise
    except (OSError, TypeError, ValueError, IndexError, OverflowError) as error:
        raise ValueError(f"{label} must be decodable mono or multichannel PCM audio.") from error
    if samples.ndim != 2 or samples.shape[0] <= 0 or samples.shape[1] <= 0:
        raise ValueError(f"{label} must contain at least one channel and one audio frame.")
    if not np.isfinite(samples).all():
        raise ValueError(f"{label} samples must all be finite.")
    if sample_rate <= 0:
        raise ValueError(f"{label} sample rate must be a positive integer.")
    duration = float(samples.shape[-1] / sample_rate)
    if not isfinite(duration) or duration <= 0 or duration > maximum_duration:
        raise ValueError(f"{label} duration must be finite, greater than 0, and at most {maximum_duration:g} seconds.")
    return ValidatedAudioInput(
        samples=np.clip(samples, -1.0, 1.0),
        sample_rate=sample_rate,
        duration_seconds=duration,
    )


def resample_audio_object(audio: dict[str, Any], target_sample_rate: int) -> dict[str, Any]:
    from math import gcd

    from scipy.signal import resample_poly

    source_sample_rate = int(audio.get("sample_rate") or 48000)
    target_sample_rate = int(target_sample_rate)
    if target_sample_rate not in {int(value) for value in AUDIO_SAMPLE_RATE_OPTIONS}:
        supported = ", ".join(AUDIO_SAMPLE_RATE_OPTIONS.values())
        raise ValueError(f"Audio sample rate must be one of: {supported}.")
    if source_sample_rate == target_sample_rate:
        return audio

    samples = np.asarray(audio["samples"], dtype=np.float32)
    divisor = gcd(source_sample_rate, target_sample_rate)
    resampled = resample_poly(
        samples,
        target_sample_rate // divisor,
        source_sample_rate // divisor,
        axis=-1,
    ).astype(np.float32, copy=False)
    return {
        **audio,
        "samples": np.clip(resampled, -1.0, 1.0),
        "sample_rate": target_sample_rate,
        "channels": int(resampled.shape[0]) if resampled.ndim > 1 else 1,
        "duration_seconds": float(resampled.shape[-1] / target_sample_rate),
    }


def _audio_array_to_object(array: np.ndarray, sample_rate: int) -> dict[str, Any]:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2 and array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
        channels = int(array.shape[0])
        duration_samples = int(array.shape[1])
    elif array.ndim == 2:
        array = array.T
        channels = int(array.shape[0])
        duration_samples = int(array.shape[1])
    else:
        array = array.reshape(1, -1)
        channels = 1
        duration_samples = int(array.shape[1])
    return {
        "samples": np.clip(array, -1.0, 1.0),
        "sample_layout": "channels_first",
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration_seconds": float(duration_samples / sample_rate) if sample_rate else 0.0,
    }


def output_to_audio_objects(result: Any, sample_rate: int = 48000) -> list[dict[str, Any]]:
    import torch

    audio = getattr(result, "audios", result)
    if isinstance(audio, dict):
        if "samples" in audio:
            return [audio]
        if "audios" in audio:
            return output_to_audio_objects(audio["audios"], sample_rate)
    if isinstance(audio, torch.Tensor):
        array = audio.detach().float().cpu().numpy()
    else:
        array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 3:
        return [_audio_array_to_object(waveform, sample_rate) for waveform in array]
    return [_audio_array_to_object(array, sample_rate)]


def output_to_audio_object(result: Any, sample_rate: int = 48000) -> dict[str, Any]:
    return output_to_audio_objects(result, sample_rate)[0]


def crop_tail(audio: dict[str, Any], start_seconds: float, duration_seconds: float | None):
    samples = np.asarray(audio["samples"], dtype=np.float32)
    sample_rate = int(audio.get("sample_rate") or 48000)
    start = max(0, int(start_seconds * sample_rate))
    end = (
        start + int(duration_seconds * sample_rate) if duration_seconds and duration_seconds > 0 else samples.shape[-1]
    )
    cropped = samples[..., start : min(end, samples.shape[-1])]
    return {
        **audio,
        "samples": cropped,
        "duration_seconds": float(cropped.shape[-1] / sample_rate) if sample_rate else 0.0,
    }


class LoadPipeline(NodeBase):
    """Load a generic Diffusers audio pipeline."""

    label = "Load Diffusers Audio Pipeline"
    category = "Diffusers Audio"
    resizable = True
    # Mode selects a generation contract but does not change the resident
    # Diffusers object. Revalidate and retag it on every graph invocation.
    cache_ignored_params = frozenset({"mode", "audio_contract"})
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "output",
            "type": "audio_diffusion_pipeline",
            "signal": {
                "direction": "output",
                "origin": "audio_contract",
                "value": DEFAULT_AUDIO_CONTRACT,
            },
        },
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
            "onChange": "update_audio_contract",
        },
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            # Keep this literal so the static registry parser can expose the
            # choices without importing Diffusers or executing this module.
            "options": [
                "AceStepPipeline",
                "StableAudioPipeline",
                "LongCatAudioDiTPipeline",
                "AudioLDM2Pipeline",
            ],
            "default": "AceStepPipeline",
            "fieldOptions": {"noValidation": True},
            "onChange": "update_audio_contract",
        },
        "mode": {
            "label": "Mode",
            "type": "string",
            "options": ["text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"],
            "default": "text_to_audio",
            "fieldOptions": {"noValidation": True},
            "onChange": "update_audio_contract",
        },
        "audio_contract": {
            "label": "Audio Contract",
            "type": "object",
            "default": DEFAULT_AUDIO_CONTRACT,
            "hidden": True,
        },
        "revision": {"label": "Revision", "type": "string", "default": ""},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "bfloat16",
        },
        "device": {"label": "Device", "type": "string", "options": DEVICE_OPTIONS, "default": DEFAULT_DEVICE},
        "auto_offload": {"label": "Auto offload", "type": "bool", "default": True},
        "offload_mode": offload_mode_param(modes=DIRECT_AUDIO_OFFLOAD_MODES),
        "execution_recipe": {
            "label": "Execution Recipe",
            "display": "input",
            "type": "diffusers_execution_recipe",
        },
        "enable_vae_tiling": {"label": "VAE tiling", "type": "bool", "default": True},
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    def update_audio_contract(self, values, ref):
        """Publish the selected generic audio task contract without loading model code."""

        values = values if isinstance(values, dict) else {}
        adapter = _loader_audio_pipeline_adapter(values)
        requested_mode = values.get("mode")
        if not isinstance(requested_mode, str) or not requested_mode or requested_mode != requested_mode.strip():
            raise ValueError("A registered Diffusers audio mode is required.")
        ref_key = ref.get("key") if isinstance(ref, dict) else None
        if requested_mode not in adapter.modes:
            if ref_key != "pipeline_class":
                adapter.contract_for_mode(requested_mode)
            requested_mode = adapter.modes[0]
        model_selection = _resolve_audio_model_selection(adapter, values.get("model_id"))
        signal_value = _audio_contract_signal(adapter, requested_mode, model_selection)
        self.set_field_params(
            "mode",
            {"options": list(adapter.modes), "default": adapter.modes[0], "value": requested_mode},
        )
        field_values = {"model_id": model_selection, "audio_contract": signal_value}
        selected_revision = values.get("revision")
        if model_selection["source"] == "local":
            field_values["revision"] = ""
        else:
            managed_revision = catalog_revision(model_selection["value"])
            if managed_revision is not None:
                field_values["revision"] = managed_revision
            elif ref_key == "model_id":
                # A model field action carries no trustworthy previous
                # repository identity. Never let a newly selected custom Hub
                # repository inherit the commit shown for its predecessor.
                field_values["revision"] = ""
            elif selected_revision not in (None, ""):
                # Class and mode actions preserve an explicit custom pin only
                # after validating that it is still an immutable revision.
                _resolve_audio_loader_revision(
                    model_selection,
                    model_selection["value"],
                    selected_revision,
                )
        self.set_field_value(field_values)
        self.set_field_params(
            "pipeline",
            {
                "signal": {
                    "direction": "output",
                    "origin": "audio_contract",
                    "value": signal_value,
                }
            },
        )

    def __call__(self, **kwargs):
        adapter = _loader_audio_pipeline_adapter(kwargs)
        mode = _loader_audio_mode(kwargs, adapter)
        values = dict(kwargs)
        values["pipeline_class"] = adapter.pipeline_class
        values["mode"] = mode
        values["model_id"] = _resolve_audio_model_selection(adapter, values.get("model_id"))
        model_id = repo_value(values["model_id"])
        values["revision"] = _resolve_audio_loader_revision(
            values["model_id"],
            model_id,
            values.get("revision"),
        )
        result = super().__call__(**values)
        pipeline = result.get("pipeline") if isinstance(result, dict) else None
        if pipeline is not None:
            self._tag_pipeline(pipeline, adapter, mode, model_id, values["revision"])
        return result

    @staticmethod
    def _tag_pipeline(
        pipeline: Any,
        adapter: AudioPipelineAdapter,
        mode: str,
        repository: str,
        revision: str | None,
    ):
        setattr(pipeline, "_modiff_audio_pipeline_class", adapter.pipeline_class)
        setattr(pipeline, "_modiff_audio_mode", mode)
        setattr(pipeline, "_modiff_audio_repo", repository or adapter.default_repo)
        setattr(pipeline, "_modiff_audio_revision", revision)

    def execute(self, **kwargs):
        adapter = _loader_audio_pipeline_adapter(kwargs)
        pipeline_class_name = adapter.pipeline_class
        mode = _loader_audio_mode(kwargs, adapter)
        model_selection = _resolve_audio_model_selection(adapter, kwargs.get("model_id"))
        model_id = repo_value(model_selection)

        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        pipeline_class = pipeline_class_from_name(pipeline_class_name)
        dtype = str_to_dtype(kwargs.get("dtype") or "bfloat16")
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode=OFFLOAD_MODE_MODEL_CPU,
            direct_device_load=True,
        )
        revision = _resolve_audio_loader_revision(model_selection, model_id, kwargs.get("revision"))

        load_kwargs = {
            "torch_dtype": dtype,
            "revision": revision,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "local_files_only": local_files_only(model_id),
            **recipe_load_kwargs,
        }
        if adapter.safe_serialization_required:
            load_kwargs["use_safetensors"] = True

        self.progress(-1, phase="loading", message=f"Loading {pipeline_class_name}")
        with self.diffusers_loading_progress():
            pipeline = pipeline_class.from_pretrained(model_id, **load_kwargs)
        self._tag_pipeline(pipeline, adapter, mode, model_id, revision)
        if recipe:
            apply_execution_recipe_to_pipeline(pipeline, recipe)
        elif kwargs.get("enable_vae_tiling", True):
            vae = getattr(pipeline, "vae", None)
            for method_name in ("enable_tiling", "enable_vae_tiling"):
                method = getattr(vae or pipeline, method_name, None)
                if callable(method):
                    method()
                    break

        self.progress(99, phase="component_placement", message=f"Applying {offload_mode} offload")
        apply_pipeline_offload(
            pipeline,
            mode=offload_mode,
            device=device,
            node_id=self.node_id,
            scope="diffusers-audio",
            prefer_pipeline_group=True,
        )
        self.mm_add(pipeline, priority=2)
        return {"pipeline": pipeline, "resolved_artifact": model_id}


class LoadAdapter(NodeBase):
    """Load a safetensors-only ACE-Step LoRA from managed cache or a local folder."""

    label = "Load Diffusers Audio LoRA"
    category = "Diffusers Audio"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "audio_diffusion_pipeline", "required": True},
        "adapter_path": {
            "label": "LoRA",
            "display": "modelselect",
            "type": "string",
            "required": True,
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "weight_name": {
            "label": "Weight name",
            "type": "string",
            "default": "adapter_model.safetensors",
            "description": "Must end with the literal lowercase .safetensors suffix.",
        },
        "revision": {
            "label": "Revision",
            "type": "string",
            "default": "",
            "description": "Required immutable 40-character commit SHA for Hub LoRAs.",
        },
        "expected_sha256": {
            "label": "Expected SHA-256",
            "type": "string",
            "default": "",
            "description": "Required content digest for Hub LoRA weights.",
        },
        "adapter_name": {"label": "Adapter name", "type": "string", "default": "audio_style"},
        "replace_existing": {"label": "Replace existing adapters", "type": "bool", "default": True},
        "scale": {
            "label": "Strength",
            "display": "slider",
            "type": "float",
            "default": 0.7,
            "min": 0,
            "max": 2,
            "step": 0.05,
        },
        "output": {"label": "Pipeline", "display": "output", "type": "audio_diffusion_pipeline"},
    }

    @staticmethod
    def _selection(value: Any) -> dict[str, str]:
        if isinstance(value, dict):
            source = _canonical_model_source(value.get("source"), label="Audio LoRA")
            raw_adapter_path = value.get("value")
            if not isinstance(raw_adapter_path, str):
                raise ValueError("Audio LoRA value must be a repository ID or local path string.")
            adapter_path = raw_adapter_path.strip()
        elif isinstance(value, str):
            source = "local"
            adapter_path = value.strip()
        else:
            raise ValueError("Audio LoRA selection must be a local path or a hub/local selection object.")
        if not adapter_path:
            raise ValueError("Audio LoRA repository ID or local path is required.")
        if source == "hub":
            adapter_path = _validated_hub_repository(adapter_path, label="Audio LoRA repository")
        else:
            try:
                local_target = Path(adapter_path).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise FileNotFoundError(f"Audio LoRA path does not exist: {adapter_path}") from error
            if not (local_target.is_file() or local_target.is_dir()):
                raise ValueError("Audio LoRA local target must be a file or directory.")
            adapter_path = str(local_target)
        return {"source": source, "value": adapter_path}

    @staticmethod
    def _weight_name(selection: dict[str, str], value: Any) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            weight_name = "adapter_model.safetensors"
        elif isinstance(value, str):
            weight_name = value.strip()
        else:
            raise ValueError("Audio LoRA weight name must be a string.")

        if selection["source"] == "hub":
            weight_name = _validated_hub_filename(weight_name, label="Audio LoRA weight name")
        else:
            selected_path = Path(selection["value"])
            if selected_path.is_file():
                weight_name = selected_path.name
        return _require_lowercase_safetensors_filename(weight_name, label="Audio LoRA weight name")

    def __call__(self, **kwargs):
        values = dict(kwargs)
        # NodeBase normally interprets legacy plain modelselect strings as the
        # first declared source (Hub). Audio LoRA plain strings have always
        # meant local paths, so canonicalize before NodeBase sees the value.
        values["adapter_path"] = self._selection(values.get("adapter_path"))
        values["weight_name"] = self._weight_name(values["adapter_path"], values.get("weight_name"))
        if values["adapter_path"]["source"] == "hub":
            values["revision"] = _resolve_audio_loader_revision(
                values["adapter_path"],
                values["adapter_path"]["value"],
                values.get("revision"),
            )
            values["expected_sha256"] = _required_sha256(
                values.get("expected_sha256"),
                label="Hub audio LoRA expected SHA-256",
            )
        else:
            values["revision"] = None
        return super().__call__(**values)

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Load Audio LoRA needs a pipeline input.")
        if getattr(pipeline, "_modiff_audio_pipeline_class", None) != "AceStepPipeline":
            raise ValueError("Audio LoRA loading is currently supported only for AceStepPipeline.")
        if not callable(getattr(pipeline, "load_lora_weights", None)):
            raise RuntimeError(
                "This Diffusers revision does not expose ACE-Step LoRA support. Install MoDiff's pinned dependencies."
            )

        selection = self._selection(kwargs.get("adapter_path"))
        source = selection["source"]
        adapter_path = selection["value"]

        weight_name = self._weight_name(selection, kwargs.get("weight_name"))
        if source == "hub":
            from utils.huggingface import cached_file_path, resolve_managed_hf_cache_file

            revision = _resolve_audio_loader_revision(selection, adapter_path, kwargs.get("revision"))
            expected = _required_sha256(
                kwargs.get("expected_sha256"),
                label="Hub audio LoRA expected SHA-256",
            )
            cached = cached_file_path(adapter_path, weight_name, revision=revision)
            if not cached:
                raise FileNotFoundError(
                    f"Audio LoRA {adapter_path}@{revision}/{weight_name} is not installed. "
                    "Install that exact revision through Model Manager first."
                )
            cached_alias = Path(cached).expanduser()
            _require_lowercase_safetensors_filename(
                cached_alias.name,
                label="Installed Hub audio LoRA cache entry",
            )
            cached_path = resolve_managed_hf_cache_file(cached)
            digest = hashlib.sha256()
            with cached_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError("The audio LoRA failed its pinned SHA-256 verification. Repair it in Model Manager.")
            # Keep the reviewed snapshot alias: its literal lowercase suffix
            # makes pinned Diffusers select safetensors. The resolved target is
            # used only for cache containment and digest verification because
            # normal Hub aliases may resolve to extensionless blob names.
            adapter_path = str(cached_alias.parent)
            weight_name = cached_alias.name
        else:
            try:
                local_target = Path(adapter_path).expanduser().resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise FileNotFoundError(f"Audio LoRA path does not exist: {adapter_path}") from error
            if local_target.is_file():
                adapter_path = str(local_target.parent)
                weight_name = local_target.name
            elif local_target.is_dir():
                requested_weight = Path(weight_name)
                if requested_weight.is_absolute():
                    raise ValueError("Audio LoRA weight name must stay inside the selected local folder.")
                try:
                    local_weight = (local_target / requested_weight).resolve(strict=True)
                    local_weight.relative_to(local_target)
                except (OSError, RuntimeError, ValueError) as error:
                    raise FileNotFoundError(
                        f"Audio LoRA weight does not exist inside the selected folder: {weight_name}"
                    ) from error
                if not local_weight.is_file():
                    raise FileNotFoundError("The selected local audio LoRA weight is not a file.")
                adapter_path = str(local_weight.parent)
                weight_name = local_weight.name
            else:
                raise FileNotFoundError(f"Audio LoRA target is not a file or folder: {local_target}")
            _require_lowercase_safetensors_filename(weight_name, label="Local audio LoRA weight name")

        adapter_name = str(kwargs.get("adapter_name") or "audio_style").strip()
        if not adapter_name:
            raise ValueError("Audio LoRA adapter name is required.")
        scale = _bounded_float(
            kwargs.get("scale"),
            default=0.7,
            label="Audio LoRA strength",
            minimum=0,
            maximum=2,
        )
        if kwargs.get("replace_existing", True) and callable(getattr(pipeline, "unload_lora_weights", None)):
            pipeline.unload_lora_weights()
        pipeline.load_lora_weights(
            adapter_path,
            weight_name=weight_name,
            adapter_name=adapter_name,
            use_safetensors=True,
        )
        if callable(getattr(pipeline, "set_adapters", None)):
            pipeline.set_adapters([adapter_name], [scale])
        return {"output": pipeline}


class SetAdapters(NodeBase):
    """Select and blend already loaded ACE-Step LoRA adapters."""

    label = "Set Audio LoRA Blend"
    category = "Diffusers Audio"
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": "audio_diffusion_pipeline",
            "required": True,
        },
        "adapter_names": {"label": "Adapter names", "type": "string", "default": "audio_style"},
        "adapter_weights": {"label": "Weights", "type": "string", "default": "0.7"},
        "output": {"label": "Pipeline", "display": "output", "type": "audio_diffusion_pipeline"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Set Audio LoRA Blend needs a pipeline input.")
        names = [part.strip() for part in str(kwargs.get("adapter_names") or "").split(",") if part.strip()]
        try:
            weights = [
                _bounded_float(
                    part.strip(),
                    default=0.7,
                    label="Audio LoRA weight",
                    minimum=0,
                    maximum=2,
                )
                for part in str(kwargs.get("adapter_weights") or "").split(",")
                if part.strip()
            ]
        except ValueError as exc:
            raise ValueError("Audio LoRA weights must be comma-separated finite values from 0 through 2.") from exc
        if not names or len(names) != len(weights):
            raise ValueError("Audio LoRA adapter names and weights must contain the same number of entries.")
        setter = getattr(pipeline, "set_adapters", None)
        if not callable(setter):
            raise RuntimeError("This pipeline does not support selecting LoRA adapters.")
        setter(names, weights)
        return {"output": pipeline}


class FuseAdapters(NodeBase):
    """Optionally fuse active ACE-Step LoRAs for lower per-step overhead."""

    label = "Fuse Audio LoRA"
    category = "Diffusers Audio"
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": "audio_diffusion_pipeline",
            "required": True,
        },
        "enabled": {"label": "Fuse", "type": "bool", "default": True},
        "safe_fusing": {"label": "Safe fusing", "type": "bool", "default": True},
        "output": {"label": "Pipeline", "display": "output", "type": "audio_diffusion_pipeline"},
    }

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Fuse Audio LoRA needs a pipeline input.")
        method_name = "fuse_lora" if kwargs.get("enabled", True) else "unfuse_lora"
        method = getattr(pipeline, method_name, None)
        if not callable(method):
            raise RuntimeError(f"This pipeline does not support {method_name}().")
        if method_name == "fuse_lora":
            try:
                method(safe_fusing=bool(kwargs.get("safe_fusing", True)))
            except TypeError:
                method()
        else:
            method()
        return {"output": pipeline}


def _has_audio_input(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and value.strip() == "")


def _pipeline_audio_contract(pipeline: Any) -> tuple[AudioPipelineAdapter, AudioModeContract]:
    runtime_class = type(pipeline).__name__
    runtime_candidates = [
        adapter for adapter in AUDIO_PIPELINE_ADAPTERS.values() if adapter.pipeline_class == runtime_class
    ]
    pipeline_class_name = getattr(pipeline, "_modiff_audio_pipeline_class", None)
    mode = getattr(pipeline, "_modiff_audio_mode", None)
    if pipeline_class_name is None:
        if runtime_class not in AUDIO_PIPELINE_ADAPTERS:
            raise ValueError(
                "The audio pipeline has no MoDiff class/mode identity and is not an exact supported Diffusers "
                "audio pipeline class. Reload it with Load Diffusers Audio Pipeline."
            )
        pipeline_class_name = runtime_class

    adapter = get_audio_pipeline_adapter(pipeline_class_name)
    if runtime_candidates and adapter not in runtime_candidates:
        runtime_names = ", ".join(candidate.pipeline_class for candidate in runtime_candidates)
        raise ValueError(
            f"Diffusers audio pipeline identity is inconsistent: runtime class {runtime_class} supports "
            f"{runtime_names}, but the pipeline is tagged as {adapter.pipeline_class}."
        )
    tagged_repo = getattr(pipeline, "_modiff_audio_repo", None)
    if isinstance(tagged_repo, str) and tagged_repo.strip():
        repository_key = tagged_repo.strip().casefold()
        managed_repo_adapters = [
            candidate
            for candidate in AUDIO_PIPELINE_ADAPTERS.values()
            if candidate.default_repo.casefold() == repository_key
        ]
        if managed_repo_adapters and adapter not in managed_repo_adapters:
            repository_names = ", ".join(candidate.pipeline_class for candidate in managed_repo_adapters)
            raise ValueError(
                "Diffusers audio pipeline identity is inconsistent: managed repository "
                f"{tagged_repo.strip()!r} supports {repository_names}, but the pipeline is tagged as "
                f"{adapter.pipeline_class}."
            )
    if mode is None:
        if len(adapter.mode_contracts) != 1:
            raise ValueError(
                f"Untagged {adapter.pipeline_class} is ambiguous across modes: {', '.join(adapter.modes)}. "
                "Reload it with an explicit audio mode."
            )
        mode = adapter.modes[0]
    contract = adapter.contract_for_mode(str(mode))
    return adapter, contract


def _finite_positive_bound(value: Any, *, default: float, label: str, maximum: float | None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number greater than 0.")
    try:
        number = float(value_or_default(value, default))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number greater than 0.") from error
    if not isfinite(number) or number <= 0 or (maximum is not None and number > maximum):
        maximum_message = f" and at most {maximum:g} seconds" if maximum is not None else ""
        raise ValueError(f"{label} must be finite and greater than 0{maximum_message}.")
    return number


@dataclass(frozen=True)
class AudioInvocation:
    adapter: AudioPipelineAdapter
    contract: AudioModeContract
    duration_seconds: float
    extension_seconds: float | None
    repaint_interval: tuple[float, float] | None
    source: ValidatedAudioInput | None
    controls: dict[str, int | float | None]


def _preflight_audio_invocation(pipeline: Any, kwargs: dict[str, Any]) -> AudioInvocation:
    adapter, contract = _pipeline_audio_contract(pipeline)
    if "task_type" not in kwargs:
        task_type = contract.task_type
    else:
        requested_task = kwargs.get("task_type")
        if not isinstance(requested_task, str) or not requested_task or requested_task != requested_task.strip():
            raise ValueError("Audio task must be an exact nonblank supported task string.")
        task_type = requested_task
    if task_type in STALE_ACE_TASK_TYPES:
        raise ValueError(
            f"ACE-Step task {task_type!r} is recognized but hidden until MoDiff provides its dedicated "
            "complete input mode. Rebuild the graph with a supported audio mode."
        )
    if task_type != contract.task_type:
        raise ValueError(
            f"Audio mode {contract.mode} requires task {contract.task_type}; received {task_type}. "
            "Refresh or rebuild the stale graph contract."
        )

    source_audio = kwargs.get("source_audio")
    reference_audio = kwargs.get("reference_audio")
    has_source = _has_audio_input(source_audio)
    has_reference = _has_audio_input(reference_audio)
    if contract.source_audio == "required" and not has_source:
        raise ValueError(f"Audio mode {contract.mode} requires source audio.")
    if contract.source_audio == "forbidden" and has_source:
        raise ValueError(f"Audio mode {contract.mode} does not accept source audio.")
    if contract.reference_audio == "required" and not has_reference:
        raise ValueError(f"Audio mode {contract.mode} requires reference audio.")
    if contract.reference_audio == "forbidden" and has_reference:
        raise ValueError(f"Audio mode {contract.mode} does not accept reference audio.")

    requested_sample_rate = _bounded_int(
        kwargs.get("sample_rate"),
        default=48000,
        label="Audio sample rate",
        minimum=min(int(value) for value in AUDIO_SAMPLE_RATE_OPTIONS),
        maximum=max(int(value) for value in AUDIO_SAMPLE_RATE_OPTIONS),
    )
    if requested_sample_rate not in {int(value) for value in AUDIO_SAMPLE_RATE_OPTIONS}:
        supported = ", ".join(AUDIO_SAMPLE_RATE_OPTIONS.values())
        raise ValueError(f"Audio sample rate must be one of: {supported}.")

    source = None
    if has_source:
        source = _validated_audio_input(
            source_audio,
            label=f"Audio mode {contract.mode} source",
            maximum_duration=contract.max_duration_seconds or ACE_MAX_DURATION_SECONDS,
        )
        source = _normalize_source_audio_channels(
            source,
            adapter=adapter,
            label=f"Audio mode {contract.mode} source",
        )

    extension_duration = None
    if contract.mode == "audio_continuation":
        extension_duration = _finite_positive_bound(
            kwargs.get("extension_duration"),
            default=ACE_CONTINUATION_DEFAULT_EXTENSION_SECONDS,
            label="ACE-Step continuation extension",
            maximum=contract.max_extension_seconds,
        )

    repaint_interval = None
    if contract.validate_repaint_interval:
        repaint_start = _bounded_float(
            kwargs.get("repainting_start"),
            default=0.0,
            label="Audio repaint start",
            minimum=0,
            maximum=contract.max_duration_seconds or ACE_MAX_DURATION_SECONDS,
        )
        repaint_end = _bounded_float(
            kwargs.get("repainting_end"),
            default=0.0,
            label="Audio repaint end",
            minimum=0,
            maximum=contract.max_duration_seconds or ACE_MAX_DURATION_SECONDS,
        )
        if repaint_end <= repaint_start:
            raise ValueError("Audio repaint requires a non-negative start and an end strictly greater than the start.")
        if source is not None and repaint_end > source.duration_seconds:
            raise ValueError(
                f"Audio repaint end {repaint_end:g}s exceeds the {source.duration_seconds:g}s source duration."
            )
        repaint_interval = (repaint_start, repaint_end)

    if contract.mode == "audio_continuation":
        duration = source.duration_seconds + extension_duration
        if contract.max_duration_seconds is not None and duration > contract.max_duration_seconds:
            raise ValueError(
                "ACE-Step continuation source plus extension must be at most "
                f"{contract.max_duration_seconds:g} seconds."
            )
    elif contract.mode == "audio_repaint":
        duration = source.duration_seconds
    else:
        duration_label = f"{adapter.pipeline_class} duration"
        duration = _finite_positive_bound(
            kwargs.get("audio_duration"),
            default=30.0,
            label=duration_label,
            maximum=contract.max_duration_seconds,
        )

    seed = _bounded_int(
        kwargs.get("seed"),
        default=0,
        label="Audio seed",
        minimum=0,
        maximum=4294967295,
    )
    if adapter.generation_kind != "ace_step":
        control_label = {
            "stable_audio": "Stable Audio",
            "longcat_audio_dit": "LongCat AudioDiT",
            "audioldm2": "AudioLDM2",
        }[adapter.generation_kind]
        controls: dict[str, int | float | None] = {
            "seed": seed,
            "steps": _bounded_int(
                kwargs.get("stable_audio_steps"),
                default=adapter.default_inference_steps,
                label=f"{control_label} steps",
                minimum=1,
                maximum=adapter.max_inference_steps,
            ),
            "guidance": _bounded_float(
                kwargs.get("stable_audio_guidance"),
                default=adapter.default_guidance_scale,
                label=f"{control_label} guidance",
                minimum=0,
                maximum=20,
            ),
            "waveforms": _bounded_int(
                kwargs.get("num_waveforms"),
                default=1,
                label=f"{control_label} variations",
                minimum=1,
                maximum=8 if adapter.supports_multiple_waveforms else 1,
            ),
        }
    else:
        raw_bpm = kwargs.get("bpm")
        bpm = None
        if raw_bpm is not None and not (isinstance(raw_bpm, str) and not raw_bpm.strip()):
            bpm_value = _bounded_int(
                raw_bpm,
                default=0,
                label="ACE-Step BPM",
                minimum=0,
                maximum=400,
            )
            bpm = bpm_value or None
        controls = {
            "seed": seed,
            "steps": _bounded_int(
                kwargs.get("num_inference_steps"),
                default=8,
                label="ACE-Step inference steps",
                minimum=1,
                maximum=100,
            ),
            "guidance": _bounded_float(
                kwargs.get("guidance_scale"),
                default=1,
                label="ACE-Step guidance",
                minimum=0,
                maximum=20,
            ),
            "shift": _bounded_float(
                kwargs.get("shift"),
                default=3,
                label="ACE-Step shift",
                minimum=0,
                maximum=10,
                minimum_inclusive=False,
            ),
            "lora_scale": _bounded_float(
                kwargs.get("lora_scale"),
                default=1,
                label="ACE-Step LoRA call strength",
                minimum=0,
                maximum=2,
            ),
            "cover_strength": (
                _bounded_float(
                    kwargs.get("audio_cover_strength"),
                    default=0.85,
                    label="ACE-Step cover strength",
                    minimum=0,
                    maximum=1,
                )
                if contract.mode == "audio_variation"
                else None
            ),
            "bpm": bpm,
        }
    controls["sample_rate"] = requested_sample_rate
    return AudioInvocation(
        adapter=adapter,
        contract=contract,
        duration_seconds=duration,
        extension_seconds=extension_duration,
        repaint_interval=repaint_interval,
        source=source,
        controls=controls,
    )


class Generate(NodeBase):
    """Generate audio with a Diffusers audio pipeline."""

    label = "Diffusers Audio Generate"
    category = "Diffusers Audio"
    resizable = True
    params = {
        "pipeline": {
            "label": "Pipeline",
            "display": "input",
            "type": "audio_diffusion_pipeline",
            "required": True,
            "onSignal": [
                {"action": "value", "target": "audio_contract"},
                {"action": "exec", "data": "update_audio_contract"},
            ],
        },
        "audio_contract": {
            "label": "Audio Contract",
            "type": "object",
            "default": DEFAULT_AUDIO_CONTRACT,
            "hidden": True,
        },
        "task_type": {
            "label": "Task",
            "type": "string",
            "options": ["text2music"],
            "default": "text2music",
            "fieldOptions": {"noValidation": True},
        },
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {
            "label": "Negative Prompt",
            "display": "textarea",
            "type": "text",
            "default": "",
            "hidden": True,
        },
        "lyrics": {"label": "Lyrics", "display": "textarea", "type": "text", "default": ""},
        "audio_duration": {
            "label": "Duration",
            "type": "float",
            "default": 30.0,
            "min": 1,
            "max": ACE_MAX_DURATION_SECONDS,
            "step": 0.5,
        },
        "extension_duration": {
            "label": "Extension",
            "type": "float",
            "default": ACE_CONTINUATION_DEFAULT_EXTENSION_SECONDS,
            "min": 1,
            "max": ACE_CONTINUATION_MAX_EXTENSION_SECONDS,
            "step": 0.5,
            "hidden": True,
        },
        "vocal_language": {"label": "Language", "type": "string", "default": "en"},
        "num_inference_steps": {
            "label": "Steps",
            "display": "slider",
            "type": "int",
            "default": 8,
            "min": 1,
            "max": 100,
            "description": "ACE-Step v1.5 XL Turbo is designed for 8 denoising steps.",
        },
        "guidance_scale": {
            "label": "Guidance",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": 0,
            "max": 20,
            "step": 0.1,
            "description": "XL Turbo is guidance-distilled; values above 1 are ignored by the Diffusers pipeline.",
        },
        "lora_scale": {
            "label": "LoRA call strength",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": 0,
            "max": 2,
            "step": 0.05,
            # Retained for backward compatibility with saved graphs. Diffusers
            # applies this per-call multiplier on top of the active adapter
            # weight. Managed ACE-Step graphs keep it at 1.0 and expose the
            # loader Strength as their single, non-duplicated control.
            "hidden": True,
            "description": (
                "Advanced per-call multiplier for active adapters. Managed ACE-Step graphs use the LoRA loader "
                "Strength and keep this multiplier at 1.0."
            ),
        },
        "shift": {
            "label": "Shift",
            "display": "slider",
            "type": "float",
            "default": 3.0,
            "min": 0.1,
            "max": 10,
            "step": 0.1,
        },
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "bpm": {"label": "BPM", "type": "int", "default": 0, "min": 0, "max": 400},
        "keyscale": {"label": "Key", "type": "string", "default": ""},
        "timesignature": {"label": "Time", "type": "string", "default": "4"},
        "source_audio": {
            "label": "Source Audio",
            "display": "input",
            "type": ["audio", "str"],
            "required": False,
            "hidden": True,
        },
        "reference_audio": {
            "label": "Reference Audio",
            "display": "input",
            "type": ["audio", "str"],
            "required": False,
            "hidden": True,
        },
        "repainting_start": {
            "label": "Repaint Start",
            "type": "float",
            "default": 0.0,
            "min": 0,
            "step": 0.01,
            "hidden": True,
        },
        "repainting_end": {
            "label": "Repaint End",
            "type": "float",
            "default": 0.0,
            "min": 0,
            "step": 0.01,
            "hidden": True,
        },
        "audio_cover_strength": {
            "label": "Cover Strength",
            "display": "slider",
            "type": "float",
            "default": 0.85,
            "min": 0,
            "max": 1,
            "step": 0.01,
            "hidden": True,
        },
        "return_continuation_tail": {
            "label": "Return Tail Only",
            "type": "bool",
            "default": True,
            "hidden": True,
        },
        "sample_rate": {
            "label": "Sample Rate",
            "type": "int",
            "default": 48000,
            "options": AUDIO_SAMPLE_RATE_OPTIONS,
        },
        "stable_audio_steps": {
            "label": "Stable Audio Steps",
            "type": "int",
            "default": 100,
            "min": 1,
            "max": 300,
            "hidden": True,
            "description": "Denoising steps for reviewed standard Diffusers audio pipelines; ignored by ACE-Step.",
        },
        "stable_audio_guidance": {
            "label": "Stable Audio Guidance",
            "type": "float",
            "default": 7,
            "min": 0,
            "max": 20,
            "hidden": True,
            "description": "Classifier-free guidance for reviewed standard Diffusers audio pipelines; ignored by ACE-Step.",
        },
        "num_waveforms": {
            "label": "Variations",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 8,
            "hidden": True,
            "description": "Number of generated waveforms when supported; ignored by ACE-Step.",
        },
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "audio_variations": {"label": "Audio Variations", "display": "output", "type": "collection"},
        "sample_rate_out": {"label": "Sample Rate", "display": "output", "type": "int"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def __call__(self, **kwargs):
        if "task_type" in kwargs:
            task_type = kwargs.get("task_type")
            if not isinstance(task_type, str) or not task_type or task_type != task_type.strip():
                raise ValueError("Audio task must be an exact nonblank supported task string.")
        for field in ("seed", "bpm", "num_inference_steps", "stable_audio_steps", "num_waveforms", "sample_rate"):
            value = kwargs.get(field)
            if (
                field not in kwargs
                or value is None
                or (field == "bpm" and isinstance(value, str) and not value.strip())
            ):
                continue
            if isinstance(value, bool):
                raise ValueError(f"{field} must be an exact finite integer.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"{field} must be an exact finite integer.") from error
            if not isfinite(numeric_value) or not numeric_value.is_integer():
                raise ValueError(f"{field} must be an exact finite integer.")
        return super().__call__(**kwargs)

    def update_audio_contract(self, values, ref):
        """Apply the loader's backend-owned contract to this generic audio form."""

        values = values if isinstance(values, dict) else {}
        signal_value = values.get("audio_contract")
        if not isinstance(signal_value, dict):
            raise ValueError("The connected audio pipeline did not publish a valid task contract.")
        adapter = get_audio_pipeline_adapter(signal_value.get("pipelineClass"))
        contract = adapter.contract_for_mode(str(signal_value.get("mode") or ""))
        expected_signal = contract.signal_value(
            adapter.pipeline_class,
            str(signal_value.get("repository") or adapter.default_repo),
        )
        if signal_value != expected_signal:
            raise ValueError("The connected audio pipeline published a stale or mismatched task contract.")

        for field, params in expected_signal["fieldParams"].items():
            self.set_field_params(field, params)

    def execute(self, **kwargs):
        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Diffusers audio pipeline is required.")
        invocation = _preflight_audio_invocation(pipeline, kwargs)
        adapter = invocation.adapter
        contract = invocation.contract

        if adapter.generation_kind != "ace_step":
            return self._execute_standard_diffusers_audio(pipeline, kwargs, invocation)

        import torch

        requested_sample_rate = int(invocation.controls["sample_rate"])
        # The decoded tensor is produced at the VAE's native rate. Labeling it
        # with a different UI/export rate changes its duration and can truncate
        # valid samples, so generation stays at the pipeline's native rate and
        # the completed audio is resampled to the requested delivery rate.
        sample_rate = int(getattr(pipeline, "sample_rate", None) or 48000)
        task_type = contract.task_type
        source = invocation.source
        source_duration = source.duration_seconds if source is not None else 0.0

        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(int(invocation.controls["seed"]))
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(int(invocation.controls["seed"]))

        call_task_type = contract.upstream_task_type
        audio_duration = invocation.duration_seconds

        call_kwargs = {
            "prompt": str(kwargs.get("prompt") or ""),
            "lyrics": str(kwargs.get("lyrics") or ""),
            "audio_duration": audio_duration,
            "vocal_language": str(kwargs.get("vocal_language") or "en"),
            "num_inference_steps": int(invocation.controls["steps"]),
            "guidance_scale": float(invocation.controls["guidance"]),
            "shift": float(invocation.controls["shift"]),
            "generator": generator,
            "output_type": "pt",
            "return_dict": True,
            "task_type": call_task_type,
        }
        if supports_arg(pipeline, "attention_kwargs"):
            call_kwargs["attention_kwargs"] = {"scale": float(invocation.controls["lora_scale"])}
        for optional in ("keyscale", "timesignature"):
            value = none_if_blank(kwargs.get(optional))
            if value is not None and supports_arg(pipeline, optional):
                call_kwargs[optional] = value
        if invocation.controls["bpm"] is not None and supports_arg(pipeline, "bpm"):
            call_kwargs["bpm"] = int(invocation.controls["bpm"])

        if source is not None:
            tensor = _audio_array_to_tensor(
                source.samples,
                source.sample_rate,
                device=device,
                target_sample_rate=sample_rate,
            )
            if task_type == "continuation":
                target_samples = int(round(audio_duration * sample_rate))
                if tensor.shape[-1] < target_samples:
                    tensor = torch.nn.functional.pad(tensor, (0, target_samples - tensor.shape[-1]))
            # P0 exposes one variation input. Route that required source as the
            # sole upstream timbre/style reference: src_audio cover conditioning
            # needs optional tokenizer modules that the reviewed artifact does
            # not publish. A separate reference input remains forbidden until a
            # true two-input adapter has an explicit merge contract.
            if task_type == "cover":
                if not supports_arg(pipeline, "reference_audio"):
                    raise ValueError(
                        "The selected ACE-Step pipeline cannot accept the required variation source as "
                        "reference_audio."
                    )
                call_kwargs["reference_audio"] = tensor
            else:
                if not supports_arg(pipeline, "src_audio"):
                    raise ValueError(
                        f"The selected ACE-Step pipeline cannot accept required {contract.mode} source audio."
                    )
                call_kwargs["src_audio"] = tensor

        if task_type in ("repaint", "continuation"):
            start, end = invocation.repaint_interval or (0.0, 0.0)
            if task_type == "continuation":
                start = source_duration
                end = audio_duration
            if supports_arg(pipeline, "repainting_start"):
                call_kwargs["repainting_start"] = start
            if supports_arg(pipeline, "repainting_end"):
                call_kwargs["repainting_end"] = end
        if task_type == "cover" and supports_arg(pipeline, "audio_cover_strength"):
            call_kwargs["audio_cover_strength"] = float(invocation.controls["cover_strength"])

        if supports_arg(pipeline, "callback_on_step_end"):
            call_kwargs["callback_on_step_end"] = self.pipe_callback
            if supports_arg(pipeline, "callback_on_step_end_tensor_inputs"):
                call_kwargs["callback_on_step_end_tensor_inputs"] = []

        self.progress(
            -1,
            phase="denoising",
            message=f"Generating audio ({task_type})",
            current_step=0,
            total_steps=call_kwargs["num_inference_steps"],
        )
        result = pipeline(**call_kwargs)
        audio = output_to_audio_object(result, sample_rate=sample_rate)
        if task_type == "continuation" and kwargs.get("return_continuation_tail", True):
            audio = crop_tail(audio, source_duration, invocation.extension_seconds)
        else:
            # Diffusion audio decoders may emit a frame-aligned tail beyond the
            # requested duration. Keep the shared node contract exact without
            # padding outputs that are genuinely shorter.
            audio = crop_tail(audio, 0.0, audio_duration)
        audio = resample_audio_object(audio, requested_sample_rate)

        return {
            "audio": audio,
            "audio_variations": [audio],
            "sample_rate_out": int(audio.get("sample_rate") or requested_sample_rate),
            "duration_seconds": float(audio.get("duration_seconds") or 0.0),
        }

    def _execute_standard_diffusers_audio(self, pipeline, kwargs, invocation: AudioInvocation):
        import torch

        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(int(invocation.controls["seed"]))
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(int(invocation.controls["seed"]))

        def callback(step, timestep, latents):
            if not hasattr(pipeline, "_num_timesteps"):
                pipeline._num_timesteps = int(invocation.controls["steps"])
            self.pipe_callback(pipeline, step, timestep, {"latents": latents})

        common_kwargs = {
            "prompt": str(kwargs.get("prompt") or ""),
            "negative_prompt": none_if_blank(kwargs.get("negative_prompt")),
            "num_inference_steps": int(invocation.controls["steps"]),
            "guidance_scale": float(invocation.controls["guidance"]),
            "generator": generator,
            "output_type": "pt",
            "return_dict": True,
        }
        if invocation.adapter.generation_kind == "stable_audio":
            result = pipeline(
                **common_kwargs,
                audio_start_in_s=0,
                audio_end_in_s=invocation.duration_seconds,
                num_waveforms_per_prompt=int(invocation.controls["waveforms"]),
                callback=callback,
                callback_steps=1,
            )
            vae = getattr(pipeline, "vae", None)
            vae_config = getattr(vae, "config", None)
            configured_rate = vae_config.get("sampling_rate") if hasattr(vae_config, "get") else None
            sample_rate = int(getattr(vae, "sampling_rate", None) or configured_rate or 44100)
            audio_variations = output_to_audio_objects(result, sample_rate)
        elif invocation.adapter.generation_kind == "longcat_audio_dit":
            result = pipeline(
                **common_kwargs,
                audio_duration_s=invocation.duration_seconds,
                callback_on_step_end=self.pipe_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
            sample_rate = int(getattr(pipeline, "sample_rate", None) or 24000)
            audio_variations = output_to_audio_objects(result, sample_rate)
        elif invocation.adapter.generation_kind == "audioldm2":
            result = pipeline(
                **common_kwargs,
                audio_length_in_s=invocation.duration_seconds,
                num_waveforms_per_prompt=int(invocation.controls["waveforms"]),
                callback=callback,
                callback_steps=1,
            )
            vocoder = getattr(pipeline, "vocoder", None)
            vocoder_config = getattr(vocoder, "config", None)
            configured_rate = (
                vocoder_config.get("sampling_rate") if hasattr(vocoder_config, "get") else None
            )
            sample_rate = int(configured_rate or 16000)
            raw_audio = getattr(result, "audios", result)
            if _is_torch_audio_tensor(raw_audio):
                raw_audio = raw_audio.detach().float().cpu().numpy()
            raw_array = np.asarray(raw_audio, dtype=np.float32)
            if raw_array.ndim == 1:
                raw_array = raw_array[None, :]
            if raw_array.ndim != 2:
                raise ValueError("AudioLDM2 returned an unsupported audio tensor shape.")
            audio_variations = [_audio_array_to_object(waveform, sample_rate) for waveform in raw_array]
        else:
            raise ValueError(f"Unsupported standard audio generation kind {invocation.adapter.generation_kind}.")

        requested_sample_rate = int(invocation.controls["sample_rate"])
        audio_variations = [
            resample_audio_object(crop_tail(audio, 0, invocation.duration_seconds), requested_sample_rate)
            for audio in audio_variations
        ]
        audio = audio_variations[0]
        return {
            "audio": audio,
            "audio_variations": audio_variations,
            "sample_rate_out": requested_sample_rate,
            "duration_seconds": float(audio["duration_seconds"]),
        }
