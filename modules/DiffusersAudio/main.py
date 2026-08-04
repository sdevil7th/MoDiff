import inspect
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
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
from modiff.model_artifact_catalog import resolve_model_revision
from utils.huggingface import local_files_only
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

logger = logging.getLogger("modiff")

ACE_STEP_DEFAULT_REPO = "ACE-Step/acestep-v15-xl-turbo-diffusers"
STABLE_AUDIO_DEFAULT_REPO = "stabilityai/stable-audio-open-1.0"
DEVICE_OPTIONS = list(DEVICE_LIST.keys())
DIRECT_AUDIO_OFFLOAD_MODES = [
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
]
ACE_TASK_TYPES = ["text2music", "cover", "repaint", "continuation", "extract", "lego", "complete"]
AUDIO_SAMPLE_RATE_OPTIONS = {
    "44100": "44.1 kHz",
    "48000": "48 kHz",
    "88200": "88.2 kHz",
    "96000": "96 kHz",
}


@dataclass(frozen=True)
class AudioPipelineAdapter:
    pipeline_class: str
    modes: frozenset[str]
    task_types: frozenset[str]

    def resolve_pipeline_class(self):
        import diffusers

        pipeline = getattr(diffusers, self.pipeline_class, None)
        if pipeline is None:
            raise ValueError(f"Diffusers does not expose audio pipeline class {self.pipeline_class}.")
        return pipeline


AUDIO_PIPELINE_ADAPTERS = {
    "AceStepPipeline": AudioPipelineAdapter(
        pipeline_class="AceStepPipeline",
        modes=frozenset({"text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"}),
        task_types=frozenset(ACE_TASK_TYPES),
    ),
    "StableAudioPipeline": AudioPipelineAdapter(
        pipeline_class="StableAudioPipeline",
        modes=frozenset({"text_to_audio"}),
        task_types=frozenset({"text2audio"}),
    ),
}


def repo_value(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or "")
    return str(value or "")


def none_if_blank(value: Any):
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def value_or_default(value: Any, default: Any):
    return default if value is None else value


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
    adapter = AUDIO_PIPELINE_ADAPTERS.get(name)
    if adapter is None:
        raise ValueError(f"Unsupported Diffusers audio pipeline class: {name}")
    return adapter.resolve_pipeline_class()


def supports_arg(pipeline: Any, arg_name: str) -> bool:
    try:
        return arg_name in inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):
        return False


def audio_to_numpy(audio: Any) -> tuple[np.ndarray, int]:
    import torch
    from scipy.io import wavfile

    data = audio
    sample_rate = 48000
    if isinstance(audio, dict):
        if "samples" in audio:
            data = audio["samples"]
        elif "audio" in audio:
            data = audio["audio"]
        elif "array" in audio:
            data = audio["array"]
        sample_rate = int(audio.get("sample_rate") or sample_rate)
    if isinstance(data, str):
        sample_rate, wav_data = wavfile.read(Path(data))
        data = wav_data
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().numpy()
    array = np.asarray(data)
    array = _pcm_to_float32(array)
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim == 2 and array.shape[0] > array.shape[1]:
        array = array.T
    return np.clip(array, -1.0, 1.0), sample_rate


def audio_to_tensor(audio: Any, device: Any, target_sample_rate: int | None = None):
    import torch
    from scipy.signal import resample_poly
    from math import gcd

    array, sample_rate = audio_to_numpy(audio)
    if target_sample_rate and sample_rate != target_sample_rate:
        divisor = gcd(sample_rate, target_sample_rate)
        array = resample_poly(
            array,
            target_sample_rate // divisor,
            sample_rate // divisor,
            axis=-1,
        ).astype(np.float32, copy=False)
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


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
    end = start + int(duration_seconds * sample_rate) if duration_seconds and duration_seconds > 0 else samples.shape[-1]
    cropped = samples[..., start:min(end, samples.shape[-1])]
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
    params = {
        "pipeline": {"label": "Pipeline", "display": "output", "type": "audio_diffusion_pipeline"},
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            # Keep this literal so the static registry parser can expose the
            # choices without importing Diffusers or executing this module.
            "options": ["AceStepPipeline", "StableAudioPipeline"],
            "default": "AceStepPipeline",
        },
        "mode": {
            "label": "Mode",
            "type": "string",
            "options": ["text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"],
            "default": "text_to_audio",
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

    def execute(self, **kwargs):
        from modules.DiffusersRuntime.main import apply_execution_recipe_to_pipeline, loader_runtime_options

        pipeline_class_name = str(kwargs.get("pipeline_class") or "AceStepPipeline")
        model_selection = kwargs.get("model_id")
        model_id = repo_value(model_selection)
        if not model_id or (pipeline_class_name == "StableAudioPipeline" and model_id == ACE_STEP_DEFAULT_REPO):
            model_id = STABLE_AUDIO_DEFAULT_REPO if pipeline_class_name == "StableAudioPipeline" else ACE_STEP_DEFAULT_REPO
            model_source = "hub"
        else:
            model_source = model_selection.get("source") if isinstance(model_selection, dict) else None
        mode = str(kwargs.get("mode") or "text_to_audio")
        adapter = AUDIO_PIPELINE_ADAPTERS.get(pipeline_class_name)
        if adapter is None or mode not in adapter.modes:
            supported = ', '.join(sorted(adapter.modes if adapter else [])) or 'none'
            raise ValueError(f"{pipeline_class_name} does not support {mode}. Supported modes: {supported}.")
        pipeline_class = pipeline_class_from_name(pipeline_class_name)
        dtype = str_to_dtype(kwargs.get("dtype") or "bfloat16")
        recipe, device, offload_mode, recipe_load_kwargs = loader_runtime_options(
            kwargs,
            default_device=DEFAULT_DEVICE,
            default_offload_mode=OFFLOAD_MODE_MODEL_CPU,
            direct_device_load=True,
        )
        revision = resolve_model_revision(
            model_id,
            none_if_blank(kwargs.get("revision")),
            source=model_source,
        )

        load_kwargs = {
            "torch_dtype": dtype,
            "revision": revision,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "local_files_only": local_files_only(model_id),
            **recipe_load_kwargs,
        }

        self.progress(-1, phase="loading", message=f"Loading {pipeline_class_name}")
        with self.diffusers_loading_progress():
            pipeline = pipeline_class.from_pretrained(model_id, **load_kwargs)
        setattr(pipeline, "_modiff_audio_pipeline_class", pipeline_class_name)
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
    """Load an ACE-Step LoRA from MoDiff's managed cache or a local folder."""

    label = "Load Diffusers Audio LoRA"
    category = "Diffusers Audio"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "audio_diffusion_pipeline", "required": True},
        "adapter_path": {
            "label": "LoRA",
            "display": "modelselect",
            "type": "string",
            "fieldOptions": {"noValidation": True, "sources": ["hub", "local"]},
        },
        "weight_name": {"label": "Weight name", "type": "string", "default": "adapter_model.safetensors"},
        "expected_sha256": {"label": "Expected SHA-256", "type": "string", "default": ""},
        "adapter_name": {"label": "Adapter name", "type": "string", "default": "audio_style"},
        "replace_existing": {"label": "Replace existing adapters", "type": "bool", "default": True},
        "scale": {"label": "Strength", "display": "slider", "type": "float", "default": 0.7, "min": 0, "max": 2, "step": 0.05},
        "output": {"label": "Pipeline", "display": "output", "type": "audio_diffusion_pipeline"},
    }

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

        selection = kwargs.get("adapter_path")
        adapter_path = repo_value(selection)
        if not adapter_path:
            return {"output": pipeline}
        weight_name = str(kwargs.get("weight_name") or "adapter_model.safetensors").strip()
        source = selection.get("source") if isinstance(selection, dict) else "local"
        if source == "hub":
            from utils.huggingface import cached_file_path

            cached = cached_file_path(adapter_path, weight_name)
            if not cached:
                raise FileNotFoundError(
                    f"Audio LoRA {adapter_path}/{weight_name} is not installed. Install it through Model Manager first."
                )
            cached_path = Path(cached)
            expected = str(kwargs.get("expected_sha256") or "").strip().lower().removeprefix("sha256:")
            if expected:
                digest = hashlib.sha256()
                with cached_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError("The audio LoRA failed its pinned SHA-256 verification. Repair it in Model Manager.")
            adapter_path = str(cached_path.parent)
            weight_name = cached_path.name

        adapter_name = str(kwargs.get("adapter_name") or "audio_style").strip()
        if kwargs.get("replace_existing", True) and callable(getattr(pipeline, "unload_lora_weights", None)):
            pipeline.unload_lora_weights()
        pipeline.load_lora_weights(adapter_path, weight_name=weight_name, adapter_name=adapter_name)
        if callable(getattr(pipeline, "set_adapters", None)):
            pipeline.set_adapters([adapter_name], [float(kwargs.get("scale", 0.7))])
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
            weights = [float(part.strip()) for part in str(kwargs.get("adapter_weights") or "").split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("Audio LoRA weights must be comma-separated numbers.") from exc
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
        },
        "task_type": {"label": "Task", "type": "string", "options": ACE_TASK_TYPES, "default": "text2music"},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
        "negative_prompt": {"label": "Negative Prompt", "display": "textarea", "type": "text", "default": ""},
        "lyrics": {"label": "Lyrics", "display": "textarea", "type": "text", "default": ""},
        "audio_duration": {"label": "Duration", "type": "float", "default": 30.0, "min": 1, "max": 240, "step": 0.5},
        "extension_duration": {"label": "Extension", "type": "float", "default": 15.0, "min": 1, "max": 180, "step": 0.5},
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
        "shift": {"label": "Shift", "display": "slider", "type": "float", "default": 3.0, "min": 0, "max": 10, "step": 0.1},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "bpm": {"label": "BPM", "type": "int", "default": 0, "min": 0, "max": 400},
        "keyscale": {"label": "Key", "type": "string", "default": ""},
        "timesignature": {"label": "Time", "type": "string", "default": "4"},
        "source_audio": {"label": "Source Audio", "display": "input", "type": ["audio", "str"], "required": False},
        "reference_audio": {"label": "Reference Audio", "display": "input", "type": ["audio", "str"], "required": False},
        "repainting_start": {"label": "Repaint Start", "type": "float", "default": 0.0, "min": 0, "step": 0.01},
        "repainting_end": {"label": "Repaint End", "type": "float", "default": 0.0, "min": 0, "step": 0.01},
        "audio_cover_strength": {"label": "Cover Strength", "display": "slider", "type": "float", "default": 0.85, "min": 0, "max": 1, "step": 0.01},
        "return_continuation_tail": {"label": "Return Tail Only", "type": "bool", "default": True},
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
            "description": "StableAudioPipeline-only denoising steps; ignored by ACE-Step.",
        },
        "stable_audio_guidance": {
            "label": "Stable Audio Guidance",
            "type": "float",
            "default": 7,
            "min": 0,
            "max": 20,
            "description": "StableAudioPipeline-only classifier-free guidance; ignored by ACE-Step.",
        },
        "num_waveforms": {
            "label": "Variations",
            "type": "int",
            "default": 1,
            "min": 1,
            "max": 8,
            "description": "Number of StableAudioPipeline waveforms; ignored by ACE-Step.",
        },
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "audio_variations": {"label": "Audio Variations", "display": "output", "type": "collection"},
        "sample_rate_out": {"label": "Sample Rate", "display": "output", "type": "int"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        import torch

        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Diffusers audio pipeline is required.")

        if getattr(pipeline, "_modiff_audio_pipeline_class", None) == "StableAudioPipeline":
            return self._execute_stable_audio(pipeline, kwargs)

        requested_sample_rate = int(kwargs.get("sample_rate") or 48000)
        if requested_sample_rate not in {int(value) for value in AUDIO_SAMPLE_RATE_OPTIONS}:
            supported = ", ".join(AUDIO_SAMPLE_RATE_OPTIONS.values())
            raise ValueError(f"Audio sample rate must be one of: {supported}.")
        # The decoded tensor is produced at the VAE's native rate. Labeling it
        # with a different UI/export rate changes its duration and can truncate
        # valid samples, so generation stays at the pipeline's native rate and
        # the completed audio is resampled to the requested delivery rate.
        sample_rate = int(getattr(pipeline, "sample_rate", None) or 48000)
        task_type = str(kwargs.get("task_type") or "text2music")
        source_audio = kwargs.get("source_audio")
        reference_audio = kwargs.get("reference_audio")
        source_duration = 0.0

        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed", 0)))
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(int(kwargs.get("seed", 0)))

        call_task_type = "repaint" if task_type == "continuation" else task_type
        audio_duration = float(kwargs.get("audio_duration") or 30.0)
        if source_audio not in (None, ""):
            source_array, source_sample_rate = audio_to_numpy(source_audio)
            source_duration = float(source_array.shape[-1] / source_sample_rate) if source_sample_rate else 0.0
            if task_type == "continuation":
                audio_duration = source_duration + float(kwargs.get("extension_duration") or 15.0)

        call_kwargs = {
            "prompt": str(kwargs.get("prompt") or ""),
            "lyrics": str(kwargs.get("lyrics") or ""),
            "audio_duration": audio_duration,
            "vocal_language": str(kwargs.get("vocal_language") or "en"),
            "num_inference_steps": int(kwargs.get("num_inference_steps") or 8),
            "guidance_scale": float(value_or_default(kwargs.get("guidance_scale"), 1.0)),
            "shift": float(value_or_default(kwargs.get("shift"), 3.0)),
            "generator": generator,
            "output_type": "pt",
            "return_dict": True,
            "task_type": call_task_type,
        }
        if supports_arg(pipeline, "attention_kwargs"):
            call_kwargs["attention_kwargs"] = {"scale": float(kwargs.get("lora_scale", 1.0))}
        for optional in ("bpm", "keyscale", "timesignature"):
            value = none_if_blank(kwargs.get(optional))
            if optional == "bpm" and value is not None:
                try:
                    value = int(float(value))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"BPM must be numeric; received {value!r}.") from exc
                if value <= 0:
                    value = None
            if value is not None and supports_arg(pipeline, optional):
                call_kwargs[optional] = value

        if source_audio not in (None, ""):
            tensor = audio_to_tensor(source_audio, device=device, target_sample_rate=sample_rate)
            if task_type == "continuation":
                target_samples = int(round(audio_duration * sample_rate))
                if tensor.shape[-1] < target_samples:
                    tensor = torch.nn.functional.pad(tensor, (0, target_samples - tensor.shape[-1]))
            # ACE-Step cover/variation treats the supplied track as a timbre
            # and style reference. Sending it as src_audio instead asks the
            # pipeline for semantic-code cover conditioning, which requires
            # optional audio tokenizer/detokenizer modules that the official
            # Diffusers artifact does not publish. Repaint and continuation,
            # by contrast, need src_audio so the VAE can preserve the source
            # waveform outside the edited interval.
            if task_type == "cover" and supports_arg(pipeline, "reference_audio"):
                call_kwargs["reference_audio"] = tensor
            elif supports_arg(pipeline, "src_audio"):
                call_kwargs["src_audio"] = tensor
        if reference_audio not in (None, "") and supports_arg(pipeline, "reference_audio"):
            call_kwargs["reference_audio"] = audio_to_tensor(
                reference_audio,
                device=device,
                target_sample_rate=sample_rate,
            )

        if task_type in ("repaint", "continuation"):
            start = float(kwargs.get("repainting_start") or 0.0)
            end = float(kwargs.get("repainting_end") or 0.0)
            if task_type == "continuation":
                start = source_duration
                end = audio_duration
            if supports_arg(pipeline, "repainting_start"):
                call_kwargs["repainting_start"] = start
            if supports_arg(pipeline, "repainting_end"):
                call_kwargs["repainting_end"] = end
        if task_type == "cover" and supports_arg(pipeline, "audio_cover_strength"):
            call_kwargs["audio_cover_strength"] = float(
                value_or_default(kwargs.get("audio_cover_strength"), 0.85)
            )

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
            audio = crop_tail(audio, source_duration, float(kwargs.get("extension_duration") or 15.0))
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

    def _execute_stable_audio(self, pipeline, kwargs):
        import torch

        if kwargs.get("source_audio") not in (None, "") or kwargs.get("reference_audio") not in (None, ""):
            raise ValueError("Stable Audio supports text-to-audio only and does not accept source audio.")
        duration = float(kwargs.get("audio_duration") or 30)
        if not 0 < duration <= 47:
            raise ValueError("Stable Audio duration must be greater than 0 and at most 47 seconds.")
        device = getattr(pipeline, "_execution_device", None) or getattr(pipeline, "device", None) or "cpu"
        try:
            generator = torch.Generator(device=device).manual_seed(int(kwargs.get("seed") or 0))
        except Exception:
            generator = torch.Generator(device="cpu").manual_seed(int(kwargs.get("seed") or 0))

        def callback(step, timestep, latents):
            if not hasattr(pipeline, "_num_timesteps"):
                pipeline._num_timesteps = int(kwargs.get("stable_audio_steps") or 100)
            self.pipe_callback(pipeline, step, timestep, {"latents": latents})

        result = pipeline(
            prompt=str(kwargs.get("prompt") or ""),
            negative_prompt=none_if_blank(kwargs.get("negative_prompt")),
            audio_start_in_s=0,
            audio_end_in_s=duration,
            num_inference_steps=int(kwargs.get("stable_audio_steps") or 100),
            guidance_scale=float(value_or_default(kwargs.get("stable_audio_guidance"), 7)),
            num_waveforms_per_prompt=int(kwargs.get("num_waveforms") or 1),
            generator=generator,
            callback=callback,
            callback_steps=1,
            output_type="pt",
            return_dict=True,
        )
        vae = getattr(pipeline, "vae", None)
        vae_config = getattr(vae, "config", None)
        configured_rate = vae_config.get("sampling_rate") if hasattr(vae_config, "get") else None
        sample_rate = int(getattr(vae, "sampling_rate", None) or configured_rate or 44100)
        requested_sample_rate = int(kwargs.get("sample_rate") or 48000)
        audio_variations = [
            resample_audio_object(crop_tail(audio, 0, duration), requested_sample_rate)
            for audio in output_to_audio_objects(result, sample_rate)
        ]
        audio = audio_variations[0]
        return {
            "audio": audio,
            "audio_variations": audio_variations,
            "sample_rate_out": requested_sample_rate,
            "duration_seconds": float(audio["duration_seconds"]),
        }
