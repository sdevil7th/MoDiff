import inspect
import logging
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
    normalize_offload_mode,
    offload_mode_param,
)
from utils.huggingface import local_files_only
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype

logger = logging.getLogger("modiff")

ACE_STEP_DEFAULT_REPO = "ACE-Step/acestep-v15-xl-turbo-diffusers"
DEVICE_OPTIONS = list(DEVICE_LIST.keys())
DIRECT_AUDIO_OFFLOAD_MODES = [
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
]
ACE_TASK_TYPES = ["text2music", "cover", "repaint", "continuation", "extract", "lego", "complete"]


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


def pipeline_class_from_name(name: str):
    if name != "AceStepPipeline":
        raise ValueError(f"Unsupported Diffusers audio pipeline class: {name}")
    from diffusers import AceStepPipeline

    return AceStepPipeline


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
    if array.dtype.kind in ("i", "u"):
        array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
    else:
        array = array.astype(np.float32, copy=False)
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim == 2 and array.shape[0] > array.shape[1]:
        array = array.T
    return np.clip(array, -1.0, 1.0), sample_rate


def audio_to_tensor(audio: Any, device: Any):
    import torch

    array, _sample_rate = audio_to_numpy(audio)
    return torch.from_numpy(array).to(device=device, dtype=torch.float32)


def output_to_audio_object(result: Any, sample_rate: int = 48000) -> dict[str, Any]:
    import torch

    audio = getattr(result, "audios", result)
    if isinstance(audio, (list, tuple)) and len(audio) > 0:
        audio = audio[0]
    if isinstance(audio, dict):
        if "samples" in audio:
            return audio
        if "audios" in audio:
            return output_to_audio_object(audio["audios"], sample_rate)
    if isinstance(audio, torch.Tensor):
        array = audio.detach().float().cpu().numpy()
    else:
        array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
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
            "options": ["AceStepPipeline"],
            "default": "AceStepPipeline",
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
        "enable_vae_tiling": {"label": "VAE tiling", "type": "bool", "default": True},
        "low_cpu_mem_usage": {"label": "Low CPU memory", "type": "bool", "default": True},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        model_id = repo_value(kwargs.get("model_id")) or ACE_STEP_DEFAULT_REPO
        pipeline_class_name = str(kwargs.get("pipeline_class") or "AceStepPipeline")
        pipeline_class = pipeline_class_from_name(pipeline_class_name)
        dtype = str_to_dtype(kwargs.get("dtype") or "bfloat16")
        device = kwargs.get("device") or DEFAULT_DEVICE
        revision = none_if_blank(kwargs.get("revision"))
        auto_offload = bool(kwargs.get("auto_offload", True))
        offload_mode = normalize_offload_mode(kwargs.get("offload_mode") or OFFLOAD_MODE_MODEL_CPU, auto_offload=auto_offload)

        load_kwargs = {
            "torch_dtype": dtype,
            "revision": revision,
            "low_cpu_mem_usage": bool(kwargs.get("low_cpu_mem_usage", True)),
            "local_files_only": local_files_only(model_id),
        }

        self.progress(-1, phase="loading", message=f"Loading {pipeline_class_name}")
        pipeline = pipeline_class.from_pretrained(model_id, **load_kwargs)
        if kwargs.get("enable_vae_tiling", True):
            vae = getattr(pipeline, "vae", None)
            for method_name in ("enable_tiling", "enable_vae_tiling"):
                method = getattr(vae or pipeline, method_name, None)
                if callable(method):
                    method()
                    break

        self.progress(-1, phase="loading", message=f"Applying {offload_mode} offload")
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


class Generate(NodeBase):
    """Generate audio with a Diffusers audio pipeline."""

    label = "Diffusers Audio Generate"
    category = "Diffusers Audio"
    resizable = True
    params = {
        "pipeline": {"label": "Pipeline", "display": "input", "type": "audio_diffusion_pipeline"},
        "task_type": {"label": "Task", "type": "string", "options": ACE_TASK_TYPES, "default": "text2music"},
        "prompt": {"label": "Prompt", "display": "textarea", "type": "text", "default": ""},
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
        "shift": {"label": "Shift", "display": "slider", "type": "float", "default": 3.0, "min": 0, "max": 10, "step": 0.1},
        "seed": {"label": "Seed", "type": "int", "display": "random", "default": 0, "min": 0, "max": 4294967295},
        "bpm": {"label": "BPM", "type": "int", "default": 0, "min": 0, "max": 400},
        "keyscale": {"label": "Key", "type": "string", "default": ""},
        "timesignature": {"label": "Time", "type": "string", "default": "4"},
        "source_audio": {"label": "Source Audio", "display": "input", "type": ["audio", "str"]},
        "reference_audio": {"label": "Reference Audio", "display": "input", "type": ["audio", "str"]},
        "repainting_start": {"label": "Repaint Start", "type": "float", "default": 0.0, "min": 0, "step": 0.01},
        "repainting_end": {"label": "Repaint End", "type": "float", "default": 0.0, "min": 0, "step": 0.01},
        "audio_cover_strength": {"label": "Cover Strength", "display": "slider", "type": "float", "default": 0.85, "min": 0, "max": 1, "step": 0.01},
        "return_continuation_tail": {"label": "Return Tail Only", "type": "bool", "default": True},
        "sample_rate": {"label": "Sample Rate", "type": "int", "default": 48000, "min": 8000, "max": 192000},
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "sample_rate_out": {"label": "Sample Rate", "display": "output", "type": "int"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        import torch

        pipeline = kwargs.get("pipeline")
        if pipeline is None:
            raise ValueError("Diffusers audio pipeline is required.")

        sample_rate = int(kwargs.get("sample_rate") or 48000)
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
            "guidance_scale": float(kwargs.get("guidance_scale") or 1.0),
            "shift": float(kwargs.get("shift") or 3.0),
            "generator": generator,
            "output_type": "pt",
            "return_dict": True,
            "task_type": call_task_type,
        }
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
            tensor = audio_to_tensor(source_audio, device=device)
            if supports_arg(pipeline, "src_audio"):
                call_kwargs["src_audio"] = tensor
            if task_type in ("repaint", "continuation") and supports_arg(pipeline, "reference_audio"):
                call_kwargs["reference_audio"] = tensor
        if reference_audio not in (None, "") and supports_arg(pipeline, "reference_audio"):
            call_kwargs["reference_audio"] = audio_to_tensor(reference_audio, device=device)

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
            call_kwargs["audio_cover_strength"] = float(kwargs.get("audio_cover_strength") or 0.85)

        if supports_arg(pipeline, "callback_on_step_end"):
            call_kwargs["callback_on_step_end"] = self.pipe_callback
            if supports_arg(pipeline, "callback_on_step_end_tensor_inputs"):
                call_kwargs["callback_on_step_end_tensor_inputs"] = []

        self.progress(0, phase="denoising", message=f"Generating audio ({task_type})")
        result = pipeline(**call_kwargs)
        audio = output_to_audio_object(result, sample_rate=sample_rate)
        if task_type == "continuation" and kwargs.get("return_continuation_tail", True):
            audio = crop_tail(audio, source_duration, float(kwargs.get("extension_duration") or 15.0))

        return {
            "audio": audio,
            "sample_rate_out": int(audio.get("sample_rate") or sample_rate),
            "duration_seconds": float(audio.get("duration_seconds") or 0.0),
        }
