import logging
from pathlib import Path

import numpy as np

from modiff.NodeBase import NodeBase
from modiff.config import CONFIG
from utils.paths import parse_filename

logger = logging.getLogger("modiff")


def _resolve_file(value):
    file = value[0] if isinstance(value, list) and value else value
    file = str(file or "")
    if not file:
        return None
    path = Path(file)
    if not path.is_absolute():
        path = Path(CONFIG.paths["work_dir"]) / path
    return path


def _collapse_single(values):
    return values[0] if len(values) == 1 else values


def _audio_to_numpy(audio):
    import torch

    if isinstance(audio, dict):
        if "samples" in audio:
            data = audio["samples"]
        elif "audio" in audio:
            data = audio["audio"]
        elif "array" in audio:
            data = audio["array"]
    else:
        data = audio

    if isinstance(data, torch.Tensor):
        array = data.detach().float().cpu().numpy()
    elif isinstance(data, np.ndarray):
        if data.dtype.kind in ("i", "u"):
            array = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
        else:
            array = data.astype(np.float32, copy=False)
    elif isinstance(data, str):
        loaded = _read_wav(Path(data))
        return loaded["samples"], loaded["sample_rate"]
    else:
        array = np.asarray(data, dtype=np.float32)

    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim == 2 and array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
        array = array.T

    sample_rate = int(audio.get("sample_rate", 48000)) if isinstance(audio, dict) else 48000
    return np.clip(array, -1.0, 1.0), sample_rate


def _read_wav(path):
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(path)
    array = np.asarray(data)
    if array.dtype.kind in ("i", "u"):
        max_value = np.iinfo(array.dtype).max
        array = array.astype(np.float32) / float(max_value)
    else:
        array = array.astype(np.float32)
    if array.ndim == 1:
        channels = 1
    else:
        channels = array.shape[1]
    duration = float(array.shape[0] / sample_rate) if sample_rate else 0.0
    return {
        "path": str(path),
        "samples": array,
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "duration_seconds": duration,
    }


def _write_wav(path, samples, sample_rate):
    from scipy.io import wavfile

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(samples, dtype=np.float32)
    array = np.clip(array, -1.0, 1.0)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    pcm = (array * 32767.0).astype(np.int16)
    wavfile.write(path, int(sample_rate), pcm)


class Load(NodeBase):
    """Load an audio file as a reusable audio object."""

    label = "Load Audio"
    category = "Audio"
    resizable = True
    params = {
        "audio": {"label": "Audio", "display": "output", "type": "audio"},
        "file": {
            "label": False,
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {
                "fileTypes": ["audio"],
                "multiple": False,
            },
        },
        "preview": {"display": "ui_audio", "type": "url", "dataSource": "filename"},
        "filename": {"label": "File", "display": "output", "type": "str"},
        "sample_rate": {"label": "Sample Rate", "display": "output", "type": "int"},
        "channels": {"label": "Channels", "display": "output", "type": "int"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        path = _resolve_file(kwargs.get("file"))
        if path is None or not path.exists():
            raise ValueError("Load Audio needs an existing audio file.")
        if path.suffix.lower() != ".wav":
            raise ValueError("Load Audio currently supports WAV files. Convert source audio to WAV before loading.")

        loaded = _read_wav(path)
        return {
            "audio": loaded,
            "filename": loaded["path"],
            "sample_rate": loaded["sample_rate"],
            "channels": loaded["channels"],
            "duration_seconds": loaded["duration_seconds"],
        }


class TrimPad(NodeBase):
    """Crop, pad, normalize, and optionally resample audio."""

    label = "Trim / Pad Audio"
    category = "Audio"
    resizable = True
    params = {
        "audio": {"label": "Audio", "display": "input", "type": ["audio", "str"]},
        "start_seconds": {"label": "Start", "type": "float", "default": 0.0, "min": 0, "step": 0.01},
        "duration_seconds": {"label": "Duration", "type": "float", "default": 0.0, "min": 0, "step": 0.01},
        "target_sample_rate": {"label": "Target SR", "type": "int", "default": 48000, "min": 8000, "max": 192000},
        "normalize_peak": {"label": "Normalize Peak", "type": "bool", "default": False},
        "output": {"label": "Audio", "display": "output", "type": "audio"},
        "sample_rate": {"label": "Sample Rate", "display": "output", "type": "int"},
        "duration": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        from scipy.signal import resample_poly
        from math import gcd

        samples, sample_rate = _audio_to_numpy(kwargs.get("audio"))
        target_sample_rate = int(kwargs.get("target_sample_rate") or sample_rate)
        if target_sample_rate != sample_rate:
            divisor = gcd(sample_rate, target_sample_rate)
            samples = resample_poly(samples, target_sample_rate // divisor, sample_rate // divisor, axis=0).astype(np.float32)
            sample_rate = target_sample_rate

        start = max(0, int(float(kwargs.get("start_seconds") or 0) * sample_rate))
        duration_value = float(kwargs.get("duration_seconds") or 0)
        end = start + int(duration_value * sample_rate) if duration_value > 0 else samples.shape[0]
        trimmed = samples[start:min(end, samples.shape[0])]
        if duration_value > 0 and trimmed.shape[0] < end - start:
            pad = np.zeros((end - start - trimmed.shape[0], trimmed.shape[1]), dtype=np.float32)
            trimmed = np.concatenate([trimmed, pad], axis=0)

        if kwargs.get("normalize_peak"):
            peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
            if peak > 0:
                trimmed = np.clip(trimmed / peak * 0.98, -1.0, 1.0)

        output = {
            "samples": trimmed,
            "sample_rate": int(sample_rate),
            "channels": int(trimmed.shape[1] if trimmed.ndim == 2 else 1),
            "duration_seconds": float(trimmed.shape[0] / sample_rate) if sample_rate else 0.0,
        }
        return {"output": output, "sample_rate": output["sample_rate"], "duration": output["duration_seconds"]}


class Export(NodeBase):
    """Save audio to a WAV file and expose a preview."""

    label = "Export Audio"
    category = "Audio"
    resizable = True
    params = {
        "audio": {"label": "Audio", "display": "input", "type": ["audio", "str"]},
        "filename": {
            "label": "File",
            "type": "str",
            "default": "{PATH:audio}/MoDiff_{HASH:6}.wav",
        },
        "sample_rate": {"label": "Sample Rate", "type": "int", "default": 48000, "min": 8000, "max": 192000},
        "preview": {"display": "ui_audio", "type": "url", "dataSource": "file"},
        "file": {"label": "File", "display": "output", "type": "audio"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        audio = kwargs.get("audio")
        if audio is None:
            raise ValueError("Export Audio needs audio input.")
        samples, detected_sample_rate = _audio_to_numpy(audio)
        sample_rate = int(kwargs.get("sample_rate") or detected_sample_rate or 48000)

        parsed_filename = Path(parse_filename(kwargs.get("filename") or "{PATH:audio}/MoDiff_{HASH:6}.wav"))
        if not parsed_filename.is_absolute():
            parsed_filename = Path(CONFIG.paths["data"]) / parsed_filename
        _write_wav(parsed_filename, samples, sample_rate)
        return {
            "file": str(parsed_filename),
            "duration_seconds": float(samples.shape[0] / sample_rate) if sample_rate else 0.0,
        }


class Preview(NodeBase):
    """Preview an audio file or audio object."""

    label = "Preview Audio"
    category = "Audio"
    resizable = True
    params = {
        "audio": {"label": "Audio", "display": "input", "type": ["audio", "str"]},
        "preview": {"display": "ui_audio", "type": "url", "dataSource": "file"},
        "file": {"label": "File", "display": "output", "type": "audio"},
    }

    def execute(self, **kwargs):
        audio = kwargs.get("audio")
        if isinstance(audio, str):
            return {"file": audio}
        samples, sample_rate = _audio_to_numpy(audio)
        path = Path(parse_filename("{PATH:audio}/MoDiff_preview_{HASH:6}.wav"))
        if not path.is_absolute():
            path = Path(CONFIG.paths["data"]) / path
        _write_wav(path, samples, sample_rate)
        return {"file": str(path)}
