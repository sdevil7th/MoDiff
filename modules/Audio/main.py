import logging
import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from modiff.NodeBase import NodeBase
from modiff.config import CONFIG
from modiff.path_identifiers import resolve_runtime_input_path
from utils.paths import parse_filename

logger = logging.getLogger("modiff")
AUDIO_SAMPLE_RATE_OPTIONS = {
    "16000": "16 kHz",
    "24000": "24 kHz",
    "44100": "44.1 kHz",
    "48000": "48 kHz",
    "88200": "88.2 kHz",
    "96000": "96 kHz",
}
AUDIO_OPERATION_MODES = ("audio_trim", "audio_join", "audio_loudness_match")
AUDIO_OPERATION_PIPELINE_CLASS = "BuiltinAudioOperationV1"
MAX_AUDIO_OPERATION_DURATION_SECONDS = 300.0
MAX_AUDIO_OPERATION_SCALAR_SAMPLES = 14_400_000
MAX_AUDIO_OPERATION_FILE_BYTES = MAX_AUDIO_OPERATION_SCALAR_SAMPLES * 8 + 1_048_576
MAX_AUDIO_OPERATION_CHANNELS = 8


def _pcm_to_float32(array):
    if array.dtype.kind == "u":
        midpoint = float(np.iinfo(array.dtype).max + 1) / 2.0
        return (array.astype(np.float32) - midpoint) / midpoint
    if array.dtype.kind == "i":
        limits = np.iinfo(array.dtype)
        scale = float(max(abs(int(limits.min)), abs(int(limits.max))))
        return array.astype(np.float32) / scale
    return array.astype(np.float32, copy=False)


def _resolve_file(value):
    file = value[0] if isinstance(value, list) and value else value
    file = str(file or "")
    if not file:
        return None
    return resolve_runtime_input_path(file)


def _collapse_single(values):
    return values[0] if len(values) == 1 else values


def _audio_to_numpy(audio):
    import torch

    sample_layout = None
    declared_channels = None
    if isinstance(audio, dict):
        sample_layout = audio.get("sample_layout")
        if sample_layout not in {None, "channels_first", "frames_first"}:
            raise ValueError("Audio sample_layout must be exactly channels_first or frames_first.")
        if "channels" in audio:
            raw_channels = audio.get("channels")
            try:
                numeric_channels = float(raw_channels)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError("Audio channels metadata must be a positive integer.") from error
            if (
                isinstance(raw_channels, bool)
                or not np.isfinite(numeric_channels)
                or not numeric_channels.is_integer()
                or numeric_channels <= 0
            ):
                raise ValueError("Audio channels metadata must be a positive integer.")
            declared_channels = int(numeric_channels)
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
        array = _pcm_to_float32(data)
    elif isinstance(data, str):
        if sample_layout not in {None, "frames_first"}:
            raise ValueError("Decoded audio files use frames_first sample_layout; the supplied metadata conflicts.")
        loaded = _read_wav(resolve_runtime_input_path(data))
        if declared_channels is not None and declared_channels != loaded["channels"]:
            raise ValueError(
                f"Audio channels metadata declares {declared_channels} channels, but the decoded file has "
                f"{loaded['channels']}."
            )
        return loaded["samples"], loaded["sample_rate"]
    else:
        array = np.asarray(data, dtype=np.float32)

    if array.ndim == 1:
        if declared_channels not in {None, 1}:
            raise ValueError(
                f"Audio channels metadata declares {declared_channels} channels, but the waveform is one-dimensional."
            )
        array = array[:, None]
    elif array.ndim == 2:
        rows, columns = (int(value) for value in array.shape)
        if sample_layout is not None:
            layout_channels = rows if sample_layout == "channels_first" else columns
            if declared_channels is not None and declared_channels != layout_channels:
                raise ValueError(
                    f"Audio channels metadata declares {declared_channels} channels, but sample_layout "
                    f"{sample_layout} identifies {layout_channels}."
                )
        elif declared_channels is not None:
            rows_match = rows == declared_channels
            columns_match = columns == declared_channels
            if rows_match and columns_match:
                if declared_channels != 1:
                    raise ValueError(
                        "Audio sample layout is ambiguous because both axes match the declared channel count."
                    )
            elif rows_match:
                sample_layout = "channels_first"
            elif columns_match:
                sample_layout = "frames_first"
            else:
                raise ValueError(
                    f"Audio channels metadata declares {declared_channels} channels, but neither sample axis matches."
                )

        if sample_layout == "channels_first":
            array = array.T
        elif sample_layout is None and array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            array = array.T

    sample_rate = int(audio.get("sample_rate", 48000)) if isinstance(audio, dict) else 48000
    return np.clip(array, -1.0, 1.0), sample_rate


def _bounded_audio_object(audio, *, label):
    samples, sample_rate = _audio_to_numpy(audio)
    if samples.ndim != 2:
        raise ValueError(f"{label} must contain a one- or two-dimensional waveform.")
    frames, channels = (int(value) for value in samples.shape)
    if not 8_000 <= sample_rate <= 192_000:
        raise ValueError(f"{label} sample rate must be between 8000 and 192000 Hz.")
    if not 1 <= channels <= MAX_AUDIO_OPERATION_CHANNELS:
        raise ValueError(f"{label} must contain between 1 and {MAX_AUDIO_OPERATION_CHANNELS} channels.")
    duration = frames / sample_rate
    if not frames or not np.isfinite(duration) or duration > MAX_AUDIO_OPERATION_DURATION_SECONDS:
        raise ValueError(
            f"{label} duration must be positive and at most {MAX_AUDIO_OPERATION_DURATION_SECONDS:g} seconds."
        )
    if frames * channels > MAX_AUDIO_OPERATION_SCALAR_SAMPLES:
        raise ValueError(f"{label} exceeds the {MAX_AUDIO_OPERATION_SCALAR_SAMPLES}-sample execution limit.")
    return {
        "samples": samples,
        "sample_layout": "frames_first",
        "sample_rate": int(sample_rate),
        "channels": channels,
        "duration_seconds": duration,
    }


def _read_wav(path):
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(path)
    array = np.asarray(data)
    array = _pcm_to_float32(array)
    if array.ndim == 1:
        channels = 1
    else:
        channels = array.shape[1]
    duration = float(array.shape[0] / sample_rate) if sample_rate else 0.0
    return {
        "path": str(path),
        "samples": array,
        "sample_layout": "frames_first",
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


def _loudnorm_filter(target_lufs, target_lra, target_peak_dbfs):
    options = [
        f"I={float(target_lufs):.2f}",
        f"LRA={float(target_lra):.2f}",
        f"TP={float(target_peak_dbfs):.2f}",
    ]
    options.append("print_format=json")
    return "loudnorm=" + ":".join(options)


def _parse_loudnorm_measurement(stderr):
    matches = re.findall(r"\{\s*\"input_i\".*?\}", stderr, flags=re.DOTALL)
    if not matches:
        raise RuntimeError("FFmpeg did not return a loudness measurement.")
    payload = json.loads(matches[-1])
    required = ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")
    measurement = {}
    for key in required:
        value = float(payload[key])
        if not np.isfinite(value):
            raise ValueError("Audio is silent or too short to measure loudness.")
        measurement[key] = value
    return measurement


def _measure_loudness(samples, sample_rate, target_lufs=-16.0, target_lra=7.0, target_peak_dbfs=-1.0):
    from imageio_ffmpeg import get_ffmpeg_exe

    with tempfile.TemporaryDirectory(prefix="modiff-audio-loudness-") as temporary_dir:
        source_path = Path(temporary_dir) / "source.wav"
        _write_wav(source_path, samples, sample_rate)
        result = subprocess.run(
            [
                get_ffmpeg_exe(),
                "-hide_banner",
                "-nostdin",
                "-i",
                str(source_path),
                "-af",
                _loudnorm_filter(target_lufs, target_lra, target_peak_dbfs),
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
        )
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg loudness analysis failed: {stderr.strip() or 'unknown error'}")
    return _parse_loudnorm_measurement(stderr)


def _atempo_factors(tempo_ratio):
    """Split a tempo ratio into conservative FFmpeg atempo stages."""

    ratio = float(tempo_ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("Audio tempo ratio must be a finite positive number.")

    factors = []
    while ratio > 2.0:
        factors.append(2.0)
        ratio /= 2.0
    while ratio < 0.5:
        factors.append(0.5)
        ratio /= 0.5
    if not factors or abs(ratio - 1.0) > 1e-9:
        factors.append(ratio)
    return factors


def _run_pitch_preserving_stretch(samples, sample_rate, tempo_ratio):
    """Time-stretch audio through FFmpeg while preserving its pitch."""

    from imageio_ffmpeg import get_ffmpeg_exe

    rubberband_filter = (
        f"rubberband=tempo={tempo_ratio:.15f}:pitch=1:"
        "transients=crisp:detector=compound:phase=laminar:window=standard:"
        "smoothing=on:formant=preserved:pitchq=quality:channels=together"
    )
    atempo_filter = ",".join(f"atempo={factor:.15f}" for factor in _atempo_factors(tempo_ratio))

    with tempfile.TemporaryDirectory(prefix="modiff-audio-fit-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        source_path = temporary_root / "source.wav"
        output_path = temporary_root / "stretched.wav"
        _write_wav(source_path, samples, sample_rate)

        command_prefix = [
            get_ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:a:0",
        ]
        command_suffix = [
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(output_path),
        ]
        errors = []
        for engine, audio_filter in (("rubberband", rubberband_filter), ("atempo", atempo_filter)):
            output_path.unlink(missing_ok=True)
            command = [*command_prefix, "-af", audio_filter, *command_suffix]
            result = subprocess.run(command, check=False, capture_output=True)
            if result.returncode == 0 and output_path.is_file():
                loaded = _read_wav(output_path)
                return loaded["samples"], engine
            errors.append(result.stderr.decode("utf-8", errors="replace").strip())
            if engine == "rubberband":
                logger.warning("FFmpeg rubberband filter is unavailable; using the atempo fallback.")

    detail = next((error for error in reversed(errors) if error), "unknown FFmpeg error")
    raise RuntimeError(f"Pitch-preserving audio fit failed: {detail}")


def _apply_half_cosine_fades(samples, sample_rate, fade_in_seconds, fade_out_seconds, content_start, content_end):
    output = np.asarray(samples, dtype=np.float32).copy()
    content_start = max(0, min(int(content_start), output.shape[0]))
    content_end = max(content_start, min(int(content_end), output.shape[0]))
    content_frames = content_end - content_start

    fade_in_frames = min(content_frames, round(max(0.0, float(fade_in_seconds)) * sample_rate))
    if fade_in_frames > 0:
        phase = np.arange(fade_in_frames, dtype=np.float64) / max(1, fade_in_frames - 1)
        gain = np.sin((np.pi / 2) * phase).astype(np.float32)
        output[content_start : content_start + fade_in_frames] *= gain[:, None]

    fade_out_frames = min(content_frames, round(max(0.0, float(fade_out_seconds)) * sample_rate))
    if fade_out_frames > 0:
        phase = np.arange(fade_out_frames, dtype=np.float64) / max(1, fade_out_frames - 1)
        gain = np.cos((np.pi / 2) * phase).astype(np.float32)
        output[content_end - fade_out_frames : content_end] *= gain[:, None]

    return output


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
            from modiff.media_import import audio_as_wav

            path = audio_as_wav(path)

        if path.stat().st_size > MAX_AUDIO_OPERATION_FILE_BYTES:
            raise ValueError(
                f"Load Audio accepts at most {MAX_AUDIO_OPERATION_FILE_BYTES} bytes per decoded WAV input."
            )

        loaded = _bounded_audio_object(_read_wav(path), label="Loaded audio")
        loaded["path"] = str(path)
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
            samples = resample_poly(samples, target_sample_rate // divisor, sample_rate // divisor, axis=0).astype(
                np.float32
            )
            sample_rate = target_sample_rate

        start = max(0, int(float(kwargs.get("start_seconds") or 0) * sample_rate))
        duration_value = float(kwargs.get("duration_seconds") or 0)
        end = start + int(duration_value * sample_rate) if duration_value > 0 else samples.shape[0]
        trimmed = samples[start : min(end, samples.shape[0])]
        if duration_value > 0 and trimmed.shape[0] < end - start:
            pad = np.zeros((end - start - trimmed.shape[0], trimmed.shape[1]), dtype=np.float32)
            trimmed = np.concatenate([trimmed, pad], axis=0)

        if kwargs.get("normalize_peak"):
            peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
            if peak > 0:
                trimmed = np.clip(trimmed / peak * 0.98, -1.0, 1.0)

        output = {
            "samples": trimmed,
            "sample_layout": "frames_first",
            "sample_rate": int(sample_rate),
            "channels": int(trimmed.shape[1] if trimmed.ndim == 2 else 1),
            "duration_seconds": float(trimmed.shape[0] / sample_rate) if sample_rate else 0.0,
        }
        return {"output": output, "sample_rate": output["sample_rate"], "duration": output["duration_seconds"]}


class FitDuration(NodeBase):
    """Fit a source window to an exact timeline without changing pitch."""

    label = "Fit Audio Duration"
    category = "Audio"
    resizable = True
    params = {
        "audio": {"label": "Audio", "display": "input", "type": ["audio", "str"]},
        "source_start_seconds": {"label": "Source Start", "type": "float", "default": 0.0, "min": 0, "step": 0.001},
        "source_duration_seconds": {
            "label": "Source Duration",
            "type": "float",
            "default": 0.0,
            "min": 0,
            "step": 0.001,
        },
        "target_duration_seconds": {
            "label": "Target Duration",
            "type": "float",
            "default": 5.0,
            "min": 0.001,
            "step": 0.001,
        },
        "delay_seconds": {"label": "Delay", "type": "float", "default": 0.0, "step": 0.001},
        "target_sample_rate": {"label": "Target SR", "type": "int", "default": 48000, "min": 8000, "max": 192000},
        "fade_in_seconds": {"label": "Fade In", "type": "float", "default": 0.0, "min": 0, "step": 0.001},
        "fade_out_seconds": {"label": "Fade Out", "type": "float", "default": 0.0, "min": 0, "step": 0.001},
        "output": {"label": "Audio", "display": "output", "type": "audio"},
        "sample_rate": {"label": "Sample Rate", "display": "output", "type": "int"},
        "duration": {"label": "Duration", "display": "output", "type": "float"},
        "tempo_ratio": {"label": "Tempo Ratio", "display": "output", "type": "float"},
        "stretch_engine": {"label": "Stretch Engine", "display": "output", "type": "str"},
    }

    def execute(self, **kwargs):
        from math import gcd

        from scipy.signal import resample_poly

        if kwargs.get("audio") is None:
            raise ValueError("Fit Audio Duration needs audio input.")

        samples, sample_rate = _audio_to_numpy(kwargs.get("audio"))
        target_sample_rate = int(kwargs.get("target_sample_rate") or sample_rate)
        if target_sample_rate < 8000 or target_sample_rate > 192000:
            raise ValueError("Target sample rate must be between 8000 and 192000 Hz.")
        if target_sample_rate != sample_rate:
            divisor = gcd(sample_rate, target_sample_rate)
            samples = resample_poly(
                samples,
                target_sample_rate // divisor,
                sample_rate // divisor,
                axis=0,
            ).astype(np.float32)
            sample_rate = target_sample_rate

        source_start = max(0, round(float(kwargs.get("source_start_seconds") or 0) * sample_rate))
        if source_start >= samples.shape[0]:
            raise ValueError("Source start is outside the input audio.")

        source_duration_seconds = float(kwargs.get("source_duration_seconds") or 0)
        if source_duration_seconds > 0:
            source_frames = max(1, round(source_duration_seconds * sample_rate))
            source_end = source_start + source_frames
            selected = samples[source_start : min(source_end, samples.shape[0])]
            if selected.shape[0] < source_frames:
                padding = np.zeros((source_frames - selected.shape[0], selected.shape[1]), dtype=np.float32)
                selected = np.concatenate([selected, padding], axis=0)
        else:
            selected = samples[source_start:]
            source_frames = selected.shape[0]

        target_duration_seconds = float(kwargs.get("target_duration_seconds") or 0)
        if not np.isfinite(target_duration_seconds) or target_duration_seconds <= 0:
            raise ValueError("Target duration must be a finite positive number.")
        target_frames = max(1, round(target_duration_seconds * sample_rate))
        tempo_ratio = source_frames / target_frames

        if source_frames == target_frames:
            fitted = selected.astype(np.float32, copy=True)
            stretch_engine = "none"
        else:
            fitted, stretch_engine = _run_pitch_preserving_stretch(selected, sample_rate, tempo_ratio)

        if fitted.ndim == 1:
            fitted = fitted[:, None]
        if fitted.shape[0] < target_frames:
            padding = np.zeros((target_frames - fitted.shape[0], fitted.shape[1]), dtype=np.float32)
            fitted = np.concatenate([fitted, padding], axis=0)
        else:
            fitted = fitted[:target_frames]

        delay_frames = round(float(kwargs.get("delay_seconds") or 0) * sample_rate)
        shifted = np.zeros((target_frames, fitted.shape[1]), dtype=np.float32)
        if delay_frames >= 0:
            retained_frames = max(0, target_frames - delay_frames)
            if retained_frames > 0:
                shifted[delay_frames : delay_frames + retained_frames] = fitted[:retained_frames]
            content_start = min(delay_frames, target_frames)
            content_end = target_frames
        else:
            source_offset = min(-delay_frames, target_frames)
            retained_frames = target_frames - source_offset
            if retained_frames > 0:
                shifted[:retained_frames] = fitted[source_offset:]
            content_start = 0
            content_end = retained_frames

        shifted = _apply_half_cosine_fades(
            shifted,
            sample_rate,
            kwargs.get("fade_in_seconds") or 0,
            kwargs.get("fade_out_seconds") or 0,
            content_start,
            content_end,
        )
        shifted = np.clip(shifted, -1.0, 1.0)
        output = {
            "samples": shifted,
            "sample_layout": "frames_first",
            "sample_rate": int(sample_rate),
            "channels": int(shifted.shape[1]),
            "duration_seconds": target_frames / sample_rate,
        }
        return {
            "output": output,
            "sample_rate": output["sample_rate"],
            "duration": output["duration_seconds"],
            "tempo_ratio": float(tempo_ratio),
            "stretch_engine": stretch_engine,
        }


class MatchLoudness(NodeBase):
    """Match generated audio to a reference window without changing its dynamics."""

    label = "Match Audio Loudness"
    category = "Audio"
    resizable = True
    params = {
        "audio": {"label": "Audio", "display": "input", "type": ["audio", "str"]},
        "reference": {"label": "Reference", "display": "input", "type": ["audio", "str"]},
        "reference_window_seconds": {
            "label": "Reference Tail",
            "type": "float",
            "default": 15.0,
            "min": 0,
            "step": 0.1,
        },
        "target_peak_dbfs": {
            "label": "Peak Ceiling",
            "type": "float",
            "default": -1.0,
            "min": -9.0,
            "max": 0.0,
            "step": 0.1,
        },
        "max_adjustment_db": {
            "label": "Max Adjustment",
            "type": "float",
            "default": 12.0,
            "min": 0.0,
            "max": 30.0,
            "step": 0.5,
        },
        "output": {"label": "Audio", "display": "output", "type": "audio"},
        "reference_lufs": {"label": "Reference LUFS", "display": "output", "type": "float"},
        "input_lufs": {"label": "Input LUFS", "display": "output", "type": "float"},
        "output_lufs": {"label": "Output LUFS", "display": "output", "type": "float"},
        "adjustment_db": {"label": "Adjustment", "display": "output", "type": "float"},
        "true_peak_dbfs": {"label": "True Peak", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        if kwargs.get("audio") is None:
            raise ValueError("Match Audio Loudness needs generated audio.")
        if kwargs.get("reference") is None:
            raise ValueError("Match Audio Loudness needs reference audio.")

        samples, sample_rate = _audio_to_numpy(kwargs.get("audio"))
        reference, reference_sample_rate = _audio_to_numpy(kwargs.get("reference"))
        window_seconds = max(0.0, float(kwargs.get("reference_window_seconds") or 0))
        if window_seconds > 0:
            window_frames = max(1, round(window_seconds * reference_sample_rate))
            reference = reference[-window_frames:]

        target_peak_dbfs = min(0.0, max(-9.0, float(kwargs.get("target_peak_dbfs") or -1.0)))
        max_adjustment_db = min(30.0, max(0.0, float(kwargs.get("max_adjustment_db") or 0.0)))
        reference_measurement = _measure_loudness(
            reference,
            reference_sample_rate,
            target_peak_dbfs=target_peak_dbfs,
        )
        reference_lufs = min(-5.0, max(-70.0, reference_measurement["input_i"]))
        input_measurement = _measure_loudness(
            samples,
            sample_rate,
            target_lufs=reference_lufs,
            target_peak_dbfs=target_peak_dbfs,
        )
        original_input_lufs = input_measurement["input_i"]
        requested_gain_db = min(
            max_adjustment_db,
            max(-max_adjustment_db, reference_lufs - original_input_lufs),
        )
        # Continuation matching must not behave like an automatic gain
        # controller. FFmpeg's dynamic loudnorm mode can apply very different
        # gain to consecutive phrases (and audibly pump quiet endings). Apply
        # one constant gain to every sample instead, capped so the measured
        # true peak remains below the requested ceiling.
        peak_limited_gain_db = target_peak_dbfs - input_measurement["input_tp"]
        applied_gain_db = min(requested_gain_db, peak_limited_gain_db)
        matched = samples * (10.0 ** (applied_gain_db / 20.0))
        if matched.ndim == 1:
            matched = matched[:, None]
        matched = np.clip(matched, -1.0, 1.0)
        output_measurement = _measure_loudness(
            matched,
            sample_rate,
            target_lufs=reference_lufs,
            target_peak_dbfs=target_peak_dbfs,
        )
        output = {
            "samples": matched,
            "sample_layout": "frames_first",
            "sample_rate": int(sample_rate),
            "channels": int(matched.shape[1] if matched.ndim == 2 else 1),
            "duration_seconds": float(matched.shape[0] / sample_rate) if sample_rate else 0.0,
        }
        return {
            "output": output,
            "reference_lufs": float(reference_lufs),
            "input_lufs": float(original_input_lufs),
            "output_lufs": float(output_measurement["input_i"]),
            "adjustment_db": float(output_measurement["input_i"] - original_input_lufs),
            "true_peak_dbfs": float(output_measurement["input_tp"]),
        }


class Join(NodeBase):
    """Append a continuation to its source while preserving exact duration."""

    label = "Join Audio"
    category = "Audio"
    resizable = True
    params = {
        "source": {"label": "Source", "display": "input", "type": ["audio", "str"]},
        "continuation": {"label": "Continuation", "display": "input", "type": ["audio", "str"]},
        "boundary_fade_seconds": {
            "label": "Boundary Fade",
            "type": "float",
            "default": 0.01,
            "min": 0.0,
            "max": 1.0,
            "step": 0.001,
        },
        "output": {"label": "Audio", "display": "output", "type": "audio"},
        "sample_rate": {"label": "Sample Rate", "display": "output", "type": "int"},
        "duration": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        from math import gcd

        from scipy.signal import resample_poly

        if kwargs.get("source") is None:
            raise ValueError("Join Audio needs source audio.")
        if kwargs.get("continuation") is None:
            raise ValueError("Join Audio needs continuation audio.")

        source, sample_rate = _audio_to_numpy(kwargs.get("source"))
        continuation, continuation_sample_rate = _audio_to_numpy(kwargs.get("continuation"))
        if continuation_sample_rate != sample_rate:
            divisor = gcd(sample_rate, continuation_sample_rate)
            continuation = resample_poly(
                continuation,
                sample_rate // divisor,
                continuation_sample_rate // divisor,
                axis=0,
            ).astype(np.float32)

        source_channels = source.shape[1]
        continuation_channels = continuation.shape[1]
        if source_channels != continuation_channels:
            if source_channels == 1:
                source = np.repeat(source, continuation_channels, axis=1)
            elif continuation_channels == 1:
                continuation = np.repeat(continuation, source_channels, axis=1)
            else:
                raise ValueError(
                    f"Join Audio cannot combine {source_channels}-channel source "
                    f"with {continuation_channels}-channel continuation."
                )

        fade_frames = min(
            source.shape[0],
            continuation.shape[0],
            round(max(0.0, float(kwargs.get("boundary_fade_seconds") or 0)) * sample_rate),
        )
        if fade_frames > 0:
            phase = np.arange(fade_frames, dtype=np.float64) / max(1, fade_frames - 1)
            source_fade = np.cos((np.pi / 2) * phase).astype(np.float32)
            continuation_fade = np.sin((np.pi / 2) * phase).astype(np.float32)
            source = source.copy()
            continuation = continuation.copy()
            source[-fade_frames:] *= source_fade[:, None]
            continuation[:fade_frames] *= continuation_fade[:, None]

        joined = np.concatenate([source, continuation], axis=0)
        output = {
            "samples": np.clip(joined, -1.0, 1.0),
            "sample_layout": "frames_first",
            "sample_rate": int(sample_rate),
            "channels": int(joined.shape[1]),
            "duration_seconds": float(joined.shape[0] / sample_rate) if sample_rate else 0.0,
        }
        return {
            "output": output,
            "sample_rate": output["sample_rate"],
            "duration": output["duration_seconds"],
        }


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
        "sample_rate": {
            "label": "Export Sample Rate",
            "type": "int",
            "default": 48000,
            "options": AUDIO_SAMPLE_RATE_OPTIONS,
        },
        "preview": {"display": "ui_audio", "type": "url", "dataSource": "file"},
        "file": {"label": "File", "display": "output", "type": "audio"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        from math import gcd

        from scipy.signal import resample_poly

        audio = kwargs.get("audio")
        if audio is None:
            raise ValueError("Export Audio needs audio input.")
        samples, detected_sample_rate = _audio_to_numpy(audio)
        sample_rate = int(kwargs.get("sample_rate") or detected_sample_rate or 48000)
        if sample_rate not in {int(value) for value in AUDIO_SAMPLE_RATE_OPTIONS}:
            supported = ", ".join(AUDIO_SAMPLE_RATE_OPTIONS.values())
            raise ValueError(f"Export sample rate must be one of: {supported}.")
        if sample_rate != detected_sample_rate:
            divisor = gcd(detected_sample_rate, sample_rate)
            samples = resample_poly(
                samples,
                sample_rate // divisor,
                detected_sample_rate // divisor,
                axis=0,
            ).astype(np.float32)

        parsed_filename = Path(parse_filename(kwargs.get("filename") or "{PATH:audio}/MoDiff_{HASH:6}.wav"))
        if not parsed_filename.is_absolute():
            parsed_filename = Path(CONFIG.paths["data"]) / parsed_filename
        _write_wav(parsed_filename, samples, sample_rate)
        return {
            "file": str(parsed_filename),
            "duration_seconds": float(samples.shape[0] / sample_rate) if sample_rate else 0.0,
        }


class ProcessAudio(NodeBase):
    """Dispatch reviewed install-free audio edits through bounded base nodes."""

    label = "Process Audio"
    category = "Audio"
    resizable = True
    params = {
        "source": {"label": "Source Audio", "display": "input", "type": ["audio", "str"]},
        "reference": {"label": "Reference Audio", "display": "input", "type": ["audio", "str"]},
        "pipeline_class": {
            "label": "Built-in Contract",
            "type": "string",
            "default": AUDIO_OPERATION_PIPELINE_CLASS,
            "hidden": True,
        },
        "operation": {
            "label": "Operation",
            "type": "string",
            "options": AUDIO_OPERATION_MODES,
            "default": "audio_trim",
        },
        "start_seconds": {"label": "Start", "type": "float", "default": 0.0, "min": 0, "step": 0.01},
        "duration_seconds": {"label": "Duration (0 = remainder)", "type": "float", "default": 0.0, "min": 0},
        "target_sample_rate": {
            "label": "Target Sample Rate",
            "type": "int",
            "default": 48_000,
            "options": AUDIO_SAMPLE_RATE_OPTIONS,
        },
        "normalize_peak": {"label": "Normalize Peak", "type": "bool", "default": False},
        "boundary_fade_seconds": {
            "label": "Join Boundary Fade",
            "type": "float",
            "default": 0.01,
            "min": 0,
            "max": 1,
            "step": 0.001,
        },
        "reference_window_seconds": {
            "label": "Reference Tail",
            "type": "float",
            "default": 15.0,
            "min": 0,
            "max": 60,
        },
        "target_peak_dbfs": {
            "label": "Peak Ceiling",
            "type": "float",
            "default": -1.0,
            "min": -9,
            "max": 0,
        },
        "max_adjustment_db": {
            "label": "Maximum Loudness Adjustment",
            "type": "float",
            "default": 12.0,
            "min": 0,
            "max": 30,
        },
        "output": {"label": "Processed Audio", "display": "output", "type": "audio"},
        "sample_rate": {"label": "Sample Rate", "display": "output", "type": "int"},
        "duration": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        if kwargs.get("pipeline_class", AUDIO_OPERATION_PIPELINE_CLASS) != AUDIO_OPERATION_PIPELINE_CLASS:
            raise ValueError("Process Audio received an unsupported built-in contract identity.")
        operation = str(kwargs.get("operation") or "audio_trim")
        if operation not in AUDIO_OPERATION_MODES:
            raise ValueError(f"Unsupported built-in audio operation {operation!r}.")
        if kwargs.get("source") is None:
            raise ValueError("Built-in audio operations require source audio.")
        source = _bounded_audio_object(kwargs.get("source"), label="Source audio")

        if operation == "audio_trim":
            start_seconds = float(kwargs.get("start_seconds") or 0)
            duration_seconds = float(kwargs.get("duration_seconds") or 0)
            raw_target_sample_rate = kwargs.get("target_sample_rate")
            target_sample_rate = int(
                source["sample_rate"] if raw_target_sample_rate is None else raw_target_sample_rate
            )
            if not np.isfinite(start_seconds) or not 0 <= start_seconds < source["duration_seconds"]:
                raise ValueError("Audio trim start must be finite and within the source duration.")
            if (
                not np.isfinite(duration_seconds)
                or duration_seconds < 0
                or duration_seconds > MAX_AUDIO_OPERATION_DURATION_SECONDS
            ):
                raise ValueError(
                    f"Audio trim duration must be between 0 and {MAX_AUDIO_OPERATION_DURATION_SECONDS:g} seconds."
                )
            if target_sample_rate not in {int(value) for value in AUDIO_SAMPLE_RATE_OPTIONS}:
                raise ValueError("Audio trim target sample rate is unsupported.")
            output_duration = duration_seconds or (source["duration_seconds"] - start_seconds)
            output_frames = round(output_duration * target_sample_rate)
            if output_frames * source["channels"] > MAX_AUDIO_OPERATION_SCALAR_SAMPLES:
                raise ValueError("Audio trim output exceeds the bounded sample limit.")
            result = TrimPad().execute(
                audio=source,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                target_sample_rate=target_sample_rate,
                normalize_peak=bool(kwargs.get("normalize_peak")),
            )
        else:
            if kwargs.get("reference") is None:
                raise ValueError(f"{operation.replace('_', ' ').title()} requires reference audio.")
            reference = _bounded_audio_object(kwargs.get("reference"), label="Reference audio")
            if operation == "audio_join":
                boundary_fade_seconds = float(kwargs.get("boundary_fade_seconds") or 0)
                if not np.isfinite(boundary_fade_seconds) or not 0 <= boundary_fade_seconds <= 1:
                    raise ValueError("Audio join boundary fade must be between 0 and 1 second.")
                reference_frames = round(
                    reference["samples"].shape[0] * source["sample_rate"] / reference["sample_rate"]
                )
                output_channels = max(source["channels"], reference["channels"])
                if (
                    source["samples"].shape[0] + reference_frames
                ) * output_channels > MAX_AUDIO_OPERATION_SCALAR_SAMPLES:
                    raise ValueError("Audio join output exceeds the bounded sample limit.")
                result = Join().execute(
                    source=source,
                    continuation=reference,
                    boundary_fade_seconds=boundary_fade_seconds,
                )
            else:
                reference_window_seconds = float(kwargs.get("reference_window_seconds") or 0)
                raw_target_peak_dbfs = kwargs.get("target_peak_dbfs")
                raw_max_adjustment_db = kwargs.get("max_adjustment_db")
                target_peak_dbfs = float(-1.0 if raw_target_peak_dbfs is None else raw_target_peak_dbfs)
                max_adjustment_db = float(12.0 if raw_max_adjustment_db is None else raw_max_adjustment_db)
                if not np.isfinite(reference_window_seconds) or not 0 <= reference_window_seconds <= 60:
                    raise ValueError("Audio loudness reference window must be between 0 and 60 seconds.")
                if not np.isfinite(target_peak_dbfs) or not -9 <= target_peak_dbfs <= 0:
                    raise ValueError("Audio loudness peak ceiling must be between -9 and 0 dBFS.")
                if not np.isfinite(max_adjustment_db) or not 0 <= max_adjustment_db <= 30:
                    raise ValueError("Audio loudness adjustment must be between 0 and 30 dB.")
                result = MatchLoudness().execute(
                    audio=source,
                    reference=reference,
                    reference_window_seconds=reference_window_seconds,
                    target_peak_dbfs=target_peak_dbfs,
                    max_adjustment_db=max_adjustment_db,
                )
        output = result["output"]
        return {
            "output": output,
            "sample_rate": int(output["sample_rate"]),
            "duration": float(output["duration_seconds"]),
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
