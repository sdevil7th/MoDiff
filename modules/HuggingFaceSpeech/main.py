"""Generic local speech-recognition nodes backed by Hugging Face Transformers."""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

from modiff.NodeBase import NodeBase
from modiff.model_artifact_catalog import IMMUTABLE_HUB_REVISION, catalog_revision
from utils.huggingface import local_files_only, validate_hf_repo_id
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST, str_to_dtype


WHISPER_TINY_REPO = "openai/whisper-tiny"
WHISPER_TINY_MODEL_TYPE = "HuggingFaceSpeechRecognitionModel"
WAV2VEC2_BASE_960H_REPO = "facebook/wav2vec2-base-960h"
WAV2VEC2_CTC_MODEL_TYPE = "HuggingFaceCTCSpeechRecognitionModel"
SPEECH_MODEL_FAMILY_SEQ2SEQ = "sequence_to_sequence"
SPEECH_MODEL_FAMILY_CTC = "ctc"
SPEECH_TASKS = {"transcribe", "translate"}
TIMESTAMP_MODES = {"none", "segment", "word"}
CTC_TIMESTAMP_MODES = {"none", "word"}
MAX_AUDIO_FILE_BYTES = 512 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 3600.0
MAX_AUDIO_SAMPLE_RATE = 192_000
MAX_TRANSCRIPT_CHARACTERS = 1_000_000
MAX_TRANSCRIPT_SEGMENTS = 100_000


def _model_selection(value: Any, *, default_repo: str = WHISPER_TINY_REPO) -> dict[str, str]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return {"source": "hub", "value": default_repo}
    if isinstance(value, str):
        source, selected = "hub", value.strip()
    elif isinstance(value, dict) and set(value).issubset({"source", "value"}):
        source = str(value.get("source") or "").strip().casefold()
        selected = str(value.get("value") or "").strip()
    else:
        raise ValueError("Speech model selection must be a repository ID or hub/local selection object.")
    if source == "hub":
        if selected.count("/") != 1:
            raise ValueError("Speech model repository must use the namespace/repository form.")
        validate_hf_repo_id(selected)
        return {
            "source": "hub",
            "value": default_repo if selected.casefold() == default_repo.casefold() else selected,
        }
    if source != "local":
        raise ValueError("Speech model source must be exactly hub or local.")
    try:
        resolved = Path(selected).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("Local speech model directory does not exist.") from error
    if not resolved.is_dir():
        raise ValueError("Local speech model selection must be a directory.")
    return {"source": "local", "value": str(resolved)}


def _model_revision(selection: dict[str, str], value: Any) -> str | None:
    if selection["source"] == "local":
        if value not in (None, ""):
            raise ValueError("A local speech model must not carry a Hub revision.")
        return None
    revision = str(value or catalog_revision(selection["value"]) or "").strip()
    if revision != revision.lower() or not IMMUTABLE_HUB_REVISION.fullmatch(revision):
        raise ValueError("Speech model revision must be an exact lowercase 40-character commit SHA.")
    reviewed = catalog_revision(selection["value"])
    if reviewed is not None and revision != reviewed:
        raise ValueError("Speech model revision does not match the reviewed catalog pin.")
    return revision


def _bounded_float(value: Any, *, field: str, default: float, minimum: float, maximum: float) -> float:
    raw = default if value is None else value
    if isinstance(raw, bool):
        raise ValueError(f"{field} must be a number.")
    try:
        parsed = float(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must be a number.") from error
    if not isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be finite and between {minimum} and {maximum}.")
    return parsed


def _normalized_audio(value: Any) -> tuple[np.ndarray, int, float]:
    from modules.Audio.main import _audio_to_numpy
    from modules.DiffusersAudio.main import _resolve_audio_source_path

    if value is None:
        raise ValueError("Transcribe Audio requires a local audio source.")
    selected = value
    if isinstance(value, (str, Path)):
        path = _resolve_audio_source_path(value)
        try:
            size = path.stat().st_size
        except OSError as error:
            raise ValueError("Speech audio source could not be inspected.") from error
        if size <= 0 or size > MAX_AUDIO_FILE_BYTES:
            raise ValueError("Speech audio files must be nonempty and at most 512 MiB.")
        selected = str(path)
    samples, sample_rate = _audio_to_numpy(selected)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[0] <= 0 or samples.shape[1] <= 0 or samples.shape[1] > 8:
        raise ValueError("Speech audio must contain one to eight channels of nonempty samples.")
    if not np.isfinite(samples).all():
        raise ValueError("Speech audio samples must be finite.")
    if sample_rate < 8_000 or sample_rate > MAX_AUDIO_SAMPLE_RATE:
        raise ValueError("Speech audio sample rate must be between 8 kHz and 192 kHz.")
    duration = float(samples.shape[0] / sample_rate)
    if not isfinite(duration) or duration <= 0 or duration > MAX_AUDIO_DURATION_SECONDS:
        raise ValueError("Speech audio duration must be positive and at most one hour.")
    mono = np.mean(samples, axis=1, dtype=np.float32)
    return np.ascontiguousarray(np.clip(mono, -1.0, 1.0)), int(sample_rate), duration


def _normalize_transcript_result(result: Any, *, task: str, timestamp_mode: str, duration: float) -> dict[str, Any]:
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise RuntimeError("Speech recognition returned an invalid transcript object.")
    text = result["text"].strip()
    if len(text) > MAX_TRANSCRIPT_CHARACTERS:
        raise RuntimeError("Speech recognition output exceeded the transcript size limit.")
    raw_chunks = result.get("chunks", [])
    if raw_chunks is None:
        raw_chunks = []
    if not isinstance(raw_chunks, list) or len(raw_chunks) > MAX_TRANSCRIPT_SEGMENTS:
        raise RuntimeError("Speech recognition output exceeded the segment count limit.")
    segments = []
    segment_characters = 0
    for chunk in raw_chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            raise RuntimeError("Speech recognition returned an invalid timestamp segment.")
        timestamp = chunk.get("timestamp")
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            raise RuntimeError("Speech recognition returned an invalid timestamp interval.")
        start, end = timestamp
        start = 0.0 if start is None else float(start)
        end = duration if end is None else float(end)
        if not isfinite(start) or not isfinite(end) or start < 0 or end < start or end > duration + 1.0:
            raise RuntimeError("Speech recognition returned an out-of-range timestamp interval.")
        chunk_text = chunk["text"].strip()
        segment_characters += len(chunk_text)
        if segment_characters > MAX_TRANSCRIPT_CHARACTERS:
            raise RuntimeError("Speech recognition output exceeded the transcript size limit.")
        segments.append({"text": chunk_text, "start": start, "end": min(end, duration)})
    return {
        "schemaVersion": 1,
        "task": task,
        "text": text,
        "timestampMode": timestamp_mode,
        "durationSeconds": duration,
        "segments": segments,
    }


class LoadSpeechRecognitionModel(NodeBase):
    """Load one reviewed local automatic-speech-recognition model."""

    label = "Load Speech Recognition Model"
    category = "Hugging Face Speech"
    resizable = True
    params = {
        "model": {"label": "Model", "display": "output", "type": "speech_recognition_model"},
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": WHISPER_TINY_REPO},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": ["WhisperForConditionalGeneration"]},
                    "local": {"className": ["WhisperForConditionalGeneration"]},
                },
            },
        },
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            "default": "AutoModelForSpeechSeq2Seq",
            "hidden": True,
            "fieldOptions": {"noValidation": True},
        },
        "execution_profile_id": {
            "label": "Execution Profile",
            "type": "string",
            "default": "",
            "hidden": True,
            "fieldOptions": {"noValidation": True},
        },
        "revision": {"label": "Revision", "type": "string", "default": ""},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "float32",
        },
        "device": {"label": "Device", "type": "string", "options": DEVICE_LIST, "default": DEFAULT_DEVICE},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        import transformers

        selection = _model_selection(kwargs.get("model_id"))
        model_id = selection["value"]
        revision = _model_revision(selection, kwargs.get("revision"))
        dtype = str_to_dtype(kwargs.get("dtype") or "float32")
        device = str(kwargs.get("device") or DEFAULT_DEVICE)
        offline = local_files_only(model_id)
        common = {
            "revision": revision,
            "local_files_only": offline,
            "trust_remote_code": False,
        }
        processor = transformers.AutoProcessor.from_pretrained(model_id, **common)
        model = transformers.AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            **common,
        )
        model.to(device)
        model.eval()
        recognizer = transformers.pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )
        return {
            "model": {
                "schemaVersion": 1,
                "pipeline": recognizer,
                "repository": model_id,
                "revision": revision,
                "source": selection["source"],
                "device": device,
                "family": SPEECH_MODEL_FAMILY_SEQ2SEQ,
            },
            "resolved_artifact": model_id,
        }


class LoadCTCSpeechRecognitionModel(NodeBase):
    """Load the reviewed Wav2Vec2 CTC automatic-speech-recognition model."""

    label = "Load CTC Speech Recognition Model"
    category = "Hugging Face Speech"
    resizable = True
    params = {
        "model": {"label": "Model", "display": "output", "type": "speech_recognition_model"},
        "model_id": {
            "label": "Model",
            "display": "modelselect",
            "type": "string",
            "value": {"source": "hub", "value": WAV2VEC2_BASE_960H_REPO},
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {
                    "hub": {"className": ["Wav2Vec2ForCTC"]},
                    "local": {"className": ["Wav2Vec2ForCTC"]},
                },
            },
        },
        "pipeline_class": {
            "label": "Pipeline Class",
            "type": "string",
            "default": "AutoModelForCTC",
            "hidden": True,
            "fieldOptions": {"noValidation": True},
        },
        "execution_profile_id": {
            "label": "Execution Profile",
            "type": "string",
            "default": "",
            "hidden": True,
            "fieldOptions": {"noValidation": True},
        },
        "revision": {"label": "Revision", "type": "string", "default": ""},
        "dtype": {
            "label": "DType",
            "type": "string",
            "options": ["float32", "float16", "bfloat16"],
            "default": "float32",
        },
        "device": {"label": "Device", "type": "string", "options": DEVICE_LIST, "default": DEFAULT_DEVICE},
        "resolved_artifact": {"label": "Resolved Artifact", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        import transformers

        selection = _model_selection(kwargs.get("model_id"), default_repo=WAV2VEC2_BASE_960H_REPO)
        model_id = selection["value"]
        revision = _model_revision(selection, kwargs.get("revision"))
        dtype = str_to_dtype(kwargs.get("dtype") or "float32")
        device = str(kwargs.get("device") or DEFAULT_DEVICE)
        offline = local_files_only(model_id)
        common = {
            "revision": revision,
            "local_files_only": offline,
            "trust_remote_code": False,
        }
        processor = transformers.AutoProcessor.from_pretrained(model_id, **common)
        model = transformers.AutoModelForCTC.from_pretrained(
            model_id,
            dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            **common,
        )
        model.to(device)
        model.eval()
        recognizer = transformers.pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )
        return {
            "model": {
                "schemaVersion": 1,
                "pipeline": recognizer,
                "repository": model_id,
                "revision": revision,
                "source": selection["source"],
                "device": device,
                "family": SPEECH_MODEL_FAMILY_CTC,
            },
            "resolved_artifact": model_id,
        }


class TranscribeAudio(NodeBase):
    """Transcribe or translate bounded local audio into a normalized transcript."""

    label = "Transcribe Audio"
    category = "Hugging Face Speech"
    resizable = True
    params = {
        "model": {
            "label": "Model",
            "display": "input",
            "type": "speech_recognition_model",
            "required": True,
        },
        "audio": {"label": "Audio", "display": "input", "type": "audio", "required": True},
        "task": {"label": "Task", "type": "string", "options": ["transcribe", "translate"], "default": "transcribe"},
        "language": {"label": "Language hint", "type": "string", "default": ""},
        "timestamps": {
            "label": "Timestamps",
            "type": "string",
            "options": ["none", "segment", "word"],
            "default": "segment",
        },
        "chunk_length_seconds": {
            "label": "Chunk length",
            "type": "float",
            "default": 30.0,
            "min": 0.0,
            "max": 30.0,
        },
        "stride_length_seconds": {
            "label": "Chunk stride",
            "type": "float",
            "default": 5.0,
            "min": 0.0,
            "max": 10.0,
        },
        "transcript": {"label": "Transcript", "display": "output", "type": "any"},
        "text": {"label": "Text", "display": "output", "type": "string"},
        "segments": {"label": "Segments", "display": "output", "type": "collection"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        model = kwargs.get("model")
        if (
            not isinstance(model, dict)
            or model.get("schemaVersion") != 1
            or model.get("family") != SPEECH_MODEL_FAMILY_SEQ2SEQ
            or not callable(model.get("pipeline"))
        ):
            raise ValueError("Transcribe Audio requires a model from Load Speech Recognition Model.")
        task = str(kwargs.get("task") or "transcribe").strip().casefold()
        if task not in SPEECH_TASKS:
            raise ValueError("Speech task must be exactly transcribe or translate.")
        timestamp_mode = str(kwargs.get("timestamps") or "segment").strip().casefold()
        if timestamp_mode not in TIMESTAMP_MODES:
            raise ValueError("Speech timestamps must be exactly none, segment, or word.")
        language = str(kwargs.get("language") or "").strip()
        if len(language) > 64 or any(character in language for character in "\r\n\0"):
            raise ValueError("Speech language hint must be at most 64 characters on one line.")
        chunk_length = _bounded_float(
            kwargs.get("chunk_length_seconds"), field="Speech chunk length", default=30.0, minimum=0.0, maximum=30.0
        )
        stride_length = _bounded_float(
            kwargs.get("stride_length_seconds"), field="Speech chunk stride", default=5.0, minimum=0.0, maximum=10.0
        )
        if chunk_length and (chunk_length < 5.0 or stride_length * 2 >= chunk_length):
            raise ValueError("Speech chunk length must be 5-30 seconds with total stride smaller than the chunk.")
        if not chunk_length and stride_length:
            raise ValueError("Speech chunk stride must be zero when chunking is disabled.")
        samples, sample_rate, duration = _normalized_audio(kwargs.get("audio"))
        generate_kwargs = {"task": task}
        if language:
            generate_kwargs["language"] = language
        call_kwargs: dict[str, Any] = {
            "generate_kwargs": generate_kwargs,
            "return_timestamps": False if timestamp_mode == "none" else "word" if timestamp_mode == "word" else True,
        }
        if chunk_length:
            call_kwargs["chunk_length_s"] = chunk_length
            call_kwargs["stride_length_s"] = stride_length
        result = model["pipeline"]({"raw": samples, "sampling_rate": sample_rate}, **call_kwargs)
        transcript = _normalize_transcript_result(
            result,
            task=task,
            timestamp_mode=timestamp_mode,
            duration=duration,
        )
        return {
            "transcript": transcript,
            "text": transcript["text"],
            "segments": transcript["segments"],
            "duration_seconds": duration,
        }


class TranscribeCTCAudio(NodeBase):
    """Transcribe bounded local audio with a CTC model without generative controls."""

    label = "Transcribe CTC Audio"
    category = "Hugging Face Speech"
    resizable = True
    params = {
        "model": {
            "label": "Model",
            "display": "input",
            "type": "speech_recognition_model",
            "required": True,
        },
        "audio": {"label": "Audio", "display": "input", "type": "audio", "required": True},
        "timestamps": {
            "label": "Timestamps",
            "type": "string",
            "options": ["none", "word"],
            "default": "word",
        },
        "chunk_length_seconds": {
            "label": "Chunk length",
            "type": "float",
            "default": 30.0,
            "min": 0.0,
            "max": 30.0,
        },
        "stride_length_seconds": {
            "label": "Chunk stride",
            "type": "float",
            "default": 5.0,
            "min": 0.0,
            "max": 10.0,
        },
        "transcript": {"label": "Transcript", "display": "output", "type": "any"},
        "text": {"label": "Text", "display": "output", "type": "string"},
        "segments": {"label": "Segments", "display": "output", "type": "collection"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        model = kwargs.get("model")
        if (
            not isinstance(model, dict)
            or model.get("schemaVersion") != 1
            or model.get("family") != SPEECH_MODEL_FAMILY_CTC
            or not callable(model.get("pipeline"))
        ):
            raise ValueError("Transcribe CTC Audio requires a model from Load CTC Speech Recognition Model.")
        timestamp_mode = str(kwargs.get("timestamps") or "word").strip().casefold()
        if timestamp_mode not in CTC_TIMESTAMP_MODES:
            raise ValueError("CTC speech timestamps must be exactly none or word.")
        chunk_length = _bounded_float(
            kwargs.get("chunk_length_seconds"),
            field="Speech chunk length",
            default=30.0,
            minimum=0.0,
            maximum=30.0,
        )
        stride_length = _bounded_float(
            kwargs.get("stride_length_seconds"),
            field="Speech chunk stride",
            default=5.0,
            minimum=0.0,
            maximum=10.0,
        )
        if chunk_length and (chunk_length < 5.0 or stride_length * 2 >= chunk_length):
            raise ValueError("Speech chunk length must be 5-30 seconds with total stride smaller than the chunk.")
        if not chunk_length and stride_length:
            raise ValueError("Speech chunk stride must be zero when chunking is disabled.")
        samples, sample_rate, duration = _normalized_audio(kwargs.get("audio"))
        call_kwargs: dict[str, Any] = {
            "return_timestamps": "word" if timestamp_mode == "word" else False,
        }
        if chunk_length:
            call_kwargs["chunk_length_s"] = chunk_length
            call_kwargs["stride_length_s"] = stride_length
        result = model["pipeline"]({"raw": samples, "sampling_rate": sample_rate}, **call_kwargs)
        transcript = _normalize_transcript_result(
            result,
            task="transcribe",
            timestamp_mode=timestamp_mode,
            duration=duration,
        )
        return {
            "transcript": transcript,
            "text": transcript["text"],
            "segments": transcript["segments"],
            "duration_seconds": duration,
        }
