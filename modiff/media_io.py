"""Capability-driven media probing, normalization, and delivery exports.

Original user files remain untouched.  Model loaders and download actions may
derive cached representations from those originals, keyed by the source digest
and the complete conversion request.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import subprocess
import tempfile
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from imageio_ffmpeg import get_ffmpeg_exe
from PIL import Image, ImageOps, UnidentifiedImageError, features


MEDIA_CAPABILITY_VERSION = 1
SUPPORTED_AUDIO_SAMPLE_RATES = (44100, 48000, 88200, 96000)
_MEDIA_EXPORT_LOCKS = tuple(threading.Lock() for _ in range(64))


def _media_export_lock(path: Path):
    return _MEDIA_EXPORT_LOCKS[hash(str(path)) % len(_MEDIA_EXPORT_LOCKS)]

_AUDIO_FORMATS = {
    "wav": {
        "label": "WAV",
        "extension": ".wav",
        "mimeType": "audio/wav",
        "encoder": "pcm_s16le",
        "preset": "Lossless",
        "sampleRates": SUPPORTED_AUDIO_SAMPLE_RATES,
    },
    "flac": {
        "label": "FLAC",
        "extension": ".flac",
        "mimeType": "audio/flac",
        "encoder": "flac",
        "preset": "Lossless",
        "sampleRates": SUPPORTED_AUDIO_SAMPLE_RATES,
    },
    "mp3": {
        "label": "MP3",
        "extension": ".mp3",
        "mimeType": "audio/mpeg",
        "encoder": "libmp3lame",
        "preset": "Compatible",
        "sampleRates": (44100, 48000),
    },
    "m4a": {
        "label": "M4A (AAC)",
        "extension": ".m4a",
        "mimeType": "audio/mp4",
        "encoder": "aac",
        "preset": "Compatible",
        "sampleRates": SUPPORTED_AUDIO_SAMPLE_RATES,
    },
    "aac": {
        "label": "AAC",
        "extension": ".aac",
        "mimeType": "audio/aac",
        "encoder": "aac",
        "preset": "Compatible",
        "sampleRates": SUPPORTED_AUDIO_SAMPLE_RATES,
    },
    "opus": {
        "label": "Opus",
        "extension": ".opus",
        "mimeType": "audio/ogg",
        "encoder": "libopus",
        "preset": "Compact",
        # Opus is encoded internally at 48 kHz.  Do not offer a selector that
        # implies a different encoded rate.
        "sampleRates": (48000,),
    },
}

_IMAGE_FORMATS = {
    "png": {
        "label": "PNG",
        "extension": ".png",
        "mimeType": "image/png",
        "pillowFormat": "PNG",
        "preset": "Lossless",
        "supportsAlpha": True,
    },
    "jpeg": {
        "label": "JPEG",
        "extension": ".jpg",
        "mimeType": "image/jpeg",
        "pillowFormat": "JPEG",
        "preset": "Compatible",
        "supportsAlpha": False,
    },
    "webp": {
        "label": "WebP",
        "extension": ".webp",
        "mimeType": "image/webp",
        "pillowFormat": "WEBP",
        "preset": "Compact",
        "supportsAlpha": True,
    },
    "avif": {
        "label": "AVIF",
        "extension": ".avif",
        "mimeType": "image/avif",
        "pillowFormat": "AVIF",
        "preset": "Compact",
        "supportsAlpha": True,
        "feature": "avif",
    },
    "tiff": {
        "label": "TIFF",
        "extension": ".tiff",
        "mimeType": "image/tiff",
        "pillowFormat": "TIFF",
        "preset": "Editing",
        "supportsAlpha": True,
    },
}

_VIDEO_FORMATS = {
    "mp4": {
        "label": "MP4 (H.264 + AAC)",
        "extension": ".mp4",
        "mimeType": "video/mp4",
        "encoders": ("libx264", "aac"),
        "preset": "Compatible",
    },
    "webm": {
        "label": "WebM (VP9 + Opus)",
        "extension": ".webm",
        "mimeType": "video/webm",
        "encoders": ("libvpx-vp9", "libopus"),
        "preset": "Web",
    },
    "mov": {
        "label": "MOV (ProRes + PCM)",
        "extension": ".mov",
        "mimeType": "video/quicktime",
        "encoders": ("prores", "pcm_s16le"),
        "preset": "Editing",
    },
    "gif": {
        "label": "Animated GIF",
        "extension": ".gif",
        "mimeType": "image/gif",
        "encoders": ("gif",),
        "preset": "Loop",
        "hasAudio": False,
    },
}

_IMPORT_EXTENSIONS = {
    "audio": (
        ".wav",
        ".wave",
        ".bwf",
        ".aif",
        ".aiff",
        ".flac",
        ".mp3",
        ".m4a",
        ".aac",
        ".mp4",
        ".ogg",
        ".oga",
        ".opus",
        ".wma",
    ),
    "image": (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".avif",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".ico",
    ),
    "video": (
        ".mp4",
        ".m4v",
        ".mov",
        ".webm",
        ".mkv",
        ".avi",
        ".mpeg",
        ".mpg",
        ".ts",
        ".mts",
        ".m2ts",
        ".wmv",
        ".flv",
    ),
}


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@lru_cache(maxsize=1)
def ffmpeg_encoders() -> frozenset[str]:
    """Return encoder names exposed by the exact bundled FFmpeg binary."""

    result = subprocess.run(
        [get_ffmpeg_exe(), "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        return frozenset()
    names: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.match(r"^\s*[A-Z\.]{6}\s+([^\s]+)", line)
        if match:
            names.add(match.group(1))
    return frozenset(names)


def _format_descriptor(value: str, descriptor: dict[str, Any]) -> dict[str, Any]:
    internal_keys = {"encoder", "encoders", "pillowFormat", "feature"}
    return {
        "value": value,
        **{
            key: list(item) if isinstance(item, tuple) else item
            for key, item in descriptor.items()
            if key not in internal_keys
        },
    }


def media_capabilities() -> dict[str, Any]:
    """Describe only the import and export choices this runtime can fulfill."""

    encoders = ffmpeg_encoders()
    audio = [
        _format_descriptor(value, descriptor)
        for value, descriptor in _AUDIO_FORMATS.items()
        if descriptor["encoder"] in encoders
    ]
    image = [
        _format_descriptor(value, descriptor)
        for value, descriptor in _IMAGE_FORMATS.items()
        if not descriptor.get("feature") or features.check(str(descriptor["feature"]))
    ]
    video = [
        _format_descriptor(value, descriptor)
        for value, descriptor in _VIDEO_FORMATS.items()
        if all(encoder in encoders for encoder in descriptor["encoders"])
    ]
    return {
        "version": MEDIA_CAPABILITY_VERSION,
        "media": {
            "audio": {"importExtensions": list(_IMPORT_EXTENSIONS["audio"]), "exportFormats": audio},
            "image": {"importExtensions": list(_IMPORT_EXTENSIONS["image"]), "exportFormats": image},
            "video": {"importExtensions": list(_IMPORT_EXTENSIONS["video"]), "exportFormats": video},
            "text": {
                "importExtensions": [".txt", ".md", ".json", ".csv", ".srt", ".vtt"],
                "exportFormats": [
                    {
                        "value": "original",
                        "label": "Original",
                        "extension": "",
                        "mimeType": "application/octet-stream",
                        "preset": "Original",
                    }
                ],
            },
        },
    }


def _ffmpeg_probe(path: Path) -> tuple[str | None, str]:
    result = subprocess.run(
        [get_ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path), "-t", "0", "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    detail = result.stderr or result.stdout
    has_video = bool(re.search(r"Stream #.*Video:", detail))
    has_audio = bool(re.search(r"Stream #.*Audio:", detail))
    if has_video:
        return "video", detail
    if has_audio:
        return "audio", detail
    return None, detail


def probe_media_file(path: str | Path, expected_kind: str | None = None) -> dict[str, Any]:
    """Decode enough of a file to classify its real content, not its suffix."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Media file does not exist: {source}")
    expected = str(expected_kind or "").rstrip("s").lower() or None
    metadata: dict[str, Any] = {
        "filename": source.name,
        "extension": source.suffix.lower(),
        "sizeBytes": source.stat().st_size,
        "sha256": _sha256(source),
    }
    try:
        with Image.open(source) as image:
            image.verify()
        with Image.open(source) as image:
            metadata.update(
                {
                    "kind": "image",
                    "mimeType": Image.MIME.get(image.format, mimetypes.guess_type(source.name)[0]),
                    "width": image.width,
                    "height": image.height,
                    "frames": int(getattr(image, "n_frames", 1)),
                }
            )
    except (UnidentifiedImageError, OSError, ValueError):
        kind, detail = _ffmpeg_probe(source)
        if kind is None:
            if expected == "text":
                try:
                    source.read_text(encoding="utf-8")
                    metadata.update({"kind": "text", "mimeType": mimetypes.guess_type(source.name)[0] or "text/plain"})
                except UnicodeDecodeError as exc:
                    raise ValueError("The selected file is not valid UTF-8 text or supported media.") from exc
            else:
                message = detail.strip().splitlines()[-1] if detail.strip() else "unsupported or corrupt media"
                raise ValueError(f"The selected file could not be decoded: {message}")
        else:
            metadata.update({"kind": kind, "mimeType": mimetypes.guess_type(source.name)[0]})
            duration = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", detail)
            if duration:
                metadata["durationSeconds"] = (
                    int(duration.group(1)) * 3600 + int(duration.group(2)) * 60 + float(duration.group(3))
                )
            sample_rate = re.search(r"Audio:.*?(\d+)\s+Hz", detail)
            if sample_rate:
                metadata["sampleRate"] = int(sample_rate.group(1))
            dimensions = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", detail)
            if dimensions:
                metadata["width"] = int(dimensions.group(1))
                metadata["height"] = int(dimensions.group(2))
    if expected and metadata["kind"] != expected:
        raise ValueError(f"The selected file contains {metadata['kind']} data, not {expected} data.")
    return metadata


def _safe_stem(value: str) -> str:
    stem = Path(value).stem
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip(".-")
    return sanitized or "MoDiff-output"


def _validated_sample_rate(value: Any, descriptor: dict[str, Any]) -> int:
    rates = tuple(int(rate) for rate in descriptor.get("sampleRates", ()))
    if not rates:
        raise ValueError("This format does not expose a sample-rate choice.")
    try:
        sample_rate = int(value)
    except (TypeError, ValueError):
        sample_rate = rates[0]
    if sample_rate not in rates:
        readable = ", ".join(f"{rate / 1000:g} kHz" for rate in rates)
        raise ValueError(f"The selected format supports these sample rates: {readable}.")
    return sample_rate


def _ffmpeg_export(source: Path, destination: Path, kind: str, format_id: str, options: dict[str, Any]) -> None:
    command = [get_ffmpeg_exe(), "-y", "-v", "error", "-nostdin", "-i", str(source)]
    if kind == "audio":
        descriptor = _AUDIO_FORMATS[format_id]
        sample_rate = _validated_sample_rate(options.get("sampleRate"), descriptor)
        command.extend(["-vn", "-ar", str(sample_rate)])
        if channels := options.get("channels"):
            command.extend(["-ac", str(max(1, min(8, int(channels))))])
        if format_id == "wav":
            bit_depth = int(options.get("bitDepth") or 16)
            codec = "pcm_s24le" if bit_depth == 24 else "pcm_s16le"
            command.extend(["-c:a", codec])
        elif format_id == "flac":
            command.extend(
                ["-c:a", "flac", "-compression_level", str(max(0, min(12, int(options.get("compression") or 8))))]
            )
        elif format_id == "mp3":
            command.extend(["-c:a", "libmp3lame", "-b:a", f"{max(64, min(320, int(options.get('bitrate') or 320)))}k"])
        elif format_id == "m4a":
            command.extend(
                [
                    "-c:a",
                    "aac",
                    "-b:a",
                    f"{max(64, min(512, int(options.get('bitrate') or 256)))}k",
                    "-movflags",
                    "+faststart",
                ]
            )
        elif format_id == "aac":
            command.extend(["-c:a", "aac", "-b:a", f"{max(64, min(512, int(options.get('bitrate') or 256)))}k"])
        elif format_id == "opus":
            command.extend(["-c:a", "libopus", "-b:a", f"{max(32, min(512, int(options.get('bitrate') or 192)))}k"])
    elif kind == "video":
        if format_id == "mp4":
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    str(options.get("speed") or "medium"),
                    "-crf",
                    str(max(0, min(51, int(options.get("quality") or 18)))),
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                ]
            )
        elif format_id == "webm":
            command.extend(
                [
                    "-c:v",
                    "libvpx-vp9",
                    "-crf",
                    str(max(0, min(63, int(options.get("quality") or 30)))),
                    "-b:v",
                    "0",
                    "-c:a",
                    "libopus",
                    "-b:a",
                    "160k",
                ]
            )
        elif format_id == "mov":
            command.extend(["-c:v", "prores", "-profile:v", "3", "-c:a", "pcm_s16le"])
        elif format_id == "gif":
            fps = max(1, min(30, int(options.get("fps") or 12)))
            width = max(160, min(1920, int(options.get("width") or 640)))
            command.extend(["-an", "-vf", f"fps={fps},scale={width}:-2:flags=lanczos", "-c:v", "gif"])
    command.append(str(destination))
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3600)
    if result.returncode or not destination.is_file():
        destination.unlink(missing_ok=True)
        detail = (result.stderr or result.stdout or "unsupported conversion").strip()
        raise ValueError(f"Could not create the requested {format_id.upper()} file: {detail}")


def _image_export(source: Path, destination: Path, format_id: str, options: dict[str, Any]) -> None:
    descriptor = _IMAGE_FORMATS[format_id]
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        image.load()
        if format_id == "jpeg":
            if image.mode not in {"RGB", "L"}:
                background = Image.new("RGB", image.size, str(options.get("background") or "#ffffff"))
                if image.mode in {"RGBA", "LA"}:
                    background.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
                else:
                    background.paste(image.convert("RGB"))
                image = background
            else:
                image = image.convert("RGB")
        save_options: dict[str, Any] = {"format": descriptor["pillowFormat"]}
        if format_id in {"jpeg", "webp", "avif"}:
            save_options["quality"] = max(1, min(100, int(options.get("quality") or 90)))
        if format_id == "png":
            save_options["compress_level"] = max(0, min(9, int(options.get("compression") or 6)))
        image.save(destination, **save_options)


def export_media_file(
    source: str | Path,
    *,
    kind: str,
    format_id: str,
    options: dict[str, Any] | None = None,
    cache_root: str | Path,
) -> tuple[Path, str, str]:
    """Create or reuse a delivery export and return path, MIME type, filename."""

    source_path = Path(source).expanduser().resolve()
    normalized_kind = str(kind).rstrip("s").lower()
    format_value = str(format_id).lower()
    options = dict(options or {})
    metadata = probe_media_file(source_path, normalized_kind)
    if format_value == "original":
        mime_type = metadata.get("mimeType") or "application/octet-stream"
        return source_path, str(mime_type), source_path.name
    formats = {"audio": _AUDIO_FORMATS, "image": _IMAGE_FORMATS, "video": _VIDEO_FORMATS}.get(normalized_kind)
    if not formats or format_value not in formats:
        raise ValueError(f"{format_value or 'The requested format'} is not available for {normalized_kind}.")
    available = {item["value"] for item in media_capabilities()["media"][normalized_kind]["exportFormats"]}
    if format_value not in available:
        raise ValueError(f"{formats[format_value]['label']} export is not available in this MoDiff runtime.")
    descriptor = formats[format_value]
    request_key = json.dumps(
        {"source": metadata["sha256"], "kind": normalized_kind, "format": format_value, "options": options},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(request_key.encode("utf-8")).hexdigest()
    destination_root = Path(cache_root).expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_stem(source_path.name)}-{digest[:10]}{descriptor['extension']}"
    destination = destination_root / filename
    if not destination.is_file():
        with _media_export_lock(destination):
            if not destination.is_file():
                with tempfile.NamedTemporaryFile(
                    dir=destination_root,
                    prefix=f".{destination.stem}-",
                    suffix=destination.suffix,
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                try:
                    if normalized_kind == "image":
                        _image_export(source_path, temporary, format_value, options)
                    else:
                        _ffmpeg_export(source_path, temporary, normalized_kind, format_value, options)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
    return destination, str(descriptor["mimeType"]), filename


def export_media_bytes(
    body: bytes,
    *,
    source_suffix: str,
    kind: str,
    format_id: str,
    options: dict[str, Any] | None,
    cache_root: str | Path,
) -> tuple[Path, str, str]:
    """Export in-memory node output through the same file-backed contract."""

    digest = hashlib.sha256(body).hexdigest()
    source_root = Path(cache_root).expanduser().resolve() / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    suffix = source_suffix if str(source_suffix).startswith(".") else f".{source_suffix}"
    source = source_root / f"{digest}{suffix}"
    if not source.is_file():
        with tempfile.NamedTemporaryFile(dir=source_root, prefix=".source-", suffix=suffix, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(body)
        temporary.replace(source)
    return export_media_file(
        source,
        kind=kind,
        format_id=format_id,
        options=options,
        cache_root=Path(cache_root) / "exports",
    )
