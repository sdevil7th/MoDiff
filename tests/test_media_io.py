from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.io import wavfile

from modiff.media_io import (
    export_media_file,
    get_ffmpeg_exe,
    media_capabilities,
    probe_media_file,
)


def _write_tone(path: Path, sample_rate: int = 48000) -> None:
    timeline = np.arange(sample_rate // 10, dtype=np.float32) / sample_rate
    samples = (np.sin(timeline * 440 * 2 * np.pi) * 0.2 * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, samples)


def _write_video(path: Path, frame_root: Path) -> None:
    frame_root.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        Image.new("RGB", (160, 120), (20 + index * 40, 70, 160 - index * 20)).save(
            frame_root / f"frame-{index:02d}.png"
        )
    subprocess.run(
        [
            get_ffmpeg_exe(),
            "-y",
            "-v",
            "error",
            "-nostdin",
            "-framerate",
            "4",
            "-i",
            str(frame_root / "frame-%02d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )


def test_capabilities_only_advertise_runtime_backed_popular_formats():
    capabilities = media_capabilities()

    assert capabilities["version"] == 1
    assert {item["value"] for item in capabilities["media"]["audio"]["exportFormats"]} >= {
        "wav",
        "flac",
        "mp3",
        "m4a",
        "aac",
        "opus",
    }
    assert {item["value"] for item in capabilities["media"]["image"]["exportFormats"]} >= {
        "png",
        "jpeg",
        "webp",
    }
    assert {item["value"] for item in capabilities["media"]["video"]["exportFormats"]} >= {
        "mp4",
        "webm",
        "mov",
        "gif",
    }


def test_probe_uses_content_instead_of_extension(tmp_path):
    source = tmp_path / "misleading.bin"
    Image.new("RGBA", (23, 17), (10, 20, 30, 128)).save(source, format="PNG")

    metadata = probe_media_file(source, "image")

    assert metadata["kind"] == "image"
    assert metadata["mimeType"] == "image/png"
    assert metadata["width"] == 23
    assert metadata["height"] == 17
    with pytest.raises(ValueError, match="not audio"):
        probe_media_file(source, "audio")


@pytest.mark.parametrize("format_id", ["wav", "flac", "mp3", "m4a", "aac", "opus"])
def test_audio_delivery_exports_are_real_decodable_files(tmp_path, format_id):
    source = tmp_path / "source.wav"
    _write_tone(source)

    exported, mime_type, filename = export_media_file(
        source,
        kind="audio",
        format_id=format_id,
        options={"sampleRate": 48000 if format_id == "opus" else 44100},
        cache_root=tmp_path / "exports",
    )

    assert exported.is_file()
    assert exported.stat().st_size > 0
    assert filename.endswith(exported.suffix)
    assert mime_type.startswith("audio/")
    metadata = probe_media_file(exported, "audio")
    assert metadata["kind"] == "audio"
    expected_rate = 48000 if format_id == "opus" else 44100
    assert metadata["sampleRate"] == expected_rate


def test_wav_export_encodes_selected_rate_in_file_header(tmp_path):
    source = tmp_path / "source.wav"
    _write_tone(source)

    exported, _, _ = export_media_file(
        source,
        kind="audio",
        format_id="wav",
        options={"sampleRate": 88200, "bitDepth": 24},
        cache_root=tmp_path / "exports",
    )

    with wave.open(str(exported), "rb") as handle:
        assert handle.getframerate() == 88200
        assert handle.getsampwidth() == 3


@pytest.mark.parametrize("format_id", ["png", "jpeg", "webp", "avif", "tiff"])
def test_image_delivery_exports_are_decodable(tmp_path, format_id):
    available = {item["value"] for item in media_capabilities()["media"]["image"]["exportFormats"]}
    if format_id not in available:
        pytest.skip(f"{format_id} is not available in this Pillow runtime")
    source = tmp_path / "source.png"
    Image.new("RGBA", (31, 19), (10, 20, 30, 128)).save(source)

    exported, mime_type, _ = export_media_file(
        source,
        kind="image",
        format_id=format_id,
        options={"quality": 84},
        cache_root=tmp_path / "exports",
    )

    with Image.open(exported) as image:
        assert image.size == (31, 19)
    assert mime_type.startswith("image/")


@pytest.mark.parametrize(
    ("format_id", "suffix", "mime_type"),
    [
        ("mp4", ".mp4", "video/mp4"),
        ("webm", ".webm", "video/webm"),
        ("mov", ".mov", "video/quicktime"),
        ("gif", ".gif", "image/gif"),
    ],
)
def test_video_delivery_exports_are_real_decodable_files(tmp_path, format_id, suffix, mime_type):
    available = {item["value"] for item in media_capabilities()["media"]["video"]["exportFormats"]}
    if format_id not in available:
        pytest.skip(f"{format_id} is not available in this FFmpeg runtime")
    source = tmp_path / "source.mp4"
    _write_video(source, tmp_path / "frames")

    exported, actual_mime_type, filename = export_media_file(
        source,
        kind="video",
        format_id=format_id,
        options={"quality": 24, "fps": 4, "width": 160},
        cache_root=tmp_path / "exports",
    )

    assert exported.is_file()
    assert exported.stat().st_size > 0
    assert exported.suffix == suffix
    assert filename.endswith(suffix)
    assert actual_mime_type == mime_type
    if format_id == "gif":
        with Image.open(exported) as image:
            assert image.is_animated
            assert image.size == (160, 120)
    else:
        metadata = probe_media_file(exported, "video")
        assert metadata["kind"] == "video"
        assert metadata["width"] == 160
        assert metadata["height"] == 120


def test_export_cache_key_includes_all_delivery_options(tmp_path):
    source = tmp_path / "source.wav"
    _write_tone(source)

    first, _, _ = export_media_file(
        source,
        kind="audio",
        format_id="wav",
        options={"sampleRate": 44100},
        cache_root=tmp_path / "exports",
    )
    same, _, _ = export_media_file(
        source,
        kind="audio",
        format_id="wav",
        options={"sampleRate": 44100},
        cache_root=tmp_path / "exports",
    )
    different, _, _ = export_media_file(
        source,
        kind="audio",
        format_id="wav",
        options={"sampleRate": 48000},
        cache_root=tmp_path / "exports",
    )

    assert same == first
    assert different != first
