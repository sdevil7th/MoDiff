import numpy as np
import pytest
from PIL import Image

from modiff import media_assets
from modules.Video.main import (
    Compose,
    ConcatenateAssets,
    Export,
    ExportAsset,
    ExportWithAudio,
    LyricOverlay,
    MuxAudioAsset,
)
from modules import MODULE_MAP


def solid(color, count=16, size=(160, 90)):
    return [Image.new("RGB", size, color) for _ in range(count)]


def test_compose_crossfades_model_agnostic_frame_lists():
    result = Compose("compose-test").execute(
        clip_1=solid("red"),
        clip_2=solid("blue"),
        transition_seconds=0.25,
        fps=16,
    )
    assert result["frames"] == 28
    assert result["duration_seconds"] == 1.75
    assert result["video"][0].getpixel((0, 0)) == (255, 0, 0)
    assert result["video"][-1].getpixel((0, 0)) == (0, 0, 255)


def test_compose_inputs_are_visible_to_static_node_registry():
    params = MODULE_MAP["modules.Video"]["Compose"]["params"]
    assert params["clip_1"]["display"] == "input"
    assert "video_collection" in params["clip_1"]["type"]


def test_plain_video_export_uses_delivery_quality_by_default():
    assert Export.params["quality"]["default"] == 8


def test_lyric_overlay_requires_and_renders_lrc_timeline():
    result = LyricOverlay("lyrics-test").execute(
        video=solid("black", count=20),
        lrc="[00:00.00]First line\n[00:00.50]Second line",
        fps=10,
        font_size=18,
        bottom_margin=12,
    )
    assert len(result["output"]) == 20
    assert np.asarray(result["output"][0]).max() > 0
    assert np.asarray(result["output"][10]).max() > 0


def test_export_with_audio_muxes_mp4(tmp_path):
    samples = np.zeros((48000, 2), dtype=np.float32)
    samples[:, 0] = 0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi, 48000))
    samples[:, 1] = samples[:, 0]
    output = tmp_path / "lyric.mp4"
    result = ExportWithAudio("export-av-test").execute(
        video=solid("navy", count=16),
        audio={"samples": samples, "sample_rate": 48000},
        filename=str(output),
        fps=16,
        quality=5,
    )
    assert output.exists()
    assert output.stat().st_size > 0
    assert result["frames"] == 16
    assert result["duration_seconds"] == 1


@pytest.mark.parametrize(("clip_count", "fps"), [(2, 8), (3, 8), (3, 16)])
def test_retained_segments_stitch_and_mux_audio_without_materializing_the_join(tmp_path, monkeypatch, clip_count, fps):
    monkeypatch.setattr(media_assets, "asset_root", lambda root=None: tmp_path)
    first = ExportAsset("retain-red").execute(video=solid("red", count=fps, size=(32, 24)), fps=fps, pin=True)
    second = ExportAsset("retain-blue").execute(video=solid("blue", count=fps, size=(32, 24)), fps=fps, pin=True)

    joined = ConcatenateAssets("join-segments").execute(
        clips=[first["asset"], second["asset"], first["asset"]][:clip_count],
        transition_seconds=0.25,
        pin=True,
    )
    expected_duration = clip_count - 0.25 * (clip_count - 1)
    samples = np.zeros((int(expected_duration * 48000), 1), dtype=np.float32)
    muxed = MuxAudioAsset("mux-segments").execute(
        video=joined["asset"],
        audio={"samples": samples, "sample_rate": 48000},
        fit="match_video",
        pin=True,
    )

    assert joined["frames"] == int(expected_duration * fps)
    assert joined["duration_seconds"] == expected_duration
    assert muxed["frames"] == joined["frames"]
    assert muxed["duration_seconds"] == joined["duration_seconds"]
    assert media_assets.coerce_video_asset(muxed["asset"])["path"] == muxed["file"]
