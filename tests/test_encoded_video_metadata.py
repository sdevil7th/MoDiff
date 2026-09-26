"""Encoded geometry must be measured after the codec's macroblock resize."""
import copy
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from modiff.server import WebServer
from modules.Video.main import Export


@pytest.mark.parametrize("nested", [False, True])
def test_export_reports_encoded_geometry_without_changing_encoder_defaults(tmp_path, nested):
    frames = [np.full((23, 17, 3), 100, dtype=np.uint8) for _ in range(3)]
    result = Export().execute(video=[frames] if nested else frames,
                              filename=str(tmp_path / "odd.mp4"), fps=3)
    filename = result["file"][0] if nested else result["file"]
    reader = imageio.get_reader(filename)
    try:
        actual = reader.get_meta_data()["size"]
        assert actual == (32, 32)  # Existing imageio macroblock behavior retained.
        assert (result["width"], result["height"]) == actual
        assert result["frames"] == reader.count_frames() == 3
    finally:
        reader.close()


def test_retained_video_metadata_separates_encoded_size_from_requested_controls(tmp_path, monkeypatch):
    clip = tmp_path / "source.mp4"
    writer = imageio.get_writer(clip, fps=3, codec="libx264")
    try:
        for _ in range(3):
            writer.append_data(np.full((23, 17, 3), 100, dtype=np.uint8))
    finally:
        writer.close()
    server = WebServer.__new__(WebServer)
    server.data_dir = tmp_path / "data"
    server.current_task = None
    monkeypatch.setattr(server, "_share_media_bytes_from_url", lambda url: {
        "bytes": clip.read_bytes(), "contentType": "video/mp4", "filename": "source.mp4"})
    original = {"id": "encoded-regression", "displayType": "video", "width": 640, "height": 800,
                "formSnapshot": {"width": 640, "height": 800},
                "mediaItems": [{"index": 0, "url": "local-fixture", "displayType": "video"}]}
    result = server._normalize_studio_output(copy.deepcopy(original))
    assert result["width"] == 640 and result["height"] == 800
    assert result["formSnapshot"] == original["formSnapshot"]
    metadata = result["mediaItems"][0]["mediaMetadata"]
    assert metadata["source"] == "encoded-file"
    assert (metadata["width"], metadata["height"], metadata["frame_count"]) == (32, 32, 3)
    assert metadata["fps"] == 3
    assert result["backendProvenance"]["mediaItems"][0]["mediaMetadata"] == metadata
    retained = next((Path(server.data_dir) / "studio" / "outputs").glob("*.mp4"))
    assert retained.read_bytes() == clip.read_bytes()
