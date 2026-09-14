"""Real export regressions for the NumPy batches returned by Helios decode."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from modules.Video.main import Export


def test_helios_numpy_batch_writes_each_clip_and_preserves_frames(tmp_path):
    videos = np.zeros((2, 5, 16, 16, 3), dtype=np.float32)
    videos[0, :, :, :, 0] = 1.0
    videos[1, :, :, :, 2] = 1.0
    result = Export().execute(video=videos, filename=str(tmp_path / "clip_{HASH:6}.mp4"), fps=5, quality=8)
    assert len(result["file"]) == 2
    assert len(set(result["file"])) == 2
    assert (result["width"], result["height"], result["frames"]) == (16, 16, 5)
    for index, filename in enumerate(result["file"]):
        assert Path(filename).is_file()
        reader = imageio.get_reader(filename)
        try:
            decoded = list(reader.iter_data())
            assert len(decoded) == 5
            assert reader.get_meta_data()["fps"] == 5
            assert decoded[0].shape == (16, 16, 3)
            assert decoded[0][:, :, (0, 2)[index]].mean() > 240
        finally:
            reader.close()


@pytest.mark.parametrize(
    "video", [None, [], {}, np.zeros((0, 5, 16, 16, 3)), np.zeros((1, 0, 16, 16, 3)), np.zeros((2, 3))]
)
def test_empty_or_unsupported_export_does_not_publish_a_nonexistent_file(tmp_path, video):
    with pytest.raises(ValueError, match="Export Video"):
        Export().execute(video=video, filename=str(tmp_path / "missing.mp4"))
    assert not (tmp_path / "missing.mp4").exists()


def test_existing_frame_list_export_still_writes_video(tmp_path):
    frames = [np.full((16, 16, 3), 127, dtype=np.uint8) for _ in range(3)]
    result = Export().execute(video=frames, filename=str(tmp_path / "frames.mp4"), fps=3)
    assert Path(result["file"]).is_file()
    assert (result["width"], result["height"], result["frames"]) == (16, 16, 3)


def test_encoder_without_output_cannot_report_success(tmp_path, monkeypatch):
    class EmptyWriter:
        def append_data(self, frame):
            pass

        def close(self):
            pass

    monkeypatch.setattr("imageio.get_writer", lambda *args, **kwargs: EmptyWriter())
    with pytest.raises(RuntimeError, match="did not produce"):
        Export().execute(video=[np.zeros((16, 16, 3), dtype=np.uint8)], filename=str(tmp_path / "missing.mp4"))
