"""Reviewed video ports retain media semantics despite upstream PIL frame types."""
from modules.ModularDiffusers.reviewed_blocks import _reviewed_block_input_params


def test_video_ports_remain_video_and_image_ports_remain_image():
    params = _reviewed_block_input_params()
    assert params["driving_video"]["type"] == "video"
    assert params["video"]["type"] == "video"
    assert params["image"]["type"] == "image"
    assert params["driving_video"]["display"] == "input"
    assert params["driving_video"]["default"] is None
