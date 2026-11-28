"""Image-save failures stay actionable and cannot become successful cache hits."""

import pytest
from PIL import Image

from modules.Image.main import Save


@pytest.mark.parametrize("source", [None, [], [None]])
def test_save_refuses_an_empty_output_set_with_an_actionable_error(source, tmp_path):
    with pytest.raises(RuntimeError, match="Save Image could not save any selected image") as caught:
        Save("empty-save")(image=source, filename=str(tmp_path / "empty.png"))
    assert isinstance(caught.value.__cause__, ValueError)
    assert list(tmp_path.iterdir()) == []


def test_save_reports_an_unsupported_format_and_can_retry_without_reusing_failure(tmp_path):
    node = Save("retry-save")
    image = Image.new("RGB", (3, 2), (10, 20, 30))
    with pytest.raises(RuntimeError, match="Save Image could not save any selected image") as caught:
        node(image=image, filename=str(tmp_path / "image.unsupported"))
    assert isinstance(caught.value.__cause__, ValueError)
    assert isinstance(caught.value.__cause__.__cause__, ValueError)
    assert not node._cache_valid
    target = tmp_path / "image.png"
    result = node(image=image, filename=str(target))
    assert result["output"] == str(target)
    assert result["saved_image_data"].tobytes() == image.tobytes()
    with Image.open(target) as saved:
        assert saved.tobytes() == image.tobytes()


def test_save_retains_the_filesystem_failure_as_its_cause(tmp_path):
    target = tmp_path / "already-a-directory.png"
    target.mkdir()
    node = Save("write-failure")
    with pytest.raises(RuntimeError, match="Save Image could not save any selected image") as caught:
        node(image=Image.new("RGB", (1, 1)), filename=str(target))
    assert isinstance(caught.value.__cause__, ValueError)
    assert isinstance(caught.value.__cause__.__cause__, OSError)
    assert not node._cache_valid
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_save_retains_batch_order_and_independent_saved_images(tmp_path):
    images = [Image.new("RGB", (3, 2), "red"), Image.new("RGB", (3, 2), "blue")]
    result = Save("batch-save")(
        image=images, filename=str(tmp_path / "image_{INDEX:02}.png")
    )
    assert [str(tmp_path / f"image_{index:02}.png") for index in range(2)] == result["output"]
    for source, filename, retained in zip(images, result["output"], result["saved_image_data"], strict=True):
        with Image.open(filename) as saved:
            assert saved.tobytes() == source.tobytes()
        assert retained is not source
        assert retained.tobytes() == source.tobytes()
