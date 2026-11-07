"""File transparency must survive the ordinary image-node dispatch boundary."""

import pytest
from PIL import Image

from modules.Image.main import Load
from modules.ImageOperations.main import AdjustImage, FilterImage, MaskComposite, StitchImages


@pytest.fixture(params=["P", "RGB", "L", "RGBA", "LA"])
def transparent_image(request):
    mode = request.param
    image = Image.new(mode, (3, 2))
    if mode == "P":
        image.putpalette([255, 0, 0, 0, 255, 0, 0, 0, 255] + [0] * 759)
        image.putdata([0, 1, 2, 0, 1, 2])
        image.info["transparency"] = bytes([0, 128, 255])
    elif mode == "RGB":
        image.putdata([(10, 20, 30), (40, 50, 60), (70, 80, 90)] * 2)
        image.info["transparency"] = (10, 20, 30)
    elif mode == "L":
        image.putdata([10, 40, 70] * 2)
        image.info["transparency"] = 10
    elif mode == "RGBA":
        image.putdata([(10, 20, 30, 0), (40, 50, 60, 128), (70, 80, 90, 255)] * 2)
    else:
        image.putdata([(10, 0), (40, 128), (70, 255)] * 2)
    original_pixels, original_info = image.tobytes(), image.info.copy()
    yield image
    assert image.tobytes() == original_pixels
    assert image.info == original_info


def test_load_png_mask_uses_palette_color_key_and_explicit_alpha(transparent_image, tmp_path):
    source = tmp_path / "transparent.png"
    transparent_image.save(source)
    expected_mask = transparent_image.convert("RGBA").getchannel("A")
    result = Load("transparent-load")(file=str(source), alpha_channel="ignore")
    assert result["mask"].tobytes() == expected_mask.tobytes()
    assert result["image"].convert("RGBA").tobytes() == transparent_image.convert("RGBA").tobytes()


def test_explicit_remove_alpha_applies_to_every_transparent_png_mode(transparent_image, tmp_path):
    source = tmp_path / "transparent.png"
    transparent_image.save(source)
    result = Load("opaque-load")(file=str(source), alpha_channel="remove alpha")
    assert result["image"].mode == "RGB"
    assert "transparency" not in result["image"].info
    assert result["mask"].tobytes() == transparent_image.convert("RGBA").getchannel("A").tobytes()


@pytest.mark.parametrize("node_class", [AdjustImage, FilterImage])
def test_adjustment_and_filter_preserve_transparency(transparent_image, node_class):
    expected = transparent_image.convert("RGBA").getchannel("A")
    result = node_class("transparent-operation")(image=transparent_image)["output"]
    assert result.mode == "RGBA"
    assert result.getchannel("A").tobytes() == expected.tobytes()


def test_stitch_composites_transparent_pngs_against_the_selected_background(transparent_image):
    result = StitchImages("transparent-stitch")(
        image=[transparent_image, transparent_image], background="white"
    )["output"]
    expected = Image.new("RGBA", result.size, "white")
    expected.alpha_composite(transparent_image.convert("RGBA"), (0, 0))
    expected.alpha_composite(transparent_image.convert("RGBA"), (3, 0))
    assert result.convert("RGBA").tobytes() == expected.tobytes()


def test_mask_composite_retains_transparent_foreground(transparent_image):
    result = MaskComposite("transparent-composite")(
        background=Image.new("RGB", transparent_image.size, "white"),
        foreground=transparent_image,
        mask=Image.new("L", transparent_image.size, 255),
    )["output"]
    assert result.mode == "RGBA"
    assert result.tobytes() == transparent_image.convert("RGBA").tobytes()
