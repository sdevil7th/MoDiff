"""The portable custom example must be safe, deterministic and graph-executable."""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modiff.custom_extensions import ExtensionStore


@pytest.fixture
def director(tmp_path, monkeypatch):
    from modules import MODULE_MAP

    store = ExtensionStore(tmp_path / "custom")
    source = Path(__file__).resolve().parents[1] / "examples/custom_nodes/LightPaletteDirector"
    item = store.stage(kind="local", source=str(source), name="PaletteTest")
    assert not item["enabled"]
    assert item["preview"]["nodes"]["LightPaletteDirector"]["params"]["mask"]["type"] == "image"
    registry = store.enable("PaletteTest", code_hash=item["codeHash"], consent=True)
    monkeypatch.setitem(MODULE_MAP, "custom.PaletteTest", registry)
    yield sys.modules["custom.PaletteTest.main"].LightPaletteDirector("palette-example")
    store.unload("PaletteTest")


def gradient():
    return Image.fromarray(np.tile(np.arange(64, dtype=np.uint8) * 4, (48, 1))).convert("RGB")


def test_deterministic_image_mask_and_diagnostics(director):
    image = gradient()
    before = image.tobytes()
    result = director.execute(image)
    assert result["out_image"].mode == "RGB"
    assert result["mask"].mode == "L"
    assert result["out_image"].size == result["mask"].size == image.size
    assert 0 < result["diagnostics"]["mask_coverage"] < 1
    assert result["mask"].getpixel((0, 0)) == 0
    assert result["mask"].getpixel((32, 24)) == 255
    assert result["out_image"].getpixel((0, 0)) == image.getpixel((0, 0))
    assert result["out_image"].tobytes() != before
    assert image.tobytes() == before
    assert director.execute(image)["out_image"].tobytes() == result["out_image"].tobytes()


def test_zero_strength_and_transparency(director):
    image = gradient()
    assert director.execute(image, strength=0)["out_image"].tobytes() == image.tobytes()
    transparent = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
    result = director.execute(transparent, strength=0)
    assert result["out_image"].getextrema() == ((255, 255),) * 3


@pytest.mark.parametrize("key,value", [
    ("strength", float("nan")), ("feather", 0), ("center_x", float("inf")),
    ("radius_x", -1), ("light_angle", 181), ("strength", True),
    ("shadow", "red"), ("highlight", "#XXXXXX"),
])
def test_rejects_invalid_controls(director, key, value):
    with pytest.raises(ValueError):
        director.execute(gradient(), **{key: value})


def test_bounded_batches_and_decoded_media_only(director):
    for value in ([], [gradient()] * 5, "latents", np.zeros((3, 16, 16))):
        with pytest.raises(ValueError):
            director.execute(value)
    with pytest.raises(ValueError, match="pixels"):
        director.execute(Image.new("L", (2049, 2048)))
    batch = director.execute([gradient(), gradient()])
    assert len(batch["out_image"]) == len(batch["mask"]) == len(batch["diagnostics"]) == 2


def test_normal_node_dispatch_and_changed_control_invalidates_cache(director):
    image = gradient()
    first = director(image=image, strength=0.2)
    second = director(image=image, strength=0.9)
    assert first["out_image"].tobytes() != second["out_image"].tobytes()
    assert second["diagnostics"]["settings"]["strength"] == 0.9
