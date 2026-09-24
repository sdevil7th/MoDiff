"""Art-direction controls, bounded mask algebra and protected graph dispatch."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modiff.custom_extensions import ExtensionStore


@pytest.fixture
def nodes(tmp_path, monkeypatch):
    from modules import MODULE_MAP

    store = ExtensionStore(tmp_path / "custom")
    item = store.stage(kind="local", source=str(Path(__file__).resolve().parents[1] /
                       "examples/custom_nodes/EditorialRegions"), name="EditorialTest")
    assert not item["enabled"]
    assert set(item["preview"]["nodes"]) == {"EditorialRegions", "ProtectedComposite"}
    registry = store.enable("EditorialTest", code_hash=item["codeHash"], consent=True)
    monkeypatch.setitem(MODULE_MAP, "custom.EditorialTest", registry)
    module = sys.modules["custom.EditorialTest.main"]
    yield module.EditorialRegions("regions"), module.ProtectedComposite("composite")
    store.unload("EditorialTest")


def polygon(points, operation="add"):
    return {"operation": operation, "points": points}


def test_mask_algebra_and_protection(nodes):
    director, composite = nodes
    source = Image.new("RGB", (100, 100), (60, 80, 100))
    before = source.tobytes()
    regions = json.dumps([
        polygon([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]),
        polygon([[0.4, 0.4], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]], "subtract"),
    ])
    result = director(image=source, regions=regions, feather=0, exposure=1)
    assert result["mask"].getpixel((50, 50)) == 0
    assert result["mask"].getpixel((20, 20)) == 255
    assert result["out_image"].getpixel((50, 50)) == source.getpixel((50, 50))
    assert result["out_image"].getpixel((20, 20)) != source.getpixel((20, 20))
    generated = Image.new("RGB", source.size, "red")
    finished = composite(original=source, edited=[generated], mask=result["mask"])["out_image"]
    zero = np.asarray(result["mask"]) == 0
    assert np.array_equal(np.asarray(finished)[zero], np.asarray(source)[zero])
    assert finished.getpixel((20, 20)) == (255, 0, 0)
    assert source.tobytes() == before


def test_intersection_and_feather(nodes):
    director, _ = nodes
    regions = json.dumps([
        polygon([[0, 0], [1, 0], [1, 1], [0, 1]]),
        polygon([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]], "intersect"),
    ])
    mask = np.asarray(director.execute(Image.new("RGB", (100, 100)), regions, feather=0.03)["mask"])
    assert mask[50, 50] == 255 and mask[0, 0] == 0
    assert ((mask > 0) & (mask < 255)).any()


def test_neutral_deterministic_and_changed_controls(nodes):
    director, _ = nodes
    source = Image.fromarray(np.random.default_rng(7).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    neutral = director(image=[source])
    assert neutral["out_image"].tobytes() == source.tobytes()
    first = director(image=source, temperature=0.8)["out_image"]
    assert first.tobytes() != source.tobytes()
    assert director(image=source, temperature=0.8)["out_image"].tobytes() == first.tobytes()
    assert director(image=source, saturation=0)["out_image"].getpixel((16, 16))[0] == director(
        image=source, saturation=0)["out_image"].getpixel((16, 16))[1]


@pytest.mark.parametrize("kwargs", [
    {"regions": "not json"}, {"regions": "[]"}, {"regions": "{}"},
    {"regions": "x" * 16385}, {"regions": '[{"operation":"add","points":[[0,0],[1,0],[2,1]]}]'},
    {"regions": '[{"operation":"add","points":[[0,0],[1,0],[true,1]]}]'},
    {"regions": '[{"operation":"unknown","points":[[0,0],[1,0],[1,1]]}]'},
    {"regions": json.dumps([polygon([[0, 0], [1, 0], [1, 1]])] * 17)},
    {"exposure": float("nan")}, {"temperature": 2}, {"saturation": -1}, {"feather": True},
])
def test_invalid_controls(nodes, kwargs):
    with pytest.raises(ValueError):
        nodes[0].execute(Image.new("RGB", (16, 16)), **kwargs)


def test_invalid_media_and_dimensions(nodes):
    director, composite = nodes
    for image in ([], [Image.new("RGB", (16, 16))] * 2, "tensor", Image.new("L", (2049, 2048))):
        with pytest.raises(ValueError):
            director.execute(image)
    source = Image.new("RGB", (16, 16))
    with pytest.raises(ValueError, match="dimensions"):
        composite.execute(source, Image.new("RGB", (32, 32)), Image.new("L", (16, 16)))
    with pytest.raises(ValueError, match="grayscale"):
        composite.execute(source, source, source)
    transparent = Image.new("RGBA", (16, 16), (255, 0, 0, 0))
    assert director.execute(transparent)["out_image"].getextrema() == ((255, 255),) * 3
