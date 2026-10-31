"""Selected image interpolation must reach Pillow through ordinary node dispatch."""

import numpy as np
import pytest
from PIL import Image, ImageOps

from modules.Image.main import Resize
from utils import image as image_utils


METHODS = ["stretch", "cover", "contain", "fit", "pad"]
FILTERS = [
    ("nearest", Image.Resampling.NEAREST),
    ("box", Image.Resampling.BOX),
    ("bilinear", Image.Resampling.BILINEAR),
    ("hamming", Image.Resampling.HAMMING),
    ("bicubic", Image.Resampling.BICUBIC),
    ("lanczos", Image.Resampling.LANCZOS),
]


def reference(source, method, resample):
    if method == "stretch":
        return source.resize((13, 11), resample=resample)
    options = {"method": resample}
    if method == "pad":
        options["color"] = (0, 0, 0)
    return getattr(ImageOps, method)(source, (13, 11), **options)


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("name,resample", FILTERS)
def test_resize_node_consumes_the_requested_interpolation(mode, method, name, resample):
    channels = 3 if mode == "RGB" else 4
    pixels = np.random.default_rng(42).integers(0, 256, (6, 8, channels), dtype=np.uint8)
    source = Image.fromarray(pixels)
    before = source.tobytes()
    expected = reference(source, method, resample)
    actual = Resize("resampling-" + method)(
        image=source, method=method, width=13, height=11,
        multiple_of=1, interpolation=name,
    )["output"]
    assert (actual.mode, actual.size, actual.tobytes()) == (expected.mode, expected.size, expected.tobytes())
    assert source.tobytes() == before


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("selected", ["NEAREST", "NeArEsT", Image.Resampling.NEAREST])
def test_geometry_helpers_preserve_case_insensitive_enum_and_ordered_list_inputs(method, selected):
    source = Image.fromarray(np.random.default_rng(42).integers(0, 256, (6, 8, 3), dtype=np.uint8))
    other = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    helper = image_utils.resize if method == "stretch" else getattr(image_utils, method)
    result = helper([source, other], 13, 11, selected)
    assert [item.tobytes() for item in result] == [
        reference(item, method, Image.Resampling.NEAREST).tobytes() for item in (source, other)
    ]


@pytest.mark.parametrize("method", METHODS)
def test_geometry_defaults_and_unknown_names_keep_bicubic_compatibility(method):
    source = Image.fromarray(np.random.default_rng(42).integers(0, 256, (6, 8, 3), dtype=np.uint8))
    helper = image_utils.resize if method == "stretch" else getattr(image_utils, method)
    expected = reference(source, method, Image.Resampling.BICUBIC)
    assert helper(source, 13, 11).tobytes() == expected.tobytes()
    assert helper(source, 13, 11, "unknown").tobytes() == expected.tobytes()
