"""Semantic media must retain array storage instead of materializing scalar lists."""
import numpy as np
import pytest

from modiff.NodeBase import recursive_type_cast


class NoScalarMaterialization(np.ndarray):
    def tolist(self):
        raise AssertionError("semantic media must not be materialized into Python scalars")


@pytest.mark.parametrize("semantic", ["video", "image", "audio", "tensor", "latents"])
@pytest.mark.parametrize("nested", [False, True])
def test_semantic_array_preserves_storage_shape_dtype_and_views(semantic, nested):
    array = np.arange(2 * 3 * 16 * 16 * 3, dtype=np.float32).reshape(2, 3, 16, 16, 3)
    array = array[:, :, ::2].view(NoScalarMaterialization)
    array.flags.writeable = False
    value = {"clips": [array]} if nested else array
    result = recursive_type_cast(value, semantic, "media")
    actual = result["clips"][0] if nested else result
    assert actual is array
    assert not actual.flags.writeable
    assert actual.dtype == np.float32
    assert actual.shape == (2, 3, 8, 16, 3)


def test_primitive_numeric_array_cast_is_unchanged():
    values = np.array([1.7, -2.3], dtype=np.float32)
    actual = recursive_type_cast(values, "int", "count")
    np.testing.assert_array_equal(actual, np.array([1, -2], dtype=np.float32))
    assert actual is not values


def test_nested_primitive_cast_is_unchanged():
    assert recursive_type_cast({"a": ["1.0", "", "2"]}, "int", "count") == {"a": [1, 0, 2]}
    assert recursive_type_cast(["false", "yes"], "bool", "enabled") == [False, True]
