"""Finite ordinary media unions retain strict Block boundary semantics."""
import pytest

from modiff.block_definition_v2 import _block_value_types_are_compatible_v2


@pytest.mark.parametrize(('public', 'runtime', 'expected'), [
    ('video|str|image', ['video', 'str', 'image'], True),
    ('video|str|image', 'video', True),
    ('video_asset|video|str', 'video_asset', True),
    ('video|str', 'string', True),
    ('video|str', 'audio', False),
    ('list[video|image]', 'image', False),
    ('video|opaque', 'audio', True),
])
def test_finite_scalar_union_matches_only_its_declared_alternatives(public, runtime, expected):
    assert _block_value_types_are_compatible_v2(public, runtime) is expected
