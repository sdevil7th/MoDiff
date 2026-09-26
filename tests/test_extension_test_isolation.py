"""Unit test discovery must never import or rewrite operator extensions."""
from pathlib import Path

from modiff.custom_extensions import ExtensionStore


def test_default_extension_store_is_outside_the_operator_checkout():
    operator_root = Path(__file__).resolve().parents[1] / 'custom'
    assert ExtensionStore().root != operator_root


def test_explicit_extension_fixture_root_is_preserved(tmp_path):
    assert ExtensionStore(tmp_path / 'explicit').root == tmp_path / 'explicit'
