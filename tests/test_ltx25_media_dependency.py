"""Exercise the actual official conditioning codec without downloading weights."""
from pathlib import Path
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


def test_conditioning_codec_is_installed_by_ordinary_setup_and_checked_before_runs():
    from modiff.preflight import PACKAGE_CHECKS
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    assert 'av==18.1.0' in project['project']['dependencies']
    assert ('av', 'av') in PACKAGE_CHECKS['required']
    assert ('av', 'av') not in PACKAGE_CHECKS['optional']
    lock = tomllib.loads((root / 'uv.lock').read_text(encoding='utf-8'))
    package = next(package for package in lock['package'] if package['name'] == 'av')
    assert package['version'] == '18.1.0'
    assert package['wheels'] and all(wheel['hash'].startswith('sha256:') for wheel in package['wheels'])


def test_official_ltx_image_conditioning_recompression_uses_the_installed_codec():
    from modiff.preflight import package_status
    from diffusers.pipelines.ltx2.utils import apply_image_conditioning_crf

    assert package_status('av', 'av')['available']
    # Chroma and edge detail exercise a real RGB -> H.264 -> RGB round trip.
    y, x = np.indices((64, 96))
    image = np.stack((x * 2, y * 3, (x + y) % 256), axis=-1).astype(np.uint8)
    result = apply_image_conditioning_crf(image, crf=33)
    assert result.shape == image.shape and result.dtype == np.uint8
    assert np.isfinite(result).all() and result.std() > 10
    assert not np.array_equal(result, image), 'The actual upstream compression must execute.'
    assert np.abs(result.astype(float) - image).mean() < 20


def test_preflight_rejects_pyav_with_a_missing_h264_encoder(monkeypatch):
    from modiff import preflight

    codec = Mock(side_effect=LookupError('Unknown codec: libx264'))
    monkeypatch.setattr(preflight.importlib, 'import_module', lambda name: SimpleNamespace(codec=SimpleNamespace(Codec=codec)))
    status = preflight.package_status('av', 'av')
    assert not status['available']
    assert status['error'] == 'Unknown codec: libx264'
    codec.assert_called_once_with('libx264', 'w')
