"""Ordinary media sources declare their existing mandatory file contract."""
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('medium', ['Image', 'Video', 'Audio'])
def test_ast_registry_requires_the_file_for_ordinary_media_loaders(medium):
    import modules

    source = Path(__file__).parents[1] / 'modules' / medium / 'main.py'
    package = SimpleNamespace(__name__='modules.' + medium)
    node = modules.parse_module_file(str(source), package)['Load']
    assert node['params']['file']['required'] is True
    assert node['params']['file']['display'] == 'filebrowser'
    assert node['skipParamsCheck'] is False
