"""Real official single-file loading through the managed Hub alias topology.

Tiny local original-format fixtures prove loader selection and containment only;
they do not qualify pretrained weights, memory fit, or image quality.
"""
import hashlib
import os

import pytest

import test_modular_weight_formats as weight_format_tests
from modiff import modular_weight_formats as formats

PAIRS = weight_format_tests.PAIRS
runtime = weight_format_tests.runtime


def blob_alias(runtime, pair, scheme):
    snapshot = runtime.snapshots[pair[0]]
    alias = snapshot / 'variants' / scheme / 'model.safetensors'
    digest = hashlib.sha256(alias.read_bytes()).hexdigest()
    blob = snapshot.parent.parent / 'blobs' / digest
    blob.parent.mkdir(exist_ok=True)
    alias.rename(blob)
    alias.symlink_to(os.path.relpath(blob, alias.parent))
    assert alias.is_symlink() and alias.resolve() == blob
    assert not blob.suffix
    return alias, blob


@pytest.mark.parametrize('pair', PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize('scheme', ['fp8r', 'nvfp4'])
def test_official_single_file_uses_safetensors_snapshot_alias(runtime, monkeypatch, pair, scheme):
    alias, blob = blob_alias(runtime, pair, scheme)
    # A Safetensors-only selected artifact must never enter Torch's pickle
    # loader (nor its legacy text-error reader) because the blob has no suffix.
    import torch
    calls = []

    def refuse_pickle(*args, **kwargs):
        calls.append(args[0] if args else None)
        raise AssertionError('Safetensors cache alias entered the legacy reader.')

    monkeypatch.setattr(torch, 'load', refuse_pickle)
    node = runtime.call(pair, scheme)
    binding = node._weight_format_binding
    spec = node.loader.get_component_spec('transformer')
    assert binding.weight == blob
    assert spec.pretrained_model_name_or_path == str(alias)
    assert not calls
    binding.assert_current()
    # A warm owner must still revalidate the same real blob, with no copied
    # checkpoint or second component incarnation.
    load_id = node.loader.transformer._diffusers_load_id
    assert runtime.call(pair, scheme, node=node) is node
    assert node.loader.transformer._diffusers_load_id == load_id
    assert alias.is_symlink() and alias.resolve() == blob


def test_alias_retarget_is_refused_before_loading(runtime):
    pair, scheme = PAIRS[1], 'nvfp4'
    alias, blob = blob_alias(runtime, pair, scheme)
    binding = formats.resolve_weight_format(model_type=pair[0], metadata={'weight_formats': ['native', scheme]},
        repository=pair[1], revision=pair[2], weight_format=scheme, offload_mode='none', auto_offload=False)
    other = blob.with_name('f' * 64)
    other.write_bytes(blob.read_bytes())
    alias.unlink()
    alias.symlink_to(os.path.relpath(other, alias.parent))
    with pytest.raises(ValueError, match='changed during loading'):
        binding.assert_current()


def test_foreign_snapshot_alias_is_refused_without_legacy_reader(runtime, monkeypatch, tmp_path):
    pair, scheme = PAIRS[1], 'nvfp4'
    alias, blob = blob_alias(runtime, pair, scheme)
    outside = tmp_path / 'foreign.safetensors'
    outside.write_bytes(blob.read_bytes())
    alias.unlink()
    alias.symlink_to(outside)
    import torch
    monkeypatch.setattr(torch, 'load', lambda *_args, **_kwargs: pytest.fail('Legacy loader was reached.'))
    with pytest.raises(FileNotFoundError, match='managed snapshot'):
        runtime.call(pair, scheme)
    assert not runtime.manager.components
