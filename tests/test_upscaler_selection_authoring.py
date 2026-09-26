"""Native model edits must persist exact installed single-file identities."""
import hashlib
from importlib import import_module
from unittest.mock import Mock, patch

import pytest

from modiff import controlled_artifacts
from modiff.field_metadata import is_metadata_field_action, metadata_field_callback
from modiff.upscaler_contracts import real_esrgan_x2_model_selection
from utils.huggingface import CONFIG


@pytest.fixture
def cached_upscaler(tmp_path, monkeypatch):
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(tmp_path))
    repository = tmp_path / 'models--example--upscaler'
    weight = repository / 'snapshots' / ('a' * 40) / 'weights' / 'upscale.pth'
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b'installed-upscaler')
    (repository / 'refs').mkdir()
    (repository / 'refs' / 'main').write_text('a' * 40)
    return repository, weight


def test_cached_file_selection_is_pinned_before_admission(cached_upscaler):
    _, weight = cached_upscaler
    selection = {'source': 'hub', 'value': 'example/upscaler/weights/upscale.pth'}
    with patch('huggingface_hub.hf_hub_download', side_effect=AssertionError('no download')):
        pinned = controlled_artifacts.pin_upscaler_model_selection(selection)
        resolved = controlled_artifacts.resolve_upscaler_artifact(pinned)
    assert pinned == {**selection, 'revision': 'a' * 40,
                      'sha256': hashlib.sha256(weight.read_bytes()).hexdigest(),
                      'byteSize': weight.stat().st_size}
    assert resolved.path == weight
    assert 'revision' not in selection
    # Once authored, changing the cache's main ref must not change this workflow.
    (cached_upscaler[0] / 'refs' / 'main').write_text('b' * 40)
    assert controlled_artifacts.resolve_upscaler_artifact(pinned).path == weight


def test_complete_reviewed_pin_can_be_authored_without_installed_weights():
    selection = real_esrgan_x2_model_selection()
    with patch('utils.huggingface.cached_file_path', side_effect=AssertionError('no cache inspection')):
        assert controlled_artifacts.pin_upscaler_model_selection(selection) == selection


def test_missing_cache_ref_never_selects_newest_snapshot(cached_upscaler):
    repository, _ = cached_upscaler
    (repository / 'refs' / 'main').unlink()
    with pytest.raises(FileNotFoundError, match='installed'):
        controlled_artifacts.pin_upscaler_model_selection(
            {'source': 'hub', 'value': 'example/upscaler/weights/upscale.pth'})


def test_explicit_revision_is_honored_and_local_selection_gets_integrity(cached_upscaler):
    _, weight = cached_upscaler
    selected = {'source': 'hub', 'value': 'example/upscaler/weights/upscale.pth', 'revision': 'a' * 40}
    assert controlled_artifacts.pin_upscaler_model_selection(selected)['revision'] == 'a' * 40
    local = controlled_artifacts.pin_upscaler_model_selection({'source': 'local', 'value': str(weight)})
    assert 'revision' not in local
    assert local['byteSize'] == weight.stat().st_size
    assert controlled_artifacts.resolve_upscaler_artifact(local).path == weight


@pytest.mark.parametrize('metadata', [
    {'revision': 'main'}, {'sha256': '0' * 64}, {'byteSize': 1}, {'byteSize': True},
])
def test_invalid_or_mismatched_declared_identity_is_not_repaired_silently(cached_upscaler, metadata):
    with pytest.raises(ValueError):
        controlled_artifacts.pin_upscaler_model_selection(
            {'source': 'hub', 'value': 'example/upscaler/weights/upscale.pth', **metadata})


def test_cache_alias_outside_exact_repository_is_rejected(cached_upscaler, tmp_path):
    _, weight = cached_upscaler
    outside = tmp_path / 'other.pth'
    outside.write_bytes(b'outside')
    weight.unlink()
    weight.symlink_to(outside)
    with pytest.raises(ValueError, match='outside'):
        controlled_artifacts.pin_upscaler_model_selection(
            {'source': 'hub', 'value': 'example/upscaler/weights/upscale.pth'})


def test_newly_resolved_cached_ref_still_checks_predeclared_integrity(cached_upscaler):
    _, weight = cached_upscaler
    with pytest.raises(ValueError, match='SHA-256'):
        controlled_artifacts.pin_upscaler_model_selection({
            'source': 'hub', 'value': 'example/upscaler/weights/upscale.pth',
            'sha256': '0' * 64, 'byteSize': weight.stat().st_size,
        })


@pytest.mark.parametrize('module,action', [('modules.Spandrel', 'Upscaler'), ('modules.Video', 'UpscaleVideo')])
def test_authoring_callback_has_no_model_owner_and_publishes_pin(cached_upscaler, module, action):
    from modules import MODULE_MAP
    method = MODULE_MAP[module][action]['params']['model_id'].get('onChange')
    assert method == 'update_model_selection'
    assert is_metadata_field_action({'module': module, 'action': action, 'fn': method})
    assert not is_metadata_field_action({'module': module, 'action': action, 'fn': method, 'queue': True})
    node_class = getattr(import_module(module + '.main'), action)
    with patch.object(node_class, '__init__', side_effect=AssertionError('no executable owner')):
        callback = metadata_field_callback(action, method, module=module, node_id='editing', sid='browser')
    callback.__self__.set_field_value = Mock()
    callback({'model_id': {'source': 'hub', 'value': 'example/upscaler/weights/upscale.pth'}}, {'key': 'model_id'})
    value = callback.__self__.set_field_value.call_args.args[0]['model_id']
    assert value['revision'] == 'a' * 40
    assert not hasattr(callback.__self__, 'execute')
    callback.__self__.set_field_value.reset_mock()
    callback({'model_id': {'source': 'hub', 'value': ''}}, {'key': 'model_id'})
    callback.__self__.set_field_value.assert_not_called()
