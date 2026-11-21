"""Native deterministic media receipts, distinct from model qualification."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from modiff.builtin_media_evidence import begin_media_capture, snapshot_video_inputs


def owner(module='modules.ImageOperations', action='ProcessImage'):
    return SimpleNamespace(module_name=module, class_name=action, _execution_input_record={})


def test_capture_is_disabled_by_default_and_does_not_touch_media(monkeypatch):
    monkeypatch.delenv('MODIFF_CAPTURE_BUILTIN_MEDIA', raising=False)
    node = owner()
    capture = begin_media_capture(node, {'operation': 'image_crop'}, inputs={'image': object()})
    capture.finish({'output': object()})
    assert node._execution_input_record == {}


def test_disabled_capture_does_not_inspect_the_owner(monkeypatch):
    monkeypatch.delenv('MODIFF_CAPTURE_BUILTIN_MEDIA', raising=False)

    class Uninspected:
        def __getattribute__(self, name):
            raise AssertionError('disabled capture must not inspect its owner')

    begin_media_capture(Uninspected(), {}, inputs={'image': object()}).finish({'output': object()})


def test_enabled_capture_ignores_opaque_owners(monkeypatch):
    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    begin_media_capture(object(), {}, inputs={'image': object()}).finish({'output': object()})


def test_call_boundary_keeps_order_pixels_masks_and_controls(monkeypatch):
    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    node = owner()
    images = [Image.new('RGB', (3, 2), color) for color in ['red', 'blue']]
    mask = Image.new('L', (3, 2), 128)
    capture = begin_media_capture(node, {'operation': 'mask_composite', 'width': 0},
                                  inputs={'image': images, 'mask': mask})
    # Mutating a later object must not rewrite the already observed inputs.
    images[0].putpixel((0, 0), (0, 255, 0))
    capture.finish({'output': images[1]})
    evidence = node._execution_input_record['mediaEvidence']
    assert evidence['status'] == 'complete'
    assert evidence['inputs']['image'][0]['sha256'] == hashlib.sha256(bytes([255, 0, 0]) * 6).hexdigest()
    assert evidence['inputs']['image'][1] == evidence['outputs']['output'][0]
    assert evidence['inputs']['mask'][0]['mode'] == 'L'
    assert evidence['controls'] == {'operation': 'mask_composite', 'width': 0}


def test_audio_capture_binds_rate_layout_and_finite_samples(monkeypatch):
    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    node = owner('modules.Audio', 'ProcessAudio')
    audio = {'samples': np.array([[0.0], [0.5]], dtype=np.float32), 'sample_rate': 48000,
             'sample_layout': 'frames_first', 'channels': 1}
    capture = begin_media_capture(node, {'operation': 'audio_trim'}, inputs={'source': audio})
    capture.finish({'output': audio})
    evidence = node._execution_input_record['mediaEvidence']
    assert evidence['status'] == 'complete'
    assert evidence['inputs']['source'][0]['shape'] == [2, 1]
    assert evidence['inputs']['source'][0]['sampleRate'] == 48000
    assert evidence['inputs']['source'] == evidence['outputs']['output']
    audio['samples'][0, 0] = np.nan
    begin_media_capture(node, {}, inputs={'source': audio}).finish({'output': audio})
    assert node._execution_input_record['mediaEvidence']['status'] == 'unavailable'


def test_video_consumes_the_same_snapshot_that_was_hashed(tmp_path, monkeypatch):
    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    from modiff.config import CONFIG
    monkeypatch.setitem(CONFIG.paths, 'data', str(tmp_path))
    source = tmp_path / 'input.mp4'
    source.write_bytes(b'original-video')
    with snapshot_video_inputs([str(source)]) as (actual, files):
        source.write_bytes(b'replaced-video')
        from pathlib import Path
        assert Path(actual[0]).read_bytes() == b'original-video'
        assert files[0]['sha256'] == hashlib.sha256(b'original-video').hexdigest()
        node = owner('modules.Video', 'ProcessVideo')
        begin_media_capture(node, {}, inputs={'videos': actual}, files={'videos': files}).finish({'video': actual[0]})
        evidence = node._execution_input_record['mediaEvidence']
        assert evidence['status'] == 'complete'
        assert evidence['inputs']['videos'][0]['sha256'] == files[0]['sha256']
        retained = Path(actual[0])
    assert not retained.exists()
    assert source.read_bytes() == b'replaced-video'


def test_capture_never_traverses_unknown_objects_or_custom_nodes(monkeypatch):
    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    class Hostile:
        def __str__(self):
            raise AssertionError('must not stringify')
    node = owner('custom.Unreviewed', 'ProcessImage')
    begin_media_capture(node, {'secret': Hostile()}, inputs={'image': Hostile()}).finish({'output': Hostile()})
    assert node._execution_input_record == {}


def test_optional_media_budget_preserves_legacy_records(monkeypatch):
    from modiff.execution_input_provenance import build_resolved_execution_inputs, capture_generation_inputs
    graph = {'nodes': {'op': {'module': 'modules.ImageOperations', 'action': 'ProcessImage', 'params': {}}}}
    record = capture_generation_inputs('op', graph['nodes']['op'], {'width': 12})
    record['mediaEvidence'] = {'schemaVersion': 1, 'status': 'complete', 'huge': 'a' * 300_000}
    result = build_resolved_execution_inputs(graph, {'op': record}, task_id='test', attempt_index=0, node_id='op')
    assert result['truncated'] is False
    assert result['summary'] == {'width': 12}
    assert result['mediaEvidenceStatus'] == {'schemaVersion': 1, 'truncated': True}
    assert 'mediaEvidence' not in result['nodes'][0]


def test_missing_video_snapshot_cannot_be_reported_complete(tmp_path, monkeypatch):
    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    path = tmp_path / 'source.mp4'
    path.write_bytes(b'small encoded-file fixture')
    node = owner('modules.Video', 'ProcessVideo')
    begin_media_capture(node, {}, inputs={'videos': [str(path)]}, files={'videos': []}).finish({'video': str(path)})
    assert node._execution_input_record['mediaEvidence']['status'] == 'unavailable'


@pytest.mark.parametrize('browser_selection', [False, True])
def test_audio_loader_hashes_the_same_snapshot_it_decodes(tmp_path, monkeypatch, browser_selection):
    from modules.Audio.main import Load
    from scipy.io import wavfile
    import io
    path = tmp_path / 'source.wav'
    wavfile.write(path, 48000, np.array([0, 1000, -1000], dtype=np.int16))
    original = path.read_bytes()
    real_read = wavfile.read
    def replace_source_before_decode(value, **kwargs):
        assert isinstance(value, io.BytesIO)
        path.write_bytes(b'replacement must not be decoded')
        return real_read(value, **kwargs)
    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    monkeypatch.setattr(wavfile, 'read', replace_source_before_decode)
    node = Load('audio-input')
    result = node(file=[str(path)] if browser_selection else str(path))
    evidence = node._execution_input_record['mediaEvidence']
    assert evidence['files']['file'] == [{
        'sourceValue': str(path),
        'sha256': hashlib.sha256(original).hexdigest(),
        'byteSize': len(original),
        'converted': False,
    }]
    assert evidence['outputs']['audio'][0]['shape'] == [3, 1]
    assert np.allclose(result['audio']['samples'][:, 0], [0, 1000 / 32768, -1000 / 32768])
