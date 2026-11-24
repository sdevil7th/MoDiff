"""Native deterministic media receipts, distinct from model qualification."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from modiff.builtin_media_evidence import begin_media_capture, snapshot_video_inputs, video_frame_identity


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


def test_video_loader_receipt_keeps_source_path_and_decodes_only_snapshotted_bytes(tmp_path, monkeypatch):
    import imageio
    from modiff.config import CONFIG
    from modules.Video.main import Load

    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    monkeypatch.setitem(CONFIG.paths, 'data', str(tmp_path))
    path = tmp_path / 'guide.mp4'
    path.write_bytes(b'original-encoded-source')
    frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in range(20)]

    class Reader:
        def get_meta_data(self):
            return {'size': (3, 2), 'fps': 12}

        def count_frames(self):
            return len(frames)

        def __iter__(self):
            return iter(frames)

        def close(self):
            pass

    def open_snapshot(filename, format):
        from pathlib import Path
        path.write_bytes(b'source-replaced-during-decode')
        assert filename != str(path) and Path(filename).read_bytes() == b'original-encoded-source'
        assert format == 'ffmpeg'
        return Reader()

    monkeypatch.setattr(imageio, 'get_reader', open_snapshot)
    node = owner('modules.Video', 'Load')
    node._interrupt = False
    output = Load.execute(node, file=['@data/guide.mp4'])
    evidence = node._execution_input_record['mediaEvidence']
    assert output['filename'] == str(path)
    assert evidence['files']['file'][0]['sourceValue'] == '@data/guide.mp4'
    assert evidence['files']['file'][0]['sha256'] == hashlib.sha256(b'original-encoded-source').hexdigest()
    video = evidence['outputs']['video'][0]
    assert video == {
        'kind': 'video', 'width': 3, 'height': 2, 'mode': 'RGB', 'frameCount': 20, 'byteSize': 360,
        'sha256': hashlib.sha256(b''.join(frame.tobytes() for frame in frames)).hexdigest(),
    }
    assert video_frame_identity(output['video'][::-1])['sha256'] != video['sha256']
    assert not list((tmp_path / '.media-evidence').iterdir())


def test_video_projection_rejects_mismatched_or_oversized_frame_lists(monkeypatch):
    frame = Image.new('RGB', (3, 2))
    with pytest.raises(ValueError, match='nonuniform-video-frames'):
        video_frame_identity([frame, Image.new('L', (3, 2))])
    monkeypatch.setattr('modiff.builtin_media_evidence.MAX_FILE_BYTES', 20)
    with pytest.raises(ValueError, match='video-frame-capture-limit'):
        video_frame_identity([frame, frame])


def test_unavailable_snapshot_directory_preserves_normal_video_input(tmp_path, monkeypatch):
    from modiff.config import CONFIG

    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    monkeypatch.setitem(CONFIG.paths, 'data', str(tmp_path))
    (tmp_path / '.media-evidence').write_bytes(b'not-a-directory')
    with snapshot_video_inputs('@data/guide.mp4') as (actual, files):
        assert actual == '@data/guide.mp4'
        assert files is None


@pytest.mark.parametrize('mode', ['character_animate', 'character_replace'])
def test_wan_motion_capture_describes_the_actual_library_inputs(monkeypatch, mode):
    from modules.DiffusersVideo.main import Generate

    monkeypatch.setenv('MODIFF_CAPTURE_BUILTIN_MEDIA', '1')
    reference = Image.new('RGB', (4, 3))
    pose = [Image.new('RGB', (4, 3), (step, 0, 0)) for step in range(20)]
    face = pose[::-1]
    background = pose if mode == 'character_replace' else None
    mask = [Image.new('L', (4, 3), 255)] * len(pose) if background else None

    class Pipeline:
        _modiff_video_pipeline_class = 'WanAnimatePipeline'
        _execution_device = 'cpu'

        def __call__(self, **arguments):
            self.arguments = arguments
            return SimpleNamespace(frames=[pose])

    pipeline = Pipeline()
    node = Generate()
    output = node.execute(pipeline=pipeline, mode=mode, reference_images=[reference], pose_video=pose,
                          face_video=face, background_video=background, mask=mask)
    assert '_native_media_capture' not in output
    recorded = node._execution_input_record['mediaEvidence']
    assert recorded['controls'] == {'mode': mode}
    for socket in ('pose_video', 'face_video', 'background_video', 'mask'):
        value = pipeline.arguments['mask_video' if socket == 'mask' else socket]
        assert recorded['inputs'][socket] == ([video_frame_identity(value)] if value else [])
    assert recorded['inputs']['pose_video'] != recorded['inputs']['face_video']


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
