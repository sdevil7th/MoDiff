"""Opt-in, bounded call-boundary evidence for reviewed media handling.

No model objects or arbitrary suppliers are inspected. Snapshot limits constrain
evidence capture only; they never change an ordinary workflow's runtime limits.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile


IMAGE_CONTROLS = '''pipeline_class operation brightness contrast saturation sharpness gamma temperature tint
    filter_operation amount threshold seed x y width height resize_width resize_height resize_fit_mode
    resize_resampling stitch_columns stitch_spacing stitch_background stitch_match_size rows columns
    channel composite_mask_channel composite_invert_mask'''.split()
AUDIO_CONTROLS = '''pipeline_class operation start_seconds duration_seconds target_sample_rate normalize_peak
    boundary_fade_seconds reference_window_seconds target_peak_dbfs max_adjustment_db'''.split()
VIDEO_CONTROLS = '''pipeline_class operation selection_mode indices timecodes every_n fps interpolation_fps
    transition_seconds start_seconds end_seconds columns sync gap background'''.split()
CONTROL_FIELDS = {
    ('Image', 'Load'): ['alpha_channel'],
    ('Image', 'Preview'): ['device', 'export'],
    ('ImageOperations', 'ProcessImage'): IMAGE_CONTROLS,
    ('Audio', 'Load'): [],
    ('Audio', 'ProcessAudio'): AUDIO_CONTROLS,
    ('Audio', 'Export'): ['filename', 'sample_rate'],
    ('HuggingFaceSpeech', 'TranscribeAudio'): [],
    ('HuggingFaceSpeech', 'TranscribeCTCAudio'): [],
    ('Video', 'ProcessVideo'): VIDEO_CONTROLS,
    ('Video', 'Export'): ['filename', 'quality', 'fps'],
}
MAX_CAPTURE_ITEMS = 16
MAX_CAPTURE_BYTES = 32 * 1024
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_PIXEL_BYTES = 64 * 1024 * 1024


def enabled():
    return os.environ.get('MODIFF_CAPTURE_BUILTIN_MEDIA') == '1'


def file_identity(path):
    digest = hashlib.sha256()
    size = 0
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            if size > MAX_FILE_BYTES:
                raise ValueError('file-capture-limit')
            digest.update(chunk)
    return {'kind': 'file', 'sha256': digest.hexdigest(), 'byteSize': size}


def media_identity(value):
    from PIL import Image
    if isinstance(value, Image.Image):
        if value.width * value.height * len(value.getbands()) > MAX_PIXEL_BYTES:
            raise ValueError('pixel-capture-limit')
        data = value.tobytes()
        return {'kind': 'image', 'sha256': hashlib.sha256(data).hexdigest(), 'byteSize': len(data),
                'width': value.width, 'height': value.height, 'mode': value.mode}
    if type(value) is dict and 'samples' in value:
        import numpy as np
        samples = value['samples']
        if not isinstance(samples, np.ndarray) or value.get('sample_layout') != 'frames_first':
            raise ValueError('unsupported-audio-layout')
        if samples.nbytes > MAX_PIXEL_BYTES or samples.ndim != 2 or not np.isfinite(samples).all():
            raise ValueError('unsupported-audio-samples')
        if type(value.get('sample_rate')) is not int or value['sample_rate'] <= 0:
            raise ValueError('unsupported-audio-rate')
        if value.get('channels') != samples.shape[1]:
            raise ValueError('unsupported-audio-channels')
        data = np.ascontiguousarray(samples, dtype='<f4').tobytes()
        return {'kind': 'audio', 'sha256': hashlib.sha256(data).hexdigest(), 'byteSize': len(data),
                'shape': list(samples.shape), 'sampleRate': value['sample_rate'], 'dtype': 'float32-le'}
    if isinstance(value, (str, Path)):
        return file_identity(value)
    raise ValueError('unsupported-media-value')


def identities(value):
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    if len(values) > MAX_CAPTURE_ITEMS:
        raise ValueError('collection-capture-limit')
    return [media_identity(item) for item in values]


class _Capture:
    def __init__(self, node, controls, inputs, files):
        self.node = node
        self.record = None
        if not enabled():
            return
        module_name = getattr(node, 'module_name', None)
        class_name = getattr(node, 'class_name', None)
        if not isinstance(module_name, str) or not isinstance(class_name, str):
            return
        scope = (module_name.removeprefix('modules.'), class_name)
        if scope not in CONTROL_FIELDS:
            return
        try:
            if scope == ('Video', 'ProcessVideo'):
                value = inputs.get('videos')
                count = len(value) if isinstance(value, (list, tuple)) else 1
                if not files or not isinstance(files.get('videos'), list) or len(files['videos']) != count:
                    raise ValueError('video-snapshot-unavailable')
            values = {}
            for name in CONTROL_FIELDS[scope]:
                if name not in controls:
                    continue
                value = controls[name]
                if value is not None and type(value) not in (str, int, float, bool):
                    raise ValueError('unsupported-control-value')
                if isinstance(value, str) and len(value) > 16384:
                    raise ValueError('control-capture-limit')
                values[name] = value
            self.record = {'schemaVersion': 1, 'source': 'native-builtin-media-v1', 'status': 'complete',
                           'controls': values, 'inputs': {key: identities(value) for key, value in inputs.items()},
                           'outputs': {}, 'files': files or {}}
            json.dumps(self.record, allow_nan=False)
        except (ValueError, TypeError, OSError, MemoryError):
            self.record = {'schemaVersion': 1, 'status': 'unavailable', 'reason': 'unsupported-or-bounded-input'}

    def finish(self, outputs):
        if self.record is None:
            return
        if self.record['status'] == 'complete':
            try:
                self.record['outputs'] = {key: identities(value) for key, value in outputs.items()}
                size = len(json.dumps(self.record, allow_nan=False, ensure_ascii=True).encode('ascii'))
                if size > MAX_CAPTURE_BYTES:
                    self.record = {'schemaVersion': 1, 'status': 'truncated', 'reason': 'capture-size-limit'}
            except (ValueError, TypeError, OSError, MemoryError):
                self.record = {'schemaVersion': 1, 'status': 'unavailable', 'reason': 'unsupported-or-bounded-output'}
        record = getattr(self.node, '_execution_input_record', None)
        if isinstance(record, dict):
            record['mediaEvidence'] = self.record


def begin_media_capture(node, controls, *, inputs, files=None):
    return _Capture(node, controls, inputs, files)


@contextmanager
def snapshot_video_inputs(value):
    """Copy once while hashing, then consume only owned snapshot bytes.

    Unsupported/oversized capture leaves normal behavior intact and yields no
    qualifying evidence. Retained structured assets are not silently rewritten.
    """
    if not enabled():
        yield value, None
        return
    values = value if isinstance(value, (list, tuple)) else [value]
    if not 1 <= len(values) <= MAX_CAPTURE_ITEMS or any(type(item) is not str or not item for item in values):
        yield value, None
        return
    from modiff.config import CONFIG
    from modiff.path_identifiers import resolve_runtime_input_path
    root = Path(CONFIG.paths['data']) / '.media-evidence'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix='inputs-') as temporary:
        actual, files = [], []
        try:
            for index, original in enumerate(values):
                source = resolve_runtime_input_path(original)
                target = Path(temporary) / f'{index:02d}{source.suffix.lower()}'
                digest, size = hashlib.sha256(), 0
                with source.open('rb') as stream, target.open('xb') as output:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        size += len(chunk)
                        if size > MAX_FILE_BYTES:
                            raise ValueError('file-capture-limit')
                        output.write(chunk)
                        digest.update(chunk)
                actual.append(str(target))
                files.append({'sourceValue': original, 'sha256': digest.hexdigest(), 'byteSize': size})
        except (ValueError, OSError):
            yield value, None
            return
        yield actual, files
