"""Video/audio export preserves borrowed inputs and commits only completed media."""
from pathlib import Path
import subprocess

import numpy as np
import imageio.v2 as imageio
from PIL import Image
import pytest
from scipy.io import wavfile

from modules.Video import main as video


def clip():
    return [Image.new('RGB', (32, 32), 'navy') for _ in range(8)]


def files(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob('*') if path.is_file()}


def test_export_preserves_borrowed_audio_even_when_its_name_looks_temporary(tmp_path, monkeypatch):
    source = tmp_path / 'source.audio.wav'
    wavfile.write(source, 48000, np.zeros((48000, 1), dtype=np.int16))
    original = source.read_bytes()
    monkeypatch.setattr(video, 'resolve_runtime_input_path', lambda value: Path(value))
    output = tmp_path / 'result.mp4'
    result = video.ExportWithAudio('audio-preservation').execute(
        video=clip(), audio=str(source), filename=str(output), fps=8, quality=5)
    assert source.is_file() and source.read_bytes() == original
    assert output.is_file() and result['frames'] == 8
    assert set(tmp_path.iterdir()) == {source, output}


@pytest.mark.parametrize('fail_mux', [False, True])
def test_export_never_clobbers_existing_intermediate_names_and_cleans_failures(tmp_path, monkeypatch, fail_mux):
    output = tmp_path / 'result.mp4'
    (tmp_path / 'result.silent.mp4').write_bytes(b'pre-existing video input')
    (tmp_path / 'result.audio.wav').write_bytes(b'pre-existing audio input')
    if fail_mux:
        output.write_bytes(b'previous completed output')
    before = files(tmp_path)
    if fail_mux:
        def fail(*args, **kwargs):
            Path(args[0][-1]).write_bytes(b'incomplete mux')
            raise subprocess.CalledProcessError(1, args[0])
        monkeypatch.setattr(subprocess, 'run', fail)
    def export():
        return video.ExportWithAudio('temporary-ownership').execute(
            video=clip(), audio={'samples': np.zeros((48000, 1), dtype=np.float32), 'sample_rate': 48000},
            filename=str(output), fps=8, quality=5)
    if fail_mux:
        with pytest.raises(subprocess.CalledProcessError):
            export()
        assert files(tmp_path) == before
    else:
        export()
        after = files(tmp_path)
        assert {name: after.get(name) for name in before} == before
        assert set(after) == {*before, output.name}


@pytest.mark.parametrize('hardlink', [False, True])
def test_export_rejects_destination_that_aliases_its_audio_input(tmp_path, monkeypatch, hardlink):
    source = tmp_path / 'source.wav'
    wavfile.write(source, 48000, np.zeros((48000, 1), dtype=np.int16))
    output = source
    if hardlink:
        output = tmp_path / 'alias.wav'
        try:
            output.hardlink_to(source)
        except OSError as error:
            pytest.skip(f'Hard links unavailable: {error}')
    monkeypatch.setattr(video, 'resolve_runtime_input_path', lambda value: Path(value))
    before = files(tmp_path)
    with pytest.raises(ValueError, match='audio input'):
        video.ExportWithAudio('source-alias').execute(
            video=clip(), audio=str(source), filename=str(output), fps=8, quality=5)
    assert files(tmp_path) == before


@pytest.mark.parametrize('frame_count', [25, 124])
@pytest.mark.parametrize('audio_adjustment', [-16000, -1, 0, 1, 16000])
def test_audio_rounding_never_discards_trailing_video_frames(tmp_path, frame_count, audio_adjustment):
    frames = [Image.fromarray(np.full((32, 32, 3), index * 2 % 256, dtype=np.uint8))
              for index in range(frame_count)]
    count = int(frame_count / 24 * 32000) + audio_adjustment
    output = tmp_path / 'complete.mp4'
    result = video.ExportWithAudio('complete-frames').execute(video=frames,
        audio={'samples': np.zeros((count, 2), dtype=np.float32), 'sample_rate': 32000},
        filename=str(output), fps=24, quality=8)
    reader = imageio.get_reader(output)
    try:
        assert reader.count_frames() == result['frames'] == frame_count
    finally:
        reader.close()
    from imageio_ffmpeg import get_ffmpeg_exe
    raw_audio = subprocess.check_output([get_ffmpeg_exe(), '-v', 'error', '-i', str(output),
        '-map', '0:a:0', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
    # AAC can pad its last1024-sample packet; it must neither leave long audio
    # after the video nor stop early when a borrowed soundtrack is shorter.
    assert abs(len(raw_audio) / (4 * 2 * 32000) - frame_count / 24) <= 1024 / 32000


@pytest.mark.parametrize('fail_mux', [False, True])
def test_retained_video_mux_preserves_existing_audio_named_like_an_intermediate(tmp_path, monkeypatch, fail_mux):
    from modiff import media_assets
    source = tmp_path / 'clip.mp4'
    source.write_bytes(b'retained video')
    sibling = tmp_path / '.clip-audio.wav'
    sibling.write_bytes(b'important existing audio')
    before = {path: path.read_bytes() for path in (source, sibling)}
    output = tmp_path / 'new-mux.mp4'
    monkeypatch.setattr(video, '_file_asset_collection', lambda value: [{'path': str(source), 'duration_seconds': 1.}])
    monkeypatch.setattr(media_assets, 'allocate_video_path', lambda **kw: ('owned-asset', output))
    monkeypatch.setattr(video, '_derived_asset_result', lambda *args, **kw: {'file': str(output)})

    def mux(arguments, destination):
        destination.write_bytes(b'new output')
        if fail_mux:
            raise RuntimeError('controlled mux failure')

    monkeypatch.setattr(media_assets, 'run_ffmpeg', mux)
    def execute():
        return video.MuxAudioAsset('owned-audio').execute(video=str(source),
            audio={'samples': np.zeros((8000, 2)), 'sample_rate': 8000}, fit='match_video')
    if fail_mux:
        with pytest.raises(RuntimeError, match='controlled mux failure'):
            execute()
    else:
        execute()
    assert all(path.read_bytes() == body for path, body in before.items())
    assert not [path for path in tmp_path.iterdir() if path.is_dir()]
