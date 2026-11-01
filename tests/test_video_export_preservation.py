"""Video/audio export preserves borrowed inputs and commits only completed media."""
from pathlib import Path
import subprocess

import numpy as np
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
