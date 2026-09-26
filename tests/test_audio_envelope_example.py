import importlib.util
from pathlib import Path

import numpy as np
import pytest


def test_audio_envelope_example_preserves_source_and_declares_exact_layout():
    path = Path(__file__).resolve().parents[1] / "examples/custom_nodes/AudioEnvelope.py"
    spec = importlib.util.spec_from_file_location("audio_envelope_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = np.full((48000, 2), 0.5, dtype=np.float32)
    audio = {"samples": source, "sample_rate": 48000, "sample_layout": "frames_first", "channels": 2}
    execute = module.AudioEnvelope.execute
    result = execute(None, audio, 0.2, 0.5, -6, 0.1)["output"]
    assert result["samples"].shape == (24000, 2)
    assert result["sample_layout"] == "frames_first"
    assert result["sample_rate"] == 48000
    assert result["duration_seconds"] == 0.5
    assert np.max(result["samples"]) == pytest.approx(0.5 * 10 ** (-6 / 20))
    assert np.all(result["samples"][[0, -1]] == 0)
    assert np.all(source == 0.5)
    with pytest.raises(ValueError, match="beyond"):
        execute(None, audio, 2, 0.5, 0, 0)
    with pytest.raises(ValueError, match="gain_db"):
        execute(None, audio, 0, 0.5, float("nan"), 0)
