"""Standalone CPU audio example: bounded trim, gain and equal-power fades.

Drop this file on the canvas, then connect Load Audio -> Audio Envelope ->
Preview Audio. Uses MoDiff's normalized waveform boundary; never loads a model.
"""

from modiff.NodeBase import NodeBase

MODIFF_RUNTIME_ROLE = "data"


class AudioEnvelope(NodeBase):
    label = "Audio Envelope"
    category = "Audio"
    params = {
        "audio": {"type": "audio", "display": "input", "label": "Audio"},
        "start_seconds": {"type": "float", "default": 0.0, "min": 0, "max": 300},
        "duration_seconds": {"type": "float", "default": 3.0, "min": 0.01, "max": 300},
        "gain_db": {"type": "float", "default": -6.0, "min": -60, "max": 12},
        "fade_seconds": {"type": "float", "default": 0.3, "min": 0, "max": 10},
        "output": {"type": "audio", "display": "output", "label": "Audio"},
    }

    def execute(self, audio, start_seconds, duration_seconds, gain_db, fade_seconds):
        import math
        import numpy as np
        from modules.Audio.main import _bounded_audio_object

        for name, value, minimum, maximum in (
            ("start_seconds", start_seconds, 0, 300),
            ("duration_seconds", duration_seconds, 0.01, 300),
            ("gain_db", gain_db, -60, 12),
            ("fade_seconds", fade_seconds, 0, 10),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be finite and between {minimum} and {maximum}.")
        source = _bounded_audio_object(audio, label="Audio Envelope input")
        rate = source["sample_rate"]
        start = round(start_seconds * rate)
        samples = source["samples"][start:start + round(duration_seconds * rate)].copy()
        if not len(samples):
            raise ValueError("Start is beyond the end of the source audio.")
        samples *= 10 ** (gain_db / 20)
        count = min(round(fade_seconds * rate), len(samples) // 2)
        if count:
            ramp = np.sin(np.linspace(0, np.pi / 2, count, dtype=np.float32))[:, None]
            samples[:count] *= ramp
            samples[-count:] *= ramp[::-1]
        return {"output": {
            "samples": np.clip(samples, -1, 1), "sample_layout": "frames_first",
            "sample_rate": rate, "channels": source["channels"],
            "duration_seconds": len(samples) / rate,
        }}
