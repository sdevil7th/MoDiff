import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.Audio.main import (
    Export,
    FitDuration,
    Join,
    MatchLoudness,
    TrimPad,
    _atempo_factors,
    _audio_to_numpy,
    _read_wav,
)


class AudioExportTests(unittest.TestCase):
    def test_explicit_sample_layout_preserves_square_stereo_payloads(self):
        square = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        frames_first, _ = _audio_to_numpy(
            {
                "samples": square,
                "sample_rate": 48000,
                "channels": 2,
                "sample_layout": "frames_first",
            }
        )
        channels_first, _ = _audio_to_numpy(
            {
                "samples": square,
                "sample_rate": 48000,
                "channels": 2,
                "sample_layout": "channels_first",
            }
        )

        np.testing.assert_array_equal(frames_first, square)
        np.testing.assert_array_equal(channels_first, square.T)

    def test_trim_pad_consumes_a_square_diffusers_channels_first_audio_object(self):
        channels_first = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

        result = TrimPad().execute(
            audio={
                "samples": channels_first,
                "sample_layout": "channels_first",
                "sample_rate": 48000,
                "channels": 2,
                "duration_seconds": 2 / 48000,
            },
            target_sample_rate=48000,
        )

        self.assertEqual(result["output"]["sample_layout"], "frames_first")
        self.assertEqual(result["output"]["channels"], 2)
        np.testing.assert_array_equal(result["output"]["samples"], channels_first.T)

    def test_trim_pad_rejects_contradictory_layout_and_channel_metadata(self):
        cases = (
            (
                {
                    "samples": np.zeros((2, 3), dtype=np.float32),
                    "sample_layout": "frames_first",
                    "sample_rate": 48000,
                    "channels": 2,
                },
                "frames_first identifies 3",
            ),
            (
                {
                    "samples": np.zeros((3, 2), dtype=np.float32),
                    "sample_layout": "channels_first",
                    "sample_rate": 48000,
                    "channels": 2,
                },
                "channels_first identifies 3",
            ),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    TrimPad().execute(audio=payload, target_sample_rate=48000)

        for channels in (True, 0, -1, 1.5, float("nan"), float("inf"), "two"):
            with self.subTest(channels=channels):
                with self.assertRaisesRegex(ValueError, "channels metadata must be a positive integer"):
                    TrimPad().execute(
                        audio={
                            "samples": np.zeros((3, 2), dtype=np.float32),
                            "sample_layout": "frames_first",
                            "sample_rate": 48000,
                            "channels": channels,
                        },
                        target_sample_rate=48000,
                    )

    def test_trim_pad_uses_decoder_provenance_for_path_backed_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "two-channel.wav"
            wavfile.write(path, 48000, np.zeros((3, 2), dtype=np.int16))

            with patch("modules.Audio.main._read_wav") as decoder:
                with self.assertRaisesRegex(ValueError, "Decoded audio files use frames_first"):
                    TrimPad().execute(
                        audio={
                            "samples": str(path),
                            "sample_layout": "channels_first",
                            "sample_rate": 48000,
                            "channels": 2,
                        },
                        target_sample_rate=48000,
                    )
                decoder.assert_not_called()

            with self.assertRaisesRegex(ValueError, "declares 1 channels, but the decoded file has 2"):
                TrimPad().execute(
                    audio={
                        "samples": str(path),
                        "sample_layout": "frames_first",
                        "sample_rate": 48000,
                        "channels": 1,
                    },
                    target_sample_rate=48000,
                )

            result = TrimPad().execute(
                audio={
                    "samples": str(path),
                    "sample_layout": "frames_first",
                    "sample_rate": 48000,
                    "channels": 2,
                },
                target_sample_rate=48000,
            )
        self.assertEqual(result["output"]["samples"].shape, (3, 2))
        self.assertEqual(result["output"]["sample_layout"], "frames_first")

    def test_unsigned_pcm_midpoint_is_silence_for_arrays_and_wav_files(self):
        source = np.asarray([0, 128, 255], dtype=np.uint8)
        converted, _sample_rate = _audio_to_numpy({"samples": source, "sample_rate": 8000})

        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "unsigned.wav"
            wavfile.write(wav_path, 8000, source)
            loaded = _read_wav(wav_path)

        np.testing.assert_allclose(converted[:, 0], [-1.0, 0.0, 127 / 128], atol=1e-7)
        np.testing.assert_allclose(loaded["samples"], [-1.0, 0.0, 127 / 128], atol=1e-7)

    def test_sample_rate_options_include_music_delivery_rates(self):
        options = Export.params["sample_rate"]["options"]

        self.assertEqual(set(options), {"44100", "48000", "88200", "96000"})
        self.assertEqual(options["44100"], "44.1 kHz")
        self.assertEqual(options["48000"], "48 kHz")
        self.assertEqual(options["88200"], "88.2 kHz")
        self.assertEqual(options["96000"], "96 kHz")

    def test_export_resamples_without_changing_duration_or_pitch(self):
        source_rate = 48000
        target_rate = 44100
        frequency = 440
        times = np.arange(source_rate, dtype=np.float32) / source_rate
        source = np.sin(2 * np.pi * frequency * times).astype(np.float32)[:, None]

        with tempfile.TemporaryDirectory(prefix="modiff-audio-export-") as temporary_dir:
            output_path = Path(temporary_dir) / "export.wav"
            result = Export("audio-export-sample-rate-test")(
                audio={"samples": source, "sample_rate": source_rate},
                filename=str(output_path),
                sample_rate=target_rate,
            )
            written_rate, written = wavfile.read(output_path)

        self.assertEqual(written_rate, target_rate)
        self.assertEqual(written.shape[0], target_rate)
        self.assertAlmostEqual(result["duration_seconds"], 1.0, places=6)
        frequencies = np.fft.rfftfreq(written.shape[0], d=1 / written_rate)
        peak_frequency = float(frequencies[np.argmax(np.abs(np.fft.rfft(written)))])
        self.assertAlmostEqual(peak_frequency, frequency, delta=1)


class AudioFitDurationTests(unittest.TestCase):
    def test_exact_window_and_positive_delay_do_not_time_warp(self):
        sample_rate = 8000
        frame_count = sample_rate * 2
        source = np.linspace(-0.8, 0.8, frame_count, dtype=np.float32)[:, None]
        source = np.repeat(source, 2, axis=1)

        result = FitDuration().execute(
            audio={"samples": source, "sample_rate": sample_rate},
            source_start_seconds=0.25,
            source_duration_seconds=1.0,
            target_duration_seconds=1.0,
            delay_seconds=0.1,
            target_sample_rate=sample_rate,
        )

        output = result["output"]["samples"]
        delay_frames = 800
        expected = source[2000 : 2000 + sample_rate - delay_frames]
        self.assertEqual(output.shape, (sample_rate, 2))
        np.testing.assert_array_equal(output[:delay_frames], 0)
        np.testing.assert_allclose(output[delay_frames:], expected, atol=0, rtol=0)
        self.assertEqual(result["duration"], 1.0)
        self.assertEqual(result["tempo_ratio"], 1.0)
        self.assertEqual(result["stretch_engine"], "none")

    def test_fades_are_applied_to_shifted_content_boundaries(self):
        sample_rate = 8000
        source = np.ones((sample_rate, 1), dtype=np.float32)

        result = FitDuration().execute(
            audio={"samples": source, "sample_rate": sample_rate},
            source_duration_seconds=1.0,
            target_duration_seconds=1.0,
            delay_seconds=0.1,
            target_sample_rate=sample_rate,
            fade_in_seconds=0.01,
            fade_out_seconds=0.02,
        )

        output = result["output"]["samples"][:, 0]
        self.assertTrue(np.all(output[:800] == 0))
        self.assertAlmostEqual(float(output[800]), 0.0, places=6)
        self.assertGreater(float(output[879]), 0.99)
        self.assertGreater(float(output[-161]), 0.99)
        self.assertAlmostEqual(float(output[-1]), 0.0, places=6)

    def test_pitch_preserving_fit_has_exact_duration_and_retains_tone(self):
        sample_rate = 8000
        source_duration = 1.0
        target_duration = 0.75
        times = np.arange(round(sample_rate * source_duration), dtype=np.float32) / sample_rate
        source = np.sin(2 * np.pi * 440 * times).astype(np.float32)[:, None]

        result = FitDuration().execute(
            audio={"samples": source, "sample_rate": sample_rate},
            source_duration_seconds=source_duration,
            target_duration_seconds=target_duration,
            target_sample_rate=sample_rate,
        )

        output = result["output"]["samples"][:, 0]
        analysis = output[800:-800]
        frequencies = np.fft.rfftfreq(analysis.shape[0], d=1 / sample_rate)
        peak_frequency = float(frequencies[np.argmax(np.abs(np.fft.rfft(analysis)))])
        self.assertEqual(output.shape[0], round(sample_rate * target_duration))
        self.assertAlmostEqual(result["tempo_ratio"], source_duration / target_duration, places=6)
        self.assertIn(result["stretch_engine"], {"rubberband", "atempo"})
        self.assertAlmostEqual(peak_frequency, 440, delta=12)

    def test_atempo_factors_stay_in_the_high_quality_range(self):
        for ratio in (0.1, 0.49, 1.0, 2.1, 8.0):
            factors = _atempo_factors(ratio)
            self.assertTrue(all(0.5 <= factor <= 2.0 for factor in factors))
            self.assertAlmostEqual(float(np.prod(factors)), ratio, places=9)


class AudioMatchLoudnessTests(unittest.TestCase):
    def test_matches_reference_loudness_without_exceeding_peak_ceiling(self):
        sample_rate = 48000
        times = np.arange(sample_rate * 6, dtype=np.float32) / sample_rate
        reference = (0.45 * np.sin(2 * np.pi * 220 * times)).astype(np.float32)[:, None]
        generated = (0.08 * np.sin(2 * np.pi * 220 * times)).astype(np.float32)[:, None]

        result = MatchLoudness().execute(
            audio={"samples": generated, "sample_rate": sample_rate},
            reference={"samples": reference, "sample_rate": sample_rate},
            reference_window_seconds=6,
            target_peak_dbfs=-1,
            max_adjustment_db=20,
        )

        output = result["output"]["samples"]
        self.assertEqual(output.shape, generated.shape)
        self.assertLess(abs(result["output_lufs"] - result["reference_lufs"]), 0.5)
        self.assertGreater(result["adjustment_db"], 10)
        self.assertLessEqual(result["true_peak_dbfs"], -0.7)

    def test_preserves_internal_dynamics_with_one_constant_gain(self):
        sample_rate = 48000
        times = np.arange(sample_rate * 4, dtype=np.float32) / sample_rate
        carrier = np.sin(2 * np.pi * 220 * times).astype(np.float32)
        envelope = np.repeat(
            np.asarray([0.04, 0.12, 0.025, 0.08], dtype=np.float32),
            sample_rate,
        )
        generated = (carrier * envelope)[:, None]
        reference = (0.35 * carrier)[:, None]

        result = MatchLoudness().execute(
            audio={"samples": generated, "sample_rate": sample_rate},
            reference={"samples": reference, "sample_rate": sample_rate},
            reference_window_seconds=4,
            target_peak_dbfs=-1,
            max_adjustment_db=20,
        )

        output = result["output"]["samples"]
        nonzero = np.abs(generated[:, 0]) > 1e-5
        sample_gain = output[nonzero, 0] / generated[nonzero, 0]
        self.assertLess(float(np.max(sample_gain) - np.min(sample_gain)), 1e-4)


class AudioJoinTests(unittest.TestCase):
    def test_appends_resampled_continuation_without_shortening_total_duration(self):
        source_rate = 48000
        continuation_rate = 24000
        source = np.ones((source_rate * 2, 2), dtype=np.float32) * 0.25
        continuation = np.ones((continuation_rate, 1), dtype=np.float32) * 0.5

        result = Join().execute(
            source={"samples": source, "sample_rate": source_rate},
            continuation={"samples": continuation, "sample_rate": continuation_rate},
            boundary_fade_seconds=0.01,
        )

        output = result["output"]["samples"]
        self.assertEqual(output.shape, (source_rate * 3, 2))
        self.assertEqual(result["sample_rate"], source_rate)
        self.assertEqual(result["duration"], 3.0)
        self.assertAlmostEqual(float(output[source_rate * 2 - 1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(output[source_rate * 2, 0]), 0.0, places=6)
        self.assertGreater(float(output[source_rate * 2 + 480, 0]), 0.49)


if __name__ == "__main__":
    unittest.main()
