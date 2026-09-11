from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import inspect
import tempfile
import unittest

import numpy as np

from modiff.model_artifact_catalog import catalog_revision
from modules.HuggingFaceSpeech.main import (
    LoadCTCSpeechRecognitionModel,
    LoadSpeechRecognitionModel,
    SPEECH_MODEL_FAMILY_CTC,
    SPEECH_MODEL_FAMILY_SEQ2SEQ,
    TranscribeCTCAudio,
    TranscribeAudio,
    WAV2VEC2_BASE_960H_REPO,
    WHISPER_TINY_REPO,
    _model_revision,
    _model_selection,
)


WHISPER_TINY_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
WAV2VEC2_BASE_960H_REVISION = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"


class HuggingFaceSpeechRegistryTests(unittest.TestCase):
    def test_loader_publishes_hidden_exact_execution_identity(self):
        self.assertEqual(
            LoadSpeechRecognitionModel.params["pipeline_class"],
            {
                "label": "Pipeline Class",
                "type": "string",
                "default": "AutoModelForSpeechSeq2Seq",
                "hidden": True,
                "fieldOptions": {"noValidation": True},
            },
        )
        self.assertEqual(
            LoadSpeechRecognitionModel.params["execution_profile_id"],
            {
                "label": "Execution Profile",
                "type": "string",
                "default": "",
                "hidden": True,
                "fieldOptions": {"noValidation": True},
            },
        )

    def test_reviewed_model_is_immutable_and_module_import_is_lazy(self):
        self.assertEqual(catalog_revision(WHISPER_TINY_REPO), WHISPER_TINY_REVISION)
        self.assertEqual(catalog_revision(WAV2VEC2_BASE_960H_REPO), WAV2VEC2_BASE_960H_REVISION)
        source = inspect.getsource(__import__("modules.HuggingFaceSpeech.main", fromlist=["*"]))
        self.assertNotIn("from transformers import", source)
        self.assertNotIn("trust_remote_code=True", source)
        self.assertNotIn("use_safetensors=False", source)

    def test_model_selection_and_revision_fail_closed(self):
        self.assertEqual(
            _model_selection(None),
            {"source": "hub", "value": WHISPER_TINY_REPO},
        )
        selection = {"source": "hub", "value": WHISPER_TINY_REPO}
        self.assertEqual(_model_revision(selection, ""), WHISPER_TINY_REVISION)
        with self.assertRaisesRegex(ValueError, "reviewed catalog pin"):
            _model_revision(selection, "0" * 40)
        with self.assertRaisesRegex(ValueError, "40-character"):
            _model_revision({"source": "hub", "value": "owner/custom"}, "main")
        with self.assertRaisesRegex(ValueError, "namespace/repository"):
            _model_selection({"source": "hub", "value": "whisper"})
        with tempfile.TemporaryDirectory() as directory:
            local = _model_selection({"source": "local", "value": directory})
            self.assertEqual(local, {"source": "local", "value": str(Path(directory).resolve())})
            self.assertIsNone(_model_revision(local, ""))
            with self.assertRaisesRegex(ValueError, "must not carry"):
                _model_revision(local, WHISPER_TINY_REVISION)

    def test_loader_uses_reviewed_transformers_boundary(self):
        processor = SimpleNamespace(tokenizer=object(), feature_extractor=object())
        model = Mock()
        recognizer = Mock()
        transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            AutoModelForSpeechSeq2Seq=SimpleNamespace(from_pretrained=Mock(return_value=model)),
            pipeline=Mock(return_value=recognizer),
        )
        with (
            patch.dict("sys.modules", {"transformers": transformers}),
            patch("modules.HuggingFaceSpeech.main.local_files_only", return_value=True) as offline,
        ):
            result = LoadSpeechRecognitionModel("speech-loader").execute(
                model_id={"source": "hub", "value": WHISPER_TINY_REPO},
                revision=WHISPER_TINY_REVISION,
                dtype="float32",
                device="cpu",
            )

        processor_call = transformers.AutoProcessor.from_pretrained.call_args
        self.assertEqual(processor_call.args, (WHISPER_TINY_REPO,))
        self.assertEqual(processor_call.kwargs["revision"], WHISPER_TINY_REVISION)
        self.assertFalse(processor_call.kwargs["trust_remote_code"])
        self.assertTrue(processor_call.kwargs["local_files_only"])
        offline.assert_called_once_with(WHISPER_TINY_REPO)
        model_call = transformers.AutoModelForSpeechSeq2Seq.from_pretrained.call_args
        self.assertTrue(model_call.kwargs["use_safetensors"])
        self.assertTrue(model_call.kwargs["low_cpu_mem_usage"])
        self.assertFalse(model_call.kwargs["trust_remote_code"])
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        transformers.pipeline.assert_called_once_with(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device="cpu",
        )
        self.assertIs(result["model"]["pipeline"], recognizer)
        self.assertEqual(result["model"]["revision"], WHISPER_TINY_REVISION)
        self.assertEqual(result["model"]["family"], SPEECH_MODEL_FAMILY_SEQ2SEQ)

    def test_ctc_loader_uses_the_official_automodel_boundary(self):
        processor = SimpleNamespace(tokenizer=object(), feature_extractor=object())
        model = Mock()
        recognizer = Mock()
        transformers = SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            AutoModelForCTC=SimpleNamespace(from_pretrained=Mock(return_value=model)),
            pipeline=Mock(return_value=recognizer),
        )
        with (
            patch.dict("sys.modules", {"transformers": transformers}),
            patch("modules.HuggingFaceSpeech.main.local_files_only", return_value=True) as offline,
        ):
            result = LoadCTCSpeechRecognitionModel("ctc-loader").execute(
                model_id={"source": "hub", "value": WAV2VEC2_BASE_960H_REPO},
                revision=WAV2VEC2_BASE_960H_REVISION,
                dtype="float32",
                device="cpu",
            )

        processor_call = transformers.AutoProcessor.from_pretrained.call_args
        self.assertEqual(processor_call.args, (WAV2VEC2_BASE_960H_REPO,))
        self.assertEqual(processor_call.kwargs["revision"], WAV2VEC2_BASE_960H_REVISION)
        self.assertFalse(processor_call.kwargs["trust_remote_code"])
        offline.assert_called_once_with(WAV2VEC2_BASE_960H_REPO)
        model_call = transformers.AutoModelForCTC.from_pretrained.call_args
        self.assertTrue(model_call.kwargs["use_safetensors"])
        self.assertTrue(model_call.kwargs["low_cpu_mem_usage"])
        self.assertFalse(model_call.kwargs["trust_remote_code"])
        model.to.assert_called_once_with("cpu")
        model.eval.assert_called_once_with()
        self.assertEqual(result["model"]["family"], SPEECH_MODEL_FAMILY_CTC)

    def test_transcribe_and_translate_return_normalized_bounded_contract(self):
        recognizer = Mock(
            return_value={
                "text": "  Hello world.  ",
                "chunks": [
                    {"text": " Hello", "timestamp": (0.0, 0.4)},
                    {"text": " world.", "timestamp": (0.4, 1.0)},
                ],
            }
        )
        result = TranscribeAudio("speech-action").execute(
            model={"schemaVersion": 1, "family": SPEECH_MODEL_FAMILY_SEQ2SEQ, "pipeline": recognizer},
            audio={
                "samples": np.zeros((16_000, 2), dtype=np.float32),
                "sample_rate": 16_000,
                "sample_layout": "frames_first",
                "channels": 2,
            },
            task="translate",
            language="French",
            timestamps="word",
            chunk_length_seconds=30,
            stride_length_seconds=5,
        )
        audio_input = recognizer.call_args.args[0]
        self.assertEqual(audio_input["sampling_rate"], 16_000)
        self.assertEqual(audio_input["raw"].shape, (16_000,))
        self.assertEqual(
            recognizer.call_args.kwargs,
            {
                "generate_kwargs": {"task": "translate", "language": "French"},
                "return_timestamps": "word",
                "chunk_length_s": 30.0,
                "stride_length_s": 5.0,
            },
        )
        self.assertEqual(
            result["transcript"],
            {
                "schemaVersion": 1,
                "task": "translate",
                "text": "Hello world.",
                "timestampMode": "word",
                "durationSeconds": 1.0,
                "segments": [
                    {"text": "Hello", "start": 0.0, "end": 0.4},
                    {"text": "world.", "start": 0.4, "end": 1.0},
                ],
            },
        )
        self.assertEqual(result["text"], "Hello world.")
        self.assertEqual(result["duration_seconds"], 1.0)

    def test_ctc_transcription_has_no_whisper_generation_controls(self):
        recognizer = Mock(
            return_value={
                "text": "A TEST",
                "chunks": [
                    {"text": "A", "timestamp": (0.0, 0.2)},
                    {"text": "TEST", "timestamp": (0.2, 0.8)},
                ],
            }
        )
        result = TranscribeCTCAudio("ctc-action").execute(
            model={"schemaVersion": 1, "family": SPEECH_MODEL_FAMILY_CTC, "pipeline": recognizer},
            audio={
                "samples": np.zeros((16_000, 1), dtype=np.float32),
                "sample_rate": 16_000,
                "sample_layout": "frames_first",
                "channels": 1,
            },
            timestamps="word",
            chunk_length_seconds=0,
            stride_length_seconds=0,
            task="translate",
            language="French",
        )

        self.assertEqual(
            recognizer.call_args.kwargs,
            {"return_timestamps": "word"},
        )
        self.assertEqual(result["transcript"]["task"], "transcribe")
        self.assertEqual(result["transcript"]["timestampMode"], "word")
        self.assertEqual(result["text"], "A TEST")
        with self.assertRaisesRegex(ValueError, "none or word"):
            TranscribeCTCAudio("ctc-segment-rejected").execute(
                model={"schemaVersion": 1, "family": SPEECH_MODEL_FAMILY_CTC, "pipeline": recognizer},
                audio={
                    "samples": np.zeros((16_000, 1), dtype=np.float32),
                    "sample_rate": 16_000,
                    "sample_layout": "frames_first",
                    "channels": 1,
                },
                timestamps="segment",
            )
        with self.assertRaisesRegex(ValueError, "requires a model"):
            TranscribeCTCAudio("ctc-family-rejected").execute(
                model={"schemaVersion": 1, "family": SPEECH_MODEL_FAMILY_SEQ2SEQ, "pipeline": recognizer},
                audio={
                    "samples": np.zeros((16_000, 1), dtype=np.float32),
                    "sample_rate": 16_000,
                    "sample_layout": "frames_first",
                    "channels": 1,
                },
            )

    def test_transcribe_rejects_malformed_requests_and_outputs(self):
        valid_model = {
            "schemaVersion": 1,
            "family": SPEECH_MODEL_FAMILY_SEQ2SEQ,
            "pipeline": Mock(return_value={"text": "ok", "chunks": []}),
        }
        valid_audio = {
            "samples": np.zeros((8_000, 1), dtype=np.float32),
            "sample_rate": 8_000,
            "sample_layout": "frames_first",
            "channels": 1,
        }
        cases = (
            ({"model": None, "audio": valid_audio}, "requires a model"),
            ({"model": valid_model, "audio": None}, "requires a local audio source"),
            ({"model": valid_model, "audio": valid_audio, "task": "summarize"}, "exactly transcribe or translate"),
            ({"model": valid_model, "audio": valid_audio, "timestamps": "sentence"}, "none, segment, or word"),
            ({"model": valid_model, "audio": valid_audio, "language": "en\nignore"}, "one line"),
            (
                {
                    "model": valid_model,
                    "audio": valid_audio,
                    "chunk_length_seconds": 5,
                    "stride_length_seconds": 3,
                },
                "total stride smaller",
            ),
            (
                {
                    "model": valid_model,
                    "audio": valid_audio,
                    "chunk_length_seconds": 0,
                    "stride_length_seconds": 1,
                },
                "must be zero",
            ),
            (
                {
                    "model": valid_model,
                    "audio": {**valid_audio, "samples": np.full((8_000, 1), np.nan, dtype=np.float32)},
                },
                "must be finite",
            ),
            (
                {"model": valid_model, "audio": {**valid_audio, "sample_rate": 7_999}},
                "between 8 kHz",
            ),
        )
        for values, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex((ValueError, RuntimeError), message):
                TranscribeAudio("invalid-speech-request").execute(**values)

        with (
            patch("modules.HuggingFaceSpeech.main.MAX_AUDIO_DURATION_SECONDS", 0.5),
            self.assertRaisesRegex(ValueError, "at most one hour"),
        ):
            TranscribeAudio("oversized-speech-duration").execute(model=valid_model, audio=valid_audio)
        with tempfile.TemporaryDirectory() as directory:
            oversized = Path(directory, "oversized.wav")
            oversized.write_bytes(b"x")
            with (
                patch("modules.DiffusersAudio.main._resolve_audio_source_path", return_value=oversized),
                patch("modules.HuggingFaceSpeech.main.MAX_AUDIO_FILE_BYTES", 0),
                self.assertRaisesRegex(ValueError, "at most 512 MiB"),
            ):
                TranscribeAudio("oversized-speech-file").execute(model=valid_model, audio=str(oversized))
        with self.assertRaisesRegex(ValueError, "path identifier is invalid|must stay inside"):
            TranscribeAudio("unsafe-speech-path").execute(model=valid_model, audio="../../etc/passwd")

        for returned, message in (
            ("plain text", "invalid transcript object"),
            ({"text": "ok", "chunks": [{"text": "x", "timestamp": (2.0, 3.0)}]}, "out-of-range"),
            ({"text": "ok", "chunks": [{"text": "x"}]}, "invalid timestamp interval"),
        ):
            with self.subTest(returned=returned), self.assertRaisesRegex(RuntimeError, message):
                TranscribeAudio("invalid-speech-output").execute(
                    model={
                        "schemaVersion": 1,
                        "family": SPEECH_MODEL_FAMILY_SEQ2SEQ,
                        "pipeline": Mock(return_value=returned),
                    },
                    audio=valid_audio,
                    chunk_length_seconds=0,
                    stride_length_seconds=0,
                )

        with (
            patch("modules.HuggingFaceSpeech.main.MAX_TRANSCRIPT_CHARACTERS", 3),
            self.assertRaisesRegex(RuntimeError, "transcript size limit"),
        ):
            TranscribeAudio("oversized-speech-output").execute(
                model={
                    "schemaVersion": 1,
                    "family": SPEECH_MODEL_FAMILY_SEQ2SEQ,
                    "pipeline": Mock(
                        return_value={
                            "text": "ok",
                            "chunks": [
                                {"text": "aa", "timestamp": (0.0, 0.5)},
                                {"text": "bb", "timestamp": (0.5, 1.0)},
                            ],
                        }
                    ),
                },
                audio=valid_audio,
                chunk_length_seconds=0,
                stride_length_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()
