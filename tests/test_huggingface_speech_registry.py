from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import inspect
import os
import tempfile
import unittest

import numpy as np

from modiff.execution_input_provenance import bind_generation_input_origins
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

    def test_speech_receipts_observe_actual_controls_and_bound_audio(self):
        from scipy.io import wavfile
        from modules.Audio.main import Load

        calls = []

        def recognize(audio, **kwargs):
            calls.append((audio, kwargs))
            return {"text": "example"}

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"MODIFF_CAPTURE_BUILTIN_MEDIA": "1"}):
            path = Path(directory, "synthetic.wav")
            wavfile.write(path, 16000, np.array([[4000, -2000], [8000, 4000]], dtype=np.int16))
            loader = Load("source")
            source = loader(file=str(path))["audio"]
            upstream = loader._execution_input_record["mediaEvidence"]["outputs"]["audio"]
            for action, family, controls, expected_call, inactive in (
                (
                    TranscribeAudio, SPEECH_MODEL_FAMILY_SEQ2SEQ,
                    {"task": "transcribe", "language": "", "timestamps": "segment",
                     "chunk_length_seconds": 30, "stride_length_seconds": 5},
                    {"generate_kwargs": {"task": "transcribe"}, "return_timestamps": True,
                     "chunk_length_s": 30.0, "stride_length_s": 5.0},
                    {"language"},
                ),
                (
                    TranscribeCTCAudio, SPEECH_MODEL_FAMILY_CTC,
                    {"timestamps": "word", "chunk_length_seconds": 0, "stride_length_seconds": 0},
                    {"return_timestamps": "word"},
                    {"chunk_length_seconds", "stride_length_seconds"},
                ),
            ):
                with self.subTest(action=action.__name__):
                    node = action("speech")
                    model = {"schemaVersion": 1, "family": family, "pipeline": recognize}
                    outputs = node(model=model, audio=source, **controls)
                    self.assertEqual(outputs["text"], "example")
                    recognizer_input, called = calls[-1]
                    self.assertEqual(called, expected_call)
                    self.assertEqual(recognizer_input["sampling_rate"], 16000)
                    np.testing.assert_array_equal(recognizer_input["raw"], [1000 / 32768, 6000 / 32768])

                    params = {key: {"value": value} for key, value in controls.items()}
                    params["model"] = {"sourceId": "loader", "sourceKey": "model"}
                    params["audio"] = {"sourceId": "source", "sourceKey": "audio"}
                    receipt = bind_generation_input_origins("speech", {
                        "module": "modules.HuggingFaceSpeech", "action": action.__name__, "params": params,
                    }, node._execution_input_record)
                    accounting = receipt["fieldAccounting"]
                    self.assertTrue(accounting["complete"])
                    self.assertEqual(set(accounting["fields"]), set(controls))
                    self.assertEqual(accounting["inputProjections"], {})
                    self.assertEqual(accounting["outputProjections"], {})
                    self.assertEqual(receipt["fields"], {})
                    self.assertEqual(receipt["mediaEvidence"], {
                        "schemaVersion": 1, "source": "native-builtin-media-v1", "status": "complete",
                        "controls": {}, "inputs": {"audio": upstream}, "outputs": {}, "files": {},
                    })
                    self.assertEqual(upstream[0]["shape"], [2, 2])
                    effective = {
                        "task": expected_call.get("generate_kwargs", {}).get("task"),
                        "timestamps": expected_call["return_timestamps"],
                        "chunk_length_seconds": expected_call.get("chunk_length_s"),
                        "stride_length_seconds": expected_call.get("stride_length_s"),
                    }
                    for key, value in controls.items():
                        self.assertEqual(accounting["fields"][key], {
                            "source": "literal", "inputValue": value,
                            **({"disposition": "inactive", "value": value, "reason": "feature-disabled"}
                               if key in inactive else {"disposition": "consumed", "value": effective[key]}),
                        })
                    count = len(calls)
                    node(model=model, audio=source, **controls)
                    self.assertEqual(len(calls), count)
                    self.assertEqual(node._execution_input_record["mediaEvidence"], receipt["mediaEvidence"])

    def test_speech_direct_file_keeps_existing_path_and_accounting_without_file_claim(self):
        from scipy.io import wavfile

        recognizer = Mock(return_value={"text": "example"})
        model = {"schemaVersion": 1, "family": SPEECH_MODEL_FAMILY_SEQ2SEQ, "pipeline": recognizer}
        node = TranscribeAudio("speech")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "source.wav")
            wavfile.write(path, 16000, np.zeros((4, 2), dtype=np.int16))
            with (
                patch.dict(os.environ, {"MODIFF_CAPTURE_BUILTIN_MEDIA": "1"}),
                patch("modules.DiffusersAudio.main._resolve_audio_source_path", return_value=path),
                patch("modiff.builtin_media_evidence.file_identity") as evidence_file,
            ):
                output = node(model=model, audio="source.wav", task="translate", language=" French ",
                              timestamps="none", chunk_length_seconds=0, stride_length_seconds=0)
                evidence_file.assert_not_called()
        self.assertEqual(output["transcript"]["task"], "translate")
        self.assertEqual(recognizer.call_args.kwargs, {
            "generate_kwargs": {"task": "translate", "language": "French"}, "return_timestamps": False,
        })
        self.assertEqual(node._execution_input_record["fieldAccounting"]["fields"]["language"], {
            "disposition": "consumed", "inputValue": " French ", "value": "French",
        })
        self.assertEqual(node._execution_input_record["mediaEvidence"], {
            "schemaVersion": 1, "status": "unavailable", "reason": "unsupported-or-bounded-input",
        })

    def test_speech_accounting_uses_successful_retry_dispatch(self):
        fail = False

        def recognize(_audio, **_kwargs):
            if fail:
                raise ValueError("test recognizer failure")
            return {"text": "example"}

        node = TranscribeCTCAudio("speech")
        model = {"schemaVersion": 1, "family": SPEECH_MODEL_FAMILY_CTC, "pipeline": recognize}
        audio = {"samples": np.zeros((12, 1), dtype=np.float32), "sample_layout": "frames_first",
                 "sample_rate": 16000, "channels": 1}
        with patch.dict(os.environ, {"MODIFF_CAPTURE_BUILTIN_MEDIA": "0"}):
            node(model=model, audio=audio, timestamps="word")
            fail = True
            with self.assertRaisesRegex(RuntimeError, "test recognizer failure"):
                node(model=model, audio=audio, timestamps="none")
            self.assertFalse(node._cache_valid)
            fail = False
            node(model=model, audio=audio, timestamps="none")

        self.assertEqual(node._execution_input_record["fieldAccounting"]["fields"]["timestamps"], {
            "disposition": "consumed", "inputValue": "none", "value": False,
        })
        self.assertNotIn("mediaEvidence", node._execution_input_record)

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
