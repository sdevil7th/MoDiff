from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES, TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
from modules.HuggingFaceTransformers.main import (
    GenerateImageVideoText,
    GenerateText,
    LoadImageTextToTextModel,
    LoadTextGenerationModel,
    SECURITY_CONTRACT,
    _batch_to_device,
    _generation_controls,
    _local_config_receipt,
    _model_revision,
    _model_selection,
    _normalized_media,
    _receipt_digest_is_valid,
)
from utils.torch_utils import DEFAULT_DEVICE


REVISION = "0123456789abcdef0123456789abcdef01234567"
REPOSITORY = "owner/model"


def _transformers_runtime(*, tokenizer=None, processor=None, model=None):
    return SimpleNamespace(
        __version__="5.14.1",
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )


def _load_text_handle(*, input_ids=None, output_ids=None, decoded="answer"):
    input_ids = torch.tensor([[1, 2]]) if input_ids is None else input_ids
    output_ids = torch.tensor([[1, 2, 3]]) if output_ids is None else output_ids
    tokenizer = Mock()
    tokenizer.return_value = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    tokenizer.decode.return_value = decoded
    model = Mock()
    model.config = SimpleNamespace(model_type="test", architectures=["TestForCausalLM"])
    model.generate.return_value = output_ids
    runtime = _transformers_runtime(tokenizer=tokenizer, model=model)
    with patch.dict("sys.modules", {"transformers": runtime}):
        loaded = LoadTextGenerationModel("load-text-test").execute(
            model_id={"source": "hub", "value": REPOSITORY},
            revision=REVISION,
            dtype="float32",
            device=DEFAULT_DEVICE,
        )
    return loaded, runtime, tokenizer, model


def _load_multimodal_handle(*, input_ids=None, output_ids=None, decoded="caption"):
    input_ids = torch.tensor([[10, 11]]) if input_ids is None else input_ids
    output_ids = torch.tensor([[10, 11, 12, 13]]) if output_ids is None else output_ids
    processor = Mock()
    processor.apply_chat_template.return_value = "<image><video> describe"
    processor.return_value = {"input_ids": input_ids, "pixel_values": torch.zeros((1, 3, 2, 2))}
    processor.decode.return_value = decoded
    model = Mock()
    model.config = SimpleNamespace(model_type="vision-test", architectures=["TestForImageTextToText"])
    model.generate.return_value = output_ids
    runtime = _transformers_runtime(processor=processor, model=model)
    with patch.dict("sys.modules", {"transformers": runtime}):
        loaded = LoadImageTextToTextModel("load-multimodal-test").execute(
            model_id=REPOSITORY,
            revision=REVISION,
            dtype="bfloat16",
            device=DEFAULT_DEVICE,
        )
    return loaded, runtime, processor, model


class HuggingFaceTransformersRegistryTests(unittest.TestCase):
    def test_registry_discovers_exact_generic_actions_and_import_is_lazy(self):
        from modules import MODULE_MAP

        self.assertEqual(
            set(MODULE_MAP["modules.HuggingFaceTransformers"]),
            {
                "LoadTextGenerationModel",
                "GenerateText",
                "LoadImageTextToTextModel",
                "GenerateImageVideoText",
            },
        )
        source = inspect.getsource(__import__("modules.HuggingFaceTransformers.main", fromlist=["*"]))
        self.assertNotIn("from transformers import", source)
        self.assertNotIn("\nimport transformers", source)
        self.assertNotIn("trust_remote_code=True", source)
        self.assertNotIn("use_safetensors=False", source)
        self.assertNotIn("snapshot_download", source)
        self.assertNotIn("InferenceClient", source)
        for class_name in MODULE_MAP["modules.HuggingFaceTransformers"]:
            self.assertEqual(
                MODULE_MAP["modules.HuggingFaceTransformers"][class_name]["category"],
                "Hugging Face Transformers",
            )

    def test_optional_runtime_qualifies_both_new_auto_model_symbols(self):
        transformers = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID].packages[0]
        expected = {"AutoModelForCausalLM", "AutoModelForImageTextToText"}
        self.assertTrue(expected.issubset(transformers.required_symbols))
        self.assertTrue(expected.issubset(transformers.required_class_symbols))
        self.assertEqual(transformers.version, "5.14.1")

    def test_model_selection_and_revision_require_an_explicit_immutable_source(self):
        for value in (None, "", "  "):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "no repository is selected"):
                _model_selection(value)
        self.assertEqual(_model_selection(REPOSITORY), {"source": "hub", "value": REPOSITORY})
        self.assertEqual(
            _model_selection({"source": "HUB", "value": REPOSITORY}),
            {"source": "hub", "value": REPOSITORY},
        )
        for value, message in (
            ("model", "namespace/repository"),
            ({"source": "remote", "value": REPOSITORY}, "exactly hub or local"),
            ({"source": "hub", "value": REPOSITORY, "revision": REVISION}, "repository ID or"),
            ({"source": "hub", "value": "owner/model\x00bad"}, "empty or invalid"),
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                _model_selection(value)
        hub = {"source": "hub", "value": REPOSITORY}
        self.assertEqual(_model_revision(hub, REVISION), REVISION)
        for revision in (None, "", "main", "A" * 40, "0" * 39, "0" * 41):
            with self.subTest(revision=revision), self.assertRaisesRegex(ValueError, "lowercase 40-character"):
                _model_revision(hub, revision)

    def test_local_selection_config_and_revision_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(
                json.dumps({"model_type": "safe", "architectures": ["SafeModel"]}),
                encoding="utf-8",
            )
            local = _model_selection({"source": "local", "value": directory})
            self.assertEqual(local, {"source": "local", "value": str(root.resolve())})
            self.assertIsNone(_model_revision(local, ""))
            with self.assertRaisesRegex(ValueError, "must not carry"):
                _model_revision(local, REVISION)
            receipt = _local_config_receipt(directory)
            self.assertRegex(receipt["configSha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(receipt["modelType"], "safe")
            self.assertEqual(receipt["architectures"], ["SafeModel"])

            config.write_text(json.dumps({"nested": {"auto_map": {"AutoModel": "evil.py"}}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not declare auto_map"):
                _local_config_receipt(directory)
            config.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain an object"):
                _local_config_receipt(directory)
            config.write_text("not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid bounded JSON"):
                _local_config_receipt(directory)
            config.write_text("{}", encoding="utf-8")
            with patch("modules.HuggingFaceTransformers.main.MAX_LOCAL_CONFIG_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "excessive size"):
                    _local_config_receipt(directory)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            (root / "config.json").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "non-symlink"):
                _local_config_receipt(directory)

    def test_text_loader_enforces_local_only_safe_exact_flags_and_sealed_receipt(self):
        loaded, runtime, _tokenizer, model = _load_text_handle()
        tokenizer_call = runtime.AutoTokenizer.from_pretrained.call_args
        self.assertEqual(tokenizer_call.args, (REPOSITORY,))
        self.assertEqual(
            tokenizer_call.kwargs,
            {
                "use_fast": True,
                "local_files_only": True,
                "trust_remote_code": False,
                "revision": REVISION,
            },
        )
        model_call = runtime.AutoModelForCausalLM.from_pretrained.call_args
        self.assertEqual(model_call.args, (REPOSITORY,))
        self.assertTrue(model_call.kwargs["use_safetensors"])
        self.assertTrue(model_call.kwargs["weights_only"])
        self.assertTrue(model_call.kwargs["low_cpu_mem_usage"])
        self.assertTrue(model_call.kwargs["local_files_only"])
        self.assertFalse(model_call.kwargs["trust_remote_code"])
        self.assertEqual(model_call.kwargs["revision"], REVISION)
        model.to.assert_called_once_with(DEFAULT_DEVICE)
        model.eval.assert_called_once_with()
        receipt = loaded["receipt"]
        self.assertIsNot(receipt, loaded["model"]["receipt"])
        self.assertEqual(receipt, loaded["model"]["receipt"])
        self.assertTrue(_receipt_digest_is_valid(receipt))
        self.assertEqual(receipt["source"], {"kind": "hub", "repository": REPOSITORY, "revision": REVISION})
        self.assertEqual(receipt["security"], SECURITY_CONTRACT)
        self.assertEqual(receipt["loader"]["modelAutoClass"], "AutoModelForCausalLM")

    def test_local_loader_omits_revision_and_records_config_digest(self):
        tokenizer = Mock()
        model = Mock()
        model.config = SimpleNamespace(model_type="local", architectures=[])
        runtime = _transformers_runtime(tokenizer=tokenizer, model=model)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text('{"model_type":"local"}', encoding="utf-8")
            with patch.dict("sys.modules", {"transformers": runtime}):
                loaded = LoadTextGenerationModel("local-loader").execute(
                    model_id={"source": "local", "value": directory},
                    revision="",
                    dtype="float16",
                    device=DEFAULT_DEVICE,
                )
        self.assertNotIn("revision", runtime.AutoTokenizer.from_pretrained.call_args.kwargs)
        self.assertNotIn("revision", runtime.AutoModelForCausalLM.from_pretrained.call_args.kwargs)
        self.assertEqual(loaded["receipt"]["source"]["kind"], "local")
        self.assertRegex(loaded["receipt"]["source"]["configSha256"], r"^sha256:[0-9a-f]{64}$")

    def test_text_generation_is_bounded_and_normalized(self):
        loaded, _runtime, tokenizer, model = _load_text_handle()
        result = GenerateText("generate-text-test").execute(
            model=loaded["model"],
            prompt="  Tell me something  ",
            max_new_tokens=4,
            min_new_tokens=1,
            do_sample=False,
            num_beams=2,
            repetition_penalty=1.2,
        )
        tokenizer.assert_called_once_with("Tell me something", return_tensors="pt", truncation=False)
        generation = model.generate.call_args.kwargs
        self.assertEqual(generation["max_new_tokens"], 4)
        self.assertEqual(generation["min_new_tokens"], 1)
        self.assertFalse(generation["do_sample"])
        self.assertEqual(generation["num_beams"], 2)
        self.assertNotIn("temperature", generation)
        tokenizer.decode.assert_called_once()
        self.assertTrue(torch.equal(tokenizer.decode.call_args.args[0], torch.tensor([3])))
        self.assertEqual(result["text"], "answer")
        self.assertEqual(result["result"]["inputTokens"], 2)
        self.assertEqual(result["result"]["generatedTokens"], 1)
        self.assertEqual(result["result"]["finishReason"], "stop")
        self.assertTrue(_receipt_digest_is_valid(result["result"]["modelReceipt"]))

    def test_sampling_controls_are_exact_and_invalid_controls_fail(self):
        controls = _generation_controls(
            {
                "max_new_tokens": 8,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
            }
        )
        self.assertEqual((controls["temperature"], controls["top_p"], controls["top_k"]), (0.7, 0.9, 20))
        for values, message in (
            ({"max_new_tokens": True}, "must be an integer"),
            ({"max_new_tokens": 0}, "between 1"),
            ({"max_new_tokens": 2, "min_new_tokens": 3}, "between 0 and 2"),
            ({"do_sample": "yes"}, "must be a boolean"),
            ({"num_beams": 9}, "between 1 and 8"),
            ({"repetition_penalty": float("nan")}, "must be finite"),
            ({"do_sample": True, "temperature": 0}, "between 0.01"),
            ({"do_sample": True, "top_p": 2}, "between 0.01"),
            ({"do_sample": True, "top_k": -1}, "between 0"),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                _generation_controls(values)

    def test_text_generation_rejects_tampering_and_token_contract_violations(self):
        loaded, _runtime, _tokenizer, _model = _load_text_handle()
        tampered = copy.copy(loaded["model"])
        tampered["receipt"] = copy.deepcopy(tampered["receipt"])
        tampered["receipt"]["security"]["trustRemoteCode"] = True
        with self.assertRaisesRegex(ValueError, "has been modified"):
            GenerateText("tampered-receipt").execute(model=tampered, prompt="hello")
        extra = {**loaded["model"], "unexpected": True}
        with self.assertRaisesRegex(ValueError, "intact"):
            GenerateText("extra-handle-field").execute(model=extra, prompt="hello")
        with self.assertRaisesRegex(ValueError, "wrong.*handle type"):
            GenerateImageVideoText("wrong-handle").execute(
                model=loaded["model"], prompt="hello", images=np.zeros((2, 2, 3), dtype=np.uint8)
            )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            GenerateText("empty-prompt").execute(model=loaded["model"], prompt="   ")
        with self.assertRaisesRegex(ValueError, "bounded text size"):
            GenerateText("large-prompt").execute(model=loaded["model"], prompt="x" * 65_537)

        oversized, _runtime, _tokenizer, _model = _load_text_handle(input_ids=torch.ones((1, 4), dtype=torch.long))
        with patch("modules.HuggingFaceTransformers.main.MAX_INPUT_TOKENS", 3):
            with self.assertRaisesRegex(RuntimeError, "input token limit"):
                GenerateText("large-input").execute(model=oversized["model"], prompt="hello")
        wrong_prefix, _runtime, _tokenizer, _model = _load_text_handle(output_ids=torch.tensor([[9, 9, 3]]))
        with self.assertRaisesRegex(RuntimeError, "preserve the input token prefix"):
            GenerateText("wrong-prefix").execute(model=wrong_prefix["model"], prompt="hello")
        too_many, _runtime, _tokenizer, _model = _load_text_handle(output_ids=torch.tensor([[1, 2, 3, 4]]))
        with self.assertRaisesRegex(RuntimeError, "exceeded max_new_tokens"):
            GenerateText("too-many-output-tokens").execute(model=too_many["model"], prompt="hello", max_new_tokens=1)

    def test_batch_validation_rejects_malformed_and_excessive_preprocessor_outputs(self):
        for batch, message in (
            (None, "non-mapping"),
            ({}, "empty or excessive"),
            ({"input_ids": torch.ones((2, 3), dtype=torch.long)}, "exactly one"),
            ({"input_ids": torch.ones((1, 0), dtype=torch.long)}, "exactly one"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(RuntimeError, message):
                _batch_to_device(batch, device=DEFAULT_DEVICE)
        batch = {
            "input_ids": torch.ones((1, 2), dtype=torch.long),
            "pixel_values": torch.zeros((1, 3, 3, 3)),
        }
        with patch("modules.HuggingFaceTransformers.main.MAX_PROCESSED_TENSOR_ELEMENTS", 10):
            with self.assertRaisesRegex(RuntimeError, "tensor size limit"):
                _batch_to_device(batch, device=DEFAULT_DEVICE)

    def test_multimodal_loader_and_generation_use_chat_placeholders_and_in_memory_media(self):
        loaded, runtime, processor, model = _load_multimodal_handle()
        processor_loader = runtime.AutoProcessor.from_pretrained.call_args
        self.assertEqual(processor_loader.args, (REPOSITORY,))
        self.assertEqual(
            processor_loader.kwargs,
            {"local_files_only": True, "trust_remote_code": False, "revision": REVISION},
        )
        model_loader = runtime.AutoModelForImageTextToText.from_pretrained.call_args
        self.assertTrue(model_loader.kwargs["use_safetensors"])
        self.assertTrue(model_loader.kwargs["weights_only"])
        self.assertEqual(loaded["receipt"]["loader"]["modelAutoClass"], "AutoModelForImageTextToText")

        image = np.zeros((2, 3, 3), dtype=np.uint8)
        frame = torch.ones((3, 2, 2), dtype=torch.float32)
        result = GenerateImageVideoText("generate-multimodal-test").execute(
            model=loaded["model"],
            prompt="  describe this  ",
            images=image,
            video=[frame],
            use_chat_template=True,
            max_new_tokens=4,
        )
        messages = processor.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual([item["type"] for item in messages[0]["content"]], ["image", "video", "text"])
        self.assertEqual(messages[0]["content"][-1]["text"], "describe this")
        self.assertEqual(
            processor.apply_chat_template.call_args.kwargs,
            {"tokenize": False, "add_generation_prompt": True},
        )
        processor_call = processor.call_args.kwargs
        self.assertEqual(processor_call["text"], "<image><video> describe")
        self.assertEqual(len(processor_call["images"]), 1)
        self.assertEqual(len(processor_call["videos"]), 1)
        self.assertEqual(len(processor_call["videos"][0]), 1)
        self.assertEqual(processor_call["images"][0].mode, "RGB")
        self.assertEqual(result["text"], "caption")
        self.assertEqual(result["result"]["imageCount"], 1)
        self.assertEqual(result["result"]["videoFrameCount"], 1)
        self.assertEqual(result["result"]["mediaPixels"], 10)
        self.assertEqual(result["result"]["generatedTokens"], 2)
        self.assertIn("max_new_tokens", model.generate.call_args.kwargs)

    def test_multimodal_raw_prompt_works_without_chat_template(self):
        loaded, _runtime, processor, _model = _load_multimodal_handle()
        del processor.apply_chat_template
        result = GenerateImageVideoText("raw-multimodal-prompt").execute(
            model=loaded["model"],
            prompt="caption",
            images=np.zeros((2, 2), dtype=np.uint8),
            use_chat_template=False,
        )
        self.assertEqual(processor.call_args.kwargs["text"], "caption")
        self.assertEqual(result["text"], "caption")

    def test_media_normalization_rejects_paths_counts_dimensions_and_bad_values(self):
        valid = np.zeros((2, 2, 3), dtype=np.uint8)
        images, video, pixels = _normalized_media(valid, [torch.zeros((3, 2, 2))])
        self.assertEqual((len(images), len(video), pixels), (1, 1, 8))
        for images_value, video_value, message in (
            (None, None, "at least one"),
            ("/tmp/image.png", None, "not a path"),
            (np.zeros((2, 2), dtype=np.bool_), None, "non-boolean"),
            (np.full((2, 2), np.nan, dtype=np.float32), None, "finite values"),
            (np.full((2, 2), 2.0, dtype=np.float32), None, r"in \[0, 1\]"),
            (np.full((2, 2), 256, dtype=np.int16), None, r"in \[0, 255\]"),
            (np.zeros((2, 2, 2), dtype=np.uint8), None, "channels"),
            ([[valid]], None, "in-memory PIL"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _normalized_media(images_value, video_value)
        with patch("modules.HuggingFaceTransformers.main.MAX_VIDEO_FRAMES", 1):
            with self.assertRaisesRegex(ValueError, "count limit"):
                _normalized_media(None, [valid, valid])
        with patch("modules.HuggingFaceTransformers.main.MAX_MEDIA_SIDE", 1):
            with self.assertRaisesRegex(ValueError, "dimensions"):
                _normalized_media(valid, None)
        with patch("modules.HuggingFaceTransformers.main.MAX_TOTAL_MEDIA_PIXELS", 3):
            with self.assertRaisesRegex(ValueError, "total pixel"):
                _normalized_media(valid, None)

    def test_multimodal_rejects_missing_template_and_bounded_render_or_output(self):
        loaded, _runtime, processor, _model = _load_multimodal_handle()
        processor.apply_chat_template = None
        with self.assertRaisesRegex(ValueError, "no chat template"):
            GenerateImageVideoText("missing-chat-template").execute(
                model=loaded["model"], prompt="caption", images=np.zeros((2, 2), dtype=np.uint8)
            )
        loaded, _runtime, processor, _model = _load_multimodal_handle(decoded="too long")
        processor.apply_chat_template.return_value = "long rendered prompt"
        with patch("modules.HuggingFaceTransformers.main.MAX_RENDERED_PROMPT_CHARACTERS", 5):
            with self.assertRaisesRegex(ValueError, "bounded text size"):
                GenerateImageVideoText("large-rendered-prompt").execute(
                    model=loaded["model"], prompt="caption", images=np.zeros((2, 2), dtype=np.uint8)
                )
        with patch("modules.HuggingFaceTransformers.main.MAX_OUTPUT_CHARACTERS", 3):
            with self.assertRaisesRegex(RuntimeError, "output text limit"):
                GenerateImageVideoText("large-decoded-output").execute(
                    model=loaded["model"],
                    prompt="caption",
                    images=np.zeros((2, 2), dtype=np.uint8),
                    use_chat_template=False,
                )


if __name__ == "__main__":
    unittest.main()
