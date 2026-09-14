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
from PIL import Image

from modiff.diffusers_profiles import resolve_execution_profiles_for_loader
from modiff.optional_runtimes import OPTIONAL_RUNTIME_PROFILES, TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
from modules.HuggingFaceTransformers.main import (
    ANY_TO_ANY_ADAPTER_CONTRACTS,
    GenerateAnyToAny,
    GenerateImageVideoText,
    GenerateText,
    LoadAnyToAnyModel,
    LoadImageTextToTextModel,
    LoadTextGenerationModel,
    SMOLLM2_135M_INSTRUCT_REPO,
    SMOLLM2_135M_INSTRUCT_REVISION,
    SECURITY_CONTRACT,
    _batch_to_device,
    _generation_controls,
    _janus_runtime_contract,
    _local_config_receipt,
    _model_revision,
    _model_selection,
    _normalized_media,
    _receipt_digest_is_valid,
)
from utils.torch_utils import DEFAULT_DEVICE


REVISION = "0123456789abcdef0123456789abcdef01234567"
REPOSITORY = "owner/model"


class JanusProcessor:
    def __init__(self, *, image_tokens=4, input_ids=None, decoded="janus answer"):
        self.num_image_tokens = image_tokens
        self.image_token = "<image>"
        self.input_ids = torch.tensor([[1, 2]]) if input_ids is None else input_ids
        self.calls = Mock()
        self.decode = Mock(return_value=decoded)
        self.postprocess = Mock(
            return_value={"pixel_values": np.ones((1, 8, 8, 3), dtype=np.float32)}
        )

    def __call__(self, **kwargs):
        self.calls(**kwargs)
        result = {"input_ids": self.input_ids, "attention_mask": torch.ones_like(self.input_ids)}
        if kwargs.get("images"):
            result["pixel_values"] = torch.zeros((1, 3, 8, 8))
        return result


class JanusForConditionalGeneration:
    input_modalities = ("image", "text")
    output_modalities = ("image", "text")

    def __init__(self, *, image_size=8, patch_size=4, image_tokens=4):
        vision_config = SimpleNamespace(
            image_size=image_size,
            patch_size=patch_size,
            num_image_tokens=image_tokens,
        )
        self.config = SimpleNamespace(
            model_type="janus",
            architectures=["JanusForConditionalGeneration"],
            vision_config=vision_config,
        )
        self.model = SimpleNamespace(
            vision_model=SimpleNamespace(config=SimpleNamespace(num_image_tokens=image_tokens))
        )
        self.to = Mock()
        self.eval = Mock()
        self.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4]]))
        self.decode_image_tokens = Mock(return_value=torch.ones((1, 8, 8, 3)))
        self.prepare_static_cache_calls = Mock()

    def _prepare_static_cache(
        self,
        cache_implementation,
        batch_size,
        max_cache_len,
        prefill_chunk_size,
        model_kwargs,
    ):
        self.prepare_static_cache_calls(
            cache_implementation=cache_implementation,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            prefill_chunk_size=prefill_chunk_size,
            model_kwargs=model_kwargs,
        )
        return SimpleNamespace(kind="static-cache")


class DecodedDeviceBatch:
    shape = (1, 8, 8, 3)

    def __init__(self):
        self.float = Mock(return_value=self)
        self.detach = Mock(return_value=self)
        self.cpu = Mock(return_value=torch.ones((1, 8, 8, 3)))


class AnyToAnyPipeline:
    instances = []

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = Mock()
        self.text_tokens = torch.tensor([1, 2, 3])
        self.image = Image.new("RGB", (8, 8), "white")
        self.__class__.instances.append(self)

    def __call__(self, invocation, **kwargs):
        self.calls(invocation, **kwargs)
        if kwargs.get("generation_mode") == "image":
            return [{"input_text": invocation["text"], "generated_image": self.image}]
        return [{"input_text": invocation["text"], "generated_token_ids": self.text_tokens}]


def _transformers_runtime(*, tokenizer=None, processor=None, model=None, config=None, pipeline_class=None):
    return SimpleNamespace(
        __version__="5.14.1",
        AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        AutoModelForImageTextToText=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        BitsAndBytesConfig=Mock(return_value=SimpleNamespace(kind="bnb-4bit")),
        AnyToAnyPipeline=pipeline_class or AnyToAnyPipeline,
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


def _load_janus_handle(*, image_size=8, patch_size=4, image_tokens=4, processor_tokens=None):
    AnyToAnyPipeline.instances.clear()
    processor = JanusProcessor(image_tokens=image_tokens if processor_tokens is None else processor_tokens)
    model = JanusForConditionalGeneration(
        image_size=image_size,
        patch_size=patch_size,
        image_tokens=image_tokens,
    )
    runtime = _transformers_runtime(
        processor=processor,
        model=model,
        config=SimpleNamespace(model_type="janus"),
        pipeline_class=AnyToAnyPipeline,
    )
    with patch.dict("sys.modules", {"transformers": runtime}):
        loaded = LoadAnyToAnyModel("load-any-to-any-test").execute(
            model_id=REPOSITORY,
            revision=REVISION,
            dtype="float32",
            device=DEFAULT_DEVICE,
        )
    return loaded, runtime, processor, model, AnyToAnyPipeline.instances[-1]


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
                "LoadAnyToAnyModel",
                "GenerateAnyToAny",
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

    def test_generic_loaders_publish_exact_fail_closed_execution_identities(self):
        cases = (
            (
                LoadTextGenerationModel,
                "LoadTextGenerationModel",
                "AutoModelForCausalLM",
                "smollm2-135m-instruct:direct",
            ),
            (
                LoadImageTextToTextModel,
                "LoadImageTextToTextModel",
                "AutoModelForImageTextToText",
                "smolvlm-256m-instruct:direct",
            ),
            (LoadAnyToAnyModel, "LoadAnyToAnyModel", "JanusForConditionalGeneration", "janus-pro-1b:direct"),
        )
        for loader, action, pipeline_class, profile_id in cases:
            with self.subTest(action=action):
                definition = loader.params["pipeline_class"]
                self.assertTrue(definition["hidden"])
                self.assertEqual(definition["default"], pipeline_class)
                profiles, reason = resolve_execution_profiles_for_loader(
                    "modules.HuggingFaceTransformers",
                    action,
                    {
                        "pipeline_class": pipeline_class,
                        "model_id": {"source": "hub", "value": REPOSITORY},
                    },
                )
                self.assertIsNone(reason)
                self.assertEqual([profile.id for profile in profiles], [profile_id])
                _profiles, missing_reason = resolve_execution_profiles_for_loader(
                    "modules.HuggingFaceTransformers",
                    action,
                    {"model_id": {"source": "hub", "value": REPOSITORY}},
                )
                self.assertEqual(missing_reason, "loader_identity_missing")

        for loader, expected in (
            (LoadTextGenerationModel, "AutoModelForCausalLM"),
            (LoadImageTextToTextModel, "AutoModelForImageTextToText"),
            (LoadAnyToAnyModel, "JanusForConditionalGeneration"),
        ):
            with self.subTest(loader=loader.__name__), self.assertRaisesRegex(ValueError, "requires"):
                loader("wrong-execution-identity").execute(pipeline_class=expected + "Wrong")

    def test_optional_runtime_qualifies_both_new_auto_model_symbols(self):
        transformers = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID].packages[0]
        expected = {
            "AutoConfig",
            "AutoModelForCausalLM",
            "AutoModelForImageTextToText",
            "AutoModelForMultimodalLM",
            "AnyToAnyPipeline",
        }
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

    def test_smollm2_bnb4_is_exact_nvidia_only_and_skips_post_load_move(self):
        tokenizer = Mock()
        model = Mock()
        model.config = SimpleNamespace(model_type="llama", architectures=["LlamaForCausalLM"])
        runtime = _transformers_runtime(tokenizer=tokenizer, model=model)
        with (
            patch.dict("sys.modules", {"transformers": runtime}),
            patch("modules.HuggingFaceTransformers.main._normalized_device", return_value="cuda"),
        ):
            loaded = LoadTextGenerationModel("smollm2-bnb4-load").execute(
                model_id={"source": "hub", "value": SMOLLM2_135M_INSTRUCT_REPO},
                revision=SMOLLM2_135M_INSTRUCT_REVISION,
                dtype="bfloat16",
                device="cuda",
                quantization_mode="bnb_4bit",
            )

        config_call = runtime.BitsAndBytesConfig.call_args.kwargs
        self.assertTrue(config_call["load_in_4bit"])
        self.assertEqual(config_call["bnb_4bit_quant_type"], "nf4")
        self.assertTrue(config_call["bnb_4bit_use_double_quant"])
        self.assertEqual(config_call["bnb_4bit_compute_dtype"], torch.bfloat16)
        model_call = runtime.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertIs(model_call["quantization_config"], runtime.BitsAndBytesConfig.return_value)
        self.assertEqual(model_call["device_map"], {"": "cuda"})
        model.to.assert_not_called()
        self.assertEqual(
            loaded["receipt"]["runtime"]["quantization"],
            {
                "mode": "bnb_4bit",
                "quantType": "nf4",
                "doubleQuant": True,
                "computeDtype": "bfloat16",
            },
        )

        with (
            patch.dict("sys.modules", {"transformers": runtime}),
            patch("modules.HuggingFaceTransformers.main._normalized_device", return_value="cuda"),
        ):
            with self.assertRaisesRegex(ValueError, "qualified only for pinned SmolLM2"):
                LoadTextGenerationModel("unreviewed-bnb4-load").execute(
                    model_id={"source": "hub", "value": REPOSITORY},
                    revision=REVISION,
                    dtype="bfloat16",
                    device="cuda",
                    quantization_mode="bnb_4bit",
                )

    def test_text_generation_is_bounded_and_normalized(self):
        loaded, _runtime, tokenizer, model = _load_text_handle()
        result = GenerateText("generate-text-test").execute(
            model=loaded["model"],
            prompt="  Tell me something  ",
            use_chat_template=False,
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

    def test_text_generation_applies_a_bounded_chat_template(self):
        loaded, _runtime, tokenizer, _model = _load_text_handle()
        tokenizer.apply_chat_template.return_value = "<chat> Tell me something"

        result = GenerateText("generate-chat-text-test").execute(
            model=loaded["model"],
            prompt="  Tell me something  ",
            use_chat_template=True,
            max_new_tokens=4,
        )

        self.assertEqual(
            tokenizer.apply_chat_template.call_args.args[0],
            [{"role": "user", "content": "Tell me something"}],
        )
        self.assertEqual(
            tokenizer.apply_chat_template.call_args.kwargs,
            {"tokenize": False, "add_generation_prompt": True},
        )
        tokenizer.assert_called_once_with("<chat> Tell me something", return_tensors="pt", truncation=False)
        self.assertEqual(result["text"], "answer")

    def test_text_generation_rejects_missing_or_excessive_chat_templates(self):
        loaded, _runtime, tokenizer, _model = _load_text_handle()
        tokenizer.apply_chat_template = None
        with self.assertRaisesRegex(ValueError, "no chat template"):
            GenerateText("missing-text-chat-template").execute(model=loaded["model"], prompt="hello")

        loaded, _runtime, tokenizer, _model = _load_text_handle()
        tokenizer.apply_chat_template.return_value = "rendered prompt"
        with (
            patch("modules.HuggingFaceTransformers.main.MAX_RENDERED_PROMPT_CHARACTERS", 5),
            self.assertRaisesRegex(ValueError, "bounded text size"),
        ):
            GenerateText("large-text-chat-template").execute(model=loaded["model"], prompt="hello")

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
            GenerateText("tampered-receipt").execute(model=tampered, prompt="hello", use_chat_template=False)
        extra = {**loaded["model"], "unexpected": True}
        with self.assertRaisesRegex(ValueError, "intact"):
            GenerateText("extra-handle-field").execute(model=extra, prompt="hello", use_chat_template=False)
        with self.assertRaisesRegex(ValueError, "wrong.*handle type"):
            GenerateImageVideoText("wrong-handle").execute(
                model=loaded["model"], prompt="hello", images=np.zeros((2, 2, 3), dtype=np.uint8)
            )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            GenerateText("empty-prompt").execute(model=loaded["model"], prompt="   ", use_chat_template=False)
        with self.assertRaisesRegex(ValueError, "bounded text size"):
            GenerateText("large-prompt").execute(model=loaded["model"], prompt="x" * 65_537, use_chat_template=False)

        oversized, _runtime, _tokenizer, _model = _load_text_handle(input_ids=torch.ones((1, 4), dtype=torch.long))
        with patch("modules.HuggingFaceTransformers.main.MAX_INPUT_TOKENS", 3):
            with self.assertRaisesRegex(RuntimeError, "input token limit"):
                GenerateText("large-input").execute(model=oversized["model"], prompt="hello", use_chat_template=False)
        wrong_prefix, _runtime, _tokenizer, _model = _load_text_handle(output_ids=torch.tensor([[9, 9, 3]]))
        with self.assertRaisesRegex(RuntimeError, "preserve the input token prefix"):
            GenerateText("wrong-prefix").execute(model=wrong_prefix["model"], prompt="hello", use_chat_template=False)
        too_many, _runtime, _tokenizer, _model = _load_text_handle(output_ids=torch.tensor([[1, 2, 3, 4]]))
        with self.assertRaisesRegex(RuntimeError, "exceeded max_new_tokens"):
            GenerateText("too-many-output-tokens").execute(
                model=too_many["model"],
                prompt="hello",
                use_chat_template=False,
                max_new_tokens=1,
            )

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

    def test_any_to_any_registry_keeps_qwen_omni_contract_only_for_safetensors_boundary(self):
        janus = ANY_TO_ANY_ADAPTER_CONTRACTS["janus"]
        qwen = ANY_TO_ANY_ADAPTER_CONTRACTS["qwen2_5_omni"]
        self.assertEqual((janus["status"], janus["adapterId"]), ("executable", "janus-v1"))
        self.assertEqual(qwen["status"], "contract-only")
        self.assertEqual(qwen["audioOutputSampleRate"], 24_000)
        self.assertIn("spk_dict.pt", qwen["blocker"])
        self.assertIn("safetensors-only", qwen["blocker"])

        runtime = _transformers_runtime(
            processor=Mock(),
            model=Mock(),
            config=SimpleNamespace(model_type="qwen2_5_omni"),
        )
        with (
            patch.dict("sys.modules", {"transformers": runtime}),
            self.assertRaisesRegex(ValueError, "contract-only.*spk_dict.pt"),
        ):
            LoadAnyToAnyModel("blocked-qwen-any-to-any").execute(
                model_id=REPOSITORY,
                revision=REVISION,
                dtype="float32",
                device=DEFAULT_DEVICE,
            )
        self.assertEqual(
            runtime.AutoConfig.from_pretrained.call_args.kwargs,
            {"local_files_only": True, "trust_remote_code": False, "revision": REVISION},
        )
        runtime.AutoProcessor.from_pretrained.assert_not_called()
        runtime.AutoModelForMultimodalLM.from_pretrained.assert_not_called()

    def test_any_to_any_janus_loader_enforces_adapter_geometry_and_exact_safe_flags(self):
        loaded, runtime, processor, model, pipeline = _load_janus_handle()
        self.assertEqual(
            runtime.AutoConfig.from_pretrained.call_args.kwargs,
            {"local_files_only": True, "trust_remote_code": False, "revision": REVISION},
        )
        self.assertEqual(
            runtime.AutoProcessor.from_pretrained.call_args.kwargs,
            {"local_files_only": True, "trust_remote_code": False, "revision": REVISION},
        )
        model_call = runtime.AutoModelForMultimodalLM.from_pretrained.call_args
        self.assertTrue(model_call.kwargs["local_files_only"])
        self.assertFalse(model_call.kwargs["trust_remote_code"])
        self.assertTrue(model_call.kwargs["use_safetensors"])
        self.assertTrue(model_call.kwargs["weights_only"])
        self.assertTrue(model_call.kwargs["low_cpu_mem_usage"])
        self.assertEqual(model_call.kwargs["revision"], REVISION)
        model.to.assert_called_once_with(DEFAULT_DEVICE)
        model.eval.assert_called_once_with()
        self.assertIs(pipeline.init_kwargs["model"], model)
        self.assertIs(pipeline.init_kwargs["processor"], processor)
        self.assertEqual(pipeline.init_kwargs["device"], DEFAULT_DEVICE)
        receipt = loaded["receipt"]
        self.assertTrue(_receipt_digest_is_valid(receipt))
        self.assertEqual(receipt["adapter"]["adapterId"], "janus-v1")
        self.assertEqual(receipt["adapter"]["imageGeneration"]["tokenCount"], 4)
        self.assertEqual(receipt["loader"]["modelAutoClass"], "AutoModelForMultimodalLM")
        self.assertEqual(receipt["loader"]["pipelineClass"], "AnyToAnyPipeline")
        self.assertEqual(receipt["security"], SECURITY_CONTRACT)

    def test_any_to_any_janus_runtime_geometry_fails_closed(self):
        processor = JanusProcessor(image_tokens=4)
        model = JanusForConditionalGeneration(image_tokens=5)
        with self.assertRaisesRegex(ValueError, "image-token geometry"):
            _janus_runtime_contract(model, processor)
        model = JanusForConditionalGeneration(image_size=9, patch_size=4, image_tokens=4)
        with self.assertRaisesRegex(ValueError, "image geometry"):
            _janus_runtime_contract(model, processor)
        model = JanusForConditionalGeneration()
        model.output_modalities = ("image", "text", "audio")
        with self.assertRaisesRegex(ValueError, "output modalities"):
            _janus_runtime_contract(model, processor)

    def test_any_to_any_janus_text_generation_is_token_bounded_and_normalized(self):
        loaded, _runtime, processor, _model, pipeline = _load_janus_handle()
        result = GenerateAnyToAny("generate-any-text").execute(
            model=loaded["model"],
            prompt="  describe  ",
            images=np.zeros((2, 2, 3), dtype=np.uint8),
            generation_mode="text",
            max_new_tokens=4,
            num_beams=8,
            do_sample=False,
        )
        preview = processor.calls.call_args.kwargs
        self.assertEqual(preview["text"], ["<image>describe"])
        self.assertEqual(preview["generation_mode"], "text")
        self.assertEqual(preview["images"][0].mode, "RGB")
        invocation = pipeline.calls.call_args.args[0]
        call = pipeline.calls.call_args.kwargs
        self.assertEqual(invocation["text"], ["<image>describe"])
        self.assertTrue(call["return_tensors"])
        self.assertEqual(call["processor_kwargs"], {"generation_mode": "text"})
        self.assertEqual(call["generate_kwargs"]["max_new_tokens"], 4)
        self.assertEqual(call["generate_kwargs"]["num_beams"], 1)
        self.assertTrue(torch.equal(processor.decode.call_args.args[0], torch.tensor([3])))
        self.assertEqual(result["text"], "janus answer")
        self.assertIsNone(result["image"])
        self.assertNotIn("audio", result)
        self.assertEqual(result["result"]["inputTokens"], 2)
        self.assertEqual(result["result"]["generatedTokens"], 1)
        self.assertEqual(result["result"]["inputImageCount"], 1)
        self.assertEqual(result["result"]["inputMediaPixels"], 4)

    def test_any_to_any_janus_image_generation_is_fixed_token_and_geometry_bounded(self):
        loaded, _runtime, processor, _model, pipeline = _load_janus_handle()
        decoded_device_batch = DecodedDeviceBatch()
        _model.decode_image_tokens.return_value = decoded_device_batch
        result = GenerateAnyToAny("generate-any-image").execute(
            model=loaded["model"],
            prompt="an owl",
            generation_mode="image",
            max_new_tokens=2,
            do_sample=True,
            temperature=0.7,
        )
        self.assertEqual(processor.calls.call_args.kwargs["generation_mode"], "image")
        pipeline.calls.assert_not_called()
        model = loaded["model"]["model"]
        call = model.generate.call_args.kwargs
        self.assertEqual(call["generation_mode"], "image")
        self.assertEqual(call["num_beams"], 1)
        self.assertEqual(call["temperature"], 0.7)
        self.assertEqual(call["past_key_values"].kind, "static-cache")
        self.assertTrue(torch.equal(call["input_ids"].detach().cpu(), torch.tensor([[1, 2]])))
        model.decode_image_tokens.assert_called_once()
        self.assertTrue(
            torch.equal(
                model.decode_image_tokens.call_args.args[0].detach().cpu(),
                torch.tensor([[1, 2, 3, 4]]),
            )
        )
        processor.postprocess.assert_called_once()
        self.assertEqual(processor.postprocess.call_args.kwargs, {"return_tensors": "np"})
        decoded_device_batch.float.assert_called_once_with()
        decoded_device_batch.detach.assert_called_once_with()
        decoded_device_batch.cpu.assert_called_once_with()
        model.prepare_static_cache_calls.assert_called_once_with(
            cache_implementation="static",
            batch_size=2,
            max_cache_len=6,
            prefill_chunk_size=None,
            model_kwargs={},
        )
        self.assertIsNone(result["text"])
        self.assertEqual(result["image"].size, (8, 8))
        self.assertEqual(result["result"]["generatedTokens"], 4)
        self.assertEqual(result["result"]["finishReason"], "fixed-image-token-contract")
        self.assertEqual(result["result"]["image"], {"width": 8, "height": 8, "pixels": 64})

    def test_any_to_any_janus_action_rejects_unsupported_inputs_outputs_and_tampering(self):
        loaded, _runtime, _processor, _model, pipeline = _load_janus_handle()
        for values, message in (
            ({"generation_mode": "audio"}, "exactly text or image"),
            ({"generation_mode": "text", "video": [np.zeros((2, 2, 3), dtype=np.uint8)]}, "only text"),
            ({"generation_mode": "text", "audio_input": np.zeros(10)}, "only text"),
            (
                {"generation_mode": "image", "images": np.zeros((2, 2, 3), dtype=np.uint8)},
                "without source images",
            ),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                GenerateAnyToAny("invalid-any-input").execute(model=loaded["model"], prompt="prompt", **values)

        tampered = copy.copy(loaded["model"])
        tampered["receipt"] = copy.deepcopy(tampered["receipt"])
        tampered["receipt"]["adapter"]["imageGeneration"]["tokenCount"] = 1
        with self.assertRaisesRegex(ValueError, "has been modified"):
            GenerateAnyToAny("tampered-any-handle").execute(model=tampered, prompt="prompt")

        pipeline.text_tokens = torch.tensor([9, 9, 3])
        with self.assertRaisesRegex(RuntimeError, "preserve the input token prefix"):
            GenerateAnyToAny("invalid-any-prefix").execute(model=loaded["model"], prompt="prompt")
        pipeline.text_tokens = torch.tensor([1, 2, 3, 4])
        with self.assertRaisesRegex(RuntimeError, "exceeded max_new_tokens"):
            GenerateAnyToAny("oversized-any-text").execute(model=loaded["model"], prompt="prompt", max_new_tokens=1)
        loaded["model"]["model"].decode_image_tokens.return_value = torch.ones((1, 7, 8, 3))
        _processor.postprocess.return_value = {
            "pixel_values": np.ones((1, 7, 8, 3), dtype=np.float32)
        }
        with self.assertRaisesRegex(RuntimeError, "reviewed output geometry"):
            GenerateAnyToAny("invalid-any-image").execute(
                model=loaded["model"], prompt="prompt", generation_mode="image"
            )


if __name__ == "__main__":
    unittest.main()
