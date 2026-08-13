import hashlib
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import modules as module_registry
from modiff.model_artifact_catalog import catalog_revision
from modiff.server import WebServer
from modules.DiffusersImage import (
    ControlGenerate,
    Edit,
    Generate,
    Inpaint,
    LoadAdapter,
    LoadPipeline,
    MODULE_MAP,
    PredictMap,
    UnconditionalGenerate,
)
from modules.DiffusersImage.main import (
    AURAFLOW_V03_REPO,
    CHROMA1_HD_REPO,
    COGVIEW3_PLUS_REPO,
    COGVIEW4_6B_REPO,
    CONSISTENCY_IMAGENET64_REPO,
    DDPM_CIFAR10_REPO,
    DREAMLITE_BASE_REPO,
    DREAMLITE_MOBILE_REPO,
    ERNIE_IMAGE_TURBO_REPO,
    GLM_IMAGE_REPO,
    FLUX2_KLEIN_REPO,
    FLUX_CANNY_REPO,
    FLUX_DEPTH_REPO,
    FLUX_DEV_REPO,
    FLUX_FILL_REPO,
    FLUX_KONTEXT_REPO,
    FLUX_KREA_REPO,
    FLUX_SCHNELL_REPO,
    HUNYUAN_DIT_CONTROLNET_CANNY_REPO,
    HUNYUAN_DIT_DISTILLED_REPO,
    IMAGE_MODE_FIELD_CONTRACTS,
    IMAGE_PIPELINE_CLASSES,
    QWEN_IMAGE_2512_REPO,
    QWEN_IMAGE_EDIT_PLUS_REPO,
    QWEN_IMAGE_EDIT_REPO,
    LCM_DREAMSHAPER_REPO,
    MARIGOLD_DEPTH_LCM_REPO,
    PIXART_SIGMA_REPO,
    SD15_BASE_REPO,
    SD15_CONTROLNET_CANNY_REPO,
    SANA_REPO,
    SANA_SPRINT_REPO,
    SDXL_BASE_REPO,
    SDXL_CONTROLNET_CANNY_REPO,
    SDXL_INSTRUCT_PIX2PIX_REPO,
    SDXL_T2I_ADAPTER_CANNY_REPO,
    SDXL_TURBO_REPO,
    Z_IMAGE_REPO,
    FluxReduxPipelineBundle,
    ImageModeFieldContract,
    _tag_image_pipeline,
    add_progress_callback,
    image_pipeline_contract,
    output_image_dimensions,
    pipeline_class_from_name,
    quant_config_for,
    resolve_image_model_selection,
    resolve_image_pipeline_revision,
    validate_image_action,
)
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS
from utils.huggingface import resolve_managed_hf_cache_file

CUSTOM_IMAGE_REVISION = "a" * 40
HUB_ADAPTER_REVISION = "b" * 40


def tag_test_image_pipeline(pipeline, pipeline_class, mode, *, repo=None, revision=None):
    adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_class]
    pipeline.__class__.__name__ = adapter.allowed_runtime_classes[0]
    repository = repo or adapter.default_repo
    resolved_revision = revision or catalog_revision(repository) or CUSTOM_IMAGE_REVISION
    conditioning_repo = adapter.default_conditioning_repo
    _tag_image_pipeline(
        pipeline,
        adapter,
        mode,
        repository,
        "hub",
        resolved_revision,
        conditioning_repo=conditioning_repo,
        conditioning_revision=(catalog_revision(conditioning_repo) if conditioning_repo else None),
    )
    return pipeline


class DiffusersImageRegistryTests(unittest.TestCase):
    def test_output_dimensions_support_pil_numpy_and_torch_layouts(self):
        self.assertEqual(output_image_dimensions([Image.new("RGB", (31, 19))], "pil"), (31, 19))
        self.assertEqual(output_image_dimensions(np.zeros((2, 19, 31, 3)), "np"), (31, 19))
        fake_tensor = SimpleNamespace(shape=(2, 3, 19, 31), size=lambda: 2 * 3 * 19 * 31)
        self.assertEqual(output_image_dimensions(fake_tensor, "pt"), (31, 19))

    def test_unconditional_adapters_are_exact_generic_pipeline_pairs(self):
        expected = {
            "DDPMPipeline": (DDPM_CIFAR10_REPO, ()),
            "DDIMPipeline": (DDPM_CIFAR10_REPO, ("eta",)),
            "ConsistencyModelPipeline": (CONSISTENCY_IMAGENET64_REPO, ("class_label",)),
        }
        for pipeline_class, (repo, optional_fields) in expected.items():
            with self.subTest(pipeline=pipeline_class):
                adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_class]
                self.assertEqual(adapter.default_repo, repo)
                self.assertEqual(adapter.mode_options, ("unconditional_image",))
                self.assertEqual(adapter.unconditional_optional_fields, optional_fields)
                contract = image_pipeline_contract(adapter, "unconditional_image")
                self.assertEqual(contract["actions"], {"UnconditionalGenerate": ["unconditional_image"]})

    def test_unconditional_generate_normalizes_outputs_and_only_passes_supported_arguments(self):
        cases = (
            ("DDPMPipeline", {}, {"batch_size", "generator", "num_inference_steps", "output_type", "return_dict"}),
            (
                "DDIMPipeline",
                {"eta": 0.25},
                {"batch_size", "generator", "num_inference_steps", "eta", "output_type", "return_dict"},
            ),
            (
                "ConsistencyModelPipeline",
                {"class_label": 145},
                {
                    "batch_size",
                    "class_labels",
                    "generator",
                    "num_inference_steps",
                    "output_type",
                    "return_dict",
                    "callback",
                    "callback_steps",
                },
            ),
        )
        for pipeline_class, extra, expected_keys in cases:
            with self.subTest(pipeline=pipeline_class):
                received = {}

                def call(self, **kwargs):
                    received.update(kwargs)
                    if callback := kwargs.get("callback"):
                        callback(0, 0, object())
                    return SimpleNamespace(images=[Image.new("RGB", (32, 24), "white")])

                parameters = {
                    "batch_size": inspect.Parameter("batch_size", inspect.Parameter.KEYWORD_ONLY),
                    "generator": inspect.Parameter("generator", inspect.Parameter.KEYWORD_ONLY),
                    "num_inference_steps": inspect.Parameter("num_inference_steps", inspect.Parameter.KEYWORD_ONLY),
                    "output_type": inspect.Parameter("output_type", inspect.Parameter.KEYWORD_ONLY),
                    "return_dict": inspect.Parameter("return_dict", inspect.Parameter.KEYWORD_ONLY),
                }
                if pipeline_class == "DDIMPipeline":
                    parameters["eta"] = inspect.Parameter("eta", inspect.Parameter.KEYWORD_ONLY)
                if pipeline_class == "ConsistencyModelPipeline":
                    parameters.update(
                        {
                            "class_labels": inspect.Parameter("class_labels", inspect.Parameter.KEYWORD_ONLY),
                            "callback": inspect.Parameter("callback", inspect.Parameter.KEYWORD_ONLY),
                            "callback_steps": inspect.Parameter("callback_steps", inspect.Parameter.KEYWORD_ONLY),
                        }
                    )
                call.__signature__ = inspect.Signature(
                    [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD), *parameters.values()]
                )
                fake_type = type(pipeline_class, (), {"_execution_device": "cpu", "__call__": call})
                pipeline = tag_test_image_pipeline(fake_type(), pipeline_class, "unconditional_image")
                node = UnconditionalGenerate()
                node.progress = Mock()

                result = node.execute(
                    pipeline=pipeline,
                    batch_size=1,
                    seed=7,
                    num_inference_steps=2,
                    output_type="pil",
                    **extra,
                )

                self.assertEqual(set(received), expected_keys)
                self.assertEqual((result["width_out"], result["height_out"]), (32, 24))
                self.assertEqual(len(result["images"]), 1)

    def test_marigold_depth_uses_the_generic_versioned_prediction_map_contract(self):
        adapter = IMAGE_PIPELINE_ADAPTERS["MarigoldDepthPipeline"]
        self.assertEqual(adapter.default_repo, MARIGOLD_DEPTH_LCM_REPO)
        self.assertEqual(adapter.mode_options, ("depth_estimation",))
        contract = image_pipeline_contract(adapter, "depth_estimation")
        self.assertEqual(contract["actions"], {"PredictMap": ["depth_estimation"]})
        self.assertEqual(
            contract["predictionMap"],
            {
                "schemaVersion": 1,
                "kinds": ["depth"],
                "semantics": "relative_depth",
                "layout": "NHWC",
                "dtype": "float32",
                "valueRange": [0.0, 1.0],
                "nearValue": 0.0,
                "farValue": 1.0,
            },
        )
        parameters = set(inspect.signature(pipeline_class_from_name("MarigoldDepthPipeline").__call__).parameters)
        self.assertTrue(
            {
                "image",
                "num_inference_steps",
                "ensemble_size",
                "processing_resolution",
                "match_input_resolution",
                "generator",
                "output_type",
                "output_uncertainty",
                "output_latent",
                "return_dict",
            }.issubset(parameters)
        )

    def test_predict_map_normalizes_depth_and_passes_only_the_reviewed_upstream_request(self):
        received = {}

        def call(
            self,
            *,
            image,
            num_inference_steps,
            ensemble_size,
            processing_resolution,
            match_input_resolution,
            generator,
            output_type,
            output_uncertainty,
            output_latent,
            return_dict,
        ):
            received.update(locals())
            return SimpleNamespace(
                prediction=np.array(
                    [[[[0.0], [0.25]], [[0.75], [1.0]]]],
                    dtype=np.float64,
                )
            )

        fake_type = type("MarigoldDepthPipeline", (), {"_execution_device": "cpu", "__call__": call})
        pipeline = tag_test_image_pipeline(fake_type(), "MarigoldDepthPipeline", "depth_estimation")
        source = Image.new("RGB", (16, 12), "white")
        result = PredictMap("depth-probe").execute(
            pipeline=pipeline,
            image=[source],
            prediction_kind="depth",
            seed=7,
            num_inference_steps=1,
            processing_resolution=768,
            match_input_resolution=True,
        )

        self.assertEqual(
            set(received) - {"self", "received"},
            {
                "image",
                "num_inference_steps",
                "ensemble_size",
                "processing_resolution",
                "match_input_resolution",
                "generator",
                "output_type",
                "output_uncertainty",
                "output_latent",
                "return_dict",
            },
        )
        self.assertIs(received["image"], source)
        self.assertEqual(received["ensemble_size"], 1)
        self.assertEqual(received["output_type"], "np")
        self.assertFalse(received["output_uncertainty"])
        prediction_map = result["prediction_map"]
        self.assertEqual(
            {key: value for key, value in prediction_map.items() if key != "prediction"},
            {
                "schemaVersion": 1,
                "kind": "depth",
                "semantics": "relative_depth",
                "layout": "NHWC",
                "dtype": "float32",
                "shape": [1, 2, 2, 1],
                "valueRange": [0.0, 1.0],
                "nearValue": 0.0,
                "farValue": 1.0,
                "width": 2,
                "height": 2,
            },
        )
        self.assertEqual(prediction_map["prediction"].dtype, np.float32)
        self.assertEqual(result["preview_images"][0].size, (2, 2))
        self.assertEqual(result["preview_images"][0].mode, "RGB")
        self.assertEqual((result["width_out"], result["height_out"]), (2, 2))

    def test_predict_map_rejects_malformed_inputs_and_outputs(self):
        class MarigoldDepthPipeline:
            _execution_device = "cpu"

            def __init__(self, prediction):
                self.prediction = prediction

            def __call__(self, **_kwargs):
                return SimpleNamespace(prediction=self.prediction)

        source = Image.new("RGB", (16, 16), "white")
        for label, prediction, message in (
            ("batch", np.zeros((2, 4, 4, 1)), "exactly one single-channel"),
            ("channels", np.zeros((1, 4, 4, 3)), "exactly one single-channel"),
            ("nonfinite", np.full((1, 4, 4, 1), np.nan), "only finite"),
            ("range", np.full((1, 4, 4, 1), 1.1), "normalized"),
        ):
            with self.subTest(label=label):
                pipeline = tag_test_image_pipeline(
                    MarigoldDepthPipeline(prediction),
                    "MarigoldDepthPipeline",
                    "depth_estimation",
                )
                with self.assertRaisesRegex(ValueError, message):
                    PredictMap(f"invalid-{label}").execute(pipeline=pipeline, image=source)

        valid_pipeline = tag_test_image_pipeline(
            MarigoldDepthPipeline(np.zeros((1, 4, 4, 1))),
            "MarigoldDepthPipeline",
            "depth_estimation",
        )
        for values, message in (
            ({"image": [source, source]}, "at most 1"),
            ({"image": source, "prediction_kind": "normal"}, "exactly depth"),
            ({"image": source, "match_input_resolution": 1}, "must be a boolean"),
            ({"image": source, "processing_resolution": 65}, "increments of 8"),
        ):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, message):
                PredictMap("invalid-input").execute(pipeline=valid_pipeline, **values)

    def test_torchao_int8_weight_only_passes_an_aobase_config_instance(self):
        int8_config = object()
        torchao_config = object()
        with (
            patch.dict(
                "sys.modules",
                {
                    "torchao": SimpleNamespace(),
                    "torchao.quantization": SimpleNamespace(
                        Int8WeightOnlyConfig=Mock(return_value=int8_config),
                    ),
                },
            ),
            patch("diffusers.TorchAoConfig", return_value=torchao_config) as config,
        ):
            result = quant_config_for("torchao_int8_weight_only", None, ["proj_out"])

        self.assertIs(result, torchao_config)
        self.assertIs(config.call_args.kwargs["quant_type"], int8_config)
        self.assertEqual(config.call_args.kwargs["modules_to_not_convert"], ["proj_out"])

    def test_quanto_float8_uses_the_current_weights_dtype_argument(self):
        received = {}

        class FakeQuantoConfig:
            def __init__(self, **kwargs):
                received.update(kwargs)

        with patch("diffusers.QuantoConfig", FakeQuantoConfig):
            quant_config_for("quanto_float8", None)

        self.assertEqual(received, {"weights_dtype": "float8", "modules_to_not_convert": None})

    def test_quanto_int8_uses_weight_only_int8(self):
        received = {}

        class FakeQuantoConfig:
            def __init__(self, **kwargs):
                received.update(kwargs)

        with patch("diffusers.QuantoConfig", FakeQuantoConfig):
            quant_config_for("quanto_int8", None, ["norm"])

        self.assertEqual(received, {"weights_dtype": "int8", "modules_to_not_convert": ["norm"]})

    def test_torchao_float8_passes_an_aobase_config_instance(self):
        import sys
        from types import ModuleType

        received = {}

        class FakeFloat8WeightOnlyConfig:
            pass

        class FakeTorchAoConfig:
            def __init__(self, **kwargs):
                received.update(kwargs)

        fake_quantization = ModuleType("torchao.quantization")
        fake_quantization.Float8WeightOnlyConfig = FakeFloat8WeightOnlyConfig
        fake_torchao = ModuleType("torchao")
        fake_torchao.quantization = fake_quantization

        with (
            patch("diffusers.TorchAoConfig", FakeTorchAoConfig),
            patch.dict(
                sys.modules,
                {"torchao": fake_torchao, "torchao.quantization": fake_quantization},
            ),
        ):
            quant_config_for("torchao_float8", None)

        self.assertIsInstance(received["quant_type"], FakeFloat8WeightOnlyConfig)

    def test_torchao_float8_explains_when_optional_dependency_is_missing(self):
        import builtins

        real_import = builtins.__import__

        def import_without_torchao(name, *args, **kwargs):
            if name == "torchao.quantization":
                raise ImportError("torchao unavailable")
            return real_import(name, *args, **kwargs)

        with (
            patch("diffusers.TorchAoConfig", object),
            patch("builtins.__import__", side_effect=import_without_torchao),
        ):
            with self.assertRaisesRegex(RuntimeError, "optional torchao quantization package"):
                quant_config_for("torchao_float8", None)

    def test_inherited_nodes_are_registered_with_their_live_contracts(self):
        expected_inputs = {
            "Edit": {"pipeline", "image"},
            "Inpaint": {"pipeline", "image", "mask_image"},
            "ControlGenerate": {"pipeline", "control_image"},
        }

        for name, required_inputs in expected_inputs.items():
            with self.subTest(node=name):
                entry = MODULE_MAP[name]
                self.assertEqual(entry["type"], "custom")
                self.assertEqual(entry["category"], "Diffusers Image")
                self.assertTrue(entry["resizable"])
                self.assertTrue(required_inputs.issubset(entry["params"]))
                self.assertEqual(entry["params"]["images"]["display"], "output")

    def test_graph_contract_marks_mode_independent_image_inputs_as_required(self):
        self.assertTrue(Edit.params["pipeline"]["required"])
        self.assertTrue(Edit.params["image"]["required"])
        self.assertTrue(Inpaint.params["mask_image"]["required"])
        self.assertTrue(ControlGenerate.params["control_image"]["required"])
        self.assertTrue(LoadAdapter.params["pipeline"]["required"])

    def test_registered_classes_can_be_constructed(self):
        for node_class in (Edit, Inpaint, ControlGenerate):
            with self.subTest(node=node_class.__name__):
                node = node_class("registry-probe")
                self.assertEqual(node.node_id, "registry-probe")
                self.assertTrue(node.resizable)

    def test_loader_rejects_missing_null_malformed_and_noncanonical_class_or_mode_before_nodebase(self):
        invalid_values = (
            ("class-absent", {"mode": "text_to_image"}),
            ("class-null", {"pipeline_class": None, "mode": "text_to_image"}),
            ("class-false", {"pipeline_class": False, "mode": "text_to_image"}),
            ("class-zero", {"pipeline_class": 0, "mode": "text_to_image"}),
            ("class-object", {"pipeline_class": {}, "mode": "text_to_image"}),
            ("class-array", {"pipeline_class": [], "mode": "text_to_image"}),
            ("class-blank", {"pipeline_class": "", "mode": "text_to_image"}),
            ("class-spaced", {"pipeline_class": " FluxPipeline ", "mode": "text_to_image"}),
            ("mode-absent", {"pipeline_class": "FluxPipeline"}),
            ("mode-null", {"pipeline_class": "FluxPipeline", "mode": None}),
            ("mode-false", {"pipeline_class": "FluxPipeline", "mode": False}),
            ("mode-zero", {"pipeline_class": "FluxPipeline", "mode": 0}),
            ("mode-object", {"pipeline_class": "FluxPipeline", "mode": {}}),
            ("mode-array", {"pipeline_class": "FluxPipeline", "mode": []}),
            ("mode-blank", {"pipeline_class": "FluxPipeline", "mode": ""}),
            ("mode-spaced", {"pipeline_class": "FluxPipeline", "mode": " text_to_image "}),
        )
        for label, values in invalid_values:
            with self.subTest(case=label):
                node = LoadPipeline(f"strict-{label}")
                node.execute = Mock()
                with self.assertRaises(ValueError):
                    node(
                        model_id={"source": "hub", "value": FLUX_SCHNELL_REPO},
                        **values,
                    )
                node.execute.assert_not_called()
                self.assertEqual(node.params, {})

    def test_loader_canonicalizes_complete_identity_before_real_nodebase_cache(self):
        class FluxPipeline:
            pass

        pipeline = FluxPipeline()
        node = LoadPipeline("canonical-image-loader")
        node.execute = Mock(return_value={"pipeline": pipeline, "resolved_artifact": FLUX_SCHNELL_REPO})
        with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
            result = node(
                model_id={"source": "HUB", "value": " BLACK-FOREST-LABS/FLUX.1-SCHNELL "},
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                revision=None,
            )

        expected_revision = catalog_revision(FLUX_SCHNELL_REPO)
        self.assertEqual(
            node.params,
            {
                "model_id": {"source": "hub", "value": FLUX_SCHNELL_REPO},
                "pipeline_class": "FluxPipeline",
                "mode": "text_to_image",
                "revision": expected_revision,
                "conditioning_kind": "none",
                "conditioning_model_id": {"source": "hub", "value": ""},
                "conditioning_revision": "",
            },
        )
        self.assertIs(result["pipeline"], pipeline)
        self.assertEqual(pipeline._modiff_image_pipeline_class, "FluxPipeline")
        self.assertEqual(pipeline._modiff_image_mode, "text_to_image")
        self.assertEqual(pipeline._modiff_image_repo, FLUX_SCHNELL_REPO)
        self.assertEqual(pipeline._modiff_image_source, "hub")
        self.assertEqual(pipeline._modiff_image_revision, expected_revision)

    def test_model_selection_source_value_and_catalog_spelling_are_canonical_or_rejected(self):
        adapter = IMAGE_PIPELINE_ADAPTERS["FluxPipeline"]
        accepted = (
            (None, {"source": "hub", "value": FLUX_SCHNELL_REPO}),
            ("", {"source": "hub", "value": FLUX_SCHNELL_REPO}),
            (
                {"source": "HUB", "value": " BLACK-FOREST-LABS/FLUX.1-SCHNELL "},
                {"source": "hub", "value": FLUX_SCHNELL_REPO},
            ),
            ({"source": "Local", "value": " models/custom "}, {"source": "local", "value": "models/custom"}),
            (" org/custom ", {"source": "hub", "value": "org/custom"}),
        )
        for value, expected in accepted:
            with self.subTest(value=value):
                self.assertEqual(resolve_image_model_selection(adapter, value), expected)

        rejected = (
            {"source": "local", "value": ""},
            {"source": " hub ", "value": FLUX_SCHNELL_REPO},
            {"source": "remote", "value": FLUX_SCHNELL_REPO},
            {"value": FLUX_SCHNELL_REPO},
            {"source": None, "value": FLUX_SCHNELL_REPO},
            {"source": 7, "value": FLUX_SCHNELL_REPO},
            {"source": "hub", "value": [FLUX_SCHNELL_REPO]},
            [],
        )
        for value in rejected:
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_image_model_selection(adapter, value)

        with self.assertRaisesRegex(ValueError, "existing local filesystem target"):
            resolve_image_model_selection(
                adapter,
                {"source": "hub", "value": "modules/DiffusersImage"},
            )

    def test_loader_revision_shape_local_boundary_and_custom_commit_fail_closed_before_execute(self):
        managed = {"source": "hub", "value": FLUX_SCHNELL_REPO}
        expected_pin = catalog_revision(FLUX_SCHNELL_REPO)
        for value in (None, "", expected_pin):
            with self.subTest(valid_revision=value):
                self.assertEqual(resolve_image_pipeline_revision(managed, value), expected_pin)

        for value in (" ", False, 0, {}, [], "main", "A" * 40):
            with self.subTest(invalid_revision=value), self.assertRaises(ValueError):
                resolve_image_pipeline_revision(managed, value)

        custom = {"source": "hub", "value": "org/custom-image"}
        self.assertEqual(resolve_image_pipeline_revision(custom, CUSTOM_IMAGE_REVISION), CUSTOM_IMAGE_REVISION)
        for value in (None, "", "main", "A" * 40, "a" * 39):
            with self.subTest(custom_revision=value), self.assertRaises(ValueError):
                resolve_image_pipeline_revision(custom, value)

        node = LoadPipeline("local-image-contract-only")
        node.execute = Mock()
        with self.assertRaisesRegex(ValueError, "reviewed local pipeline-directory index"):
            node(
                model_id={"source": "local", "value": "models/local-pipeline"},
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                revision="",
            )
        node.execute.assert_not_called()

    def test_class_only_changes_replace_the_inherited_managed_repository(self):
        cases = {
            "ZImagePipeline": ("text_to_image", Z_IMAGE_REPO),
            "Flux2KleinPipeline": ("text_to_image", FLUX2_KLEIN_REPO),
            "FluxFillPipeline": ("inpaint", FLUX_FILL_REPO),
            "FluxControlPipeline": ("control_image", FLUX_DEPTH_REPO),
            "FluxKontextPipeline": ("edit_image", FLUX_KONTEXT_REPO),
        }
        loaded = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **_kwargs):
                loaded.append(repo)
                return cls()

        node = LoadPipeline("class-only-default-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            for pipeline_class, (mode, expected_repo) in cases.items():
                with self.subTest(pipeline_class=pipeline_class):
                    result = node.execute(
                        model_id={"source": "hub", "value": FLUX_SCHNELL_REPO},
                        pipeline_class=pipeline_class,
                        mode=mode,
                        auto_offload=False,
                        offload_mode="none",
                    )
                    self.assertEqual(result["resolved_artifact"], expected_repo)
                    self.assertEqual(result["pipeline"]._modiff_image_mode, mode)

        self.assertEqual(loaded, [expected for _mode, expected in cases.values()])

    def test_model_resolution_preserves_explicit_and_shared_compatible_repositories(self):
        flux2 = IMAGE_PIPELINE_ADAPTERS["Flux2KleinPipeline"]
        custom_hub = {"source": "hub", "value": "example/custom-flux2-compatible"}
        local = {"source": "local", "value": FLUX_SCHNELL_REPO}
        self.assertEqual(resolve_image_model_selection(flux2, custom_hub), custom_hub)
        self.assertEqual(resolve_image_model_selection(flux2, local), local)

        flux = IMAGE_PIPELINE_ADAPTERS["FluxPipeline"]
        for repo in (FLUX_SCHNELL_REPO, FLUX_DEV_REPO, FLUX_KREA_REPO):
            with self.subTest(repo=repo):
                selection = {"source": "hub", "value": repo}
                self.assertEqual(resolve_image_model_selection(flux, selection), selection)

        control = IMAGE_PIPELINE_ADAPTERS["FluxControlPipeline"]
        canny = {"source": "hub", "value": FLUX_CANNY_REPO}
        self.assertEqual(resolve_image_model_selection(control, canny), canny)

    def test_every_image_adapter_owns_a_default_and_all_reviewed_compatible_repositories(self):
        all_managed_repos = {repo for adapter in IMAGE_PIPELINE_ADAPTERS.values() for repo in adapter.managed_repos}
        for pipeline_class, adapter in IMAGE_PIPELINE_ADAPTERS.items():
            with self.subTest(pipeline_class=pipeline_class):
                self.assertTrue(adapter.default_repo)
                contract = image_pipeline_contract(adapter, adapter.mode_options[0])
                implemented_modes = {mode for action_modes in contract["actions"].values() for mode in action_modes}
                self.assertEqual(implemented_modes, adapter.modes)
                self.assertEqual(
                    resolve_image_model_selection(adapter, None),
                    {"source": "hub", "value": adapter.default_repo},
                )
                for repo in adapter.managed_repos:
                    selection = {"source": "hub", "value": repo}
                    self.assertEqual(resolve_image_model_selection(adapter, selection), selection)
                    self.assertEqual(
                        resolve_image_pipeline_revision(selection, ""),
                        catalog_revision(repo),
                    )

                incompatible = next(repo for repo in all_managed_repos if repo not in adapter.managed_repos)
                self.assertEqual(
                    resolve_image_model_selection(
                        adapter,
                        {"source": "hub", "value": incompatible},
                    ),
                    {"source": "hub", "value": adapter.default_repo},
                )

    def test_image_field_contracts_cover_every_adapter_mode_and_selected_values(self):
        self.assertEqual(set(IMAGE_MODE_FIELD_CONTRACTS), set(IMAGE_PIPELINE_ADAPTERS))
        expected_fields = {
            "negative_prompt",
            "width",
            "height",
            "guidance_scale",
            "strength",
            "padding_mask_crop",
            "max_sequence_length",
            "reference_strength",
            "pag_scale",
            "pag_adaptive_scale",
            "conditioning_scale",
            "image_guidance_scale",
        }
        for pipeline_class, adapter in IMAGE_PIPELINE_ADAPTERS.items():
            with self.subTest(pipeline_class=pipeline_class):
                self.assertEqual(tuple(IMAGE_MODE_FIELD_CONTRACTS[pipeline_class]), adapter.mode_options)
                for mode in adapter.mode_options:
                    self.assertEqual(
                        set(image_pipeline_contract(adapter, mode)["fieldParams"]),
                        expected_fields,
                    )

        flux_text = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["FluxPipeline"], "text_to_image")
        self.assertTrue(flux_text["fieldParams"]["strength"]["hidden"])
        self.assertTrue(flux_text["fieldParams"]["padding_mask_crop"]["hidden"])
        sdxl_edit = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["StableDiffusionXLImg2ImgPipeline"], "edit_image")
        self.assertFalse(sdxl_edit["fieldParams"]["strength"]["hidden"])
        self.assertTrue(sdxl_edit["fieldParams"]["width"]["hidden"])
        redux_multi = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["FluxReduxPipeline"], "multi_image_reference_edit"
        )
        self.assertFalse(redux_multi["fieldParams"]["reference_strength"]["hidden"])
        qwen_edit = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["QwenImageEditPlusPipeline"], "multi_image_reference_edit"
        )
        self.assertTrue(qwen_edit["fieldParams"]["strength"]["hidden"])
        self.assertTrue(qwen_edit["fieldParams"]["reference_strength"]["hidden"])
        pag = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["StableDiffusionPAGPipeline"], "text_to_image")
        self.assertFalse(pag["fieldParams"]["pag_scale"]["hidden"])
        self.assertFalse(pag["fieldParams"]["pag_adaptive_scale"]["hidden"])
        sdxl_pag = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["StableDiffusionXLPAGPipeline"], "text_to_image"
        )
        self.assertFalse(sdxl_pag["fieldParams"]["pag_scale"]["hidden"])
        self.assertFalse(sdxl_pag["fieldParams"]["pag_adaptive_scale"]["hidden"])
        sana = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["SanaPipeline"], "text_to_image")
        self.assertFalse(sana["fieldParams"]["negative_prompt"]["hidden"])
        self.assertFalse(sana["fieldParams"]["max_sequence_length"]["hidden"])
        auraflow = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["AuraFlowPipeline"], "text_to_image")
        self.assertEqual(auraflow["fieldParams"]["width"]["max"], 1536)
        self.assertEqual(auraflow["fieldParams"]["height"]["max"], 1536)
        chroma = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["ChromaPipeline"], "text_to_image")
        self.assertEqual(chroma["fieldParams"]["width"]["max"], 1024)
        self.assertEqual(chroma["fieldParams"]["height"]["max"], 1024)
        cogview3 = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["CogView3PlusPipeline"], "text_to_image"
        )
        for field in ("width", "height"):
            self.assertEqual(
                {key: cogview3["fieldParams"][field][key] for key in ("min", "max", "step")},
                {"min": 512, "max": 2048, "step": 32},
            )
        cogview4 = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["CogView4Pipeline"], "text_to_image")
        for field in ("width", "height"):
            self.assertEqual(
                {key: cogview4["fieldParams"][field][key] for key in ("min", "max", "step")},
                {"min": 512, "max": 2048, "step": 32},
            )
        self.assertEqual(cogview4["maxOutputPixels"], 2**21)
        self.assertEqual(cogview4["fieldParams"]["max_sequence_length"]["max"], 1024)
        ernie = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["ErnieImagePipeline"], "text_to_image")
        for field in ("width", "height"):
            self.assertEqual(
                {key: ernie["fieldParams"][field][key] for key in ("min", "max", "step")},
                {"min": 1024, "max": 1024, "step": 32},
            )
        self.assertEqual(ernie["maxOutputPixels"], 1024 * 1024)
        self.assertTrue(ernie["fieldParams"]["negative_prompt"]["hidden"])
        self.assertTrue(ernie["fieldParams"]["guidance_scale"]["hidden"])
        self.assertTrue(ernie["fieldParams"]["max_sequence_length"]["hidden"])
        glm_image = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["GlmImagePipeline"], "text_to_image")
        for field in ("width", "height"):
            self.assertEqual(
                {key: glm_image["fieldParams"][field][key] for key in ("min", "max", "step")},
                {"min": 1024, "max": 1024, "step": 32},
            )
        self.assertEqual(glm_image["maxOutputPixels"], 1024 * 1024)
        self.assertTrue(glm_image["fieldParams"]["negative_prompt"]["hidden"])
        self.assertFalse(glm_image["fieldParams"]["guidance_scale"]["hidden"])
        self.assertFalse(glm_image["fieldParams"]["max_sequence_length"]["hidden"])
        sana_sprint = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["SanaSprintPipeline"], "text_to_image"
        )
        self.assertTrue(sana_sprint["fieldParams"]["negative_prompt"]["hidden"])
        self.assertFalse(sana_sprint["fieldParams"]["max_sequence_length"]["hidden"])
        sana_sprint_edit = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["SanaSprintImg2ImgPipeline"], "edit_image"
        )
        self.assertFalse(sana_sprint_edit["fieldParams"]["width"]["hidden"])
        self.assertFalse(sana_sprint_edit["fieldParams"]["height"]["hidden"])
        self.assertFalse(sana_sprint_edit["fieldParams"]["strength"]["hidden"])
        for pipeline_name, mode in (
            ("StableDiffusionXLPAGImg2ImgPipeline", "edit_image"),
            ("StableDiffusionXLPAGInpaintPipeline", "inpaint"),
        ):
            with self.subTest(pipeline=pipeline_name):
                contract = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS[pipeline_name], mode)
                self.assertFalse(contract["fieldParams"]["pag_scale"]["hidden"])
                self.assertFalse(contract["fieldParams"]["pag_adaptive_scale"]["hidden"])
                self.assertFalse(contract["fieldParams"]["strength"]["hidden"])
        controlnet = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["StableDiffusionControlNetPipeline"], "control_image"
        )
        self.assertFalse(controlnet["fieldParams"]["conditioning_scale"]["hidden"])
        sdxl_controlnet = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["StableDiffusionXLControlNetPipeline"], "control_image"
        )
        self.assertFalse(sdxl_controlnet["fieldParams"]["conditioning_scale"]["hidden"])
        sdxl_adapter = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["StableDiffusionXLAdapterPipeline"], "control_image"
        )
        self.assertFalse(sdxl_adapter["fieldParams"]["conditioning_scale"]["hidden"])
        turbo = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["StableDiffusionXLTurboPipeline"], "text_to_image"
        )
        self.assertTrue(turbo["fieldParams"]["negative_prompt"]["hidden"])
        self.assertTrue(turbo["fieldParams"]["guidance_scale"]["hidden"])
        instruct = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["StableDiffusionXLInstructPix2PixPipeline"], "edit_image"
        )
        self.assertFalse(instruct["fieldParams"]["image_guidance_scale"]["hidden"])
        self.assertTrue(instruct["fieldParams"]["strength"]["hidden"])

    def test_image_field_contract_rejects_unknown_or_duplicate_visibility_fields(self):
        for fields in (("unknown",), ("strength", "strength")):
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, "unique reviewed"):
                ImageModeFieldContract(fields)

    def test_new_standard_image_adapters_match_pinned_generic_action_signatures(self):
        expected = {
            "StableDiffusionPAGPipeline": ({"text_to_image"}, SD15_BASE_REPO, {"prompt"}),
            "LatentConsistencyModelPipeline": ({"text_to_image"}, LCM_DREAMSHAPER_REPO, {"prompt"}),
            "StableDiffusionPipeline": ({"text_to_image"}, SD15_BASE_REPO, {"prompt"}),
            "StableDiffusionControlNetPipeline": (
                {"control_image"},
                SD15_BASE_REPO,
                {"prompt", "image"},
            ),
            "StableDiffusionImg2ImgPipeline": ({"edit_image"}, SD15_BASE_REPO, {"prompt", "image"}),
            "StableDiffusionInpaintPipeline": (
                {"inpaint", "outpaint"},
                SD15_BASE_REPO,
                {"prompt", "image", "mask_image"},
            ),
            "StableDiffusionXLPipeline": ({"text_to_image"}, SDXL_BASE_REPO, {"prompt"}),
            "StableDiffusionXLTurboPipeline": ({"text_to_image"}, SDXL_TURBO_REPO, {"prompt"}),
            "StableDiffusionXLInstructPix2PixPipeline": (
                {"edit_image"},
                SDXL_INSTRUCT_PIX2PIX_REPO,
                {"prompt", "image", "image_guidance_scale"},
            ),
            "StableDiffusionXLControlNetPipeline": (
                {"control_image"},
                SDXL_BASE_REPO,
                {"prompt", "image"},
            ),
            "HunyuanDiTControlNetPipeline": (
                {"control_image"},
                HUNYUAN_DIT_DISTILLED_REPO,
                {"prompt", "control_image"},
            ),
            "StableDiffusionXLAdapterPipeline": (
                {"control_image"},
                SDXL_BASE_REPO,
                {"prompt", "image"},
            ),
            "StableDiffusionXLPAGPipeline": ({"text_to_image"}, SDXL_BASE_REPO, {"prompt"}),
            "StableDiffusionXLPAGImg2ImgPipeline": (
                {"edit_image"},
                SDXL_BASE_REPO,
                {"prompt", "image"},
            ),
            "StableDiffusionXLPAGInpaintPipeline": (
                {"inpaint"},
                SDXL_BASE_REPO,
                {"prompt", "image", "mask_image"},
            ),
            "SanaPipeline": ({"text_to_image"}, SANA_REPO, {"prompt"}),
            "SanaSprintPipeline": ({"text_to_image"}, SANA_SPRINT_REPO, {"prompt"}),
            "SanaSprintImg2ImgPipeline": (
                {"edit_image"},
                SANA_SPRINT_REPO,
                {"prompt", "image"},
            ),
            "PixArtSigmaPipeline": ({"text_to_image"}, PIXART_SIGMA_REPO, {"prompt"}),
            "AuraFlowPipeline": ({"text_to_image"}, AURAFLOW_V03_REPO, {"prompt"}),
            "ChromaPipeline": ({"text_to_image"}, CHROMA1_HD_REPO, {"prompt"}),
            "CogView3PlusPipeline": ({"text_to_image"}, COGVIEW3_PLUS_REPO, {"prompt"}),
            "CogView4Pipeline": ({"text_to_image"}, COGVIEW4_6B_REPO, {"prompt"}),
            "ErnieImagePipeline": ({"text_to_image"}, ERNIE_IMAGE_TURBO_REPO, {"prompt"}),
            "GlmImagePipeline": ({"text_to_image"}, GLM_IMAGE_REPO, {"prompt"}),
            "DreamLitePipeline": (
                {"text_to_image", "edit_image"},
                DREAMLITE_BASE_REPO,
                {"prompt", "image"},
            ),
            "DreamLiteMobilePipeline": (
                {"text_to_image", "edit_image"},
                DREAMLITE_MOBILE_REPO,
                {"prompt", "image"},
            ),
            "StableDiffusionXLImg2ImgPipeline": ({"edit_image"}, SDXL_BASE_REPO, {"prompt", "image"}),
            "StableDiffusionXLInpaintPipeline": (
                {"inpaint", "outpaint"},
                SDXL_BASE_REPO,
                {"prompt", "image", "mask_image"},
            ),
            "QwenImageImg2ImgPipeline": ({"edit_image"}, QWEN_IMAGE_2512_REPO, {"prompt", "image"}),
            "QwenImageInpaintPipeline": (
                {"inpaint", "outpaint"},
                QWEN_IMAGE_2512_REPO,
                {"prompt", "image", "mask_image"},
            ),
            "QwenImageEditPipeline": ({"edit_image"}, QWEN_IMAGE_EDIT_REPO, {"prompt", "image"}),
            "QwenImageEditPlusPipeline": (
                {"edit_image", "multi_image_reference_edit"},
                QWEN_IMAGE_EDIT_PLUS_REPO,
                {"prompt", "image"},
            ),
            "ZImageImg2ImgPipeline": ({"edit_image"}, Z_IMAGE_REPO, {"prompt", "image"}),
            "ZImageInpaintPipeline": (
                {"inpaint", "outpaint"},
                Z_IMAGE_REPO,
                {"prompt", "image", "mask_image"},
            ),
            "FluxKontextInpaintPipeline": (
                {"inpaint", "outpaint"},
                FLUX_KONTEXT_REPO,
                {"prompt", "image", "mask_image"},
            ),
            "Flux2KleinInpaintPipeline": (
                {"inpaint", "outpaint"},
                FLUX2_KLEIN_REPO,
                {"prompt", "image", "mask_image"},
            ),
        }
        for pipeline_name, (modes, repository, required_inputs) in expected.items():
            with self.subTest(pipeline=pipeline_name):
                adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_name]
                parameters = set(
                    inspect.signature(pipeline_class_from_name(adapter.load_pipeline_class).__call__).parameters
                )
                self.assertEqual(adapter.modes, frozenset(modes))
                self.assertEqual(adapter.default_repo, repository)
                self.assertTrue(required_inputs.issubset(parameters))
                self.assertIn("num_inference_steps", parameters)
                self.assertIn("generator", parameters)
                self.assertIn("output_type", parameters)
                if adapter.guidance_parameter is not None:
                    self.assertIn(adapter.guidance_parameter, parameters)

        for deferred in (
            "Flux2Pipeline",
            "Flux2KleinKVPipeline",
            "FluxControlImg2ImgPipeline",
            "FluxControlInpaintPipeline",
            "FluxControlNetPipeline",
            "FluxControlNetImg2ImgPipeline",
            "FluxControlNetInpaintPipeline",
            "QwenImageControlNetPipeline",
            "QwenImageControlNetInpaintPipeline",
            "QwenImageLayeredPipeline",
            "ZImageControlNetPipeline",
            "ZImageControlNetInpaintPipeline",
            "ZImageOmniPipeline",
        ):
            with self.subTest(deferred=deferred):
                self.assertNotIn(deferred, IMAGE_PIPELINE_ADAPTERS)

    def test_new_standard_image_adapters_execute_only_their_pinned_signature(self):
        image = Image.new("RGB", (16, 16), "black")
        mask = Image.new("L", (16, 16), "white")
        cases = (
            ("StableDiffusionPAGPipeline", "text_to_image", Generate, {}),
            ("LatentConsistencyModelPipeline", "text_to_image", Generate, {}),
            ("StableDiffusionPipeline", "text_to_image", Generate, {}),
            ("StableDiffusionImg2ImgPipeline", "edit_image", Edit, {"image": image}),
            ("StableDiffusionInpaintPipeline", "inpaint", Inpaint, {"image": image, "mask_image": mask}),
            ("StableDiffusionXLPipeline", "text_to_image", Generate, {}),
            ("StableDiffusionXLTurboPipeline", "text_to_image", Generate, {}),
            ("StableDiffusionXLPAGPipeline", "text_to_image", Generate, {}),
            ("StableDiffusionXLPAGImg2ImgPipeline", "edit_image", Edit, {"image": image}),
            (
                "StableDiffusionXLPAGInpaintPipeline",
                "inpaint",
                Inpaint,
                {"image": image, "mask_image": mask},
            ),
            ("SanaPipeline", "text_to_image", Generate, {}),
            ("SanaSprintPipeline", "text_to_image", Generate, {}),
            ("SanaSprintImg2ImgPipeline", "edit_image", Edit, {"image": image}),
            ("PixArtSigmaPipeline", "text_to_image", Generate, {}),
            ("AuraFlowPipeline", "text_to_image", Generate, {}),
            ("ChromaPipeline", "text_to_image", Generate, {}),
            ("CogView3PlusPipeline", "text_to_image", Generate, {}),
            ("CogView4Pipeline", "text_to_image", Generate, {}),
            ("ErnieImagePipeline", "text_to_image", Generate, {}),
            ("GlmImagePipeline", "text_to_image", Generate, {}),
            ("DreamLitePipeline", "text_to_image", Generate, {}),
            ("DreamLitePipeline", "edit_image", Edit, {"image": image}),
            ("DreamLiteMobilePipeline", "text_to_image", Generate, {}),
            ("DreamLiteMobilePipeline", "edit_image", Edit, {"image": image}),
            ("StableDiffusionXLInstructPix2PixPipeline", "edit_image", Edit, {"image": image}),
            (
                "HunyuanDiTControlNetPipeline",
                "control_image",
                ControlGenerate,
                {"control_image": image},
            ),
            ("StableDiffusionXLImg2ImgPipeline", "edit_image", Edit, {"image": image}),
            ("StableDiffusionXLInpaintPipeline", "inpaint", Inpaint, {"image": image, "mask_image": mask}),
            ("QwenImageImg2ImgPipeline", "edit_image", Edit, {"image": image}),
            ("QwenImageInpaintPipeline", "inpaint", Inpaint, {"image": image, "mask_image": mask}),
            ("QwenImageEditPipeline", "edit_image", Edit, {"image": image}),
            ("QwenImageEditPlusPipeline", "edit_image", Edit, {"image": image}),
            ("ZImageImg2ImgPipeline", "edit_image", Edit, {"image": image}),
            ("ZImageInpaintPipeline", "inpaint", Inpaint, {"image": image, "mask_image": mask}),
            ("FluxKontextInpaintPipeline", "inpaint", Inpaint, {"image": image, "mask_image": mask}),
            ("Flux2KleinInpaintPipeline", "inpaint", Inpaint, {"image": image, "mask_image": mask}),
        )
        aliases = {
            "negative_prompt": "negative_prompt",
            "width": "width",
            "height": "height",
            "max_sequence_length": "max_sequence_length",
            "strength": "strength",
            "padding_mask_crop": "padding_mask_crop",
            "reference_strength": "reference_strength",
            "pag_scale": "pag_scale",
            "pag_adaptive_scale": "pag_adaptive_scale",
            "image_guidance_scale": "image_guidance_scale",
        }
        for pipeline_name, mode, action_class, action_inputs in cases:
            with self.subTest(pipeline=pipeline_name):
                adapter = IMAGE_PIPELINE_ADAPTERS[pipeline_name]
                upstream_signature = inspect.signature(
                    pipeline_class_from_name(adapter.load_pipeline_class).__call__
                )
                upstream_parameters = set(upstream_signature.parameters)
                received = {}

                def call(_self, **kwargs):
                    received.update(kwargs)
                    return SimpleNamespace(images=[Image.new("RGB", (16, 16), "white")])

                call.__signature__ = upstream_signature
                fake_type = type(
                    adapter.load_pipeline_class,
                    (),
                    {"_execution_device": "cpu", "__call__": call},
                )
                pipeline = tag_test_image_pipeline(fake_type(), pipeline_name, mode)
                values = {
                    "pipeline": pipeline,
                    "prompt": "render the reviewed fixture",
                    "negative_prompt": "artifact",
                    "width": max(32, adapter.min_output_side),
                    "height": max(32, adapter.min_output_side),
                    "num_inference_steps": 2,
                    "guidance_scale": (
                        adapter.fixed_guidance_scale if adapter.fixed_guidance_scale is not None else 4.0
                    ),
                    "strength": 0.75,
                    "padding_mask_crop": 16,
                    "max_sequence_length": 128,
                    "pag_scale": 3.0,
                    "pag_adaptive_scale": 0.5,
                    "image_guidance_scale": 1.5,
                    "output_type": "pil",
                    **action_inputs,
                }
                initial = {
                    "prompt",
                    "num_inference_steps",
                    "generator",
                    "output_type",
                    "return_dict",
                    *action_inputs.keys(),
                }
                if action_class is Generate:
                    initial.update({"width", "height"})
                expected_keys = initial | {
                    destination
                    for source, destination in aliases.items()
                    if values.get(source) is not None and destination in upstream_parameters
                    and source not in adapter.ignored_generation_parameters
                }
                if adapter.guidance_parameter is not None:
                    expected_keys.add(adapter.guidance_parameter)
                if adapter.conditioning_scale_parameter is not None:
                    expected_keys.add(adapter.conditioning_scale_parameter)

                with patch("modules.DiffusersImage.main.add_progress_callback"):
                    action_class(f"signature-{pipeline_name}").execute(**values)

                self.assertEqual(set(received), expected_keys)
                if adapter.guidance_parameter is not None:
                    self.assertEqual(
                        received[adapter.guidance_parameter],
                        adapter.fixed_guidance_scale if adapter.fixed_guidance_scale is not None else 4.0,
                    )
                else:
                    self.assertNotIn("guidance_scale", received)
                if "negative_prompt" in upstream_parameters:
                    self.assertEqual(received["negative_prompt"], "artifact")

    def test_modern_flux_true_cfg_and_negative_prompt_use_the_reviewed_parameters(self):
        class ModernFlux:
            def __call__(
                self,
                *,
                negative_prompt=None,
                true_cfg_scale=1.0,
                guidance_scale=3.5,
            ):
                return None

        for pipeline_name in (
            "FluxPipeline",
            "FluxImg2ImgPipeline",
            "FluxInpaintPipeline",
            "FluxKontextPipeline",
            "FluxKontextInpaintPipeline",
        ):
            with self.subTest(pipeline=pipeline_name):
                target = {}
                IMAGE_PIPELINE_ADAPTERS[pipeline_name].apply_generation_parameters(
                    ModernFlux(),
                    {"negative_prompt": "artifact", "guidance_scale": 5.0},
                    target,
                )
                self.assertEqual(
                    target,
                    {"negative_prompt": "artifact", "true_cfg_scale": 5.0},
                )

    def test_flux_controlnet_is_not_advertised_without_component_assembly(self):
        self.assertNotIn("FluxControlNetPipeline", IMAGE_PIPELINE_CLASSES)
        node = LoadPipeline("removed-controlnet-probe")
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name") as resolve_pipeline,
            self.assertRaisesRegex(ValueError, "requires a separately loaded FluxControlNetModel"),
        ):
            node(
                model_id=FLUX_DEV_REPO,
                pipeline_class="FluxControlNetPipeline",
                mode="control_image",
            )
        resolve_pipeline.assert_not_called()

    def test_t2i_adapter_is_not_advertised_with_legacy_only_official_weights(self):
        self.assertNotIn("StableDiffusionAdapterPipeline", IMAGE_PIPELINE_CLASSES)
        node = LoadPipeline("removed-t2i-adapter-probe")
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name") as resolve_pipeline,
            self.assertRaisesRegex(ValueError, "legacy PyTorch .bin weights only"),
        ):
            node(
                model_id=SD15_BASE_REPO,
                pipeline_class="StableDiffusionAdapterPipeline",
                mode="control_image",
            )
        resolve_pipeline.assert_not_called()

    def test_pipeline_field_action_publishes_backend_owned_options_defaults_and_signal(self):
        self.assertEqual(LoadPipeline.params["model_id"]["onChange"], "update_pipeline_contract")
        self.assertEqual(
            LoadPipeline.params["pipeline"]["signal"]["value"],
            image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["FluxPipeline"], "text_to_image"),
        )
        node = LoadPipeline("image-contract-action")
        node.set_field_params = Mock()
        node.set_field_value = Mock()

        node.update_pipeline_contract(
            {
                "pipeline_class": "FluxKontextPipeline",
                "mode": "control_image",
                "model_id": {"source": "hub", "value": FLUX_SCHNELL_REPO},
            },
            {"key": "pipeline_class"},
        )

        node.set_field_value.assert_any_call({"mode": "edit_image"})
        node.set_field_value.assert_any_call({"model_id": {"source": "hub", "value": FLUX_KONTEXT_REPO}})
        node.set_field_value.assert_any_call({"revision": catalog_revision(FLUX_KONTEXT_REPO)})
        mode_update = next(call for call in node.set_field_params.call_args_list if call.args[0] == "mode")
        self.assertEqual(mode_update.args[1]["options"], ["edit_image", "multi_image_reference_edit"])
        self.assertEqual(mode_update.args[1]["default"], "edit_image")
        model_update = next(call for call in node.set_field_params.call_args_list if call.args[0] == "model_id")
        self.assertEqual(
            model_update.args[1]["fieldOptions"]["filter"]["hub"]["className"],
            ["FluxKontextPipeline"],
        )
        signal_update = next(call for call in node.set_field_params.call_args_list if call.args[0] == "pipeline")
        signal = signal_update.args[1]["signal"]
        self.assertEqual(signal["origin"], "pipeline_class")
        self.assertEqual(
            signal["value"],
            image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["FluxKontextPipeline"], "edit_image"),
        )

    def test_generate_field_action_applies_only_the_exact_selected_image_contract(self):
        self.assertEqual(
            Generate.params["pipeline"]["onSignal"],
            [
                {"action": "value", "target": "image_contract"},
                {"action": "exec", "data": "update_image_contract"},
            ],
        )
        node = Edit("image-generate-contract-action")
        node.set_field_params = Mock()
        selected = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS["StableDiffusionXLImg2ImgPipeline"], "edit_image")

        node.update_image_contract({"image_contract": selected}, {"key": "pipeline"})

        updates = {call.args[0]: call.args[1] for call in node.set_field_params.call_args_list}
        self.assertEqual(updates["strength"], {"hidden": False})
        self.assertEqual(updates["width"], {"hidden": True})
        self.assertEqual(updates["max_sequence_length"], {"hidden": True})

        tampered = {**selected, "fieldParams": {**selected["fieldParams"], "strength": {"hidden": True}}}
        with self.assertRaisesRegex(ValueError, "stale or mismatched"):
            node.update_image_contract({"image_contract": tampered}, {"key": "pipeline"})
        with self.assertRaisesRegex(ValueError, "does not support this generic image action"):
            Generate("wrong-image-action").update_image_contract(
                {"image_contract": selected},
                {"key": "pipeline"},
            )

    def test_control_generate_field_action_accepts_the_exact_sd15_controlnet_contract(self):
        node = ControlGenerate("sd15-controlnet-contract-action")
        node.set_field_params = Mock()
        selected = image_pipeline_contract(
            IMAGE_PIPELINE_ADAPTERS["StableDiffusionControlNetPipeline"],
            "control_image",
        )

        node.update_image_contract({"image_contract": selected}, {"key": "pipeline"})

        updates = {call.args[0]: call.args[1] for call in node.set_field_params.call_args_list}
        self.assertEqual(updates["conditioning_scale"], {"hidden": False})
        self.assertEqual(updates["strength"], {"hidden": True})

    def test_model_field_action_couples_repository_and_exact_revision_without_stale_pins(self):
        node = LoadPipeline("image-model-revision-action")
        node.set_field_params = Mock()
        node.set_field_value = Mock()

        node.update_pipeline_contract(
            {
                "pipeline_class": "FluxPipeline",
                "mode": "text_to_image",
                "model_id": {"source": "hub", "value": "org/custom-image"},
                "revision": CUSTOM_IMAGE_REVISION,
            },
            {"key": "model_id"},
        )
        node.set_field_value.assert_any_call({"revision": ""})

        node.set_field_value.reset_mock()
        node.update_pipeline_contract(
            {
                "pipeline_class": "FluxPipeline",
                "mode": "text_to_image",
                "model_id": {"source": "hub", "value": FLUX_DEV_REPO},
                "revision": CUSTOM_IMAGE_REVISION,
            },
            {"key": "model_id"},
        )
        node.set_field_value.assert_any_call({"revision": catalog_revision(FLUX_DEV_REPO)})

        node.set_field_value.reset_mock()
        node.update_pipeline_contract(
            {
                "pipeline_class": "FluxPipeline",
                "mode": "text_to_image",
                "model_id": {"source": "local", "value": "models/custom"},
                "revision": CUSTOM_IMAGE_REVISION,
            },
            {"key": "model_id"},
        )
        node.set_field_value.assert_any_call({"revision": ""})

        node.set_field_value.reset_mock()
        node.update_pipeline_contract(
            {
                "pipeline_class": "FluxPipeline",
                "mode": "text_to_image",
                "model_id": {"source": "hub", "value": "org/custom-image"},
                "revision": CUSTOM_IMAGE_REVISION,
            },
            {"key": "mode"},
        )
        self.assertNotIn(
            {"revision": ""},
            [call.args[0] for call in node.set_field_value.call_args_list],
        )

    def test_pipeline_field_action_rejects_noncanonical_identity_and_source(self):
        node = LoadPipeline("invalid-image-contract-action")
        invalid = (
            {"pipeline_class": " FluxPipeline ", "mode": "text_to_image", "model_id": FLUX_SCHNELL_REPO},
            {"pipeline_class": "FluxPipeline", "mode": " text_to_image ", "model_id": FLUX_SCHNELL_REPO},
            {
                "pipeline_class": "FluxPipeline",
                "mode": "text_to_image",
                "model_id": {"source": " hub ", "value": FLUX_SCHNELL_REPO},
            },
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                node.update_pipeline_contract(values, {"key": "pipeline_class"})

    def test_unsupported_mode_fails_before_pipeline_resolution(self):
        node = LoadPipeline("mode-probe")
        with self.assertRaisesRegex(ValueError, "does not support outpaint"):
            node.execute(model_id="example/model", pipeline_class="FluxPipeline", mode="outpaint")

    def test_flux2_klein_supports_generation_and_reference_edits(self):
        node = LoadPipeline("flux2-mode-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _repo, **_kwargs):
                return cls()

        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            for mode in ("text_to_image", "edit_image", "multi_image_reference_edit"):
                node.execute(
                    model_id="black-forest-labs/FLUX.2-klein-4B",
                    pipeline_class="Flux2KleinPipeline",
                    mode=mode,
                    auto_offload=False,
                    offload_mode="none",
                )

    def test_flux_img2img_exposes_only_single_image_edit(self):
        adapter = IMAGE_PIPELINE_ADAPTERS["FluxImg2ImgPipeline"]
        self.assertEqual(adapter.mode_options, ("edit_image",))
        self.assertEqual(adapter.max_reference_images, 1)
        self.assertNotIn(
            "multi_image_reference_edit",
            image_pipeline_contract(adapter, "edit_image")["modes"],
        )

    def test_flux2_klein_reuses_one_resident_loader_when_only_mode_changes(self):
        loaded = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **_kwargs):
                loaded.append(repo)
                return cls()

        node = LoadPipeline("flux2-cache-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        common = {
            "model_id": "black-forest-labs/FLUX.2-klein-4B",
            "pipeline_class": "Flux2KleinPipeline",
            "auto_offload": False,
            "offload_mode": "none",
        }
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            first = node(mode="text_to_image", **common)
            second = node(mode="edit_image", **common)
            self.assertTrue(node._has_changed)
            third = node(mode="edit_image", **common)

        self.assertIs(first["pipeline"], second["pipeline"])
        self.assertIs(second["pipeline"], third["pipeline"])
        self.assertEqual(loaded, ["black-forest-labs/FLUX.2-klein-4B"])
        self.assertEqual(node.params["mode"], "edit_image")
        self.assertFalse(node._has_changed)
        self.assertEqual(second["pipeline"]._modiff_image_pipeline_class, "Flux2KleinPipeline")
        self.assertEqual(second["pipeline"]._modiff_image_mode, "edit_image")
        self.assertEqual(second["pipeline"]._modiff_image_repo, FLUX2_KLEIN_REPO)

    def test_mode_retag_invalidates_cached_generate_edit_and_inpaint_outputs(self):
        image = Image.new("RGB", (8, 8), "black")
        mask = Image.new("L", (8, 8), "white")
        cases = (
            (
                Generate,
                "Flux2KleinPipeline",
                FLUX2_KLEIN_REPO,
                "text_to_image",
                "edit_image",
                {"prompt": "same prompt"},
                "Generate requires one of: text_to_image",
            ),
            (
                Edit,
                "Flux2KleinPipeline",
                FLUX2_KLEIN_REPO,
                "edit_image",
                "text_to_image",
                {"prompt": "same prompt", "image": image},
                "Edit requires one of: edit_image, multi_image_reference_edit",
            ),
            (
                Inpaint,
                "FluxFillPipeline",
                FLUX_FILL_REPO,
                "inpaint",
                "outpaint",
                {
                    "prompt": "same prompt",
                    "image": image,
                    "mask_image": mask,
                    "output_type": "pil",
                },
                None,
            ),
        )

        for action_class, pipeline_class, repository, first_mode, second_mode, values, error in cases:
            with self.subTest(action=action_class.__name__, mode=f"{first_mode}->{second_mode}"):

                class FakePipeline:
                    _execution_device = "cpu"

                    def __init__(self):
                        self.calls = 0

                    def __call__(self, **_kwargs):
                        self.calls += 1
                        return SimpleNamespace(images=[Image.new("RGB", (8, 8), "white")])

                FakePipeline.__name__ = IMAGE_PIPELINE_ADAPTERS[pipeline_class].allowed_runtime_classes[0]
                pipeline = FakePipeline()
                loader = LoadPipeline(f"{action_class.__name__}-loader")
                loader.execute = Mock(return_value={"pipeline": pipeline, "resolved_artifact": repository})
                task = action_class(f"{action_class.__name__}-task")
                server = object.__new__(WebServer)
                server.modules = module_registry.MODULE_MAP
                server.node_cache = {"loader": loader, "task": task}
                server.current_task = None
                server.queue_message = lambda *_args, **_kwargs: None
                graph_node = {
                    "module": "modules.DiffusersImage",
                    "action": action_class.__name__,
                    "params": {
                        "pipeline": {"sourceId": "loader", "sourceKey": "pipeline"},
                        **{key: {"value": value} for key, value in values.items()},
                    },
                }
                common = {
                    "pipeline_class": pipeline_class,
                    "model_id": {"source": "hub", "value": repository},
                }

                with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
                    loader(mode=first_mode, **common)
                    server.execute_node("task", graph_node, "test", quiet=True)
                    loader(mode=second_mode, **common)

                self.assertTrue(loader._has_changed)
                self.assertEqual(loader.execute.call_count, 1)
                if error is not None:
                    with self.assertRaisesRegex(ValueError, error):
                        server.execute_node("task", graph_node, "test", quiet=True)
                    self.assertEqual(pipeline.calls, 1)
                else:
                    server.execute_node("task", graph_node, "test", quiet=True)
                    self.assertTrue(task._has_changed)
                    self.assertEqual(pipeline.calls, 2)

    def test_task_nodes_reject_a_pipeline_loaded_for_another_mode_before_inference(self):
        calls = []

        class TaggedPipeline:
            _execution_device = "cpu"

            def __call__(self, **_kwargs):
                calls.append(True)
                return type("Result", (), {"images": [Image.new("RGB", (8, 8), "white")]})()

        image = Image.new("RGB", (8, 8), "black")
        mask = Image.new("L", (8, 8), "white")
        cases = (
            (
                Generate("wrong-generate-mode"),
                "Flux2KleinPipeline",
                "edit_image",
                {"prompt": "test"},
                "Generate requires one of: text_to_image",
            ),
            (
                Edit("wrong-edit-mode"),
                "Flux2KleinPipeline",
                "text_to_image",
                {"image": image, "prompt": "test"},
                "Edit requires one of: edit_image, multi_image_reference_edit",
            ),
            (
                Inpaint("wrong-inpaint-mode"),
                "FluxControlPipeline",
                "control_image",
                {"image": image, "mask_image": mask, "prompt": "test"},
                "Inpaint requires one of: inpaint, outpaint",
            ),
            (
                ControlGenerate("wrong-control-mode"),
                "FluxFillPipeline",
                "inpaint",
                {"control_image": image, "prompt": "test"},
                "ControlGenerate requires one of: control_image",
            ),
        )
        for node, pipeline_class, mode, kwargs, message in cases:
            with self.subTest(node=type(node).__name__):
                pipeline = tag_test_image_pipeline(TaggedPipeline(), pipeline_class, mode)
                with self.assertRaisesRegex(ValueError, message):
                    node.execute(pipeline=pipeline, **kwargs)

        self.assertEqual(calls, [])

    def test_untagged_multi_action_pipeline_requires_an_exact_loaded_mode(self):
        class Flux2KleinPipeline:
            def __call__(self, **_kwargs):
                self.fail("inference must not run")

        with self.assertRaisesRegex(ValueError, "missing its exact loaded image mode"):
            Generate("untagged-flux2-probe").execute(
                pipeline=Flux2KleinPipeline(),
                prompt="test",
            )

    def test_exact_untagged_single_action_pipeline_has_safe_legacy_recovery(self):
        received = []

        class FluxPipeline:
            _execution_device = "cpu"

            def __call__(self, **kwargs):
                received.append(kwargs)
                return type("Result", (), {"images": [Image.new("RGB", (8, 8), "white")]})()

        result = Generate("legacy-flux-generate-probe").execute(
            pipeline=FluxPipeline(),
            prompt="test",
            width=16,
            height=16,
            num_inference_steps=1,
        )

        self.assertEqual(result["images"][0].size, (8, 8))
        self.assertEqual(received[0]["prompt"], "test")

    def test_unknown_untagged_pipeline_is_rejected_before_inference(self):
        calls = []

        class UnknownPipeline:
            def __call__(self, **_kwargs):
                calls.append(True)

        with self.assertRaisesRegex(ValueError, "Cannot recover an exact Diffusers image adapter"):
            Generate("unknown-image-pipeline-probe").execute(
                pipeline=UnknownPipeline(),
                prompt="test",
            )

        self.assertEqual(calls, [])

    def test_tagged_pipeline_requires_complete_runtime_source_repo_and_revision_consistency(self):
        class PartialPipeline:
            pass

        partial = PartialPipeline()
        partial._modiff_image_pipeline_class = "FluxPipeline"
        partial._modiff_image_mode = "text_to_image"
        with self.assertRaisesRegex(ValueError, "identity is incomplete"):
            validate_image_action(partial, "Generate")

        class FluxPipeline:
            pass

        runtime_mismatch = FluxPipeline()
        qwen = IMAGE_PIPELINE_ADAPTERS["QwenImagePipeline"]
        _tag_image_pipeline(
            runtime_mismatch,
            qwen,
            "text_to_image",
            qwen.default_repo,
            "hub",
            catalog_revision(qwen.default_repo),
        )
        with self.assertRaisesRegex(ValueError, "runtime class 'FluxPipeline' is tagged as QwenImagePipeline"):
            validate_image_action(runtime_mismatch, "Generate")

        class Flux2KleinPipeline:
            pass

        incompatible_repo = Flux2KleinPipeline()
        flux2 = IMAGE_PIPELINE_ADAPTERS["Flux2KleinPipeline"]
        _tag_image_pipeline(
            incompatible_repo,
            flux2,
            "text_to_image",
            FLUX_SCHNELL_REPO,
            "hub",
            catalog_revision(FLUX_SCHNELL_REPO),
        )
        with self.assertRaisesRegex(ValueError, "is not compatible with Flux2KleinPipeline"):
            validate_image_action(incompatible_repo, "Generate")

        valid = tag_test_image_pipeline(FluxPipeline(), "FluxPipeline", "text_to_image")
        self.assertIs(validate_image_action(valid, "Generate"), IMAGE_PIPELINE_ADAPTERS["FluxPipeline"])

        bundle = FluxReduxPipelineBundle(SimpleNamespace(), SimpleNamespace())
        tag_test_image_pipeline(bundle, "FluxReduxPipeline", "edit_image")
        self.assertIs(validate_image_action(bundle, "Edit"), IMAGE_PIPELINE_ADAPTERS["FluxReduxPipeline"])

    def test_all_image_actions_validate_numeric_and_media_contracts_before_torch(self):
        calls = []

        def pipeline(runtime_name):
            pipeline_type = type(runtime_name, (), {"__call__": lambda _self, **_kwargs: calls.append(runtime_name)})
            return pipeline_type()

        image = Image.new("RGB", (16, 16), "black")
        mask = Image.new("L", (16, 16), "white")
        cases = (
            (Generate(), pipeline("FluxPipeline"), {"width": 15}, "width"),
            (Edit(), pipeline("FluxImg2ImgPipeline"), {"image": image, "height": 31}, "height"),
            (
                Inpaint(),
                pipeline("FluxFillPipeline"),
                {"image": image, "mask_image": mask, "num_inference_steps": 0},
                "num_inference_steps",
            ),
            (
                ControlGenerate(),
                pipeline("FluxControlPipeline"),
                {"control_image": image, "guidance_scale": float("nan")},
                "guidance_scale",
            ),
            (Edit(), pipeline("FluxImg2ImgPipeline"), {"image": []}, "empty image list"),
            (
                Inpaint(),
                pipeline("FluxFillPipeline"),
                {"image": "not-an-image", "mask_image": "not-a-mask"},
                "must be a PIL image",
            ),
            (
                ControlGenerate(),
                pipeline("FluxControlPipeline"),
                {"control_image": "not-an-image"},
                "PIL image, NumPy array, or Torch tensor",
            ),
        )
        for node, selected_pipeline, values, message in cases:
            with self.subTest(node=type(node).__name__, message=message):
                with (
                    patch.dict(sys.modules, {"torch": None}),
                    self.assertRaisesRegex(ValueError, message) as raised,
                ):
                    node.execute(pipeline=selected_pipeline, **values)
                self.assertNotIn("torch halted", str(raised.exception))
        self.assertEqual(calls, [])

    def test_generate_preflight_rejects_every_declared_bound_and_nonfinite_value(self):
        class FluxPipeline:
            def __call__(self, **_kwargs):
                raise AssertionError("inference must not run")

        invalid = (
            ("width", 2064),
            ("height", 17),
            ("seed", 4294967296),
            ("num_inference_steps", 101),
            ("guidance_scale", float("inf")),
            ("pag_scale", -0.1),
            ("pag_adaptive_scale", float("inf")),
            ("strength", -0.01),
            ("padding_mask_crop", 7),
            ("max_sequence_length", 513),
            ("output_type", "latent"),
        )
        for field, value in invalid:
            with self.subTest(field=field, value=value), patch.dict(sys.modules, {"torch": None}):
                with self.assertRaises(ValueError) as raised:
                    Generate().execute(pipeline=FluxPipeline(), **{field: value})
                self.assertNotIn("torch halted", str(raised.exception))

    def test_action_preflight_runs_through_real_nodebase_before_torch_or_upstream(self):
        calls = []

        class FluxPipeline:
            def __call__(self, **_kwargs):
                calls.append(True)

        node = Generate("real-nodebase-image-preflight")
        with patch.dict(sys.modules, {"torch": None}), self.assertRaisesRegex(ValueError, "width") as raised:
            node(pipeline=FluxPipeline(), width=15)

        self.assertNotIn("torch halted", str(raised.exception))
        self.assertEqual(calls, [])

    def test_facade_rejects_raw_nodebase_coercion_inputs_before_torch(self):
        class FluxPipeline:
            def __call__(self, **_kwargs):
                raise AssertionError("inference must not run")

        image = Image.new("RGB", (16, 16), "black")
        mask = Image.new("L", (16, 16), "white")
        cases = (
            (Generate("raw-generate-bool"), FluxPipeline(), {"width": True}),
            (Generate("raw-generate-blank"), FluxPipeline(), {"num_inference_steps": ""}),
            (Edit("raw-edit-container"), type("FluxImg2ImgPipeline", (), {})(), {"image": image, "height": []}),
            (
                Inpaint("raw-inpaint-bool"),
                type("FluxFillPipeline", (), {})(),
                {"image": image, "mask_image": mask, "padding_mask_crop": False},
            ),
            (
                ControlGenerate("raw-control-container"),
                type("FluxControlPipeline", (), {})(),
                {"control_image": image, "guidance_scale": {}},
            ),
        )
        for node, pipeline, values in cases:
            with self.subTest(node=node.node_id), patch.dict(sys.modules, {"torch": None}):
                with self.assertRaises(ValueError):
                    node(pipeline=pipeline, **values)

    def test_media_shape_duck_types_are_not_accepted_as_images(self):
        class FluxImg2ImgPipeline:
            pass

        with (
            patch.dict(sys.modules, {"torch": None}),
            self.assertRaisesRegex(ValueError, "PIL image, NumPy array, or Torch tensor"),
        ):
            Edit().execute(
                pipeline=FluxImg2ImgPipeline(),
                image=SimpleNamespace(shape=(16, 16, 3)),
            )

    def test_reference_count_and_pixel_limits_fail_before_stitching_or_torch(self):
        class FluxImg2ImgPipeline:
            def __call__(self, **_kwargs):
                raise AssertionError("inference must not run")

        oversized_shape = np.broadcast_to(np.zeros((1, 1, 3), dtype=np.uint8), (8192, 8192, 3))
        with patch.dict(sys.modules, {"torch": None}), self.assertRaisesRegex(ValueError, "cumulative input limit"):
            Edit().execute(pipeline=FluxImg2ImgPipeline(), image=oversized_shape)

        references = [Image.new("RGB", (1, 1), "black") for _ in range(2)]
        with patch.dict(sys.modules, {"torch": None}), self.assertRaisesRegex(ValueError, "at most 1"):
            Edit().execute(pipeline=FluxImg2ImgPipeline(), image=references)

        class Flux2KleinPipeline:
            pass

        multi_mode = tag_test_image_pipeline(Flux2KleinPipeline(), "Flux2KleinPipeline", "multi_image_reference_edit")
        with patch.dict(sys.modules, {"torch": None}), self.assertRaisesRegex(ValueError, "at most 8"):
            Edit().execute(
                pipeline=multi_mode,
                image=[Image.new("RGB", (1, 1), "black") for _ in range(9)],
            )

        single_mode = tag_test_image_pipeline(Flux2KleinPipeline(), "Flux2KleinPipeline", "edit_image")
        with patch.dict(sys.modules, {"torch": None}), self.assertRaisesRegex(ValueError, "at most 1"):
            Edit().execute(
                pipeline=single_mode,
                image=[Image.new("RGB", (1, 1)), Image.new("RGB", (1, 1))],
            )

    def test_generate_preflight_preserves_valid_zero_and_boundary_values(self):
        received = {}

        class FluxPipeline:
            _execution_device = "cpu"

            def __call__(
                self,
                *,
                prompt,
                width,
                height,
                num_inference_steps,
                generator,
                output_type,
                return_dict,
                true_cfg_scale,
                strength,
                max_sequence_length,
            ):
                received.update(
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    output_type=output_type,
                    true_cfg_scale=true_cfg_scale,
                    strength=strength,
                    max_sequence_length=max_sequence_length,
                    seed=generator.initial_seed(),
                )
                return SimpleNamespace(images=[Image.new("RGB", (16, 16), "white")])

        Generate().execute(
            pipeline=FluxPipeline(),
            width=16,
            height=2048,
            seed=4294967295,
            num_inference_steps=1,
            guidance_scale=0,
            strength=0,
            padding_mask_crop=0,
            max_sequence_length=512,
            output_type="pil",
        )

        self.assertEqual(
            received,
            {
                "width": 16,
                "height": 2048,
                "num_inference_steps": 1,
                "output_type": "pil",
                "true_cfg_scale": 0.0,
                "strength": 0.0,
                "max_sequence_length": 512,
                "seed": 4294967295,
            },
        )

    def test_cross_workflow_loader_reuse_removes_residual_lora(self):
        unloads = []
        pipeline = type("Pipeline", (), {"unload_lora_weights": lambda _self: unloads.append(True)})()
        node = LoadPipeline("image-loader-reuse")
        node.output["pipeline"] = pipeline

        node.prepare_for_workflow_reuse()

        self.assertEqual(unloads, [True])
        self.assertIs(node.output["pipeline"], pipeline)

    def test_two_repositories_use_the_same_generic_loader_contract(self):
        loaded = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.append(repo)
                return cls()

        node = LoadPipeline("repo-switch-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            for repo in ("org/flux-compatible-a", "org/flux-compatible-b"):
                result = node.execute(
                    model_id=repo,
                    pipeline_class="FluxPipeline",
                    mode="text_to_image",
                    revision=CUSTOM_IMAGE_REVISION,
                    auto_offload=False,
                    offload_mode="none",
                )
                self.assertEqual(result["resolved_artifact"], repo)

        self.assertEqual(loaded, ["org/flux-compatible-a", "org/flux-compatible-b"])

    def test_sdxl_turbo_loads_the_reviewed_upstream_fp16_safetensors_variant(self):
        loaded = {}

        class StableDiffusionXLPipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        node = LoadPipeline("sdxl-turbo-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=StableDiffusionXLPipeline,
            ) as resolve_pipeline,
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=SDXL_TURBO_REPO,
                pipeline_class="StableDiffusionXLTurboPipeline",
                mode="text_to_image",
                revision=catalog_revision(SDXL_TURBO_REPO),
                dtype="float16",
                auto_offload=False,
                offload_mode="none",
            )

        resolve_pipeline.assert_called_once_with("StableDiffusionXLPipeline")
        self.assertEqual(loaded["repo"], SDXL_TURBO_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(SDXL_TURBO_REPO))
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertEqual(loaded["kwargs"]["variant"], "fp16")
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        self.assertEqual(result["pipeline"]._modiff_image_pipeline_class, "StableDiffusionXLTurboPipeline")

    def test_sdxl_turbo_rejects_non_native_steps_and_guidance(self):
        class StableDiffusionXLPipeline:
            def __call__(self, **_kwargs):
                raise AssertionError("invalid Turbo settings must fail before inference")

        pipeline = tag_test_image_pipeline(
            StableDiffusionXLPipeline(),
            "StableDiffusionXLTurboPipeline",
            "text_to_image",
        )
        for kwargs, message in (
            ({"num_inference_steps": 5, "guidance_scale": 0.0}, "between 1 and 4"),
            ({"num_inference_steps": 1, "guidance_scale": 0.1}, "requires guidance_scale=0"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                Generate("sdxl-turbo-native-contract").execute(
                    pipeline=pipeline,
                    prompt="reviewed fixture",
                    width=32,
                    height=32,
                    **kwargs,
                )

    def test_sdxl_pag_loads_the_reviewed_fp16_safetensors_variant(self):
        for pipeline_name, mode in (
            ("StableDiffusionXLPAGPipeline", "text_to_image"),
            ("StableDiffusionXLPAGImg2ImgPipeline", "edit_image"),
            ("StableDiffusionXLPAGInpaintPipeline", "inpaint"),
        ):
            with self.subTest(pipeline=pipeline_name):
                loaded = {}

                class ReviewedPAGPipeline:
                    @classmethod
                    def from_pretrained(cls, repo, **kwargs):
                        loaded.update({"repo": repo, "kwargs": kwargs})
                        return cls()

                node = LoadPipeline(f"sdxl-pag-load-probe-{mode}")
                node.progress = lambda *args, **kwargs: None
                node.mm_add = lambda *args, **kwargs: None
                with (
                    patch(
                        "modules.DiffusersImage.main.pipeline_class_from_name",
                        return_value=ReviewedPAGPipeline,
                    ) as resolve_pipeline,
                    patch("modules.DiffusersImage.main.apply_pipeline_offload"),
                ):
                    result = node.execute(
                        model_id=SDXL_BASE_REPO,
                        pipeline_class=pipeline_name,
                        mode=mode,
                        revision=catalog_revision(SDXL_BASE_REPO),
                        dtype="float16",
                        auto_offload=False,
                        offload_mode="none",
                    )

                resolve_pipeline.assert_called_once_with(pipeline_name)
                self.assertEqual(loaded["repo"], SDXL_BASE_REPO)
                self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(SDXL_BASE_REPO))
                self.assertTrue(loaded["kwargs"]["use_safetensors"])
                self.assertEqual(loaded["kwargs"]["variant"], "fp16")
                self.assertNotIn("trust_remote_code", loaded["kwargs"])
                self.assertEqual(result["pipeline"]._modiff_image_pipeline_class, pipeline_name)

    def test_sana_loads_the_reviewed_mixed_precision_safetensors_variant(self):
        loaded = {}

        class PrecisionComponent:
            def __init__(self):
                self.to_calls = []

            def to(self, dtype):
                self.to_calls.append(dtype)
                return self

        class SanaPipeline:
            def __init__(self):
                self.text_encoder = PrecisionComponent()
                self.vae = PrecisionComponent()

            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        node = LoadPipeline("sana-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=SanaPipeline),
            patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=SANA_REPO,
                pipeline_class="SanaPipeline",
                mode="text_to_image",
                revision=catalog_revision(SANA_REPO),
                dtype="float16",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], SANA_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(SANA_REPO))
        self.assertEqual(loaded["kwargs"]["torch_dtype"], "float16")
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertEqual(loaded["kwargs"]["variant"], "fp16")
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        self.assertEqual(result["pipeline"].text_encoder.to_calls, ["bfloat16"])
        self.assertEqual(result["pipeline"].vae.to_calls, ["bfloat16"])

    def test_sana_sprint_loads_native_bfloat16_safetensors_and_bounds_steps(self):
        loaded = {}

        class SanaSprintPipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(self, **_kwargs):
                raise AssertionError("invalid Sprint settings must fail before inference")

        node = LoadPipeline("sana-sprint-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=SanaSprintPipeline),
            patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=SANA_SPRINT_REPO,
                pipeline_class="SanaSprintPipeline",
                mode="text_to_image",
                revision=catalog_revision(SANA_SPRINT_REPO),
                dtype="bfloat16",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], SANA_SPRINT_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(SANA_SPRINT_REPO))
        self.assertEqual(loaded["kwargs"]["torch_dtype"], "bfloat16")
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertNotIn("variant", loaded["kwargs"])
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        with self.assertRaisesRegex(ValueError, "between 1 and 4"):
            Generate("sana-sprint-step-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=32,
                height=32,
                num_inference_steps=5,
                guidance_scale=4.5,
            )

    def test_pixart_sigma_loads_only_the_pinned_safetensors_and_bounds_recipe(self):
        loaded = {}

        class PixArtSigmaPipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(self, **_kwargs):
                raise AssertionError("invalid PixArt settings must fail before inference")

        node = LoadPipeline("pixart-sigma-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=PixArtSigmaPipeline,
            ),
            patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=PIXART_SIGMA_REPO,
                pipeline_class="PixArtSigmaPipeline",
                mode="text_to_image",
                revision=catalog_revision(PIXART_SIGMA_REPO),
                dtype="float16",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], PIXART_SIGMA_REPO)
        self.assertEqual(
            loaded["kwargs"]["revision"],
            catalog_revision(PIXART_SIGMA_REPO),
        )
        self.assertEqual(loaded["kwargs"]["torch_dtype"], "float16")
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertNotIn("variant", loaded["kwargs"])
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            Generate("pixart-sigma-step-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1024,
                height=1024,
                num_inference_steps=51,
                guidance_scale=4.5,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 300"):
            Generate("pixart-sigma-sequence-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1024,
                height=1024,
                num_inference_steps=20,
                guidance_scale=4.5,
                max_sequence_length=301,
            )

    def test_auraflow_loads_only_the_pinned_fp16_partition_and_bounds_native_recipe(self):
        loaded = {}

        class AuraFlowPipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(self, **_kwargs):
                raise AssertionError("invalid AuraFlow settings must fail before inference")

        node = LoadPipeline("auraflow-v0.3-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=AuraFlowPipeline,
            ),
            patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=AURAFLOW_V03_REPO,
                pipeline_class="AuraFlowPipeline",
                mode="text_to_image",
                revision=catalog_revision(AURAFLOW_V03_REPO),
                dtype="float16",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], AURAFLOW_V03_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(AURAFLOW_V03_REPO))
        self.assertEqual(loaded["kwargs"]["torch_dtype"], "float16")
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertEqual(loaded["kwargs"]["variant"], "fp16")
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        for field, value in (("width", 1552), ("height", 1552)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "between 16 and 1536"):
                Generate(f"auraflow-{field}-contract").execute(
                    pipeline=result["pipeline"],
                    prompt="reviewed fixture",
                    width=value if field == "width" else 1536,
                    height=value if field == "height" else 768,
                    num_inference_steps=50,
                    guidance_scale=3.5,
                    max_sequence_length=256,
                )
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            Generate("auraflow-step-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1536,
                height=768,
                num_inference_steps=51,
                guidance_scale=3.5,
                max_sequence_length=256,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 256"):
            Generate("auraflow-sequence-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1536,
                height=768,
                num_inference_steps=50,
                guidance_scale=3.5,
                max_sequence_length=257,
            )

    def test_chroma_loads_only_the_pinned_safe_partition_and_bounds_reviewed_recipe(self):
        loaded = {}

        class ChromaPipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(self, **_kwargs):
                raise AssertionError("invalid Chroma settings must fail before inference")

        node = LoadPipeline("chroma1-hd-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=ChromaPipeline,
            ),
            patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=CHROMA1_HD_REPO,
                pipeline_class="ChromaPipeline",
                mode="text_to_image",
                revision=catalog_revision(CHROMA1_HD_REPO),
                dtype="bfloat16",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], CHROMA1_HD_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(CHROMA1_HD_REPO))
        self.assertEqual(loaded["kwargs"]["torch_dtype"], "bfloat16")
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertNotIn("variant", loaded["kwargs"])
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        for field, value in (("width", 1040), ("height", 1040)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "between 16 and 1024"):
                Generate(f"chroma-{field}-contract").execute(
                    pipeline=result["pipeline"],
                    prompt="reviewed fixture",
                    width=value if field == "width" else 1024,
                    height=value if field == "height" else 1024,
                    num_inference_steps=40,
                    guidance_scale=3.0,
                    max_sequence_length=512,
                )
        with self.assertRaisesRegex(ValueError, "between 1 and 40"):
            Generate("chroma-step-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1024,
                height=1024,
                num_inference_steps=41,
                guidance_scale=3.0,
                max_sequence_length=512,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 512"):
            Generate("chroma-sequence-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1024,
                height=1024,
                num_inference_steps=40,
                guidance_scale=3.0,
                max_sequence_length=513,
            )

    def test_cogview3_loads_only_the_pinned_bfloat16_partition_and_bounds_native_recipe(self):
        loaded = {}

        class CogView3PlusPipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(self, **_kwargs):
                raise AssertionError("invalid CogView3 settings must fail before inference")

        node = LoadPipeline("cogview3-plus-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=CogView3PlusPipeline,
            ),
            patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=COGVIEW3_PLUS_REPO,
                pipeline_class="CogView3PlusPipeline",
                mode="text_to_image",
                revision=catalog_revision(COGVIEW3_PLUS_REPO),
                dtype="bfloat16",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], COGVIEW3_PLUS_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(COGVIEW3_PLUS_REPO))
        self.assertEqual(loaded["kwargs"]["torch_dtype"], "bfloat16")
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertNotIn("variant", loaded["kwargs"])
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        for field, value, message in (
            ("width", 480, "between 512 and 2048"),
            ("height", 2080, "between 512 and 2048"),
            ("width", 528, "increments of 32 from 512"),
        ):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                Generate(f"cogview3-{field}-{value}-contract").execute(
                    pipeline=result["pipeline"],
                    prompt="reviewed fixture",
                    width=value if field == "width" else 1024,
                    height=value if field == "height" else 1024,
                    num_inference_steps=50,
                    guidance_scale=7.0,
                    max_sequence_length=224,
                )
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            Generate("cogview3-step-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1024,
                height=1024,
                num_inference_steps=51,
                guidance_scale=7.0,
                max_sequence_length=224,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 224"):
            Generate("cogview3-sequence-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1024,
                height=1024,
                num_inference_steps=50,
                guidance_scale=7.0,
                max_sequence_length=225,
            )

    def test_cogview4_loads_only_the_pinned_bfloat16_partition_and_bounds_native_recipe(self):
        loaded = {}

        class CogView4Pipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(self, **_kwargs):
                raise AssertionError("invalid CogView4 settings must fail before inference")

        node = LoadPipeline("cogview4-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=CogView4Pipeline,
            ),
            patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=COGVIEW4_6B_REPO,
                pipeline_class="CogView4Pipeline",
                mode="text_to_image",
                revision=catalog_revision(COGVIEW4_6B_REPO),
                dtype="bfloat16",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], COGVIEW4_6B_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(COGVIEW4_6B_REPO))
        self.assertEqual(loaded["kwargs"]["torch_dtype"], "bfloat16")
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertNotIn("variant", loaded["kwargs"])
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        for field, value, message in (
            ("width", 480, "between 512 and 2048"),
            ("height", 2080, "between 512 and 2048"),
            ("width", 528, "increments of 32 from 512"),
        ):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, message):
                Generate(f"cogview4-{field}-{value}-contract").execute(
                    pipeline=result["pipeline"],
                    prompt="reviewed fixture",
                    width=value if field == "width" else 1024,
                    height=value if field == "height" else 1024,
                    num_inference_steps=50,
                    guidance_scale=3.5,
                    max_sequence_length=1024,
                )
        with self.assertRaisesRegex(ValueError, "cannot exceed 2097152 pixels"):
            Generate("cogview4-pixel-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=2048,
                height=1056,
                num_inference_steps=50,
                guidance_scale=3.5,
                max_sequence_length=1024,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            Generate("cogview4-step-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1024,
                height=1024,
                num_inference_steps=51,
                guidance_scale=3.5,
                max_sequence_length=1024,
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 1024"):
            Generate("cogview4-sequence-contract").execute(
                pipeline=result["pipeline"],
                prompt="reviewed fixture",
                width=1024,
                height=1024,
                num_inference_steps=50,
                guidance_scale=3.5,
                max_sequence_length=1025,
            )

    def test_ernie_image_turbo_loads_safe_bfloat16_and_enforces_the_exact_recipe(self):
        loaded = {}
        called = {}

        class ErnieImagePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(
                self,
                prompt=None,
                negative_prompt="",
                height=1024,
                width=1024,
                num_inference_steps=50,
                guidance_scale=4.0,
                generator=None,
                output_type="pil",
                return_dict=True,
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
                use_pe=True,
            ):
                called.update(locals())
                return SimpleNamespace(images=[Image.new("RGB", (width, height))])

        node = LoadPipeline("ernie-image-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=ErnieImagePipeline,
            ),
            patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=ERNIE_IMAGE_TURBO_REPO,
                pipeline_class="ErnieImagePipeline",
                mode="text_to_image",
                revision=catalog_revision(ERNIE_IMAGE_TURBO_REPO),
                dtype="bfloat16",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], ERNIE_IMAGE_TURBO_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(ERNIE_IMAGE_TURBO_REPO))
        self.assertEqual(loaded["kwargs"]["torch_dtype"], "bfloat16")
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertNotIn("variant", loaded["kwargs"])
        self.assertNotIn("trust_remote_code", loaded["kwargs"])

        generate = Generate("ernie-image-generate-probe")
        generate.progress = lambda *args, **kwargs: None
        generated = generate.execute(
            pipeline=result["pipeline"],
            prompt="reviewed fixture",
            negative_prompt="",
            width=1024,
            height=1024,
            num_inference_steps=8,
            guidance_scale=1.0,
            max_sequence_length=2048,
            seed=7,
            output_type="pil",
        )
        self.assertEqual(generated["width_out"], 1024)
        self.assertEqual(generated["height_out"], 1024)
        self.assertEqual(called["num_inference_steps"], 8)
        self.assertEqual(called["guidance_scale"], 1.0)
        self.assertTrue(called["use_pe"])
        self.assertNotIn("max_sequence_length", called)

        for field, value, message in (
            ("width", 1008, "between 1024 and 1024"),
            ("height", 1056, "between 1024 and 1024"),
            ("num_inference_steps", 9, "between 1 and 8"),
            ("guidance_scale", 1.1, "requires guidance_scale=1"),
        ):
            values = {
                "pipeline": result["pipeline"],
                "prompt": "reviewed fixture",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 8,
                "guidance_scale": 1.0,
                "max_sequence_length": 2048,
            }
            values[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                Generate(f"ernie-image-{field}-contract").execute(**values)

    def test_glm_image_loads_safe_bfloat16_with_fp32_t5_and_enforces_the_exact_recipe(self):
        loaded = {}
        called = {}
        placements = []

        class TextEncoder:
            def to(self, dtype):
                placements.append(dtype)
                return self

        class GlmImagePipeline:
            def __init__(self):
                self.text_encoder = TextEncoder()

            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(
                self,
                prompt=None,
                height=1024,
                width=1024,
                num_inference_steps=50,
                guidance_scale=1.5,
                max_sequence_length=2048,
                negative_prompt_embeds=None,
                generator=None,
                output_type="pil",
                return_dict=True,
                callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None,
            ):
                called.update(locals())
                return SimpleNamespace(images=[Image.new("RGB", (width, height))])

        node = LoadPipeline("glm-image-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=GlmImagePipeline,
            ),
            patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=GLM_IMAGE_REPO,
                pipeline_class="GlmImagePipeline",
                mode="text_to_image",
                revision=catalog_revision(GLM_IMAGE_REPO),
                dtype="bfloat16",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], GLM_IMAGE_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(GLM_IMAGE_REPO))
        self.assertEqual(loaded["kwargs"]["torch_dtype"], "bfloat16")
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertNotIn("variant", loaded["kwargs"])
        self.assertNotIn("trust_remote_code", loaded["kwargs"])
        self.assertEqual(placements, ["float32"])

        generate = Generate("glm-image-generate-probe")
        generate.progress = lambda *args, **kwargs: None
        generated = generate.execute(
            pipeline=result["pipeline"],
            prompt="reviewed fixture",
            negative_prompt="must not reach the package API",
            width=1024,
            height=1024,
            num_inference_steps=50,
            guidance_scale=1.5,
            max_sequence_length=2048,
            seed=7,
            output_type="pil",
        )
        self.assertEqual(generated["width_out"], 1024)
        self.assertEqual(generated["height_out"], 1024)
        self.assertEqual(called["num_inference_steps"], 50)
        self.assertEqual(called["guidance_scale"], 1.5)
        self.assertEqual(called["max_sequence_length"], 2048)
        self.assertIsNone(called["negative_prompt_embeds"])

        for field, value, message in (
            ("width", 992, "between 1024 and 1024"),
            ("height", 1056, "between 1024 and 1024"),
            ("num_inference_steps", 51, "between 1 and 50"),
            ("max_sequence_length", 2049, "between 1 and 2048"),
        ):
            values = {
                "pipeline": result["pipeline"],
                "prompt": "reviewed fixture",
                "width": 1024,
                "height": 1024,
                "num_inference_steps": 50,
                "guidance_scale": 1.5,
                "max_sequence_length": 2048,
            }
            values[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                Generate(f"glm-image-{field}-contract").execute(**values)

    def test_dreamlite_loaders_are_exact_safe_and_bound_base_and_mobile_recipes(self):
        loaded = {}

        def pipeline_class(name):
            def from_pretrained(cls, repo, **kwargs):
                loaded[name] = {"repo": repo, "kwargs": kwargs}
                return cls()

            def call(self, **_kwargs):
                raise AssertionError("invalid DreamLite settings must fail before inference")

            return type(name, (), {"from_pretrained": classmethod(from_pretrained), "__call__": call})

        for name, repository, mode in (
            ("DreamLitePipeline", DREAMLITE_BASE_REPO, "text_to_image"),
            ("DreamLiteMobilePipeline", DREAMLITE_MOBILE_REPO, "edit_image"),
        ):
            with self.subTest(pipeline=name):
                node = LoadPipeline(f"{name}-load-probe")
                node.progress = lambda *args, **kwargs: None
                node.mm_add = lambda *args, **kwargs: None
                with (
                    patch(
                        "modules.DiffusersImage.main.pipeline_class_from_name",
                        return_value=pipeline_class(name),
                    ),
                    patch("modules.DiffusersImage.main.str_to_dtype", side_effect=lambda value: value),
                    patch("modules.DiffusersImage.main.apply_pipeline_offload"),
                ):
                    node.execute(
                        model_id=repository,
                        pipeline_class=name,
                        mode=mode,
                        revision=catalog_revision(repository),
                        dtype="bfloat16",
                        auto_offload=False,
                        offload_mode="none",
                    )

                self.assertEqual(loaded[name]["repo"], repository)
                self.assertEqual(loaded[name]["kwargs"]["revision"], catalog_revision(repository))
                self.assertEqual(loaded[name]["kwargs"]["torch_dtype"], "bfloat16")
                self.assertTrue(loaded[name]["kwargs"]["use_safetensors"])
                self.assertNotIn("variant", loaded[name]["kwargs"])
                self.assertNotIn("trust_remote_code", loaded[name]["kwargs"])

        base = tag_test_image_pipeline(
            pipeline_class("DreamLitePipeline")(),
            "DreamLitePipeline",
            "text_to_image",
        )
        with self.assertRaisesRegex(ValueError, "between 1 and 50"):
            Generate("dreamlite-base-step-contract").execute(
                pipeline=base,
                prompt="reviewed fixture",
                num_inference_steps=51,
                guidance_scale=3.5,
            )

        mobile = tag_test_image_pipeline(
            pipeline_class("DreamLiteMobilePipeline")(),
            "DreamLiteMobilePipeline",
            "text_to_image",
        )
        with self.assertRaisesRegex(ValueError, "requires guidance_scale=0"):
            Generate("dreamlite-mobile-guidance-contract").execute(
                pipeline=mobile,
                prompt="reviewed fixture",
                num_inference_steps=4,
                guidance_scale=1.0,
            )

    def test_sdxl_instruct_pix2pix_loads_safetensors_and_enforces_image_guidance(self):
        loaded = {}

        class StableDiffusionXLInstructPix2PixPipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

            def __call__(self, **_kwargs):
                raise AssertionError("invalid image guidance must fail before inference")

        node = LoadPipeline("sdxl-instruct-load-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                return_value=StableDiffusionXLInstructPix2PixPipeline,
            ) as resolve_pipeline,
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id=SDXL_INSTRUCT_PIX2PIX_REPO,
                pipeline_class="StableDiffusionXLInstructPix2PixPipeline",
                mode="edit_image",
                revision=catalog_revision(SDXL_INSTRUCT_PIX2PIX_REPO),
                dtype="float16",
                auto_offload=False,
                offload_mode="none",
            )

        resolve_pipeline.assert_called_once_with("StableDiffusionXLInstructPix2PixPipeline")
        self.assertEqual(loaded["repo"], SDXL_INSTRUCT_PIX2PIX_REPO)
        self.assertEqual(loaded["kwargs"]["revision"], catalog_revision(SDXL_INSTRUCT_PIX2PIX_REPO))
        self.assertTrue(loaded["kwargs"]["use_safetensors"])
        self.assertNotIn("variant", loaded["kwargs"])
        self.assertNotIn("trust_remote_code", loaded["kwargs"])

        with self.assertRaisesRegex(ValueError, "image_guidance_scale must be finite and between 1.0 and 20.0"):
            Edit("sdxl-instruct-guidance-contract").execute(
                pipeline=result["pipeline"],
                prompt="turn the sky cloudy",
                image=Image.new("RGB", (16, 16), "blue"),
                num_inference_steps=30,
                guidance_scale=3.0,
                image_guidance_scale=0.9,
            )

    def test_execution_recipe_controls_load_placement_attention_and_offload(self):
        loaded = {}
        offload = {}

        class FakeTransformer:
            def __init__(self):
                self.backends = []

            def set_attention_backend(self, backend):
                self.backends.append(backend)

        class FakePipeline:
            def __init__(self):
                self.transformer = FakeTransformer()

            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        def capture_offload(_pipeline, **kwargs):
            offload.update(kwargs)

        quant_config = object()
        recipe = {
            "quantization_config": quant_config,
            "device_map": {"transformer": 0, "text_encoder_2": "cpu"},
            "max_memory": {0: "16GiB", "cpu": "48GiB"},
            "offload_mode": "group_cpu",
            "device": "cuda:0",
            "attention_backend": "native",
            "attention_components": ["transformer"],
            "vae_slicing": False,
            "vae_tiling": False,
        }

        node = LoadPipeline("recipe-loader-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload", side_effect=capture_offload),
        ):
            with self.assertRaisesRegex(ValueError, "PipelineQuantizationConfig"):
                node.execute(
                    model_id="org/runtime-recipe-model",
                    pipeline_class="FluxPipeline",
                    mode="text_to_image",
                    revision=CUSTOM_IMAGE_REVISION,
                    execution_recipe=recipe,
                )
            recipe["quantization_config"] = None
            result = node.execute(
                model_id="org/runtime-recipe-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                revision=CUSTOM_IMAGE_REVISION,
                execution_recipe=recipe,
            )

        self.assertNotIn("quantization_config", loaded["kwargs"])
        self.assertEqual(loaded["kwargs"]["device_map"], recipe["device_map"])
        self.assertEqual(loaded["kwargs"]["max_memory"], recipe["max_memory"])
        self.assertEqual(offload["mode"], "group_cpu")
        self.assertEqual(offload["device"], "cuda:0")
        self.assertEqual(result["pipeline"].transformer.backends, ["native"])

    def test_execution_recipe_applies_cache_and_compile_to_direct_image_pipeline(self):
        applied = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _repo, **_kwargs):
                return cls()

        recipe = {
            "offload_mode": "none",
            "device": "cuda:0",
            "denoiser_cache": "first_block",
            "cache_threshold": 0.08,
            "regional_compile": True,
            "compile_components": ["transformer"],
        }
        node = LoadPipeline("direct-image-runtime-recipe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
            patch(
                "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                side_effect=lambda pipeline, runtime_recipe: applied.update(
                    {"pipeline": pipeline, "recipe": runtime_recipe}
                )
                or {"cache": {"requested": runtime_recipe["denoiser_cache"]}},
            ),
        ):
            result = node.execute(
                model_id="org/direct-image-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                revision=CUSTOM_IMAGE_REVISION,
                execution_recipe=recipe,
                enable_vae_slicing=False,
                enable_vae_tiling=False,
            )

        self.assertIs(applied["pipeline"], result["pipeline"])
        self.assertEqual(applied["recipe"]["denoiser_cache"], "first_block")
        self.assertEqual(applied["recipe"]["cache_threshold"], 0.08)
        self.assertTrue(applied["recipe"]["regional_compile"])
        self.assertFalse(applied["recipe"]["vae_slicing"])
        self.assertFalse(applied["recipe"]["vae_tiling"])
        self.assertEqual(
            result["pipeline"]._modiff_runtime_config,
            {"cache": {"requested": "first_block"}},
        )

    def test_direct_image_defaults_still_apply_vae_memory_without_a_recipe_node(self):
        captured = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _repo, **_kwargs):
                return cls()

        node = LoadPipeline("direct-image-default-runtime")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
            patch(
                "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                side_effect=lambda _pipeline, runtime_recipe: captured.update(runtime_recipe) or {},
            ),
        ):
            node.execute(
                model_id="org/direct-image-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                revision=CUSTOM_IMAGE_REVISION,
                auto_offload=False,
                offload_mode="none",
                enable_vae_slicing=False,
                enable_vae_tiling=True,
            )

        self.assertFalse(captured["vae_slicing"])
        self.assertTrue(captured["vae_tiling"])

    def test_direct_device_map_streams_a_no_offload_pipeline_to_cuda_during_load(self):
        loaded = {}
        offload = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        node = LoadPipeline("direct-device-map-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch(
                "modules.DiffusersImage.main.apply_pipeline_offload",
                side_effect=lambda _pipeline, **kwargs: offload.update(kwargs),
            ),
        ):
            node.execute(
                model_id="org/native-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                revision=CUSTOM_IMAGE_REVISION,
                device="cuda:0",
                device_map="cuda",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["kwargs"]["device_map"], "cuda")
        self.assertEqual(offload["mode"], "none")
        self.assertEqual(offload["device"], "cuda:0")

    def test_direct_device_map_overrides_a_neutral_recipe_default(self):
        loaded = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, _repo, **kwargs):
                loaded.update(kwargs)
                return cls()

        node = LoadPipeline("direct-map-precedence-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            node.execute(
                model_id="org/native-model",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                revision=CUSTOM_IMAGE_REVISION,
                execution_recipe={"device_map": "none", "offload_mode": "none", "device": "cuda:0"},
                device_map="cuda",
            )

        self.assertEqual(loaded["device_map"], "cuda")

    def test_flux_redux_loader_composes_prior_with_app_cached_flux_base(self):
        loaded = []

        class FakePrior:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.append(("prior", repo, kwargs))
                return cls()

        class FakeBase:
            text_encoder = "clip"
            text_encoder_2 = "t5"
            tokenizer = "clip-tokenizer"
            tokenizer_2 = "t5-tokenizer"

            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.append(("base", repo, kwargs))
                return cls()

            def register_modules(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        node = LoadPipeline("redux-loader-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("diffusers.FluxPriorReduxPipeline", FakePrior),
            patch("diffusers.FluxPipeline", FakeBase),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            result = node.execute(
                model_id="black-forest-labs/FLUX.1-Redux-dev",
                pipeline_class="FluxReduxPipeline",
                mode="edit_image",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertIsInstance(result["pipeline"], FluxReduxPipelineBundle)
        self.assertEqual(loaded[0][0:2], ("base", FLUX_DEV_REPO))
        self.assertEqual(loaded[0][2]["revision"], "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21")
        self.assertEqual(loaded[1][0:2], ("prior", "black-forest-labs/FLUX.1-Redux-dev"))
        self.assertEqual(loaded[1][2]["revision"], catalog_revision("black-forest-labs/FLUX.1-Redux-dev"))
        self.assertEqual(loaded[1][2]["text_encoder"], "clip")
        self.assertEqual(loaded[1][2]["text_encoder_2"], "t5")
        self.assertEqual(loaded[1][2]["tokenizer"], "clip-tokenizer")
        self.assertEqual(loaded[1][2]["tokenizer_2"], "t5-tokenizer")
        self.assertTrue(loaded[0][2]["local_files_only"])
        self.assertTrue(loaded[1][2]["local_files_only"])
        self.assertIsNone(result["pipeline"].base.text_encoder)
        self.assertIsNone(result["pipeline"].base.text_encoder_2)

    def test_curated_image_loader_uses_only_its_catalog_pin(self):
        loaded = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.append((repo, kwargs))
                return cls()

        node = LoadPipeline("image-revision-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            node.execute(
                model_id="black-forest-labs/FLUX.1-schnell",
                pipeline_class="FluxPipeline",
                mode="text_to_image",
                auto_offload=False,
                offload_mode="none",
            )
            with self.assertRaisesRegex(ValueError, "must use its reviewed commit"):
                node.execute(
                    model_id="black-forest-labs/FLUX.1-schnell",
                    pipeline_class="FluxPipeline",
                    mode="text_to_image",
                    revision="main",
                    auto_offload=False,
                    offload_mode="none",
                )

        self.assertEqual(loaded[0][1]["revision"], "741f7c3ce8b383c54771c7003378a50191e9efe9")
        self.assertEqual(len(loaded), 1)

    def test_conditioned_loader_assembles_pinned_safetensors_component(self):
        calls = {}

        class FakeControlNet:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                calls["component"] = (repo, kwargs)
                return cls()

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                calls["pipeline"] = (repo, kwargs)
                return cls()

        node = LoadPipeline("sd15-controlnet-assembly")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                side_effect=lambda name: {
                    "ControlNetModel": FakeControlNet,
                    "StableDiffusionControlNetPipeline": FakePipeline,
                }[name],
            ),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
            patch(
                "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                return_value={},
            ),
        ):
            result = node.execute(
                model_id={"source": "hub", "value": SD15_BASE_REPO},
                revision=catalog_revision(SD15_BASE_REPO),
                pipeline_class="StableDiffusionControlNetPipeline",
                mode="control_image",
                conditioning_kind="controlnet",
                conditioning_model_id={"source": "hub", "value": SD15_CONTROLNET_CANNY_REPO},
                conditioning_revision=catalog_revision(SD15_CONTROLNET_CANNY_REPO),
                dtype="float32",
                device="cpu",
                auto_offload=False,
                offload_mode="none",
            )

        component_repo, component_kwargs = calls["component"]
        self.assertEqual(component_repo, SD15_CONTROLNET_CANNY_REPO)
        self.assertEqual(component_kwargs["revision"], catalog_revision(SD15_CONTROLNET_CANNY_REPO))
        self.assertTrue(component_kwargs["use_safetensors"])
        self.assertNotIn("trust_remote_code", component_kwargs)
        pipeline_repo, pipeline_kwargs = calls["pipeline"]
        self.assertEqual(pipeline_repo, SD15_BASE_REPO)
        self.assertEqual(pipeline_kwargs["revision"], catalog_revision(SD15_BASE_REPO))
        self.assertTrue(pipeline_kwargs["use_safetensors"])
        self.assertIsInstance(pipeline_kwargs["controlnet"], FakeControlNet)
        self.assertEqual(result["pipeline"]._modiff_conditioning_kind, "controlnet")
        self.assertEqual(
            result["pipeline"]._modiff_conditioning_revision,
            catalog_revision(SD15_CONTROLNET_CANNY_REPO),
        )

    def test_sdxl_conditioned_loader_uses_only_pinned_fp16_safetensors_variants(self):
        calls = {}

        class FakeControlNet:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                calls["component"] = (repo, kwargs)
                return cls()

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                calls["pipeline"] = (repo, kwargs)
                return cls()

        node = LoadPipeline("sdxl-controlnet-assembly")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                side_effect=lambda name: {
                    "ControlNetModel": FakeControlNet,
                    "StableDiffusionXLControlNetPipeline": FakePipeline,
                }[name],
            ),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
            patch(
                "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                return_value={},
            ),
        ):
            result = node.execute(
                model_id={"source": "hub", "value": SDXL_BASE_REPO},
                revision=catalog_revision(SDXL_BASE_REPO),
                pipeline_class="StableDiffusionXLControlNetPipeline",
                mode="control_image",
                conditioning_kind="controlnet",
                conditioning_model_id={"source": "hub", "value": SDXL_CONTROLNET_CANNY_REPO},
                conditioning_revision=catalog_revision(SDXL_CONTROLNET_CANNY_REPO),
                dtype="float16",
                device="cpu",
                auto_offload=False,
                offload_mode="none",
            )

        component_repo, component_kwargs = calls["component"]
        self.assertEqual(component_repo, SDXL_CONTROLNET_CANNY_REPO)
        self.assertEqual(component_kwargs["revision"], catalog_revision(SDXL_CONTROLNET_CANNY_REPO))
        self.assertTrue(component_kwargs["use_safetensors"])
        self.assertEqual(component_kwargs["variant"], "fp16")
        self.assertNotIn("trust_remote_code", component_kwargs)
        pipeline_repo, pipeline_kwargs = calls["pipeline"]
        self.assertEqual(pipeline_repo, SDXL_BASE_REPO)
        self.assertEqual(pipeline_kwargs["revision"], catalog_revision(SDXL_BASE_REPO))
        self.assertTrue(pipeline_kwargs["use_safetensors"])
        self.assertEqual(pipeline_kwargs["variant"], "fp16")
        self.assertIsInstance(pipeline_kwargs["controlnet"], FakeControlNet)
        self.assertEqual(result["pipeline"]._modiff_conditioning_kind, "controlnet")
        self.assertEqual(
            result["pipeline"]._modiff_conditioning_revision,
            catalog_revision(SDXL_CONTROLNET_CANNY_REPO),
        )

    def test_hunyuan_dit_controlnet_uses_exact_safe_assembly_and_bounded_canny_action(self):
        calls = {}

        class FakeControlNet:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                calls["component"] = (repo, kwargs)
                return cls()

        class HunyuanDiTControlNetPipeline:
            _execution_device = "cpu"

            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                calls["pipeline"] = (repo, kwargs)
                return cls()

            def __call__(
                self,
                prompt=None,
                control_image=None,
                controlnet_conditioning_scale=1.0,
                width=None,
                height=None,
                **kwargs,
            ):
                calls["generation"] = {
                    "prompt": prompt,
                    "control_image": control_image,
                    "controlnet_conditioning_scale": controlnet_conditioning_scale,
                    "width": width,
                    "height": height,
                    "kwargs": kwargs,
                }
                return SimpleNamespace(images=[Image.new("RGB", (width, height), "white")])

        loader = LoadPipeline("hunyuan-dit-controlnet-assembly")
        loader.progress = lambda *args, **kwargs: None
        loader.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                side_effect=lambda name: {
                    "HunyuanDiT2DControlNetModel": FakeControlNet,
                    "HunyuanDiTControlNetPipeline": HunyuanDiTControlNetPipeline,
                }[name],
            ),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
            patch(
                "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                return_value={},
            ),
        ):
            result = loader.execute(
                model_id={"source": "hub", "value": HUNYUAN_DIT_DISTILLED_REPO},
                revision=catalog_revision(HUNYUAN_DIT_DISTILLED_REPO),
                pipeline_class="HunyuanDiTControlNetPipeline",
                mode="control_image",
                conditioning_kind="controlnet",
                conditioning_model_id={"source": "hub", "value": HUNYUAN_DIT_CONTROLNET_CANNY_REPO},
                conditioning_revision=catalog_revision(HUNYUAN_DIT_CONTROLNET_CANNY_REPO),
                dtype="float16",
                device="cpu",
                auto_offload=True,
                offload_mode="model_cpu",
            )

        component_repo, component_kwargs = calls["component"]
        self.assertEqual(component_repo, HUNYUAN_DIT_CONTROLNET_CANNY_REPO)
        self.assertEqual(component_kwargs["revision"], catalog_revision(HUNYUAN_DIT_CONTROLNET_CANNY_REPO))
        self.assertTrue(component_kwargs["use_safetensors"])
        self.assertNotIn("trust_remote_code", component_kwargs)
        base_repo, base_kwargs = calls["pipeline"]
        self.assertEqual(base_repo, HUNYUAN_DIT_DISTILLED_REPO)
        self.assertEqual(base_kwargs["revision"], catalog_revision(HUNYUAN_DIT_DISTILLED_REPO))
        self.assertTrue(base_kwargs["use_safetensors"])
        self.assertIsInstance(base_kwargs["controlnet"], FakeControlNet)

        control_image = Image.new("RGB", (1024, 1024), "black")
        action = ControlGenerate("hunyuan-dit-controlnet-generate")
        generated = action.execute(
            pipeline=result["pipeline"],
            prompt="a bilingual controlled image",
            negative_prompt="artifact",
            control_image=control_image,
            width=1024,
            height=1024,
            num_inference_steps=50,
            guidance_scale=6,
            conditioning_scale=1,
            seed=42,
            output_type="pil",
        )
        self.assertEqual(generated["width_out"], 1024)
        self.assertEqual(generated["height_out"], 1024)
        self.assertIs(calls["generation"]["control_image"], control_image)
        self.assertEqual(calls["generation"]["controlnet_conditioning_scale"], 1)
        with self.assertRaisesRegex(ValueError, "between 1024 and 1024"):
            action.execute(
                pipeline=result["pipeline"],
                prompt="reject an unreviewed size",
                control_image=control_image,
                width=768,
                height=1024,
                num_inference_steps=50,
                guidance_scale=6,
                output_type="pil",
            )

    def test_sdxl_t2i_adapter_loader_and_action_use_pinned_generic_contract(self):
        calls = {}

        class FakeAdapter:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                calls["component"] = (repo, kwargs)
                return cls()

        class StableDiffusionXLAdapterPipeline:
            device = "cpu"

            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                calls["pipeline"] = (repo, kwargs)
                return cls()

            def __call__(
                self,
                prompt=None,
                image=None,
                adapter_conditioning_scale=1.0,
                width=None,
                height=None,
                **kwargs,
            ):
                calls["generation"] = {
                    "prompt": prompt,
                    "image": image,
                    "adapter_conditioning_scale": adapter_conditioning_scale,
                    "kwargs": kwargs,
                }
                return SimpleNamespace(images=[Image.new("RGB", (width, height), "white")])

        node = LoadPipeline("sdxl-t2i-adapter-assembly")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch(
                "modules.DiffusersImage.main.pipeline_class_from_name",
                side_effect=lambda name: {
                    "T2IAdapter": FakeAdapter,
                    "StableDiffusionXLAdapterPipeline": StableDiffusionXLAdapterPipeline,
                }[name],
            ),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
            patch(
                "modules.DiffusersRuntime.main.apply_execution_recipe_to_pipeline",
                return_value={},
            ),
        ):
            result = node.execute(
                model_id={"source": "hub", "value": SDXL_BASE_REPO},
                revision=catalog_revision(SDXL_BASE_REPO),
                pipeline_class="StableDiffusionXLAdapterPipeline",
                mode="control_image",
                conditioning_kind="t2i_adapter",
                conditioning_model_id={"source": "hub", "value": SDXL_T2I_ADAPTER_CANNY_REPO},
                conditioning_revision=catalog_revision(SDXL_T2I_ADAPTER_CANNY_REPO),
                dtype="float16",
                device="cpu",
                auto_offload=False,
                offload_mode="none",
            )

        component_repo, component_kwargs = calls["component"]
        self.assertEqual(component_repo, SDXL_T2I_ADAPTER_CANNY_REPO)
        self.assertEqual(component_kwargs["revision"], catalog_revision(SDXL_T2I_ADAPTER_CANNY_REPO))
        self.assertTrue(component_kwargs["use_safetensors"])
        self.assertEqual(component_kwargs["variant"], "fp16")
        self.assertNotIn("trust_remote_code", component_kwargs)
        pipeline_repo, pipeline_kwargs = calls["pipeline"]
        self.assertEqual(pipeline_repo, SDXL_BASE_REPO)
        self.assertEqual(pipeline_kwargs["revision"], catalog_revision(SDXL_BASE_REPO))
        self.assertTrue(pipeline_kwargs["use_safetensors"])
        self.assertEqual(pipeline_kwargs["variant"], "fp16")
        self.assertIsInstance(pipeline_kwargs["adapter"], FakeAdapter)
        self.assertEqual(result["pipeline"]._modiff_conditioning_kind, "t2i_adapter")

        control = Image.new("RGB", (32, 32), "black")
        generated = ControlGenerate("sdxl-t2i-adapter-generate").execute(
            pipeline=result["pipeline"],
            control_image=control,
            prompt="rights-safe edge fixture",
            width=32,
            height=32,
            num_inference_steps=30,
            guidance_scale=7.5,
            conditioning_scale=0.8,
        )
        self.assertEqual(generated["images"][0].size, (32, 32))
        self.assertIs(calls["generation"]["image"], control)
        self.assertEqual(calls["generation"]["adapter_conditioning_scale"], 0.8)

    def test_conditioned_control_action_uses_upstream_image_and_scale_parameters(self):
        calls = {}

        class StableDiffusionControlNetPipeline:
            device = "cpu"

            def __call__(
                self,
                prompt=None,
                image=None,
                controlnet_conditioning_scale=1.0,
                width=None,
                height=None,
                **kwargs,
            ):
                calls.update(
                    prompt=prompt,
                    image=image,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    kwargs=kwargs,
                )
                return SimpleNamespace(images=[Image.new("RGB", (width, height), "white")])

        pipeline = StableDiffusionControlNetPipeline()
        tag_test_image_pipeline(
            pipeline,
            "StableDiffusionControlNetPipeline",
            "control_image",
        )
        control = Image.new("RGB", (32, 32), "black")
        result = ControlGenerate("sd15-controlnet-generate").execute(
            pipeline=pipeline,
            control_image=control,
            prompt="rights-safe edge fixture",
            width=32,
            height=32,
            num_inference_steps=1,
            guidance_scale=1.0,
            conditioning_scale=0.65,
        )

        self.assertIs(calls["image"], control)
        self.assertEqual(calls["controlnet_conditioning_scale"], 0.65)
        self.assertNotIn("control_image", calls["kwargs"])
        self.assertEqual(result["width_out"], 32)
        self.assertEqual(result["height_out"], 32)

    def test_flux_redux_bundle_delegates_multiple_reference_fusion_to_diffusers(self):
        import torch

        calls = {}

        class PriorOutput:
            # FluxPriorReduxPipeline already reduces the reference batch to
            # one weighted conditioning sample.
            prompt_embeds = torch.tensor([[[3.0, 5.0]]])
            pooled_prompt_embeds = torch.tensor([[4.0, 6.0]])

        class FakePrior:
            def __call__(self, **kwargs):
                calls["prior"] = kwargs
                return PriorOutput()

        class FakeBase:
            device = "cpu"

            def __call__(self, **kwargs):
                calls["base"] = kwargs
                return type("Result", (), {"images": [Image.new("RGB", (32, 32), "white")]})()

        bundle = FluxReduxPipelineBundle(FakePrior(), FakeBase())
        references = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")]
        result = Edit("redux-edit-probe").execute(
            pipeline=bundle,
            image=references,
            prompt="combine material and silhouette",
            width=32,
            height=32,
            num_inference_steps=4,
            guidance_scale=2.5,
        )

        self.assertIs(calls["prior"]["image"], references)
        self.assertEqual(calls["prior"]["prompt"], "combine material and silhouette")
        self.assertEqual(calls["prior"]["prompt_embeds_scale"], [1.0, 1.0])
        self.assertEqual(calls["prior"]["pooled_prompt_embeds_scale"], [1.0, 1.0])
        torch.testing.assert_close(calls["base"]["prompt_embeds"], torch.tensor([[[3.0, 5.0]]]))
        torch.testing.assert_close(calls["base"]["pooled_prompt_embeds"], torch.tensor([[4.0, 6.0]]))
        self.assertEqual(calls["base"]["width"], 32)
        self.assertEqual(calls["base"]["height"], 32)
        self.assertEqual(result["images"][0].size, (32, 32))

    def test_flux_redux_bundle_scales_secondary_references_through_generic_conditioning(self):
        import torch

        calls = {}

        class PriorOutput:
            prompt_embeds = torch.tensor([[[1.0, 3.0]]])
            pooled_prompt_embeds = torch.tensor([[2.0, 4.0]])

        class FakePrior:
            def __call__(self, **kwargs):
                calls["prior"] = kwargs
                return PriorOutput()

        class FakeBase:
            device = "cpu"

            def __call__(self, **kwargs):
                return type("Result", (), {"images": [Image.new("RGB", (32, 32), "white")]})()

        bundle = FluxReduxPipelineBundle(FakePrior(), FakeBase())
        references = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")]
        Edit("redux-weight-probe").execute(
            pipeline=bundle,
            image=references,
            prompt="architecture with a restrained material cue",
            width=32,
            height=32,
            num_inference_steps=4,
            guidance_scale=2.5,
            reference_strength=0.2,
        )

        self.assertEqual(calls["prior"]["prompt_embeds_scale"], [1.0, 0.2])
        self.assertEqual(calls["prior"]["pooled_prompt_embeds_scale"], [1.0, 0.2])

    def test_flux_kontext_stitches_multiple_references_through_generic_edit(self):
        received = {}

        class FluxKontextPipeline:
            _execution_device = "cpu"

            def __call__(self, **kwargs):
                received.update(kwargs)
                return type("Result", (), {"images": [Image.new("RGB", (32, 32), "white")]})()

        references = [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (8, 16), "blue")]
        result = Edit("kontext-multi-probe").execute(
            pipeline=FluxKontextPipeline(),
            image=references,
            prompt="use first for identity and second for style",
            width=32,
            height=32,
            num_inference_steps=2,
        )

        self.assertIsInstance(received["image"], Image.Image)
        self.assertEqual(received["image"].size, (24, 16))
        self.assertEqual(received["image"].getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(received["image"].getpixel((23, 0)), (0, 0, 255))
        self.assertEqual(result["images"][0].size, (32, 32))

    def test_qwen_inpaint_uses_generic_guidance_and_crop_aliases(self):
        received = {}

        class FakeResult:
            images = [Image.new("RGB", (16, 16), "white")]

        class FakeQwenPipeline:
            _execution_device = "cpu"
            _modiff_image_pipeline_class = "QwenImageEditInpaintPipeline"
            _modiff_image_mode = "inpaint"

            def __call__(
                self,
                *,
                image,
                mask_image,
                prompt,
                num_inference_steps,
                generator,
                output_type,
                return_dict,
                true_cfg_scale,
                padding_mask_crop,
                strength,
            ):
                received.update(
                    true_cfg_scale=true_cfg_scale,
                    padding_mask_crop=padding_mask_crop,
                    strength=strength,
                )
                return FakeResult()

        node = Inpaint("qwen-generic-probe")
        result = node.execute(
            pipeline=tag_test_image_pipeline(FakeQwenPipeline(), "QwenImageEditInpaintPipeline", "inpaint"),
            image=Image.new("RGB", (16, 16), "black"),
            mask_image=Image.new("L", (16, 16), "white"),
            prompt="replace the object",
            num_inference_steps=2,
            guidance_scale=4.0,
            padding_mask_crop=128,
            strength=1.0,
            output_type="pil",
        )

        self.assertEqual(received, {"true_cfg_scale": 4.0, "padding_mask_crop": 128, "strength": 1.0})
        self.assertEqual(result["images"][0].size, (16, 16))

    def test_zero_padding_mask_crop_maps_to_diffusers_no_crop(self):
        received = {}

        class Pipeline:
            def __call__(self, padding_mask_crop=None, **kwargs):
                return None

        adapter = IMAGE_PIPELINE_ADAPTERS["QwenImageEditInpaintPipeline"]
        adapter.apply_generation_parameters(
            Pipeline(),
            {"padding_mask_crop": 0},
            received,
        )

        self.assertNotIn("padding_mask_crop", received)

    def test_qwen_generic_loader_uses_the_official_default_repository(self):
        loaded = []

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **_kwargs):
                loaded.append(repo)
                return cls()

        node = LoadPipeline("qwen-loader-probe")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersImage.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersImage.main.apply_pipeline_offload"),
        ):
            node.execute(
                pipeline_class="QwenImageEditInpaintPipeline",
                mode="inpaint",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded, ["Qwen/Qwen-Image-Edit"])

    def test_generic_inpaint_preserves_every_black_mask_pixel(self):
        self.assertEqual(Inpaint.params["output_type"]["options"], ["pil"])

        class FakeResult:
            images = [Image.new("RGB", (4, 2), (200, 210, 220))]

        class FakePipeline:
            _execution_device = "cpu"
            _modiff_image_pipeline_class = "FluxFillPipeline"
            _modiff_image_mode = "inpaint"

            def __call__(self, **_kwargs):
                return FakeResult()

        source = Image.new("RGB", (4, 2), (10, 20, 30))
        mask = Image.new("L", (4, 2), 0)
        for x in (2, 3):
            for y in (0, 1):
                mask.putpixel((x, y), 255)

        result = Inpaint("mask-contract-probe").execute(
            pipeline=tag_test_image_pipeline(FakePipeline(), "FluxFillPipeline", "inpaint"),
            image=source,
            mask_image=mask,
            prompt="replace",
            num_inference_steps=1,
            output_type="pil",
        )

        self.assertEqual(result["images"][0].getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(result["images"][0].getpixel((3, 1)), (200, 210, 220))

    def test_generic_step_callback_propagates_interrupt_to_pipeline(self):
        class FakePipeline:
            _interrupt = False
            _num_timesteps = 4

            def __call__(self, *, callback_on_step_end=None, callback_on_step_end_tensor_inputs=None):
                pass

        node = Inpaint("interrupt-probe")
        node._interrupt = True
        progress = []
        node.progress = lambda *args, **kwargs: progress.append((args, kwargs))
        pipeline = FakePipeline()
        call_kwargs = {}

        add_progress_callback(node, pipeline, call_kwargs, 4)
        self.assertEqual(progress[0][0], (0,))
        self.assertEqual(progress[0][1]["phase"], "denoising")
        self.assertEqual(progress[0][1]["current_step"], 0)
        self.assertEqual(progress[0][1]["total_steps"], 4)
        with self.assertRaisesRegex(InterruptedError, "interrupted by the user"):
            call_kwargs["callback_on_step_end"](pipeline, 0, 1, {})

        self.assertTrue(pipeline._interrupt)

    def test_generic_execution_exposes_active_pipeline_during_inference(self):
        node = Inpaint("active-pipeline-probe")
        observed = []

        class FakeResult:
            images = [Image.new("RGB", (16, 16), "white")]

        class FakePipeline:
            _execution_device = "cpu"
            _modiff_image_pipeline_class = "FluxFillPipeline"
            _modiff_image_mode = "inpaint"

            def __call__(self, **_kwargs):
                observed.append(node._active_pipeline is self)
                return FakeResult()

        node.execute(
            pipeline=tag_test_image_pipeline(FakePipeline(), "FluxFillPipeline", "inpaint"),
            image=Image.new("RGB", (16, 16), "black"),
            mask_image=Image.new("L", (16, 16), "white"),
            prompt="replace",
            num_inference_steps=1,
        )

        self.assertEqual(observed, [True])
        self.assertIsNone(node._active_pipeline)

    def test_adapter_uses_only_the_app_managed_cached_weight(self):
        calls = []

        class FakePipeline:
            def load_lora_weights(self, path, **kwargs):
                calls.append((path, kwargs))

        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "hub"
            cache_root.mkdir()
            cached_weight = cache_root / "adapter.safetensors"
            cached_weight.write_bytes(b"cached adapter")
            expected = hashlib.sha256(cached_weight.read_bytes()).hexdigest()
            with (
                patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(cache_root)}),
                patch(
                    "modules.DiffusersImage.main.cached_file_path",
                    return_value=str(cached_weight),
                ) as cached,
            ):
                LoadAdapter("adapter-probe").execute(
                    pipeline=FakePipeline(),
                    adapter_path={"source": "hub", "value": "unit/adapter"},
                    weight_name="adapter.safetensors",
                    revision=HUB_ADAPTER_REVISION,
                    expected_sha256=expected,
                    adapter_name="gallery",
                    scale=0.8,
                )

        cached.assert_called_once_with("unit/adapter", "adapter.safetensors", revision=HUB_ADAPTER_REVISION)
        self.assertEqual(calls[0][0], str(cached_weight.parent))
        self.assertEqual(calls[0][1]["weight_name"], "adapter.safetensors")
        self.assertTrue(calls[0][1]["use_safetensors"])

    def test_adapter_missing_from_app_cache_fails_before_pipeline_load(self):
        class FakePipeline:
            def load_lora_weights(self, *_args, **_kwargs):
                raise AssertionError("must not download or load")

        with patch("modules.DiffusersImage.main.cached_file_path", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Model Manager"):
                LoadAdapter("missing-adapter-probe").execute(
                    pipeline=FakePipeline(),
                    adapter_path={"source": "hub", "value": "unit/adapter"},
                    weight_name="adapter.safetensors",
                    revision=HUB_ADAPTER_REVISION,
                    expected_sha256="0" * 64,
                )

    def test_adapter_rejects_nonliteral_safetensors_suffixes_before_cache_or_pipeline_mutation(self):
        class FakePipeline:
            def load_lora_weights(self, *_args, **_kwargs):
                raise AssertionError("adapter loading must not run")

        with patch("modules.DiffusersImage.main.cached_file_path") as cached:
            for weight_name in ("adapter.bin", "adapter.SAFETENSORS", "adapter.SafeTensors"):
                with (
                    self.subTest(weight_name=weight_name),
                    self.assertRaisesRegex(ValueError, "contained \\.safetensors"),
                ):
                    LoadAdapter("unsafe-hub-adapter").execute(
                        pipeline=FakePipeline(),
                        adapter_path={"source": "hub", "value": "unit/adapter"},
                        weight_name=weight_name,
                        revision=HUB_ADAPTER_REVISION,
                        expected_sha256="0" * 64,
                    )
        cached.assert_not_called()

    def test_adapter_source_variants_cannot_bypass_the_app_cache_boundary(self):
        class FakePipeline:
            def __init__(self):
                self.calls = []

            def load_lora_weights(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        for source in ("Hub", "HUB"):
            with self.subTest(canonical_source=source):
                pipeline = FakePipeline()
                with (
                    patch("modules.DiffusersImage.main.cached_file_path", return_value=False) as cached,
                    self.assertRaisesRegex(FileNotFoundError, "Model Manager"),
                ):
                    LoadAdapter("canonical-adapter-source").execute(
                        pipeline=pipeline,
                        adapter_path={"source": source, "value": "unit/adapter"},
                        weight_name="adapter.safetensors",
                        revision=HUB_ADAPTER_REVISION,
                        expected_sha256="0" * 64,
                    )
                cached.assert_called_once_with("unit/adapter", "adapter.safetensors", revision=HUB_ADAPTER_REVISION)
                self.assertEqual(pipeline.calls, [])

        invalid = (" hub ", "remote", None, 7)
        for source in invalid:
            with self.subTest(invalid_source=source):
                pipeline = FakePipeline()
                with self.assertRaisesRegex(ValueError, "source must be exactly hub or local"):
                    LoadAdapter("invalid-adapter-source").execute(
                        pipeline=pipeline,
                        adapter_path={"source": source, "value": "unit/adapter"},
                        weight_name="adapter.safetensors",
                        revision=HUB_ADAPTER_REVISION,
                        expected_sha256="0" * 64,
                    )
                self.assertEqual(pipeline.calls, [])

    def test_adapter_truthy_missing_cache_entry_fails_before_pipeline_mutation(self):
        events = []

        class FakePipeline:
            def unload_lora_weights(self):
                events.append("unload")

            def load_lora_weights(self, *_args, **_kwargs):
                events.append("load")

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.safetensors"
            with (
                patch("modules.DiffusersImage.main.cached_file_path", return_value=str(missing)),
                self.assertRaisesRegex(FileNotFoundError, "cache entry does not exist"),
            ):
                LoadAdapter("truthy-missing-adapter").execute(
                    pipeline=FakePipeline(),
                    adapter_path={"source": "hub", "value": "unit/adapter"},
                    weight_name=missing.name,
                    revision=HUB_ADAPTER_REVISION,
                    expected_sha256="0" * 64,
                )
        self.assertEqual(events, [])

    def test_managed_cache_resolution_accepts_contained_files_and_rejects_escapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "hub"
            cache_root.mkdir()
            contained = cache_root / "blob.safetensors"
            contained.write_bytes(b"contained")
            outside = root / "outside.safetensors"
            outside.write_bytes(b"outside")
            with patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(cache_root)}):
                self.assertEqual(resolve_managed_hf_cache_file(contained), contained.resolve())
                with self.assertRaisesRegex(ValueError, "outside the managed cache root"):
                    resolve_managed_hf_cache_file(outside)

    def test_managed_cache_resolution_preserves_snapshot_to_blob_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "hub"
            repository_root = cache_root / "models--unit--adapter"
            blob = repository_root / "blobs" / "abc"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"blob")
            snapshot_file = repository_root / "snapshots" / HUB_ADAPTER_REVISION / "adapter.safetensors"
            snapshot_file.parent.mkdir(parents=True)
            try:
                os.symlink(blob, snapshot_file)
            except OSError as error:
                self.skipTest(f"Symlinks are unavailable on this Windows runtime: {error}")
            with patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(cache_root)}):
                self.assertEqual(resolve_managed_hf_cache_file(snapshot_file), blob.resolve())

    def test_hub_adapter_load_preserves_safe_snapshot_alias_for_extensionless_blob(self):
        calls = []

        class FakePipeline:
            def load_lora_weights(self, path, **kwargs):
                calls.append((path, kwargs))

        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "hub"
            repository_root = cache_root / "models--unit--adapter"
            blob = repository_root / "blobs" / "abc"
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"extensionless safetensors blob probe")
            snapshot_file = repository_root / "snapshots" / HUB_ADAPTER_REVISION / "adapter.safetensors"
            snapshot_file.parent.mkdir(parents=True)
            try:
                os.symlink(blob, snapshot_file)
            except OSError as error:
                self.skipTest(f"Symlinks are unavailable on this Windows runtime: {error}")

            with (
                patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(cache_root)}),
                patch("modules.DiffusersImage.main.cached_file_path", return_value=str(snapshot_file)),
            ):
                LoadAdapter("snapshot-alias-adapter").execute(
                    pipeline=FakePipeline(),
                    adapter_path={"source": "hub", "value": "unit/adapter"},
                    weight_name="adapter.safetensors",
                    revision=HUB_ADAPTER_REVISION,
                    expected_sha256=hashlib.sha256(blob.read_bytes()).hexdigest(),
                )

        self.assertEqual(calls[0][0], str(snapshot_file.parent))
        self.assertEqual(calls[0][1]["weight_name"], "adapter.safetensors")
        self.assertTrue(calls[0][1]["use_safetensors"])

    def test_hub_adapter_cache_escape_fails_before_pipeline_mutation(self):
        events = []

        class FakePipeline:
            def unload_lora_weights(self):
                events.append("unload")

            def load_lora_weights(self, *_args, **_kwargs):
                events.append("load")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "hub"
            cache_root.mkdir()
            outside = root / "outside.safetensors"
            outside.write_bytes(b"outside")
            with (
                patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(cache_root)}),
                patch(
                    "modules.DiffusersImage.main.cached_file_path",
                    return_value=str(outside),
                ),
                self.assertRaisesRegex(FileNotFoundError, "cache entry does not exist"),
            ):
                LoadAdapter("escaped-hub-adapter").execute(
                    pipeline=FakePipeline(),
                    adapter_path={"source": "hub", "value": "unit/adapter"},
                    weight_name=outside.name,
                    revision=HUB_ADAPTER_REVISION,
                    expected_sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
                )
        self.assertEqual(events, [])

    def test_hub_label_cannot_turn_an_existing_local_path_into_a_repository(self):
        class FakePipeline:
            def load_lora_weights(self, *_args, **_kwargs):
                raise AssertionError("adapter loading must not run")

        with self.assertRaisesRegex(ValueError, "existing local filesystem target"):
            LoadAdapter("hub-local-confusion").execute(
                pipeline=FakePipeline(),
                adapter_path={"source": "hub", "value": "modules/DiffusersImage"},
                weight_name="adapter.safetensors",
                revision=HUB_ADAPTER_REVISION,
                expected_sha256="0" * 64,
            )

    def test_adapter_facade_rejects_raw_nodebase_coercion_before_cache_or_mutation(self):
        class FakePipeline:
            def load_lora_weights(self, *_args, **_kwargs):
                raise AssertionError("adapter loading must not run")

        base = {
            "pipeline": FakePipeline(),
            "adapter_path": {"source": "hub", "value": "unit/adapter"},
            "weight_name": "adapter.safetensors",
            "revision": HUB_ADAPTER_REVISION,
            "expected_sha256": "0" * 64,
        }
        invalid = (
            {"scale": False},
            {"scale": []},
            {"replace_existing": "false"},
            {"revision": False},
            {"expected_sha256": []},
            {"weight_name": 0},
            {"adapter_path": False},
        )
        with patch("modules.DiffusersImage.main.cached_file_path") as cached:
            for index, override in enumerate(invalid):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    LoadAdapter(f"raw-adapter-{index}")(**{**base, **override})
        cached.assert_not_called()

    def test_local_adapter_requires_an_existing_file_or_contained_directory_weight(self):
        calls = []

        class FakePipeline:
            def load_lora_weights(self, path, **kwargs):
                calls.append((path, kwargs))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "direct.safetensors"
            direct.write_bytes(b"direct")
            nested = root / "weights"
            nested.mkdir()
            nested_weight = nested / "nested.safetensors"
            nested_weight.write_bytes(b"nested")
            selected = root / "selected"
            selected.mkdir()
            (root / "outside.safetensors").write_bytes(b"outside")

            LoadAdapter("local-file-adapter").execute(
                pipeline=FakePipeline(),
                adapter_path={"source": "LOCAL", "value": str(direct)},
                adapter_name="direct",
            )
            LoadAdapter("local-folder-adapter").execute(
                pipeline=FakePipeline(),
                adapter_path={"source": "local", "value": str(root)},
                weight_name="weights/nested.safetensors",
                adapter_name="nested",
            )
            self.assertEqual(calls[0][0], str(root.resolve()))
            self.assertEqual(calls[0][1]["weight_name"], direct.name)
            self.assertTrue(calls[0][1]["use_safetensors"])
            self.assertEqual(calls[1][0], str(nested.resolve()))
            self.assertEqual(calls[1][1]["weight_name"], nested_weight.name)
            self.assertTrue(calls[1][1]["use_safetensors"])

            before = len(calls)
            for value, weight_name in (
                (str(root / "missing.safetensors"), ""),
                ("unit/adapter", "adapter.safetensors"),
                (str(selected), "../outside.safetensors"),
            ):
                with (
                    self.subTest(value=value, weight_name=weight_name),
                    self.assertRaises((FileNotFoundError, ValueError)),
                ):
                    LoadAdapter("invalid-local-adapter").execute(
                        pipeline=FakePipeline(),
                        adapter_path={"source": "local", "value": value},
                        weight_name=weight_name,
                    )
            self.assertEqual(len(calls), before)

            for unsafe_name in ("unsafe.bin", "unsafe.SAFETENSORS", "unsafe.SafeTensors"):
                unsafe = root / unsafe_name
                unsafe.write_bytes(b"unsafe format probe")
                with (
                    self.subTest(unsafe_name=unsafe_name),
                    self.assertRaisesRegex(ValueError, "lowercase \\.safetensors"),
                ):
                    LoadAdapter("unsafe-local-adapter").execute(
                        pipeline=FakePipeline(),
                        adapter_path={"source": "local", "value": str(unsafe)},
                    )
            self.assertEqual(len(calls), before)

    def test_adapter_verifies_pinned_hash_and_replaces_previous_pipeline_adapters(self):
        events = []

        class FakePipeline:
            def unload_lora_weights(self):
                events.append("unload")

            def load_lora_weights(self, path, **kwargs):
                events.append(("load", path, kwargs))

            def set_adapters(self, names, weights):
                events.append(("activate", names, weights))

        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "hub"
            cache_root.mkdir()
            adapter_file = cache_root / "adapter.safetensors"
            adapter_file.write_bytes(b"pinned adapter bytes")
            expected = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
            with (
                patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(cache_root)}),
                patch(
                    "modules.DiffusersImage.main.cached_file_path",
                    return_value=str(adapter_file),
                ),
            ):
                LoadAdapter("verified-adapter-probe").execute(
                    pipeline=FakePipeline(),
                    adapter_path={"source": "hub", "value": "unit/adapter"},
                    weight_name=adapter_file.name,
                    revision=HUB_ADAPTER_REVISION,
                    expected_sha256=expected,
                    adapter_name="theme",
                    scale=0.75,
                )

        self.assertEqual(events[0], "unload")
        self.assertEqual(events[1][0], "load")
        self.assertEqual(events[2], ("activate", ["theme"], [0.75]))

    def test_chained_adapters_preserve_prior_adapter_names_and_scales(self):
        events = []

        class FakePipeline:
            def unload_lora_weights(self):
                events.append("unload")

            def load_lora_weights(self, path, **kwargs):
                events.append(("load", path, kwargs))

            def set_adapters(self, names, weights):
                events.append(("activate", names, weights))

        pipeline = FakePipeline()
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "hub"
            cache_root.mkdir()
            first = cache_root / "first.safetensors"
            second = cache_root / "second.safetensors"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            with (
                patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(cache_root)}),
                patch(
                    "modules.DiffusersImage.main.cached_file_path",
                    side_effect=[str(first), str(second)],
                ),
            ):
                LoadAdapter("first-adapter").execute(
                    pipeline=pipeline,
                    adapter_path={"source": "hub", "value": "unit/first"},
                    weight_name=first.name,
                    revision=HUB_ADAPTER_REVISION,
                    expected_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),
                    adapter_name="cinematic",
                    scale=0.8,
                    replace_existing=True,
                )
                LoadAdapter("second-adapter").execute(
                    pipeline=pipeline,
                    adapter_path={"source": "hub", "value": "unit/second"},
                    weight_name=second.name,
                    revision=HUB_ADAPTER_REVISION,
                    expected_sha256=hashlib.sha256(second.read_bytes()).hexdigest(),
                    adapter_name="render_3d",
                    scale=0.18,
                    replace_existing=False,
                )

        self.assertEqual(events.count("unload"), 1)
        self.assertEqual(events[-1], ("activate", ["cinematic", "render_3d"], [0.8, 0.18]))
        self.assertEqual(pipeline._modiff_adapter_scales, {"cinematic": 0.8, "render_3d": 0.18})

    def test_adapter_scale_zero_remains_zero(self):
        activations = []

        class FakePipeline:
            def load_lora_weights(self, *_args, **_kwargs):
                pass

            def set_adapters(self, names, weights):
                activations.append((names, weights))

        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "hub"
            cache_root.mkdir()
            cached_weight = cache_root / "adapter.safetensors"
            cached_weight.write_bytes(b"cached adapter")
            with (
                patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(cache_root)}),
                patch(
                    "modules.DiffusersImage.main.cached_file_path",
                    return_value=str(cached_weight),
                ),
            ):
                LoadAdapter("zero-scale-adapter").execute(
                    pipeline=FakePipeline(),
                    adapter_path={"source": "hub", "value": "unit/adapter"},
                    weight_name="adapter.safetensors",
                    revision=HUB_ADAPTER_REVISION,
                    expected_sha256=hashlib.sha256(cached_weight.read_bytes()).hexdigest(),
                    adapter_name="optional",
                    scale=0,
                    replace_existing=False,
                )

        self.assertEqual(activations, [(["optional"], [0.0])])

    def test_adapter_hash_mismatch_does_not_mutate_pipeline(self):
        class FakePipeline:
            def unload_lora_weights(self):
                raise AssertionError("hash validation must happen before pipeline mutation")

            def load_lora_weights(self, *_args, **_kwargs):
                raise AssertionError("hash validation must happen before adapter loading")

        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory) / "hub"
            cache_root.mkdir()
            adapter_file = cache_root / "adapter.safetensors"
            adapter_file.write_bytes(b"unexpected bytes")
            with (
                patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(cache_root)}),
                patch(
                    "modules.DiffusersImage.main.cached_file_path",
                    return_value=str(adapter_file),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "pinned SHA-256"):
                    LoadAdapter("invalid-adapter-probe").execute(
                        pipeline=FakePipeline(),
                        adapter_path={"source": "hub", "value": "unit/adapter"},
                        weight_name=adapter_file.name,
                        revision=HUB_ADAPTER_REVISION,
                        expected_sha256="0" * 64,
                    )


class FakeRequest:
    match_info = {}


class DiffusersImageNodeRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_nodes_route_exposes_flattened_facade_contracts(self):
        server = WebServer(module_registry.MODULE_MAP)
        response = await server.nodes(FakeRequest())
        nodes = json.loads(response.text)["nodes"]

        for action in ("Edit", "Inpaint", "ControlGenerate"):
            with self.subTest(node=action):
                key = f"modules.DiffusersImage.{action}"
                self.assertIn(key, nodes)
                self.assertTrue(nodes[key]["resizable"])
                self.assertEqual(nodes[key]["params"]["images"]["display"], "output")


if __name__ == "__main__":
    unittest.main()
