"""Depth adapters retain native values and the existing normalized map contract."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from utils.torch_utils import DEFAULT_DEVICE

from modules.HuggingFaceTransformers.depth import depth_geometry, depth_convention, predict_depth
from modules.HuggingFaceTransformers.main import LoadDepthEstimationModel, PredictDepth, _sealed_receipt, SECURITY_CONTRACT


class DPTImageProcessor:
    size = {"height": 28, "width": 28}
    ensure_multiple_of = 14
    keep_aspect_ratio = True
    do_pad = False
    do_resize = True

    def __call__(self, *, images, size, return_tensors):
        _, shape = depth_geometry(self, images[0], size["height"])
        return {"pixel_values": torch.ones(shape)}

    def post_process_depth_estimation(self, output, target_sizes):
        value = output.predicted_depth
        if target_sizes is not None:
            value = torch.nn.functional.interpolate(value[:, None], size=target_sizes[0], mode="bicubic", align_corners=False)[:, 0]
        return [{"predicted_depth": value[0]}]


class DepthModel:
    def __init__(self, kind="relative"):
        self.config = SimpleNamespace(model_type="depth_anything", depth_estimation_type=kind, max_depth=80)
        self.inputs = []

    def forward(self, pixel_values):
        self.inputs.append(pixel_values)
        h, w = pixel_values.shape[-2:]
        return SimpleNamespace(predicted_depth=torch.linspace(2, 22, h * w).reshape(1, h, w))

    __call__ = forward


def handle(model=None, processor=None):
    model = model or DepthModel()
    processor = processor or DPTImageProcessor()
    receipt = _sealed_receipt({
        "schemaVersion": 1, "library": "transformers", "task": "depth-estimation",
        "source": {"kind": "hub", "repository": "owner/model", "revision": "a" * 40},
        "security": SECURITY_CONTRACT,
        "loader": {"preprocessorAutoClass": "AutoImageProcessor", "modelAutoClass": "AutoModelForDepthEstimation"},
        "runtime": {"dtype": "float32", "device": "cpu"},
    })
    return {"schemaVersion": 1, "kind": "transformers-depth-estimation", "model": model, "preprocessor": processor, "receipt": receipt}


@pytest.mark.parametrize("kind,near_is_larger", [("relative", True), ("metric", False)])
def test_native_depth_is_not_replaced_by_its_normalized_preview(kind, near_is_larger):
    selected = handle(DepthModel(kind))
    result = PredictDepth.execute(None, pipeline=selected, image=Image.new("RGB", (28, 28)))
    native = result["native_depth"]
    assert native.shape == (28, 28)
    assert float(native.min()) == 2 and float(native.max()) == 22
    prediction = result["prediction_map"]
    assert prediction["shape"] == [1, 28, 28, 1]
    assert prediction["nearValue"] == 0 and prediction["farValue"] == 1
    assert float(prediction["prediction"][0, 0, 0, 0]) == (1 if near_is_larger else 0)
    assert result["preview_images"][0].size == (28, 28)
    assert result["result"]["modelReceipt"] == selected["receipt"]


def test_output_resolution_uses_official_postprocessing_and_keeps_model_reusable():
    selected = handle()
    for size, match in [((56, 28), True), ((112, 56), False)]:
        output = PredictDepth.execute(None, pipeline=selected, image=Image.new("RGB", size), match_input_resolution=match)
        assert output["native_depth"].shape == ((size[1], size[0]) if match else (28, 56))
    assert len(selected["model"].inputs) == 2


def test_invalid_geometry_is_rejected_before_preprocessing_or_forward():
    processor = DPTImageProcessor()
    model = DepthModel()
    with pytest.raises(ValueError, match="geometry"):
        predict_depth(model, processor, Image.new("RGB", (4096, 1)), receipt=handle()["receipt"], resolution=2048, match_input=True, convention="model_default")
    assert not model.inputs


@pytest.mark.parametrize("image", [None, [], [Image.new("RGB", (28, 28))] * 2])
def test_exactly_one_source_image_is_required(image):
    with pytest.raises(ValueError):
        PredictDepth.execute(None, pipeline=handle(), image=image)


def test_nonfinite_output_is_rejected():
    model = DepthModel()
    model.forward = Mock(return_value=SimpleNamespace(predicted_depth=torch.full((1, 28, 28), float("nan"))))
    class BadModel(DepthModel):
        def __call__(self, **kwargs):
            return model.forward(**kwargs)
    with pytest.raises(RuntimeError, match="nonfinite"):
        PredictDepth.execute(None, pipeline=handle(BadModel()), image=Image.new("RGB", (28, 28)))


def test_unreviewed_depth_polarity_requires_an_explicit_choice():
    model = SimpleNamespace(config=SimpleNamespace(model_type="another_model"))
    with pytest.raises(ValueError, match="convention"):
        depth_convention(model, "model_default")
    assert depth_convention(model, "far_is_larger") == ("far_is_larger", "operator_selected")


def test_loader_uses_existing_local_only_model_loading_boundary():
    with patch("modules.HuggingFaceTransformers.main._load_model", return_value=("model", "receipt")) as load:
        result = LoadDepthEstimationModel.execute(None, model_id="owner/model", revision="a" * 40, dtype="float32", device="cpu")
    assert result == {"pipeline": "model", "receipt": "receipt"}
    assert load.call_args.kwargs == {"selection_value": "owner/model", "revision_value": "a" * 40, "dtype_value": "float32", "device_value": "cpu", "task": "depth-estimation"}


def test_curated_models_share_generic_connected_nodes_and_exact_artifact_pins():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    contracts = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})[0]
    for profile in ("depth-anything-v2-small:direct", "depth-anything-v2-metric-outdoor-small:direct"):
        starter = resolve_operation_starter(MODULE_MAP, contracts, {"pipelineClass": "AutoModelForDepthEstimation", "task": "depth_estimation", "executionProfileId": profile})
        loader = next(n for n in starter["nodes"] if n["action"] == "LoadDepthEstimationModel")
        assert len(loader["values"]["revision"]) == 40
        assert any(n["action"] == "PredictDepth" for n in starter["nodes"])
        assert any(e["sourceHandle"] == "pipeline" and e["targetHandle"] == "pipeline" for e in starter["edges"])


def test_official_auto_loader_preserves_immutable_local_only_safetensors_flags():
    model = DepthModel()
    model.to = Mock()
    model.eval = Mock()
    processor = DPTImageProcessor()
    runtime = SimpleNamespace(
        __version__="5.14.1",
        AutoImageProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        AutoModelForDepthEstimation=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    with patch.dict("sys.modules", {"transformers": runtime}):
        loaded = LoadDepthEstimationModel.execute(None, model_id="owner/model", revision="a" * 40, dtype="float32", device=DEFAULT_DEVICE)
    common = {"revision": "a" * 40, "local_files_only": True, "trust_remote_code": False}
    runtime.AutoImageProcessor.from_pretrained.assert_called_once_with("owner/model", **common)
    runtime.AutoModelForDepthEstimation.from_pretrained.assert_called_once_with(
        "owner/model", **common, dtype=torch.float32, use_safetensors=True, weights_only=True, low_cpu_mem_usage=True,
    )
    assert loaded["pipeline"]["model"] is model
    assert loaded["receipt"]["task"] == "depth-estimation"


@pytest.mark.parametrize("revision", ["main", "", "a" * 39])
def test_depth_loader_rejects_mutable_identity_before_loading(revision):
    runtime = SimpleNamespace(AutoImageProcessor=Mock(), AutoModelForDepthEstimation=Mock())
    with patch.dict("sys.modules", {"transformers": runtime}), pytest.raises(ValueError, match="commit"):
        LoadDepthEstimationModel.execute(None, model_id="owner/model", revision=revision, dtype="float32", device="cpu")
    runtime.AutoImageProcessor.assert_not_called()
    runtime.AutoModelForDepthEstimation.assert_not_called()


def test_depth_controls_are_in_the_bounded_consumed_input_receipt():
    from modiff.execution_input_provenance import capture_generation_inputs
    record = capture_generation_inputs("depth", {"module": "modules.HuggingFaceTransformers", "action": "PredictDepth"}, {
        "processing_resolution": 0, "match_input_resolution": False, "depth_convention": "model_default",
        "pipeline": object(), "image": Image.new("RGB", (28, 28)),
    })
    assert record["fields"] == {
        "processing_resolution": {"source": "literal", "value": 0},
        "match_input_resolution": {"source": "literal", "value": False},
        "depth_convention": {"source": "literal", "value": "model_default"},
    }
