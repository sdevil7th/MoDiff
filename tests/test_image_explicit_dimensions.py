"""Keep spatial defaults separate from the generic image action's valid range."""

import importlib.util
import hashlib
import inspect
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.DiffusersImage.main import (
    _compatible_image_contract,
    _tag_image_pipeline,
    catalog_revision,
    get_image_pipeline_adapter,
    image_pipeline_contract,
    preflight_image_action,
)


def test_saved_ernie_fixed_size_presentation_keeps_the_current_contract_boundary():
    adapter = get_image_pipeline_adapter('ErnieImagePipeline')
    current = image_pipeline_contract(adapter, 'text_to_image')
    saved = deepcopy(current)
    for field in ('width', 'height'):
        saved['fieldParams'][field].update(min=1024, max=1024)
    assert _compatible_image_contract(saved, current)
    saved['fieldParams']['width']['step'] = 16
    assert not _compatible_image_contract(saved, current)


# Read the real source graphs even after their metadata is refreshed. The saved
# signals below reconstruct only the former bounds and match hashes from the
# published pre-refresh graphs before they exercise saved-file compatibility.
_REVIEWED_SAVED_SIGNAL_HASHES = {
    "studio/glm-image-pipeline/text-to-image.json": "a612a4eecf846913c931bc3c3186973da003c7c9d0a901271917780ec335cf39",
    "studio/hunyuan-di-t-control-net-pipeline/control-image.json": "5c79b923d093204f72dc530de4ca066907239ee35d5ee4223f345f854a2e33ac",
    "studio/hunyuan-di-tpag-pipeline/text-to-image.json": "080d434590d833d25adc0655ca8561630c35b3432e70f5ad00b72a190f7645c3",
    "studio/hunyuan-di-t-pipeline/text-to-image.json": "0be0bfddcad69291d60d104bb772dd60b14340c848317fcd2e6ab1fb5117850f",
    "studio/joy-image-edit-pipeline/edit-image.json": "7fc3d51b88d48543601deaf51538afdcdf9466a381e18bc49decf17320a35865",
    "studio/joy-image-edit-pipeline/text-to-image.json": "de22e030e4bd9152000bc87b856ae3f5978e27b2a62a48391fdbd6982f5e57e9",
    "studio/joy-image-edit-plus-pipeline/edit-image.json": "1433f2ef550603682e24fd4b20d9e38589eee6621e0c36ef5f5e643703a6cf5d",
    "studio/joy-image-edit-plus-pipeline/multi-image-reference-edit.json": "1479ced82586ffefda5add89b42392f24a6eb0fe9aa248527f570f8988d6887b",
    "studio/kandinsky3-pipeline/edit-image.json": "c66c52fcfc639fef8c01df8cf268a08dad8668eff55afabc80e7618572ef089c",
    "studio/kandinsky3-pipeline/text-to-image.json": "41556e4eb5fd80a8234ea020c6c768dcb84f8bf4b86dc38b8d5d0e161d82f706",
}


@pytest.mark.parametrize("path,expected_digest", _REVIEWED_SAVED_SIGNAL_HASHES.items(), ids=_REVIEWED_SAVED_SIGNAL_HASHES.keys())
def test_reviewed_saved_image_presentation_publishes_live_bounds(path, expected_digest, monkeypatch):
    from modules.DiffusersImage import main as image_module

    graph_path = Path(__file__).resolve().parents[1] / "data" / "graphs" / path
    graph_bytes = graph_path.read_bytes()
    graph = json.loads(graph_bytes)
    [node] = [item["data"] for item in graph["nodes"] if "image_contract" in item["data"].get("params", {})]
    params = node["params"]
    saved = deepcopy(params["image_contract"]["value"])
    adapter = get_image_pipeline_adapter(saved["pipelineClass"])
    current = image_pipeline_contract(adapter, saved["mode"])
    if saved["pipelineClass"].startswith("JoyImageEdit"):
        saved["fieldParams"]["max_sequence_length"]["max"] = 2048
        expected_bounds = {"max_sequence_length": {"max": 4096}}
        assert params["max_sequence_length"]["value"] == 2048
    else:
        for field in ("width", "height"):
            saved["fieldParams"][field].update(min=1024, max=1024)
            assert params[field]["value"] == 1024
        expected_bounds = {field: {"min": 512, "max": 2048} for field in ("width", "height")}
    serialized = json.dumps(saved, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(serialized.encode("utf-8")).hexdigest() == expected_digest

    assert _compatible_image_contract(saved, current)
    action = getattr(image_module, node["action"])
    observed = {}
    monkeypatch.setattr(action, "set_field_params", lambda self, key, value: observed.update({key: value}))
    arguments = {"image_contract": saved}
    before = deepcopy(arguments)
    action("saved-image-preset").update_image_contract(arguments, None)
    assert arguments == before
    for field, bounds in expected_bounds.items():
        assert {key: observed[field][key] for key in bounds} == bounds
        assert observed[field] == current["fieldParams"][field]
    assert graph_path.read_bytes() == graph_bytes

    wrong_identity = {**saved, "pipelineClass": "StableDiffusionPipeline"}
    wrong_mode = {**saved, "mode": "not_a_reviewed_task"}
    wrong_limit = {**saved, "maxOutputPixels": current.get("maxOutputPixels", 0) + 1}
    altered = deepcopy(saved)
    field = next(iter(expected_bounds))
    altered["fieldParams"][field]["max"] -= 1
    for signal in (wrong_identity, wrong_mode, wrong_limit, altered):
        assert not _compatible_image_contract(signal, current)


def test_unreviewed_pipeline_does_not_inherit_fixed_size_compatibility():
    adapter = get_image_pipeline_adapter("StableDiffusionPipeline")
    current = image_pipeline_contract(adapter, "text_to_image")
    unrelated = deepcopy(current)
    for field in ("width", "height"):
        unrelated["fieldParams"][field].update(min=1024, max=1024)
    assert not _compatible_image_contract(unrelated, current)


@pytest.mark.skipif(not os.getenv('MODIFF_SAVED_ERNIE_GRAPH'), reason='requires the retained original Ernie run')
def test_retained_ernie_workflow_applies_current_size_controls_without_mutating_saved_metadata(monkeypatch):
    from modules.DiffusersImage.main import Generate

    original = Path(os.environ['MODIFF_SAVED_ERNIE_GRAPH']).read_bytes()
    graph = json.loads(original)
    signal = next(node['params']['image_contract']['value'] for node in graph['nodes'].values()
                  if node['module'] == 'modules.DiffusersImage' and node['action'] == 'Generate')
    old = deepcopy(signal)
    observed = {}
    monkeypatch.setattr(Generate, 'set_field_params', lambda self, key, params: observed.update({key: params}))
    Generate('saved-ernie').update_image_contract({'image_contract': signal}, None)
    assert signal == old
    assert signal['fieldParams']['width']['min'] == signal['fieldParams']['width']['max'] == 1024
    assert (observed['width']['min'], observed['width']['max'], observed['width']['step']) == (512, 2048, 32)
    assert Path(os.environ['MODIFF_SAVED_ERNIE_GRAPH']).read_bytes() == original


@pytest.mark.parametrize(
    "name,step",
    [
        ("Kandinsky3Pipeline", 64),
        ("Kandinsky3Img2ImgPipeline", 64),
        ("ErnieImagePipeline", 32),
        ("GlmImagePipeline", 32),
    ],
)
def test_spatial_contract_allows_non_square_sizes_without_raising_pixel_ceiling(name, step):
    adapter = get_image_pipeline_adapter(name)
    contract = image_pipeline_contract(adapter, adapter.mode_options[0])
    for key in ("width", "height"):
        params = contract["fieldParams"][key]
        assert (params["min"], params["max"], params["step"]) == (512, 2048, step)
    assert contract["maxOutputPixels"] == 1024 * 1024


@pytest.mark.parametrize("name", ["Kandinsky3Pipeline", "ErnieImagePipeline", "GlmImagePipeline"])
def test_text_to_image_preflight_preserves_requested_size_and_rejects_oversized_or_unaligned(name):
    adapter = get_image_pipeline_adapter(name)
    pipeline = type(name, (), {})()
    _tag_image_pipeline(
        pipeline, adapter, "text_to_image", adapter.default_repo, "hub", catalog_revision(adapter.default_repo)
    )
    values = {
        "width": 1152,
        "height": 896,
        "num_inference_steps": adapter.max_inference_steps,
        "guidance_scale": adapter.fixed_guidance_scale or 1.5,
    }
    _, accepted = preflight_image_action(pipeline, "Generate", values)
    assert (accepted["width"], accepted["height"]) == (1152, 896)
    for size, message in [((1152, 1024), "cannot exceed"), ((1153, 896), "increments"), ((480, 896), "between")]:
        with pytest.raises(ValueError, match=message):
            preflight_image_action(pipeline, "Generate", {**values, "width": size[0], "height": size[1]})


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="requires reviewed optional runtime")
@pytest.mark.parametrize("name", ["Kandinsky3Pipeline", "ErnieImagePipeline", "GlmImagePipeline"])
def test_pinned_upstream_accepts_explicit_dimensions(name):
    import diffusers

    factory = getattr(diffusers, name)
    parameters = inspect.signature(factory.__call__).parameters
    for dimension in ("width", "height"):
        assert parameters[dimension].default == (None if name == "GlmImagePipeline" else 1024)
        assert parameters[dimension].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    if name == "Kandinsky3Pipeline":
        from diffusers.pipelines.kandinsky3.pipeline_kandinsky3 import downscale_height_and_width

        latent_height, latent_width = downscale_height_and_width(896, 1152)
        assert (latent_height * 8, latent_width * 8) == (896, 1152)
    elif name == "GlmImagePipeline":
        # Invoke the installed library's validation without loading any weights.
        pipeline = SimpleNamespace(
            vae_scale_factor=8, transformer=SimpleNamespace(config=SimpleNamespace(patch_size=2))
        )
        factory.check_inputs(pipeline, "fixture", 896, 1152, None)
        with pytest.raises(ValueError, match="divisible"):
            factory.check_inputs(pipeline, "fixture", 897, 1152, None)
