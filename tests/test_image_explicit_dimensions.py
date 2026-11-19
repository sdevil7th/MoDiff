"""Keep spatial defaults separate from the generic image action's valid range."""

import importlib.util
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
