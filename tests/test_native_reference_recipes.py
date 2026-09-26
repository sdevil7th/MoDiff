"""Reference ordering/composition must agree with the exact upstream variant."""
from types import SimpleNamespace

import pytest
import importlib.util
import torch
from PIL import Image

from modules import MODULE_MAP
from modiff.operation_catalog import build_operation_catalog
from modiff.operation_starters import resolve_operation_starter
from modules.ImageOperations.main import StitchImages, stitch_reference_images
from modules.ModularDiffusers.latents import prepare_image_for_vae_pipeline


@pytest.mark.parametrize("pipeline,profile", [
    ("FluxKontextModularPipeline", "flux-kontext:modular"),
    ("Flux2ModularPipeline", "flux2:modular"),
    ("Flux2KleinModularPipeline", "flux2-klein:modular"),
])
def test_native_reference_task_selects_real_stages_and_only_kontext_composes(pipeline, profile):
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    result = resolve_operation_starter(MODULE_MAP, contracts, {
        "pipelineClass": pipeline, "task": "multi_image_reference_edit", "executionProfileId": profile,
    })
    nodes = {node["operation"]["operationId"]: node for node in result["nodes"]}
    assert "diffusion.encode_image" in nodes and "diffusion.denoise" in nodes
    if pipeline == "FluxKontextModularPipeline":
        assert nodes["diffusion.compose_references"]["values"]["layout"] == "horizontal_reference"
        assert {"source": "diffusion.compose_references", "sourceHandle": "output",
                "target": "diffusion.encode_image", "targetHandle": "image"} in result["edges"]
        assert {"operationId": "diffusion.compose_references", "field": "image"} in result["requiredInputs"]
    else:
        assert "diffusion.compose_references" not in nodes


def test_reference_canvas_matches_standard_adapter_preserves_order_and_bounds():
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, prepare_reference_images
    images = [Image.new("RGB", (4, 2), "red"), Image.new("RGB", (2, 4), "blue")]
    node = object.__new__(StitchImages)
    result = node.execute(image=images, layout="horizontal_reference")
    standard = prepare_reference_images(images, IMAGE_PIPELINE_ADAPTERS["FluxKontextPipeline"])
    assert result["output"].size == (10, 4) and result["count"] == 2
    assert result["output"].tobytes() == standard.tobytes()
    assert result["output"].getpixel((0, 0)) == (255, 0, 0)
    assert result["output"].getpixel((9, 0)) == (0, 0, 255)
    with pytest.raises(ValueError, match="eight"):
        stitch_reference_images([images[0]] * 9)
    with pytest.raises(ValueError, match="execution limit"):
        stitch_reference_images([Image.new("RGB", (8192, 1)), Image.new("RGB", (1, 2))])


@pytest.mark.parametrize("pipeline", ["Flux2ModularPipeline", "Flux2KleinModularPipeline"])
def test_flux2_references_remain_ordered_individual_images_and_are_bounded(pipeline):
    images = [Image.new("RGB", (32, 64), "red"), Image.new("RGB", (64, 32), "blue")]
    cls = type(pipeline, (), {})
    assert prepare_image_for_vae_pipeline(images, cls) is images
    for invalid in ([], images * 5):
        with pytest.raises(ValueError, match="eight"):
            prepare_image_for_vae_pipeline(invalid, cls)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="requires the staged optional Transformers runtime")
def test_actual_upstream_flux2_packs_separate_reference_tokens_and_position_ids():
    from diffusers.modular_pipelines import PipelineState
    from diffusers.modular_pipelines.flux2.before_denoise import Flux2PrepareImageLatentsStep
    state = PipelineState()
    state.set("image_latents", [torch.ones(1, 4, 2, 3), torch.full((1, 4, 3, 2), 2.0)])
    state.set("batch_size", 1)
    state.set("num_images_per_prompt", 1)
    _, state = Flux2PrepareImageLatentsStep()(SimpleNamespace(_execution_device=torch.device("cpu")), state)
    latents = state.get("image_latents")
    ids = state.get("image_latent_ids")
    assert latents.shape == (1, 12, 4)
    assert torch.all(latents[:, :6] == 1) and torch.all(latents[:, 6:] == 2)
    assert torch.all(ids[:, :6, 0] == 10) and torch.all(ids[:, 6:, 0] == 20)
