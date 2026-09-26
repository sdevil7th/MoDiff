"""Exercise real pinned input/scheduler validation without allocating model weights."""
import inspect
from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("transformers")

from diffusers import SCMScheduler, SanaSprintImg2ImgPipeline, SanaSprintPipeline
from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, Generate, Edit, _tag_image_pipeline
from modiff.model_artifact_catalog import catalog_revision


@pytest.mark.parametrize("upstream,action,mode", [
    (SanaSprintPipeline, Generate, "text_to_image"),
    (SanaSprintImg2ImgPipeline, Edit, "edit_image"),
])
@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_generic_dispatch_preserves_requested_sprint_steps(upstream, action, mode, steps):
    signature = inspect.signature(upstream.__call__)
    received = {}

    def call(self, **kwargs):
        received.update(kwargs)
        bound = signature.bind(self, **kwargs)
        bound.apply_defaults()
        checks = {key: bound.arguments[key] for key in inspect.signature(upstream.check_inputs).parameters if key != "self"}
        upstream.check_inputs(self, **checks)
        scheduler = SCMScheduler()
        scheduler.set_timesteps(
            bound.arguments["num_inference_steps"],
            max_timesteps=bound.arguments["max_timesteps"],
            intermediate_timesteps=bound.arguments["intermediate_timesteps"],
        )
        assert len(scheduler.timesteps) == steps + 1
        if steps == 2:
            assert float(scheduler.timesteps[1]) == pytest.approx(1.3)
        return SimpleNamespace(images=[Image.new("RGB", (64, 64), "white")])

    call.__signature__ = signature
    pipeline = type(upstream.__name__, (), {
        "__call__": call, "_execution_device": "cpu", "_callback_tensor_inputs": ["latents"],
    })()
    adapter = IMAGE_PIPELINE_ADAPTERS[upstream.__name__]
    _tag_image_pipeline(pipeline, adapter, mode, adapter.default_repo, "hub", catalog_revision(adapter.default_repo))
    values = {"pipeline": pipeline, "prompt": "Brass instrument on blue velvet", "seed": "123",
              "num_inference_steps": str(steps), "width": 1024, "height": 1024}
    if action is Edit:
        values.update(image=Image.new("RGB", (1024, 1024)), strength=1.0)
    node = action()
    node(**values)
    assert received["num_inference_steps"] == steps
    if steps == 2:
        assert "intermediate_timesteps" not in received
    else:
        assert received["intermediate_timesteps"] is None
