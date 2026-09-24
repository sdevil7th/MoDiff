"""The selected upstream workflow, not an unpruned family, owns the bundle."""
from types import SimpleNamespace
import importlib.util

import pytest
from diffusers import QwenImageEditModularPipeline

from modules.ModularDiffusers.loaders import _loader_text_encoder_component_names


@pytest.mark.parametrize("workflow", [None, "image_conditioned", "image_conditioned_inpainting"])
@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="requires the staged optional Transformers runtime")
def test_qwen_edit_text_bundle_survives_workflow_flattening(workflow):
    pipeline = QwenImageEditModularPipeline(workflow=workflow)
    original_keys = list(pipeline.blocks.sub_blocks)
    if workflow:
        assert "text_encoder" not in original_keys
        assert "text_encoder.encode" in original_keys
    assert set(_loader_text_encoder_component_names(pipeline)) == {"text_encoder", "processor"}
    assert list(pipeline.blocks.sub_blocks) == original_keys


def test_loader_does_not_invent_a_text_bundle_for_other_stages():
    loader = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={"vae_encoder.encode": object()}))
    assert _loader_text_encoder_component_names(loader) == []
