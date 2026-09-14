"""No-weight regression using the exact reviewed Mid index and installed blocks."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("transformers")
from diffusers import ClassifierFreeGuidance, ClassifierFreeZeroStarGuidance, ComponentsManager

from modules.ModularDiffusers.loaders import (
    _instantiate_reviewed_builtin_pipeline,
    _validate_reviewed_pipeline_index,
)
from tests.test_helios_index_contract import CASES, index_document


REPOSITORY = "BestWishYsh/Helios-Mid"
REVISION = "477c55427ec0ea774bdebd0fbe736313cfc5a312"
INDEX = json.loads((Path(__file__).parent / "fixtures/helios_mid_modular_model_index.json").read_text())


def test_exact_reviewed_mid_index_accepts_native_zero_star_guider():
    with patch("modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
               return_value=("modular_model_index.json", INDEX)):
        assert _validate_reviewed_pipeline_index("HeliosPyramidModularPipeline", REPOSITORY, REVISION) == (
            "modular_model_index.json", INDEX)


@pytest.mark.parametrize("workflow", ["text2video", "image2video", "video2video"])
def test_mid_constructor_retains_native_guider_and_unloaded_weights(workflow):
    with patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("no download allowed")):
        pipeline = _instantiate_reviewed_builtin_pipeline(
            "HeliosPyramidModularPipeline", REPOSITORY,
            index_filename="modular_model_index.json", index_document=INDEX,
            components_manager=ComponentsManager(), collection="mid-guider-regression", workflow_id=workflow)
    assert type(pipeline.guider) is ClassifierFreeZeroStarGuidance
    assert pipeline.guider.config.guidance_scale == 5.0
    assert pipeline.guider.config.zero_init_steps == 2
    assert pipeline.get_component_spec("guider").type_hint is ClassifierFreeZeroStarGuidance
    assert all(getattr(pipeline, name) is None for name in ("transformer", "vae", "text_encoder", "tokenizer"))


@pytest.mark.parametrize("case", [CASES[0], CASES[2]])
def test_other_helios_variants_keep_cfg_and_reject_zero_star_index(case):
    model_type, blocks, variant, revision, scheduler = case
    repository = "BestWishYsh/Helios-" + variant
    document = index_document(model_type, blocks, repository, scheduler)
    pipeline = _instantiate_reviewed_builtin_pipeline(
        model_type, repository, index_filename="modular_model_index.json", index_document=document,
        components_manager=ComponentsManager(), collection="other-guider-regression", workflow_id="text2video")
    assert type(pipeline.guider) is ClassifierFreeGuidance
    document["guider"] = ["diffusers", "ClassifierFreeZeroStarGuidance"]
    with patch("modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
               return_value=("modular_model_index.json", document)):
        with pytest.raises(ValueError, match="guider"):
            _validate_reviewed_pipeline_index(model_type, repository, revision)
