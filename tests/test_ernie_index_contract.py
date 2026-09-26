"""Exact ERNIE tokenizer factory/concrete mappings, without model loading."""

from copy import deepcopy
import importlib.util
from unittest.mock import patch

import pytest

from modules.ModularDiffusers.loaders import _validate_reviewed_pipeline_index

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="requires the staged optional Transformers runtime",
)


REPOSITORY = "baidu/ERNIE-Image-Turbo"
REVISION = "bc68c81e2a1730a394d5fc9fae70713dee940140"
DOCUMENT = {
    "_class_name": "ErnieImagePipeline",
    "_diffusers_version": "0.36.0",
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "pe": ["transformers", "Ministral3ForCausalLM"],
    "pe_tokenizer": ["transformers", "TokenizersBackend"],
    "text_encoder": ["transformers", "Mistral3Model"],
    "tokenizer": ["transformers", "TokenizersBackend"],
    "transformer": ["diffusers", "ErnieImageTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


def test_ernie_pinned_index_accepts_the_two_reviewed_concrete_tokenizers():
    with patch("modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
               return_value=("model_index.json", DOCUMENT)):
        assert _validate_reviewed_pipeline_index(
            "ErnieImageModularPipeline", REPOSITORY, REVISION,
        ) == ("model_index.json", DOCUMENT)


@pytest.mark.parametrize("component", ["pe_tokenizer", "tokenizer", "pe", "text_encoder", "transformer", "vae"])
def test_ernie_index_still_rejects_unreviewed_component_types(component):
    document = deepcopy(DOCUMENT)
    document[component] = ["unreviewed_package", "UnreviewedModel"]
    with patch("modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
               return_value=("model_index.json", document)):
        with pytest.raises(ValueError, match=f"component '{component}'"):
            _validate_reviewed_pipeline_index("ErnieImageModularPipeline", REPOSITORY, REVISION)
