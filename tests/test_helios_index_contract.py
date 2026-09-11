"""No-weight validation of the exact public Helios Modular index contracts."""

import importlib.util
import unittest
from unittest.mock import patch

from modules.ModularDiffusers.loaders import _validate_reviewed_pipeline_index


CASES = (
    (
        "HeliosModularPipeline",
        "HeliosAutoBlocks",
        "Base",
        "5c50b6bc90eae9bd815d2a50b0c9877e3fd2cf88",
        "HeliosScheduler",
    ),
    (
        "HeliosPyramidModularPipeline",
        "HeliosPyramidAutoBlocks",
        "Mid",
        "477c55427ec0ea774bdebd0fbe736313cfc5a312",
        "HeliosScheduler",
    ),
    (
        "HeliosPyramidDistilledModularPipeline",
        "HeliosPyramidDistilledAutoBlocks",
        "Distilled",
        "b991c0379a018f4de3227d95468237f56066f5bb",
        "HeliosDMDScheduler",
    ),
)


def index_document(model_type, blocks, repository, scheduler):
    def component(library, name, subfolder):
        return [
            library,
            name,
            {
                "pretrained_model_name_or_path": repository,
                "revision": None,
                "subfolder": subfolder,
                "type_hint": [library, name],
                "variant": None,
            },
        ]

    return {
        "_class_name": model_type,
        "_blocks_class_name": blocks,
        "_diffusers_version": "0.38.0.dev0",
        "guider": ["diffusers", "ClassifierFreeZeroStarGuidance"
                   if repository == "BestWishYsh/Helios-Mid" else "ClassifierFreeGuidance"],
        "scheduler": component("diffusers", scheduler, "scheduler"),
        "text_encoder": component("transformers", "UMT5EncoderModel", "text_encoder"),
        "tokenizer": component("transformers", "T5Tokenizer", "tokenizer"),
        "transformer": component("diffusers", "HeliosTransformer3DModel", "transformer"),
        "vae": component("diffusers", "AutoencoderKLWan", "vae"),
    }


@unittest.skipUnless(importlib.util.find_spec("transformers"), "requires the staged optional Transformers runtime")
class HeliosIndexContractTests(unittest.TestCase):
    def test_pinned_helios_variants_accept_their_concrete_tokenizer(self):
        for model_type, blocks, variant, revision, scheduler in CASES:
            repository = "BestWishYsh/Helios-" + variant
            document = index_document(model_type, blocks, repository, scheduler)
            with (
                self.subTest(variant=variant),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("modular_model_index.json", document),
                ),
            ):
                self.assertEqual(
                    _validate_reviewed_pipeline_index(model_type, repository, revision),
                    ("modular_model_index.json", document),
                )

    def test_alias_does_not_admit_other_tokenizers_or_repositories(self):
        for model_type, blocks, variant, revision, scheduler in CASES:
            repository = "BestWishYsh/Helios-" + variant
            for tokenizer, repo in (
                ("T5TokenizerFast", repository),
                ("PreTrainedTokenizerFast", repository),
                ("T5Tokenizer", "unit/unreviewed-helios"),
            ):
                document = index_document(model_type, blocks, repo, scheduler)
                document["tokenizer"][1] = tokenizer
                document["tokenizer"][2]["type_hint"][1] = tokenizer
                with (
                    self.subTest(variant=variant, tokenizer=tokenizer, repository=repo),
                    patch(
                        "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                        return_value=("modular_model_index.json", document),
                    ),
                    self.assertRaisesRegex(ValueError, "tokenizer"),
                ):
                    _validate_reviewed_pipeline_index(model_type, repo, revision)

    def test_two_field_config_components_do_not_weaken_weight_or_class_validation(self):
        model_type, blocks, variant, revision, scheduler = CASES[0]
        repository = "BestWishYsh/Helios-" + variant
        for name, value in (
            ("guider", ["diffusers", "ClassifierFreeZeroStarGuidance"]),
            ("text_encoder", ["transformers", "UMT5EncoderModel"]),
            ("scheduler", ["diffusers", "HeliosDMDScheduler", {"type_hint": ["diffusers", "HeliosDMDScheduler"]}]),
            ("unexpected_component", ["diffusers", "ClassifierFreeGuidance"]),
        ):
            document = index_document(model_type, blocks, repository, scheduler)
            document[name] = value
            with (
                self.subTest(component=name),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("modular_model_index.json", document),
                ),
                self.assertRaisesRegex(ValueError, name),
            ):
                _validate_reviewed_pipeline_index(model_type, repository, revision)
