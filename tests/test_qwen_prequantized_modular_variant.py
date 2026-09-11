"""The reviewed low-memory checkpoint keeps the existing Qwen Cluster contract."""

from copy import deepcopy
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from modules.ModularDiffusers.loaders import ModelsLoader, _validate_reviewed_pipeline_index


REPOSITORY = "unsloth/Qwen-Image-2512-unsloth-bnb-4bit"
REVISION = "f50b8c24fe21e9265509b15113b7cca82d0a4443"
MODEL_TYPE = "QwenImageModularPipeline"


def test_saved_quantization_offloads_each_component_before_loading_the_next():
    from modules.ModularDiffusers.loaders import load_components_strict, should_incrementally_group_offload

    events = []
    def spec(name):
        def load(**kwargs):
            events.append(("load", name))
            return object()
        return SimpleNamespace(pretrained_model_name_or_path=REPOSITORY, load=load)

    pipeline = SimpleNamespace(
        _component_specs={name: spec(name) for name in ("text_encoder", "transformer")},
        register_components=lambda **components: None,
    )
    def offload(pipeline, *, component_names, **kwargs):
        events.append(("offload", component_names[0]))
        return SimpleNamespace(applied=True, components=component_names, method="group", mode="group_cpu", disk_path=None)

    policy = {"device": "cuda:0", "mode": "group_cpu", "node_id": "test", "scope": "modular-diffusers"}
    with patch("modules.ModularDiffusers.loaders.apply_component_group_offload", side_effect=offload), \
         patch("modules.ModularDiffusers.loaders.torch.cuda.is_available", return_value=False):
        load_components_strict(
            pipeline, ["text_encoder", "transformer"], required_names={"text_encoder", "transformer"},
            model_id=REPOSITORY, dtype="bfloat16", offload_mode="group_cpu", quant_config=None,
            diagnostics={}, component_load_kwargs={},
            incremental_group_offload=policy if should_incrementally_group_offload(
                use_group_offload=True, quant_config=None,
            ) else None,
        )
    assert events == [("load", "text_encoder"), ("offload", "text_encoder"),
                      ("load", "transformer"), ("offload", "transformer")]


def test_cluster_qualification_and_loader_share_the_workflow_variant_review():
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    from modiff.huggingface_cluster_runtime import _effective_reviewed_artifact

    profile = DIFFUSERS_EXECUTION_PROFILES["qwen-image:modular"]
    admission = {
        "artifact": {"repo": "Qwen/Qwen-Image-2512", "revision": "25468b98e3276ca6700de15c6628e51b7de54a26"},
        "executionParameterSources": ["modelVariant"],
    }
    assert _effective_reviewed_artifact(
        admission, profile, REPOSITORY, workflow_id="text2image",
    ) == {"repo": REPOSITORY, "revision": REVISION}
    with pytest.raises(ValueError, match="not an admitted same-pipeline variant"):
        _effective_reviewed_artifact(admission, profile, REPOSITORY, workflow_id="image2image")
    with pytest.raises(ValueError, match="does not expose"):
        _effective_reviewed_artifact(
            {**admission, "executionParameterSources": []}, profile, REPOSITORY, workflow_id="text2image",
        )


def test_prequantized_variant_resolves_repository_and_immutable_revision_together():
    selector, revision = ModelsLoader._effective_builtin_selector(
        model_type=MODEL_TYPE,
        repo_id={"source": "hub", "value": "Qwen/Qwen-Image-2512"},
        revision="25468b98e3276ca6700de15c6628e51b7de54a26",
        workflow_id="text2image",
        reviewed_variant=REPOSITORY,
    )
    assert selector == {"source": "hub", "value": REPOSITORY}
    assert revision == REVISION
    assert ModelsLoader._reviewed_builtin_selection(
        model_type=MODEL_TYPE, repo_id=selector, revision=revision,
    ) == ("hub", REPOSITORY, REVISION)
    with pytest.raises(ValueError, match="requires reviewed revision"):
        ModelsLoader._reviewed_builtin_selection(
            model_type=MODEL_TYPE, repo_id=selector, revision="f" * 40,
        )


@pytest.mark.parametrize("workflow", ["image2image", "inpaint"])
def test_prequantized_variant_is_not_implicitly_admitted_to_other_workflows(workflow):
    with pytest.raises(ValueError, match="does not admit model variant"):
        ModelsLoader._effective_builtin_selector(
            model_type=MODEL_TYPE,
            repo_id={"source": "hub", "value": "Qwen/Qwen-Image-2512"},
            revision=None, workflow_id=workflow, reviewed_variant=REPOSITORY,
        )


def test_prequantized_standard_index_uses_only_the_reviewed_component_classes():
    pytest.importorskip("transformers")
    document = {
        "_class_name": "QwenImagePipeline",
        "_diffusers_version": "0.36.0",
        "_name_or_path": "Qwen/Qwen-Image-2512",
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
        "tokenizer": ["transformers", "Qwen2Tokenizer"],
        "transformer": ["diffusers", "QwenImageTransformer2DModel"],
        "vae": ["diffusers", "AutoencoderKLQwenImage"],
    }
    with patch("modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
               return_value=("model_index.json", document)):
        assert _validate_reviewed_pipeline_index(MODEL_TYPE, REPOSITORY, REVISION) == (
            "model_index.json", document,
        )
    for component in ("text_encoder", "transformer", "tokenizer", "vae", "scheduler"):
        tampered = deepcopy(document)
        tampered[component] = ["unreviewed_package", "UnreviewedModel"]
        with patch("modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                   return_value=("model_index.json", tampered)):
            with pytest.raises(ValueError, match=component):
                _validate_reviewed_pipeline_index(MODEL_TYPE, REPOSITORY, REVISION)
