"""Image pipelines must load the installed runtime files, not unrelated Hub weights."""

import importlib.util
import inspect
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from modules.DiffusersImage.main import load_cached_image_component

REPO = "example/image-pipeline"
REVISION = "a" * 40


def pipeline_factory():
    return type("Pipeline", (), {"config_name": "model_index.json", "from_pretrained": Mock()})


def test_pinned_pipeline_uses_exact_installed_directory_without_hub_tree_check(tmp_path):
    factory = pipeline_factory()
    snapshot = tmp_path / REVISION
    kwargs = dict(revision=REVISION, local_files_only=True, use_safetensors=True, variant="fp16")
    with patch("utils.huggingface.exact_cached_snapshot_path", return_value=snapshot) as resolve:
        result = load_cached_image_component(factory, REPO, **kwargs)
    resolve.assert_called_once_with(REPO, REVISION)
    factory.from_pretrained.assert_called_once_with(snapshot, **kwargs)
    assert result is factory.from_pretrained.return_value


@pytest.mark.parametrize(
    "error",
    [FileNotFoundError("Install the exact reviewed snapshot"), ValueError("Snapshot escaped its managed cache")],
)
def test_invalid_or_missing_snapshot_never_falls_back_to_hub_loading(error):
    factory = pipeline_factory()
    with patch("utils.huggingface.exact_cached_snapshot_path", side_effect=error):
        with pytest.raises(type(error), match=str(error)):
            load_cached_image_component(factory, REPO, revision=REVISION, local_files_only=True)
    factory.from_pretrained.assert_not_called()


def test_missing_required_pipeline_component_still_fails_from_local_directory():
    factory = pipeline_factory()
    factory.from_pretrained.side_effect = OSError("Required transformer shard is missing")
    snapshot = Path("/reviewed/snapshots") / REVISION
    with patch("utils.huggingface.exact_cached_snapshot_path", return_value=snapshot):
        with pytest.raises(OSError, match="Required transformer shard"):
            load_cached_image_component(factory, REPO, revision=REVISION, local_files_only=True)
    assert factory.from_pretrained.call_args.args == (snapshot,)


def test_local_pipeline_and_individual_component_keep_their_existing_loader_paths():
    pipeline = pipeline_factory()
    component = type("Component", (), {"config_name": "config.json", "from_pretrained": Mock()})
    with patch("utils.huggingface.exact_cached_snapshot_path") as resolve:
        load_cached_image_component(pipeline, "/operator/local-pipeline", revision=None, local_files_only=True)
        load_cached_image_component(component, REPO, revision=REVISION, local_files_only=True)
    resolve.assert_not_called()
    pipeline.from_pretrained.assert_called_once_with("/operator/local-pipeline", revision=None, local_files_only=True)
    component.from_pretrained.assert_called_once_with(REPO, revision=REVISION, local_files_only=True)


def test_mutable_revision_cannot_bypass_exact_snapshot_validation():
    factory = pipeline_factory()
    with pytest.raises(ValueError, match="40-character commit"):
        load_cached_image_component(factory, REPO, revision="main", local_files_only=True)
    factory.from_pretrained.assert_not_called()


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="requires reviewed optional runtime")
@pytest.mark.parametrize(
    "name",
    [
        "OvisImagePipeline",
        "LongCatImagePipeline",
        "HunyuanDiTPipeline",
        "HunyuanDiTPAGPipeline",
        "HunyuanDiTControlNetPipeline",
    ],
)
def test_pinned_upstream_classes_expose_reviewed_snapshot_and_call_contracts(name, tmp_path):
    import diffusers

    factory = getattr(diffusers, name)
    assert factory.config_name == "model_index.json"
    signature = inspect.signature(factory.__call__).parameters
    if name.startswith("HunyuanDiT"):
        assert signature["use_resolution_binning"].default is True
    if name == "LongCatImagePipeline":
        assert "callback_on_step_end" not in signature
    snapshot = tmp_path / REVISION
    with (
        patch("utils.huggingface.exact_cached_snapshot_path", return_value=snapshot),
        patch.object(factory, "from_pretrained") as load,
    ):
        load_cached_image_component(factory, REPO, revision=REVISION, local_files_only=True)
    load.assert_called_once_with(snapshot, revision=REVISION, local_files_only=True)
