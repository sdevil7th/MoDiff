"""Real tiny tokenizer/processor loads from an isolated pinned offline snapshot."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from diffusers import ComponentSpec
from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
from PIL import Image
from transformers import Qwen2Tokenizer, Qwen2VLImageProcessor, Qwen2VLProcessor
from transformers import BertConfig, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from modules.ModularDiffusers.loaders import RequiredComponentLoadError, load_components_strict
from modules.ModularDiffusers import loaders


@pytest.fixture
def pinned_processor_snapshot(tmp_path):
    repo = "test-owned-processor/tiny-multimodal"
    revision = "a" * 40
    cache = tmp_path / "hub"
    snapshot = cache / "models--test-owned-processor--tiny-multimodal" / "snapshots" / revision
    processor = snapshot / "processor"
    processor.mkdir(parents=True)
    Qwen2Tokenizer(vocab={"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2,
                         "<|image_pad|>": 3, "<|video_pad|>": 4, "t": 5, "e": 6, "s": 7},
                   merges=[]).save_pretrained(processor)
    Qwen2VLImageProcessor(min_pixels=3136, max_pixels=12845056).save_pretrained(processor)
    encoder = Qwen2_5_VLForConditionalGeneration(Qwen2_5_VLConfig(
        text_config={"vocab_size": 32, "hidden_size": 16, "intermediate_size": 32,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "num_key_value_heads": 2},
        vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                       "num_heads": 2, "out_hidden_size": 16, "patch_size": 4,
                       "temporal_patch_size": 2, "spatial_merge_size": 2},
    ))
    encoder.save_pretrained(snapshot / "text_encoder")
    return repo, revision, cache, snapshot


def component_pipeline(repo, revision):
    specs = {
        "text_encoder": ComponentSpec("text_encoder", Qwen2_5_VLForConditionalGeneration,
                                      pretrained_model_name_or_path=repo, subfolder="text_encoder", revision=revision),
        "processor": ComponentSpec("processor", Qwen2VLProcessor,
                                   pretrained_model_name_or_path=repo, subfolder="processor", revision=revision),
    }
    pipeline = SimpleNamespace(_component_specs=specs, text_encoder=None, processor=None)
    pipeline.register_components = lambda **values: [setattr(pipeline, key, value) for key, value in values.items()]
    return pipeline


def test_processor_legacy_layout_requires_model_config_for_auto_tokenizer(pinned_processor_snapshot):
    repo, revision, cache, snapshot = pinned_processor_snapshot
    assert not (snapshot / "processor" / "config.json").exists()
    with pytest.raises(OSError, match="couldn't find|connect"):
        Qwen2VLProcessor.from_pretrained(repo, revision=revision, subfolder="processor",
                                        cache_dir=cache, local_files_only=True)


def test_same_owner_config_loads_real_processor_without_new_snapshot_files(pinned_processor_snapshot):
    repo, revision, cache, snapshot = pinned_processor_snapshot
    before = {str(p.relative_to(snapshot)): p.read_bytes() for p in snapshot.rglob("*") if p.is_file()}
    pipeline = component_pipeline(repo, revision)
    diagnostics = {}
    load_components_strict(pipeline, ["text_encoder", "processor"], required_names={"text_encoder", "processor"},
                           model_id=repo, dtype=torch.float32, offload_mode="none", quant_config=None,
                           diagnostics=diagnostics, component_load_kwargs={"cache_dir": str(cache),
                                                                         "local_files_only": True})
    assert isinstance(pipeline.processor, Qwen2VLProcessor)
    assert isinstance(pipeline.processor.tokenizer, Qwen2Tokenizer)
    assert pipeline.processor.image_processor.patch_size == 14
    assert pipeline.processor._diffusers_load_id == pipeline._component_specs["processor"].load_id
    assert diagnostics["components_loaded"] == ["text_encoder", "processor"]
    baseline = Qwen2VLProcessor.from_pretrained(repo, revision=revision, subfolder="processor", cache_dir=cache,
                                               local_files_only=True, config=deepcopy(pipeline.text_encoder.config))
    image = Image.new("RGB", (28, 28), (24, 60, 120))
    expected = baseline(text="test", images=image, return_tensors="pt")
    actual = pipeline.processor(text="test", images=image, return_tensors="pt")
    assert set(actual) == set(expected)
    assert all(torch.equal(actual[key], expected[key]) for key in actual)
    after = {str(p.relative_to(snapshot)): p.read_bytes() for p in snapshot.rglob("*") if p.is_file()}
    assert before == after
    assert not (snapshot / "processor" / "config.json").exists()


def test_explicit_model_config_already_loads_real_processor(pinned_processor_snapshot):
    repo, revision, cache, _ = pinned_processor_snapshot
    config = Qwen2_5_VLConfig()
    before = deepcopy(config.to_dict())
    processor = Qwen2VLProcessor.from_pretrained(repo, revision=revision, subfolder="processor", cache_dir=cache,
                                                local_files_only=True, config=config)
    assert isinstance(processor.tokenizer, Qwen2Tokenizer)
    assert config.to_dict() == before


def selected_encoder(pipeline, cache):
    return pipeline._component_specs["text_encoder"].load(cache_dir=cache, local_files_only=True)


def load_processor(pipeline, repo, cache, selected_components=None, **kwargs):
    diagnostics = {}
    load_components_strict(pipeline, ["processor"], required_names={"processor"}, model_id=repo,
                           dtype=torch.float32, offload_mode="none", quant_config=None, diagnostics=diagnostics,
                           component_load_kwargs={"cache_dir": str(cache), "local_files_only": True, **kwargs},
                           selected_components=selected_components)
    return diagnostics


def test_exact_selected_reuse_copies_config_without_mutating_source(pinned_processor_snapshot, monkeypatch):
    repo, revision, cache, _ = pinned_processor_snapshot
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    before = deepcopy(encoder.config.to_dict())
    observed = []
    spec = pipeline._component_specs["processor"]
    original = spec.load

    def actual_load(**kwargs):
        observed.append(kwargs["config"])
        return original(**kwargs)

    monkeypatch.setattr(spec, "load", actual_load)
    diagnostics = load_processor(pipeline, repo, cache, {"text_encoder": encoder})
    assert observed[0] is not encoder.config
    assert observed[0].vision_config is not encoder.config.vision_config
    assert observed[0].to_dict() == before
    assert encoder.config.to_dict() == before
    assert diagnostics["processor_config_bindings"]["processor"]["load_id"] == encoder._diffusers_load_id


@pytest.mark.parametrize("change", ["foreign_repo", "foreign_revision", "stale_load_id", "missing_load_id",
                                   "wrong_config_class", "ambient_only", "ambiguous_sources"])
def test_processor_cannot_borrow_foreign_stale_ambient_or_ambiguous_config(pinned_processor_snapshot, change):
    repo, revision, cache, _ = pinned_processor_snapshot
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    selected = {"text_encoder": encoder}
    source = pipeline._component_specs["text_encoder"]
    if change == "foreign_repo":
        source.pretrained_model_name_or_path = "foreign/repository"
        encoder._diffusers_load_id = source.load_id
    elif change == "foreign_revision":
        source.revision = "b" * 40
        encoder._diffusers_load_id = source.load_id
    elif change == "stale_load_id":
        encoder._diffusers_load_id = source.load_id.replace(revision, "b" * 40)
    elif change == "missing_load_id":
        del encoder._diffusers_load_id
    elif change == "wrong_config_class":
        encoder.config = BertConfig()
    elif change == "ambient_only":
        pipeline.text_encoder = encoder
        selected = None
    elif change == "ambiguous_sources":
        pipeline._component_specs["second_encoder"] = deepcopy(source)
        selected["second_encoder"] = encoder
    with pytest.raises(RequiredComponentLoadError, match="processor"):
        load_processor(pipeline, repo, cache, selected)
    assert pipeline.processor is None


def test_explicit_processor_config_is_preserved_even_with_foreign_selected_model(pinned_processor_snapshot, monkeypatch):
    repo, revision, cache, _ = pinned_processor_snapshot
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    encoder.config = BertConfig()
    explicit = Qwen2_5_VLConfig()
    spec = pipeline._component_specs["processor"]
    original = spec.load
    observed = []

    def actual_load(**kwargs):
        observed.append(kwargs["config"])
        return original(**kwargs)

    monkeypatch.setattr(spec, "load", actual_load)
    diagnostics = load_processor(pipeline, repo, cache, {"text_encoder": encoder}, config=explicit)
    assert observed == [explicit]
    assert "processor_config_bindings" not in diagnostics


def test_existing_corrupt_processor_model_config_is_not_hidden_by_peer_config(pinned_processor_snapshot):
    repo, revision, cache, snapshot = pinned_processor_snapshot
    (snapshot / "processor" / "config.json").write_text("{not valid json")
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    with pytest.raises(RequiredComponentLoadError, match="valid JSON"):
        load_processor(pipeline, repo, cache, {"text_encoder": encoder})
    assert pipeline.processor is None


def test_existing_valid_processor_config_remains_authoritative(pinned_processor_snapshot, monkeypatch):
    repo, revision, cache, snapshot = pinned_processor_snapshot
    own_config = Qwen2_5_VLConfig()
    own_config.save_pretrained(snapshot / "processor")
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    spec = pipeline._component_specs["processor"]
    original = spec.load
    observed = []

    def actual_load(**kwargs):
        observed.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(spec, "load", actual_load)
    diagnostics = load_processor(pipeline, repo, cache, {"text_encoder": encoder})
    assert len(observed) == 1 and "config" not in observed[0]
    assert "processor_config_bindings" not in diagnostics


def test_binding_preserves_upstream_tokenizer_class_selection(pinned_processor_snapshot):
    repo, revision, cache, snapshot = pinned_processor_snapshot
    tokenizer_path = snapshot / "processor" / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_path.read_text())
    tokenizer_config["tokenizer_class"] = "NotAnInstalledTokenizer"
    tokenizer_path.write_text(json.dumps(tokenizer_config))
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    # Transformers may select its backend fallback for an unfamiliar class.
    # Preserve that exact upstream behavior; do not force a tokenizer class.
    baseline = Qwen2VLProcessor.from_pretrained(repo, revision=revision, subfolder="processor", cache_dir=cache,
                                               local_files_only=True, config=deepcopy(encoder.config))
    load_processor(pipeline, repo, cache, {"text_encoder": encoder})
    assert type(pipeline.processor.tokenizer) is type(baseline.tokenizer)
    assert pipeline.processor.tokenizer("test") == baseline.tokenizer("test")


def test_binding_does_not_bypass_explicit_tokenizer_selector_validation(pinned_processor_snapshot):
    repo, revision, cache, _ = pinned_processor_snapshot
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    with pytest.raises(RequiredComponentLoadError, match="tokenizer_type.*not_registered.*does not exist"):
        load_processor(pipeline, repo, cache, {"text_encoder": encoder}, tokenizer_type="not_registered")
    assert pipeline.processor is None


def test_binding_does_not_replace_missing_image_processor_config(pinned_processor_snapshot):
    repo, revision, cache, snapshot = pinned_processor_snapshot
    filename = "preprocessor_config.json"
    (snapshot / "processor" / filename).unlink()
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    with pytest.raises(RequiredComponentLoadError):
        load_processor(pipeline, repo, cache, {"text_encoder": encoder})
    assert pipeline.processor is None
    assert not (snapshot / "processor" / filename).exists()


def test_local_repository_config_resolution_is_preserved(pinned_processor_snapshot, monkeypatch):
    _, revision, cache, snapshot = pinned_processor_snapshot
    repo = str(snapshot)
    Qwen2_5_VLConfig().save_pretrained(snapshot / "processor")
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    spec = pipeline._component_specs["processor"]
    original = spec.load
    calls = []

    def actual_load(**kwargs):
        calls.append(dict(kwargs))
        return original(**kwargs)

    monkeypatch.setattr(spec, "load", actual_load)
    diagnostics = load_processor(pipeline, repo, cache, {"text_encoder": encoder})
    assert len(calls) == 1 and "config" not in calls[0]
    assert isinstance(pipeline.processor, Qwen2VLProcessor)
    assert "processor_config_bindings" not in diagnostics


@pytest.mark.parametrize("field,value", [("pretrained_model_name_or_path", "foreign/repository"),
                                       ("revision", "b" * 40), ("subfolder", "other_processor"),
                                       ("variant", "foreign")])
def test_effective_loading_identity_overrides_cannot_borrow_config(pinned_processor_snapshot, monkeypatch, field, value):
    repo, revision, cache, _ = pinned_processor_snapshot
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    calls = []

    def fail_effective_load(**kwargs):
        calls.append(dict(kwargs))
        raise OSError("effective alternate artifact has no local processor")

    monkeypatch.setattr(pipeline._component_specs["processor"], "load", fail_effective_load)
    with pytest.raises(RequiredComponentLoadError, match="effective alternate artifact"):
        load_processor(pipeline, repo, cache, {"text_encoder": encoder}, **{field: value})
    assert len(calls) == 1 and "config" not in calls[0]
    assert calls[0][field] == value
    assert pipeline.processor is None


@pytest.mark.parametrize("succeeds", [True, False])
def test_online_unknown_config_uses_ordinary_load_without_binding_or_retry(pinned_processor_snapshot, monkeypatch, succeeds):
    repo, revision, cache, _ = pinned_processor_snapshot
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    monkeypatch.setattr(loaders.hub_constants, "HF_HUB_OFFLINE", False)
    assert try_to_load_from_cache(repo, "processor/config.json", cache_dir=cache, revision=revision) is None
    calls = []
    expected_processor = object()

    def online_load(**kwargs):
        calls.append(dict(kwargs))
        if not succeeds:
            raise OSError("connection or access failure does not establish config absence")
        return expected_processor

    monkeypatch.setattr(pipeline._component_specs["processor"], "load", online_load)
    if succeeds:
        diagnostics = load_processor(pipeline, repo, cache, {"text_encoder": encoder}, local_files_only=False)
        assert pipeline.processor is expected_processor
        assert "processor_config_bindings" not in diagnostics
    else:
        with pytest.raises(RequiredComponentLoadError, match="connection or access failure"):
            load_processor(pipeline, repo, cache, {"text_encoder": encoder}, local_files_only=False)
        assert pipeline.processor is None
    assert len(calls) == 1 and "config" not in calls[0]


def test_online_exact_missing_config_marker_allows_one_real_processor_retry(pinned_processor_snapshot, monkeypatch):
    import httpx
    import huggingface_hub.file_download as hub_download
    from huggingface_hub.errors import RemoteEntryNotFoundError

    repo, revision, cache, _ = pinned_processor_snapshot
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    monkeypatch.setattr(loaders.hub_constants, "HF_HUB_OFFLINE", False)
    assert try_to_load_from_cache(repo, "processor/config.json", cache_dir=cache, revision=revision) is None
    original = pipeline._component_specs["processor"].load
    calls = []

    def missing_metadata(**kwargs):
        assert kwargs["url"].endswith(f"/{revision}/processor/config.json")
        response = httpx.Response(404, headers={"X-Repo-Commit": revision},
                                  request=httpx.Request("HEAD", kwargs["url"]))
        raise RemoteEntryNotFoundError("processor/config.json does not exist", response=response)

    monkeypatch.setattr(hub_download, "get_hf_file_metadata", missing_metadata)

    def actual_load(**kwargs):
        calls.append(dict(kwargs))
        if "config" not in kwargs:
            # Exercise the actual Hub 404 path and exact-pin negative marker;
            # this in-memory metadata response performs no network request.
            hub_download.hf_hub_download(repo, "processor/config.json", revision=revision, cache_dir=cache,
                                        local_files_only=False)
        return original(**{**kwargs, "local_files_only": True})

    monkeypatch.setattr(pipeline._component_specs["processor"], "load", actual_load)
    diagnostics = load_processor(pipeline, repo, cache, {"text_encoder": encoder}, local_files_only=False)
    assert try_to_load_from_cache(repo, "processor/config.json", cache_dir=cache, revision=revision) is _CACHED_NO_EXIST
    assert len(calls) == 2 and "config" not in calls[0] and "config" in calls[1]
    assert isinstance(pipeline.processor, Qwen2VLProcessor)
    assert diagnostics["processor_config_bindings"]["processor"]["load_id"] == encoder._diffusers_load_id


def test_online_already_known_absence_binds_before_load(pinned_processor_snapshot, monkeypatch):
    repo, revision, cache, snapshot = pinned_processor_snapshot
    marker = snapshot.parent.parent / ".no_exist" / revision / "processor" / "config.json"
    marker.parent.mkdir(parents=True)
    marker.touch()
    assert try_to_load_from_cache(repo, "processor/config.json", cache_dir=cache, revision=revision) is _CACHED_NO_EXIST
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    monkeypatch.setattr(loaders.hub_constants, "HF_HUB_OFFLINE", False)
    original = pipeline._component_specs["processor"].load
    calls = []

    def actual_load(**kwargs):
        calls.append(dict(kwargs))
        return original(**{**kwargs, "local_files_only": True})

    monkeypatch.setattr(pipeline._component_specs["processor"], "load", actual_load)
    load_processor(pipeline, repo, cache, {"text_encoder": encoder}, local_files_only=False)
    assert len(calls) == 1 and "config" in calls[0]


@pytest.mark.parametrize("marker_revision,filename", [("b" * 40, "processor/config.json"),
                                                     ("a" * 40, "processor/other_config.json")])
def test_online_failure_does_not_retry_for_an_unrelated_negative_marker(pinned_processor_snapshot, monkeypatch,
                                                                     marker_revision, filename):
    repo, revision, cache, snapshot = pinned_processor_snapshot
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    monkeypatch.setattr(loaders.hub_constants, "HF_HUB_OFFLINE", False)
    calls = []

    def failing_load(**kwargs):
        calls.append(dict(kwargs))
        marker = snapshot.parent.parent / ".no_exist" / marker_revision / filename
        marker.parent.mkdir(parents=True)
        marker.touch()
        raise OSError("original unrelated failure")

    monkeypatch.setattr(pipeline._component_specs["processor"], "load", failing_load)
    with pytest.raises(RequiredComponentLoadError, match="original unrelated failure"):
        load_processor(pipeline, repo, cache, {"text_encoder": encoder}, local_files_only=False)
    assert len(calls) == 1 and "config" not in calls[0]
    assert pipeline.processor is None


def test_retry_failure_is_not_retried_again_and_retains_original_diagnostics(pinned_processor_snapshot, monkeypatch):
    repo, revision, cache, snapshot = pinned_processor_snapshot
    pipeline = component_pipeline(repo, revision)
    encoder = selected_encoder(pipeline, cache)
    monkeypatch.setattr(loaders.hub_constants, "HF_HUB_OFFLINE", False)
    calls = []

    def failing_load(**kwargs):
        calls.append(dict(kwargs))
        if "config" not in kwargs:
            marker = snapshot.parent.parent / ".no_exist" / revision / "processor/config.json"
            marker.parent.mkdir(parents=True)
            marker.touch()
            raise OSError("initial missing config")
        raise ValueError("invalid processor after binding")

    monkeypatch.setattr(pipeline._component_specs["processor"], "load", failing_load)
    diagnostics = {}
    with pytest.raises(RequiredComponentLoadError, match="invalid processor after binding"):
        load_components_strict(pipeline, ["processor"], required_names={"processor"}, model_id=repo,
                               dtype=torch.float32, offload_mode="none", quant_config=None, diagnostics=diagnostics,
                               component_load_kwargs={"cache_dir": str(cache), "local_files_only": False},
                               selected_components={"text_encoder": encoder})
    assert len(calls) == 2 and "config" not in calls[0] and "config" in calls[1]
    assert diagnostics["processor_config_retries"]["processor"]["error"] == "initial missing config"
    assert "initial missing config" in diagnostics["processor_config_retries"]["processor"]["traceback"]
    assert diagnostics["components_failed"][0]["error"] == "invalid processor after binding"
    assert pipeline.processor is None
