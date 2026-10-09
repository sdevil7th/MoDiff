"""Mandatory safety stays exact, owned, fail-closed, and separate from qualification."""
import ast
import hashlib
import os
import sys
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import pytest
import torch

from modiff.cosmos_safety_contract import (
    COSMOS_SAFETY_BUNDLE_MEMBER, COSMOS_SAFETY_RUNTIME_PROFILE_ID,
    cosmos_safety_artifacts, verify_cosmos_safety_snapshot,
)
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES, assert_optional_runtime_distribution_compatibility,
    project_optional_runtime_qualification,
)
from modules.ModularDiffusers import cosmos_safety as safety
from modules.ModularDiffusers import loaders
from modules.ModularDiffusers.loaders import annotate_modular_loader_outputs
from modules.ModularDiffusers.route_state import issue_pipeline_instance_token, require_component_binding
from utils.memory_menager import _model_size_bytes

MODEL_TYPE = "Cosmos3OmniModularPipeline"


def artifact_snapshot(tmp_path, *, global_blob=False):
    artifact = deepcopy(cosmos_safety_artifacts()[0])
    artifact["files"] = []
    snapshot = tmp_path / "hub" / ("models--" + artifact["repository"].replace("/", "--")) / "snapshots" / artifact["revision"]
    snapshot.mkdir(parents=True)
    raw = b"reviewed safety bytes"
    digest = hashlib.sha256(raw).hexdigest()
    name = "e9" + "2" * 62
    target = snapshot if not global_blob else tmp_path / "hub/blobs/e9"
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot / "marker.txt"
    if global_blob:
        (target / name).write_bytes(raw)
        path.symlink_to(target / name)
    else:
        path.write_bytes(raw)
    artifact["marker"] = "marker.txt"
    artifact["files"] = [{"path": "marker.txt", "byteSize": len(raw), "blobHash": digest}]
    return snapshot, artifact, path


@pytest.mark.parametrize("global_blob", [False, True])
def test_safety_verifies_content_not_storage_name(tmp_path, global_blob):
    snapshot, artifact, _ = artifact_snapshot(tmp_path, global_blob=global_blob)
    assert verify_cosmos_safety_snapshot(snapshot, artifact) == snapshot


@pytest.mark.parametrize("global_blob", [False, True])
@pytest.mark.parametrize("changed_field", ["atime", "mode", "inode", "device", "size", "mtime", "ctime"])
def test_safety_rechecks_alias_identity_after_read_without_rejecting_access_time(
    tmp_path, monkeypatch, global_blob, changed_field,
):
    snapshot, artifact, alias = artifact_snapshot(tmp_path, global_blob=global_blob)
    target = alias.resolve(strict=True)
    original_open = Path.open
    original_lstat = Path.lstat
    bytes_read = bytearray()

    @contextmanager
    def reading_handle(handle):
        with handle:
            def read(size):
                data = handle.read(size)
                bytes_read.extend(data)
                return data
            yield SimpleNamespace(read=read, fileno=handle.fileno)

    def open_file(path, *args, **kwargs):
        handle = original_open(path, *args, **kwargs)
        return reading_handle(handle) if path == target and args == ("rb",) else handle

    def alias_stat(path, *args, **kwargs):
        value = original_lstat(path, *args, **kwargs)
        if path != alias or not bytes_read:
            return value
        fields = list(value)
        index = {"mode": 0, "inode": 1, "device": 2, "size": 6, "atime": 7, "mtime": 8, "ctime": 9}[changed_field]
        fields[index] += 1
        nanoseconds = {f"st_{field}_ns": getattr(value, f"st_{field}_ns") for field in ("atime", "mtime", "ctime")}
        if changed_field in {"atime", "mtime", "ctime"}:
            nanoseconds[f"st_{changed_field}_ns"] += 1_000_000_000
        return os.stat_result(fields, nanoseconds)

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(Path, "lstat", alias_stat)
    if changed_field == "atime":
        assert verify_cosmos_safety_snapshot(snapshot, artifact) == snapshot
    else:
        with pytest.raises(ValueError, match="reviewed digest"):
            verify_cosmos_safety_snapshot(snapshot, artifact)
    assert bytes(bytes_read) == b"reviewed safety bytes"


def test_safety_checks_git_blob_digest_for_small_config_bytes(tmp_path):
    snapshot, artifact, path = artifact_snapshot(tmp_path)
    raw = path.read_bytes()
    artifact["files"][0]["blobHash"] = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    assert verify_cosmos_safety_snapshot(snapshot, artifact) == snapshot
    path.write_bytes(b"x" * len(raw))
    with pytest.raises(ValueError, match="digest"):
        verify_cosmos_safety_snapshot(snapshot, artifact)


@pytest.mark.parametrize("fault", ["digest", "size", "missing", "outside", "foreign_repository", "revision"])
def test_safety_rejects_incomplete_or_changed_artifact(tmp_path, fault):
    snapshot, artifact, path = artifact_snapshot(tmp_path, global_blob=True)
    if fault == "digest":
        path.resolve().write_bytes(b"x" * artifact["files"][0]["byteSize"])
    elif fault == "size":
        path.resolve().write_bytes(b"x")
    elif fault == "missing":
        path.unlink()
    elif fault in {"outside", "foreign_repository"}:
        target = tmp_path / ("outside" if fault == "outside" else "hub/models--foreign--repo/blobs/other")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"reviewed safety bytes")
        path.unlink()
        path.symlink_to(target)
    else:
        artifact["revision"] = "b" * 40
    with pytest.raises((ValueError, OSError)):
        verify_cosmos_safety_snapshot(snapshot, artifact)


def test_optional_runtime_has_exact_single_provider_package_closure():
    profile = OPTIONAL_RUNTIME_PROFILES[COSMOS_SAFETY_RUNTIME_PROFILE_ID]
    assert profile.label == "Cosmos Guardrail 0.3.1"
    assert profile.contract_state == "candidate_unqualified"
    assert not any((profile.cutover_ready, profile.install_action_available, profile.activation_available))
    assert {value.distribution for value in profile.packages} == {
        "cosmos-guardrail", "nltk", "better-profanity", "retinaface-py", "opencv-python", "scikit-image",
        "lazy-loader", "tifffile", "joblib", "defusedxml",
    }
    assert "opencv-python-headless" in profile.incompatible_distributions
    assert "cv2" not in {value.import_name for value in profile.base_packages}
    assert {value["distribution"] for value in profile.artifact_locks} == {value.distribution for value in profile.packages}
    assert profile.to_spec_dict()["incompatibleDistributions"] == list(profile.incompatible_distributions)
    assert len(profile.spec_digest) == 71


def test_linux_cosmos_package_qualification_keeps_other_targets_pending():
    candidate = OPTIONAL_RUNTIME_PROFILES[COSMOS_SAFETY_RUNTIME_PROFILE_ID]
    projected = project_optional_runtime_qualification(candidate, platform_name="linux", machine="x86_64")
    assert projected is candidate and projected.spec_digest == candidate.spec_digest
    assert candidate.contract_for_target(platform_name="linux", machine="x86_64").contract_state == "qualified"
    target = projected.contract_for_target(platform_name="linux", machine="x86_64")
    assert target.contract_state == "qualified" and target.install_action_available and target.activation_available
    for platform in ("macos", "windows"):
        target = projected.contract_for_target(platform_name=platform, machine="x86_64")
        assert target.contract_state == "candidate_unqualified"
        assert not target.install_action_available and not target.activation_available


def test_cosmos_exact_linux_wheel_plan_is_complete_and_platform_bounded():
    from modiff import optimization_packages as packages
    from packaging.utils import parse_wheel_filename
    profile = OPTIONAL_RUNTIME_PROFILES[COSMOS_SAFETY_RUNTIME_PROFILE_ID]
    tags = set()
    for lock in profile.artifact_locks:
        tags.update(parse_wheel_filename(lock["filename"])[3])
    with patch.object(packages, "_platform_name", return_value="linux"), patch.object(packages, "_machine_name", return_value="x86_64"), patch("packaging.tags.sys_tags", return_value=iter(tags)):
        plan = packages._artifact_install_plan(profile)
    assert {row["distribution"] for row in plan} == {row.distribution for row in profile.packages}
    assert len(plan) == 10 and sum(row["byteSize"] for row in plan) == 91_263_675
    with patch.object(packages, "_platform_name", return_value="windows"), patch.object(packages, "_machine_name", return_value="x86_64"):
        with pytest.raises(RuntimeError, match="incomplete"):
            packages._artifact_install_plan(profile)


def test_exact_qualification_codec_branch_uses_real_cpu_processing_without_default_model_load():
    pytest.importorskip("cosmos_guardrail")
    pytest.importorskip("skimage", reason="Exact scikit-image wheel is not installed in this private CPU source-review environment")
    path = Path(__file__).resolve().parents[1] / "scripts/qualify_optional_runtime.py"
    tree = ast.parse(path.read_text())
    workload = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "_WORKLOAD_SCRIPT" for target in node.targets))
    node = next(node for node in ast.parse(workload).body if isinstance(node, ast.If) and
                "cosmos-guardrail" in ast.unparse(node.test))
    namespace = {"candidate": OPTIONAL_RUNTIME_PROFILES[COSMOS_SAFETY_RUNTIME_PROFILE_ID]}
    # The tested codec is the installed CPU cv2. Actual GUI-provider runtime
    # qualification still requires the separately installed exact wheel; only
    # its distribution collision check is substituted for this source test.
    with patch("modiff.optional_runtimes.assert_optional_runtime_distribution_compatibility"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    assert namespace["cosmos_safety"]["pngRoundtrip"] == "passed"
    assert namespace["cosmos_safety"]["cannyShape"] == [16, 16]


@pytest.mark.parametrize("conflicting", ["opencv-python-headless", "opencv-contrib-python", "opencv-contrib-python-headless"])
def test_single_provider_rejects_installed_conflicting_distribution(conflicting):
    def version(name):
        if name == conflicting:
            return "5.0.0.93"
        raise metadata.PackageNotFoundError(name)
    with pytest.raises(RuntimeError, match="single provider"):
        assert_optional_runtime_distribution_compatibility(
            OPTIONAL_RUNTIME_PROFILES[COSMOS_SAFETY_RUNTIME_PROFILE_ID], version_resolver=version,
        )


def test_missing_conflicting_distribution_is_allowed_but_unreadable_metadata_is_not():
    def missing(name):
        raise metadata.PackageNotFoundError(name)
    profile = OPTIONAL_RUNTIME_PROFILES[COSMOS_SAFETY_RUNTIME_PROFILE_ID]
    assert_optional_runtime_distribution_compatibility(profile, version_resolver=missing)
    with pytest.raises(OSError, match="metadata failure"):
        assert_optional_runtime_distribution_compatibility(
            profile, version_resolver=lambda _name: (_ for _ in ()).throw(OSError("metadata failure")),
        )


def test_non_cosmos_stage_never_imports_optional_package():
    with patch.dict(sys.modules, {"cosmos_guardrail": None}):
        safety.attach_cosmos_safety_checker(object(), {}, model_type="QwenImageModularPipeline")


@pytest.fixture
def genuine_factory(tmp_path, monkeypatch):
    """Use the genuine optional package; tiny CPU models stand in for pretrained weights."""
    upstream = pytest.importorskip("cosmos_guardrail.cosmos_guardrail")
    nltk = pytest.importorskip("nltk")
    from better_profanity import profanity
    snapshots = []
    artifacts = deepcopy(cosmos_safety_artifacts())
    for index, artifact in enumerate(artifacts):
        snapshot = tmp_path / str(index)
        snapshot.mkdir()
        snapshots.append(snapshot)
    root = snapshots[0] / "blocklist"
    for name, raw in {
        "custom/blocked": b"blockedword\n", "whitelist/allowed": b"allowedword\n", "exact_match/blocked": b"forbiddenphrase\n",
        "nltk_data/tokenizers/punkt_tab/english/abbrev_types.txt": b"",
        "nltk_data/tokenizers/punkt_tab/english/collocations.tab": b"",
        "nltk_data/tokenizers/punkt_tab/english/ortho_context.tab": b"",
        "nltk_data/tokenizers/punkt_tab/english/sent_starters.txt": b"",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    archive = root / "nltk_data/corpora/wordnet.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as output:
        for name in ("lexnames", "index.sense", "index.noun", "index.verb", "index.adj", "index.adv", "data.noun", "data.verb", "data.adj", "data.adv", "noun.exc", "verb.exc", "adj.exc", "adv.exc"):
            output.writestr("wordnet/" + name, b"")
    with zipfile.ZipFile(root / "nltk_data/tokenizers/punkt_tab.zip", "w") as output:
        for path in (root / "nltk_data/tokenizers/punkt_tab/english").iterdir():
            output.writestr("punkt_tab/english/" + path.name, path.read_bytes())
    artifacts[0]["files"] = [{"path": path.relative_to(snapshots[0]).as_posix()} for path in root.rglob("*") if path.is_file()]
    calls = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(safety, "require_cosmos_safety_runtime"))
        stack.enter_context(patch.object(safety, "cosmos_safety_artifacts", return_value=artifacts))
        stack.enter_context(patch.object(safety, "exact_cached_snapshot_path", side_effect=snapshots))
        stack.enter_context(patch.object(safety, "verify_cosmos_safety_snapshot", side_effect=lambda p, _a: p))
        stack.enter_context(patch("transformers.AutoModelForCausalLM.from_pretrained", side_effect=lambda *a, **k: calls.append(("model", a, k)) or torch.nn.Linear(2, 2)))
        stack.enter_context(patch("transformers.AutoTokenizer.from_pretrained", side_effect=lambda *a, **k: calls.append(("tokenizer", a, k)) or object()))
        stack.enter_context(patch.object(upstream, "RetinaFace", side_effect=lambda **_k: torch.nn.Linear(2, 2)))
        stack.enter_context(patch.object(upstream, "load_model", side_effect=lambda model, _path: model))
        paths_before = list(nltk.data.path)
        profanity_before = list(profanity.CENSOR_WORDSET)
        config_before = deepcopy(upstream.cfg_re50)
        checker = safety.build_cosmos_safety_checker(owner_node_id="owner", model_type=MODEL_TYPE)
        yield checker, upstream, calls
        assert nltk.data.path == paths_before
        assert profanity.CENSOR_WORDSET == profanity_before
        assert upstream.cfg_re50 == config_before


def sealed_bundle(checker, *, owner="owner", model_type=MODEL_TYPE):
    model_id = safety.memory_manager.add(checker)
    bundle = {COSMOS_SAFETY_BUNDLE_MEMBER: {"model_id": model_id, "owner_node_id": owner}}
    outputs = {"pipeline_components": bundle}
    token = issue_pipeline_instance_token(model_type=model_type, repo_id="owner/cosmos", repo_source="hub", revision="a" * 40)
    annotate_modular_loader_outputs(outputs, repo_id="owner/cosmos", repo_source="hub", model_type=model_type,
                                   revision="a" * 40, trust_remote_code=False, pipeline_instance_token=token)
    return model_id, bundle


def test_genuine_factory_owns_models_and_preserves_package_methods(genuine_factory):
    checker, upstream, calls = genuine_factory
    assert isinstance(checker, upstream.CosmosSafetyChecker)
    assert checker.check_text_safety.__func__ is upstream.CosmosSafetyChecker.check_text_safety
    assert checker.check_video_safety.__func__ is upstream.CosmosSafetyChecker.check_video_safety
    assert _model_size_bytes(checker) == sum(p.numel() * p.element_size() for p in checker.parameters()) > 0
    assert checker.to("cpu") is checker
    assert str(checker.device) == "cpu"
    assert all(kwargs["local_files_only"] is True and kwargs["trust_remote_code"] is False for _, _, kwargs in calls)
    assert calls[0][2]["use_safetensors"] is True
    assert checker.text_guardrail.safety_models[0].is_safe("allowedword") == (True, "Input is safe")
    assert checker.text_guardrail.safety_models[0].is_safe("blockedword")[0] is False


def test_genuine_text_guard_failure_propagates_instead_of_allowing(genuine_factory):
    checker, _, _ = genuine_factory
    model = checker.text_guardrail.safety_models[1]
    model.extract_label_and_categories = lambda _prompt: (_ for _ in ()).throw(RuntimeError("moderation failed"))
    with pytest.raises(RuntimeError, match="moderation failed"):
        checker.check_text_safety("allowedword")
    assert checker.check_text_safety("blockedword") is False


def test_same_managed_checker_attaches_to_each_stage_and_retains_output_transform(genuine_factory):
    checker, _, _ = genuine_factory
    model_id, bundle = sealed_bundle(checker)
    try:
        stages = [SimpleNamespace(enable_safety_checker=lambda value: attached.append(value)) for _ in range(3)]
        attached = []
        with patch.object(safety, "require_cosmos_safety_runtime"):
            for stage in stages:
                safety.attach_cosmos_safety_checker(stage, bundle, model_type=MODEL_TYPE)
        assert attached == [checker] * 3
        checker.video_guardrail.postprocessors[0].postprocess = lambda value: value + 1
        assert checker.check_video_safety(4) == 5
    finally:
        safety.memory_manager.remove(model_id)


@pytest.mark.parametrize("fault", ["unsealed", "changed_id", "changed_owner", "different_model", "evicted"])
def test_stage_rejects_stale_foreign_or_mutated_checker(genuine_factory, fault):
    checker, _, _ = genuine_factory
    model_id, bundle = sealed_bundle(checker)
    try:
        if fault == "unsealed":
            bundle = {COSMOS_SAFETY_BUNDLE_MEMBER: bundle[COSMOS_SAFETY_BUNDLE_MEMBER]}
        elif fault == "changed_id":
            bundle[COSMOS_SAFETY_BUNDLE_MEMBER]["model_id"] = "foreign"
        elif fault == "changed_owner":
            bundle[COSMOS_SAFETY_BUNDLE_MEMBER]["owner_node_id"] = "foreign"
        elif fault == "evicted":
            safety.memory_manager.remove(model_id)
        with patch.object(safety, "require_cosmos_safety_runtime"), pytest.raises(ValueError):
            safety.attach_cosmos_safety_checker(
                SimpleNamespace(enable_safety_checker=lambda _value: pytest.fail("foreign checker installed")), bundle,
                model_type="Cosmos3DistilledModularPipeline" if fault == "different_model" else MODEL_TYPE,
            )
    finally:
        safety.memory_manager.remove(model_id)


@pytest.fixture
def ordinary_cosmos_loader(monkeypatch, genuine_factory):
    """Normal ModelsLoader/NodeBase cache with genuine, tiny owned checkers."""
    original, _, _ = genuine_factory
    builds, instances, nodes = [], [], []

    class EmptyPipeline:
        pretrained_component_names = []
        _component_specs = {}
        components = {}
        def update_components(self, **values):
            assert not values

    def instantiate(*_args, **_kwargs):
        owner = EmptyPipeline()
        instances.append(owner)
        return owner

    def build(*, owner_node_id, model_type):
        checker = type(original)()
        checker._modiff_cosmos_owned_checker = True
        checker._modiff_cosmos_owner = owner_node_id
        checker._modiff_cosmos_model_type = model_type
        builds.append(checker)
        return checker

    monkeypatch.setattr(safety, "build_cosmos_safety_checker", build)
    monkeypatch.setattr(loaders, "components", SimpleNamespace(collections={}, _lookup_ids=lambda **_: []))
    monkeypatch.setattr(loaders, "_instantiate_reviewed_builtin_pipeline", instantiate)
    monkeypatch.setattr(loaders, "_reviewed_builtin_workflow_id", lambda *_: None)
    monkeypatch.setattr(loaders, "_reviewed_loader_component_outputs", lambda *_: ())
    monkeypatch.setattr(loaders, "reviewed_modular_weight_variant", lambda *_: None)
    monkeypatch.setattr(loaders, "_loader_text_encoder_component_names", lambda *_: [])
    monkeypatch.setattr(loaders, "pin_modular_component_revisions", lambda *_: {})
    monkeypatch.setattr(loaders, "_primary_component_cache_dirs", lambda *_: {})
    monkeypatch.setattr(loaders, "configure_components_manager_offload", lambda *_a, **_k: None)
    monkeypatch.setattr(loaders, "load_components_strict", lambda *_a, **_k: None)
    monkeypatch.setattr(loaders, "apply_modular_runtime_policy", lambda *_a, **_k: {"attention": None, "vae": None})
    monkeypatch.setattr(loaders, "place_pipeline_components", lambda *_a, **_k: [])
    monkeypatch.setattr(loaders, "configure_required_regional_compile", lambda *_a, **_k: None)
    monkeypatch.setattr(loaders, "record_pipeline_component_runtime_policy", lambda *_a, **_k: None)

    def node(owner="owner"):
        value = loaders.ModelsLoader(owner)
        value.default_params = {}
        value._skip_params_check = True
        monkeypatch.setattr(value, "_effective_builtin_selector", lambda **kwargs: (kwargs["repo_id"], "a" * 40))
        monkeypatch.setattr(value, "_preflight_reviewed_builtin_selection", lambda **_: (
            "hub", "owner/cosmos", "a" * 40, "modular_model_index.json", {"_class_name": MODEL_TYPE},
        ))
        nodes.append(value)
        return value

    kwargs = dict(model_type=MODEL_TYPE, repo_id={"source": "hub", "value": "owner/cosmos"},
                  revision="a" * 40, device="cpu", dtype=torch.float32, auto_offload=False, offload_mode="none")
    yield node, kwargs, builds, instances
    for value in nodes:
        for model_id in list(value._mm_models):
            safety.memory_manager.remove(model_id)
        value._mm_models.clear()


def test_ordinary_loader_cache_keeps_one_checker_and_eviction_reloads(ordinary_cosmos_loader):
    make, kwargs, builds, instances = ordinary_cosmos_loader
    node = make()
    first = node(**kwargs)["pipeline_components"]
    member = first[COSMOS_SAFETY_BUNDLE_MEMBER]
    assert node._mm_models == [member["model_id"]]
    assert safety.memory_manager.get_model(member["model_id"]) is builds[0]
    assert node(**kwargs)["pipeline_components"] is first
    assert len(builds) == len(instances) == 1
    assert node._cache_reason == "unchanged_inputs"
    safety.memory_manager.remove(member["model_id"])
    replacement = node(**kwargs)["pipeline_components"]
    assert node._cache_reason == "models_evicted"
    assert require_component_binding(replacement, label="replacement") is not require_component_binding(first, label="original")
    assert replacement[COSMOS_SAFETY_BUNDLE_MEMBER]["model_id"] != member["model_id"]
    assert len(builds) == len(instances) == 2


def test_loader_owner_reassignment_and_separate_owners_preserve_identity(ordinary_cosmos_loader):
    make, kwargs, builds, _ = ordinary_cosmos_loader
    node = make()
    bundle = node(**kwargs)["pipeline_components"]
    model_id = bundle[COSMOS_SAFETY_BUNDLE_MEMBER]["model_id"]
    token = require_component_binding(bundle, label="original")
    node.rebind_cache_owner("adopted-owner")
    assert node.node_id == bundle[COSMOS_SAFETY_BUNDLE_MEMBER]["owner_node_id"] == builds[0]._modiff_cosmos_owner == "adopted-owner"
    assert node(**kwargs)["pipeline_components"] is bundle
    assert require_component_binding(bundle, label="adopted") is token
    attached = []
    safety.attach_cosmos_safety_checker(SimpleNamespace(enable_safety_checker=attached.append), bundle, model_type=MODEL_TYPE)
    assert attached == [builds[0]]
    other = make("other-owner")
    foreign = other(**kwargs)["pipeline_components"]
    assert foreign[COSMOS_SAFETY_BUNDLE_MEMBER]["model_id"] != model_id
    assert builds[1] is not builds[0]
    assert builds[1]._modiff_cosmos_owner == "other-owner"


def test_cached_loader_rechecks_runtime_before_return(ordinary_cosmos_loader, monkeypatch):
    make, kwargs, builds, _ = ordinary_cosmos_loader
    node = make()
    original = node(**kwargs)
    monkeypatch.setattr(safety, "require_cosmos_safety_runtime", lambda: (_ for _ in ()).throw(RuntimeError("runtime withdrawn")))
    with pytest.raises(RuntimeError, match="runtime withdrawn"):
        node(**kwargs)
    assert node.output is original
    assert len(builds) == 1


@pytest.mark.parametrize("failure", ["component_load", "bundle_publication"])
def test_loader_failure_never_publishes_or_leaks_checker(ordinary_cosmos_loader, monkeypatch, failure):
    make, kwargs, builds, _ = ordinary_cosmos_loader
    node = make()
    def fail(*_args, **_kwargs):
        raise RuntimeError("required publication failed")
    monkeypatch.setattr(loaders, "load_components_strict" if failure == "component_load" else "annotate_modular_loader_outputs", fail)
    with pytest.raises(RuntimeError, match="required publication failed"):
        node(**kwargs)
    assert len(builds) == 1
    assert node._mm_models == []
    assert not node._cache_valid
    assert not any(safety.memory_manager.get_model(key) is builds[0] for key in safety.memory_manager.cache)
    assert all(value is None for value in node.output.values())
