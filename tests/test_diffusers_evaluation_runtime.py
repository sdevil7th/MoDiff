"""No-download exact-artifact admission; never trained or accelerator proof."""
import ast
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from modiff import base_runtime, diffusers_evaluation_runtime as evaluation, runtime_overlays


@pytest.fixture(autouse=True)
def isolated_selection(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.delenv(evaluation.PROFILE_ENV, raising=False)
    monkeypatch.delenv(evaluation.WHEEL_ENV, raising=False)


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    site = tmp_path / "site"
    package = site / "diffusers"
    package.mkdir(parents=True)
    metadata = site / "diffusers-0.41.0.dev0.dist-info"
    metadata.mkdir()
    wheel = tmp_path / "diffusers-0.41.0.dev0-py3-none-any.whl"
    files = {"diffusers/__init__.py": b'raise AssertionError("observer must never import model code")\n',
             "diffusers/component.py": b"reviewed = True\n",
             "diffusers-0.41.0.dev0.dist-info/METADATA": b"Name: diffusers\nVersion: 0.41.0.dev0\n",
             "diffusers-0.41.0.dev0.dist-info/WHEEL": b"Wheel-Version: 1.0\n"}
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
            (site / name).write_bytes(data)
        archive.writestr("diffusers-0.41.0.dev0.dist-info/RECORD", "")
    direct = {"url": wheel.as_uri(), "archive_info": {}}
    distribution = SimpleNamespace(version="0.41.0.dev0", _path=metadata,
        locate_file=lambda name: site / name,
        read_text=lambda name: json.dumps(direct) if name == "direct_url.json" else "record\n" if name == "RECORD" else None)
    contract = {"kind": "official-source-evaluation", "distribution": "diffusers", "version": distribution.version,
                "repository": "https://github.com/huggingface/diffusers.git", "revision": "a" * 40,
                "sourceTree": "b" * 40, "wheelFilename": wheel.name, "wheelByteSize": wheel.stat().st_size,
                "wheelSha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schemaVersion": 1, "profiles": {"fixture": contract}}))
    monkeypatch.setattr(evaluation, "PROFILE_PATH", profile)
    monkeypatch.setattr(evaluation, "find_spec", lambda _name: SimpleNamespace(
        origin=str(package / "__init__.py"), submodule_search_locations=[str(package)]))
    monkeypatch.setenv(evaluation.PROFILE_ENV, "fixture")
    monkeypatch.setenv(evaluation.WHEEL_ENV, str(wheel))
    return SimpleNamespace(site=site, package=package, wheel=wheel, distribution=distribution,
                           direct=direct, profile=profile, contract=contract, files=files)


def observed(artifact):
    return evaluation.selected_diffusers_evaluation_contract(distribution_resolver=lambda _name: artifact.distribution)


def test_default_never_observes_an_evaluation_distribution():
    def forbidden(_name):
        pytest.fail("Ordinary setup must not observe an evaluation package")
    assert evaluation.selected_diffusers_evaluation_contract(distribution_resolver=forbidden) is None


def test_unselected_wheel_is_not_consent_to_evaluation(artifact, monkeypatch):
    monkeypatch.delenv(evaluation.PROFILE_ENV)
    with pytest.raises(RuntimeError, match="explicit reviewed profile"):
        observed(artifact)


@pytest.mark.parametrize("selected", ["unknown", "fixture ", "../fixture"])
def test_unknown_profile_never_relaxes_ordinary_pin(artifact, monkeypatch, selected):
    monkeypatch.setenv(evaluation.PROFILE_ENV, selected)
    with pytest.raises(RuntimeError, match="not reviewed"):
        observed(artifact)


def test_exact_wheel_and_installed_body_are_verified_without_model_import(artifact):
    value = observed(artifact)
    assert value["dependency"] == "diffusers==0.41.0.dev0"
    assert value["installedFiles"] == {name: hashlib.sha256(data).hexdigest() for name, data in artifact.files.items()}
    assert value["wheelSha256"] == artifact.contract["wheelSha256"]


@pytest.mark.parametrize("fault", ["body", "extra_file", "missing_file", "version", "record", "metadata_origin", "direct_url", "hash"])
def test_forged_or_changed_installed_distribution_refuses(artifact, fault):
    if fault == "body":
        (artifact.package / "component.py").write_bytes(b"reviewed = Evil\n")
    elif fault == "extra_file":
        (artifact.package / "unreviewed.py").write_bytes(b"execute = True\n")
    elif fault == "missing_file":
        (artifact.package / "component.py").unlink()
    elif fault == "version":
        artifact.distribution.version = "0.41.0.dev1"
    elif fault == "record":
        original = artifact.distribution.read_text
        artifact.distribution.read_text = lambda name: None if name == "RECORD" else original(name)
    elif fault == "metadata_origin":
        artifact.distribution._path = artifact.site / "elsewhere"
    elif fault == "direct_url":
        artifact.direct["url"] = "https://example.invalid/other.whl"
    elif fault == "hash":
        artifact.direct["archive_info"] = {"hashes": {"sha256": "0" * 64}}
    with pytest.raises((RuntimeError, OSError)):
        observed(artifact)


def test_wheel_change_after_success_is_not_hidden_by_cached_metadata(artifact):
    observed(artifact)
    with artifact.wheel.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(RuntimeError, match="exact reviewed artifact"):
        observed(artifact)


def test_installed_body_change_after_success_is_rechecked(artifact):
    observed(artifact)
    (artifact.package / "component.py").write_bytes(b"reviewed = None\n")
    with pytest.raises(RuntimeError, match="differ"):
        observed(artifact)


def test_checkout_shadowing_is_not_installed_source_proof(artifact, monkeypatch):
    monkeypatch.setattr(evaluation, "find_spec", lambda _name: SimpleNamespace(
        origin=str(artifact.site / "checkout/__init__.py"), submodule_search_locations=[str(artifact.site / "checkout")]))
    with pytest.raises(RuntimeError, match="import resolution"):
        observed(artifact)


def test_exact_evaluation_changes_execution_without_changing_catalog(artifact, tmp_path, monkeypatch):
    project = tmp_path / "pyproject.toml"
    project.write_text('[project]\nrequires-python=">=3.12,<3.13"\ndependencies=["diffusers==0.41.0"]\n')
    monkeypatch.setattr(base_runtime, "PROJECT_METADATA", project)
    catalog = base_runtime.base_model_runtime_contract()
    status = base_runtime.base_runtime_status(version_resolver=lambda _name: artifact.distribution.version,
        distribution_resolver=lambda _name: artifact.distribution)
    assert status["verified"] and status["evaluationRuntime"]["profileId"] == "fixture"
    assert "installedFiles" not in status["evaluationRuntime"]
    assert base_runtime.base_model_runtime_contract() == catalog
    assert catalog["dependencies"] == ["diffusers==0.41.0"]
    monkeypatch.delenv(evaluation.PROFILE_ENV)
    monkeypatch.delenv(evaluation.WHEEL_ENV)
    assert not base_runtime.base_runtime_status(version_resolver=lambda _name: artifact.distribution.version)["verified"]


def test_invalid_selected_profile_returns_incompatible_status(artifact, monkeypatch):
    monkeypatch.setenv(evaluation.PROFILE_ENV, "unknown")
    status = base_runtime.base_runtime_status(version_resolver=lambda _name: "0.41.0")
    assert not status["verified"] and any("evaluation contract is invalid" in issue for issue in status["issues"])


def emitted_base_diffusers_validation():
    # Select the actual emitted statements by their AST boundaries. The base
    # census is shared by ordinary/E installs and a declared overlay provider;
    # the later provider-origin branch is independently tested in full scripts.
    tree = ast.parse(runtime_overlays._VALIDATION_SCRIPT)
    starts = [index for index, node in enumerate(tree.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "diffusers" for target in node.targets)]
    assert len(starts) == 1
    start = starts[0]
    stops = [index for index, node in enumerate(tree.body) if index > start
        and isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name) and node.test.left.id == "provider"
        and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.IsNot)
        and len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value is None]
    assert len(stops) == 1
    statements = tree.body[start:stops[0]]
    assert any(isinstance(node, ast.FunctionDef) and node.name == "verify_evaluation_installation"
               for node in statements)
    return compile(ast.Module(body=statements, type_ignores=[]), "<actual-emitted-base-validation>", "exec")


def test_emitted_validator_rechecks_selected_installed_file_census(artifact, tmp_path, monkeypatch):
    selected = observed(artifact)
    record = artifact.distribution.read_text("RECORD")
    identity = {"version": artifact.distribution.version, "declaration": selected["dependency"],
        "evaluationRuntime": selected, "directUrl": artifact.direct,
        "recordDigest": "sha256:" + hashlib.sha256(record.encode()).hexdigest(),
        "metadataOrigin": str(artifact.distribution._path)}
    monkeypatch.setattr(runtime_overlays.metadata, "distribution", lambda _name: artifact.distribution)
    section = emitted_base_diffusers_validation()
    import importlib
    namespace = {"metadata": runtime_overlays.metadata, "json": json, "hashlib": hashlib,
                 "Path": Path, "importlib": importlib, "os": os, "stat": stat, "sys": sys, "provider": None, "payload": {"binding": {"diffusers": identity}}}
    emitted_packages = [{"distribution": "bitsandbytes", "version": "fixture"}]
    namespace["observed"] = emitted_packages
    exec(section, namespace)
    assert namespace["observed"] is emitted_packages, "Evaluation checks overwrote the complete validator's package-result list"
    (artifact.package / "component.py").write_bytes(b"late mutation\n")
    with pytest.raises(RuntimeError, match="file identity drifted"):
        exec(section, namespace)


@pytest.mark.parametrize("fault", ["late_extra_module", "intermediate_reparse_metadata", "late_bytecode", "missing_no_bytecode_flag"])
def test_emitted_validator_refuses_late_census_or_ancestor_drift(artifact, monkeypatch, fault):
    # Actual tiny installed files and the real emitted validation section.
    # The reparse case supplies synthetic Win32 metadata on the actual package
    # directory; it does not claim a native Windows junction observation.
    selected = observed(artifact)
    record = artifact.distribution.read_text("RECORD")
    identity = {"version": artifact.distribution.version, "declaration": selected["dependency"],
        "evaluationRuntime": selected, "directUrl": artifact.direct,
        "recordDigest": "sha256:" + hashlib.sha256(record.encode()).hexdigest(),
        "metadataOrigin": str(artifact.distribution._path)}
    monkeypatch.setattr(runtime_overlays.metadata, "distribution", lambda _name: artifact.distribution)
    section = emitted_base_diffusers_validation()
    import importlib
    namespace = {"metadata": runtime_overlays.metadata, "json": json, "hashlib": hashlib,
        "Path": Path, "importlib": importlib, "os": os, "stat": stat, "sys": sys,
        "provider": None, "payload": {"binding": {"diffusers": identity}}}
    exec(section, namespace)
    if fault == "late_extra_module":
        (artifact.package / "unreviewed_late.py").write_bytes(b"unreviewed = True\n")
        expected = "package census drifted"
    elif fault == "late_bytecode":
        cache = artifact.package / "__pycache__"
        cache.mkdir()
        (cache / "component.cpython-312.pyc").write_bytes(b"unreviewed-bytecode")
        expected = "bytecode"
    elif fault == "missing_no_bytecode_flag":
        monkeypatch.setattr(sys, "dont_write_bytecode", False)
        expected = "no-bytecode"
    else:
        original = Path.lstat
        def reparse_details(path, *args, **kwargs):
            actual = original(path, *args, **kwargs)
            if path == artifact.package:
                return SimpleNamespace(st_mode=actual.st_mode, st_dev=actual.st_dev, st_ino=actual.st_ino,
                    st_file_attributes=getattr(actual, "st_file_attributes", 0) | 1024)
            return actual
        monkeypatch.setattr(Path, "lstat", reparse_details)
        monkeypatch.setattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024, raising=False)
        expected = "reparse point"
    with pytest.raises(RuntimeError, match=expected):
        exec(section, namespace)


def test_selected_evaluation_requires_no_bytecode_flag(artifact, monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    with pytest.raises(RuntimeError, match="no-bytecode"):
        observed(artifact)


@pytest.mark.parametrize("name", ["component.pyc", "__pycache__/component.cpython-312.pyc", "__pycache__/component.cpython-312.opt-1.pyc"])
def test_selected_evaluation_refuses_existing_bytecode(artifact, name):
    target = artifact.package / name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b"unreviewed-bytecode")
    with pytest.raises(RuntimeError, match="bytecode"):
        observed(artifact)


def test_unselected_ordinary_runtime_does_not_require_no_bytecode(monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    def forbidden(_name):
        pytest.fail("Ordinary setup must not inspect an evaluation distribution")
    assert evaluation.selected_diffusers_evaluation_contract(distribution_resolver=forbidden) is None


def _metadata_profiles(artifact, values):
    document = json.loads(artifact.profile.read_text())
    document["profiles"]["fixture"]["authoringExecutionProfileIds"] = values
    artifact.profile.write_text(json.dumps(document))


def test_authoring_metadata_keeps_exact_distribution_identity_and_base_digest(artifact, tmp_path, monkeypatch):
    before = observed(artifact)
    project = tmp_path / "project.toml"
    project.write_text('[project]\nrequires-python=">=3.12,<3.13"\ndependencies=["diffusers==0.41.0"]\n')
    monkeypatch.setattr(base_runtime, "PROJECT_METADATA", project)
    version = lambda _name: artifact.distribution.version
    distribution = lambda _name: artifact.distribution
    status = base_runtime.base_runtime_status(version_resolver=version, distribution_resolver=distribution)
    _metadata_profiles(artifact, ["flux3-klein:modular", "flux3-klein-base:modular"])
    assert observed(artifact) == before
    assert evaluation.selected_diffusers_evaluation_authoring_profile_ids(distribution_resolver=distribution) == (
        "flux3-klein:modular", "flux3-klein-base:modular")
    after = base_runtime.base_runtime_status(version_resolver=version, distribution_resolver=distribution)
    assert after["current_digest"] == status["current_digest"]
    assert after["evaluationRuntime"] == status["evaluationRuntime"]
    assert "authoringExecutionProfileIds" not in after["evaluationRuntime"]


@pytest.mark.parametrize("value", [None, "profile", [True], ["profile", "profile"], ["../profile"],
                                  ["p" + str(index) for index in range(17)]])
def test_metadata_only_profile_list_is_closed_without_affecting_base_identity(artifact, value):
    before = observed(artifact)
    _metadata_profiles(artifact, value)
    assert observed(artifact) == before
    with pytest.raises(RuntimeError, match="authoring metadata"):
        evaluation.selected_diffusers_evaluation_authoring_profile_ids(
            distribution_resolver=lambda _name: artifact.distribution)


def test_metadata_helper_without_explicit_selection_never_inspects_distribution(monkeypatch):
    monkeypatch.delenv(evaluation.PROFILE_ENV, raising=False)
    monkeypatch.delenv(evaluation.WHEEL_ENV, raising=False)
    assert evaluation.selected_diffusers_evaluation_authoring_profile_ids(
        distribution_resolver=lambda _name: pytest.fail("Unexpected package observation")) == ()


def test_metadata_helper_rechecks_full_installed_body_after_success(artifact):
    _metadata_profiles(artifact, ["flux3-klein:modular"])
    resolver = lambda _name: artifact.distribution
    assert evaluation.selected_diffusers_evaluation_authoring_profile_ids(distribution_resolver=resolver)
    (artifact.package / "component.py").write_bytes(b"changed = True\n")
    with pytest.raises(RuntimeError, match="differ"):
        evaluation.selected_diffusers_evaluation_authoring_profile_ids(distribution_resolver=resolver)


def _private_metadata_loader(identity="flux3-klein:modular", workflow="text2image"):
    # A synthetic minimal schema projection, using the actual immutable profile
    # and repository catalog. It does not construct or execute a model/node.
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    profile = DIFFUSERS_EXECUTION_PROFILES[identity]
    return profile, {"module": profile.loader_module, "action": profile.loader_action,
        "operation": {"binding": {"pipelineClass": profile.pipeline_class}, "workflowId": workflow,
                      "decomposition": "loader"},
        "params": {"repo_id": {}, "revision": {}, "execution_profile_id": {}},
        "values": {"model_type": profile.pipeline_class, "workflow_id": workflow}}


def test_real_private_profile_binding_stays_refused_without_exact_evaluation_authoring(artifact, monkeypatch):
    from modiff.operation_starters import _bind_execution_profile
    profile, loader = _private_metadata_loader()
    assert profile.public is False
    monkeypatch.delenv(evaluation.PROFILE_ENV)
    monkeypatch.delenv(evaluation.WHEEL_ENV)
    with pytest.raises(ValueError, match="does not belong"):
        _bind_execution_profile(loader, "text_to_image", profile.id)
    assert loader["params"]["repo_id"] == {}


@pytest.mark.parametrize("identity", ["flux3-klein:modular", "flux3-klein-base:modular"])
@pytest.mark.parametrize("workflow,task", [("text2image", "text_to_image"), ("image_conditioned", "edit_image")])
def test_declared_private_metadata_uses_existing_profile_artifact_workflow_checks(artifact, monkeypatch, identity, workflow, task):
    from modiff.operation_starters import _bind_execution_profile
    _metadata_profiles(artifact, [identity])
    monkeypatch.setattr(evaluation.metadata, "distribution", lambda _name: artifact.distribution)
    profile, loader = _private_metadata_loader(identity, workflow)
    _bind_execution_profile(loader, task, identity)
    assert loader["params"]["repo_id"]["value"] == {"source": "hub", "value": profile.default_repo}
    assert loader["params"]["execution_profile_id"]["value"] == identity
    assert profile.public is False and profile.live_proof is False
    with pytest.raises(ValueError, match="exact reviewed artifact"):
        _bind_execution_profile(_private_metadata_loader(identity, workflow)[1], task, identity, "unrelated/model")


def test_valid_evaluation_does_not_unhide_unlisted_private_profiles(artifact, monkeypatch):
    from modiff.operation_starters import _bind_execution_profile
    _metadata_profiles(artifact, ["flux3-klein:modular"])
    monkeypatch.setattr(evaluation.metadata, "distribution", lambda _name: artifact.distribution)
    profile, loader = _private_metadata_loader("flux3-klein-base:modular")
    with pytest.raises(ValueError, match="does not belong"):
        _bind_execution_profile(loader, "text_to_image", profile.id)


def test_request_cannot_supply_evaluation_metadata_permission():
    from modiff.operation_starters import resolve_operation_starter
    with pytest.raises(ValueError):
        resolve_operation_starter({}, [], {"pipelineClass": "Flux3DistilledModularPipeline", "task": "text_to_image",
            "executionProfileId": "flux3-klein:modular", "authoringExecutionProfileIds": ["flux3-klein:modular"]})


def test_public_profile_metadata_never_calls_evaluation_authorizer(monkeypatch):
    from modiff.operation_starters import _bind_execution_profile
    monkeypatch.setattr(evaluation, "selected_diffusers_evaluation_authoring_profile_ids",
        lambda: pytest.fail("Public metadata must not depend on evaluation availability"))
    profile, loader = _private_metadata_loader("flux2-klein:modular")
    assert profile.public is True
    _bind_execution_profile(loader, "text_to_image", profile.id)
    assert loader["params"]["repo_id"]["value"] == {"source": "hub", "value": profile.default_repo}


def test_private_metadata_invalid_artifact_refuses_before_value_projection(artifact, monkeypatch):
    from modiff.operation_starters import _bind_execution_profile
    _metadata_profiles(artifact, ["flux3-klein:modular"])
    monkeypatch.setattr(evaluation.metadata, "distribution", lambda _name: artifact.distribution)
    profile, loader = _private_metadata_loader()
    with artifact.wheel.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="authoring metadata is unavailable"):
        _bind_execution_profile(loader, "text_to_image", profile.id)
    assert loader["params"]["repo_id"] == {}


def _authoring_catalog(artifact, monkeypatch, identities):
    import modules
    from modules.ModularDiffusers import modular_utils

    _metadata_profiles(artifact, identities)
    # DTO-only fixture: supply the actual reviewed schemas on base releases
    # without these future exports. These empty identity classes cannot execute.
    registry = modular_utils._get_registry_instance()
    configs = dict(registry.get_all())
    names = {owner.__name__ for owner in configs}
    for name, config in (("Flux3ModularPipeline", modular_utils.FLUX_3_PIPELINE_CONFIG),
                         ("Flux3DistilledModularPipeline", modular_utils.FLUX_3_DISTILLED_PIPELINE_CONFIG)):
        if name not in names:
            configs[type(name, (), {})] = config
    monkeypatch.setattr(registry, "get_all", lambda: configs)
    observer = evaluation.selected_diffusers_evaluation_authoring_profile_ids
    monkeypatch.setattr(evaluation, "selected_diffusers_evaluation_authoring_profile_ids",
        lambda: observer(distribution_resolver=lambda _name: artifact.distribution))
    def build():
        from modiff.operation_catalog import build_evaluation_authoring_catalog
        return build_evaluation_authoring_catalog(modules.MODULE_MAP, catalog_resolver=lambda: {})
    return build


def test_capabilities_separate_evaluation_authoring_from_all_public_metadata(artifact, monkeypatch, tmp_path):
    import modules
    from modiff.config import CONFIG
    from modiff.server import WebServer

    _authoring_catalog(artifact, monkeypatch, ["flux3-klein:modular", "flux3-klein-base:modular"])
    paths = {key: str(tmp_path / key) for key in CONFIG.paths if key != "app_root"}
    for path in paths.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(CONFIG, "paths", {**CONFIG.paths, **paths})
    monkeypatch.setattr(WebServer, "_optimization_runtime_context", lambda self: (None, {}, {}))
    server = WebServer(modules.MODULE_MAP)
    with monkeypatch.context() as ordinary:
        ordinary.setattr(evaluation, "selected_diffusers_evaluation_authoring_profile_ids", lambda: ())
        baseline = server._build_model_capabilities_payload()
    payload = server._build_model_capabilities_payload()
    namespace = payload.pop("evaluationAuthoring")
    assert payload == baseline
    assert len(namespace["executionProfiles"]) == 2
    for capability in payload["capabilities"]:
        if capability["modelType"].startswith("Flux3"):
            assert capability["executionProfiles"] == capability["runnableModes"] == []
            assert all(capability[key] is False for key in
                ("autoEligible", "templateEligible", "galleryEligible", "liveProof"))
    filtered = server._build_model_capabilities_payload("klein-base")["evaluationAuthoring"]
    assert [profile["id"] for profile in filtered["executionProfiles"]] == ["flux3-klein-base:modular"]
    assert [row["pipelineClass"] for row in filtered["pipelineSupport"]] == ["Flux3ModularPipeline"]
    assert len(filtered["authoringRecipes"]) == 2
    assert all(row["executionProfileId"] == "flux3-klein-base:modular" for row in filtered["authoringRecipes"])
    exact = server._build_model_capabilities_payload("flux3modularpipeline")["evaluationAuthoring"]
    assert [profile["id"] for profile in exact["executionProfiles"]] == ["flux3-klein-base:modular"]
    assert "evaluationAuthoring" not in server._build_model_capabilities_payload("no-such-model")
    assert "evaluationAuthoring" not in server._build_model_capabilities_payload("flux3-klein:modular")


def test_evaluation_discovery_projects_only_exact_selected_private_profiles_and_tasks(artifact, monkeypatch):
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    from modiff.model_artifact_catalog import catalog_revision
    identities = ["flux3-klein:modular", "flux3-klein-base:modular"]
    build = _authoring_catalog(artifact, monkeypatch, identities)
    result = build()
    assert result["schemaVersion"] == 1 and result["scope"] == "expert-authoring"
    assert all(result[key] is False for key in ("autoEligible", "templateEligible", "galleryEligible"))
    profiles = result["executionProfiles"]
    assert [profile["id"] for profile in profiles] == identities
    assert all(profile["public"] is False and profile["live_proof"] is False for profile in profiles)
    assert {row["pipelineClass"] for row in result["pipelineSupport"]} == {
        DIFFUSERS_EXECUTION_PROFILES[identity].pipeline_class for identity in identities}
    for row in result["pipelineSupport"]:
        expected = [identity for identity in identities
                    if DIFFUSERS_EXECUTION_PROFILES[identity].pipeline_class == row["pipelineClass"]]
        assert {task["task"] for task in row["tasks"]} == {"text_to_image", "edit_image"}
        assert all(task["executionProfileIds"] == expected and task["execution"] == "adapter"
                   for task in row["tasks"])
    for profile in profiles:
        assert profile["repository_revisions"] == [{"repo": profile["default_repo"],
            "revision": catalog_revision(profile["default_repo"])}]


def test_evaluation_discovery_absent_without_selection_and_invalid_runtime_refuses(artifact, monkeypatch):
    build = _authoring_catalog(artifact, monkeypatch, ["flux3-klein:modular"])
    assert build()["executionProfiles"]
    artifact.distribution.version = "0.41.0"
    with pytest.raises(RuntimeError, match="version"):
        build()
    monkeypatch.delenv(evaluation.PROFILE_ENV)
    monkeypatch.delenv(evaluation.WHEEL_ENV)
    assert build() is None


@pytest.mark.parametrize("identities", [["unknown:profile"], ["z-image:auto"], ["flux3-klein:modular"] * 2])
def test_evaluation_discovery_malformed_or_unreviewed_profile_selection_fails_closed(artifact, monkeypatch, identities):
    build = _authoring_catalog(artifact, monkeypatch, identities)
    with pytest.raises((ValueError, RuntimeError), match="authoring"):
        build()


def test_evaluation_discovery_never_imports_model_libraries_constructs_nodes_or_acquires_packages(artifact, monkeypatch):
    import builtins
    import subprocess
    import huggingface_hub
    from modiff.NodeBase import NodeBase

    build = _authoring_catalog(artifact, monkeypatch, ["flux3-klein:modular"])
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "diffusers", "peft"}:
            pytest.fail("Evaluation discovery imported a model library")
        return original_import(name, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("Evaluation discovery constructed a node or acquired packages/artifacts")
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(NodeBase, "__init__", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", forbidden)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", forbidden)
    assert len(build()["executionProfiles"]) == 1


def test_evaluation_discovery_refuses_missing_registered_task_support(artifact, monkeypatch):
    from modiff import operation_catalog
    build = _authoring_catalog(artifact, monkeypatch, ["flux3-klein:modular"])
    monkeypatch.setattr(operation_catalog, "build_operation_catalog", lambda *args, **kwargs: ([], []))
    with pytest.raises(ValueError, match="exact reviewed pipeline/task"):
        build()


def test_normal_evaluation_selections_resolve_current_parent_without_enabling_pup(artifact, monkeypatch):
    import modules
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter
    from test_flux3_normal_authoring import SELECTIONS, CONTROLS

    build = _authoring_catalog(artifact, monkeypatch, ["flux3-klein:modular", "flux3-klein-base:modular"])
    metadata = build()
    assert metadata["authoringRecipes"] == SELECTIONS
    contracts, _ = build_operation_catalog(modules.MODULE_MAP, [], catalog_resolver=lambda: {})
    for selection in metadata["authoringRecipes"]:
        request = {key: value for key, value in selection.items()
                   if key not in {"rawAuthoringRecipeId", "rawAuthoringRecipeChoices"}}
        starter = resolve_operation_starter(modules.MODULE_MAP, contracts, request)
        assert starter["authoring"]["id"] == selection["authoringRecipeId"]
        assert len(starter["authoring"]["ancillaryOperations"]) == 1
        assert starter["authoring"]["ancillaryOperations"][0]["layoutPolicy"] == "append-details-preserve-layout-v1"
        assert not any(node["module"] == "modules.HuggingFaceTransformers" for node in starter["nodes"])
        ordinary = resolve_operation_starter(modules.MODULE_MAP, contracts,
            {key: value for key, value in request.items() if key != "authoringRecipeId"})
        assert "authoring" not in ordinary
        assert ordinary["nodes"] == starter["nodes"] and ordinary["edges"] == starter["edges"]
        raw = resolve_operation_starter(modules.MODULE_MAP, contracts,
            {**request, "authoringRecipeId": selection["rawAuthoringRecipeId"]})
        assert raw["authoring"]["id"] == selection["rawAuthoringRecipeId"]
        assert "layoutPolicy" not in raw["authoring"]["ancillaryOperations"][0]
        assert raw["nodes"] == starter["nodes"] and raw["edges"] == starter["edges"]
        for choice in selection.get("rawAuthoringRecipeChoices", []):
            alternative = resolve_operation_starter(modules.MODULE_MAP, contracts,
                {**request, "authoringRecipeId": choice["authoringRecipeId"]})
            assert alternative["authoring"]["id"] == choice["authoringRecipeId"]
            assert "layoutPolicy" not in alternative["authoring"]["ancillaryOperations"][0]
            assert alternative["nodes"] == starter["nodes"] and alternative["edges"] == starter["edges"]
    for profile in metadata["executionProfiles"]:
        assert profile["adapterAuthoring"]["requiredControls"] == CONTROLS
        assert profile["adapterAuthoring"]["repositoryRevisions"] == profile["repository_revisions"]
        assert profile["public"] is profile["live_proof"] is False


@pytest.mark.parametrize("fault", ["owner", "offload", "quantizer", "repository"])
def test_evaluation_adapter_policy_requires_actual_registered_controls_and_pins(artifact, monkeypatch, fault):
    from copy import deepcopy
    import modules
    from modiff import operation_catalog, model_artifact_catalog

    _authoring_catalog(artifact, monkeypatch, ["flux3-klein:modular"])
    registry = dict(modules.MODULE_MAP)
    registry["modules.ModularDiffusers"] = dict(registry["modules.ModularDiffusers"])
    owner = deepcopy(registry["modules.ModularDiffusers"]["ModelsLoader"])
    registry["modules.ModularDiffusers"]["ModelsLoader"] = owner
    if fault == "owner":
        registry["modules.ModularDiffusers"]["ModelsLoader"] = None
    elif fault == "offload":
        owner["params"]["offload_mode"]["options"] = ["group_cpu"]
    elif fault == "quantizer":
        owner["params"]["quant_config"]["required"] = True
    else:
        actual = model_artifact_catalog.catalog_profile_repository_revisions
        monkeypatch.setattr(model_artifact_catalog, "catalog_profile_repository_revisions",
            lambda profiles: {identity: [{**row, "revision": "a" * 40} for row in rows]
                              for identity, rows in actual(profiles).items()})
    with pytest.raises(ValueError, match="[Aa]dapter authoring"):
        operation_catalog.build_evaluation_authoring_catalog(registry, catalog_resolver=lambda: {})


def test_normal_declarations_preserve_all_fourteen_existing_parent_dtos(artifact, monkeypatch):
    from dataclasses import replace
    import modules
    from modiff.ancillary_execution import PRIVATE_PARENTS
    from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter

    _authoring_catalog(artifact, monkeypatch, ["flux3-klein:modular", "flux3-klein-base:modular"])
    contracts, _ = build_operation_catalog(modules.MODULE_MAP, [], catalog_resolver=lambda: {})
    def resolve_all():
        return {identity: resolve_operation_starter(modules.MODULE_MAP, contracts,
            {"pipelineClass": parent["pipelineClass"], "task": parent["task"],
             "executionProfileId": parent["profileId"], "authoringRecipeId": identity})
            for identity, parent in PRIVATE_PARENTS.items()
            if identity not in {
                "flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v1",
                "flux3-klein:edit-image:prompt-preview-nf4-authoring:v1",
                "flux3-klein-base:edit-image:prompt-preview-nf4-authoring:v1",
            }}
    current = resolve_all()
    assert len(current) == 14
    with monkeypatch.context() as previous:
        for identity in ("flux3-klein:modular", "flux3-klein-base:modular"):
            previous.setitem(DIFFUSERS_EXECUTION_PROFILES, identity,
                replace(DIFFUSERS_EXECUTION_PROFILES[identity], adapter_authoring=None))
        assert resolve_all() == current


def test_sealed_provider_reobserves_original_evaluation_via_real_base_pathfinder(artifact, monkeypatch):
    from modiff import base_runtime, runtime_overlays
    selected = observed(artifact)
    identity = {'distribution': 'diffusers', 'version': artifact.distribution.version,
        'declaration': selected['dependency'], 'evaluationRuntime': selected, 'directUrl': artifact.direct,
        'recordDigest': 'sha256:' + hashlib.sha256(artifact.distribution.read_text('RECORD').encode()).hexdigest(),
        'metadataOrigin': str(artifact.distribution._path),
        'importIdentity': {'origin': str(artifact.package / '__init__.py'), 'searchLocations': [str(artifact.package)]}}
    monkeypatch.setattr(base_runtime, 'base_model_runtime_contract', lambda: {'dependencies': ['diffusers==0.41.0']})
    monkeypatch.setattr(runtime_overlays.metadata.Distribution, 'at', lambda _origin: artifact.distribution)
    monkeypatch.setattr(evaluation, 'find_spec', lambda _name: pytest.fail('must use actual base-only PathFinder'))
    assert runtime_overlays._diffusers_identity_at(identity) == identity
    (artifact.package / 'component.py').write_bytes(b'changed source\n')
    with pytest.raises(RuntimeError, match='Installed Diffusers files differ'):
        runtime_overlays._diffusers_identity_at(identity)


def test_selected_provider_does_not_inherit_old_evaluation_authoring_permission(monkeypatch):
    from modiff import runtime_overlays
    monkeypatch.setattr(runtime_overlays, 'active_diffusers_provider', lambda: {'profileId': 'reviewed-provider'})
    monkeypatch.setattr(evaluation, 'selected_diffusers_evaluation_contract',
                        lambda **_kwargs: pytest.fail('must not reinterpret the old evaluation selector'))
    assert evaluation.selected_diffusers_evaluation_authoring_profile_ids() == ()
