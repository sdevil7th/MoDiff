"""Historical dynamic generators have one isolated, explicit source boundary."""

import os
import subprocess
import sys

import pytest

from modiff import catalog_source_audit as audit
from modiff.upstream_coverage import UpstreamCoverageError, reviewed_diffusers_source


def test_audit_child_rebuilds_paths_and_keeps_credentials_and_caches_private(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated/private/modules")
    monkeypatch.setenv("HF_TOKEN", "fixture-secret")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "fixture-secret")
    monkeypatch.setenv("HF_TOKEN_PATH", "/unrelated/operator/token")
    monkeypatch.setenv("HF_HOME", "/unrelated/operator/cache")
    source = tmp_path / "reviewed/diffusers"
    workspace = tmp_path / "owned-audit"
    before = dict(os.environ)
    result = audit._audit_environment(source, workspace, "generate_modular_block_contracts.py")
    assert result["PYTHONPATH"].split(os.pathsep) == [str(source.parent), str(audit.ROOT)]
    assert result["HF_HOME"] == str(workspace / "hf")
    assert result["HF_TOKEN_PATH"] == str(workspace / "hf/token")
    assert result["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert result["HF_HUB_CACHE"] == str(workspace / "hf/hub")
    assert result["MODIFF_MANAGED_ROOT"] == str(workspace / "managed")
    assert result["HF_HUB_OFFLINE"] == result["TRANSFORMERS_OFFLINE"] == "1"
    assert result["CUDA_VISIBLE_DEVICES"] == "-1"
    assert "HF_TOKEN" not in result and "HUGGING_FACE_HUB_TOKEN" not in result
    assert os.environ == before


def test_audit_accepts_only_the_three_known_cli_paths(tmp_path):
    with pytest.raises(UpstreamCoverageError, match="limited"):
        audit.prepare_catalog_source_audit(tmp_path / "generate_modular_block_contracts.py")


def test_historical_child_cannot_use_the_actual_installed_stable_package():
    source = reviewed_diffusers_source()
    with pytest.raises(UpstreamCoverageError, match="different Diffusers"):
        audit._verify_imported_source(source)


@pytest.mark.parametrize("name", sorted(audit._SCRIPTS))
def test_generator_requires_explicit_audit_source_before_dynamic_import(name):
    environment = dict(os.environ)
    environment.pop("MODIFF_DIFFUSERS_CATALOG_SOURCE", None)
    environment.pop(audit._CHILD, None)
    environment.pop(audit._WORKSPACE, None)
    environment["PYTHONPATH"] = str(audit.ROOT)
    result = subprocess.run([sys.executable, str(audit.ROOT / "scripts" / name), "--check"],
                            capture_output=True, text=True, env=environment, timeout=15)
    assert result.returncode != 0
    assert "requires explicit MODIFF_DIFFUSERS_CATALOG_SOURCE" in result.stderr
    assert "CATALOG_SOURCE_AUDIT" not in result.stdout
    assert "Loading modules" not in result.stdout


@pytest.mark.parametrize("name", sorted(audit._SCRIPTS))
def test_real_generator_keeps_relative_output_in_the_callers_workspace(tmp_path, name):
    source = reviewed_diffusers_source()
    caller = tmp_path / "caller workspace"
    caller.mkdir()
    environment = dict(os.environ)
    environment["MODIFF_DIFFUSERS_CATALOG_SOURCE"] = str(source)
    environment.pop(audit._CHILD, None)
    environment.pop(audit._WORKSPACE, None)
    script = audit.ROOT / "scripts" / name

    def invoke(*arguments):
        return subprocess.run([sys.executable, str(script), *arguments], cwd=caller,
                              capture_output=True, text=True, env=environment, timeout=60)

    output = caller / "generated catalog.json"
    generated = invoke("--output", output.name)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert output.is_file(), "The successful audit CLI must retain its relative output for the caller."
    rendered = output.read_bytes()
    assert rendered.startswith(b"{")
    checked = invoke("--check", "--output=" + output.name)
    assert checked.returncode == 0, checked.stdout + checked.stderr

    nested = caller / "nested directory"
    nested.mkdir()
    equal_output = nested / "equal output.json"
    written_equal = invoke("--output=nested directory/equal output.json")
    assert written_equal.returncode == 0, written_equal.stdout + written_equal.stderr
    assert equal_output.read_bytes() == rendered
    checked_separate = invoke("--check", "--output", "nested directory/equal output.json")
    assert checked_separate.returncode == 0, checked_separate.stdout + checked_separate.stderr
    equal_output.write_text("{}\n")
    stale = invoke("--check", "--output=nested directory/equal output.json")
    assert stale.returncode != 0
    assert "is stale" in stale.stderr


@pytest.mark.parametrize("arguments", [("--output",), ("--output", "--check"), ("--out",)])
def test_real_generator_missing_output_argument_keeps_argparse_error(tmp_path, arguments):
    source = reviewed_diffusers_source()
    environment = dict(os.environ)
    environment["MODIFF_DIFFUSERS_CATALOG_SOURCE"] = str(source)
    environment.pop(audit._CHILD, None)
    environment.pop(audit._WORKSPACE, None)
    result = subprocess.run([sys.executable, str(audit.ROOT / "scripts/generate_modular_block_contracts.py"),
                             *arguments], cwd=tmp_path, capture_output=True, text=True,
                            env=environment, timeout=60)
    assert result.returncode == 2
    assert "argument --output: expected one argument" in result.stderr


def test_audit_output_rewrite_changes_only_explicit_relative_output_paths(tmp_path):
    absolute = str(tmp_path / "already absolute.json")
    arguments = ["--check", "--output", "relative name.json", "--output=second path.json",
                 "--output", absolute, "--unrelated=relative.json"]
    assert audit._caller_output_arguments(arguments, tmp_path) == [
        "--check", "--output", str(tmp_path / "relative name.json"),
        "--output=" + str(tmp_path / "second path.json"), "--output", absolute,
        "--unrelated=relative.json",
    ]
    assert arguments[2] == "relative name.json"
    assert audit._caller_output_arguments(["--output"], tmp_path) == ["--output"]
    assert audit._caller_output_arguments(["--output", "--check"], tmp_path) == ["--output", "--check"]


@pytest.mark.parametrize("option", ["--o", "--ou", "--out", "--outp", "--outpu", "--output"])
def test_audit_preserves_argparse_output_abbreviations(tmp_path, option):
    assert audit._caller_output_arguments([option, "relative name.json"], tmp_path) == [
        option, str(tmp_path / "relative name.json"),
    ]
    assert audit._caller_output_arguments([option + "=equal output.json"], tmp_path) == [
        option + "=" + str(tmp_path / "equal output.json"),
    ]
    assert audit._caller_output_arguments([option, "--check"], tmp_path) == [option, "--check"]


@pytest.mark.parametrize("arguments, filename", [
    (("--out", "abbreviated output.json"), "abbreviated output.json"),
    (("--o=equal abbreviated.json",), "equal abbreviated.json"),
    (("--outp", "-1"), "-1"),
])
def test_real_generator_preserves_accepted_output_abbreviations(tmp_path, arguments, filename):
    source = reviewed_diffusers_source()
    environment = dict(os.environ)
    environment["MODIFF_DIFFUSERS_CATALOG_SOURCE"] = str(source)
    environment.pop(audit._CHILD, None)
    environment.pop(audit._WORKSPACE, None)
    script = audit.ROOT / "scripts/generate_modular_block_contracts.py"
    generated = subprocess.run([sys.executable, str(script), *arguments], cwd=tmp_path,
                               capture_output=True, text=True, env=environment, timeout=60)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert (tmp_path / filename).is_file()
    checked = subprocess.run([sys.executable, str(script), "--check", *arguments], cwd=tmp_path,
                             capture_output=True, text=True, env=environment, timeout=60)
    assert checked.returncode == 0, checked.stdout + checked.stderr
