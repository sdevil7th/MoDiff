"""Historical catalog proof verifies actual package bytes, not index stat hints."""

import shutil
import subprocess

import pytest

from modiff.upstream_coverage import UpstreamCoverageError, _git_diffusers_revision


@pytest.fixture
def checkout(tmp_path):
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is required for an immutable historical source audit")
    root = tmp_path / "upstream"
    root.mkdir()
    source = root / "src/diffusers"
    (source / "modular_pipelines/cosmos").mkdir(parents=True)
    (source / "__init__.py").write_text("__version__ = 'reviewed'\n")
    decoder = source / "modular_pipelines/cosmos/decoders.py"
    decoder.write_text("class Decoder:\n    pass\n")
    (root / ".gitignore").write_text("ignored/\n__pycache__/\n")

    def run(*args):
        return subprocess.run([git, "-C", str(root), *args], check=True, capture_output=True, text=True).stdout.strip()

    run("init")
    run("config", "user.name", "Source audit fixture")
    run("config", "user.email", "source-audit@example.invalid")
    run("config", "core.autocrlf", "false")
    run("remote", "add", "origin", "https://github.com/huggingface/diffusers.git")
    run("add", ".")
    run("commit", "-m", "Exact historical source fixture")
    return root, source, decoder, run, run("rev-parse", "HEAD")


def test_exact_package_bytes_allow_clean_source_and_bytecode(checkout):
    _root, source, _decoder, _run, head = checkout
    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"not Python source")
    assert _git_diffusers_revision(source) == head


@pytest.mark.parametrize("change", ["unstaged", "staged", "index-only", "assume-unchanged", "missing", "symlink"])
def test_non_init_package_source_drift_rejects_even_when_index_hints_hide_it(checkout, change):
    _root, source, decoder, run, _head = checkout
    original = decoder.read_bytes()
    if change == "assume-unchanged":
        run("update-index", "--assume-unchanged", "src/diffusers/modular_pipelines/cosmos/decoders.py")
    if change == "missing":
        decoder.unlink()
    elif change == "symlink":
        target = decoder.with_suffix(".txt")
        target.write_bytes(original)
        decoder.unlink()
        try:
            decoder.symlink_to(target)
        except OSError:
            pytest.skip("This host cannot create symlinks")
    else:
        decoder.write_bytes(original + b"\n# non-init source drift\n")
        if change in {"staged", "index-only"}:
            run("add", "src/diffusers/modular_pipelines/cosmos/decoders.py")
        if change == "index-only":
            decoder.write_bytes(original)
    assert (source / "__init__.py").read_text() == "__version__ = 'reviewed'\n"
    with pytest.raises(UpstreamCoverageError, match="package source"):
        _git_diffusers_revision(source)


@pytest.mark.parametrize("relative", ["new_decoder.py", "ignored/extra.py", "ignored/extra.pyi"])
def test_untracked_package_python_including_ignored_files_rejects(checkout, relative):
    _root, source, _decoder, _run, _head = checkout
    path = source / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("class Unreviewed: pass\n")
    with pytest.raises(UpstreamCoverageError, match="untracked Python"):
        _git_diffusers_revision(source)


def test_historical_source_cannot_be_proven_by_an_installed_git_receipt(tmp_path, monkeypatch):
    import importlib.metadata
    from modiff import upstream_coverage
    from modiff.modular_workflow_contracts import PINNED_DIFFUSERS_REVISION

    source = tmp_path / "diffusers"
    source.mkdir()
    (source / "__init__.py").write_text("__version__ = 'unverified'\n")
    monkeypatch.setattr(upstream_coverage, "_git_diffusers_revision", lambda _source: None)
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _name: pytest.fail(
        "An installed wheel receipt cannot substitute for immutable historical source bytes"
    ))
    with pytest.raises(UpstreamCoverageError, match="unverifiable source"):
        upstream_coverage._verify_diffusers_source_revision(source, PINNED_DIFFUSERS_REVISION)


def test_git_replacement_cannot_substitute_different_source_for_reviewed_commit(checkout):
    _root, source, decoder, run, head = checkout
    decoder.write_bytes(decoder.read_bytes() + b"\n# replacement source\n")
    run("add", ".")
    run("commit", "-m", "Replacement tree")
    replacement = run("rev-parse", "HEAD")
    run("replace", head, replacement)
    run("reset", "--hard", head)
    assert run("rev-parse", "HEAD") == head
    assert b"replacement source" in decoder.read_bytes()
    with pytest.raises(UpstreamCoverageError, match="package source"):
        _git_diffusers_revision(source)
