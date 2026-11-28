"""Hub source resolution is metadata inspection, not staging or approval."""

from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def online_hub_metadata(monkeypatch):
    # Each test controls the Hub's offline flag independently of the test runner.
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)


@pytest.mark.parametrize("source,revision", [
    ("owner/repo", "a" * 40),
    ("https://huggingface.co/owner/repo/tree/" + "a" * 40, None),
    ("https://huggingface.co/owner/repo", "a" * 40),
])
def test_offline_exact_hub_revision_resolves_without_network_or_staging(monkeypatch, source, revision):
    from modiff.custom_extension_source import resolve_hub_extension

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: pytest.fail("No metadata request"))
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: pytest.fail("No source staging"))
    assert resolve_hub_extension(source, revision) == {
        "kind": "hub", "source": "owner/repo", "requestedRevision": "a" * 40, "revision": "a" * 40,
    }


@pytest.mark.parametrize("revision", [None, "main", "release", "A" * 40, "a" * 39])
def test_offline_mutable_or_noncanonical_revisions_require_an_exact_commit(monkeypatch, revision):
    from modiff.custom_extension_source import resolve_hub_extension

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: pytest.fail("No metadata request"))
    with pytest.raises(ValueError, match="offline.*40-character"):
        resolve_hub_extension("owner/repo", revision)


def test_hub_url_resolves_without_staging_or_import(monkeypatch):
    from modiff.custom_extension_source import resolve_hub_extension

    calls = []

    def info(_self, repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", info)
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: pytest.fail("No source download"))
    result = resolve_hub_extension("https://huggingface.co/example/block/tree/reviewed")
    assert result == {"kind": "hub", "source": "example/block", "requestedRevision": "reviewed", "revision": "a" * 40}
    assert calls == [("example/block", {"revision": "reviewed", "timeout": 20})]


@pytest.mark.parametrize(
    "source,revision",
    [
        ("https://huggingface.co.evil.test/owner/repo", None),
        ("https://user:secret@huggingface.co/owner/repo", None),
        ("https://huggingface.co/owner/repo?token=secret", None),
        ("http://huggingface.co/owner/repo", None),
        ("https://huggingface.co:443/owner/repo", None),
        ("https://huggingface.co/datasets/owner/repo", None),
        ("https://huggingface.co/owner/repo/blob/main/block.py", None),
        ("https://huggingface.co/owner/repo/tree/main/extra", None),
        ("https://huggingface.co/owner/repo/tree/main", "different"),
        ("../owner/repo", None),
        ("owner/repo", "bad\nrevision"),
        ("owner/repo", 42),
    ],
)
@pytest.mark.parametrize("offline", [False, True])
def test_invalid_sources_fail_before_network(monkeypatch, source, revision, offline):
    from modiff.custom_extension_source import resolve_hub_extension

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", offline)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: pytest.fail("No network"))
    with pytest.raises(ValueError):
        resolve_hub_extension(source, revision)


def test_branches_are_pinned_and_exact_requests_cannot_be_retargeted(monkeypatch):
    from modiff.custom_extension_source import resolve_hub_extension

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: SimpleNamespace(sha="b" * 40))
    assert resolve_hub_extension("owner/repo")["requestedRevision"] == "main"
    assert resolve_hub_extension("owner/repo", "release")["revision"] == "b" * 40
    with pytest.raises(ValueError, match="different revision"):
        resolve_hub_extension("owner/repo", "a" * 40)
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", lambda *a, **k: SimpleNamespace(sha="main"))
    with pytest.raises(ValueError, match="40-character"):
        resolve_hub_extension("owner/repo")


def test_resolution_errors_do_not_expose_remote_details(monkeypatch):
    from modiff.custom_extension_source import resolve_hub_extension

    def fail(*args, **kwargs):
        raise RuntimeError("https://user:secret@example.test/private")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", fail)
    with pytest.raises(ValueError, match="Check the repository") as error:
        resolve_hub_extension("owner/repo")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("ref,rows,expected", [
    (None, [("a", "HEAD")], "a"),
    ("main", [("b", "refs/heads/main")], "b"),
    ("release", [("c", "refs/tags/release"), ("d", "refs/tags/release^{}")], "d"),
    ("release", [("a", "refs/heads/release"), ("c", "refs/tags/release"), ("d", "refs/tags/release^{}")], None),
])
def test_git_resolution_is_pinned_bounded_and_rejects_ambiguity(monkeypatch, ref, rows, expected):
    from modiff.custom_extension_source import resolve_git_extension
    def run(args, **kwargs):
        assert args[:4] == ["git", "-c", "credential.interactive=false", "ls-remote"]
        assert kwargs["timeout"] == 30
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        return SimpleNamespace(returncode=0, stdout="".join(f"{sha * 40}\t{name}\n" for sha, name in rows).encode())
    monkeypatch.setattr("subprocess.run", run)
    if expected:
        assert resolve_git_extension("https://example.com/nodes.git", ref)["revision"] == expected * 40
    else:
        with pytest.raises(ValueError, match="ambiguous"):
            resolve_git_extension("https://example.com/nodes.git", ref)


@pytest.mark.parametrize("source,revision", [
    ("file:///tmp/nodes", None), ("https://user:pass@example.com/nodes", None),
    ("https://example.com/nodes?token=secret", None), ("https://example.com/nodes", "--upload-pack=evil"),
])
def test_git_invalid_source_does_not_launch_git(monkeypatch, source, revision):
    from modiff.custom_extension_source import resolve_git_extension
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("No Git invocation"))
    with pytest.raises(ValueError):
        resolve_git_extension(source, revision)
