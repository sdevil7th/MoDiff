"""Resolve an operator-selected Hub source without downloading or importing code."""

import re
from urllib.parse import unquote, urlsplit

from modiff.custom_extensions import ExtensionError, immutable_revision


def resolve_git_extension(source, revision=None):
    """Resolve remote refs without checkout, hooks, terminal prompts or code imports."""
    import os
    import subprocess

    if not isinstance(source, str) or len(source) > 2048:
        raise ExtensionError("Enter an HTTPS Git repository URL.")
    url = urlsplit(source)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ExtensionError("Use an HTTPS Git URL without credentials, query or fragment.")
    requested = revision or "HEAD"
    if not isinstance(requested, str) or len(requested) > 256 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", requested) or ".." in requested:
        raise ExtensionError("The revision must be a branch, tag or exact commit.")
    if re.fullmatch(r"[a-f0-9]{40}", requested):
        resolved = requested
    else:
        refs = [requested] if requested == "HEAD" else [f"refs/heads/{requested}", f"refs/tags/{requested}", f"refs/tags/{requested}^{{}}"]
        try:
            result = subprocess.run(
                ["git", "-c", "credential.interactive=false", "ls-remote", "--exit-code", source, *refs],
                capture_output=True, timeout=30,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"},
            )
            matches = [line.split() for line in result.stdout.decode("utf-8").splitlines()]
            if result.returncode or not matches:
                raise ValueError("Missing ref")
            # Prefer an annotated tag's peeled commit; reject ambiguous branch/tag names.
            peeled = {row[0] for row in matches if len(row) == 2 and row[1].endswith("^{}")}
            commits = {row[0] for row in matches if len(row) == 2 and not row[1].startswith("refs/tags/")}
            commits |= peeled or {row[0] for row in matches if len(row) == 2 and row[1].startswith("refs/tags/")}
            if len(commits) != 1:
                raise ValueError("Ambiguous ref")
            resolved = immutable_revision(commits.pop())
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            raise ExtensionError("Could not resolve Git revision. Check URL, access and revision; use an exact commit for ambiguous names.") from error
    return {"kind": "git", "source": source, "requestedRevision": requested, "revision": resolved}


def resolve_hub_extension(source, revision=None):
    from huggingface_hub import HfApi
    from huggingface_hub.utils import validate_repo_id

    if not isinstance(source, str) or not source.strip() or len(source) > 2048:
        raise ExtensionError("Enter a Hugging Face repository ID or HTTPS repository URL.")
    source = source.strip()
    url_revision = None
    if "://" in source:
        url = urlsplit(source)
        if url.scheme != "https" or url.netloc != "huggingface.co" or url.query or url.fragment:
            raise ExtensionError("Use a huggingface.co HTTPS repository URL without credentials, query or fragment.")
        parts = url.path.strip("/").split("/")
        if len(parts) == 4 and parts[2] == "tree":
            url_revision = unquote(parts.pop())
            parts.pop()
        if len(parts) not in {1, 2} or parts[0] in {"datasets", "spaces"}:
            raise ExtensionError("Use the repository root or its /tree/<revision> URL, not a file or subfolder.")
        source = "/".join(parts)
    validate_repo_id(source)
    if revision is not None and not isinstance(revision, str):
        raise ExtensionError("The revision must be a branch, tag or exact commit.")
    revision = (revision or "").strip()
    if url_revision and revision and url_revision != revision:
        raise ExtensionError("The URL and revision field disagree. Choose one revision before resolving.")
    requested = revision or url_revision or "main"
    if len(requested) > 256 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", requested) or ".." in requested:
        raise ExtensionError("The revision must be a branch, tag or exact commit.")
    try:
        info = HfApi(endpoint="https://huggingface.co").model_info(source, revision=requested, timeout=20)
    except Exception as error:
        # Provider exceptions can contain authenticated URLs or request details.
        raise ExtensionError(
            "Could not resolve the source. Check the repository, revision, network and Hub access."
        ) from error
    resolved = immutable_revision(info.sha)
    if re.fullmatch(r"[a-f0-9]{40}", requested) and resolved != requested:
        raise ExtensionError("Hugging Face returned a different revision. No source was staged.")
    return {"kind": "hub", "source": source, "requestedRevision": requested, "revision": resolved}
