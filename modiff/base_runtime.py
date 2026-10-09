"""Observe standard application dependencies without importing model libraries."""

from __future__ import annotations

import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import tomllib


PROJECT_METADATA = Path(__file__).resolve().parents[1] / "pyproject.toml"


def base_model_runtime_contract() -> dict:
    """Declare model-library requirements without observing/importing packages.

    Catalog identity must be deterministic on every machine. Runtime receipts
    separately bind the versions actually installed in the worker.
    """
    from packaging.requirements import Requirement

    document = tomllib.loads(PROJECT_METADATA.read_text(encoding="utf-8"))
    names = {"transformers", "peft", "diffusers", "accelerate", "safetensors", "torch", "huggingface-hub"}
    declarations = sorted(
        declaration for declaration in document["project"]["dependencies"]
        if Requirement(declaration).name.lower().replace("_", "-") in names
    )
    digest = hashlib.sha256(json.dumps(declarations, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"id": "modiff.base-model-runtime", "delivery": "base", "dependencies": declarations, "digest": f"sha256:{digest}"}


def base_runtime_status(*, version_resolver=None, distribution_resolver=None) -> dict:
    """Verify declared base versions and any explicitly declared source identity, read-only.

    Installation is always an explicit uv operation. This observation is also
    suitable for graph inspection: it never imports Transformers, PEFT or Torch.
    """
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    version_resolver = version_resolver or metadata.version
    distribution_resolver = distribution_resolver or metadata.distribution
    document = tomllib.loads(PROJECT_METADATA.read_text(encoding="utf-8"))
    packages = {}
    issues = []
    python_version = ".".join(map(str, sys.version_info[:3]))
    python_requirement = document["project"].get("requires-python", "")
    if not SpecifierSet(python_requirement).contains(python_version):
        issues.append(f"Python {python_version} does not satisfy {python_requirement}.")
    source_revisions = {}
    for declaration in document["project"]["dependencies"]:
        requirement = Requirement(declaration)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            version = version_resolver(requirement.name)
        except metadata.PackageNotFoundError:
            issues.append(f"Missing required package: {requirement.name}.")
            continue
        packages[requirement.name] = version
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=False):
            issues.append(f"{requirement.name} {version} does not satisfy {requirement.specifier}.")
        if requirement.url and requirement.url.startswith("git+"):
            repository, commit = requirement.url.removeprefix("git+").rsplit("@", 1)
            try:
                direct = json.loads(distribution_resolver(requirement.name).read_text("direct_url.json") or "{}")
            except (OSError, ValueError, TypeError, metadata.PackageNotFoundError):
                direct = {}
            if not isinstance(direct, dict):
                direct = {}
            vcs_info = direct.get("vcs_info")
            if not isinstance(vcs_info, dict):
                vcs_info = {}
            installed_repository = direct.get("url")
            same_repository = (
                isinstance(installed_repository, str)
                and installed_repository.removesuffix(".git") == repository.removesuffix(".git")
            )
            if vcs_info.get("commit_id") != commit or vcs_info.get("vcs") != "git" or not same_repository:
                issues.append(f"{requirement.name} must use the declared source revision {commit}.")
            source_revisions[requirement.name] = vcs_info.get("commit_id")
    identity = {
        "python": list(sys.version_info[:2]),
        "packages": packages,
        "sourceRevisions": source_revisions,
        "dependencies": document["project"]["dependencies"],
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "status": "verified" if not issues else "incompatible",
        "verified": not issues,
        "matches": not issues,
        "installationMode": "uv",
        "packages": packages,
        "issues": issues,
        "current_digest": digest,
    }
