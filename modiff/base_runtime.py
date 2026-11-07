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
    evaluation, provider, underlying = None, None, None
    try:
        from modiff.runtime_overlays import active_diffusers_provider, base_diffusers_identity
        provider = active_diffusers_provider()
        underlying = base_diffusers_identity() if provider is not None else None
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, metadata.PackageNotFoundError) as error:
        issues.append(f"Diffusers provider contract is invalid: {error}")
    try:
        from modiff.diffusers_evaluation_runtime import selected_diffusers_evaluation_contract
        evaluation = underlying.get("evaluationRuntime") if underlying is not None else selected_diffusers_evaluation_contract(
            distribution_resolver=distribution_resolver)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, metadata.PackageNotFoundError) as error:
        issues.append(f"Diffusers evaluation contract is invalid: {error}")
    for declaration in document["project"]["dependencies"]:
        if evaluation and Requirement(declaration).name.lower().replace("_", "-") == "diffusers":
            declaration = evaluation["dependency"]
        requirement = Requirement(declaration)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            version = version_resolver(requirement.name)
        except metadata.PackageNotFoundError:
            issues.append(f"Missing required package: {requirement.name}.")
            continue
        packages[requirement.name] = version
        checked_version = underlying["version"] if underlying is not None and requirement.name.lower() == "diffusers" else version
        if provider is not None and requirement.name.lower() == "diffusers" and version != provider["version"]:
            issues.append("The selected Diffusers provider version changed.")
        if requirement.specifier and not requirement.specifier.contains(checked_version, prereleases=bool(evaluation and requirement.name.lower() == "diffusers")):
            issues.append(f"{requirement.name} {checked_version} does not satisfy {requirement.specifier}.")
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
    evaluation_identity = {key: value for key, value in evaluation.items() if key not in {"installedFiles", "wheelPath"}} if evaluation else None
    provider_identity = ({key: provider[key] for key in ("profileId", "specDigest", "distribution", "version", "recordDigest")}
                         if provider is not None else None)
    underlying_identity = ({key: underlying[key] for key in ("version", "declaration", "recordDigest")}
                           if underlying is not None else None)
    identity = {
        "python": list(sys.version_info[:2]),
        "packages": packages,
        "sourceRevisions": source_revisions,
        "dependencies": document["project"]["dependencies"],
        **({"evaluationRuntime": evaluation_identity} if evaluation_identity else {}),
        **({"diffusersProvider": provider_identity, "underlyingDiffusers": underlying_identity} if provider_identity else {}),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    # Overlay validation remains bound to the original base, while public
    # runtime identity also records the separately selected provider.
    base_digest = None
    if underlying is not None:
        original_identity = {key: value for key, value in identity.items()
                             if key not in {"diffusersProvider", "underlyingDiffusers"}}
        original_identity["packages"] = {**packages, "diffusers": underlying["version"]}
        base_digest = hashlib.sha256(json.dumps(original_identity, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "status": "verified" if not issues else "incompatible",
        "verified": not issues,
        "matches": not issues,
        "installationMode": "uv",
        **({"evaluationRuntime": evaluation_identity} if evaluation_identity else {}),
        **({"diffusersProvider": provider_identity, "underlyingDiffusers": underlying_identity} if provider_identity else {}),
        "packages": packages,
        "issues": issues,
        "current_digest": digest,
        **({"underlying_base_digest": base_digest} if base_digest is not None else {}),
    }
