from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "model-artifact-catalog.json"
AUTO_TRUST_LEVELS = {"official", "modiff_qualified"}
COMMUNITY_DISPLAY_MIN_DOWNLOADS = 1_000
COMMUNITY_DISPLAY_MIN_LIKES = 10
IMMUTABLE_HUB_REVISION = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


def read_model_artifact_catalog() -> dict[str, Any]:
    with CATALOG_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schemaVersion") != 1 or not isinstance(payload.get("models"), list):
        raise ValueError("Unsupported model artifact catalog schema.")
    defaults = payload.get("artifactDefaults") if isinstance(payload.get("artifactDefaults"), dict) else {}
    checked_at = payload.get("checkedAt")
    for model in payload["models"]:
        if not isinstance(model, dict):
            continue
        model.setdefault("baseRevision", None)
        artifacts = model.get("artifacts") if isinstance(model.get("artifacts"), list) else []
        expanded = []
        for raw in artifacts:
            if not isinstance(raw, dict):
                continue
            artifact = {**defaults, **raw}
            artifact["popularitySnapshot"] = {
                "checkedAt": checked_at,
                "downloads": int(raw.get("downloads") or 0),
                "likes": int(raw.get("likes") or 0),
            }
            trust = str(raw.get("trust") or "community")
            if "qualificationEvidence" not in raw:
                artifact["qualificationEvidence"] = {
                    "status": "runtime-qualified" if trust == "modiff_qualified" else "documented" if trust == "official" else "community-option",
                    "source": "modiff-qualification" if trust == "modiff_qualified" else "publisher" if trust == "official" else "hub-popularity-snapshot",
                }
            format_name = str(raw.get("format") or "native")
            if "supportedBackends" not in raw:
                artifact["supportedBackends"] = (
                    ["cuda"]
                    if format_name in {"diffusers-bnb", "nvfp4", "mxfp8"}
                    else ["cuda", "rocm"]
                    if format_name == "fp8"
                    else ["cuda", "rocm", "mps", "cpu"]
                    if format_name == "gguf"
                    else defaults.get("supportedBackends", [])
                )
            if "supportedPlatforms" not in raw and format_name in {"nvfp4", "mxfp8"}:
                artifact["supportedPlatforms"] = ["linux"]
            expanded.append(artifact)
        model["artifacts"] = expanded
    return payload


def catalog_model(model_type: str) -> dict[str, Any] | None:
    for model in read_model_artifact_catalog()["models"]:
        if isinstance(model, dict) and model.get("modelType") == model_type:
            return model
    return None


def catalog_artifact(model_type: str, repo: str) -> dict[str, Any] | None:
    model = catalog_model(model_type)
    for artifact in (model or {}).get("artifacts") or []:
        if isinstance(artifact, dict) and str(artifact.get("repo") or "").lower() == repo.lower():
            return artifact
    return None


def catalog_repository_pin(repo: str, *, model_type: str | None = None) -> dict[str, Any] | None:
    """Return the reviewed immutable pin for a cataloged Hub repository.

    Base models, optional artifacts, and fixed auxiliary repositories all use
    the same lookup so loaders cannot accidentally follow a moving branch for
    one part of a curated pipeline. Repository matching is case-insensitive,
    but the catalog remains the source of truth for the canonical spelling.
    """

    repository = str(repo or "").strip()
    if not repository:
        return None
    repository_lower = repository.lower()
    catalog = read_model_artifact_catalog()
    models = catalog["models"]
    if model_type:
        scoped = [model for model in models if model.get("modelType") == model_type]
        models = scoped + [model for model in models if model not in scoped]
    for model in models:
        if str(model.get("baseRepo") or "").lower() == repository_lower:
            return {
                "repo": model["baseRepo"],
                "revision": model.get("baseRevision"),
                "license": model.get("baseLicense"),
                "kind": "base",
                "modelType": model.get("modelType"),
            }
        for artifact in model.get("artifacts") or []:
            if str(artifact.get("repo") or "").lower() == repository_lower:
                return {**artifact, "kind": "artifact", "modelType": model.get("modelType")}
    for pin in catalog.get("repositoryPins") or []:
        if isinstance(pin, dict) and str(pin.get("repo") or "").lower() == repository_lower:
            return {**pin, "kind": "auxiliary"}
    return None


def catalog_revision(repo: str, *, model_type: str | None = None) -> str | None:
    """Return and validate the immutable revision recorded for ``repo``."""

    pin = catalog_repository_pin(repo, model_type=model_type)
    if pin is None:
        return None
    revision = str(pin.get("revision") or "").strip()
    if not IMMUTABLE_HUB_REVISION.fullmatch(revision):
        raise ValueError(f"Cataloged Hugging Face repository {repo!r} is missing a 40-character commit revision.")
    return revision.lower()


def resolve_model_revision(
    repo: str,
    revision: Any = None,
    *,
    model_type: str | None = None,
    source: str | None = None,
) -> str | None:
    """Preserve an explicit revision or pin a known curated Hub repository.

    Unknown repositories retain normal user-selected Hugging Face semantics.
    Local selections never inherit a Hub pin, even if their display value
    happens to match a cataloged repository ID.
    """

    explicit = str(revision or "").strip() or None
    if explicit is not None:
        return explicit
    if source is not None and str(source).strip().lower() != "hub":
        return None
    return catalog_revision(repo, model_type=model_type)


def require_catalog_revision(repo: str, *, model_type: str | None = None) -> str:
    """Return a reviewed pin for a fixed built-in repository or fail closed."""

    revision = catalog_revision(repo, model_type=model_type)
    if revision is None:
        raise ValueError(f"Built-in Hugging Face repository {repo!r} has no reviewed catalog revision.")
    return revision


def community_artifact_is_discoverable(artifact: dict[str, Any]) -> bool:
    return bool(
        int(artifact.get("downloads") or 0) >= COMMUNITY_DISPLAY_MIN_DOWNLOADS
        or int(artifact.get("likes") or 0) >= COMMUNITY_DISPLAY_MIN_LIKES
    )


def public_model_artifact_catalog() -> dict[str, Any]:
    catalog = read_model_artifact_catalog()
    return {
        **catalog,
        "policy": {
            "autoTrustLevels": sorted(AUTO_TRUST_LEVELS),
            "communityDisplayMinimumDownloads": COMMUNITY_DISPLAY_MIN_DOWNLOADS,
            "communityDisplayMinimumLikes": COMMUNITY_DISPLAY_MIN_LIKES,
            "popularityIsCompatibilityProof": False,
        },
    }


def refreshed_hub_metadata(*, model_type: str | None = None, repo: str | None = None) -> dict[str, Any]:
    """Return a live discovery overlay; never mutate vetted catalog selection fields."""
    from huggingface_hub import HfApi
    from modiff.config import CONFIG

    catalog = read_model_artifact_catalog()
    api = HfApi(token=CONFIG.hf.get("token"), library_name="MoDiff")
    refreshed = []
    for model in catalog["models"]:
        if model_type and model.get("modelType") != model_type:
            continue
        for artifact in model.get("artifacts") or []:
            artifact_repo = str(artifact.get("repo") or "")
            if not artifact_repo or (repo and artifact_repo.lower() != repo.lower()):
                continue
            try:
                info = api.model_info(artifact_repo)
                refreshed.append({
                    "modelType": model.get("modelType"),
                    "repo": artifact_repo,
                    "liveRevision": getattr(info, "sha", None),
                    "downloads": int(getattr(info, "downloads", 0) or 0),
                    "likes": int(getattr(info, "likes", 0) or 0),
                    "status": "available",
                })
            except Exception as exc:
                refreshed.append({
                    "modelType": model.get("modelType"),
                    "repo": artifact_repo,
                    "status": "unavailable",
                    "message": str(exc),
                })
    return {
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "selectionChanged": False,
        "artifacts": refreshed,
    }
