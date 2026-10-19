"""Explicit immutable evaluation artifacts, separate from ordinary release setup.

This read-only observer imports no model package, installs nothing and grants no
model/Auto/hardware qualification. The normal project dependency remains pinned.
"""
from __future__ import annotations

import hashlib
from importlib import metadata
from importlib.util import find_spec
import io
import json
import os
import re
from pathlib import Path, PurePosixPath
import stat
import sys
import zipfile

PROFILE_PATH = Path(__file__).resolve().parents[1] / "data/diffusers-evaluation-runtime.v1.json"
PROFILE_ENV = "MODIFF_DIFFUSERS_EVALUATION_PROFILE"
WHEEL_ENV = "MODIFF_DIFFUSERS_EVALUATION_WHEEL"
MAX_WHEEL_BYTES = 16 * 1024 * 1024
MAX_INSTALLED_FILES = 4096
MAX_EXPANDED_BYTES = 128 * 1024 * 1024


def _plain(path: Path, *, directory: bool = False) -> os.stat_result:
    details = path.lstat()
    if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
        raise RuntimeError("The evaluation artifact contains a link or reparse point.")
    if not (stat.S_ISDIR(details.st_mode) if directory else stat.S_ISREG(details.st_mode)):
        raise RuntimeError("The evaluation artifact has an unexpected filesystem entry.")
    return details


def _directory_identity(details: os.stat_result) -> tuple:
    return details.st_dev, details.st_ino, details.st_mode, getattr(details, "st_file_attributes", 0)


def _body(path: Path, limit: int, *, checked_parents: dict | None = None) -> bytes:
    checked_parents = {} if checked_parents is None else checked_parents
    for parent in path.parents:
        if parent not in checked_parents:
            checked_parents[parent] = _directory_identity(_plain(parent, directory=True))
    before = _plain(path)
    if before.st_size > limit:
        raise RuntimeError("The evaluation artifact exceeds its declared bound.")
    with path.open("rb") as stream:
        body = stream.read(before.st_size + 1)
    after = _plain(path)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if len(body) != before.st_size or any(getattr(before, key) != getattr(after, key) for key in fields):
        raise RuntimeError("The evaluation artifact changed while it was read.")
    return body


def selected_diffusers_evaluation_contract(*, distribution_resolver=None,
        _authoring_profile_ids: list[str] | None = None) -> dict | None:
    """Validate the explicitly selected exact wheel and its installed file census.

    An unknown/missing/changed artifact fails closed; it never falls back to a
    permissive development-version range. No selector means ordinary setup.
    """
    selected = os.environ.get(PROFILE_ENV, "")
    wheel_value = os.environ.get(WHEEL_ENV, "")
    if not selected:
        if wheel_value:
            raise RuntimeError("An evaluation wheel requires an explicit reviewed profile.")
        return None
    if not sys.dont_write_bytecode:
        raise RuntimeError("Explicit Diffusers evaluation requires a no-bytecode (-B) Python launch.")
    checked_parents = {}
    document = json.loads(_body(PROFILE_PATH, 16 * 1024, checked_parents=checked_parents))
    if document.get("schemaVersion") != 1 or selected not in document.get("profiles", {}):
        raise RuntimeError("The Diffusers evaluation profile is not reviewed.")
    contract = document["profiles"][selected]
    if (contract.get("kind") != "official-source-evaluation" or contract.get("distribution") != "diffusers"
            or contract.get("repository") != "https://github.com/huggingface/diffusers.git"):
        raise RuntimeError("The evaluation profile has invalid official-library provenance.")
    if not wheel_value or not Path(wheel_value).is_absolute():
        raise RuntimeError("The reviewed evaluation wheel must be selected by absolute path.")
    wheel = Path(wheel_value)
    body = _body(wheel, MAX_WHEEL_BYTES, checked_parents=checked_parents)
    wheel_digest = hashlib.sha256(body).hexdigest()
    if (wheel.name != contract["wheelFilename"] or len(body) != contract["wheelByteSize"]
            or wheel_digest != contract["wheelSha256"]):
        raise RuntimeError("The evaluation wheel does not match the exact reviewed artifact.")
    distribution = (distribution_resolver or metadata.distribution)("diffusers")
    if str(distribution.version) != contract["version"]:
        raise RuntimeError("The installed Diffusers version does not match the evaluation artifact.")
    record = distribution.read_text("RECORD")
    if not isinstance(record, str) or not record or len(record) > 2 * 1024 * 1024:
        raise RuntimeError("The evaluation installation has no bounded distribution record.")
    direct = json.loads(distribution.read_text("direct_url.json") or "null")
    if not isinstance(direct, dict) or set(direct) != {"url", "archive_info"} or direct["url"] != wheel.as_uri():
        raise RuntimeError("Diffusers was not installed from the selected evaluation wheel.")
    archive_info = direct["archive_info"]
    if (not isinstance(archive_info, dict) or set(archive_info) - {"hash", "hashes"}
            or ("hash" in archive_info and archive_info["hash"] != "sha256=" + wheel_digest)
            or ("hashes" in archive_info and archive_info["hashes"] != {"sha256": wheel_digest})):
        raise RuntimeError("The evaluation wheel installation provenance is inconsistent.")
    site = Path(distribution.locate_file("")).absolute()
    _plain(site, directory=True)
    if Path(distribution._path).absolute() != site / f"diffusers-{contract['version']}.dist-info":
        raise RuntimeError("The evaluation distribution metadata has an unexpected origin.")
    specification = find_spec("diffusers")
    if (specification is None or specification.origin is None
            or Path(specification.origin).absolute() != site / "diffusers/__init__.py"
            or {Path(value).absolute() for value in specification.submodule_search_locations or []} != {site / "diffusers"}):
        raise RuntimeError("Diffusers import resolution differs from the reviewed installed package.")
    installed = {}
    expanded = 0
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or len(names) > MAX_INSTALLED_FILES:
            raise RuntimeError("The evaluation wheel has an invalid file census.")
        for member in archive.infolist():
            name = member.filename
            relative = PurePosixPath(name)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                    or not relative.parts or relative.parts[0] not in {"diffusers", f"diffusers-{contract['version']}.dist-info"}):
                raise RuntimeError("The evaluation wheel contains an unexpected package path.")
            if member.is_dir() or name.endswith(".dist-info/RECORD"):
                continue
            expanded += member.file_size
            if expanded > MAX_EXPANDED_BYTES:
                raise RuntimeError("The evaluation wheel exceeds its expanded bound.")
            expected = archive.read(member)
            actual = _body(site / name, MAX_EXPANDED_BYTES, checked_parents=checked_parents)
            if actual != expected:
                raise RuntimeError("Installed Diffusers files differ from the reviewed evaluation wheel.")
            installed[name] = hashlib.sha256(actual).hexdigest()
    package_root = site / "diffusers"
    actual_package = set()
    observed_entries = 0
    for directory, directories, files in os.walk(package_root, followlinks=False):
        parent = Path(directory)
        _plain(parent, directory=True)
        observed_entries += 1 + len(directories) + len(files)
        if observed_entries > MAX_INSTALLED_FILES * 4 or len(parent.relative_to(package_root).parts) > 64:
            raise RuntimeError("The installed Diffusers census exceeds its traversal bound.")
        for child in directories:
            _plain(parent / child, directory=True)
        for child in files:
            path = parent / child
            _plain(path)
            if path.suffix.lower() == ".pyc":
                raise RuntimeError("The evaluation installation contains bytecode; use a clean source-only installation and -B.")
            actual_package.add(path.relative_to(site).as_posix())
            if len(actual_package) > MAX_INSTALLED_FILES:
                raise RuntimeError("The installed Diffusers census exceeds its bound.")
    if actual_package != {name for name in installed if name.startswith("diffusers/")}:
        raise RuntimeError("Installed Diffusers contains files outside the reviewed evaluation wheel.")
    # Check common ancestors once at each end of this observation. Every file
    # body is still read and compared; no package-content result is cached.
    for parent, identity in checked_parents.items():
        if _directory_identity(_plain(parent, directory=True)) != identity:
            raise RuntimeError("An evaluation artifact ancestor changed during observation.")
    # Collect metadata permission from this exact validated read, separately
    # from installed-distribution identity and optional-runtime sealing.
    if _authoring_profile_ids is not None:
        profiles = contract.get("authoringExecutionProfileIds", [])
        if (not isinstance(profiles, list) or len(profiles) > 16
                or any(not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._:-]{0,127}", value)
                       for value in profiles) or len(set(profiles)) != len(profiles)):
            raise RuntimeError("The evaluation authoring metadata profile list is invalid.")
        _authoring_profile_ids.extend(profiles)
    census_digest = hashlib.sha256(json.dumps(installed, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"profileId": selected, "dependency": f"diffusers=={contract['version']}",
            "repository": contract["repository"], "revision": contract["revision"],
            "sourceTree": contract["sourceTree"], "wheelSha256": wheel_digest,
            "packageCensusSha256": census_digest,
            "recordSha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
            "wheelPath": str(wheel), "installedFiles": installed,
            "scope": "exact-installed-source-artifact; no model or hardware qualification"}


def selected_diffusers_evaluation_authoring_profile_ids(*, distribution_resolver=None) -> tuple[str, ...]:
    """Metadata-only visibility after the full exact artifact observation.

    The ordinary identity returned above is unchanged. These IDs grant neither
    public discovery, model execution, resource eligibility nor qualification.
    No HTTP request or caller-supplied allowlist can select them.
    """
    profiles: list[str] = []
    selected_diffusers_evaluation_contract(distribution_resolver=distribution_resolver,
        _authoring_profile_ids=profiles)
    return tuple(profiles)
