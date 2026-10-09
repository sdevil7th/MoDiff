"""Isolated historical source runtime for the three no-weight catalog CLIs.

Application startup and runtime checks never call this helper. Only the audit
child imports the verified historical package; the parent's installed release
and import path are unchanged.
"""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path
import subprocess
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from modiff.upstream_coverage import (
    UpstreamCoverageError, _static_package_version, reviewed_diffusers_source,
)

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = frozenset({
    "generate_modular_block_contracts.py",
    "generate_modular_conditional_contracts.py",
    "generate_modular_workflow_contracts.py",
})
_CHILD = "MODIFF_CATALOG_SOURCE_AUDIT_CHILD"
_WORKSPACE = "MODIFF_CATALOG_SOURCE_AUDIT_WORKSPACE"


def _audit_environment(source: Path, workspace: Path, script_name: str) -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN", "HF_TOKEN_PATH", "HF_STORED_TOKENS_PATH"):
        environment.pop(name, None)
    environment.update({
        "PYTHONPATH": os.pathsep.join((str(source.parent), str(ROOT))),
        "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "-1", "HF_HOME": str(workspace / "hf"),
        "HF_HUB_CACHE": str(workspace / "hf/hub"), "HF_TOKEN_PATH": str(workspace / "hf/token"),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "MODIFF_MANAGED_ROOT": str(workspace / "managed"),
        "MODIFF_DIFFUSERS_CATALOG_SOURCE": str(source), _CHILD: script_name, _WORKSPACE: str(workspace),
    })
    return environment


def _caller_output_arguments(arguments: list[str], caller: Path) -> list[str]:
    """Preserve the three CLIs' accepted output paths in the caller's cwd."""
    result = list(arguments)
    for index, argument in enumerate(arguments):
        option, equal, value = argument.partition("=")
        if not option.startswith("--") or len(option) <= 2 or not "--output".startswith(option):
            continue
        if equal:
            if not Path(value).is_absolute():
                result[index] = option + "=" + str((caller / value).resolve())
        elif index + 1 < len(arguments):
            value = arguments[index + 1]
            # argparse accepts a bare dash and negative numbers as values.
            # Leave missing/option arguments to its original error handling.
            if value.startswith("-") and value != "-" and not re.fullmatch(r"-\d+|-\d*\.\d+", value):
                continue
            if not Path(value).is_absolute():
                result[index + 1] = str((caller / value).resolve())
    return result


def _verify_imported_source(source: Path):
    diffusers = importlib.import_module("diffusers")
    expected_version = _static_package_version((source / "__init__.py").read_text(encoding="utf-8"),
                                              label="Historical Diffusers __init__.py")
    if Path(diffusers.__file__).resolve() != source / "__init__.py" or diffusers.__version__ != expected_version:
        raise UpstreamCoverageError("The catalog audit child imported a different Diffusers source or version.")
    return diffusers


def prepare_catalog_source_audit(script: Path) -> None:
    """Reexecute one known CLI in its private verified-source audit child."""
    script = script.resolve()
    if script.name not in _SCRIPTS or script != ROOT / "scripts" / script.name:
        raise UpstreamCoverageError("Historical source audit is limited to the reviewed catalog generators.")
    declared = os.environ.get("MODIFF_DIFFUSERS_CATALOG_SOURCE")
    if not declared:
        raise UpstreamCoverageError("Historical generation requires explicit MODIFF_DIFFUSERS_CATALOG_SOURCE.")
    source = reviewed_diffusers_source(Path(declared))
    if os.environ.get(_CHILD) != script.name:
        arguments = _caller_output_arguments(sys.argv[1:], Path.cwd())
        with TemporaryDirectory(prefix="modiff-historical-catalog-audit-") as temporary:
            workspace = Path(temporary).resolve()
            result = subprocess.run([sys.executable, str(script), *arguments], cwd=workspace,
                                    env=_audit_environment(source, workspace, script.name), check=False)
        raise SystemExit(result.returncode)

    workspace = Path(os.environ.get(_WORKSPACE, "")).resolve()
    if workspace != Path.cwd().resolve() or not workspace.name.startswith("modiff-historical-catalog-audit-"):
        raise UpstreamCoverageError("The catalog audit child requires its private workspace.")
    sys.dont_write_bytecode = True
    # Bootstrap only this child without reading operator config, dotenv, Hub
    # credentials or approved custom extensions. Ordinary tests keep their real
    # fixture methods and normal runtime imports keep the installed release.
    from modiff import secret_config
    with patch("configparser.ConfigParser.read", return_value=[]), patch("os.makedirs"), patch.object(
        secret_config, "huggingface_token", return_value=(None, None)
    ):
        from modiff.config import CONFIG
    CONFIG.hf.update({"token": None, "token_source": None, "online_status": "Offline",
                      "cache_dir": str(workspace / "hf/hub")})
    for name in CONFIG.paths:
        CONFIG.paths[name] = str(workspace / name)
        Path(CONFIG.paths[name]).mkdir(parents=True, exist_ok=True)
    # Copy the small source tree so Windows needs no symlink privilege. No
    # operator custom extensions or frontend/media payloads are copied.
    shutil.copytree(ROOT / "modules", workspace / "modules",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _verify_imported_source(source)
    from modiff.custom_extensions import ExtensionStore
    with patch.object(ExtensionStore, "load_enabled", return_value=None):
        importlib.import_module("modules")
    print("CATALOG_SOURCE_AUDIT " + json.dumps({
        "scope": "Isolated historical source metadata; no generation or runtime qualification",
        "diffusersOrigin": str(source / "__init__.py"),
        "diffusersVersion": _static_package_version((source / "__init__.py").read_text(encoding="utf-8"),
                                                    label="Historical Diffusers __init__.py"),
        "configAndCache": "private", "extensionDiscovery": "disabled",
    }), flush=True)
