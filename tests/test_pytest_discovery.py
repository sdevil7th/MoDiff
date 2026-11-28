"""Default verification must not import preserved review scripts or snapshots."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_default_pytest_discovery_excludes_preserved_evidence(tmp_path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "pytest.ini").write_bytes((root / "pytest.ini").read_bytes())
    maintained = tmp_path / "tests"
    maintained.mkdir()
    (maintained / "test_maintained.py").write_text("def test_current():\n    assert True\n", encoding="utf-8")
    for directory in ["reviews", "checkpoints", "maintenance"]:
        archive = tmp_path / directory
        archive.mkdir()
        (archive / "test_archived_helper.py").write_text(
            "raise RuntimeError('Preserved evidence must not execute during default discovery')\n",
            encoding="utf-8",
        )
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
