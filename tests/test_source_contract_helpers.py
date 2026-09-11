import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from source_contract_helpers import source_sha256


def test_source_hash_allows_checkout_newlines_but_detects_changed_code(tmp_path):
    source = tmp_path / "source.py"
    original = b"VALUE = 1\n"
    expected = hashlib.sha256(original).hexdigest()
    source.write_bytes(original)
    assert source_sha256(source) == expected
    source.write_bytes(original.replace(b"\n", b"\r\n"))
    assert source_sha256(source) == expected
    source.write_bytes(b"VALUE = 2\r\n")
    assert source_sha256(source) != expected


def test_required_catalog_snapshots_are_present_and_not_ignored():
    root = Path(__file__).resolve().parents[1]
    required = [
        "data/modular-block-contracts.json",
        "data/modular-conditional-contracts.json",
        "data/modular-workflow-contracts.json",
        "data/anima-artifact-review.json",
        "data/flux2-klein-base-4b-artifact-review.json",
        "data/minimax-music3-artifact-review.json",
    ]
    assert all((root / path).is_file() for path in required)
    if not shutil.which("git") or not (root / ".git").exists():
        pytest.skip("Ignore-rule check requires a Git checkout")
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *required],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
