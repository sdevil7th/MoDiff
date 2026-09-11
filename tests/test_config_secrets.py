import configparser
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from modiff.config import Config
from modiff.secret_config import dotenv_value, huggingface_token, set_dotenv_value


def test_dotenv_reader_ignores_unrelated_entries_and_accepts_export(tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "# local secrets\nUNRELATED=value\nexport HF_TOKEN='dotenv-token'\n",
        encoding="utf-8",
    )

    assert dotenv_value(dotenv_path, "HF_TOKEN") == "dotenv-token"
    assert dotenv_value(dotenv_path, "MISSING") is None


def test_huggingface_token_prefers_environment_then_dotenv_then_config(tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HF_TOKEN=dotenv-token\n", encoding="utf-8")
    cfg = configparser.ConfigParser()
    cfg.read_string("[huggingface]\ntoken = config-token\n")

    with patch.dict("os.environ", {"HF_TOKEN": "environment-token"}, clear=True):
        assert huggingface_token(cfg, dotenv_path) == (
            "environment-token",
            "environment:HF_TOKEN",
        )

    with patch.dict("os.environ", {}, clear=True):
        assert huggingface_token(cfg, dotenv_path) == ("dotenv-token", "dotenv")

    dotenv_path.unlink()
    with patch.dict("os.environ", {}, clear=True):
        assert huggingface_token(cfg, dotenv_path) == ("config-token", "config")


def test_config_loads_huggingface_token_from_explicit_dotenv(tmp_path: Path):
    config_path = tmp_path / "config.ini"
    config_path.write_text("[huggingface]\nonline_status = Auto\n", encoding="utf-8")
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HF_TOKEN=dotenv-token\n", encoding="utf-8")

    with patch.dict("os.environ", {}, clear=True):
        config = Config(config_path=config_path, dotenv_path=dotenv_path)

    assert config.hf["token"] == "dotenv-token"
    assert config.hf["token_source"] == "dotenv"


def test_set_dotenv_value_preserves_unrelated_entries_and_restricts_permissions(tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "UNRELATED=value\nHF_TOKEN=old-token\nexport HF_TOKEN=duplicate-token\n",
        encoding="utf-8",
    )

    set_dotenv_value(dotenv_path, "HF_TOKEN", "new-token")

    assert dotenv_path.read_text(encoding="utf-8") == (
        "UNRELATED=value\nHF_TOKEN=new-token\n"
    )
    if os.name != "nt":
        assert dotenv_path.stat().st_mode & 0o777 == 0o600


def test_set_dotenv_value_without_fchmod(tmp_path: Path, monkeypatch):
    monkeypatch.delattr(os, "fchmod", raising=False)
    dotenv_path = tmp_path / ".env"
    set_dotenv_value(dotenv_path, "HF_TOKEN", "synthetic-token")
    assert dotenv_value(dotenv_path, "HF_TOKEN") == "synthetic-token"
    assert list(tmp_path.iterdir()) == [dotenv_path]


def test_permission_failure_closes_temporary_file_and_preserves_original(tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("HF_TOKEN=original\n", encoding="utf-8")
    with patch.object(os, "fchmod", side_effect=PermissionError("denied"), create=True):
        with pytest.raises(PermissionError, match="denied"):
            set_dotenv_value(dotenv_path, "HF_TOKEN", "replacement")
    assert dotenv_path.read_text(encoding="utf-8") == "HF_TOKEN=original\n"
    assert list(tmp_path.iterdir()) == [dotenv_path]


def test_set_dotenv_value_rejects_multiline_values_and_symlinks(tmp_path: Path):
    dotenv_path = tmp_path / ".env"

    try:
        set_dotenv_value(dotenv_path, "HF_TOKEN", "first\nsecond")
    except ValueError:
        pass
    else:
        raise AssertionError("multiline dotenv values must be rejected")

    target = tmp_path / "target"
    target.write_text("HF_TOKEN=unchanged\n", encoding="utf-8")
    try:
        dotenv_path.symlink_to(target)
    except OSError as error:
        if os.name == "nt" and error.winerror == 1314:
            pytest.skip("Windows symlink creation requires Developer Mode or privilege")
        raise
    try:
        set_dotenv_value(dotenv_path, "HF_TOKEN", "replacement")
    except ValueError:
        pass
    else:
        raise AssertionError("symlinked dotenv files must be rejected")
    assert target.read_text(encoding="utf-8") == "HF_TOKEN=unchanged\n"
