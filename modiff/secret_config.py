import configparser
import os
from pathlib import Path
import tempfile


def dotenv_value(path: Path, key: str) -> str | None:
    """Read one simple dotenv value without exporting unrelated entries."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None

    value = None
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, candidate = line.partition("=")
        if not separator or name.strip() != key:
            continue
        candidate = candidate.strip()
        if (
            len(candidate) >= 2
            and candidate[0] == candidate[-1]
            and candidate[0] in {"'", '"'}
        ):
            candidate = candidate[1:-1]
        value = candidate.strip() or None
    return value


def huggingface_token(
    cfg: configparser.ConfigParser,
    dotenv_path: Path,
) -> tuple[str | None, str | None]:
    """Resolve the local Hugging Face credential and identify its source."""

    for environment_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(environment_name, "").strip()
        if value:
            return value, f"environment:{environment_name}"

    value = dotenv_value(dotenv_path, "HF_TOKEN")
    if value:
        return value, "dotenv"

    value = cfg.get("huggingface", "token", fallback="").strip()
    if value:
        return value, "config"
    return None, None


def set_dotenv_value(path: Path, key: str, value: str) -> None:
    """Atomically set one dotenv entry while preserving unrelated values."""

    if not key or not key.replace("_", "").isalnum() or key.upper() != key:
        raise ValueError("Dotenv secret names must use uppercase letters, numbers, and underscores.")
    if not value or any(character in value for character in ("\r", "\n", "\x00")):
        raise ValueError("Dotenv secret values must be non-empty single-line strings.")
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlinked dotenv file.")

    try:
        existing_lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        existing_lines = []

    replacement = f"{key}={value}"
    updated_lines = []
    replaced = False
    for raw_line in existing_lines:
        candidate = raw_line.strip()
        if candidate.startswith("export "):
            candidate = candidate[7:].lstrip()
        name, separator, _ = candidate.partition("=")
        if separator and name.strip() == key:
            if not replaced:
                updated_lines.append(replacement)
                replaced = True
            continue
        updated_lines.append(raw_line)
    if not replaced:
        updated_lines.append(replacement)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            # Python 3.12 on Windows has no fchmod. Its access control comes
            # from the containing directory's ACL, rather than POSIX mode bits.
            # Own the descriptor before setting permissions so failures close
            # it before the temporary file is removed.
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), 0o600)
            else:
                os.chmod(temporary_path, 0o600)
            handle.write("\n".join(updated_lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
