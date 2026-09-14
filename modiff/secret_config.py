import configparser
import os
from pathlib import Path
import tempfile


def _restrict_secret_descriptor(descriptor: int) -> None:
    """Restrict the held temporary file before writing any secret bytes."""
    if os.name != "nt":
        os.fchmod(descriptor, 0o600)
        return

    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    pointer = ctypes.c_void_p
    kernel32.ReOpenFile.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD)
    kernel32.ReOpenFile.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = (pointer,)
    kernel32.LocalFree.restype = pointer
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(pointer), pointer,
    )
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    advapi32.GetSecurityDescriptorDacl.argtypes = (
        pointer, ctypes.POINTER(wintypes.BOOL), ctypes.POINTER(pointer), ctypes.POINTER(wintypes.BOOL),
    )
    advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.SetSecurityInfo.argtypes = (
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, pointer, pointer, pointer, pointer,
    )
    advapi32.SetSecurityInfo.restype = wintypes.DWORD
    # Reopen the same object, never a pathname, for WRITE_DAC. OW grants only
    # the file owner access; the protected DACL removes inherited grants.
    handle = kernel32.ReOpenFile(msvcrt.get_osfhandle(descriptor), 0x00060000, 0x3, 0)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    security = pointer()
    try:
        if not advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            "D:P(A;;FA;;;OW)", 1, ctypes.byref(security), None,
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        present, defaulted, dacl = wintypes.BOOL(), wintypes.BOOL(), pointer()
        if not advapi32.GetSecurityDescriptorDacl(
            security, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        if not present.value or not dacl.value:
            raise OSError("An owner-only secret DACL could not be constructed.")
        error = advapi32.SetSecurityInfo(handle, 1, 0x80000004, None, None, dacl, None)
        if error:
            raise ctypes.WinError(error)
    finally:
        if security.value:
            kernel32.LocalFree(security)
        kernel32.CloseHandle(handle)


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
            _restrict_secret_descriptor(handle.fileno())
            handle.write("\n".join(updated_lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
