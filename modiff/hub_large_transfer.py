"""Bound the official Hub Xet transport for blobs above its HTTP size ceiling.

Ordinary transfers keep their existing HTTP implementation. Only large blobs
use a disposable app-owned process: a stuck native reconstruction can then be
terminated without leaving a thread holding the server's download lease.
Credentials travel over stdin, never command arguments, files or progress logs.
"""

import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile


TRANSFER_TIMEOUT_SECONDS = 1800
MAX_REQUEST_BYTES = 2 * 1024 * 1024


def requires_large_transfer(plan):
    from huggingface_hub.constants import MAX_HTTP_DOWNLOAD_SIZE

    return any(
        isinstance(row.get("size"), int) and row["size"] > MAX_HTTP_DOWNLOAD_SIZE
        for row in plan.get("files", [])
        if isinstance(row, dict)
    )


def validate_request(values):
    from huggingface_hub.utils import validate_repo_id

    if (
        not isinstance(values, dict)
        or set(values) - {"repo_id", "revision", "allow_patterns", "cache_dir", "token", "force_download"}
        or values.get("force_download", False) is not False
    ):
        raise ValueError("Large Hub downloads require the app-owned resumable request contract.")
    validate_repo_id(values.get("repo_id", ""))
    if not re.fullmatch("[0-9a-f]{40}", str(values.get("revision", ""))):
        raise ValueError("Large Hub downloads require an immutable revision.")
    names = values.get("allow_patterns")
    if not isinstance(names, list) or not 1 <= len(names) <= 10000:
        raise ValueError("Large Hub downloads require a finite reviewed file selection.")
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 1024
            or "\\" in name
            or any(character in name for character in "*?[]\0:")
            or PurePosixPath(name).is_absolute()
            or ".." in PurePosixPath(name).parts
            or name != PurePosixPath(name).as_posix()
        ):
            raise ValueError("Large Hub download paths must be exact repository files.")
    if not isinstance(values.get("cache_dir"), str) or not values["cache_dir"]:
        raise ValueError("Large Hub downloads require the configured cache directory.")


def download(values):
    validate_request(values)
    payload = json.dumps(values).encode("utf-8")
    if len(payload) > MAX_REQUEST_BYTES:
        raise ValueError("Large Hub download request is too large.")
    environment = {**os.environ, "HF_HUB_DISABLE_XET": "0", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    with tempfile.TemporaryFile() as output:
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "modiff.hub_large_transfer"],
                input=payload,
                stdout=output,
                stderr=subprocess.DEVNULL,
                env=environment,
                cwd=str(Path(__file__).resolve().parents[1]),
                timeout=TRANSFER_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                "The large Hub transfer exceeded its 30-minute transfer window. "
                "Partial cache files are preserved; retry to resume."
            ) from error
        output.seek(0)
        try:
            result = json.loads(output.read(16385))
        except (ValueError, UnicodeDecodeError) as error:
            raise RuntimeError("The isolated Hub transfer returned an invalid result.") from error
    if completed.returncode or not isinstance(result, dict) or not isinstance(result.get("snapshot"), str):
        kind = result.get("errorType", "worker failure") if isinstance(result, dict) else "worker failure"
        raise RuntimeError(f"Large Hub transfer failed ({kind}); partial cache files are preserved.")
    return result["snapshot"]


def _main():
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("Request too large.")
        values = json.loads(raw)
        validate_request(values)
        from huggingface_hub import snapshot_download
        if importlib.util.find_spec('hf_xet') is None:
            raise RuntimeError("hf_xet_unavailable")
        snapshot = snapshot_download(**values, max_workers=4)
        print(json.dumps({"snapshot": str(snapshot)}), flush=True)
        return 0
    except Exception as error:
        # Avoid echoing request data, signed URLs or exception text containing
        # credentials from any downstream transport.
        print(json.dumps({"errorType": type(error).__name__}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(_main())
