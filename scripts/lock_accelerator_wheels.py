"""Generate a hash-locked direct-wheel requirements file for a release manifest profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from modiff.runtime_profile import MANIFEST_PATH


def parse_direct_wheel_requirements(path: Path) -> tuple[list[str], str]:
    """Return direct HTTPS wheel URLs and the editable project requirement.

    This maintenance command is intentionally limited to profiles made from
    direct wheel URLs. Rejecting index-based and ordinary pinned profiles
    prevents an accidental invocation from replacing their requirements with
    only the editable project line.
    """

    urls: list[str] = []
    editable_requirements: list[str] = []
    unsupported: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        first_token = line.split(maxsplit=1)[0]
        if first_token.startswith("https://"):
            urls.append(first_token)
        elif line.startswith("-e "):
            editable_requirements.append(line)
        else:
            unsupported.append(line)

    if unsupported:
        raise ValueError(
            "profile is not direct-wheel-only; unsupported requirement lines: "
            + ", ".join(unsupported)
        )
    if not urls:
        raise ValueError("profile contains no direct HTTPS wheel URLs")
    if len(editable_requirements) > 1:
        raise ValueError("profile contains more than one editable project requirement")
    return urls, editable_requirements[0] if editable_requirements else "-e ."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="amd-rocm-linux")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict) or args.profile not in profiles:
        parser.error(f"unknown accelerator profile: {args.profile}")
    profile = profiles[args.profile]
    requirement = Path(profile["requirements"])
    repository_root = Path(__file__).resolve().parents[1]
    if args.output is None and (requirement.is_absolute() or ".." in requirement.parts):
        parser.error(f"profile requirements path must stay inside the repository: {requirement}")
    output = args.output or repository_root / requirement
    try:
        urls, editable_requirement = parse_direct_wheel_requirements(output)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    lines = []
    for url in urls:
        digest = hashlib.sha256()
        with urllib.request.urlopen(url) as response:
            while chunk := response.read(8 * 1024 * 1024):
                digest.update(chunk)
        lines.append(f"{url} --hash=sha256:{digest.hexdigest()}")
    lines.append(editable_requirement)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
