"""Generate a hash-locked direct-wheel requirements file for a release manifest profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

from modiff.runtime_profile import MANIFEST_PATH


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="amd-rocm-linux")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    profile = manifest["profiles"][args.profile]
    requirement = Path(profile["requirements"])
    output = args.output or Path(__file__).resolve().parents[1] / requirement
    urls = [line.strip().split()[0] for line in output.read_text(encoding="utf-8").splitlines() if line.strip().startswith("https://")]
    lines = []
    with tempfile.TemporaryDirectory(prefix="modiff-wheel-lock-") as temporary:
        for url in urls:
            name = urllib.parse.unquote(url.rsplit("/", 1)[-1])
            target = Path(temporary) / name
            digest = hashlib.sha256()
            with urllib.request.urlopen(url) as response, target.open("wb") as handle:
                while chunk := response.read(8 * 1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
            lines.append(f"{url} --hash=sha256:{digest.hexdigest()}")
    lines.append("-e .")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
