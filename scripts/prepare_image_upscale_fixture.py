#!/usr/bin/env python3
"""Create a reproducible low-resolution input for an image-upscale review run."""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=72)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        parser.error("fixture dimensions must be positive")
    if not 1 <= args.jpeg_quality <= 95:
        parser.error("JPEG quality must be between 1 and 95")

    with Image.open(args.source) as opened:
        source = opened.convert("RGB")
    resized = source.resize((args.width, args.height), Image.Resampling.LANCZOS)
    encoded = BytesIO()
    resized.save(encoded, format="JPEG", quality=args.jpeg_quality, subsampling=2, optimize=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded.getvalue())
    record = {
        "schemaVersion": 1,
        "kind": "image_upscale_input_fixture",
        "source": {
            "path": str(args.source.resolve()),
            "sha256": _sha256(args.source),
            "width": source.width,
            "height": source.height,
        },
        "fixture": {
            "path": str(args.output.resolve()),
            "sha256": _sha256(args.output),
            "width": args.width,
            "height": args.height,
            "format": "JPEG",
        },
        "recipe": {
            "resize": "Pillow LANCZOS",
            "jpegQuality": args.jpeg_quality,
            "jpegSubsampling": 2,
            "purpose": "A realistic low-resolution, compressed input for exact native-2x restoration review.",
        },
        "rights": {
            "sourceUserQualityApproval": "approved",
            "publicationApproval": "pending",
            "derivedFixturePublicationApproval": "pending",
        },
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
