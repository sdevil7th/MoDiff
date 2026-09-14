#!/usr/bin/env python3
"""Screen one model-upscaled image and create a human-review comparison card."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.asset_quality import assess_image_upscale, build_image_upscale_review_card  # noqa: E402


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
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    args = parser.parse_args()
    with Image.open(args.source) as source_image, Image.open(args.output) as output_image:
        source = source_image.convert("RGB")
        output = output_image.convert("RGB")
    assessment = assess_image_upscale(source, output)
    document = assessment.to_dict()
    document["source"] = {"path": str(args.source.resolve()), "sha256": _sha256(args.source)}
    document["output"] = {"path": str(args.output.resolve()), "sha256": _sha256(args.output)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.card.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_image_upscale_review_card(source, output, assessment).save(args.card, format="WEBP", quality=94, method=6)
    print(json.dumps(document, sort_keys=True))
    return 0 if assessment.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
