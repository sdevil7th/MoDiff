#!/usr/bin/env python3
"""Reject objective text-to-image defects and build a human-review card."""

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

from modiff.asset_quality import assess_text_to_image, build_text_to_image_review_card  # noqa: E402
from modiff.local_review_receipts import canonical_content_hash  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _comparison_images(root: Path, output: Path) -> tuple[list[Image.Image], list[str]]:
    images = []
    paths = []
    if not root.is_dir():
        return images, paths
    for path in sorted(root.glob("*/*")):
        if path.resolve() == output.resolve() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "comparison-card" in path.name:
            continue
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
                paths.append(str(path.resolve()))
        except (OSError, ValueError):
            continue
    return images, paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, default=ROOT / "review-pending")
    parser.add_argument("--title", default="TEXT TO IMAGE · NATIVE SHOWCASE")
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    prompt = (recipe.get("recipe") or {}).get("prompt")
    width = (recipe.get("recipe") or {}).get("width")
    height = (recipe.get("recipe") or {}).get("height")
    if not isinstance(prompt, str) or not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("Recipe must bind a prompt and integer width/height.")
    with Image.open(args.output) as source:
        output = source.convert("RGB")
    comparison_images, comparison_paths = _comparison_images(args.comparison_root, args.output)
    assessment = assess_text_to_image(
        output,
        expected_size=(width, height),
        comparison_images=comparison_images,
    )
    document = assessment.to_dict()
    document["recipe"] = {
        "path": str(args.recipe.resolve()),
        "sha256": _sha256(args.recipe),
        "recipeContentHash": canonical_content_hash(recipe["recipe"]),
    }
    document["output"] = {"path": str(args.output.resolve()), "sha256": _sha256(args.output)}
    document["comparisonSet"] = {
        "root": str(args.comparison_root.resolve()),
        "imageCount": len(comparison_paths),
        "paths": comparison_paths,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.card.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_text_to_image_review_card(
        output,
        assessment,
        title=args.title,
        prompt=prompt,
    ).save(args.card, format="WEBP", quality=94, method=6)
    print(json.dumps(document, sort_keys=True))
    return 0 if assessment.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
