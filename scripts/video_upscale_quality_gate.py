#!/usr/bin/env python3
"""Screen one model-upscaled video and create a multi-timepoint review card."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import imageio.v3 as iio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modiff.asset_quality import assess_video_upscale, build_video_upscale_review_card  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_video(path: Path):
    metadata = iio.immeta(path, plugin="FFMPEG")
    frames = iio.imread(path, plugin="FFMPEG")
    if len(frames) > 300:
        raise ValueError(f"Quality review is bounded to 300 frames; {path} contains {len(frames)}.")
    return frames, float(metadata.get("fps") or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    args = parser.parse_args()
    source_frames, source_fps = _read_video(args.source)
    output_frames, output_fps = _read_video(args.output)
    assessment = assess_video_upscale(
        source_frames,
        output_frames,
        source_fps=source_fps,
        output_fps=output_fps,
    )
    document = assessment.to_dict()
    document["source"] = {"path": str(args.source.resolve()), "sha256": _sha256(args.source)}
    document["output"] = {"path": str(args.output.resolve()), "sha256": _sha256(args.output)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.card.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_video_upscale_review_card(source_frames, output_frames, assessment).save(
        args.card,
        format="WEBP",
        quality=94,
        method=6,
    )
    print(json.dumps(document, sort_keys=True))
    return 0 if assessment.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
