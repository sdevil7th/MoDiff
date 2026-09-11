#!/usr/bin/env python3
"""Create a reproducible motion-rich low-resolution video-upscale fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as iio
import imageio.v3 as iio_v3
import numpy as np
from PIL import Image, ImageOps


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
    parser.add_argument("--height", type=int, default=288)
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
        parser.error("fixture dimensions must be positive multiples of 16")

    metadata = iio_v3.immeta(args.source, plugin="FFMPEG")
    source_fps = float(metadata.get("fps") or 0)
    source_frames = iio_v3.imread(args.source, plugin="FFMPEG")
    if source_fps <= 0 or not 24 <= len(source_frames) <= 300:
        parser.error("source clip must contain 24 to 300 frames with a positive frame rate")
    transformed = [
        ImageOps.fit(
            Image.fromarray(frame[..., :3].astype("uint8"), "RGB"),
            (args.width, args.height),
            method=Image.Resampling.LANCZOS,
        )
        for frame in source_frames
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with iio.get_writer(
        args.output,
        format="FFMPEG",
        mode="I",
        fps=source_fps,
        codec="libx264",
        quality=6,
        macro_block_size=16,
        ffmpeg_log_level="error",
        output_params=["-pix_fmt", "yuv420p"],
    ) as writer:
        for frame in transformed:
            writer.append_data(np.asarray(frame))

    record = {
        "schemaVersion": 1,
        "kind": "video_upscale_input_fixture",
        "source": {
            "path": str(args.source.resolve()),
            "sha256": _sha256(args.source),
            "width": int(source_frames.shape[2]),
            "height": int(source_frames.shape[1]),
            "frames": len(source_frames),
            "fps": source_fps,
        },
        "fixture": {
            "path": str(args.output.resolve()),
            "sha256": _sha256(args.output),
            "width": args.width,
            "height": args.height,
            "frames": len(source_frames),
            "fps": source_fps,
            "format": "H.264 MP4",
        },
        "recipe": {
            "fit": "Pillow LANCZOS center crop",
            "codec": "libx264",
            "imageioQuality": 6,
            "purpose": "A five-second motion-rich low-resolution input for exact native-2x video restoration review.",
        },
        "rights": {
            "sourceUserReview": "improve_if_possible_else_keep_fallback",
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
