#!/usr/bin/env python3
"""Validate a showcase campaign and build a private, local review index."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import wave
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def validate_image(path: Path) -> tuple[dict, list[dict]]:
    with Image.open(path) as source:
        source.load()
        image = source.convert("RGB")
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256)
        probabilities = histogram[histogram > 0] / histogram.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        result = {
            "kind": "image",
            "path": path.name,
            "sha256": sha256_file(path),
            "byteSize": path.stat().st_size,
            "format": source.format,
            "width": image.width,
            "height": image.height,
            "mode": source.mode,
            "luminanceMean": round(float(gray.mean()), 4),
            "luminanceStdDev": round(float(gray.std()), 4),
            "luminanceEntropyBits": round(entropy, 4),
            "technicalChecks": {
                "decodes": True,
                "nonUniform": bool(gray.std() >= 2.0),
                "meaningfulDimensions": image.width >= 256 and image.height >= 256,
            },
        }
    return result, []


def sampled_video_frames(path: Path, sample_count: int = 9) -> tuple[list[np.ndarray], dict]:
    metadata = iio.immeta(path)
    fps = float(metadata.get("fps") or 0)
    duration = float(metadata.get("duration") or 0)
    expected_frames = max(1, int(round(duration * fps))) if duration > 0 and fps > 0 else 1
    wanted = set(np.linspace(0, max(0, expected_frames - 1), min(sample_count, expected_frames)).round().astype(int))
    frames: list[np.ndarray] = []
    for index, frame in enumerate(iio.imiter(path)):
        if index in wanted:
            frames.append(np.asarray(frame)[..., :3])
        if index >= max(wanted):
            break
    return frames, metadata


def video_contact_sheet(
    frames: list[np.ndarray],
    destination: Path,
    *,
    labels: list[str] | None = None,
) -> None:
    thumbnails = []
    for index, frame in enumerate(frames):
        image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        image.thumbnail((320, 200), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (336, 232), "#111827")
        tile.paste(image, ((336 - image.width) // 2, 8))
        draw = ImageDraw.Draw(tile)
        label = labels[index] if labels else f"sample {index + 1}/{len(frames)}"
        draw.text((12, 210), label, fill="#f9fafb")
        thumbnails.append(tile)
    columns = 3
    rows = math.ceil(len(thumbnails) / columns)
    sheet = Image.new("RGB", (columns * 336, rows * 232), "#030712")
    for index, tile in enumerate(thumbnails):
        sheet.paste(tile, ((index % columns) * 336, (index // columns) * 232))
    sheet.save(destination, format="PNG")


def temporal_continuity_metrics(frames: list[np.ndarray]) -> dict:
    """Summarize every adjacent transition without claiming aesthetic approval.

    Earlier review evidence sampled distant frames and could mistake frame jumps
    for desirable motion. This screen inspects every adjacent pair at a bounded
    resolution, reports robust outliers, and leaves final smoothness judgment to
    the reviewer looking at the native video and consecutive-frame sheet.
    """

    reduced: list[np.ndarray] = []
    for frame in frames:
        image = Image.fromarray(frame.astype(np.uint8), mode="RGB").convert("L")
        image.thumbnail((192, 108), Image.Resampling.BILINEAR)
        reduced.append(np.asarray(image, dtype=np.float32))
    differences = np.asarray(
        [float(np.mean(np.abs(right - left))) for left, right in zip(reduced, reduced[1:], strict=False)],
        dtype=np.float64,
    )
    correlations = []
    for left, right in zip(reduced, reduced[1:], strict=False):
        left_flat = left.ravel()
        right_flat = right.ravel()
        if left_flat.std() <= 1e-8 or right_flat.std() <= 1e-8:
            correlations.append(1.0 if np.array_equal(left_flat, right_flat) else 0.0)
        else:
            correlations.append(float(np.corrcoef(left_flat, right_flat)[0, 1]))
    if not differences.size:
        return {
            "adjacentPairCount": 0,
            "adjacentLuminanceMae": [],
            "adjacentCorrelation": [],
            "abruptTransitionIndices": [],
            "worstTransitionIndices": [],
            "checks": {"hasAdjacentPairs": False, "noRobustAbruptTransitionOutliers": False},
            "needsReviewerAttention": True,
            "policy": "Automated continuity screening may flag transitions but cannot approve perceived smoothness.",
        }

    median = float(np.median(differences))
    median_absolute_deviation = float(np.median(np.abs(differences - median)))
    robust_sigma = 1.4826 * median_absolute_deviation
    abrupt_threshold = max(8.0, median * 2.75, median + 6.0 * robust_sigma)
    abrupt = np.flatnonzero(differences > abrupt_threshold).astype(int).tolist()
    worst = np.argsort(differences)[::-1][: min(3, differences.size)].astype(int).tolist()
    correlation_values = np.asarray(correlations, dtype=np.float64)
    correlation_p05 = float(np.quantile(correlation_values, 0.05))
    difference_p95 = float(np.quantile(differences, 0.95))
    meaningful_adjacent_motion = median >= 0.75 or difference_p95 >= 1.0
    continuity_checks = {
        "hasAdjacentPairs": True,
        "noRobustAbruptTransitionOutliers": not abrupt,
        "noLowAdjacentStructuralCorrelationTail": correlation_p05 >= 0.50,
        "boundedTransitionDispersion": difference_p95 <= max(12.0, median * 1.90),
    }
    return {
        "adjacentPairCount": int(differences.size),
        "adjacentLuminanceMae": [round(float(value), 4) for value in differences],
        "adjacentCorrelation": [round(float(value), 6) for value in correlation_values],
        "luminanceMaeMedian": round(median, 4),
        "luminanceMaeP95": round(difference_p95, 4),
        "luminanceMaeMaximum": round(float(np.max(differences)), 4),
        "adjacentCorrelationP05": round(correlation_p05, 6),
        "meaningfulAdjacentMotion": meaningful_adjacent_motion,
        "robustAbruptThreshold": round(abrupt_threshold, 4),
        "abruptTransitionIndices": abrupt,
        "worstTransitionIndices": worst,
        "checks": continuity_checks,
        "needsReviewerAttention": not all(continuity_checks.values()) or not meaningful_adjacent_motion,
        "policy": (
            "This inspects all adjacent frames, flags abrupt luminance/structure changes, and separately screens "
            "for meaningful visual motion. It cannot approve perceived smoothness or prompt fidelity; inspect the "
            "native video and consecutive-frame evidence."
        ),
    }


def consecutive_review_frames(frames: list[np.ndarray], transition_indices: list[int]) -> tuple[list[np.ndarray], list[str]]:
    indices: list[int] = []
    anchors = [0, max(0, len(frames) // 2 - 2), max(0, len(frames) - 5)]
    anchors.extend(max(0, transition - 1) for transition in transition_indices)
    for anchor in anchors:
        for index in range(anchor, min(len(frames), anchor + 5)):
            if index not in indices:
                indices.append(index)
    return [frames[index] for index in indices], [f"consecutive frame {index}" for index in indices]


def validate_video(path: Path, evidence_dir: Path) -> tuple[dict, list[dict]]:
    metadata = iio.immeta(path)
    all_frames = [np.asarray(frame)[..., :3] for frame in iio.imiter(path)]
    if not all_frames:
        raise ValueError(f"Video decoded no frames: {path}")
    sampled_indices = np.linspace(0, len(all_frames) - 1, min(9, len(all_frames))).round().astype(int).tolist()
    frames = [all_frames[index] for index in sampled_indices]
    grayscale = [np.asarray(Image.fromarray(frame).convert("L"), dtype=np.float32) for frame in frames]
    differences = [float(np.mean(np.abs(right - left))) for left, right in zip(grayscale, grayscale[1:], strict=False)]
    contact_sheet = evidence_dir / "contact-sheet.png"
    video_contact_sheet(frames, contact_sheet, labels=[f"sample frame {index}" for index in sampled_indices])
    continuity = temporal_continuity_metrics(all_frames)
    consecutive_frames, consecutive_labels = consecutive_review_frames(
        all_frames,
        continuity["worstTransitionIndices"],
    )
    consecutive_sheet = evidence_dir / "consecutive-frame-sheet.png"
    video_contact_sheet(consecutive_frames, consecutive_sheet, labels=consecutive_labels)
    fps = float(metadata.get("fps") or 0)
    duration = float(metadata.get("duration") or 0)
    width, height = metadata.get("size") or metadata.get("source_size") or (0, 0)
    result = {
        "kind": "video",
        "path": path.name,
        "sha256": sha256_file(path),
        "byteSize": path.stat().st_size,
        "codec": metadata.get("codec"),
        "pixelFormat": metadata.get("pix_fmt"),
        "width": int(width),
        "height": int(height),
        "fps": fps,
        "durationSeconds": duration,
        "estimatedFrameCount": int(round(fps * duration)) if fps and duration else None,
        "decodedFrameCount": len(all_frames),
        "sampledFrameCount": len(frames),
        "sampledMeanAbsoluteFrameDifferences": [round(value, 4) for value in differences],
        "sampledMotionMean": round(float(np.mean(differences)), 4) if differences else 0.0,
        "sampledMotionMaximum": round(float(np.max(differences)), 4) if differences else 0.0,
        "temporalContinuity": continuity,
        "technicalChecks": {
            "decodes": True,
            "hasMultipleFrames": len(frames) > 1,
            "nonStatic": bool(
                differences
                and np.mean(differences) >= 0.75
                and continuity.get("meaningfulAdjacentMotion") is True
            ),
            "meaningfulDimensions": int(width) >= 256 and int(height) >= 256,
        },
    }
    return result, [
        {"kind": "image", "label": "Whole-clip samples", "path": contact_sheet.name},
        {
            "kind": "image",
            "label": "Consecutive frames, including worst measured transitions",
            "path": consecutive_sheet.name,
        },
    ]


def pcm_float(data: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if sample_width == 3:
        raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        signed = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        signed = (signed ^ 0x800000) - 0x800000
        return signed.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width}")


def audio_waveform(samples: np.ndarray, channels: int, destination: Path) -> None:
    channel_samples = samples.reshape(-1, channels).mean(axis=1)
    width, height = 1200, 320
    bucket = max(1, math.ceil(channel_samples.size / width))
    padded = np.pad(channel_samples, (0, (-channel_samples.size) % bucket))
    envelope = np.max(np.abs(padded.reshape(-1, bucket)), axis=1)[:width]
    canvas = Image.new("RGB", (width, height), "#030712")
    draw = ImageDraw.Draw(canvas)
    center = height // 2
    draw.line((0, center, width, center), fill="#334155", width=1)
    for x, amplitude in enumerate(envelope):
        radius = max(1, int(amplitude * (height // 2 - 12)))
        draw.line((x, center - radius, x, center + radius), fill="#facc15", width=1)
    canvas.save(destination, format="PNG")


def validate_audio(path: Path, evidence_dir: Path) -> tuple[dict, list[dict]]:
    with wave.open(str(path), "rb") as stream:
        channels = stream.getnchannels()
        sample_rate = stream.getframerate()
        sample_width = stream.getsampwidth()
        frames = stream.getnframes()
        compression = stream.getcomptype()
        raw = stream.readframes(frames)
    if compression != "NONE":
        raise ValueError(f"Unsupported compressed WAV: {compression}")
    samples = pcm_float(raw, sample_width)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    clipping = float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 0.0
    waveform = evidence_dir / "waveform.png"
    audio_waveform(samples, channels, waveform)
    duration = frames / sample_rate if sample_rate else 0.0
    result = {
        "kind": "audio",
        "path": path.name,
        "sha256": sha256_file(path),
        "byteSize": path.stat().st_size,
        "channels": channels,
        "sampleRate": sample_rate,
        "sampleWidthBytes": sample_width,
        "frameCount": frames,
        "durationSeconds": round(duration, 6),
        "peakAmplitude": round(peak, 6),
        "rmsAmplitude": round(rms, 6),
        "clippedSampleRatio": round(clipping, 8),
        "technicalChecks": {
            "decodes": True,
            "nonSilent": rms >= 0.001,
            "notHeavilyClipped": clipping <= 0.01,
            "meaningfulDuration": duration >= 10,
            "sampleRateAtLeast44100": sample_rate >= 44_100,
        },
    }
    return result, [{"kind": "image", "label": "Waveform", "path": waveform.name}]


def media_file(evidence_dir: Path, media_type: str, filename: str | None = None) -> Path | None:
    suffixes = {"image": {".png", ".webp", ".jpg", ".jpeg"}, "video": {".mp4"}, "audio": {".wav"}}[media_type]
    ignored = {
        "cluster-generated.png",
        "frontend-generated.png",
        "frontend-model-ready.png",
        "frontend-download-started.png",
        "contact-sheet.png",
        "consecutive-frame-sheet.png",
        "waveform.png",
    }

    def is_browser_screenshot(path: Path) -> bool:
        if path.name in ignored or path.name.startswith("frontend-"):
            return True
        if media_type != "image" or path.suffix.lower() != ".png":
            return False
        # Live browser qualifications save the full-page screenshot as
        # `<scenario>-generated.png` and the actual backend asset beside it as
        # `<scenario>-generated.webp`. Prefer the real asset deterministically;
        # otherwise the review page can accidentally display app chrome instead
        # of the generated image.
        return any(path.with_suffix(suffix).is_file() for suffix in (".webp", ".jpg", ".jpeg"))

    if filename is not None:
        # A qualification can produce several reviewable assets in one evidence
        # directory (for example, multiple reference-image modes or RGBA layers).
        # Keep the queue declarative and prevent it from escaping its evidence
        # directory while allowing each item to select its exact generated asset.
        if Path(filename).name != filename:
            raise ValueError(f"mediaFile must be a filename without directories: {filename!r}")
        selected = evidence_dir / filename
        if selected.suffix.lower() not in suffixes:
            raise ValueError(f"mediaFile has the wrong type for {media_type}: {filename!r}")
        return selected if selected.is_file() else None

    candidates = [
        path
        for path in evidence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes and not is_browser_screenshot(path)
    ]
    return sorted(candidates, key=lambda item: ("showcase" not in item.name, item.name))[0] if candidates else None


def item_html(item: dict) -> str:
    media = item.get("media")
    body = '<p class="waiting">Generated asset is not ready yet.</p>'
    if media:
        path = html.escape(f"{item['evidenceDirectory']}/{media['path']}")
        if item["mediaType"] == "image":
            body = f'<a href="{path}"><img src="{path}" alt="{html.escape(item["id"])}"></a>'
        elif item["mediaType"] == "video":
            body = f'<video controls preload="metadata" src="{path}"></video>'
        else:
            body = f'<audio controls preload="metadata" src="{path}"></audio>'
    materials = "".join(
        f'<figure><img src="{html.escape(item["evidenceDirectory"] + "/" + material["path"])}" '
        f'alt="{html.escape(material["label"])}"><figcaption>{html.escape(material["label"])}</figcaption></figure>'
        for material in item.get("reviewMaterials", [])
    )
    checks = media.get("technicalChecks", {}) if media else {}
    return (
        f'<article><h2>{html.escape(item["id"])}</h2><p>{html.escape(item["definitionId"])}</p>'
        f'<p>Status: <strong>{html.escape(item["status"])}</strong></p>{body}{materials}'
        f'<pre>{html.escape(json.dumps(checks, indent=2))}</pre></article>'
    )


def build(campaign_root: Path) -> dict:
    campaign_root = campaign_root.resolve()
    queue = json.loads((campaign_root / "review-queue.json").read_text(encoding="utf-8"))
    items = []
    for queued in queue["items"]:
        evidence_dir = campaign_root / queued["evidenceDirectory"]
        evidence_dir.mkdir(parents=True, exist_ok=True)
        path = media_file(evidence_dir, queued["mediaType"], queued.get("mediaFile"))
        media = None
        materials: list[dict] = []
        if path:
            if queued["mediaType"] == "image":
                media, materials = validate_image(path)
            elif queued["mediaType"] == "video":
                media, materials = validate_video(path, evidence_dir)
            else:
                media, materials = validate_audio(path, evidence_dir)
            write_json(evidence_dir / "media-validation.json", media)
        requested_status = queued["status"]
        status = (
            "pending_review"
            if media and requested_status in {"queued", "pending"}
            else requested_status
        )
        items.append(
            {
                **queued,
                "status": status,
                "media": media,
                "reviewMaterials": materials,
            }
        )
    document = {"schemaVersion": 1, "campaignId": queue["campaignId"], "items": items}
    write_json(campaign_root / "review-index.json", document)
    cards = "".join(item_html(item) for item in items)
    (campaign_root / "index.html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>MoDiff showcase review</title>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;background:#070b12;color:#e5e7eb}"
        "article{border:1px solid #334155;padding:20px;margin:24px 0;background:#111827}img,video{max-width:100%;height:auto}"
        "audio{width:100%}figure{margin:20px 0}pre{white-space:pre-wrap;color:#cbd5e1}.waiting{color:#facc15}</style>"
        "</head><body><h1>MoDiff Diffusers showcase review</h1><p>Private qualification outputs; no item is public or "
        f"approved until a reviewer records that decision.</p>{cards}</body></html>\n",
        encoding="utf-8",
    )
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    document = build(args.campaign_root)
    ready = sum(1 for item in document["items"] if item["media"])
    print(f"Built private review index with {ready}/{len(document['items'])} generated assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
