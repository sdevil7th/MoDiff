"""Original procedural input fixtures for hidden candidates that still need media.

The sealed authoring ledger records 92 workflows as
``draft_complete_input_selection_pending``. This module writes original MoDiff
geometric stills, sine tones, and motion clips into ``review-pending/`` so a
human can review rights and suitability. It never selects those bytes into the
authoring ledger, copies third-party media, or claims Gallery approval.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import wave

from PIL import Image, ImageDraw

from modiff.local_review_receipts import canonical_content_hash


SCHEMA_VERSION = 1
AUTHORING_SPEC_PATH = Path("data/template-authoring-specs.v1.json")
AUTHORING_PENDING_STATE = "draft_complete_input_selection_pending"
KIT_DIRNAME = "kit"
LEDGER_NAME = "ledger.v1.json"
HTML_NAME = "index.html"
_BOUNDARY = {
    "selectsInputAssets": False,
    "authoringSpecMutated": False,
    "thirdPartyMedia": False,
    "rightsApproved": False,
    "galleryRegistered": False,
    "datasetPublished": False,
    "publicTemplateMutated": False,
    "executesWorkflows": False,
}
_FIELD_KEYS = {
    ("image", "referenceImages"): ("still_life.png", "tram_stop.png"),
    ("image", "controlImage"): ("control_lines.png",),
    ("image", "maskImage"): ("mask.png",),
    ("image", "lastImage"): ("tram_stop.png",),
    ("video", "sourceVideo"): ("motion.mp4",),
    ("video", "controlVideo"): ("control_motion.mp4",),
    ("video", "referenceVideos"): ("motion.mp4", "tram_motion.mp4"),
    ("video", "poseVideo"): ("pose_motion.mp4",),
    ("video", "faceVideo"): ("face_motion.mp4",),
    ("video", "backgroundVideo"): ("background_motion.mp4",),
    ("video", "maskVideo"): ("mask_motion.mp4",),
    ("audio", "sourceAudio"): ("tone_a.wav",),
    ("audio", "referenceAudio"): ("tone_b.wav",),
}


class OriginalInputFixtureError(ValueError):
    """Raised when original input fixtures cannot be authored fail-closed."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _require_imageio():
    try:
        import imageio.v2 as imageio
        import numpy as np
    except ImportError as error:
        raise OriginalInputFixtureError(
            "Original video fixtures require imageio and numpy in the project environment."
        ) from error
    return imageio, np


def _still_life(size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), (232, 221, 204))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([96, 168, 416, 372], radius=28, fill=(36, 78, 140))
    draw.rectangle([118, 198, 286, 338], fill=(58, 98, 156))
    for y in range(210, 330, 12):
        draw.line([(128, y), (276, y)], fill=(186, 204, 222), width=2)
    draw.ellipse([304, 198, 394, 288], fill=(212, 154, 64))
    draw.ellipse([338, 232, 360, 254], fill=(90, 62, 28))
    draw.rectangle([168, 372, 208, 404], fill=(90, 74, 58))
    draw.rectangle([304, 372, 344, 404], fill=(90, 74, 58))
    return image


def _tram_stop(size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), (46, 58, 74))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 340, size, size], fill=(78, 86, 96))
    draw.polygon([(210, 140), (360, 320), (60, 320)], fill=(176, 42, 42))
    draw.rectangle([248, 320, 264, 400], fill=(54, 40, 36))
    draw.ellipse([40, 48, 120, 128], fill=(214, 196, 148))
    return image


def _control_lines(size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), (250, 250, 250))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle([96, 168, 416, 372], radius=28, outline=(20, 20, 20), width=6)
    draw.ellipse([304, 198, 394, 288], outline=(20, 20, 20), width=5)
    draw.rectangle([118, 198, 286, 338], outline=(20, 20, 20), width=4)
    return image


def _mask(size: int = 512) -> Image.Image:
    image = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle([96, 168, 416, 372], fill=(255, 255, 255))
    return image


def _write_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def _write_tone(path: Path, *, frequency: float, seconds: float = 2.0, sample_rate: int = 48000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    amplitude = 12000
    frames = int(seconds * sample_rate)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        payload = bytearray()
        for index in range(frames):
            sample = int(amplitude * math.sin(2 * math.pi * frequency * index / sample_rate))
            payload.extend(int(sample).to_bytes(2, "little", signed=True))
        handle.writeframes(bytes(payload))


def _write_video(path: Path, frames: list[Image.Image], *, fps: int = 8) -> None:
    imageio, np = _require_imageio()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame.convert("RGB")))
    finally:
        writer.close()


def _motion_frames(base: Image.Image, *, count: int = 8) -> list[Image.Image]:
    frames = []
    for index in range(count):
        canvas = base.copy()
        draw = ImageDraw.Draw(canvas)
        x = 24 + index * 18
        draw.rectangle([x, 24, x + 48, 72], fill=(240, 232, 210))
        frames.append(canvas)
    return frames


def _line_motion_frames(size: int = 256, *, count: int = 8) -> list[Image.Image]:
    frames = []
    for index in range(count):
        image = Image.new("RGB", (size, size), (250, 250, 250))
        draw = ImageDraw.Draw(image)
        y = 40 + index * 18
        draw.line([(32, y), (size - 32, y + 24)], fill=(20, 20, 20), width=6)
        draw.rectangle([80, 80, 176, 176], outline=(20, 20, 20), width=4)
        frames.append(image)
    return frames


def _pose_frames(size: int = 256, *, count: int = 8) -> list[Image.Image]:
    frames = []
    for index in range(count):
        image = Image.new("RGB", (size, size), (236, 232, 224))
        draw = ImageDraw.Draw(image)
        cx = 128 + index * 4
        draw.ellipse([cx - 16, 48, cx + 16, 80], outline=(30, 30, 30), width=4)
        draw.line([(cx, 80), (cx, 150)], fill=(30, 30, 30), width=4)
        draw.line([(cx, 100), (cx - 30, 130)], fill=(30, 30, 30), width=4)
        draw.line([(cx, 100), (cx + 30, 130 + index)], fill=(30, 30, 30), width=4)
        draw.line([(cx, 150), (cx - 24, 210)], fill=(30, 30, 30), width=4)
        draw.line([(cx, 150), (cx + 24, 210)], fill=(30, 30, 30), width=4)
        frames.append(image)
    return frames


def _face_frames(size: int = 256, *, count: int = 8) -> list[Image.Image]:
    frames = []
    for index in range(count):
        image = Image.new("RGB", (size, size), (244, 228, 210))
        draw = ImageDraw.Draw(image)
        draw.ellipse([48, 48, 208, 208], fill=(232, 196, 168), outline=(90, 62, 48), width=4)
        eye_y = 110 + (index % 3)
        draw.ellipse([88, eye_y, 112, eye_y + 16], fill=(40, 32, 28))
        draw.ellipse([144, eye_y, 168, eye_y + 16], fill=(40, 32, 28))
        draw.arc([96, 140, 160, 180], start=20, end=160, fill=(90, 42, 42), width=4)
        frames.append(image)
    return frames


def _background_frames(size: int = 256, *, count: int = 8) -> list[Image.Image]:
    frames = []
    for index in range(count):
        shift = 8 * index
        image = Image.new("RGB", (size, size), (70 + shift, 96, 128))
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 170, size, size], fill=(48, 64, 52))
        frames.append(image)
    return frames


def _mask_frames(size: int = 256, *, count: int = 8) -> list[Image.Image]:
    frames = []
    for index in range(count):
        image = Image.new("RGB", (size, size), (0, 0, 0))
        draw = ImageDraw.Draw(image)
        x = 40 + index * 12
        draw.ellipse([x, 70, x + 96, 166], fill=(255, 255, 255))
        frames.append(image)
    return frames


def write_fixture_kit(destination: Path) -> dict[str, dict]:
    kit = destination / KIT_DIRNAME
    if kit.exists():
        for child in kit.iterdir():
            if child.is_file() and not child.is_symlink():
                child.unlink()
    kit.mkdir(parents=True, exist_ok=True)

    still = _still_life()
    tram = _tram_stop()
    files = {
        "still_life.png": lambda path: _write_png(path, still),
        "tram_stop.png": lambda path: _write_png(path, tram),
        "control_lines.png": lambda path: _write_png(path, _control_lines()),
        "mask.png": lambda path: _write_png(path, _mask()),
        "tone_a.wav": lambda path: _write_tone(path, frequency=440.0),
        "tone_b.wav": lambda path: _write_tone(path, frequency=330.0),
        "motion.mp4": lambda path: _write_video(path, _motion_frames(still.resize((256, 256)))),
        "tram_motion.mp4": lambda path: _write_video(path, _motion_frames(tram.resize((256, 256)))),
        "control_motion.mp4": lambda path: _write_video(path, _line_motion_frames()),
        "pose_motion.mp4": lambda path: _write_video(path, _pose_frames()),
        "face_motion.mp4": lambda path: _write_video(path, _face_frames()),
        "background_motion.mp4": lambda path: _write_video(path, _background_frames()),
        "mask_motion.mp4": lambda path: _write_video(path, _mask_frames()),
    }
    assets = {}
    for name, writer in files.items():
        path = kit / name
        writer(path)
        if path.is_symlink() or not path.is_file():
            raise OriginalInputFixtureError(f"Original fixture was not written: {path}")
        suffix = path.suffix.lower()
        media_kind = {".png": "image", ".wav": "audio", ".mp4": "video"}[suffix]
        assets[name] = {
            "fileName": name,
            "relativePath": f"{KIT_DIRNAME}/{name}",
            "mediaKind": media_kind,
            "byteSize": path.stat().st_size,
            "sha256": _sha256_file(path),
            "provenance": "original_modiff_procedural_v1",
        }
    return assets


def _pending_specifications(authoring: dict) -> list[dict]:
    specifications = authoring.get("specifications")
    if not isinstance(specifications, list):
        raise OriginalInputFixtureError("Authoring ledger is missing specifications.")
    pending = [
        specification
        for specification in specifications
        if specification.get("authoringState") == AUTHORING_PENDING_STATE
    ]
    if not pending:
        raise OriginalInputFixtureError("No input-selection-pending authoring specifications were found.")
    return pending


def _map_item(item: dict, assets: dict[str, dict]) -> dict:
    field = item.get("field")
    media_kind = item.get("mediaKind")
    minimum = item.get("minimumCount", 1)
    if not isinstance(field, str) or not isinstance(media_kind, str):
        raise OriginalInputFixtureError("Authoring input item is missing field or mediaKind.")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise OriginalInputFixtureError(f"Authoring input item {field} has an invalid minimumCount.")
    names = _FIELD_KEYS.get((media_kind, field))
    if names is None:
        raise OriginalInputFixtureError(f"No original fixture mapping exists for {media_kind}:{field}.")
    if len(names) < minimum:
        raise OriginalInputFixtureError(f"Original fixture kit does not cover {minimum} assets for {field}.")
    selected = [assets[name] for name in names[:minimum]]
    return {
        "field": field,
        "mediaKind": media_kind,
        "minimumCount": minimum,
        "selectionState": "original_fixture_staged_review_required",
        "rightsState": "review_required",
        "authoringLedgerSelected": False,
        "fixtures": [
            {
                "fileName": asset["fileName"],
                "relativePath": asset["relativePath"],
                "sha256": asset["sha256"],
                "byteSize": asset["byteSize"],
            }
            for asset in selected
        ],
        "technicalRequirements": list(item.get("technicalRequirements") or []),
    }


def build_original_input_fixture_ledger(
    root: Path,
    *,
    authoring: dict | None = None,
    assets: dict[str, dict],
) -> dict:
    root = root.resolve()
    if authoring is None:
        path = root / AUTHORING_SPEC_PATH
        if path.is_symlink() or not path.is_file():
            raise OriginalInputFixtureError("Authoring specification ledger is missing.")
        authoring = json.loads(path.read_text(encoding="utf-8"))

    mappings = []
    for specification in _pending_specifications(authoring):
        workflow_id = specification.get("canonicalWorkflowId")
        items = ((specification.get("inputPlan") or {}).get("items")) or []
        if not isinstance(workflow_id, str) or not workflow_id or not items:
            raise OriginalInputFixtureError("Pending authoring specification is missing workflow identity or inputs.")
        mappings.append(
            {
                "canonicalWorkflowId": workflow_id,
                "authoringSpecId": specification.get("id"),
                "mode": specification.get("mode"),
                "mediaKind": specification.get("mediaKind"),
                "inputPlanStatus": specification.get("inputPlan", {}).get("status"),
                "items": [_map_item(item, assets) for item in items],
                "claims": {
                    "inputSelected": False,
                    "rightsApproved": False,
                    "workflowExecuted": False,
                    "publicTemplate": False,
                    "authoringSpecMutated": False,
                },
            }
        )
    field_counts = Counter()
    for mapping in mappings:
        for item in mapping["items"]:
            field_counts[f"{item['mediaKind']}:{item['field']}"] += 1
    ledger = {
        "schemaVersion": SCHEMA_VERSION,
        "kind": "original_input_fixture_ledger",
        "boundary": dict(_BOUNDARY),
        "provenance": "original_modiff_procedural_v1",
        "policy": {
            "doesNotMutateAuthoringLedger": True,
            "doesNotSelectInputs": True,
            "rightsRemainReviewRequired": True,
            "thirdPartyMediaForbidden": True,
        },
        "kit": sorted(assets.values(), key=lambda item: item["fileName"]),
        "workflows": mappings,
        "stagedAt": _now(),
        "summary": {
            "workflowCount": len(mappings),
            "kitAssetCount": len(assets),
            "fieldCounts": dict(sorted(field_counts.items())),
        },
    }
    ledger["contentHash"] = canonical_content_hash(ledger)
    return ledger


def render_original_input_fixture_html(ledger: dict) -> str:
    kit = []
    for asset in ledger["kit"]:
        relative = asset["relativePath"]
        kind = asset["mediaKind"]
        if kind == "image":
            media = f'<img src="{relative}" alt="{asset["fileName"]}" />'
        elif kind == "video":
            media = f'<video controls src="{relative}"></video>'
        else:
            media = f'<audio controls src="{relative}"></audio>'
        kit.append(f"<figure>{media}<figcaption>{asset['fileName']}</figcaption></figure>")
    rows = []
    for workflow in ledger["workflows"]:
        fields = ", ".join(f"{item['field']}→{item['fixtures'][0]['fileName']}" for item in workflow["items"])
        rows.append(f"<li><code>{workflow['canonicalWorkflowId']}</code> — {fields}</li>")
    return (
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\" />"
        "<title>MoDiff original input fixtures</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px}"
        "img,video{max-width:240px;height:auto}figure{display:inline-block;margin:8px 16px 16px 0}"
        "code{font-size:12px}</style></head><body>"
        "<h1>Original input fixtures</h1>"
        "<p>These files are original MoDiff procedural media. They are not selected into the "
        "sealed authoring ledger and are not Gallery examples. Rights review is still required.</p>"
        "<h2>Kit</h2>"
        + "".join(kit)
        + f"<h2>Mapped workflows ({ledger['summary']['workflowCount']})</h2><ul>"
        + "".join(rows)
        + "</ul></body></html>\n"
    )


def write_original_input_fixtures(root: Path, destination: Path | None = None) -> dict:
    root = root.resolve()
    target = destination if destination is not None else root / "review-pending" / "input-fixtures"
    target.mkdir(parents=True, exist_ok=True)
    assets = write_fixture_kit(target)
    ledger = build_original_input_fixture_ledger(root, assets=assets)
    _write_json(target / LEDGER_NAME, ledger)
    (target / HTML_NAME).write_text(render_original_input_fixture_html(ledger), encoding="utf-8")
    return ledger
