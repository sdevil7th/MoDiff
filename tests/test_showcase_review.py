import json

import numpy as np

from PIL import Image

from scripts.build_showcase_review import build, media_file, temporal_continuity_metrics


def test_review_builder_preserves_closed_human_decisions(tmp_path):
    evidence = tmp_path / "approved-image"
    evidence.mkdir()
    Image.new("RGB", (256, 256), "navy").save(evidence / "showcase.webp")
    (tmp_path / "review-queue.json").write_text(
        json.dumps(
            {
                "campaignId": "review-status-proof",
                "items": [
                    {
                        "id": "approved",
                        "definitionId": "definition",
                        "mediaType": "image",
                        "status": "approved",
                        "evidenceDirectory": evidence.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build(tmp_path)

    assert result["items"][0]["status"] == "approved"


def test_media_file_prefers_generated_asset_over_companion_browser_screenshot(tmp_path):
    Image.new("RGB", (800, 450), "red").save(tmp_path / "qwen-proof-generated.png")
    Image.new("RGB", (256, 256), "navy").save(tmp_path / "qwen-proof-generated.webp")

    selected = media_file(tmp_path, "image")

    assert selected == tmp_path / "qwen-proof-generated.webp"


def test_review_builder_selects_an_explicit_asset_from_a_multi_output_directory(tmp_path):
    evidence = tmp_path / "multi-output"
    evidence.mkdir()
    Image.new("RGB", (256, 256), "navy").save(evidence / "first.webp")
    Image.new("RGB", (384, 256), "teal").save(evidence / "second.webp")
    (tmp_path / "review-queue.json").write_text(
        json.dumps(
            {
                "campaignId": "explicit-media-proof",
                "items": [
                    {
                        "id": "second-output",
                        "definitionId": "definition",
                        "mediaType": "image",
                        "mediaFile": "second.webp",
                        "status": "pending",
                        "evidenceDirectory": evidence.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build(tmp_path)

    assert result["items"][0]["media"]["path"] == "second.webp"
    assert result["items"][0]["media"]["width"] == 384


def test_media_file_rejects_an_explicit_path_outside_the_evidence_directory(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "without directories"):
        media_file(tmp_path, "image", "../outside.webp")


def test_temporal_continuity_inspects_every_adjacent_pair_and_flags_a_single_jump():
    yy, xx = np.mgrid[:72, :128]
    base = np.stack((xx * 2, yy * 3, (xx + yy) * 2), axis=-1).clip(0, 255).astype(np.uint8)
    smooth = [np.roll(base, index, axis=1) for index in range(12)]
    smooth_metrics = temporal_continuity_metrics(smooth)

    jumped = smooth.copy()
    jumped[6] = 255 - jumped[5]
    jumped_metrics = temporal_continuity_metrics(jumped)

    assert smooth_metrics["adjacentPairCount"] == 11
    assert smooth_metrics["checks"]["noRobustAbruptTransitionOutliers"] is True
    assert smooth_metrics["needsReviewerAttention"] is False
    assert jumped_metrics["checks"]["noRobustAbruptTransitionOutliers"] is False
    assert jumped_metrics["needsReviewerAttention"] is True
    assert 5 in jumped_metrics["abruptTransitionIndices"]


def test_temporal_continuity_flags_a_smooth_but_effectively_static_clip():
    yy, xx = np.mgrid[:72, :128]
    base = np.stack((xx * 2, yy * 3, (xx + yy) * 2), axis=-1).clip(0, 255).astype(np.uint8)
    # Sparse sub-pixel-like changes are enough to make frames non-identical,
    # but not enough to read as purposeful motion in a showcase clip.
    nearly_static = [base.copy() for _ in range(12)]
    for index, frame in enumerate(nearly_static):
        frame[index % frame.shape[0], :, :] = np.clip(
            frame[index % frame.shape[0], :, :].astype(np.int16) + 1,
            0,
            255,
        ).astype(np.uint8)

    metrics = temporal_continuity_metrics(nearly_static)

    assert metrics["checks"]["noRobustAbruptTransitionOutliers"] is True
    assert metrics["meaningfulAdjacentMotion"] is False
    assert metrics["needsReviewerAttention"] is True
