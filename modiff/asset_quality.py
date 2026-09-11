"""Deterministic screening and review-card helpers for generated assets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class ImageUpscaleAssessment:
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    expected_scale: int
    dynamic_range: float
    reconstruction_mae: float
    detail_ratio: float
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "kind": "image_upscale_quality_screen",
            "sourceSize": {"width": self.source_size[0], "height": self.source_size[1]},
            "outputSize": {"width": self.output_size[0], "height": self.output_size[1]},
            "expectedScale": self.expected_scale,
            "metrics": {
                "dynamicRange": round(self.dynamic_range, 6),
                "reconstructionMae": round(self.reconstruction_mae, 6),
                "detailRatioVersusBicubic": round(self.detail_ratio, 6),
            },
            "checks": self.checks,
            "automatedStatus": "pass" if self.passed else "reject",
            "approvalStatus": "human_review_required" if self.passed else "not_eligible",
            "policy": "Automated screening may reject but never approve a Gallery asset.",
        }


@dataclass(frozen=True)
class VideoUpscaleAssessment:
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    source_frames: int
    output_frames: int
    source_fps: float
    output_fps: float
    expected_scale: int
    temporal_mae: float
    changed_pixel_ratio: float
    reconstruction_mae: float
    detail_ratio: float
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "kind": "video_upscale_quality_screen",
            "sourceSize": {"width": self.source_size[0], "height": self.source_size[1]},
            "outputSize": {"width": self.output_size[0], "height": self.output_size[1]},
            "sourceFrames": self.source_frames,
            "outputFrames": self.output_frames,
            "sourceFps": round(self.source_fps, 6),
            "outputFps": round(self.output_fps, 6),
            "sourceDurationSeconds": round(self.source_frames / self.source_fps, 6),
            "outputDurationSeconds": round(self.output_frames / self.output_fps, 6),
            "expectedScale": self.expected_scale,
            "metrics": {
                "temporalMae": round(self.temporal_mae, 6),
                "changedPixelRatio": round(self.changed_pixel_ratio, 6),
                "reconstructionMae": round(self.reconstruction_mae, 6),
                "detailRatioVersusBicubic": round(self.detail_ratio, 6),
            },
            "checks": self.checks,
            "automatedStatus": "pass" if self.passed else "reject",
            "approvalStatus": "human_review_required" if self.passed else "not_eligible",
            "policy": "Automated screening may reject but never approve a Gallery asset.",
        }


@dataclass(frozen=True)
class TextToImageAssessment:
    output_size: tuple[int, int]
    expected_size: tuple[int, int]
    dynamic_range: float
    luminance_std: float
    edge_energy: float
    shadow_clip_ratio: float
    highlight_clip_ratio: float
    nearest_perceptual_distance: int | None
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "kind": "text_to_image_quality_screen",
            "outputSize": {"width": self.output_size[0], "height": self.output_size[1]},
            "expectedSize": {"width": self.expected_size[0], "height": self.expected_size[1]},
            "metrics": {
                "dynamicRange": round(self.dynamic_range, 6),
                "luminanceStandardDeviation": round(self.luminance_std, 6),
                "edgeEnergy": round(self.edge_energy, 6),
                "shadowClipRatio": round(self.shadow_clip_ratio, 6),
                "highlightClipRatio": round(self.highlight_clip_ratio, 6),
                "nearestPerceptualHashDistance": self.nearest_perceptual_distance,
            },
            "checks": self.checks,
            "automatedStatus": "pass" if self.passed else "reject",
            "approvalStatus": "human_review_required" if self.passed else "not_eligible",
            "policy": (
                "This screen rejects only objective technical defects and near-duplicates; it cannot approve aesthetics. "
                "Prompt fidelity, artifacts, rights, and final approval require human review."
            ),
        }


@dataclass(frozen=True)
class ImageEditAssessment:
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    expected_size: tuple[int, int]
    dynamic_range: float
    luminance_std: float
    edge_energy: float
    shadow_clip_ratio: float
    highlight_clip_ratio: float
    source_output_mae: float
    localized_change_ratio: float
    source_luminance_correlation: float
    source_aligned_luminance_correlation: float
    alignment_overlap_ratio: float
    alignment_inlier_count: int
    alignment_inlier_ratio: float
    source_perceptual_distance: int
    nearest_gallery_perceptual_distance: int | None
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "kind": "image_edit_quality_screen",
            "sourceSize": {"width": self.source_size[0], "height": self.source_size[1]},
            "outputSize": {"width": self.output_size[0], "height": self.output_size[1]},
            "expectedSize": {"width": self.expected_size[0], "height": self.expected_size[1]},
            "metrics": {
                "dynamicRange": round(self.dynamic_range, 6),
                "luminanceStandardDeviation": round(self.luminance_std, 6),
                "edgeEnergy": round(self.edge_energy, 6),
                "shadowClipRatio": round(self.shadow_clip_ratio, 6),
                "highlightClipRatio": round(self.highlight_clip_ratio, 6),
                "sourceOutputMae": round(self.source_output_mae, 6),
                "localizedChangeRatio": round(self.localized_change_ratio, 6),
                "sourceLuminanceCorrelation": round(self.source_luminance_correlation, 6),
                "sourceAlignedLuminanceCorrelation": round(self.source_aligned_luminance_correlation, 6),
                "alignmentOverlapRatio": round(self.alignment_overlap_ratio, 6),
                "alignmentInlierCount": self.alignment_inlier_count,
                "alignmentInlierRatio": round(self.alignment_inlier_ratio, 6),
                "sourcePerceptualHashDistance": self.source_perceptual_distance,
                "nearestGalleryPerceptualHashDistance": self.nearest_gallery_perceptual_distance,
            },
            "checks": self.checks,
            "automatedStatus": "pass" if self.passed else "reject",
            "approvalStatus": "human_review_required" if self.passed else "not_eligible",
            "policy": (
                "This screen rejects blank, undersized, unchanged, composition-destroying, and duplicate edits; "
                "it cannot approve edit fidelity or aesthetics. Inspect the native before/after pair."
            ),
        }


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _edge_energy(image: Image.Image) -> float:
    values = _rgb_array(image)
    luminance = values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722
    horizontal = np.abs(np.diff(luminance, axis=1)).mean()
    vertical = np.abs(np.diff(luminance, axis=0)).mean()
    return float(horizontal + vertical)


def _difference_hash(image: Image.Image, *, hash_size: int = 8) -> int:
    reduced = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    values = np.asarray(reduced, dtype=np.int16)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for value in bits.flat:
        result = (result << 1) | int(value)
    return result


def _homography_aligned_correlation(
    source_luminance: np.ndarray,
    output_luminance: np.ndarray,
) -> tuple[float, float, int, float]:
    """Measure retained structure after an evidence-backed viewpoint transform.

    Raw pixel correlation is deliberately retained as the primary cheap signal.
    This secondary path is valid only when ORB feature matches produce a
    RANSAC-supported homography with material overlap. OpenCV is an app-owned
    optional Gallery-media dependency, so callers without it safely receive no
    alignment evidence instead of a weakened check.
    """

    try:
        import cv2
    except ImportError:
        return 0.0, 0.0, 0, 0.0

    source_u8 = np.clip(source_luminance * 255.0, 0, 255).astype(np.uint8)
    output_u8 = np.clip(output_luminance * 255.0, 0, 255).astype(np.uint8)
    cv2.setRNGSeed(0)
    detector = cv2.ORB_create(nfeatures=4000, fastThreshold=10)
    source_points, source_descriptors = detector.detectAndCompute(source_u8, None)
    output_points, output_descriptors = detector.detectAndCompute(output_u8, None)
    if source_descriptors is None or output_descriptors is None:
        return 0.0, 0.0, 0, 0.0

    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(source_descriptors, output_descriptors, k=2)
    matches = [
        nearest
        for pair in pairs
        if len(pair) == 2
        for nearest, alternative in [pair]
        if nearest.distance < 0.75 * alternative.distance
    ]
    if len(matches) < 12:
        return 0.0, 0.0, 0, 0.0

    source_matches = np.float32([source_points[item.queryIdx].pt for item in matches]).reshape(-1, 1, 2)
    output_matches = np.float32([output_points[item.trainIdx].pt for item in matches]).reshape(-1, 1, 2)
    transform, inlier_mask = cv2.findHomography(source_matches, output_matches, cv2.RANSAC, 4.0)
    if transform is None or inlier_mask is None:
        return 0.0, 0.0, 0, 0.0
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count / len(matches))
    if inlier_count < 12 or inlier_ratio < 0.25:
        return 0.0, 0.0, inlier_count, inlier_ratio

    output_size = (output_luminance.shape[1], output_luminance.shape[0])
    warped_source = cv2.warpPerspective(source_luminance.astype(np.float32), transform, output_size)
    overlap_mask = cv2.warpPerspective(np.ones_like(source_u8), transform, output_size) > 0
    overlap_ratio = float(overlap_mask.mean())
    if overlap_ratio < 0.35:
        return 0.0, overlap_ratio, inlier_count, inlier_ratio
    aligned_source = warped_source[overlap_mask]
    aligned_output = output_luminance[overlap_mask]
    if aligned_source.std() <= 1e-8 or aligned_output.std() <= 1e-8:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(aligned_source, aligned_output)[0, 1])
    return correlation, overlap_ratio, inlier_count, inlier_ratio


def assess_text_to_image(
    output: Image.Image,
    *,
    expected_size: tuple[int, int] = (1024, 1024),
    comparison_images: Iterable[Image.Image] = (),
) -> TextToImageAssessment:
    """Reject objective T2I defects without pretending to automate aesthetic approval."""

    output = output.convert("RGB")
    output_values = _rgb_array(output)
    luminance = output_values[..., 0] * 0.2126 + output_values[..., 1] * 0.7152 + output_values[..., 2] * 0.0722
    dynamic_range = float(np.quantile(output_values, 0.95) - np.quantile(output_values, 0.05))
    luminance_std = float(luminance.std())
    edge_energy = _edge_energy(output)
    shadow_clip_ratio = float((luminance <= 0.01).mean())
    highlight_clip_ratio = float((luminance >= 0.99).mean())
    output_hash = _difference_hash(output)
    distances = [
        (output_hash ^ _difference_hash(candidate.convert("RGB"))).bit_count()
        for candidate in comparison_images
    ]
    nearest_distance = min(distances) if distances else None
    checks = {
        "nativeShowcaseDimensions": output.size == expected_size,
        # Portrait and landscape model contracts may have one native 768px
        # side (for example FLUX LoRA's official 768x1024 recipe). This still
        # rejects the tiny 32/64/512px technical canaries that cannot carry a
        # Gallery card without being mislabeled as native showcase media.
        "cardReadableDimensions": min(output.size) >= 768,
        "nonBlankDynamicRange": dynamic_range >= 0.30,
        "meaningfulTonalStructure": luminance_std >= 0.08,
        "meaningfulSpatialDetail": edge_energy >= 0.008,
        "noExtremeOversharpening": edge_energy <= 0.20,
        "noCrushedShadows": shadow_clip_ratio <= 0.25,
        "noBlownHighlights": highlight_clip_ratio <= 0.25,
        "notPerceptualDuplicate": nearest_distance is None or nearest_distance >= 10,
    }
    return TextToImageAssessment(
        output_size=output.size,
        expected_size=expected_size,
        dynamic_range=dynamic_range,
        luminance_std=luminance_std,
        edge_energy=edge_energy,
        shadow_clip_ratio=shadow_clip_ratio,
        highlight_clip_ratio=highlight_clip_ratio,
        nearest_perceptual_distance=nearest_distance,
        checks=checks,
    )


def assess_image_edit(
    source: Image.Image,
    output: Image.Image,
    *,
    expected_size: tuple[int, int],
    comparison_images: Iterable[Image.Image] = (),
) -> ImageEditAssessment:
    """Reject objective edit failures while leaving semantic/aesthetic approval to review."""

    source = source.convert("RGB")
    output = output.convert("RGB")
    source_aligned = source.resize(output.size, Image.Resampling.LANCZOS)
    source_values = _rgb_array(source_aligned)
    output_values = _rgb_array(output)
    source_luminance = (
        source_values[..., 0] * 0.2126 + source_values[..., 1] * 0.7152 + source_values[..., 2] * 0.0722
    )
    output_luminance = (
        output_values[..., 0] * 0.2126 + output_values[..., 1] * 0.7152 + output_values[..., 2] * 0.0722
    )
    source_output_mae = float(np.abs(output_values - source_values).mean())
    # Global MAE and perceptual hashes are intentionally insensitive to a
    # small, precise edit. Retain a high-delta pixel ratio so an otherwise
    # preserved image can prove a localized material/color change without
    # weakening the unchanged-image rejection.
    per_pixel_mae = np.abs(output_values - source_values).mean(axis=2)
    localized_change_ratio = float((per_pixel_mae >= 0.20).mean())
    if source_luminance.std() <= 1e-8 or output_luminance.std() <= 1e-8:
        luminance_correlation = 0.0
    else:
        luminance_correlation = float(np.corrcoef(source_luminance.ravel(), output_luminance.ravel())[0, 1])
    aligned_correlation, alignment_overlap, alignment_inliers, alignment_inlier_ratio = (
        _homography_aligned_correlation(source_luminance, output_luminance)
    )
    dynamic_range = float(np.quantile(output_values, 0.95) - np.quantile(output_values, 0.05))
    luminance_std = float(output_luminance.std())
    edge_energy = _edge_energy(output)
    shadow_clip_ratio = float((output_luminance <= 0.01).mean())
    highlight_clip_ratio = float((output_luminance >= 0.99).mean())
    output_hash = _difference_hash(output)
    source_distance = (output_hash ^ _difference_hash(source_aligned)).bit_count()
    gallery_distances = [
        (output_hash ^ _difference_hash(candidate.convert("RGB"))).bit_count()
        for candidate in comparison_images
    ]
    nearest_gallery_distance = min(gallery_distances) if gallery_distances else None
    checks = {
        "nativeShowcaseDimensions": output.size == expected_size,
        "cardReadableDimensions": output.width >= 1024 and output.height >= 512,
        "nonBlankDynamicRange": dynamic_range >= 0.30,
        "meaningfulTonalStructure": luminance_std >= 0.08,
        "meaningfulSpatialDetail": edge_energy >= 0.008,
        "noExtremeOversharpening": edge_energy <= 0.20,
        "noCrushedShadows": shadow_clip_ratio <= 0.25,
        "noBlownHighlights": highlight_clip_ratio <= 0.05,
        "meaningfulEdit": (source_output_mae >= 0.04 and source_distance >= 6)
        or (
            source_output_mae >= 0.02
            and localized_change_ratio >= 0.01
            and (
                luminance_correlation >= 0.75
                or (
                    aligned_correlation >= 0.75
                    and alignment_overlap >= 0.75
                    and alignment_inliers >= 24
                    and alignment_inlier_ratio >= 0.50
                )
            )
        ),
        "sourceCompositionRetained": source_output_mae <= 0.50
        and (
            luminance_correlation >= 0.30
            or (
                aligned_correlation >= 0.35
                and alignment_overlap >= 0.35
                and alignment_inliers >= 12
                and alignment_inlier_ratio >= 0.25
            )
        ),
        "notGalleryDuplicate": nearest_gallery_distance is None or nearest_gallery_distance >= 10,
    }
    return ImageEditAssessment(
        source_size=source.size,
        output_size=output.size,
        expected_size=expected_size,
        dynamic_range=dynamic_range,
        luminance_std=luminance_std,
        edge_energy=edge_energy,
        shadow_clip_ratio=shadow_clip_ratio,
        highlight_clip_ratio=highlight_clip_ratio,
        source_output_mae=source_output_mae,
        localized_change_ratio=localized_change_ratio,
        source_luminance_correlation=luminance_correlation,
        source_aligned_luminance_correlation=aligned_correlation,
        alignment_overlap_ratio=alignment_overlap,
        alignment_inlier_count=alignment_inliers,
        alignment_inlier_ratio=alignment_inlier_ratio,
        source_perceptual_distance=source_distance,
        nearest_gallery_perceptual_distance=nearest_gallery_distance,
        checks=checks,
    )


def assess_image_upscale(source: Image.Image, output: Image.Image, *, expected_scale: int = 2) -> ImageUpscaleAssessment:
    source = source.convert("RGB")
    output = output.convert("RGB")
    expected_size = (source.width * expected_scale, source.height * expected_scale)
    baseline = source.resize(output.size, Image.Resampling.BICUBIC)
    reconstructed = output.resize(source.size, Image.Resampling.LANCZOS)
    source_values = _rgb_array(source)
    output_values = _rgb_array(output)
    reconstruction_mae = float(np.abs(_rgb_array(reconstructed) - source_values).mean())
    dynamic_range = float(np.quantile(output_values, 0.95) - np.quantile(output_values, 0.05))
    baseline_detail = max(_edge_energy(baseline), 1e-8)
    detail_ratio = _edge_energy(output) / baseline_detail
    checks = {
        "nativeScale": output.size == expected_size,
        "cardReadableDimensions": output.width >= 1024 and output.height >= 512,
        "nonBlankDynamicRange": dynamic_range >= 0.25,
        "sourceContentPreserved": reconstruction_mae <= 0.12,
        "meaningfulDetailChange": detail_ratio >= 1.01,
        "noExtremeOversharpening": detail_ratio <= 2.5,
    }
    return ImageUpscaleAssessment(
        source_size=source.size,
        output_size=output.size,
        expected_scale=expected_scale,
        dynamic_range=dynamic_range,
        reconstruction_mae=reconstruction_mae,
        detail_ratio=detail_ratio,
        checks=checks,
    )


def _video_luminance(frames: np.ndarray) -> np.ndarray:
    values = frames.astype(np.float32) / 255.0
    return values[..., 0] * 0.2126 + values[..., 1] * 0.7152 + values[..., 2] * 0.0722


def assess_video_upscale(
    source_frames: np.ndarray,
    output_frames: np.ndarray,
    *,
    source_fps: float,
    output_fps: float,
    expected_scale: int = 2,
) -> VideoUpscaleAssessment:
    if source_frames.ndim != 4 or output_frames.ndim != 4 or source_frames.shape[-1] < 3 or output_frames.shape[-1] < 3:
        raise ValueError("Video quality screening requires frame arrays shaped N x H x W x RGB.")
    if len(source_frames) == 0 or len(output_frames) == 0 or source_fps <= 0 or output_fps <= 0:
        raise ValueError("Video quality screening requires non-empty clips with positive frame rates.")
    source_frames = source_frames[..., :3].astype(np.uint8)
    output_frames = output_frames[..., :3].astype(np.uint8)
    source_height, source_width = source_frames.shape[1:3]
    output_height, output_width = output_frames.shape[1:3]
    source_luminance = _video_luminance(source_frames)
    if len(source_frames) > 1:
        temporal_delta = np.abs(np.diff(source_luminance, axis=0))
        temporal_mae = float(temporal_delta.mean())
        changed_pixel_ratio = float((temporal_delta > (8.0 / 255.0)).mean())
    else:
        temporal_mae = 0.0
        changed_pixel_ratio = 0.0

    paired_count = min(len(source_frames), len(output_frames))
    sample_indices = sorted(set(np.linspace(0, paired_count - 1, min(12, paired_count), dtype=int)))
    reconstruction_errors = []
    detail_ratios = []
    for index in sample_indices:
        source_image = Image.fromarray(source_frames[index], "RGB")
        output_image = Image.fromarray(output_frames[index], "RGB")
        reconstructed = output_image.resize(source_image.size, Image.Resampling.LANCZOS)
        reconstruction_errors.append(float(np.abs(_rgb_array(reconstructed) - _rgb_array(source_image)).mean()))
        baseline = source_image.resize(output_image.size, Image.Resampling.BICUBIC)
        detail_ratios.append(_edge_energy(output_image) / max(_edge_energy(baseline), 1e-8))
    reconstruction_mae = float(np.mean(reconstruction_errors))
    detail_ratio = float(np.mean(detail_ratios))
    source_duration = len(source_frames) / source_fps
    output_duration = len(output_frames) / output_fps
    checks = {
        "frameCountPreserved": len(output_frames) == len(source_frames),
        "frameRatePreserved": abs(output_fps - source_fps) <= 0.01,
        "durationPreserved": abs(output_duration - source_duration) <= max(0.05, 1.0 / source_fps),
        "nativeScale": (output_width, output_height)
        == (source_width * expected_scale, source_height * expected_scale),
        "cardReadableDimensions": output_width >= 1024 and output_height >= 512,
        "showcaseDuration": source_duration >= 3.0 and len(source_frames) >= 24,
        "meaningfulTemporalChange": temporal_mae >= 0.005 and changed_pixel_ratio >= 0.03,
        "sourceContentPreserved": reconstruction_mae <= 0.12,
        "meaningfulDetailChange": detail_ratio >= 1.01,
        "noExtremeOversharpening": detail_ratio <= 2.5,
    }
    return VideoUpscaleAssessment(
        source_size=(source_width, source_height),
        output_size=(output_width, output_height),
        source_frames=len(source_frames),
        output_frames=len(output_frames),
        source_fps=source_fps,
        output_fps=output_fps,
        expected_scale=expected_scale,
        temporal_mae=temporal_mae,
        changed_pixel_ratio=changed_pixel_ratio,
        reconstruction_mae=reconstruction_mae,
        detail_ratio=detail_ratio,
        checks=checks,
    )


def _detail_crop_box(image: Image.Image, *, crop_width: int, crop_height: int) -> tuple[int, int, int, int]:
    luminance = np.asarray(image.convert("L"), dtype=np.float32)
    best_score = -1.0
    best = (0, 0, crop_width, crop_height)
    x_steps = sorted(set(np.linspace(0, max(0, image.width - crop_width), 7, dtype=int)))
    y_steps = sorted(set(np.linspace(0, max(0, image.height - crop_height), 5, dtype=int)))
    for top in y_steps:
        for left in x_steps:
            region = luminance[top : top + crop_height, left : left + crop_width]
            if region.size == 0:
                continue
            score = float(np.abs(np.diff(region, axis=1)).mean() + np.abs(np.diff(region, axis=0)).mean())
            if score > best_score:
                best_score = score
                best = (left, top, left + crop_width, top + crop_height)
    return best


def _fit(image: Image.Image, size: tuple[int, int], *, resample: Image.Resampling) -> Image.Image:
    contained = ImageOps.contain(image, size, method=resample)
    canvas = Image.new("RGB", size, "#090c0f")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def build_text_to_image_review_card(
    output: Image.Image,
    assessment: TextToImageAssessment,
    *,
    title: str,
    prompt: str,
) -> Image.Image:
    output = output.convert("RGB")
    card = Image.new("RGB", (1800, 1100), "#12171c")
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default(size=30)
    small = ImageFont.load_default(size=22)
    draw.text((60, 34), title, fill="#f5f7f8", font=font)
    draw.text(
        (60, 76),
        "Automated screen: " + ("PASS, HUMAN REVIEW REQUIRED" if assessment.passed else "REJECT"),
        fill="#84d8a7" if assessment.passed else "#ff8a8a",
        font=small,
    )
    panel = _fit(output, (980, 900), resample=Image.Resampling.LANCZOS)
    card.paste(panel, (60, 130))
    metrics = assessment.to_dict()["metrics"]
    lines = [
        f"Native output: {output.width} x {output.height}",
        f"Dynamic range: {metrics['dynamicRange']}",
        f"Luminance std: {metrics['luminanceStandardDeviation']}",
        f"Edge energy: {metrics['edgeEnergy']}",
        f"Nearest dHash distance: {metrics['nearestPerceptualHashDistance']}",
        "",
        "Automated checks cannot approve aesthetics",
        "or prompt fidelity. Inspect the native file.",
        "",
        "PROMPT",
    ]
    y = 150
    for line in lines:
        draw.text((1090, y), line, fill="#c9d1d9", font=small)
        y += 38
    words = prompt.split()
    wrapped: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > 50 and current:
            wrapped.append(current)
            current = word
        else:
            current = candidate
    if current:
        wrapped.append(current)
    for line in wrapped[:13]:
        draw.text((1090, y), line, fill="#f5f7f8", font=small)
        y += 32
    return card


def build_image_edit_review_card(
    source: Image.Image,
    output: Image.Image,
    assessment: ImageEditAssessment,
    *,
    title: str,
    prompt: str,
) -> Image.Image:
    source = source.convert("RGB")
    output = output.convert("RGB")
    card = Image.new("RGB", (1800, 1100), "#12171c")
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default(size=30)
    small = ImageFont.load_default(size=22)
    draw.text((60, 34), title, fill="#f5f7f8", font=font)
    draw.text(
        (60, 76),
        "Matched composition · automated screen: "
        + ("PASS, HUMAN REVIEW REQUIRED" if assessment.passed else "REJECT"),
        fill="#84d8a7" if assessment.passed else "#ff8a8a",
        font=small,
    )
    panel_size = (820, 560)
    card.paste(_fit(source, panel_size, resample=Image.Resampling.LANCZOS), (60, 140))
    card.paste(_fit(output, panel_size, resample=Image.Resampling.LANCZOS), (920, 140))
    draw.text((60, 715), f"BEFORE · {source.width} x {source.height}", fill="#c9d1d9", font=small)
    draw.text((920, 715), f"AFTER · {output.width} x {output.height}", fill="#c9d1d9", font=small)
    metrics = assessment.to_dict()["metrics"]
    draw.text(
        (60, 765),
        f"Edit MAE {metrics['sourceOutputMae']} · structure correlation "
        f"{metrics['sourceLuminanceCorrelation']} · source dHash distance "
        f"{metrics['sourcePerceptualHashDistance']}",
        fill="#c9d1d9",
        font=small,
    )
    draw.text((60, 820), "PROMPT", fill="#f5f7f8", font=small)
    words = prompt.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > 118 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    y = 858
    for line in lines[:6]:
        draw.text((60, y), line, fill="#f5f7f8", font=small)
        y += 34
    return card


def build_image_upscale_review_card(
    source: Image.Image,
    output: Image.Image,
    assessment: ImageUpscaleAssessment,
) -> Image.Image:
    source = source.convert("RGB")
    output = output.convert("RGB")
    card = Image.new("RGB", (1800, 1100), "#12171c")
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default(size=30)
    small = ImageFont.load_default(size=22)
    draw.text((60, 34), "REAL-ESRGAN x2 · IMAGE UPSCALE", fill="#f5f7f8", font=font)
    draw.text(
        (60, 76),
        "Same frame and matched detail crop · automated screen: "
        + ("PASS, HUMAN REVIEW REQUIRED" if assessment.passed else "REJECT"),
        fill="#84d8a7" if assessment.passed else "#ff8a8a",
        font=small,
    )
    panel_size = (820, 410)
    card.paste(_fit(source, panel_size, resample=Image.Resampling.NEAREST), (60, 130))
    card.paste(_fit(output, panel_size, resample=Image.Resampling.LANCZOS), (920, 130))
    draw.text((60, 552), f"BEFORE · {source.width} x {source.height}", fill="#c9d1d9", font=small)
    draw.text((920, 552), f"AFTER · {output.width} x {output.height}", fill="#c9d1d9", font=small)

    crop_width = min(192, source.width)
    crop_height = min(128, source.height)
    source_box = _detail_crop_box(source, crop_width=crop_width, crop_height=crop_height)
    output_box = tuple(coordinate * assessment.expected_scale for coordinate in source_box)
    source_crop = source.crop(source_box)
    output_crop = output.crop(output_box)
    detail_size = (700, 390)
    card.paste(_fit(source_crop, detail_size, resample=Image.Resampling.NEAREST), (130, 640))
    card.paste(_fit(output_crop, detail_size, resample=Image.Resampling.LANCZOS), (970, 640))
    draw.rectangle((125, 635, 835, 1035), outline="#68737d", width=3)
    draw.rectangle((965, 635, 1675, 1035), outline="#68737d", width=3)
    draw.text((130, 1042), "BEFORE DETAIL", fill="#c9d1d9", font=small)
    draw.text((970, 1042), "AFTER DETAIL", fill="#c9d1d9", font=small)
    return card


def build_video_upscale_review_card(
    source_frames: np.ndarray,
    output_frames: np.ndarray,
    assessment: VideoUpscaleAssessment,
) -> Image.Image:
    card = Image.new("RGB", (1800, 1100), "#12171c")
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default(size=30)
    small = ImageFont.load_default(size=22)
    draw.text((60, 34), "REAL-ESRGAN x2 · VIDEO UPSCALE", fill="#f5f7f8", font=font)
    draw.text(
        (60, 76),
        "Matched timepoints · automated screen: "
        + ("PASS, HUMAN REVIEW REQUIRED" if assessment.passed else "REJECT"),
        fill="#84d8a7" if assessment.passed else "#ff8a8a",
        font=small,
    )
    indices = sorted(set(np.linspace(0, min(len(source_frames), len(output_frames)) - 1, 4, dtype=int)))
    panel_size = (390, 360)
    x_positions = (60, 490, 920, 1350)
    for column, index in enumerate(indices):
        source_image = Image.fromarray(source_frames[index][..., :3].astype(np.uint8), "RGB")
        output_image = Image.fromarray(output_frames[index][..., :3].astype(np.uint8), "RGB")
        x = x_positions[column]
        card.paste(_fit(source_image, panel_size, resample=Image.Resampling.NEAREST), (x, 150))
        card.paste(_fit(output_image, panel_size, resample=Image.Resampling.LANCZOS), (x, 630))
        time_seconds = index / assessment.source_fps
        draw.text((x, 520), f"BEFORE · t={time_seconds:.2f}s", fill="#c9d1d9", font=small)
        draw.text((x, 1000), f"AFTER · t={time_seconds:.2f}s", fill="#c9d1d9", font=small)
    draw.text(
        (60, 1060),
        f"{assessment.source_frames} frames · {assessment.source_fps:g} fps · "
        f"{assessment.source_size[0]}x{assessment.source_size[1]} to "
        f"{assessment.output_size[0]}x{assessment.output_size[1]}",
        fill="#c9d1d9",
        font=small,
    )
    return card
