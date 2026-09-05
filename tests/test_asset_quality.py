import unittest

import numpy as np
from PIL import Image

from modiff.asset_quality import (
    assess_image_edit,
    assess_image_upscale,
    assess_text_to_image,
    assess_video_upscale,
    build_image_edit_review_card,
    build_image_upscale_review_card,
    build_text_to_image_review_card,
    build_video_upscale_review_card,
)


class ImageUpscaleQualityTests(unittest.TestCase):
    def test_exact_two_x_detailed_output_passes_but_still_requires_human_review(self):
        yy, xx = np.mgrid[:256, :512]
        pattern = np.stack(((xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 7) % 256), axis=-1).astype("uint8")
        source = Image.fromarray(pattern, "RGB")
        output = source.resize((1024, 512), Image.Resampling.NEAREST)

        assessment = assess_image_upscale(source, output)

        self.assertTrue(assessment.passed)
        self.assertEqual(assessment.to_dict()["approvalStatus"], "human_review_required")
        self.assertEqual(build_image_upscale_review_card(source, output, assessment).size, (1800, 1100))

    def test_wrong_scale_and_blank_output_fail(self):
        source = Image.new("RGB", (512, 256), "white")
        output = Image.new("RGB", (512, 256), "white")

        assessment = assess_image_upscale(source, output)

        self.assertFalse(assessment.passed)
        self.assertFalse(assessment.checks["nativeScale"])
        self.assertFalse(assessment.checks["nonBlankDynamicRange"])

    def test_plain_bicubic_resize_is_not_misclassified_as_model_restoration(self):
        yy, xx = np.mgrid[:256, :512]
        pattern = np.stack(((xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 7) % 256), axis=-1).astype("uint8")
        source = Image.fromarray(pattern, "RGB")
        output = source.resize((1024, 512), Image.Resampling.BICUBIC)

        assessment = assess_image_upscale(source, output)

        self.assertFalse(assessment.passed)
        self.assertFalse(assessment.checks["meaningfulDetailChange"])


class VideoUpscaleQualityTests(unittest.TestCase):
    def test_motion_rich_exact_two_x_clip_passes_but_requires_human_review(self):
        frames = []
        yy, xx = np.mgrid[:256, :512]
        for index in range(24):
            frame = np.stack(
                (
                    (xx * 3 + index * 11) % 256,
                    (yy * 5 + index * 7) % 256,
                    ((xx + yy) * 7 + index * 13) % 256,
                ),
                axis=-1,
            ).astype("uint8")
            frames.append(frame)
        source = np.stack(frames)
        output = np.stack(
            [np.asarray(Image.fromarray(frame).resize((1024, 512), Image.Resampling.NEAREST)) for frame in source]
        )

        assessment = assess_video_upscale(source, output, source_fps=8, output_fps=8)

        self.assertTrue(assessment.passed)
        self.assertEqual(assessment.to_dict()["approvalStatus"], "human_review_required")
        self.assertEqual(build_video_upscale_review_card(source, output, assessment).size, (1800, 1100))

    def test_static_wrong_scale_short_clip_fails(self):
        source = np.zeros((8, 128, 128, 3), dtype="uint8")
        output = np.zeros((8, 128, 128, 3), dtype="uint8")

        assessment = assess_video_upscale(source, output, source_fps=8, output_fps=16)

        self.assertFalse(assessment.passed)
        self.assertFalse(assessment.checks["nativeScale"])
        self.assertFalse(assessment.checks["frameRatePreserved"])
        self.assertFalse(assessment.checks["meaningfulTemporalChange"])


class TextToImageQualityTests(unittest.TestCase):
    def test_native_detailed_image_passes_but_still_requires_human_review(self):
        yy, xx = np.mgrid[:1024, :1024]
        pattern = np.stack(((xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 7) % 256), axis=-1).astype("uint8")
        output = Image.fromarray(pattern, "RGB")

        assessment = assess_text_to_image(output)

        self.assertTrue(assessment.passed)
        self.assertEqual(assessment.to_dict()["approvalStatus"], "human_review_required")
        self.assertIn("cannot approve aesthetics", assessment.to_dict()["policy"])
        self.assertEqual(
            build_text_to_image_review_card(output, assessment, title="TEST", prompt="Detailed test prompt").size,
            (1800, 1100),
        )

    def test_small_blank_and_perceptual_duplicate_image_fails(self):
        output = Image.new("RGB", (512, 512), "white")

        assessment = assess_text_to_image(output, comparison_images=[output.copy()])

        self.assertFalse(assessment.passed)
        self.assertFalse(assessment.checks["nativeShowcaseDimensions"])
        self.assertFalse(assessment.checks["nonBlankDynamicRange"])
        self.assertFalse(assessment.checks["notPerceptualDuplicate"])

    def test_native_768_by_1024_portrait_is_card_readable(self):
        yy, xx = np.mgrid[:1024, :768]
        pattern = np.stack(((xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 7) % 256), axis=-1).astype("uint8")
        output = Image.fromarray(pattern, "RGB")

        assessment = assess_text_to_image(output, expected_size=(768, 1024))

        self.assertTrue(assessment.checks["nativeShowcaseDimensions"])
        self.assertTrue(assessment.checks["cardReadableDimensions"])


class ImageEditQualityTests(unittest.TestCase):
    @staticmethod
    def _source() -> Image.Image:
        yy, xx = np.mgrid[:512, :1024]
        pattern = np.stack(((xx * 3) % 256, (yy * 5) % 256, ((xx + yy) * 7) % 256), axis=-1).astype("uint8")
        return Image.fromarray(pattern, "RGB")

    def test_detailed_material_edit_passes_but_requires_human_review(self):
        source = self._source()
        values = np.asarray(source).astype(np.int16)
        output = Image.fromarray(np.clip(values[..., ::-1] * 0.85 + 22, 0, 255).astype("uint8"), "RGB")

        assessment = assess_image_edit(source, output, expected_size=(1024, 512))

        self.assertTrue(assessment.passed)
        self.assertEqual(assessment.to_dict()["approvalStatus"], "human_review_required")
        self.assertIn("cannot approve", assessment.to_dict()["policy"])
        self.assertEqual(
            build_image_edit_review_card(source, output, assessment, title="TEST EDIT", prompt="Change materials").size,
            (1800, 1100),
        )

    def test_unchanged_and_blank_edits_fail(self):
        source = self._source()
        unchanged = assess_image_edit(source, source.copy(), expected_size=(1024, 512))
        blank = assess_image_edit(source, Image.new("RGB", (512, 512), "white"), expected_size=(1024, 512))

        self.assertFalse(unchanged.passed)
        self.assertFalse(unchanged.checks["meaningfulEdit"])
        self.assertFalse(blank.passed)
        self.assertFalse(blank.checks["nativeShowcaseDimensions"])
        self.assertFalse(blank.checks["nonBlankDynamicRange"])

    def test_precise_local_color_edit_passes_meaningful_edit_gate(self):
        source = self._source()
        output_values = np.asarray(source).copy()
        output_values[170:310, 430:650] = np.asarray([224, 168, 32], dtype=np.uint8)
        output = Image.fromarray(output_values, "RGB")

        assessment = assess_image_edit(source, output, expected_size=(1024, 512))

        self.assertLess(assessment.source_output_mae, 0.04)
        self.assertGreaterEqual(assessment.localized_change_ratio, 0.01)
        self.assertGreaterEqual(assessment.source_luminance_correlation, 0.75)
        self.assertTrue(assessment.checks["meaningfulEdit"])
        self.assertTrue(assessment.checks["sourceCompositionRetained"])

    def test_weak_source_correlation_does_not_claim_composition_retention(self):
        source = self._source()
        source_values = np.asarray(source).astype(np.float32)
        noise = np.random.default_rng(7).integers(0, 256, source_values.shape).astype(np.float32)
        output = Image.fromarray(np.clip(source_values * 0.20 + noise * 0.80, 0, 255).astype("uint8"), "RGB")

        assessment = assess_image_edit(source, output, expected_size=(1024, 512))

        self.assertGreaterEqual(assessment.source_luminance_correlation, 0.20)
        self.assertLess(assessment.source_luminance_correlation, 0.30)
        self.assertFalse(assessment.checks["sourceCompositionRetained"])

    def test_feature_supported_viewpoint_change_retains_composition(self):
        import cv2

        rng = np.random.default_rng(41)
        source_values = rng.integers(0, 256, (512, 1024, 3), dtype=np.uint8)
        source_values = cv2.GaussianBlur(source_values, (0, 0), 1.2)
        for index in range(20):
            cv2.circle(
                source_values,
                (50 + index * 47, 80 + (index % 5) * 70),
                15 + (index % 4) * 5,
                (index * 11 % 255, index * 29 % 255, index * 53 % 255),
                -1,
            )
        source = Image.fromarray(source_values, "RGB")
        transform = np.asarray(
            [[0.92, -0.08, 68.0], [0.04, 0.88, 34.0], [0.00008, -0.00012, 1.0]],
            dtype=np.float32,
        )
        output_values = cv2.warpPerspective(
            source_values,
            transform,
            source.size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        output = Image.fromarray(output_values, "RGB")

        assessment = assess_image_edit(source, output, expected_size=source.size)

        self.assertLess(assessment.source_luminance_correlation, 0.30)
        self.assertGreaterEqual(assessment.source_aligned_luminance_correlation, 0.35)
        self.assertGreaterEqual(assessment.alignment_overlap_ratio, 0.35)
        self.assertGreaterEqual(assessment.alignment_inlier_count, 12)
        self.assertTrue(assessment.checks["sourceCompositionRetained"])

    def test_broad_clipped_highlights_fail_edit_screen(self):
        source = self._source()
        output_values = np.asarray(source).copy()
        output_values[:, :128] = 255
        output = Image.fromarray(output_values, "RGB")

        assessment = assess_image_edit(source, output, expected_size=(1024, 512))

        self.assertGreater(assessment.highlight_clip_ratio, 0.05)
        self.assertFalse(assessment.checks["noBlownHighlights"])


if __name__ == "__main__":
    unittest.main()
