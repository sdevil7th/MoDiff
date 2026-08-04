import unittest

import numpy as np
from PIL import Image, ImageDraw

from modules.VideoConditioning import main as video_conditioning
from modules.VideoConditioning.main import EdgePreprocessor, ObjectMaskPropagate


class VideoConditioningPreprocessorTests(unittest.TestCase):
    def test_video_conditioning_has_no_independent_model_preprocessors(self):
        self.assertFalse(hasattr(video_conditioning, "DepthPreprocessor"))
        self.assertFalse(hasattr(video_conditioning, "PosePreprocessor"))

    def test_edge_preprocessor_preserves_shape_and_frame_count(self):
        source = Image.new("RGB", (16, 12), "black")
        ImageDraw.Draw(source).rectangle((4, 3, 11, 8), fill="white")
        output = EdgePreprocessor().execute(video=[source, source], algorithm="canny", low_threshold=50, high_threshold=100)["output"]

        self.assertEqual(len(output), 2)
        self.assertEqual(output[0].size, source.size)
        self.assertGreater(np.asarray(output[0]).max(), 0)

    def test_static_video_keeps_mask_and_reports_high_confidence(self):
        frame = Image.new("RGB", (16, 12), "gray")
        mask = Image.new("L", frame.size, 0)
        ImageDraw.Draw(mask).rectangle((4, 3, 10, 8), fill=255)
        result = ObjectMaskPropagate().execute(video=[frame, frame, frame], first_mask=mask, threshold=127, smooth_pixels=0)

        self.assertEqual(len(result["masks"]), 3)
        self.assertTrue(all(score > 0.99 for score in result["confidence"]))
        self.assertEqual(np.asarray(result["masks"][0]).sum(), np.asarray(result["masks"][-1]).sum())


if __name__ == "__main__":
    unittest.main()
