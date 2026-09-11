"""CPU-only edge format contract; no model downloads or inference."""
import unittest

import numpy as np
from PIL import Image, ImageDraw

from modules.ImageFilters import MODULE_MAP
from modules.ImageFilters.main import Canny


class CannyOutputModeTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 32), "black")
        ImageDraw.Draw(self.image).rectangle((8, 8, 24, 24), fill="white")

    def execute(self, **params):
        return Canny("canny-output-test").execute(
            image=params.pop("image", self.image), device="cpu", **params
        )["output"]

    def test_default_and_legacy_graph_keep_grayscale_output(self):
        self.assertEqual(MODULE_MAP["Canny"]["params"]["output_mode"]["default"], "L")
        old = self.execute()[0]
        explicit = self.execute(output_mode="L")[0]
        self.assertEqual(old.mode, "L")
        np.testing.assert_array_equal(np.array(old), np.array(explicit))
        self.assertGreater(np.count_nonzero(np.array(old)), 0)

    def test_rgb_repeats_identical_edges_without_changing_pixels(self):
        gray = np.array(self.execute()[0])
        rgb = self.execute(output_mode="RGB")[0]
        self.assertEqual(rgb.mode, "RGB")
        self.assertEqual(rgb.size, self.image.size)
        for channel in np.moveaxis(np.array(rgb), -1, 0):
            np.testing.assert_array_equal(channel, gray)

    def test_batches_retain_count_and_rgb_format(self):
        outputs = self.execute(image=[self.image, self.image], output_mode="RGB")
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(image.mode == "RGB" for image in outputs))

    def test_unknown_format_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Output Mode.*L.*RGB"):
            self.execute(output_mode="RGBA")


if __name__ == "__main__":
    unittest.main()
