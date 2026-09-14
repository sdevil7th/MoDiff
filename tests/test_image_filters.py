import unittest

import numpy as np
from PIL import Image, ImageDraw

from modules.ImageFilters.main import Canny


class ImageFilterTests(unittest.TestCase):
    def test_canny_preserves_source_extent_and_emits_edges(self):
        source = Image.new("RGB", (32, 24), "black")
        ImageDraw.Draw(source).rectangle((8, 6, 23, 17), fill="white")

        output = Canny().execute(
            image=source,
            low_threshold=0.1,
            high_threshold=0.2,
            device="cpu",
        )["output"]

        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].size, source.size)
        self.assertEqual(output[0].mode, "L")
        self.assertGreater(np.asarray(output[0]).max(), 0)


if __name__ == "__main__":
    unittest.main()
