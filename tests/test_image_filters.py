import unittest

import numpy as np
from PIL import Image, ImageDraw

from modules.ImageFilters.main import Canny
from utils.torch_utils import CPU_DEVICE


class ImageFilterTests(unittest.TestCase):
    def test_canny_reversed_thresholds_preserve_the_same_edge_map(self):
        pixels = np.random.default_rng(42).integers(0, 256, (32, 32, 3), dtype=np.uint8)
        source = Image.fromarray(pixels)
        expected = Canny().execute(
            image=source, low_threshold=0.1, high_threshold=0.7, device=CPU_DEVICE
        )["output"][0]
        node = Canny("canny-threshold-order")
        inputs = dict(image=source, low_threshold=0.7, high_threshold=0.1, device=CPU_DEVICE)
        reversed_thresholds = node(**inputs)["output"][0]
        self.assertEqual(reversed_thresholds.tobytes(), expected.tobytes())
        self.assertTrue(node._has_changed)
        self.assertEqual(node(**inputs)["output"][0].tobytes(), expected.tobytes())
        self.assertFalse(node._has_changed)

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
