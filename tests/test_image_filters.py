import unittest

import numpy as np
from PIL import Image, ImageDraw

from modules.ImageFilters import MODULE_MAP as FILTER_DEFINITIONS, main as filters
from modules.ImageFilters.main import Canny
from utils.torch_utils import CPU_DEVICE


class ImageFilterTests(unittest.TestCase):
    def test_every_published_filter_executes_with_its_creator_defaults(self):
        source = Image.new("RGB", (64, 48), "black")
        ImageDraw.Draw(source).rectangle((16, 12, 47, 35), fill="white")
        for action, definition in FILTER_DEFINITIONS.items():
            with self.subTest(action=action):
                values = {name: field["default"] for name, field in definition["params"].items() if "default" in field}
                values.update(image=source, device=CPU_DEVICE)
                result = getattr(filters, action)("defaults-" + action)(**values)["output"]
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0].size, source.size)

    def test_unsharp_integral_float_and_even_radius_keep_the_integer_recipe(self):
        source = Image.new("RGB", (64, 48), "black")
        ImageDraw.Draw(source).rectangle((16, 12, 47, 35), fill="white")
        expected = filters.UnsharpMask().execute(image=source, radius=5, device=CPU_DEVICE)["output"][0]
        for radius in (5.0, "5", 4.0):
            with self.subTest(radius=radius):
                result = filters.UnsharpMask("edited-unsharp")(
                    image=source, radius=radius, device=CPU_DEVICE
                )["output"][0]
                self.assertEqual(result.tobytes(), expected.tobytes())

    def test_unsharp_rejects_invalid_radius_before_building_a_kernel(self):
        source = Image.new("RGB", (64, 48), "black")
        for radius in (0, -1, 64, 3.5, float("nan"), float("inf"), "bad"):
            with self.subTest(radius=radius), self.assertRaisesRegex(RuntimeError, "radius must be a whole number between 1 and 63"):
                filters.UnsharpMask("invalid-unsharp")(image=source, radius=radius, device=CPU_DEVICE)

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
