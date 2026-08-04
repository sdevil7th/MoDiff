import unittest

from PIL import Image

from modules.VideoConditioning.main import AlignMask, ReferenceImages


class AlignVideoMaskTests(unittest.TestCase):
    def test_grow_pixels_expands_white_generated_region(self):
        source = Image.new("RGB", (9, 9), (10, 20, 30))
        mask = Image.new("L", (9, 9), 0)
        mask.putpixel((4, 4), 255)

        result = AlignMask().execute(video=[source], mask=[mask], threshold=127, grow_pixels=2)["output"][0]

        self.assertEqual(result.getpixel((2, 2)), (255, 255, 255))
        self.assertEqual(result.getpixel((6, 6)), (255, 255, 255))
        self.assertEqual(result.getpixel((1, 1)), (0, 0, 0))

    def test_default_does_not_expand_mask(self):
        source = Image.new("RGB", (5, 5), 0)
        mask = Image.new("L", (5, 5), 0)
        mask.putpixel((2, 2), 255)

        result = AlignMask().execute(video=[source], mask=[mask], threshold=127)["output"][0]

        self.assertEqual(result.getpixel((2, 2)), (255, 255, 255))
        self.assertEqual(result.getpixel((1, 2)), (0, 0, 0))


class VideoReferenceImagesTests(unittest.TestCase):
    def test_packages_a_single_image_without_model_specific_translation(self):
        image = Image.new("RGB", (8, 8), (10, 20, 30))

        self.assertEqual(ReferenceImages().execute(images=image)["references"], [image])

    def test_preserves_an_ordered_reference_list(self):
        images = [Image.new("RGB", (8, 8), color) for color in ((10, 20, 30), (40, 50, 60))]

        self.assertEqual(ReferenceImages().execute(images=images)["references"], images)


if __name__ == "__main__":
    unittest.main()
