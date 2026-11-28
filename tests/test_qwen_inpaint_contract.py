import unittest

from PIL import Image

from modules.DiffusersImage.main import (
    IMAGE_PIPELINE_ADAPTERS,
    Inpaint,
    LoadPipeline,
    OutpaintCanvas,
    composite_masked_pil_outputs,
)


class QwenInpaintContractTests(unittest.TestCase):
    def test_qwen_inpaint_is_a_generic_adapter_contract(self):
        adapter = IMAGE_PIPELINE_ADAPTERS["QwenImageEditInpaintPipeline"]
        self.assertEqual(adapter.guidance_parameter, "true_cfg_scale")
        self.assertEqual(adapter.modes, frozenset({"inpaint", "outpaint"}))
        self.assertEqual(Inpaint.params["output_type"]["options"], ["pil"])

    def test_loader_defaults_to_memory_bounded_vae_decode(self):
        self.assertTrue(LoadPipeline.params["enable_vae_slicing"]["default"])
        self.assertTrue(LoadPipeline.params["enable_vae_tiling"]["default"])

    def test_generated_pixels_are_exposed_only_inside_the_white_mask(self):
        source = Image.new("RGB", (4, 2), (10, 20, 30))
        generated = Image.new("RGB", (4, 2), (200, 210, 220))
        mask = Image.new("L", (4, 2), 0)
        for x in (2, 3):
            for y in (0, 1):
                mask.putpixel((x, y), 255)

        output = composite_masked_pil_outputs([generated], source, mask)[0]

        self.assertEqual(output.getpixel((0, 0)), (10, 20, 30))
        self.assertEqual(output.getpixel((1, 1)), (10, 20, 30))
        self.assertEqual(output.getpixel((2, 0)), (200, 210, 220))
        self.assertEqual(output.getpixel((3, 1)), (200, 210, 220))

    def test_mask_and_source_are_normalized_to_generated_output_size(self):
        source = Image.new("RGB", (2, 2), (10, 20, 30))
        generated = Image.new("RGB", (4, 4), (200, 210, 220))
        mask = Image.new("L", (2, 2), 255)

        output = composite_masked_pil_outputs(generated, source, mask)[0]

        self.assertEqual(output.size, (4, 4))
        self.assertEqual(output.getpixel((0, 0)), (200, 210, 220))

    def test_outpaint_canvas_is_model_neutral(self):
        result = OutpaintCanvas("canvas-test").execute(
            image=Image.new("RGB", (32, 32), "white"),
            width=64,
            height=64,
            left=16,
            right=16,
            top=16,
            bottom=16,
            overlap=0,
            feather=0,
        )

        self.assertEqual(result["canvas"].size, (64, 64))
        self.assertEqual(result["mask_image"].getpixel((0, 0)), 255)
        self.assertEqual(result["mask_image"].getpixel((32, 16)), 0)

    def test_outpaint_seam_overlap_only_reaches_expanded_edges(self):
        source = Image.new("RGB", (96, 96), (30, 90, 160))
        for width, height, left, top in ((128, 96, 32, 0), (96, 128, 0, 32)):
            with self.subTest(left=left, top=top):
                result = OutpaintCanvas("canvas-test").execute(
                    image=source, width=width, height=height,
                    left=left, right=0, top=top, bottom=0, overlap=16, feather=4,
                )
                mask = result["mask_image"]
                self.assertEqual(mask.getpixel((0, 0)), 255)
                self.assertGreater(mask.getpixel((left + 2, top + 2)), 0)
                self.assertEqual(mask.getpixel((width - 1, height - 1)), 0)
                unexpanded_edge = (left + 48, 0) if left else (0, top + 48)
                self.assertEqual(mask.getpixel(unexpanded_edge), 0)
                generated = Image.new("RGB", (width, height), "white")
                output = composite_masked_pil_outputs(generated, result["canvas"], mask)[0]
                self.assertEqual(output.getpixel(unexpanded_edge), source.getpixel((0, 0)))

    def test_outpaint_mask_includes_a_one_pixel_expansion(self):
        result = OutpaintCanvas("canvas-test").execute(
            image=Image.new("RGB", (63, 64), "white"), width=64, height=64,
            left=0, right=1, top=0, bottom=0, overlap=0, feather=0,
        )
        mask = result["mask_image"]
        self.assertEqual(mask.getpixel((62, 32)), 0)
        self.assertEqual(mask.getpixel((63, 32)), 255)


if __name__ == "__main__":
    unittest.main()
