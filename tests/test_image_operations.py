import unittest

from PIL import Image

from modules.ImageOperations.main import (
    FILTER_OPERATIONS,
    IMAGE_OPERATION_PIPELINE_CLASS,
    MAX_IMAGE_COUNT,
    AdjustImage,
    CropImage,
    FilterImage,
    ImageChannels,
    MaskComposite,
    ProcessImage,
    ResizeImage,
    StitchImages,
    TileImage,
)


class ImageOperationTests(unittest.TestCase):
    def test_adjust_is_deterministic_preserves_alpha_and_supports_collections(self):
        source = Image.new("RGBA", (3, 2), (100, 80, 60, 37))
        node = AdjustImage()
        first = node.execute(
            image=source,
            brightness=1.2,
            contrast=0.9,
            saturation=1.1,
            sharpness=1.5,
            gamma=1.2,
            temperature=0.25,
            tint=-0.1,
        )["output"]
        second = node.execute(
            image=source,
            brightness=1.2,
            contrast=0.9,
            saturation=1.1,
            sharpness=1.5,
            gamma=1.2,
            temperature=0.25,
            tint=-0.1,
        )["output"]

        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first.getchannel("A").getextrema(), (37, 37))
        collection = node.execute(image=[source, source])["output"]
        self.assertEqual(len(collection), 2)

    def test_adjust_rejects_nonfinite_or_out_of_range_values(self):
        source = Image.new("RGB", (2, 2), "white")
        for field, value in (("gamma", 0), ("brightness", float("nan")), ("tint", 2)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                AdjustImage().execute(image=source, **{field: value})

    def test_every_filter_is_deterministic_bounded_and_preserves_alpha(self):
        source = Image.new("RGBA", (7, 5), (90, 120, 150, 81))
        node = FilterImage()
        for operation in FILTER_OPERATIONS:
            with self.subTest(operation=operation):
                first = node.execute(
                    image=source,
                    operation=operation,
                    amount=2,
                    threshold=3,
                    seed=42,
                )["output"]
                second = node.execute(
                    image=source,
                    operation=operation,
                    amount=2,
                    threshold=3,
                    seed=42,
                )["output"]
                self.assertEqual(first.size, source.size)
                self.assertEqual(first.tobytes(), second.tobytes())
                self.assertEqual(first.getchannel("A").getextrema(), (81, 81))

        with self.assertRaisesRegex(ValueError, "Unsupported image filter"):
            node.execute(image=source, operation="external_plugin")

    def test_crop_clamps_to_edges_and_rejects_inconsistent_collections(self):
        source = Image.new("RGB", (10, 8), "red")
        result = CropImage().execute(image=source, x=7, y=6, width=20, height=20)
        self.assertEqual(result["output"].size, (3, 2))
        self.assertEqual((result["output_width"], result["output_height"]), (3, 2))

        with self.assertRaisesRegex(ValueError, "origin"):
            CropImage().execute(image=source, x=10, y=0)
        with self.assertRaisesRegex(ValueError, "dimensions must match"):
            CropImage().execute(
                image=[source, Image.new("RGB", (9, 8), "blue")],
                x=7,
                y=6,
            )

    def test_tile_uses_row_major_rectangles_and_preserves_every_pixel(self):
        source = Image.new("RGB", (5, 3))
        for y in range(source.height):
            for x in range(source.width):
                source.putpixel((x, y), (x * 20, y * 30, 0))
        result = TileImage().execute(image=source, rows=2, columns=3)

        self.assertEqual(result["count"], 6)
        self.assertEqual([item["index"] for item in result["layout"]], list(range(6)))
        canvas = Image.new("RGB", source.size)
        for tile, item in zip(result["tiles"], result["layout"]):
            canvas.paste(tile, (item["x"], item["y"]))
        self.assertEqual(canvas.tobytes(), source.tobytes())

        with self.assertRaisesRegex(ValueError, "exactly one"):
            TileImage().execute(image=[source, source], rows=2, columns=2)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            TileImage().execute(image=Image.new("RGB", (2, 2)), rows=3, columns=1)

    def test_stitch_is_bounded_row_major_and_preserves_alpha(self):
        images = [
            Image.new("RGBA", (2, 2), (255, 0, 0, 255)),
            Image.new("RGBA", (1, 2), (0, 255, 0, 128)),
            Image.new("RGBA", (2, 1), (0, 0, 255, 255)),
            Image.new("RGBA", (2, 2), (255, 255, 0, 255)),
        ]
        result = StitchImages().execute(
            image=images,
            columns=2,
            spacing=1,
            background="transparent",
            match_size=True,
        )

        output = result["output"]
        self.assertEqual((result["rows"], result["count"]), (2, 4))
        self.assertEqual(output.size, (5, 5))
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(output.getpixel((2, 0)), (0, 0, 0, 0))
        self.assertEqual(output.getpixel((3, 0)), (0, 255, 0, 128))
        self.assertEqual(
            output.tobytes(),
            StitchImages()
            .execute(
                image=images,
                columns=2,
                spacing=1,
                background="transparent",
                match_size=True,
            )["output"]
            .tobytes(),
        )

        with self.assertRaisesRegex(ValueError, "between 2 and 64"):
            StitchImages().execute(image=images[0])
        with self.assertRaisesRegex(ValueError, "share dimensions"):
            StitchImages().execute(image=images, match_size=False)
        with self.assertRaisesRegex(ValueError, "background"):
            StitchImages().execute(image=images, background="external")

    def test_resize_supports_bounded_traditional_interpolation_and_fit_modes(self):
        source = Image.new("RGBA", (8, 4), (20, 40, 60, 71))
        node = ResizeImage()
        contain = node.execute(
            image=source,
            width=6,
            height=6,
            fit_mode="contain",
            resampling="lanczos",
        )["output"]
        cover = node.execute(
            image=source,
            width=6,
            height=6,
            fit_mode="cover",
            resampling="bicubic",
        )["output"]
        stretch = node.execute(
            image=source,
            width=6,
            height=6,
            fit_mode="stretch",
            resampling="nearest",
        )["output"]

        self.assertEqual(contain.size, (6, 3))
        self.assertEqual(cover.size, (6, 6))
        self.assertEqual(stretch.size, (6, 6))
        self.assertEqual(contain.getchannel("A").getextrema(), (71, 71))
        self.assertEqual(
            contain.tobytes(),
            node.execute(
                image=source,
                width=6,
                height=6,
                fit_mode="contain",
                resampling="lanczos",
            )["output"].tobytes(),
        )

        with self.assertRaisesRegex(ValueError, "fit mode"):
            node.execute(image=source, width=6, height=6, fit_mode="plugin")
        with self.assertRaisesRegex(ValueError, "interpolation"):
            node.execute(image=source, width=6, height=6, resampling="ai")
        with self.assertRaisesRegex(ValueError, "at most"):
            node.execute(
                image=source,
                width=8192,
                height=8192,
                fit_mode="stretch",
            )

    def test_channels_have_exact_values_and_alpha_defaults_to_opaque(self):
        rgba = Image.new("RGBA", (2, 1), (10, 20, 30, 40))
        result = ImageChannels().execute(image=rgba, channel="green")
        self.assertEqual(result["output"].getpixel((0, 0)), 20)
        self.assertEqual(result["red"].getpixel((0, 0)), 10)
        self.assertEqual(result["blue"].getpixel((0, 0)), 30)
        self.assertEqual(result["alpha"].getpixel((0, 0)), 40)

        rgb = ImageChannels().execute(image=Image.new("RGB", (1, 1), "black"))
        self.assertEqual(rgb["alpha"].getpixel((0, 0)), 255)
        with self.assertRaisesRegex(ValueError, "Unsupported image channel"):
            ImageChannels().execute(image=rgba, channel="cyan")

    def test_mask_composite_is_deterministic_channel_explicit_and_size_strict(self):
        background = Image.new("RGB", (2, 1), "red")
        foreground = Image.new("RGB", (2, 1), "blue")
        mask = Image.new("RGB", (2, 1))
        mask.putdata([(0, 0, 0), (255, 255, 255)])
        node = MaskComposite()

        output = node.execute(
            background=background,
            foreground=foreground,
            mask=mask,
            mask_channel="luminance",
        )["output"]
        inverted = node.execute(
            background=background,
            foreground=foreground,
            mask=mask,
            mask_channel="red",
            invert_mask=True,
        )["output"]
        self.assertEqual([output.getpixel((x, 0)) for x in range(2)], [(255, 0, 0), (0, 0, 255)])
        self.assertEqual([inverted.getpixel((x, 0)) for x in range(2)], [(0, 0, 255), (255, 0, 0)])
        with self.assertRaisesRegex(ValueError, "dimensions must match"):
            node.execute(
                background=background,
                foreground=foreground,
                mask=Image.new("L", (1, 1)),
            )
        with self.assertRaisesRegex(ValueError, "Unsupported mask channel"):
            node.execute(background=background, foreground=foreground, mask=mask, mask_channel="depth")

    def test_collection_and_pixel_limits_fail_closed_before_processing(self):
        source = Image.new("RGB", (1, 1), "black")
        with self.assertRaisesRegex(ValueError, "at most"):
            AdjustImage().execute(image=[source] * (MAX_IMAGE_COUNT + 1))
        with self.assertRaisesRegex(TypeError, "must be a PIL image"):
            FilterImage().execute(image=[source, object()])

    def test_generic_facade_dispatches_every_reviewed_mode_and_locks_identity(self):
        source = Image.new("RGBA", (8, 6), (40, 80, 120, 99))
        node = ProcessImage()
        results = {
            "image_adjustment": node.execute(image=source, operation="image_adjustment"),
            "image_filter": node.execute(
                image=source,
                operation="image_filter",
                filter_operation="unsharp_mask",
            ),
            "image_crop": node.execute(
                image=source,
                operation="image_crop",
                x=1,
                y=1,
                width=4,
                height=3,
            ),
            "image_upscale": node.execute(
                image=source,
                operation="image_upscale",
                resize_width=4,
                resize_height=4,
                resize_fit_mode="contain",
                resize_resampling="bilinear",
            ),
            "image_stitch": node.execute(
                image=[source, source, source, source],
                operation="image_stitch",
                stitch_columns=2,
            ),
            "image_tile": node.execute(
                image=source,
                operation="image_tile",
                rows=2,
                columns=2,
            ),
            "image_channels": node.execute(
                image=source,
                operation="image_channels",
                channel="red",
            ),
            "mask_composite": node.execute(
                image=[source, Image.new("RGBA", source.size, (200, 40, 20, 255))],
                mask=Image.new("L", source.size, 255),
                operation="mask_composite",
            ),
        }
        self.assertEqual(set(results), set(ProcessImage.params["operation"]["options"]))
        for mode, result in results.items():
            with self.subTest(output_contract=mode):
                self.assertEqual(set(result), {"output"})
        self.assertEqual(results["image_crop"]["output"].size, (4, 3))
        self.assertEqual(results["image_upscale"]["output"].size, (4, 3))
        self.assertEqual(results["image_stitch"]["output"].size, (16, 12))
        self.assertEqual(len(results["image_tile"]["output"]), 4)
        self.assertEqual(results["image_channels"]["output"].mode, "L")
        self.assertEqual(results["mask_composite"]["output"].getpixel((0, 0)), (200, 40, 20, 255))

        with self.assertRaisesRegex(ValueError, "contract identity"):
            node.execute(
                image=source,
                operation="image_adjustment",
                pipeline_class="MutableImageOperation",
            )
        with self.assertRaisesRegex(ValueError, "Unsupported built-in"):
            node.execute(
                image=source,
                operation="external_plugin",
                pipeline_class=IMAGE_OPERATION_PIPELINE_CLASS,
            )

    def test_registry_exposes_only_the_nine_generic_operations(self):
        from modules import MODULE_MAP

        self.assertEqual(
            set(MODULE_MAP["modules.ImageOperations"]),
            {
                "AdjustImage",
                "FilterImage",
                "CropImage",
                "ResizeImage",
                "StitchImages",
                "TileImage",
                "ImageChannels",
                "MaskComposite",
                "ProcessImage",
            },
        )
        for definition in MODULE_MAP["modules.ImageOperations"].values():
            self.assertEqual(definition["category"], "Image Operations")
            self.assertTrue(definition["params"])


if __name__ == "__main__":
    unittest.main()
