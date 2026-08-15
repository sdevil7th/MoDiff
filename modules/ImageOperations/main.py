"""Model-neutral, deterministic image processing nodes."""

from __future__ import annotations

import math
import random
from typing import Any

from PIL import Image, ImageChops, ImageEnhance, ImageFilter

from modiff.NodeBase import NodeBase


MAX_IMAGE_COUNT = 64
MAX_IMAGE_PIXELS = 16_777_216
MAX_TOTAL_PIXELS = 67_108_864
MAX_TILE_COUNT = 64
FILTER_OPERATIONS = (
    "gaussian_blur",
    "box_blur",
    "sharpen",
    "unsharp_mask",
    "glow",
    "film_grain",
    "chromatic_aberration",
)
IMAGE_CHANNELS = ("red", "green", "blue", "alpha", "luminance")


def _bounded_number(
    value: Any,
    *,
    name: str,
    minimum: float,
    maximum: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return number


def _images(value: Any, *, name: str = "image") -> tuple[list[Image.Image], bool]:
    singular = isinstance(value, Image.Image)
    values = [value] if singular else value
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{name} needs one or more images.")
    if len(values) > MAX_IMAGE_COUNT:
        raise ValueError(f"{name} accepts at most {MAX_IMAGE_COUNT} images per execution.")
    output = []
    total_pixels = 0
    for index, item in enumerate(values):
        if not isinstance(item, Image.Image):
            raise TypeError(f"{name} item {index + 1} must be a PIL image.")
        pixels = item.width * item.height
        if item.width < 1 or item.height < 1 or pixels > MAX_IMAGE_PIXELS:
            raise ValueError(f"{name} item {index + 1} must contain between 1 and {MAX_IMAGE_PIXELS} pixels.")
        total_pixels += pixels
        if total_pixels > MAX_TOTAL_PIXELS:
            raise ValueError(f"{name} exceeds the {MAX_TOTAL_PIXELS}-pixel execution limit.")
        output.append(item)
    return output, singular


def _collapse(values: list[Image.Image], singular: bool) -> Image.Image | list[Image.Image]:
    return values[0] if singular else values


def _split_color_alpha(image: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    return image.convert("RGB"), alpha


def _restore_alpha(image: Image.Image, alpha: Image.Image | None) -> Image.Image:
    if alpha is None:
        return image
    output = image.convert("RGBA")
    output.putalpha(alpha)
    return output


def _point_gain(channel: Image.Image, gain: float) -> Image.Image:
    return channel.point([max(0, min(255, round(index * gain))) for index in range(256)])


class AdjustImage(NodeBase):
    """Apply bounded color and tone adjustments while preserving alpha."""

    label = "Adjust Image"
    category = "Image Operations"
    resizable = True
    params = {
        "image": {"label": "Image", "display": "input", "type": "image"},
        "brightness": {
            "label": "Brightness",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 4.0,
            "step": 0.05,
        },
        "contrast": {
            "label": "Contrast",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 4.0,
            "step": 0.05,
        },
        "saturation": {
            "label": "Saturation",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 4.0,
            "step": 0.05,
        },
        "sharpness": {
            "label": "Sharpness",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 4.0,
            "step": 0.05,
        },
        "gamma": {
            "label": "Gamma",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": 0.1,
            "max": 4.0,
            "step": 0.05,
        },
        "temperature": {
            "label": "Temperature",
            "display": "slider",
            "type": "float",
            "default": 0.0,
            "min": -1.0,
            "max": 1.0,
            "step": 0.05,
        },
        "tint": {
            "label": "Tint",
            "display": "slider",
            "type": "float",
            "default": 0.0,
            "min": -1.0,
            "max": 1.0,
            "step": 0.05,
        },
        "output": {"label": "Adjusted Image", "display": "output", "type": "image"},
    }

    def execute(self, **kwargs):
        images, singular = _images(kwargs.get("image"))
        brightness = _bounded_number(kwargs.get("brightness", 1.0), name="Brightness", minimum=0, maximum=4)
        contrast = _bounded_number(kwargs.get("contrast", 1.0), name="Contrast", minimum=0, maximum=4)
        saturation = _bounded_number(kwargs.get("saturation", 1.0), name="Saturation", minimum=0, maximum=4)
        sharpness = _bounded_number(kwargs.get("sharpness", 1.0), name="Sharpness", minimum=0, maximum=4)
        gamma = _bounded_number(kwargs.get("gamma", 1.0), name="Gamma", minimum=0.1, maximum=4)
        temperature = _bounded_number(kwargs.get("temperature", 0.0), name="Temperature", minimum=-1, maximum=1)
        tint = _bounded_number(kwargs.get("tint", 0.0), name="Tint", minimum=-1, maximum=1)
        gamma_lut = [round(((index / 255) ** (1 / gamma)) * 255) for index in range(256)]
        output = []
        for source in images:
            image, alpha = _split_color_alpha(source)
            image = ImageEnhance.Brightness(image).enhance(brightness)
            image = ImageEnhance.Contrast(image).enhance(contrast)
            image = ImageEnhance.Color(image).enhance(saturation)
            image = ImageEnhance.Sharpness(image).enhance(sharpness)
            image = image.point(gamma_lut * 3)
            red, green, blue = image.split()
            red = _point_gain(red, 1 + temperature * 0.5 - tint * 0.15)
            green = _point_gain(green, 1 + tint * 0.35)
            blue = _point_gain(blue, 1 - temperature * 0.5 - tint * 0.15)
            output.append(_restore_alpha(Image.merge("RGB", (red, green, blue)), alpha))
        return {"output": _collapse(output, singular)}


def _shift_channel(channel: Image.Image, offset: int) -> Image.Image:
    output = Image.new("L", channel.size, 0)
    width, height = channel.size
    if offset >= 0:
        if offset < width:
            output.paste(channel.crop((0, 0, width - offset, height)), (offset, 0))
    elif -offset < width:
        output.paste(channel.crop((-offset, 0, width, height)), (0, 0))
    return output


class FilterImage(NodeBase):
    """Apply one deterministic bounded filter to an image collection."""

    label = "Filter Image"
    category = "Image Operations"
    resizable = True
    params = {
        "image": {"label": "Image", "display": "input", "type": "image"},
        "operation": {
            "label": "Filter",
            "type": "string",
            "options": FILTER_OPERATIONS,
            "default": "gaussian_blur",
        },
        "amount": {
            "label": "Amount",
            "display": "slider",
            "type": "float",
            "default": 1.0,
            "min": 0.0,
            "max": 32.0,
            "step": 0.25,
        },
        "threshold": {
            "label": "Threshold",
            "display": "slider",
            "type": "int",
            "default": 3,
            "min": 0,
            "max": 255,
        },
        "seed": {"label": "Seed", "type": "int", "default": 0},
        "output": {"label": "Filtered Image", "display": "output", "type": "image"},
    }

    def execute(self, **kwargs):
        images, singular = _images(kwargs.get("image"))
        operation = str(kwargs.get("operation") or "gaussian_blur")
        allowed = set(FILTER_OPERATIONS)
        if operation not in allowed:
            raise ValueError(f"Unsupported image filter {operation!r}.")
        amount = _bounded_number(kwargs.get("amount", 1.0), name="Filter amount", minimum=0, maximum=32)
        threshold = int(_bounded_number(kwargs.get("threshold", 3), name="Filter threshold", minimum=0, maximum=255))
        try:
            seed = int(kwargs.get("seed", 0))
        except (TypeError, ValueError) as error:
            raise ValueError("Filter seed must be an integer.") from error
        output = []
        for image_index, source in enumerate(images):
            image, alpha = _split_color_alpha(source)
            if operation == "gaussian_blur":
                filtered = image.filter(ImageFilter.GaussianBlur(radius=amount))
            elif operation == "box_blur":
                filtered = image.filter(ImageFilter.BoxBlur(radius=amount))
            elif operation == "sharpen":
                filtered = ImageEnhance.Sharpness(image).enhance(1 + amount)
            elif operation == "unsharp_mask":
                filtered = image.filter(
                    ImageFilter.UnsharpMask(
                        radius=max(0.1, amount),
                        percent=min(500, round(amount * 50)),
                        threshold=threshold,
                    )
                )
            elif operation == "glow":
                blurred = image.filter(ImageFilter.GaussianBlur(radius=amount))
                filtered = ImageChops.screen(image, blurred)
            elif operation == "film_grain":
                generator = random.Random(seed + image_index)
                amplitude = min(64, round(amount * 2))
                noise = bytes(generator.randrange(256 - amplitude, 256) for _ in range(image.width * image.height))
                noise_image = Image.frombytes("L", image.size, noise).convert("RGB")
                filtered = ImageChops.multiply(image, noise_image)
            else:
                offset = round(amount)
                red, green, blue = image.split()
                filtered = Image.merge(
                    "RGB",
                    (_shift_channel(red, offset), green, _shift_channel(blue, -offset)),
                )
            output.append(_restore_alpha(filtered, alpha))
        return {"output": _collapse(output, singular)}


class CropImage(NodeBase):
    """Crop images to one explicit bounded rectangle."""

    label = "Crop Image"
    category = "Image Operations"
    params = {
        "image": {"label": "Image", "display": "input", "type": "image"},
        "x": {"label": "Left", "type": "int", "default": 0, "min": 0, "max": 32767},
        "y": {"label": "Top", "type": "int", "default": 0, "min": 0, "max": 32767},
        "width": {
            "label": "Width (0 = remaining)",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 32768,
        },
        "height": {
            "label": "Height (0 = remaining)",
            "type": "int",
            "default": 0,
            "min": 0,
            "max": 32768,
        },
        "output": {"label": "Cropped Image", "display": "output", "type": "image"},
        "output_width": {"label": "Width", "display": "output", "type": "int"},
        "output_height": {"label": "Height", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        images, singular = _images(kwargs.get("image"))
        values = {}
        for name, maximum in (("x", 32767), ("y", 32767), ("width", 32768), ("height", 32768)):
            try:
                value = int(kwargs.get(name, 0))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Crop {name} must be an integer.") from error
            if not 0 <= value <= maximum:
                raise ValueError(f"Crop {name} must be between 0 and {maximum}.")
            values[name] = value
        output = []
        output_sizes = set()
        for image in images:
            if values["x"] >= image.width or values["y"] >= image.height:
                raise ValueError("Crop origin must be inside every input image.")
            right = image.width if values["width"] == 0 else values["x"] + values["width"]
            bottom = image.height if values["height"] == 0 else values["y"] + values["height"]
            right = min(image.width, right)
            bottom = min(image.height, bottom)
            if right <= values["x"] or bottom <= values["y"]:
                raise ValueError("Crop rectangle must contain at least one pixel.")
            cropped = image.crop((values["x"], values["y"], right, bottom))
            output.append(cropped)
            output_sizes.add(cropped.size)
        if len(output_sizes) != 1:
            raise ValueError("Crop output dimensions must match across an image collection.")
        width, height = next(iter(output_sizes))
        return {
            "output": _collapse(output, singular),
            "output_width": width,
            "output_height": height,
        }


class TileImage(NodeBase):
    """Split one image into a bounded row-major grid without losing pixels."""

    label = "Split Image into Tiles"
    category = "Image Operations"
    params = {
        "image": {"label": "Image", "display": "input", "type": "image"},
        "rows": {"label": "Rows", "type": "int", "default": 2, "min": 1, "max": 8},
        "columns": {"label": "Columns", "type": "int", "default": 2, "min": 1, "max": 8},
        "tiles": {"label": "Tiles", "display": "output", "type": "image"},
        "count": {"label": "Tile Count", "display": "output", "type": "int"},
        "layout": {"label": "Tile Rectangles", "display": "output", "type": "collection"},
    }

    def execute(self, **kwargs):
        images, singular = _images(kwargs.get("image"))
        if not singular:
            raise ValueError("Split Image into Tiles accepts exactly one image per execution.")
        try:
            rows = int(kwargs.get("rows", 2))
            columns = int(kwargs.get("columns", 2))
        except (TypeError, ValueError) as error:
            raise ValueError("Tile rows and columns must be integers.") from error
        if not 1 <= rows <= 8 or not 1 <= columns <= 8 or rows * columns > MAX_TILE_COUNT:
            raise ValueError("Tile layout must contain between 1 and 64 tiles with at most 8 rows or columns.")
        image = images[0]
        if rows > image.height or columns > image.width:
            raise ValueError("Tile rows and columns cannot exceed the image dimensions.")
        x_bounds = [round(index * image.width / columns) for index in range(columns + 1)]
        y_bounds = [round(index * image.height / rows) for index in range(rows + 1)]
        tiles = []
        layout = []
        for row in range(rows):
            for column in range(columns):
                box = (
                    x_bounds[column],
                    y_bounds[row],
                    x_bounds[column + 1],
                    y_bounds[row + 1],
                )
                tiles.append(image.crop(box))
                layout.append(
                    {
                        "index": len(tiles) - 1,
                        "row": row,
                        "column": column,
                        "x": box[0],
                        "y": box[1],
                        "width": box[2] - box[0],
                        "height": box[3] - box[1],
                    }
                )
        return {"tiles": tiles, "count": len(tiles), "layout": layout}


class ImageChannels(NodeBase):
    """Extract one channel as a grayscale image while exposing all channels."""

    label = "Extract Image Channels"
    category = "Image Operations"
    params = {
        "image": {"label": "Image", "display": "input", "type": "image"},
        "channel": {
            "label": "Selected Channel",
            "type": "string",
            "options": IMAGE_CHANNELS,
            "default": "luminance",
        },
        "output": {"label": "Selected Channel", "display": "output", "type": "image"},
        "red": {"label": "Red", "display": "output", "type": "image"},
        "green": {"label": "Green", "display": "output", "type": "image"},
        "blue": {"label": "Blue", "display": "output", "type": "image"},
        "alpha": {"label": "Alpha", "display": "output", "type": "image"},
        "luminance": {"label": "Luminance", "display": "output", "type": "image"},
    }

    def execute(self, **kwargs):
        images, singular = _images(kwargs.get("image"))
        channel = str(kwargs.get("channel") or "luminance")
        allowed = set(IMAGE_CHANNELS)
        if channel not in allowed:
            raise ValueError(f"Unsupported image channel {channel!r}.")
        values = {name: [] for name in allowed}
        for image in images:
            rgba = image.convert("RGBA")
            red, green, blue, alpha = rgba.split()
            values["red"].append(red)
            values["green"].append(green)
            values["blue"].append(blue)
            values["alpha"].append(alpha)
            values["luminance"].append(image.convert("L"))
        return {name: _collapse(channel_images, singular) for name, channel_images in values.items()} | {
            "output": _collapse(values[channel], singular)
        }
