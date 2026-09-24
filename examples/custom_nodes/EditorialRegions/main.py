"""Explicit, bounded image-space art direction. No semantic mask inference."""

import json
import math

from modiff.NodeBase import NodeBase


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return float(value)


def image_value(value):
    from PIL import Image

    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, Image.Image):
        raise ValueError("Expected one decoded PIL image (or a one-image batch).")
    if not 1 <= value.width * value.height <= 4_194_304:
        raise ValueError("Image must contain at most 4,194,304 pixels.")
    return value


def rgb_image(value):
    from PIL import Image

    value = image_value(value)
    return Image.alpha_composite(Image.new("RGBA", value.size, "white"), value.convert("RGBA")).convert("RGB")


def region_mask(size, regions, feather):
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    number(feather, "feather", 0, 0.1)
    if not isinstance(regions, str) or len(regions) > 16_384:
        raise ValueError("Regions must be a JSON string of at most 16,384 characters.")
    try:
        shapes = json.loads(regions)
    except (ValueError, TypeError) as exc:
        raise ValueError("Regions must be valid JSON.") from exc
    if not isinstance(shapes, list) or not 1 <= len(shapes) <= 16:
        raise ValueError("Supply 1 to 16 region polygons.")
    result = Image.new("L", size, 0)
    for shape in shapes:
        if not isinstance(shape, dict) or set(shape) != {"operation", "points"}:
            raise ValueError("Each region requires only operation and points.")
        if shape["operation"] not in ("add", "subtract", "intersect"):
            raise ValueError("Region operation must be add, subtract or intersect.")
        points = shape["points"]
        if not isinstance(points, list) or not 3 <= len(points) <= 32:
            raise ValueError("Each polygon requires 3 to 32 normalized [x,y] points.")
        coordinates = []
        for point in points:
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("Each point must be [x,y].")
            coordinates.append(tuple(number(v, "coordinate", 0, 1) * (size[i] - 1) for i, v in enumerate(point)))
        layer = Image.new("L", size, 0)
        ImageDraw.Draw(layer).polygon(coordinates, fill=255)
        if feather:
            layer = layer.filter(ImageFilter.GaussianBlur(feather * min(size)))
        if shape["operation"] == "add":
            result = ImageChops.lighter(result, layer)
        elif shape["operation"] == "subtract":
            result = ImageChops.subtract(result, layer)
        else:
            result = ImageChops.multiply(result, layer)
    return result


class EditorialRegions(NodeBase):
    label = "Editorial Regions"
    category = "Image"
    resizable = True
    params = {
        "image": {"label": "Source image", "type": "image", "display": "input"},
        "regions": {"label": "Region polygons (JSON)", "type": "string", "display": "textarea",
                    "default": '[{"operation":"add","points":[[0,0],[1,0],[1,1],[0,1]]}]'},
        "feather": {"label": "Edge feather", "type": "float", "default": 0.015, "min": 0, "max": 0.1, "step": 0.001},
        "exposure": {"label": "Exposure stops", "type": "float", "default": 0, "min": -3, "max": 3, "step": 0.1},
        "temperature": {"label": "Warm cool split", "type": "float", "default": 0, "min": -1, "max": 1, "step": 0.05},
        "saturation": {"label": "Saturation", "type": "float", "default": 1, "min": 0, "max": 2, "step": 0.05},
        "out_image": {"label": "Directed image", "type": "image", "display": "output"},
        "mask": {"label": "Edit mask", "type": "image", "display": "output"},
        "diagnostics": {"label": "Region diagnostics", "type": "dict", "display": "output"},
    }

    def execute(self, image, regions='[{"operation":"add","points":[[0,0],[1,0],[1,1],[0,1]]}]',
                feather=0.015, exposure=0, temperature=0, saturation=1):
        import numpy as np
        from PIL import Image

        source = rgb_image(image)
        mask = region_mask(source.size, regions, feather)
        exposure = number(exposure, "exposure", -3, 3)
        temperature = number(temperature, "temperature", -1, 1)
        saturation = number(saturation, "saturation", 0, 2)
        rgb = np.asarray(source, dtype=np.float32) / 255
        # Exposure in linear light; no hidden model, downloads or file access.
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        luma = linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        directed = (luma[..., None] + saturation * (linear - luma[..., None])) * (2 ** exposure)
        # Opposed horizontal gels: warm on the left, cool on the right.
        gradient = np.linspace(1, -1, source.width, dtype=np.float32)[None, :, None]
        directed *= np.exp(gradient * temperature * np.array([0.8, 0.0, -0.8], dtype=np.float32))
        directed = np.clip(directed, 0, 1)
        srgb = np.where(directed <= 0.0031308, directed * 12.92, 1.055 * directed ** (1 / 2.4) - 0.055)
        edited = Image.fromarray(np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8))
        output = Image.composite(edited, source, mask)
        return {"out_image": output, "mask": mask,
                "diagnostics": {"width": source.width, "height": source.height,
                                "mask_coverage": float(np.asarray(mask).mean() / 255),
                                "semantic_segmentation": False}}


class ProtectedComposite(NodeBase):
    label = "Protected Editorial Composite"
    category = "Image"
    resizable = True
    params = {
        "original": {"label": "Protected original", "type": "image", "display": "input"},
        "edited": {"label": "Generated edit", "type": "image", "display": "input"},
        "mask": {"label": "Edit mask", "type": "image", "display": "input"},
        "out_image": {"label": "Final image", "type": "image", "display": "output"},
    }

    def execute(self, original, edited, mask):
        from PIL import Image

        original, edited, mask = rgb_image(original), rgb_image(edited), image_value(mask)
        if original.size != edited.size or original.size != mask.size:
            raise ValueError("Original, edit and mask must have identical dimensions; resize explicitly upstream.")
        if mask.mode != "L":
            raise ValueError("Edit mask must be grayscale L, not an ambiguous RGB image.")
        return {"out_image": Image.composite(edited, original, mask)}
