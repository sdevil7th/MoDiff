"""Deterministic, model-independent art direction and an inpainting mask."""

from modiff.NodeBase import NodeBase


def direct_image(image, *, shadow, midtone, highlight, center_x, center_y,
                 radius_x, radius_y, feather, strength, light_angle, light_intensity):
    import math

    import numpy as np
    from PIL import Image, ImageColor

    controls = {
        "center_x": (center_x, 0, 1), "center_y": (center_y, 0, 1),
        "radius_x": (radius_x, 0.01, 1), "radius_y": (radius_y, 0.01, 1),
        "feather": (feather, 0.01, 1), "strength": (strength, 0, 1),
        "light_angle": (light_angle, -180, 180), "light_intensity": (light_intensity, 0, 1),
    }
    for name, (value, low, high) in controls.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")
    if not isinstance(image, Image.Image):
        raise ValueError("Light & Palette Director expects decoded PIL images, not tensors or latents.")
    width, height = image.size
    if width < 1 or height < 1 or width * height > 4_194_304:
        raise ValueError("Light & Palette Director supports up to 4,194,304 pixels per image.")
    colors = []
    for name, value in (("shadow", shadow), ("midtone", midtone), ("highlight", highlight)):
        if not isinstance(value, str) or len(value) != 7 or not value.startswith("#"):
            raise ValueError(f"{name} must be an RGB hex color, for example #204060.")
        try:
            colors.append(np.asarray(ImageColor.getrgb(value), dtype=np.float32) / 255)
        except ValueError as exc:
            raise ValueError(f"Invalid {name} RGB color.") from exc

    # Composite transparent pixels on white, yielding an explicit RGB diffusion boundary.
    rgba = image.convert("RGBA")
    source = Image.alpha_composite(Image.new("RGBA", image.size, "white"), rgba).convert("RGB")
    rgb = np.asarray(source, dtype=np.float32) / 255
    x = (np.arange(width, dtype=np.float32) + 0.5) / width
    y = (np.arange(height, dtype=np.float32) + 0.5) / height
    dx = (x[None, :] - center_x) / radius_x
    dy = (y[:, None] - center_y) / radius_y
    distance = np.sqrt(dx * dx + dy * dy)
    mask = np.clip((1 - distance) / feather, 0, 1)
    mask = mask * mask * (3 - 2 * mask)  # Smoothstep: editable falloff for the reload demo.
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    lower = colors[0] + (colors[1] - colors[0]) * np.minimum(luma * 2, 1)[..., None]
    upper = colors[1] + (colors[2] - colors[1]) * np.maximum(luma * 2 - 1, 0)[..., None]
    palette = np.where((luma < 0.5)[..., None], lower, upper)
    # Retain source detail/chroma residual instead of replacing the image with flat colors.
    palette = palette + 0.35 * (rgb - luma[..., None])
    angle = math.radians(light_angle)
    light = np.clip((dx * math.cos(angle) + dy * math.sin(angle)) / 2, -1, 1)
    directed = np.clip(palette + light[..., None] * light_intensity * 0.3, 0, 1)
    blend = mask[..., None] * strength
    result = np.clip(np.rint((rgb * (1 - blend) + directed * blend) * 255), 0, 255).astype(np.uint8)
    return (
        Image.fromarray(result),
        Image.fromarray(np.rint(mask * 255).astype(np.uint8)),
        {"width": width, "height": height, "mask_coverage": float(mask.mean()),
         "settings": {key: value[0] for key, value in controls.items()},
         "palette": {"shadow": shadow, "midtone": midtone, "highlight": highlight}},
    )


class LightPaletteDirector(NodeBase):
    label = "Light & Palette Director"
    category = "Image"
    resizable = True
    params = {
        "image": {"label": "Image", "type": "image", "display": "input"},
        "shadow": {"label": "Shadow color", "type": "string", "default": "#173B60"},
        "midtone": {"label": "Midtone color", "type": "string", "default": "#E49C72"},
        "highlight": {"label": "Highlight color", "type": "string", "default": "#FFF1CC"},
        "center_x": {"label": "Region X", "type": "float", "default": 0.5, "min": 0, "max": 1, "step": 0.01},
        "center_y": {"label": "Region Y", "type": "float", "default": 0.5, "min": 0, "max": 1, "step": 0.01},
        "radius_x": {"label": "Region width", "type": "float", "default": 0.45, "min": 0.01, "max": 1, "step": 0.01},
        "radius_y": {"label": "Region height", "type": "float", "default": 0.48, "min": 0.01, "max": 1, "step": 0.01},
        "feather": {"label": "Feather", "type": "float", "default": 0.35, "min": 0.01, "max": 1, "step": 0.01},
        "strength": {"label": "Palette strength", "type": "float", "default": 0.65, "min": 0, "max": 1, "step": 0.01},
        "light_angle": {"label": "Light angle", "type": "float", "default": -45, "min": -180, "max": 180, "step": 1},
        "light_intensity": {"label": "Light intensity", "type": "float", "default": 0.5, "min": 0, "max": 1, "step": 0.01},
        "out_image": {"label": "Directed image", "type": "image", "display": "output"},
        "mask": {"label": "Soft mask", "type": "image", "display": "output"},
        "diagnostics": {"label": "Diagnostics", "type": "dict", "display": "output"},
    }

    def execute(self, image, shadow="#173B60", midtone="#E49C72", highlight="#FFF1CC",
                center_x=0.5, center_y=0.5, radius_x=0.45, radius_y=0.48, feather=0.35,
                strength=0.65, light_angle=-45, light_intensity=0.5):
        images = image if isinstance(image, (list, tuple)) else [image]
        if not 1 <= len(images) <= 4:
            raise ValueError("Light & Palette Director accepts one to four decoded images per run.")
        results = [direct_image(item, shadow=shadow, midtone=midtone, highlight=highlight,
                                center_x=center_x, center_y=center_y, radius_x=radius_x,
                                radius_y=radius_y, feather=feather, strength=strength,
                                light_angle=light_angle, light_intensity=light_intensity) for item in images]
        def unpack(values):
            return values[0] if len(values) == 1 else values
        return {"out_image": unpack([item[0] for item in results]),
                "mask": unpack([item[1] for item in results]),
                "diagnostics": unpack([item[2] for item in results])}
