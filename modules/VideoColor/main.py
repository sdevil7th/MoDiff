from PIL import Image, ImageEnhance

from modiff.NodeBase import NodeBase


def as_frames(video):
    if video in (None, ""):
        return []
    if isinstance(video, list):
        return video
    return [video]


def to_image(frame):
    import numpy as np
    import torch

    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().float().cpu()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype.kind == "f":
            array = (array.clip(0, 1) * 255).astype("uint8")
        return Image.fromarray(array).convert("RGB")
    if isinstance(frame, np.ndarray):
        array = frame
        if array.dtype.kind == "f":
            array = (array.clip(0, 1) * 255).astype("uint8")
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        return Image.fromarray(array).convert("RGB")
    return frame.convert("RGB")


def clamp(value):
    return max(0, min(255, int(round(value))))


def apply_temperature_tint(image, temperature, tint):
    if temperature == 0 and tint == 0:
        return image
    r_gain = 1 + temperature / 100
    b_gain = 1 - temperature / 100
    g_gain = 1 + tint / 100
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            red, green, blue = pixels[x, y]
            pixels[x, y] = (clamp(red * r_gain), clamp(green * g_gain), clamp(blue * b_gain))
    return image


def apply_gamma(image, gamma):
    if gamma <= 0 or gamma == 1:
        return image
    inv = 1.0 / gamma
    lut = [clamp(((index / 255) ** inv) * 255) for index in range(256)]
    return image.point(lut * 3)


class Adjust(NodeBase):
    """Apply deterministic color correction to each video frame."""

    label = "Adjust Video Color"
    category = "Video Color"
    resizable = True
    params = {
        "video": {"label": "Video", "display": "input", "type": "video"},
        "brightness": {"label": "Brightness", "display": "slider", "type": "float", "min": 0, "max": 2, "step": 0.05, "default": 1.0},
        "contrast": {"label": "Contrast", "display": "slider", "type": "float", "min": 0, "max": 2, "step": 0.05, "default": 1.0},
        "saturation": {"label": "Saturation", "display": "slider", "type": "float", "min": 0, "max": 2, "step": 0.05, "default": 1.0},
        "sharpness": {"label": "Sharpness", "display": "slider", "type": "float", "min": 0, "max": 2, "step": 0.05, "default": 1.0},
        "gamma": {"label": "Gamma", "display": "slider", "type": "float", "min": 0.1, "max": 3, "step": 0.05, "default": 1.0},
        "temperature": {"label": "Temperature", "display": "slider", "type": "float", "min": -1, "max": 1, "step": 0.05, "default": 0.0},
        "tint": {"label": "Tint", "display": "slider", "type": "float", "min": -1, "max": 1, "step": 0.05, "default": 0.0},
        "output": {"label": "Video", "display": "output", "type": "video"},
        "frames_out": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        frames = as_frames(kwargs.get("video"))
        output = []
        for frame in frames:
            image = to_image(frame)
            image = ImageEnhance.Brightness(image).enhance(float(kwargs.get("brightness", 1.0)))
            image = ImageEnhance.Contrast(image).enhance(float(kwargs.get("contrast", 1.0)))
            image = ImageEnhance.Color(image).enhance(float(kwargs.get("saturation", 1.0)))
            image = ImageEnhance.Sharpness(image).enhance(float(kwargs.get("sharpness", 1.0)))
            image = apply_gamma(image, float(kwargs.get("gamma", 1.0)))
            image = apply_temperature_tint(image, float(kwargs.get("temperature", 0.0)), float(kwargs.get("tint", 0.0)))
            output.append(image)
        return {"output": output, "frames_out": len(output)}


class LUTApprox(NodeBase):
    """Apply a simple 3D-LUT-like color mix from RGB multipliers."""

    label = "Video Color Mix"
    category = "Video Color"
    params = {
        "video": {"label": "Video", "display": "input", "type": "video"},
        "red": {"label": "Red", "display": "slider", "type": "float", "min": 0, "max": 2, "step": 0.05, "default": 1.0},
        "green": {"label": "Green", "display": "slider", "type": "float", "min": 0, "max": 2, "step": 0.05, "default": 1.0},
        "blue": {"label": "Blue", "display": "slider", "type": "float", "min": 0, "max": 2, "step": 0.05, "default": 1.0},
        "output": {"label": "Video", "display": "output", "type": "video"},
    }

    def execute(self, **kwargs):
        output = []
        red_gain = float(kwargs.get("red", 1.0))
        green_gain = float(kwargs.get("green", 1.0))
        blue_gain = float(kwargs.get("blue", 1.0))
        for frame in as_frames(kwargs.get("video")):
            image = to_image(frame)
            pixels = image.load()
            for y in range(image.height):
                for x in range(image.width):
                    red, green, blue = pixels[x, y]
                    pixels[x, y] = (clamp(red * red_gain), clamp(green * green_gain), clamp(blue * blue_gain))
            output.append(image)
        return {"output": output}
