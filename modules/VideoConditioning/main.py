from PIL import Image, ImageOps

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
        return frame
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().float().cpu()
        if tensor.ndim == 4:
            tensor = tensor.squeeze(0)
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype.kind == "f":
            array = (array.clip(0, 1) * 255).astype("uint8")
        return Image.fromarray(array)
    if isinstance(frame, np.ndarray):
        array = frame
        if array.dtype.kind == "f":
            array = (array.clip(0, 1) * 255).astype("uint8")
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        return Image.fromarray(array)
    return frame


def resize_frame(frame, width, height, fit):
    image = to_image(frame).convert("RGB")
    if fit == "stretch":
        return image.resize((width, height), Image.Resampling.LANCZOS)
    if fit == "contain":
        contained = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        canvas.paste(contained, ((width - contained.width) // 2, (height - contained.height) // 2))
        return canvas
    return ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS)


def sample_frames(frames, count):
    if count <= 0 or not frames:
        return []
    if len(frames) == count:
        return frames
    if count == 1:
        return [frames[0]]
    return [frames[round(index * (len(frames) - 1) / (count - 1))] for index in range(count)]


class Normalize(NodeBase):
    """Trim, resize, crop, and normalize a video frame list for video models."""

    label = "Normalize Video"
    category = "Video Conditioning"
    resizable = True
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video", "image"]},
        "width": {"label": "Width", "type": "int", "default": 832, "min": 16, "max": 2048, "step": 16},
        "height": {"label": "Height", "type": "int", "default": 480, "min": 16, "max": 2048, "step": 16},
        "num_frames": {"label": "Frames", "type": "int", "default": 81, "min": 1, "max": 241},
        "fit": {"label": "Fit", "type": "string", "options": ["cover", "contain", "stretch"], "default": "cover"},
        "output": {"label": "Video", "display": "output", "type": "video"},
        "frames_out": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        frames = as_frames(kwargs.get("video"))
        width = int(kwargs.get("width", 832))
        height = int(kwargs.get("height", 480))
        num_frames = int(kwargs.get("num_frames", 81))
        fit = kwargs.get("fit", "cover")
        normalized = [resize_frame(frame, width, height, fit) for frame in sample_frames(frames, num_frames)]
        return {"output": normalized, "frames_out": len(normalized)}


class AlignMask(NodeBase):
    """Resize a mask video to match a source video and repeat/sampling frames as needed."""

    label = "Align Video Mask"
    category = "Video Conditioning"
    resizable = True
    params = {
        "video": {"label": "Source video", "display": "input", "type": "video"},
        "mask": {"label": "Mask", "display": "input", "type": ["video", "image"]},
        "invert": {"label": "Invert mask", "type": "bool", "default": False},
        "threshold": {"label": "Threshold", "display": "slider", "type": "int", "min": 0, "max": 255, "default": 127},
        "output": {"label": "Mask video", "display": "output", "type": "video"},
    }

    def execute(self, **kwargs):
        source_frames = as_frames(kwargs.get("video"))
        mask_frames = as_frames(kwargs.get("mask"))
        if not source_frames or not mask_frames:
            return {"output": []}

        threshold = int(kwargs.get("threshold", 127))
        sampled_masks = sample_frames(mask_frames, len(source_frames))
        output = []
        for source, mask in zip(source_frames, sampled_masks):
            source_image = to_image(source)
            mask_image = to_image(mask).convert("L").resize(source_image.size, Image.Resampling.LANCZOS)
            mask_image = mask_image.point(lambda value: 255 if value > threshold else 0)
            if bool(kwargs.get("invert", False)):
                mask_image = ImageOps.invert(mask_image)
            output.append(mask_image.convert("RGB"))
        return {"output": output}


class FirstLastFrames(NodeBase):
    """Extract first and last frames from a video for reference or boundary conditioning."""

    label = "First/Last Video Frames"
    category = "Video Conditioning"
    params = {
        "video": {"label": "Video", "display": "input", "type": "video"},
        "first": {"label": "First", "display": "output", "type": "image"},
        "last": {"label": "Last", "display": "output", "type": "image"},
        "references": {"label": "References", "display": "output", "type": "image"},
    }

    def execute(self, **kwargs):
        frames = as_frames(kwargs.get("video"))
        if not frames:
            return {"first": None, "last": None, "references": []}
        first = to_image(frames[0]).convert("RGB")
        last = to_image(frames[-1]).convert("RGB")
        return {"first": first, "last": last, "references": [first, last]}


class ReferenceImages(NodeBase):
    """Pass through image references as a VACE reference-image list."""

    label = "Wan VACE References"
    category = "Video Conditioning"
    params = {
        "images": {"label": "Images", "display": "input", "type": "image"},
        "references": {"label": "References", "display": "output", "type": "image"},
    }

    def execute(self, **kwargs):
        images = kwargs.get("images")
        if images in (None, ""):
            return {"references": []}
        return {"references": images if isinstance(images, list) else [images]}
