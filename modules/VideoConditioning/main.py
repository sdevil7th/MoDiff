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
        "grow_pixels": {
            "label": "Grow generated region",
            "type": "int",
            "min": 0,
            "max": 256,
            "default": 0,
        },
        "output": {"label": "Mask video", "display": "output", "type": "video"},
    }

    def execute(self, **kwargs):
        source_frames = as_frames(kwargs.get("video"))
        mask_frames = as_frames(kwargs.get("mask"))
        if not source_frames or not mask_frames:
            return {"output": []}

        threshold = int(kwargs.get("threshold", 127))
        grow_pixels = max(0, int(kwargs.get("grow_pixels", 0)))
        sampled_masks = sample_frames(mask_frames, len(source_frames))
        output = []
        for source, mask in zip(source_frames, sampled_masks):
            source_image = to_image(source)
            mask_image = to_image(mask).convert("L").resize(source_image.size, Image.Resampling.LANCZOS)
            mask_image = mask_image.point(lambda value: 255 if value > threshold else 0)
            if bool(kwargs.get("invert", False)):
                mask_image = ImageOps.invert(mask_image)
            if grow_pixels:
                import numpy as np

                try:
                    import cv2

                    mask_array = np.asarray(mask_image)
                    kernel_size = grow_pixels * 2 + 1
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                    mask_image = Image.fromarray(cv2.dilate(mask_array, kernel, iterations=1)).convert("L")
                except ImportError:  # pragma: no cover - full installs include OpenCV
                    from PIL import ImageFilter

                    mask_image = mask_image.filter(ImageFilter.MaxFilter(grow_pixels * 2 + 1))
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
    """Package one or more images as a reusable video reference list."""

    label = "Video Reference Images"
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


class EdgePreprocessor(NodeBase):
    """Create deterministic edge or sketch control frames locally."""

    label = "Video Edge / Sketch Preprocessor"
    category = "Video Conditioning"
    resizable = True
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video", "image"]},
        "algorithm": {"label": "Algorithm", "type": "string", "options": ["canny", "sobel", "sketch"], "default": "canny"},
        "low_threshold": {"label": "Low Threshold", "type": "int", "default": 100, "min": 0, "max": 255},
        "high_threshold": {"label": "High Threshold", "type": "int", "default": 200, "min": 0, "max": 255},
        "blur_radius": {"label": "Blur", "type": "int", "default": 3, "min": 0, "max": 31},
        "invert": {"label": "Invert", "type": "bool", "default": False},
        "output": {"label": "Control Video", "display": "output", "type": "video"},
    }

    def execute(self, **kwargs):
        import numpy as np
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video edge preprocessing needs the gallery-media extra (OpenCV).") from exc

        algorithm = str(kwargs.get("algorithm") or "canny")
        blur = max(0, int(kwargs.get("blur_radius") or 0))
        if blur and blur % 2 == 0:
            blur += 1
        output = []
        for frame in as_frames(kwargs.get("video")):
            rgb = np.asarray(to_image(frame).convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if blur:
                gray = cv2.GaussianBlur(gray, (blur, blur), 0)
            if algorithm == "canny":
                edge = cv2.Canny(
                    gray,
                    int(kwargs.get("low_threshold") or 0),
                    int(kwargs.get("high_threshold") or 0),
                )
            elif algorithm == "sobel":
                x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                edge = cv2.convertScaleAbs(cv2.magnitude(x, y))
            elif algorithm == "sketch":
                inverted = 255 - gray
                smooth = cv2.GaussianBlur(inverted, (max(3, blur or 21), max(3, blur or 21)), 0)
                edge = cv2.divide(gray, 255 - smooth, scale=256)
            else:
                raise ValueError(f"Unsupported edge algorithm {algorithm!r}.")
            if bool(kwargs.get("invert", False)):
                edge = 255 - edge
            output.append(Image.fromarray(edge).convert("RGB"))
        return {"output": output}


class ObjectMaskPropagate(NodeBase):
    """Propagate one authored mask with local dense optical flow."""

    label = "Propagate Object Mask"
    category = "Video Conditioning"
    resizable = True
    params = {
        "video": {"label": "Video", "display": "input", "type": "video"},
        "first_mask": {"label": "First-frame Mask", "display": "input", "type": "image"},
        "threshold": {"label": "Mask Threshold", "type": "int", "default": 127, "min": 0, "max": 255},
        "smooth_pixels": {"label": "Smooth", "type": "int", "default": 3, "min": 0, "max": 31},
        "masks": {"label": "Mask Video", "display": "output", "type": "video"},
        "confidence": {"label": "Per-frame Confidence", "display": "output", "type": "collection"},
    }

    def execute(self, **kwargs):
        import numpy as np
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Object mask propagation needs the gallery-media extra (OpenCV).") from exc

        frames = [np.asarray(to_image(frame).convert("RGB")) for frame in as_frames(kwargs.get("video"))]
        if not frames or kwargs.get("first_mask") is None:
            raise ValueError("Propagate Object Mask needs a video and a first-frame mask.")
        height, width = frames[0].shape[:2]
        threshold = int(kwargs.get("threshold") or 0)
        mask = np.asarray(to_image(kwargs.get("first_mask")).convert("L").resize((width, height), Image.Resampling.LANCZOS))
        mask = np.where(mask > threshold, 255, 0).astype(np.uint8)
        masks = [Image.fromarray(mask).convert("RGB")]
        confidence = [1.0]
        previous_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        smooth = max(0, int(kwargs.get("smooth_pixels") or 0))
        if smooth and smooth % 2 == 0:
            smooth += 1
        for frame in frames[1:]:
            current_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            warped = cv2.remap(mask, grid_x - flow[..., 0], grid_y - flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            if smooth:
                warped = cv2.GaussianBlur(warped, (smooth, smooth), 0)
            mask = np.where(warped > threshold, 255, 0).astype(np.uint8)
            remapped_previous = cv2.remap(previous_gray, grid_x - flow[..., 0], grid_y - flow[..., 1], cv2.INTER_LINEAR)
            error = np.abs(remapped_previous.astype(np.float32) - current_gray.astype(np.float32))
            region = mask > 0
            score = 1.0 - float(error[region].mean() / 255.0) if region.any() else 0.0
            confidence.append(max(0.0, min(1.0, score)))
            masks.append(Image.fromarray(mask).convert("RGB"))
            previous_gray = current_gray
        return {"masks": masks, "confidence": confidence}
