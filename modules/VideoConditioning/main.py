from PIL import Image, ImageColor, ImageDraw, ImageOps

from modiff.NodeBase import NodeBase


def _storyboard_records(text):
    import json
    if not isinstance(text, str) or len(text.encode('utf-8')) > 1_048_576:
        raise ValueError('Storyboard JSON must be text of at most 1 MiB.')
    try:
        records = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise ValueError('Storyboard must contain valid bounded JSON.') from error
    if not isinstance(records, list) or not 1 <= len(records) <= 64:
        raise ValueError('Storyboard must contain one to 64 explicit shot records.')
    return records


class Storyboard(NodeBase):
    """Publish an authored shot list and one shared seed without executing it."""

    label = 'Storyboard'
    category = 'Video Conditioning'
    resizable = True
    params = {
        'shots_json': {'label': 'Shots (JSON)', 'type': 'text', 'display': 'textarea', 'default': '[]'},
        'base_seed': {'label': 'Base seed', 'type': 'int', 'display': 'random', 'default': 0, 'min': 0,
                      'max': 9_007_199_254_740_991},
        'storyboard': {'label': 'Storyboard', 'type': 'text', 'display': 'output'},
        'seed': {'label': 'Base seed', 'type': 'int', 'display': 'output'},
        'count': {'label': 'Shots', 'type': 'int', 'display': 'output'},
    }

    def execute(self, **kwargs):
        return evaluate_storyboard(kwargs)


class ShotInputs(NodeBase):
    """Select one bounded storyboard record for ordinary explicit video stages.

    This deterministic data operation never loads models, executes shots or
    reads another shot's images. Model-specific validity remains downstream.
    """

    label = 'Shot Inputs'
    category = 'Video Conditioning'
    resizable = True
    params = {
        'storyboard': {'label': 'Storyboard JSON', 'type': 'text', 'display': 'input', 'required': True},
        'index': {'label': 'Shot index (zero based)', 'type': 'int', 'default': 0, 'min': 0},
        'prompt_override': {'label': 'Optional shot prompt override', 'type': 'text', 'display': 'textarea', 'default': ''},
        'base_seed': {'label': 'Base seed', 'type': 'int', 'default': 0, 'min': 0},
        'use_seed_override': {'label': 'Override shot seed', 'type': 'bool', 'default': False},
        'seed_override': {'label': 'Shot seed override', 'type': 'int', 'display': 'random', 'default': 0, 'min': 0,
                          'max': 9_007_199_254_740_991},
        'negative_prompt': {'label': 'Negative prompt', 'type': 'text', 'display': 'textarea', 'default': ''},
        'width': {'label': 'Width', 'type': 'int', 'default': 832, 'min': 1},
        'height': {'label': 'Height', 'type': 'int', 'default': 480, 'min': 1},
        'num_frames': {'label': 'Frames', 'type': 'int', 'default': 81, 'min': 1},
        'fps': {'label': 'FPS', 'type': 'float', 'default': 16., 'min': .01},
        'steps': {'label': 'Steps', 'type': 'int', 'default': 40, 'min': 1},
        'guidance_scale': {'label': 'Guidance', 'type': 'float', 'default': 3.5, 'min': 0},
        'secondary_guidance_scale': {'label': 'Low-noise guidance', 'type': 'float', 'default': 3.5, 'min': 0},
        'conditioning_strength': {'label': 'Condition strength', 'type': 'float', 'default': .9, 'min': 0, 'max': 1},
        'prompt': {'label': 'Shot prompt', 'type': 'text', 'display': 'output'},
        'negative': {'label': 'Negative prompt', 'type': 'text', 'display': 'output'},
        'seed': {'label': 'Shot seed', 'type': 'int', 'display': 'output'},
        'title': {'label': 'Shot title', 'type': 'text', 'display': 'output'},
        'width_out': {'label': 'Width', 'type': 'int', 'display': 'output'},
        'height_out': {'label': 'Height', 'type': 'int', 'display': 'output'},
        'frames': {'label': 'Frames', 'type': 'int', 'display': 'output'},
        'fps_out': {'label': 'FPS', 'type': 'float', 'display': 'output'},
        'steps_out': {'label': 'Steps', 'type': 'int', 'display': 'output'},
        'guidance': {'label': 'Guidance', 'type': 'float', 'display': 'output'},
        'secondary_guidance': {'label': 'Low-noise guidance', 'type': 'float', 'display': 'output'},
        'condition_strength': {'label': 'Condition strength', 'type': 'float', 'display': 'output'},
    }

    def execute(self, **kwargs):
        return evaluate_shot_inputs(kwargs, parameters=type(self).params)


class ShotPrompt(NodeBase):
    """Select one shot's text without depending on its render or other shots."""

    label = 'Shot Prompt'
    category = 'Video Conditioning'
    resizable = True
    params = {
        'storyboard': {'label': 'Storyboard JSON', 'type': 'text', 'display': 'input', 'required': True},
        'index': {'label': 'Shot index (zero based)', 'type': 'int', 'default': 0, 'min': 0},
        'prompt_override': {'label': 'Optional shot prompt override', 'type': 'text', 'display': 'textarea', 'default': ''},
        'prompt': {'label': 'Shot prompt', 'type': 'text', 'display': 'output'},
    }

    def execute(self, **kwargs):
        return evaluate_shot_prompt(kwargs)


def evaluate_storyboard(kwargs):
    """Pure deterministic stage logic; safe for bounded resource inspection."""
    text = kwargs.get('shots_json', '[]')
    records = _storyboard_records(text)
    seed = kwargs.get('base_seed', 0)
    if type(seed) is not int or not 0 <= seed <= 9_007_199_254_740_991:
        raise ValueError('Storyboard base seed must be an exact nonnegative integer.')
    return {'storyboard': text, 'seed': seed, 'count': len(records)}


def evaluate_shot_inputs(kwargs, *, parameters=None):
    """Pure deterministic stage logic; safe for bounded resource inspection."""
    import math

    kwargs = {**{key: field['default'] for key, field in (parameters if parameters is not None else ShotInputs.params).items()
                 if 'default' in field and field.get('display') != 'output'}, **kwargs}

    records = _storyboard_records(kwargs.get('storyboard'))

    def whole(value, name, minimum=0):
        if type(value) is not int or not minimum <= value <= 9_007_199_254_740_991:
            raise ValueError(f'{name} must be a whole number of at least {minimum}.')
        return value

    def number(value, name, minimum=0, positive=False):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < minimum or positive and value == minimum):
            raise ValueError(f'{name} must be a finite positive value.' if positive else
                             f'{name} must be a finite value of at least {minimum}.')
        return float(value)

    index = whole(kwargs['index'], 'Shot index')
    if index >= len(records):
        raise ValueError('Shot index is outside this storyboard.')
    record = records[index]
    if isinstance(record, str):
        record = {'prompt': record}
    if not isinstance(record, dict) or not isinstance(record.get('prompt'), str) or not record['prompt'].strip():
        raise ValueError('The selected shot needs a nonempty prompt.')
    fps = number(kwargs['fps'], 'FPS', positive=True)
    frames = whole(kwargs['num_frames'], 'Frames', 1)
    if 'duration_seconds' in record:
        duration = number(record['duration_seconds'], 'Shot duration', positive=True)
        target = whole(max(1, round(number(duration * fps, 'Duration frames', positive=True))), 'Duration frames', 1)
        frames = whole(1 + ((target - 1 + 3) // 4) * 4, 'Frames', 1)
    seed = whole(record.get('seed', whole(kwargs['base_seed'], 'Base seed') + index), 'Shot seed')
    if kwargs['use_seed_override']:
        seed = whole(kwargs['seed_override'], 'Shot seed override')
    strength = number(record.get('conditioning_strength', kwargs['conditioning_strength']), 'Condition strength')
    if strength > 1:
        raise ValueError('Condition strength must be at most one.')
    return {
        'prompt': kwargs['prompt_override'].strip() or record['prompt'].strip(),
        'negative': kwargs['negative_prompt'], 'seed': seed,
        'title': str(record.get('title') or f'Shot {index + 1}'),
        'width_out': whole(kwargs['width'], 'Width', 1),
        'height_out': whole(kwargs['height'], 'Height', 1),
        'frames': frames, 'fps_out': fps, 'steps_out': whole(kwargs['steps'], 'Steps', 1),
        'guidance': number(kwargs['guidance_scale'], 'Guidance'),
        'secondary_guidance': number(kwargs['secondary_guidance_scale'], 'Low-noise guidance'),
        'condition_strength': strength,
    }


def evaluate_shot_prompt(kwargs):
    """Pure deterministic stage logic; safe for bounded resource inspection."""
    records = _storyboard_records(kwargs.get('storyboard'))
    index = kwargs.get('index', 0)
    if type(index) is not int or not 0 <= index < len(records):
        raise ValueError('Shot index is outside this storyboard.')
    record = records[index]
    prompt = record.get('prompt') if isinstance(record, dict) else record
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('The selected shot needs a nonempty prompt.')
    return {'prompt': str(kwargs.get('prompt_override') or '').strip() or prompt.strip()}



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


class OutpaintCanvas(NodeBase):
    """Pad prepared video frames without resizing and create matching masks."""

    label = "Prepare Video Canvas"
    category = "Video Conditioning"
    resizable = True
    params = {
        "video": {"label": "Source video", "display": "input", "type": "video", "required": True},
        "preparation": {"label": "Mask preparation", "type": "string", "default": "expand",
                        "options": ["prepared", "expand"]},
        "prepared_mask_file": {"label": "Prepared boundary mask", "type": "str", "display": "filebrowser",
                               "default": "", "required": False,
                               "requiredWhen": {"field": "preparation", "equals": "prepared"},
                               "fieldOptions": {"fileTypes": ["video"], "multiple": False}},
        "invert": {"label": "Invert prepared mask", "type": "bool", "default": False},
        "threshold": {"label": "Prepared mask threshold", "type": "int", "default": 127, "min": 0, "max": 255},
        "grow_pixels": {"label": "Grow prepared mask", "type": "int", "default": 0, "min": 0, "max": 256},
        "left": {"label": "Left border", "type": "int", "default": 0, "min": 0},
        "right": {"label": "Right border", "type": "int", "default": 0, "min": 0},
        "top": {"label": "Top border", "type": "int", "default": 0, "min": 0},
        "bottom": {"label": "Bottom border", "type": "int", "default": 0, "min": 0},
        "fill_color": {"label": "Canvas fill", "type": "string", "default": "#7f7f7f"},
        "canvas": {"label": "Expanded video", "display": "output", "type": "video"},
        "mask": {"label": "Boundary mask", "display": "output", "type": "video"},
        "width": {"label": "Canvas width", "display": "output", "type": "int"},
        "height": {"label": "Canvas height", "display": "output", "type": "int"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        frames = as_frames(kwargs.get("video"))
        if not frames:
            raise ValueError("Prepare Video Canvas requires source video frames.")
        if any(not isinstance(frame, Image.Image) or frame.mode != "RGB" for frame in frames):
            raise ValueError("Prepare Video Canvas needs prepared RGB frames; connect Normalize Video first.")
        source_width, source_height = frames[0].size
        if any(frame.size != (source_width, source_height) for frame in frames):
            raise ValueError("Prepare Video Canvas source frames must have the same dimensions.")
        preparation = kwargs.get('preparation', 'expand')
        if preparation not in ('prepared', 'expand'):
            raise ValueError('Mask preparation must be prepared or expand.')
        if preparation == 'prepared':
            # Read only the active input. An unused empty mask picker must not
            # block automatic expansion, and prepared inputs are never padded.
            from modules.Video.main import load_video_file
            loaded = load_video_file(kwargs.get('prepared_mask_file'), interrupted=lambda: self._interrupt)
            masks = AlignMask.execute(self, video=frames, mask=loaded['video'],
                                      invert=kwargs.get('invert', False), threshold=kwargs.get('threshold', 127),
                                      grow_pixels=kwargs.get('grow_pixels', 0))['output']
            return {'canvas': frames, 'mask': masks, 'width': source_width,
                    'height': source_height, 'frames': len(frames)}
        borders = {}
        for field in ("left", "right", "top", "bottom"):
            value = kwargs.get(field, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a nonnegative whole pixel count.")
            borders[field] = value
        width = source_width + borders["left"] + borders["right"]
        height = source_height + borders["top"] + borders["bottom"]
        try:
            color = ImageColor.getrgb(str(kwargs.get("fill_color") or "#7f7f7f"))[:3]
        except ValueError as error:
            raise ValueError("Canvas fill must be a CSS color name or hex color.") from error

        # RGB PIL storage uses four bytes per pixel. Budget both full output
        # collections, the L mask template and one source frame before any
        # allocation. The reserve follows the existing CPU cache policy; this
        # is an observed memory boundary, not a guessed model resolution limit.
        from modiff.hardware import system_memory_snapshot
        memory = system_memory_snapshot()
        available, total = memory.get("available_bytes"), memory.get("total_bytes")
        reserve = max(4 * 1024**3, int(total * .1)) if type(total) is int and total > 0 else 4 * 1024**3
        required = width * height * (8 * len(frames) + 1) + source_width * source_height * 4
        if type(available) is not int or required > max(0, available - reserve):
            raise MemoryError(
                f"Video canvas preparation needs {required} bytes plus the system RAM reserve; "
                "available memory is insufficient or unavailable."
            )

        mask = Image.new("L", (width, height), 255)
        left, top = borders["left"], borders["top"]
        ImageDraw.Draw(mask).rectangle(
            (left, top, left + source_width - 1, top + source_height - 1), fill=0,
        )
        canvases, masks = [], []
        for index, frame in enumerate(frames):
            if self._interrupt:
                raise InterruptedError("Video canvas preparation interrupted.")
            canvas = Image.new("RGB", (width, height), color)
            canvas.paste(frame, (left, top))
            canvases.append(canvas)
            masks.append(mask.convert("RGB"))
            self.progress(round((index + 1) / len(frames) * 100), phase="processing",
                          message=f"Preparing video canvas {index + 1}/{len(frames)}")
        return {"canvas": canvases, "mask": masks, "width": width, "height": height, "frames": len(frames)}


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
