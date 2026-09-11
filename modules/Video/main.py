# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
from modiff.NodeBase import NodeBase
from modiff.path_identifiers import resolve_runtime_input_path
from pathlib import Path
import logging
import math
from utils.paths import parse_filename
from utils.torch_utils import DEFAULT_DEVICE, DEVICE_LIST
from modiff.upscaler_contracts import real_esrgan_x2_model_selection

logger = logging.getLogger("modiff")

VIDEO_OPERATION_MODES = [
    "video_frame_extract",
    "frame_interpolation",
    "video_stitch",
    "video_trim",
    "video_reverse",
    "video_tile",
]
VIDEO_OPERATION_PIPELINE_CLASS = "BuiltinVideoOperationV1"
VIDEO_UPSCALE_MODE = "video_upscale"
VIDEO_UPSCALE_PIPELINE_CLASS = "SpandrelVideoUpscaleV1"
VIDEO_UPSCALE_MODEL_SELECTION = real_esrgan_x2_model_selection()
MAX_VIDEO_OPERATION_INPUTS = 16
MAX_VIDEO_OPERATION_FRAMES_PER_INPUT = 14_400
MAX_VIDEO_OPERATION_TOTAL_FRAMES = 57_600
MAX_VIDEO_OPERATION_PIXELS = 16_777_216
MAX_VIDEO_REVERSE_PIXEL_FRAMES = 251_658_240
MAX_VIDEO_INTERPOLATION_FPS = 120
MAX_VIDEO_INTERPOLATION_PIXEL_FRAMES = 251_658_240
MAX_EXTRACTED_FRAMES = 64
MAX_VIDEO_UPSCALE_FRAMES = 1_200
MAX_VIDEO_UPSCALE_INPUT_PIXELS = 4_194_304
MAX_VIDEO_UPSCALE_OUTPUT_PIXELS = 16_777_216


class Load(NodeBase):
    """
    Load a video from a file path.
    """

    label = "Load Video"
    category = "Video"
    resizable = True
    params = {
        "video": {
            "label": "Video",
            "display": "output",
            "type": "video",
        },
        "label": {
            "display": "ui_label",
            "value": "Load Video",
        },
        "file": {
            "label": False,
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {
                "fileTypes": ["video"],
                "multiple": False,
            },
        },
        "filename": {"label": "File Name", "display": "output", "type": "str"},
        "width": {"display": "output", "type": "int"},
        "height": {"display": "output", "type": "int"},
        "frames": {"display": "output", "type": "int"},
        "fps": {"label": "FPS", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        import imageio
        from PIL import Image

        file_value = kwargs["file"]
        file_value = file_value[0] if isinstance(file_value, list) and file_value else file_value
        if not file_value:
            raise ValueError("Load Video needs an existing video file.")
        file = resolve_runtime_input_path(file_value)
        logger.debug(f"Loading video from file: {file}")

        if not file.is_file():
            raise ValueError("Load Video needs an existing video file.")

        images = []
        reader = imageio.get_reader(str(file), "ffmpeg")
        meta = reader.get_meta_data()
        width = meta.get("size", (0, 0))[0]
        height = meta.get("size", (0, 0))[1]
        frames = reader.count_frames()
        fps = meta.get("fps", 0)

        for frame in reader:
            images.append(Image.fromarray(frame))
        reader.close()

        return {
            "filename": str(file),
            "video": images,
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
        }


class Export(NodeBase):
    """
    Save/Re-encode a video
    """

    label = "Export Video"
    category = "Video"
    resizable = True
    params = {
        "video": {"type": ["video", "str", "image"], "display": "input"},
        "filename": {
            "label": "File",
            "type": "str",
            "default": "{PATH:videos}/MoDiff_{HASH:6}.mp4",
        },
        # "codec": { "type": "str", "options": ["libx264", "vp9"], "default": "libx264" },
        # Match the app's other delivery exporters. Quality 5 produces a very
        # small, visibly blocky H.264 file for native 480p diffusion output.
        "quality": {"display": "slider", "type": "int", "min": 1, "max": 10, "default": 8},
        "fps": {"label": "FPS", "type": "float", "default": 24, "min": 1, "max": 240, "step": 0.01},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "file": {"type": "video", "display": "output"},
        "width": {"display": "output", "type": "int"},
        "height": {"display": "output", "type": "int"},
        "frames": {"display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import imageio
        import numpy as np
        from PIL import Image

        video = kwargs["video"]
        filename = kwargs.get("filename", "{PATH:videos}/MoDiff_{HASH:6}.mp4")
        quality = kwargs.get("quality", 8)
        fps = kwargs.get("fps", 24)

        def frame_to_array(frame):
            import torch

            if isinstance(frame, Image.Image):
                return np.array(frame.convert("RGB"))
            if isinstance(frame, torch.Tensor):
                tensor = frame.detach().float().cpu()
                if tensor.ndim == 4:
                    tensor = tensor.squeeze(0)
                if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                    tensor = tensor.permute(1, 2, 0)
                array = tensor.numpy()
                if array.dtype.kind == "f":
                    array = np.clip(array, 0, 1) * 255
                return array.astype(np.uint8)
            if isinstance(frame, np.ndarray):
                array = frame
                if array.dtype.kind == "f":
                    array = np.clip(array, 0, 1) * 255
                if array.ndim == 3 and array.shape[0] in (1, 3, 4):
                    array = np.moveaxis(array, 0, -1)
                return array.astype(np.uint8)
            return np.array(frame)

        def frames_shape(video_data):
            import torch

            if isinstance(video_data, torch.Tensor):
                video_data = tensor_to_frames(video_data)
            if isinstance(video_data, np.ndarray) and video_data.ndim == 4:
                video_data = [video_data[index] for index in range(video_data.shape[0])]
            if not isinstance(video_data, list) or len(video_data) == 0:
                return 0, 0, 0
            array = frame_to_array(video_data[0])
            height, width = array.shape[:2]
            return width, height, len(video_data)

        def tensor_to_frames(tensor):
            tensor = tensor.detach().cpu()
            if tensor.ndim == 5:
                tensor = tensor.squeeze(0)
            if tensor.ndim == 4:
                return [tensor[index] for index in range(tensor.shape[0])]
            return [tensor]

        def save_video(video_data):
            if video_data is None or (isinstance(video_data, list) and not video_data):
                raise ValueError("Export Video needs non-empty video frames.")

            parsed_filename = parse_filename(filename)
            Path(parsed_filename).parent.mkdir(parents=True, exist_ok=True)
            width, height, frame_count = 0, 0, 0

            if isinstance(video_data, str):
                reader = imageio.get_reader(str(resolve_runtime_input_path(video_data)))
                meta = reader.get_meta_data()
                width, height = meta.get("size", (0, 0))
                try:
                    frame_count = reader.count_frames()
                except Exception:
                    frame_count = 0
                writer = imageio.get_writer(
                    parsed_filename,
                    fps=fps,
                    quality=quality,
                    codec="libx264",
                )
                for frame in reader:
                    writer.append_data(frame)
                reader.close()
                writer.close()

            elif isinstance(video_data, list):
                width, height, frame_count = frames_shape(video_data)

                writer = imageio.get_writer(
                    parsed_filename,
                    fps=fps,
                    quality=quality,
                    codec="libx264",
                )
                for frame in video_data:
                    writer.append_data(frame_to_array(frame))
                writer.close()

            elif isinstance(video_data, np.ndarray) and video_data.ndim == 4:
                return save_video([video_data[index] for index in range(video_data.shape[0])])

            else:
                import torch

                if isinstance(video_data, torch.Tensor):
                    return save_video(tensor_to_frames(video_data))
                raise ValueError("Export Video received an unsupported video shape or type.")

            destination = Path(parsed_filename)
            if not destination.is_file() or destination.stat().st_size == 0:
                raise RuntimeError("Export Video did not produce a non-empty encoded file.")
            # The existing encoder may resize odd dimensions to a macroblock
            # boundary. Report the retained file, not the pre-encoder frames.
            from modiff.media_assets import probe_video_file

            encoded = probe_video_file(destination)
            return str(parsed_filename), encoded["width"], encoded["height"], encoded["frame_count"]

        # Diffusers VideoProcessor returns NumPy output as B,F,H,W,C. Preserve
        # the batch as separate clips, just like its existing nested PIL output.
        if isinstance(video, np.ndarray) and video.ndim == 5:
            video = [list(clip) for clip in video]

        if isinstance(video, list) and video and isinstance(video[0], list):
            files = []
            width = height = frames = 0
            for item in video:
                file, width, height, frames = save_video(item)
                files.append(file)
            return {"file": files, "width": width, "height": height, "frames": frames}

        file, width, height, frames = save_video(video)

        return {
            "file": file,
            "width": width,
            "height": height,
            "frames": frames,
        }


def _pil_frames(value):
    """Normalize any supported in-memory video value to RGB PIL frames."""
    import numpy as np
    from PIL import Image

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is part of the runtime
        torch = None

    if value is None or (isinstance(value, str) and not value):
        return []
    if isinstance(value, dict) and (value.get("path") or value.get("file")):
        value = value.get("path") or value.get("file")
    if isinstance(value, str):
        import imageio

        reader = imageio.get_reader(str(resolve_runtime_input_path(value)), "ffmpeg")
        try:
            return [Image.fromarray(frame).convert("RGB") for frame in reader]
        finally:
            reader.close()
    if torch is not None and isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
        if tensor.ndim == 5:
            tensor = tensor.squeeze(0)
        value = [tensor[index] for index in range(tensor.shape[0])] if tensor.ndim == 4 else [tensor]
    if isinstance(value, np.ndarray) and value.ndim == 4:
        value = list(value)
    if not isinstance(value, list):
        value = [value]

    frames = []
    for frame in value:
        if isinstance(frame, Image.Image):
            frames.append(frame.convert("RGB"))
            continue
        if torch is not None and isinstance(frame, torch.Tensor):
            frame = frame.detach().float().cpu()
            if frame.ndim == 4:
                frame = frame.squeeze(0)
            if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
                frame = frame.permute(1, 2, 0)
            frame = frame.numpy()
        array = np.asarray(frame)
        if array.dtype.kind == "f":
            array = np.clip(array, 0, 1) * 255
        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        frames.append(Image.fromarray(array.astype(np.uint8)).convert("RGB"))
    return frames


class MaskedComposite(NodeBase):
    """Composite generated video only inside a white mask sequence."""

    label = "Masked Video Composite"
    category = "Video"
    resizable = True
    params = {
        "source": {"label": "Source Video", "display": "input", "type": ["video", "str"]},
        "generated": {"label": "Generated Video", "display": "input", "type": ["video", "str"]},
        "mask": {"label": "White Generate Mask", "display": "input", "type": ["video", "image"]},
        "feather": {"label": "Edge Feather", "type": "float", "default": 0.0, "min": 0, "max": 128, "step": 1},
        "output": {"label": "Video", "display": "output", "type": "video"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        from PIL import Image, ImageFilter

        source = _pil_frames(kwargs.get("source"))
        generated = _pil_frames(kwargs.get("generated"))
        masks = _pil_frames(kwargs.get("mask"))
        if not source or not generated or not masks:
            raise ValueError("Masked Video Composite needs source, generated, and mask frames.")
        if len(source) != len(generated):
            raise ValueError(
                f"Masked Video Composite frame mismatch: source has {len(source)} frames and generated has "
                f"{len(generated)}."
            )
        if len(masks) == 1:
            masks = masks * len(source)
        elif len(masks) != len(source):
            raise ValueError(
                f"Masked Video Composite mask mismatch: expected 1 or {len(source)} masks; received {len(masks)}."
            )

        feather = max(0.0, float(kwargs.get("feather") or 0.0))
        output = []
        for source_frame, generated_frame, mask_frame in zip(source, generated, masks):
            size = source_frame.size
            if generated_frame.size != size:
                generated_frame = generated_frame.resize(size, Image.Resampling.LANCZOS)
            mask_image = mask_frame.convert("L")
            if mask_image.size != size:
                mask_image = mask_image.resize(size, Image.Resampling.LANCZOS)
            if feather > 0:
                mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=feather))
            output.append(Image.composite(generated_frame, source_frame, mask_image))
        return {"output": output, "frames": len(output)}


class TemporalCleanPlate(NodeBase):
    """Build a person-free plate sequence by interpolating two clean frames."""

    label = "Temporal Clean Plate"
    category = "Video"
    resizable = True
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video", "str"]},
        "start_index": {"label": "Clean Start Frame", "type": "int", "default": 0, "min": -100000},
        "end_index": {"label": "Clean End Frame", "type": "int", "default": -1, "min": -100000},
        "easing": {
            "label": "Interpolation",
            "type": "string",
            "options": ["smoothstep", "linear"],
            "default": "smoothstep",
        },
        "output": {"label": "Plate Video", "display": "output", "type": "video"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        from PIL import Image

        frames = _pil_frames(kwargs.get("video"))
        if not frames:
            raise ValueError("Temporal Clean Plate needs a non-empty video.")

        def normalize_index(value, fallback):
            index = int(value if value is not None else fallback)
            if index < 0:
                index += len(frames)
            return max(0, min(len(frames) - 1, index))

        start_index = normalize_index(kwargs.get("start_index"), 0)
        end_index = normalize_index(kwargs.get("end_index"), len(frames) - 1)
        start = frames[start_index]
        end = frames[end_index]
        if end.size != start.size:
            end = end.resize(start.size, Image.Resampling.LANCZOS)
        easing = str(kwargs.get("easing") or "smoothstep")
        output = []
        denominator = max(1, len(frames) - 1)
        for index in range(len(frames)):
            alpha = index / denominator
            if easing == "smoothstep":
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            elif easing != "linear":
                raise ValueError(f"Unsupported clean-plate interpolation {easing!r}.")
            output.append(Image.blend(start, end, alpha))
        return {"output": output, "frames": len(output)}


class ExtendCleanPlate(NodeBase):
    """Copy a clean background strip into a neighboring occluded region."""

    label = "Extend Video Clean Plate"
    category = "Video"
    resizable = True
    params = {
        "video": {"label": "Plate Video", "display": "input", "type": ["video", "str"]},
        "boundary_x": {"label": "Clean Boundary X", "type": "int", "default": 535, "min": 1, "max": 16384},
        "extend_left": {"label": "Extend Left", "type": "int", "default": 20, "min": 1, "max": 2048},
        "mode": {"label": "Extension", "type": "string", "options": ["copy", "mirror"], "default": "copy"},
        "top": {"label": "Top", "type": "int", "default": 0, "min": 0, "max": 16384},
        "bottom": {"label": "Bottom", "type": "int", "default": 230, "min": 1, "max": 16384},
        "output": {"label": "Extended Plate", "display": "output", "type": "video"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        from PIL import Image

        frames = _pil_frames(kwargs.get("video"))
        if not frames:
            raise ValueError("Extend Video Clean Plate needs a non-empty video.")
        boundary = int(kwargs.get("boundary_x", 535))
        extend = int(kwargs.get("extend_left", 20))
        top = int(kwargs.get("top", 0))
        bottom = int(kwargs.get("bottom", 230))
        mode = str(kwargs.get("mode") or "copy")
        if mode not in {"copy", "mirror"}:
            raise ValueError(f"Unsupported clean-plate extension {mode!r}.")
        output = []
        for frame in frames:
            width, height = frame.size
            if not 0 < boundary < width:
                raise ValueError(f"Clean boundary X must be inside the frame; received {boundary} for width {width}.")
            actual_extend = min(extend, boundary, width - boundary)
            actual_top = max(0, min(height - 1, top))
            actual_bottom = max(actual_top + 1, min(height, bottom))
            result = frame.copy()
            clean_strip = frame.crop((boundary, actual_top, boundary + actual_extend, actual_bottom))
            if mode == "mirror":
                clean_strip = clean_strip.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            result.paste(clean_strip, (boundary - actual_extend, actual_top))
            output.append(result)
        return {"output": output, "frames": len(output)}


class Compose(NodeBase):
    """Compose two to six model-agnostic clips into one continuous timeline."""

    label = "Compose Video"
    category = "Video"
    resizable = True
    params = {
        # Keep these explicit: MoDiff's static AST registry intentionally does
        # not execute dict comprehensions while discovering node contracts.
        "clip_1": {
            "label": "Clip 1",
            "display": "input",
            "type": ["video_collection", "video", "str"],
            "required": True,
        },
        "clip_2": {
            "label": "Clip 2",
            "display": "input",
            "type": ["video_collection", "video", "str"],
            "required": False,
        },
        "clip_3": {
            "label": "Clip 3",
            "display": "input",
            "type": ["video_collection", "video", "str"],
            "required": False,
        },
        "clip_4": {
            "label": "Clip 4",
            "display": "input",
            "type": ["video_collection", "video", "str"],
            "required": False,
        },
        "clip_5": {
            "label": "Clip 5",
            "display": "input",
            "type": ["video_collection", "video", "str"],
            "required": False,
        },
        "clip_6": {
            "label": "Clip 6",
            "display": "input",
            "type": ["video_collection", "video", "str"],
            "required": False,
        },
        "transition_seconds": {
            "label": "Crossfade",
            "type": "float",
            "default": 0.35,
            "min": 0,
            "max": 2,
            "step": 0.05,
        },
        "fps": {"label": "FPS", "type": "float", "default": 16, "min": 1, "max": 120, "step": 0.01},
        "video": {"display": "output", "type": "video"},
        "frames": {"display": "output", "type": "int"},
        "duration_seconds": {"display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        from PIL import Image

        clips = []
        for index in range(1, 7):
            value = kwargs.get(f"clip_{index}")
            if isinstance(value, list) and value and isinstance(value[0], list):
                clips.extend(_pil_frames(item) for item in value)
            else:
                clips.append(_pil_frames(value))
        clips = [clip for clip in clips if clip]
        if not clips:
            raise ValueError("Compose Video needs at least one non-empty clip.")
        fps = float(kwargs.get("fps") or 16)
        fade_frames = max(0, int(round(float(kwargs.get("transition_seconds") or 0) * fps)))
        target_size = clips[0][0].size
        clips = [
            [
                frame.resize(target_size, Image.Resampling.LANCZOS) if frame.size != target_size else frame
                for frame in clip
            ]
            for clip in clips
        ]
        output = list(clips[0])
        for clip in clips[1:]:
            overlap = min(fade_frames, len(output), len(clip))
            if overlap:
                start = len(output) - overlap
                for index in range(overlap):
                    alpha = (index + 1) / (overlap + 1)
                    output[start + index] = Image.blend(output[start + index], clip[index], alpha)
            output.extend(clip[overlap:])
        return {"video": output, "frames": len(output), "duration_seconds": len(output) / fps}


class LyricOverlay(NodeBase):
    """Render an authored LRC timeline over any video model's frames."""

    label = "Timed Lyric Overlay"
    category = "Video"
    resizable = True
    params = {
        "video": {"display": "input", "type": ["video", "str"]},
        "lrc": {"label": "Timed Lyrics (LRC)", "display": "textarea", "type": "text", "default": ""},
        "fps": {"label": "FPS", "type": "float", "default": 16, "min": 1, "max": 120, "step": 0.01},
        "font_size": {"label": "Font Size", "type": "int", "default": 42, "min": 12, "max": 160},
        "bottom_margin": {"label": "Bottom Margin", "type": "int", "default": 54, "min": 0, "max": 400},
        "output": {"display": "output", "type": "video"},
    }

    @staticmethod
    def _timeline(text):
        import re

        entries = []
        for line in str(text or "").splitlines():
            match = re.match(r"\s*\[(\d+):(\d+(?:\.\d+)?)\]\s*(.+?)\s*$", line)
            if match:
                entries.append((int(match.group(1)) * 60 + float(match.group(2)), match.group(3)))
        return sorted(entries)

    def execute(self, **kwargs):
        from PIL import ImageDraw, ImageFont

        frames = _pil_frames(kwargs.get("video"))
        timeline = self._timeline(kwargs.get("lrc"))
        if not frames or not timeline:
            raise ValueError("Timed Lyric Overlay needs video frames and at least one [mm:ss] lyric line.")
        fps = float(kwargs.get("fps") or 16)
        font_size = int(kwargs.get("font_size") or 42)
        margin = int(kwargs.get("bottom_margin") or 54)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
        output = []
        for frame_index, source in enumerate(frames):
            timestamp = frame_index / fps
            active = next((text for start, text in reversed(timeline) if start <= timestamp), "")
            frame = source.copy()
            if active:
                draw = ImageDraw.Draw(frame)
                box = draw.textbbox((0, 0), active, font=font, stroke_width=2)
                x = max(16, (frame.width - (box[2] - box[0])) // 2)
                y = max(16, frame.height - margin - (box[3] - box[1]))
                draw.text((x, y), active, font=font, fill="white", stroke_width=3, stroke_fill="black")
            output.append(frame)
        return {"output": output}


class ExportWithAudio(NodeBase):
    """Export composed frames with generated or loaded audio in one MP4."""

    label = "Export Video with Audio"
    category = "Video"
    resizable = True
    params = {
        "video": {"display": "input", "type": ["video", "str"]},
        "audio": {"display": "input", "type": ["audio", "str"]},
        "filename": {"label": "File", "type": "str", "default": "{PATH:videos}/MoDiff_{HASH:6}.mp4"},
        "fps": {"label": "FPS", "type": "float", "default": 16, "min": 1, "max": 120, "step": 0.01},
        "quality": {"display": "slider", "type": "int", "min": 1, "max": 10, "default": 8},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "file": {"type": "video", "display": "output"},
        "frames": {"display": "output", "type": "int"},
        "duration_seconds": {"display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        import imageio
        import numpy as np
        import subprocess
        from scipy.io import wavfile
        from imageio_ffmpeg import get_ffmpeg_exe

        frames = _pil_frames(kwargs.get("video"))
        if not frames:
            raise ValueError("Export Video with Audio needs non-empty video frames.")
        fps = float(kwargs.get("fps") or 16)
        destination = Path(parse_filename(kwargs.get("filename") or "{PATH:videos}/MoDiff_{HASH:6}.mp4"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        silent_path = destination.with_suffix(".silent.mp4")
        audio_path = destination.with_suffix(".audio.wav")

        audio = kwargs.get("audio")
        if isinstance(audio, str):
            audio_path = resolve_runtime_input_path(audio)
        else:
            data = audio.get("samples", audio.get("audio")) if isinstance(audio, dict) else audio
            sample_rate = int(audio.get("sample_rate", 48000)) if isinstance(audio, dict) else 48000
            samples = np.asarray(data, dtype=np.float32)
            while samples.ndim > 2:
                samples = samples[0]
            if samples.ndim == 2 and samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]:
                samples = samples.T
            wavfile.write(audio_path, sample_rate, (np.clip(samples, -1, 1) * 32767).astype(np.int16))

        writer = imageio.get_writer(silent_path, fps=fps, quality=int(kwargs.get("quality") or 8), codec="libx264")
        try:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()
        subprocess.run(
            [
                get_ffmpeg_exe(),
                "-y",
                "-v",
                "error",
                "-i",
                str(silent_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "256k",
                "-shortest",
                str(destination),
            ],
            check=True,
        )
        silent_path.unlink(missing_ok=True)
        if audio_path.parent == destination.parent and audio_path.name.endswith(".audio.wav"):
            audio_path.unlink(missing_ok=True)
        return {"file": str(destination), "frames": len(frames), "duration_seconds": len(frames) / fps}


def _video_collection(value):
    """Normalize a collection of clips without confusing one clip with many."""
    if value in (None, ""):
        return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return [_pil_frames(value)]
    if not value:
        return []
    if isinstance(value[0], list):
        return [_pil_frames(item) for item in value if item]
    if isinstance(value[0], str) and len(value) > 1:
        return [_pil_frames(item) for item in value if item]
    return [_pil_frames(value)]


def _parse_numbers(value, *, cast=float):
    if value in (None, ""):
        return []
    if isinstance(value, str):
        values = value.replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    return [cast(item) for item in values if str(item).strip()]


def _concatenate_clips(clips, *, transition_frames=0):
    from PIL import Image

    clips = [list(clip) for clip in clips if clip]
    if not clips:
        return []
    target_size = clips[0][0].size
    normalized = [
        [frame.resize(target_size, Image.Resampling.LANCZOS) if frame.size != target_size else frame for frame in clip]
        for clip in clips
    ]
    output = list(normalized[0])
    for clip in normalized[1:]:
        overlap = min(max(0, int(transition_frames)), len(output), len(clip))
        if overlap:
            start = len(output) - overlap
            for index in range(overlap):
                alpha = (index + 1) / (overlap + 1)
                output[start + index] = Image.blend(output[start + index], clip[index], alpha)
        output.extend(clip[overlap:])
    return output


class FrameExtract(NodeBase):
    """Extract ordered frames by boundary, index, timecode, or interval."""

    label = "Extract Video Frames"
    category = "Video"
    resizable = True
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video_asset", "video", "str"]},
        "mode": {
            "label": "Selection",
            "type": "string",
            "options": ["first", "last", "first_last", "indices", "timecodes", "every_n"],
            "default": "first_last",
        },
        "indices": {"label": "Frame Indices", "type": "string", "default": "0,-1"},
        "timecodes": {"label": "Times (seconds)", "type": "string", "default": "0"},
        "every_n": {"label": "Every N Frames", "type": "int", "default": 16, "min": 1},
        "fps": {"label": "FPS", "type": "float", "default": 16, "min": 0.01},
        "frames": {"label": "Frames", "display": "output", "type": "image"},
        "selected_indices": {"label": "Indices", "display": "output", "type": "collection"},
        "timestamps": {"label": "Timestamps", "display": "output", "type": "collection"},
    }

    def execute(self, **kwargs):
        value = kwargs.get("video")
        file_asset = None
        if (isinstance(value, dict) and (value.get("path") or value.get("file"))) or isinstance(value, str):
            from modiff.media_assets import coerce_video_asset

            file_asset = coerce_video_asset(value)
            frame_count = int(file_asset["frame_count"])
        else:
            frames = _pil_frames(value)
            frame_count = len(frames)
        if frame_count < 1:
            raise ValueError("Extract Video Frames needs a non-empty video.")
        mode = str(kwargs.get("mode") or "first_last")
        if mode == "first":
            indices = [0]
        elif mode == "last":
            indices = [frame_count - 1]
        elif mode == "first_last":
            indices = [0, frame_count - 1]
        elif mode == "indices":
            indices = _parse_numbers(kwargs.get("indices"), cast=int)
        elif mode == "timecodes":
            fps = float(kwargs.get("fps") or 16)
            indices = [round(value * fps) for value in _parse_numbers(kwargs.get("timecodes"), cast=float)]
        elif mode == "every_n":
            step = max(1, int(kwargs.get("every_n") or 1))
            indices = list(range(0, frame_count, step))
        else:
            raise ValueError(f"Unsupported frame selection mode {mode!r}.")
        normalized = []
        for index in indices:
            index = int(index)
            if index < 0:
                index += frame_count
            if 0 <= index < frame_count and index not in normalized:
                normalized.append(index)
        if not normalized:
            raise ValueError("The requested frame selection is outside this video.")
        if len(normalized) > MAX_EXTRACTED_FRAMES:
            raise ValueError(f"Extract Video Frames accepts at most {MAX_EXTRACTED_FRAMES} output frames.")
        fps = float(file_asset["fps"] if file_asset and file_asset.get("fps") else kwargs.get("fps") or 16)
        if file_asset:
            import imageio.v2 as imageio
            from PIL import Image

            reader = imageio.get_reader(file_asset["path"], "ffmpeg")
            try:
                selected_frames = [Image.fromarray(reader.get_data(index)).convert("RGB") for index in normalized]
            finally:
                reader.close()
        else:
            selected_frames = [frames[index] for index in normalized]
        return {
            "frames": selected_frames,
            "selected_indices": normalized,
            "timestamps": [index / fps for index in normalized],
        }


class Trim(NodeBase):
    """Trim a video by frame range or seconds."""

    label = "Trim Video"
    category = "Video"
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video", "str"]},
        "range_mode": {"label": "Range", "type": "string", "options": ["frames", "seconds"], "default": "seconds"},
        "start": {"label": "Start", "type": "float", "default": 0, "min": 0},
        "end": {"label": "End (0 = end)", "type": "float", "default": 0, "min": 0},
        "fps": {"label": "FPS", "type": "float", "default": 16, "min": 0.01},
        "output": {"label": "Video", "display": "output", "type": "video"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        frames = _pil_frames(kwargs.get("video"))
        fps = float(kwargs.get("fps") or 16)
        scale = fps if kwargs.get("range_mode") == "seconds" else 1
        start = max(0, int(round(float(kwargs.get("start") or 0) * scale)))
        end_value = float(kwargs.get("end") or 0)
        end = int(round(end_value * scale)) if end_value > 0 else len(frames)
        if end < start:
            raise ValueError("Trim Video end must not be before start.")
        output = frames[start : min(end, len(frames))]
        return {"output": output, "frames": len(output), "duration_seconds": len(output) / fps}


class Concatenate(NodeBase):
    """Concatenate an arbitrary ordered clip collection with optional crossfades."""

    label = "Concatenate Videos"
    category = "Video"
    params = {
        "clips": {"label": "Clips", "display": "input", "type": ["video_collection", "collection", "video"]},
        "transition_seconds": {"label": "Crossfade", "type": "float", "default": 0, "min": 0},
        "fps": {"label": "FPS", "type": "float", "default": 16, "min": 0.01},
        "output": {"label": "Video", "display": "output", "type": "video"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        clips = _video_collection(kwargs.get("clips"))
        if not clips:
            raise ValueError("Concatenate Videos needs at least one clip.")
        fps = float(kwargs.get("fps") or 16)
        transition = round(max(0.0, float(kwargs.get("transition_seconds") or 0)) * fps)
        output = _concatenate_clips(clips, transition_frames=transition)
        return {"output": output, "frames": len(output), "duration_seconds": len(output) / fps}


class StackTile(NodeBase):
    """Arrange a collection of videos into a synchronized video wall."""

    label = "Stack / Tile Videos"
    category = "Video"
    params = {
        "videos": {"label": "Videos", "display": "input", "type": ["video_collection", "collection"]},
        "columns": {"label": "Columns", "type": "int", "default": 2, "min": 1},
        "sync": {
            "label": "Length",
            "type": "string",
            "options": ["shortest", "longest_hold"],
            "default": "longest_hold",
        },
        "gap": {"label": "Gap", "type": "int", "default": 0, "min": 0},
        "background": {"label": "Background", "type": "string", "default": "black"},
        "output": {"label": "Video", "display": "output", "type": "video"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        from math import ceil
        from PIL import Image, ImageColor

        clips = _video_collection(kwargs.get("videos"))
        if not clips:
            raise ValueError("Stack / Tile Videos needs at least one clip.")
        width, height = clips[0][0].size
        columns = max(1, int(kwargs.get("columns") or 1))
        rows = ceil(len(clips) / columns)
        gap = max(0, int(kwargs.get("gap") or 0))
        count = min(map(len, clips)) if kwargs.get("sync") == "shortest" else max(map(len, clips))
        output = []
        for frame_index in range(count):
            canvas = Image.new(
                "RGB",
                (columns * width + max(0, columns - 1) * gap, rows * height + max(0, rows - 1) * gap),
                ImageColor.getrgb(str(kwargs.get("background") or "black")),
            )
            for clip_index, clip in enumerate(clips):
                source = clip[min(frame_index, len(clip) - 1)]
                if source.size != (width, height):
                    source = source.resize((width, height), Image.Resampling.LANCZOS)
                left = (clip_index % columns) * (width + gap)
                top = (clip_index // columns) * (height + gap)
                canvas.paste(source, (left, top))
            output.append(canvas)
        return {"output": output, "frames": len(output)}


class Reverse(NodeBase):
    """Reverse frame order, optionally excluding duplicate endpoints for looping."""

    label = "Reverse Video"
    category = "Video"
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video", "str"]},
        "exclude_endpoints": {"label": "Exclude Endpoints", "type": "bool", "default": False},
        "output": {"label": "Video", "display": "output", "type": "video"},
    }

    def execute(self, **kwargs):
        frames = _pil_frames(kwargs.get("video"))
        output = list(reversed(frames[1:-1] if kwargs.get("exclude_endpoints") and len(frames) > 2 else frames))
        return {"output": output}


class Crossfade(NodeBase):
    """Join two clips with a frame-accurate crossfade."""

    label = "Crossfade Videos"
    category = "Video"
    params = {
        "first": {"label": "First", "display": "input", "type": ["video", "str"]},
        "second": {"label": "Second", "display": "input", "type": ["video", "str"]},
        "duration_seconds": {"label": "Duration", "type": "float", "default": 0.35, "min": 0},
        "fps": {"label": "FPS", "type": "float", "default": 16, "min": 0.01},
        "output": {"label": "Video", "display": "output", "type": "video"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        fps = float(kwargs.get("fps") or 16)
        output = _concatenate_clips(
            [_pil_frames(kwargs.get("first")), _pil_frames(kwargs.get("second"))],
            transition_frames=round(max(0.0, float(kwargs.get("duration_seconds") or 0)) * fps),
        )
        if not output:
            raise ValueError("Crossfade Videos needs two non-empty clips.")
        return {"output": output, "frames": len(output)}


class FirstLastSegmentBuilder(NodeBase):
    """Turn ordered keyframes into deterministic first/last-frame generation jobs."""

    label = "Build First / Last Segments"
    category = "Video"
    params = {
        "keyframes": {"label": "Keyframes", "display": "input", "type": "image"},
        "prompts": {"label": "Segment Prompts", "display": "textarea", "type": "text", "default": "[]"},
        "settings": {"label": "Shared Settings (JSON)", "display": "textarea", "type": "text", "default": "{}"},
        "jobs": {"label": "Segment Jobs", "display": "output", "type": "collection"},
        "count": {"label": "Segments", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import json

        keyframes = kwargs.get("keyframes")
        keyframes = keyframes if isinstance(keyframes, list) else [keyframes] if keyframes is not None else []
        if len(keyframes) < 2:
            raise ValueError("Build First / Last Segments needs at least two keyframes.")
        raw_prompts = kwargs.get("prompts")
        if isinstance(raw_prompts, str):
            text = raw_prompts.strip()
            if text.startswith("["):
                prompts = json.loads(text)
            else:
                prompts = text.splitlines()
        else:
            prompts = list(raw_prompts or [])
        settings = kwargs.get("settings")
        settings = json.loads(settings or "{}") if isinstance(settings, str) else dict(settings or {})
        if not isinstance(settings, dict):
            raise ValueError("Shared Settings must be a JSON object.")
        jobs = []
        for index in range(len(keyframes) - 1):
            jobs.append(
                {
                    "index": index,
                    "first_frame": keyframes[index],
                    "last_frame": keyframes[index + 1],
                    "prompt": str(prompts[index]) if index < len(prompts) else "",
                    "settings": dict(settings),
                }
            )
        return {"jobs": jobs, "count": len(jobs)}


class KeyframeChain(NodeBase):
    """Assemble clips collected from a loop over first/last-frame segment jobs."""

    label = "Assemble Keyframe Chain"
    category = "Video"
    params = {
        "clips": {"label": "Generated Clips", "display": "input", "type": ["video_collection", "collection"]},
        "boundary": {
            "label": "Boundary",
            "type": "string",
            "options": ["keep", "drop_duplicate", "crossfade"],
            "default": "drop_duplicate",
        },
        "crossfade_seconds": {"label": "Crossfade", "type": "float", "default": 0.2, "min": 0},
        "fps": {"label": "FPS", "type": "float", "default": 16, "min": 0.01},
        "output": {"label": "Video", "display": "output", "type": "video"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        clips = _video_collection(kwargs.get("clips"))
        if not clips:
            raise ValueError("Assemble Keyframe Chain needs generated clips.")
        boundary = str(kwargs.get("boundary") or "drop_duplicate")
        if boundary == "drop_duplicate":
            clips = [clips[0], *[clip[1:] if len(clip) > 1 else [] for clip in clips[1:]]]
            transition = 0
        elif boundary == "crossfade":
            transition = round(float(kwargs.get("crossfade_seconds") or 0) * float(kwargs.get("fps") or 16))
        elif boundary == "keep":
            transition = 0
        else:
            raise ValueError(f"Unsupported keyframe boundary policy {boundary!r}.")
        output = _concatenate_clips(clips, transition_frames=transition)
        return {"output": output, "frames": len(output)}


class ExportAsset(NodeBase):
    """Stream video frames to a retained file-backed asset."""

    label = "Export Retained Video Asset"
    category = "Video"
    resizable = True
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video_asset", "video", "str"]},
        "fps": {"label": "FPS", "type": "float", "default": 16, "min": 1, "max": 120},
        "quality": {"label": "Quality", "type": "int", "default": 8, "min": 1, "max": 10},
        "pin": {"label": "Protect From Cleanup", "type": "bool", "default": False},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "asset": {"label": "Video Asset", "display": "output", "type": "video_asset"},
        "file": {"label": "File", "display": "output", "type": "video"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        import imageio
        import numpy as np
        from modiff.media_assets import (
            allocate_video_path,
            coerce_video_asset,
            current_task_id,
            register_derived_video_asset,
            register_video_asset,
            run_ffmpeg,
        )

        video = kwargs.get("video")
        task_id = current_task_id()
        asset_id, destination = allocate_video_path(task_id=task_id)
        fps = float(kwargs.get("fps") or 16)
        if isinstance(video, (str, dict)):
            source = coerce_video_asset(video)
            quality = int(kwargs.get("quality") or 8)
            run_ffmpeg(
                [
                    "-i",
                    source["path"],
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-vf",
                    f"fps={fps}",
                    "-c:v",
                    "libx264",
                    "-crf",
                    str(max(12, 32 - quality * 2)),
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                ],
                destination,
            )
            asset = register_derived_video_asset(
                destination,
                asset_id=asset_id,
                task_id=task_id,
                source_assets=[source],
                operation="retain",
                pinned=bool(kwargs.get("pin", False)),
            )
            return {"asset": asset, "file": str(destination), "duration_seconds": asset["duration_seconds"]}

        frames = _pil_frames(video)
        if not frames:
            raise ValueError("Export Retained Video Asset needs non-empty video frames.")
        writer = imageio.get_writer(
            destination,
            fps=fps,
            quality=int(kwargs.get("quality") or 8),
            codec="libx264",
        )
        try:
            for index, frame in enumerate(frames):
                writer.append_data(np.asarray(frame.convert("RGB")))
                if index % max(1, len(frames) // 100) == 0:
                    self.progress(index / len(frames), phase="encoding", message="Writing retained video")
        finally:
            writer.close()
        width, height = frames[0].size
        asset = register_video_asset(
            destination,
            asset_id=asset_id,
            task_id=task_id,
            width=width,
            height=height,
            fps=fps,
            frame_count=len(frames),
            temporary=True,
            pinned=bool(kwargs.get("pin", False)),
        )
        return {"asset": asset, "file": str(destination), "duration_seconds": asset["duration_seconds"]}


def _file_asset_collection(value):
    from modiff.media_assets import coerce_video_asset

    values = list(value) if isinstance(value, (list, tuple)) else [value]
    assets = [coerce_video_asset(item) for item in values if item not in (None, "")]
    if not assets:
        raise ValueError("At least one retained video asset or file path is required.")
    return assets


def _video_filter(asset, label, *, width, height, fps, duration=None):
    # `xfade` rejects inputs whose filter-link frame rate is unspecified.  The
    # source MP4 can be perfectly CFR while a preceding `xfade` link still
    # reports 1/0, so normalize both the rate and time base explicitly.
    fps_text = f"{float(fps):.12g}"
    expression = (
        f"[{label}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"settb=expr=1/{fps_text},setpts=PTS-STARTPTS,fps={fps_text}"
    )
    if duration is not None:
        expression += f",tpad=stop_mode=clone:stop_duration={max(0.0, duration)}"
    return expression


def _derived_asset_result(destination, asset_id, sources, operation, *, pin=False):
    from modiff.media_assets import current_task_id, register_derived_video_asset

    asset = register_derived_video_asset(
        destination,
        asset_id=asset_id,
        task_id=current_task_id(),
        source_assets=sources,
        operation=operation,
        pinned=pin,
    )
    return {
        "asset": asset,
        "file": str(destination),
        "duration_seconds": asset["duration_seconds"],
        "frames": asset["frame_count"],
    }


class TrimAsset(NodeBase):
    """Trim a retained video without loading its full frame sequence into memory."""

    label = "Trim Retained Video"
    category = "Video"
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video_asset", "str"]},
        "start_seconds": {"label": "Start", "type": "float", "default": 0, "min": 0},
        "end_seconds": {"label": "End (0 = end)", "type": "float", "default": 0, "min": 0},
        "pin": {"label": "Protect From Cleanup", "type": "bool", "default": False},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "asset": {"label": "Video Asset", "display": "output", "type": "video_asset"},
        "file": {"label": "File", "display": "output", "type": "video"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        from modiff.media_assets import allocate_video_path, current_task_id, run_ffmpeg

        source = _file_asset_collection(kwargs.get("video"))[0]
        start = max(0.0, float(kwargs.get("start_seconds") or 0))
        end = float(kwargs.get("end_seconds") or 0)
        if end and end <= start:
            raise ValueError("Trim Retained Video end must be after start.")
        asset_id, destination = allocate_video_path(task_id=current_task_id())
        args = ["-ss", str(start)]
        if end:
            args += ["-to", str(end)]
        args += [
            "-i",
            source["path"],
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
        ]
        run_ffmpeg(args, destination)
        return _derived_asset_result(destination, asset_id, [source], "trim", pin=bool(kwargs.get("pin")))


class ConcatenateAssets(NodeBase):
    """Join retained videos through FFmpeg with bounded process memory."""

    label = "Join Retained Videos"
    category = "Video"
    params = {
        "clips": {"label": "Clips", "display": "input", "type": ["video_asset_collection", "collection"]},
        "transition_seconds": {"label": "Crossfade", "type": "float", "default": 0, "min": 0, "max": 5},
        "pin": {"label": "Protect From Cleanup", "type": "bool", "default": False},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "asset": {"label": "Video Asset", "display": "output", "type": "video_asset"},
        "file": {"label": "File", "display": "output", "type": "video"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        from modiff.media_assets import allocate_video_path, current_task_id, run_ffmpeg

        sources = _file_asset_collection(kwargs.get("clips"))
        first = sources[0]
        width, height = int(first["width"]), int(first["height"])
        fps = float(first["fps"] or 16)
        transition = max(0.0, float(kwargs.get("transition_seconds") or 0))
        filters = [
            _video_filter(source, index, width=width, height=height, fps=fps) + f"[v{index}]"
            for index, source in enumerate(sources)
        ]
        if len(sources) == 1:
            filters.append("[v0]null[outv]")
        elif transition <= 0:
            filters.append(
                "".join(f"[v{index}]" for index in range(len(sources))) + f"concat=n={len(sources)}:v=1:a=0[outv]"
            )
        else:
            previous = "v0"
            elapsed = float(first["duration_seconds"])
            fps_text = f"{fps:.12g}"
            for index, source in enumerate(sources[1:], 1):
                usable = min(
                    transition, max(0.001, elapsed - 1 / fps), max(0.001, float(source["duration_seconds"]) - 1 / fps)
                )
                output = "outv" if index == len(sources) - 1 else f"x{index}"
                raw_output = f"raw_{output}"
                offset = max(0.0, elapsed - usable)
                filters.append(
                    f"[{previous}][v{index}]xfade=transition=fade:duration={usable}:offset={offset}[{raw_output}]"
                )
                # FFmpeg 7 can drop the negotiated frame-rate metadata from an
                # xfade output. Reassert it before feeding that link into the
                # next xfade; otherwise a chain of three or more clips fails
                # with `current rate of 1/0 is invalid`.
                filters.append(f"[{raw_output}]settb=expr=1/{fps_text},setpts=PTS-STARTPTS,fps={fps_text}[{output}]")
                previous = output
                elapsed += float(source["duration_seconds"]) - usable
        asset_id, destination = allocate_video_path(task_id=current_task_id())
        inputs = [part for source in sources for part in ("-i", source["path"])]
        run_ffmpeg(
            inputs
            + [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[outv]",
                "-r",
                str(fps),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
            ],
            destination,
        )
        return _derived_asset_result(destination, asset_id, sources, "concatenate", pin=bool(kwargs.get("pin")))


def _bounded_video_operation_assets(value, *, minimum):
    assets = _file_asset_collection(value)
    if not minimum <= len(assets) <= MAX_VIDEO_OPERATION_INPUTS:
        raise ValueError(
            f"Built-in video operations require between {minimum} and {MAX_VIDEO_OPERATION_INPUTS} input videos."
        )
    total_frames = 0
    for index, asset in enumerate(assets, 1):
        width = int(asset.get("width") or 0)
        height = int(asset.get("height") or 0)
        frame_count = int(asset.get("frame_count") or 0)
        fps = float(asset.get("fps") or 0)
        duration = float(asset.get("duration_seconds") or 0)
        if width < 1 or height < 1 or width * height > MAX_VIDEO_OPERATION_PIXELS:
            raise ValueError(
                f"Built-in video input {index} must contain between 1 and {MAX_VIDEO_OPERATION_PIXELS} pixels per frame."
            )
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Built-in video input {index} must declare a positive frame rate.")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Built-in video input {index} must declare a positive finite duration.")
        if not 1 <= frame_count <= MAX_VIDEO_OPERATION_FRAMES_PER_INPUT:
            raise ValueError(
                f"Built-in video input {index} must contain between 1 and "
                f"{MAX_VIDEO_OPERATION_FRAMES_PER_INPUT} frames."
            )
        total_frames += frame_count
    if total_frames > MAX_VIDEO_OPERATION_TOTAL_FRAMES:
        raise ValueError(f"Built-in video inputs exceed the {MAX_VIDEO_OPERATION_TOTAL_FRAMES}-frame execution limit.")
    return assets


def _validate_video_interpolation(source, target_fps):
    source_fps = float(source["fps"])
    target_fps = float(target_fps)
    if (
        not math.isfinite(target_fps)
        or target_fps <= source_fps
        or target_fps > MAX_VIDEO_INTERPOLATION_FPS
    ):
        raise ValueError(
            "Video frame interpolation FPS must be finite, above the source rate, "
            f"and at most {MAX_VIDEO_INTERPOLATION_FPS}."
        )
    predicted_frames = math.ceil(float(source["duration_seconds"]) * target_fps)
    if predicted_frames > MAX_VIDEO_OPERATION_TOTAL_FRAMES:
        raise ValueError(
            "Video frame interpolation exceeds the bounded "
            f"{MAX_VIDEO_OPERATION_TOTAL_FRAMES}-frame output limit."
        )
    pixel_frames = int(source["width"]) * int(source["height"]) * predicted_frames
    if pixel_frames > MAX_VIDEO_INTERPOLATION_PIXEL_FRAMES:
        raise ValueError(
            "Video frame interpolation exceeds the bounded "
            f"{MAX_VIDEO_INTERPOLATION_PIXEL_FRAMES} output pixel-frame limit."
        )
    return target_fps


class ProcessVideo(NodeBase):
    """Dispatch reviewed install-free video tasks through existing bounded nodes."""

    label = "Process Video"
    category = "Video"
    resizable = True
    params = {
        "videos": {
            "label": "Videos",
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {"fileTypes": ["video"], "multiple": True},
        },
        "pipeline_class": {
            "label": "Built-in Contract",
            "type": "string",
            "default": VIDEO_OPERATION_PIPELINE_CLASS,
            "hidden": True,
        },
        "operation": {
            "label": "Operation",
            "type": "string",
            "options": VIDEO_OPERATION_MODES,
            "default": "video_frame_extract",
        },
        "selection_mode": {
            "label": "Frame Selection",
            "type": "string",
            "options": ["first", "last", "first_last", "indices", "timecodes", "every_n"],
            "default": "first_last",
        },
        "indices": {"label": "Frame Indices", "type": "string", "default": "0,-1"},
        "timecodes": {"label": "Times (seconds)", "type": "string", "default": "0"},
        "every_n": {"label": "Every N Frames", "type": "int", "default": 16, "min": 1, "max": 14400},
        "fps": {"label": "Output FPS", "type": "float", "default": 24, "min": 1, "max": 120},
        "interpolation_fps": {
            "label": "Interpolation FPS",
            "type": "float",
            "default": 60,
            "min": 1,
            "max": MAX_VIDEO_INTERPOLATION_FPS,
            "step": 0.01,
        },
        "transition_seconds": {
            "label": "Crossfade",
            "type": "float",
            "default": 0,
            "min": 0,
            "max": 5,
            "step": 0.05,
        },
        "start_seconds": {"label": "Trim Start", "type": "float", "default": 0, "min": 0, "max": 86_400},
        "end_seconds": {"label": "Trim End (0 = end)", "type": "float", "default": 0, "min": 0, "max": 86_400},
        "columns": {"label": "Tile Columns", "type": "int", "default": 2, "min": 1, "max": 4},
        "sync": {
            "label": "Tile Length",
            "type": "string",
            "options": ["shortest", "longest_hold"],
            "default": "longest_hold",
        },
        "gap": {"label": "Tile Gap", "type": "int", "default": 0, "min": 0, "max": 256},
        "background": {
            "label": "Tile Background",
            "type": "string",
            "options": ["black", "white", "gray"],
            "default": "black",
        },
        "images": {"label": "Extracted Frames", "display": "output", "type": "image"},
        "video": {"label": "Processed Video", "display": "output", "type": "video"},
        "selected_indices": {"label": "Indices", "display": "output", "type": "collection"},
        "timestamps": {"label": "Timestamps", "display": "output", "type": "collection"},
    }

    def execute(self, **kwargs):
        if kwargs.get("pipeline_class", VIDEO_OPERATION_PIPELINE_CLASS) != VIDEO_OPERATION_PIPELINE_CLASS:
            raise ValueError("Process Video received an unsupported built-in contract identity.")
        operation = str(kwargs.get("operation") or "video_frame_extract")
        if operation not in VIDEO_OPERATION_MODES:
            raise ValueError(f"Unsupported built-in video operation {operation!r}.")
        if operation == "video_frame_extract":
            assets = _bounded_video_operation_assets(kwargs.get("videos"), minimum=1)
            if len(assets) != 1:
                raise ValueError("Video frame extraction requires exactly one input video.")
            selection_mode = str(kwargs.get("selection_mode") or "first_last")
            if selection_mode not in FrameExtract.params["mode"]["options"]:
                raise ValueError(f"Unsupported frame selection mode {selection_mode!r}.")
            if selection_mode in {"indices", "timecodes"}:
                field = selection_mode
                raw = kwargs.get(field)
                if isinstance(raw, str) and len(raw) > 4096:
                    raise ValueError("Frame selection text accepts at most 4096 characters.")
                values = _parse_numbers(raw, cast=int if selection_mode == "indices" else float)
                if len(values) > MAX_EXTRACTED_FRAMES or any(
                    isinstance(value, float) and (not math.isfinite(value) or value < 0) for value in values
                ):
                    raise ValueError(f"Frame selection accepts at most {MAX_EXTRACTED_FRAMES} finite values.")
            every_n = int(kwargs.get("every_n") or 16)
            if not 1 <= every_n <= MAX_VIDEO_OPERATION_FRAMES_PER_INPUT:
                raise ValueError(f"Frame interval must be between 1 and {MAX_VIDEO_OPERATION_FRAMES_PER_INPUT}.")
            result = FrameExtract().execute(
                video=assets[0],
                mode=selection_mode,
                indices=kwargs.get("indices", "0,-1"),
                timecodes=kwargs.get("timecodes", "0"),
                every_n=every_n,
                fps=assets[0]["fps"],
            )
            return {
                "images": result["frames"],
                "video": None,
                "selected_indices": result["selected_indices"],
                "timestamps": result["timestamps"],
            }
        if operation == "frame_interpolation":
            assets = _bounded_video_operation_assets(kwargs.get("videos"), minimum=1)
            if len(assets) != 1:
                raise ValueError("Video frame interpolation requires exactly one input video.")
            source = assets[0]
            target_fps = _validate_video_interpolation(source, kwargs.get("interpolation_fps") or 60)
            result = FrameInterpolateAsset().execute(video=source, target_fps=target_fps, pin=False)
        elif operation == "video_stitch":
            assets = _bounded_video_operation_assets(kwargs.get("videos"), minimum=2)
            transition_seconds = float(kwargs.get("transition_seconds") or 0)
            if not math.isfinite(transition_seconds) or not 0 <= transition_seconds <= 5:
                raise ValueError("Video stitch crossfade must be between 0 and 5 seconds.")
            result = ConcatenateAssets().execute(
                clips=assets,
                transition_seconds=transition_seconds,
                pin=False,
            )
        elif operation == "video_trim":
            assets = _bounded_video_operation_assets(kwargs.get("videos"), minimum=1)
            if len(assets) != 1:
                raise ValueError("Video trim requires exactly one input video.")
            start_seconds = float(kwargs.get("start_seconds") or 0)
            end_seconds = float(kwargs.get("end_seconds") or 0)
            duration = float(assets[0]["duration_seconds"])
            if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
                raise ValueError("Video trim bounds must be finite.")
            if start_seconds < 0 or start_seconds >= duration:
                raise ValueError("Video trim start must be within the source duration.")
            if end_seconds < 0 or end_seconds > duration or (end_seconds and end_seconds <= start_seconds):
                raise ValueError("Video trim end must be zero or after start within the source duration.")
            result = TrimAsset().execute(
                video=assets[0],
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                pin=False,
            )
        elif operation == "video_reverse":
            assets = _bounded_video_operation_assets(kwargs.get("videos"), minimum=1)
            if len(assets) != 1:
                raise ValueError("Video reverse requires exactly one input video.")
            source = assets[0]
            pixel_frames = int(source["width"]) * int(source["height"]) * int(source["frame_count"])
            if pixel_frames > MAX_VIDEO_REVERSE_PIXEL_FRAMES:
                raise ValueError(
                    "Video reverse exceeds the bounded in-memory FFmpeg reverse workload "
                    f"of {MAX_VIDEO_REVERSE_PIXEL_FRAMES} pixel-frames."
                )
            result = ReverseAsset().execute(video=source, pin=False)
        else:
            assets = _bounded_video_operation_assets(kwargs.get("videos"), minimum=2)
            raw_columns = kwargs.get("columns", 2)
            raw_gap = kwargs.get("gap", 0)
            columns = int(2 if raw_columns is None else raw_columns)
            gap = int(0 if raw_gap is None else raw_gap)
            sync = str(kwargs.get("sync") or "longest_hold")
            background = str(kwargs.get("background") or "black")
            if not 1 <= columns <= 4:
                raise ValueError("Video tile columns must be between 1 and 4.")
            if not 0 <= gap <= 256:
                raise ValueError("Video tile gap must be between 0 and 256 pixels.")
            if sync not in StackTileAssets.params["sync"]["options"]:
                raise ValueError(f"Unsupported video tile synchronization policy {sync!r}.")
            if background not in ProcessVideo.params["background"]["options"]:
                raise ValueError(f"Unsupported video tile background {background!r}.")
            width = int(assets[0]["width"])
            height = int(assets[0]["height"])
            used_columns = min(columns, len(assets))
            rows = math.ceil(len(assets) / columns)
            output_width = used_columns * width + (used_columns - 1) * gap
            output_height = rows * height + (rows - 1) * gap
            if output_width * output_height > MAX_VIDEO_OPERATION_PIXELS:
                raise ValueError(
                    f"Video tile output must contain at most {MAX_VIDEO_OPERATION_PIXELS} pixels per frame."
                )
            result = StackTileAssets().execute(
                videos=assets,
                columns=columns,
                sync=sync,
                gap=gap,
                background=background,
                pin=False,
            )
        return {
            "images": None,
            "video": result["file"],
            "selected_indices": [],
            "timestamps": [],
        }


class UpscaleVideo(NodeBase):
    """Stream a retained video through one exact, app-managed Spandrel model."""

    label = "Upscale Video"
    category = "Video"
    resizable = True
    params = {
        "video": {
            "label": "Source Video",
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {"fileTypes": ["video"], "multiple": False},
        },
        "pipeline_class": {
            "label": "Upscale Contract",
            "type": "string",
            "default": VIDEO_UPSCALE_PIPELINE_CLASS,
            "hidden": True,
        },
        "operation": {
            "label": "Operation",
            "type": "string",
            "options": [VIDEO_UPSCALE_MODE],
            "default": VIDEO_UPSCALE_MODE,
            "hidden": True,
        },
        "model_id": {
            "label": "Upscaler",
            "display": "modelselect",
            "type": "string",
            "default": VIDEO_UPSCALE_MODEL_SELECTION,
            "fieldOptions": {
                "noValidation": True,
                "sources": ["hub", "local"],
                "filter": {"hub": {}, "local": {"id": r"^upscalers/"}},
            },
        },
        "tile_size": {
            "label": "Tile Size",
            "type": "int",
            "default": 256,
            "min": 64,
            "max": 2048,
            "step": 32,
        },
        "tile_overlap": {
            "label": "Tile Overlap",
            "type": "int",
            "default": 32,
            "min": 0,
            "max": 256,
            "step": 8,
        },
        "device": {
            "label": "Device",
            "type": "string",
            "default": DEFAULT_DEVICE,
            "options": DEVICE_LIST,
        },
        "fps": {"label": "Output FPS", "type": "float", "default": 24, "min": 1, "max": 120},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "video_out"},
        "video_out": {"label": "Upscaled Video", "display": "output", "type": "video"},
        "width": {"display": "output", "type": "int"},
        "height": {"display": "output", "type": "int"},
        "frames": {"display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import imageio
        import numpy as np
        from PIL import Image
        from modiff.media_assets import allocate_video_path, current_task_id
        from modules.Spandrel.main import Upscaler

        if kwargs.get("pipeline_class", VIDEO_UPSCALE_PIPELINE_CLASS) != VIDEO_UPSCALE_PIPELINE_CLASS:
            raise ValueError("Upscale Video received an unsupported pipeline contract identity.")
        if kwargs.get("operation", VIDEO_UPSCALE_MODE) != VIDEO_UPSCALE_MODE:
            raise ValueError("Upscale Video received an unsupported operation.")
        assets = _bounded_video_operation_assets(kwargs.get("video"), minimum=1)
        if len(assets) != 1:
            raise ValueError("Video upscaling requires exactly one input video.")
        source = assets[0]
        frame_count = int(source["frame_count"])
        width = int(source["width"])
        height = int(source["height"])
        if frame_count > MAX_VIDEO_UPSCALE_FRAMES:
            raise ValueError(f"Video upscaling accepts at most {MAX_VIDEO_UPSCALE_FRAMES} source frames.")
        if width * height > MAX_VIDEO_UPSCALE_INPUT_PIXELS:
            raise ValueError(
                f"Video upscaling accepts at most {MAX_VIDEO_UPSCALE_INPUT_PIXELS} source pixels per frame."
            )
        tile_size = int(kwargs.get("tile_size") or 256)
        tile_overlap = int(kwargs.get("tile_overlap") or 0)
        if not 64 <= tile_size <= 2048:
            raise ValueError("Video upscale tile size must be between 64 and 2048 pixels.")
        if not 0 <= tile_overlap <= min(256, tile_size // 2):
            raise ValueError("Video upscale tile overlap exceeds the bounded tile contract.")
        output_fps = float(kwargs.get("fps") or 24)
        if not math.isfinite(output_fps) or not 1 <= output_fps <= 120:
            raise ValueError("Video upscale output FPS must be between 1 and 120.")

        asset_id, destination = allocate_video_path(task_id=current_task_id())
        reader = imageio.get_reader(source["path"], "ffmpeg")
        writer = None
        upscaler = Upscaler(self.node_id)
        decoded_frames = 0
        output_width = output_height = 0
        try:
            for raw_frame in reader:
                if decoded_frames >= frame_count or decoded_frames >= MAX_VIDEO_UPSCALE_FRAMES:
                    raise ValueError("Decoded video frames exceed the reviewed source metadata bound.")
                result = upscaler.execute(
                    image=Image.fromarray(raw_frame).convert("RGB"),
                    model_id=kwargs.get("model_id", VIDEO_UPSCALE_MODEL_SELECTION),
                    downscale=1.0,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                    device=kwargs.get("device") or DEFAULT_DEVICE,
                ).get("output")
                if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], Image.Image):
                    raise ValueError("The selected video upscaler returned an invalid frame batch.")
                frame = result[0].convert("RGB")
                if decoded_frames == 0:
                    output_width, output_height = frame.size
                    if (
                        output_width <= width
                        or output_height <= height
                        or output_width * output_height > MAX_VIDEO_UPSCALE_OUTPUT_PIXELS
                    ):
                        raise ValueError("The selected video upscaler returned an invalid output scale.")
                    writer = imageio.get_writer(
                        destination,
                        fps=output_fps,
                        quality=8,
                        codec="libx264",
                    )
                elif frame.size != (output_width, output_height):
                    raise ValueError("The selected video upscaler returned inconsistent frame dimensions.")
                writer.append_data(np.asarray(frame))
                decoded_frames += 1
                self.progress(
                    min(99, decoded_frames / frame_count * 100),
                    phase="upscaling",
                    message=f"Upscaled frame {decoded_frames}/{frame_count}",
                )
        except Exception:
            if writer is not None:
                writer.close()
                writer = None
            destination.unlink(missing_ok=True)
            raise
        finally:
            reader.close()
            if writer is not None:
                writer.close()
        if decoded_frames != frame_count:
            destination.unlink(missing_ok=True)
            raise ValueError("Decoded video frame count does not match the reviewed source metadata.")
        result = _derived_asset_result(destination, asset_id, [source], "spandrel-video-upscale")
        self.progress(100, phase="upscaling", message=f"Upscaled {decoded_frames} frames")
        return {
            "video_out": result["file"],
            "width": output_width,
            "height": output_height,
            "frames": decoded_frames,
        }


class StackTileAssets(NodeBase):
    """Build a synchronized retained-video wall without Python frame materialization."""

    label = "Tile Retained Videos"
    category = "Video"
    params = {
        "videos": {"label": "Videos", "display": "input", "type": ["video_asset_collection", "collection"]},
        "columns": {"label": "Columns", "type": "int", "default": 2, "min": 1},
        "sync": {
            "label": "Length",
            "type": "string",
            "options": ["shortest", "longest_hold"],
            "default": "longest_hold",
        },
        "gap": {"label": "Gap", "type": "int", "default": 0, "min": 0},
        "background": {"label": "Background", "type": "string", "default": "black"},
        "pin": {"label": "Protect From Cleanup", "type": "bool", "default": False},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "asset": {"label": "Video Asset", "display": "output", "type": "video_asset"},
        "file": {"label": "File", "display": "output", "type": "video"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        from modiff.media_assets import allocate_video_path, current_task_id, run_ffmpeg

        sources = _file_asset_collection(kwargs.get("videos"))
        first = sources[0]
        width, height, fps = int(first["width"]), int(first["height"]), float(first["fps"] or 16)
        columns = max(1, int(kwargs.get("columns") or 1))
        gap = max(0, int(kwargs.get("gap") or 0))
        durations = [float(source["duration_seconds"]) for source in sources]
        output_duration = min(durations) if kwargs.get("sync") == "shortest" else max(durations)
        filters = []
        for index, source in enumerate(sources):
            hold = output_duration - float(source["duration_seconds"]) if kwargs.get("sync") != "shortest" else None
            filters.append(
                _video_filter(source, index, width=width, height=height, fps=fps, duration=hold) + f"[v{index}]"
            )
        layout = "|".join(
            f"{index % columns * (width + gap)}_{index // columns * (height + gap)}" for index in range(len(sources))
        )
        filters.append(
            "".join(f"[v{index}]" for index in range(len(sources)))
            + f"xstack=inputs={len(sources)}:layout={layout}:fill={kwargs.get('background') or 'black'}[outv]"
        )
        asset_id, destination = allocate_video_path(task_id=current_task_id())
        inputs = [part for source in sources for part in ("-i", source["path"])]
        run_ffmpeg(
            inputs
            + [
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[outv]",
                "-r",
                str(fps),
                "-t",
                str(output_duration),
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
            ],
            destination,
        )
        return _derived_asset_result(destination, asset_id, sources, "tile", pin=bool(kwargs.get("pin")))


class ReverseAsset(NodeBase):
    """Reverse a retained clip on disk; audio can be remuxed after visual editing."""

    label = "Reverse Retained Video"
    category = "Video"
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video_asset", "str"]},
        "pin": {"label": "Protect From Cleanup", "type": "bool", "default": False},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "asset": {"label": "Video Asset", "display": "output", "type": "video_asset"},
        "file": {"label": "File", "display": "output", "type": "video"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        from modiff.media_assets import allocate_video_path, current_task_id, run_ffmpeg

        source = _file_asset_collection(kwargs.get("video"))[0]
        asset_id, destination = allocate_video_path(task_id=current_task_id())
        run_ffmpeg(
            [
                "-i",
                source["path"],
                "-vf",
                "reverse",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
            ],
            destination,
        )
        return _derived_asset_result(destination, asset_id, [source], "reverse", pin=bool(kwargs.get("pin")))


class FrameInterpolateAsset(NodeBase):
    """Increase a retained clip's frame rate through bounded deterministic blending."""

    label = "Interpolate Retained Video Frames"
    category = "Video"
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video_asset", "str"]},
        "target_fps": {
            "label": "Target FPS",
            "type": "float",
            "default": 60,
            "min": 1,
            "max": MAX_VIDEO_INTERPOLATION_FPS,
        },
        "pin": {"label": "Protect From Cleanup", "type": "bool", "default": False},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "asset": {"label": "Video Asset", "display": "output", "type": "video_asset"},
        "file": {"label": "File", "display": "output", "type": "video"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        from modiff.media_assets import allocate_video_path, current_task_id, run_ffmpeg

        sources = _bounded_video_operation_assets(kwargs.get("video"), minimum=1)
        if len(sources) != 1:
            raise ValueError("Interpolate Retained Video Frames requires exactly one input video.")
        source = sources[0]
        source_fps = float(source.get("fps") or 0)
        target_fps = _validate_video_interpolation(source, kwargs.get("target_fps") or 60)
        duration = float(source.get("duration_seconds") or 0)
        fps_text = f"{target_fps:.12g}"
        duration_text = f"{duration:.12g}"
        tail_padding_text = f"{max(1.0, 2.0 / source_fps):.12g}"
        asset_id, destination = allocate_video_path(task_id=current_task_id())
        run_ffmpeg(
            [
                "-i",
                source["path"],
                "-vf",
                (
                    f"tpad=stop_mode=clone:stop_duration={tail_padding_text},"
                    f"minterpolate=fps={fps_text}:mi_mode=blend,"
                    f"trim=duration={duration_text},setpts=PTS-STARTPTS"
                ),
                "-an",
                "-r",
                fps_text,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
            ],
            destination,
        )
        return _derived_asset_result(
            destination,
            asset_id,
            [source],
            "frame_interpolation_blend",
            pin=bool(kwargs.get("pin")),
        )


class CrossfadeAssets(NodeBase):
    """Crossfade two retained clips through the same scalable join implementation."""

    label = "Crossfade Retained Videos"
    category = "Video"
    params = {
        "first": {"label": "First", "display": "input", "type": ["video_asset", "str"]},
        "second": {"label": "Second", "display": "input", "type": ["video_asset", "str"]},
        "duration_seconds": {"label": "Duration", "type": "float", "default": 0.35, "min": 0, "max": 5},
        "pin": {"label": "Protect From Cleanup", "type": "bool", "default": False},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "asset": {"label": "Video Asset", "display": "output", "type": "video_asset"},
        "file": {"label": "File", "display": "output", "type": "video"},
        "duration_seconds_output": {"label": "Output Duration", "display": "output", "type": "float"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        result = ConcatenateAssets().execute(
            clips=[kwargs.get("first"), kwargs.get("second")],
            transition_seconds=kwargs.get("duration_seconds"),
            pin=kwargs.get("pin"),
        )
        result["duration_seconds_output"] = result.pop("duration_seconds")
        return result


class MuxAudioAsset(NodeBase):
    """Attach loaded or generated audio to a retained video without decoding its frames."""

    label = "Add Audio to Retained Video"
    category = "Video"
    params = {
        "video": {"label": "Video", "display": "input", "type": ["video_asset", "str"]},
        "audio": {"label": "Audio", "display": "input", "type": ["audio", "str"]},
        "fit": {
            "label": "Duration",
            "type": "string",
            "options": ["match_video", "shortest"],
            "default": "match_video",
        },
        "pin": {"label": "Protect From Cleanup", "type": "bool", "default": False},
        "preview": {"display": "ui_video", "type": "url", "dataSource": "file"},
        "asset": {"label": "Video Asset", "display": "output", "type": "video_asset"},
        "file": {"label": "File", "display": "output", "type": "video"},
        "duration_seconds": {"label": "Duration", "display": "output", "type": "float"},
        "frames": {"label": "Frames", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import numpy as np
        from scipy.io import wavfile
        from modiff.media_assets import allocate_video_path, current_task_id, run_ffmpeg

        source = _file_asset_collection(kwargs.get("video"))[0]
        audio = kwargs.get("audio")
        temporary_audio = None
        if isinstance(audio, str):
            audio_path = resolve_runtime_input_path(audio).resolve()
        elif isinstance(audio, dict) and (audio.get("path") or audio.get("file")):
            audio_path = resolve_runtime_input_path(str(audio.get("path") or audio.get("file"))).resolve()
        else:
            samples = audio.get("samples", audio.get("audio")) if isinstance(audio, dict) else audio
            if samples is None:
                raise ValueError("Add Audio to Retained Video needs loaded or generated audio.")
            sample_rate = int(audio.get("sample_rate") or 48000) if isinstance(audio, dict) else 48000
            array = np.asarray(samples, dtype=np.float32)
            while array.ndim > 2:
                array = array[0]
            if array.ndim == 2 and array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
                array = array.T
            temporary_audio = Path(source["path"]).with_name(f".{Path(source['path']).stem}-audio.wav")
            wavfile.write(temporary_audio, sample_rate, (np.clip(array, -1, 1) * 32767).astype(np.int16))
            audio_path = temporary_audio
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        asset_id, destination = allocate_video_path(task_id=current_task_id())
        args = [
            "-i",
            source["path"],
            "-i",
            str(audio_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "256k",
        ]
        if kwargs.get("fit") == "shortest":
            args.append("-shortest")
        else:
            args += ["-af", "apad", "-t", str(source["duration_seconds"])]
        args += ["-movflags", "+faststart"]
        try:
            run_ffmpeg(args, destination)
        finally:
            if temporary_audio is not None:
                temporary_audio.unlink(missing_ok=True)
        return _derived_asset_result(destination, asset_id, [source], "mux_audio", pin=bool(kwargs.get("pin")))


class CleanupAssets(NodeBase):
    """Remove retained temporary media with pinned-asset protection."""

    label = "Clean Temporary Media"
    category = "Video"
    params = {
        "scope": {
            "label": "Scope",
            "type": "string",
            "options": ["current_run", "older_than", "all_unpinned"],
            "default": "current_run",
        },
        "older_than_hours": {"label": "Older Than Hours", "type": "float", "default": 24, "min": 0},
        "report": {"label": "Cleanup Report", "display": "output", "type": "string"},
        "removed_count": {"label": "Removed", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        import json
        from modiff.media_assets import cleanup_media_assets

        scope = str(kwargs.get("scope") or "current_run")
        task_id = None
        older = None
        if scope == "current_run":
            try:
                from modiff.server import server

                task_id = (server.current_task or {}).get("task_id")
            except Exception:
                task_id = None
            if not task_id:
                raise ValueError("Current-run cleanup is only available while a task identity is active.")
        elif scope == "older_than":
            older = float(kwargs.get("older_than_hours") or 0) * 3600
        elif scope != "all_unpinned":
            raise ValueError(f"Unsupported temporary media cleanup scope {scope!r}.")
        report = cleanup_media_assets(task_id=task_id, older_than_seconds=older)
        return {"report": json.dumps(report, sort_keys=True), "removed_count": len(report["removed"])}
