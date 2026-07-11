from modiff.NodeBase import NodeBase
from modiff.config import CONFIG
from pathlib import Path
import logging
from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
from utils.paths import parse_filename

logger = logging.getLogger('modiff')

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
        'label': {
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
        "filename": { "label": "File Name", "display": "output", "type": "str" },
        "width": { "display": "output", "type": "int" },
        "height": { "display": "output", "type": "int" },
        "frames": { "display": "output", "type": "int" },
        "fps": { "label": "FPS", "display": "output", "type": "float" },
    }

    def execute(self, **kwargs):
        import imageio
        from PIL import Image
        file = kwargs["file"]
        file = Path(file[0] if isinstance(file, list) else file)
        logger.debug(f"Loading video from file: {file}")

        if file is None or file == "":
            file = ""
        if not Path(file).is_absolute():
            file = Path(CONFIG.paths['work_dir']) / file
        if not Path(file).exists():
            file = ""

        images = []
        reader = imageio.get_reader(str(file), 'ffmpeg')
        meta = reader.get_meta_data()
        width = meta.get('size', (0,0))[0]
        height = meta.get('size', (0,0))[1]
        frames = reader.count_frames()
        fps = meta.get('fps', 0)

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
        "video": { "type": ["video", "str", "image"], "display": "input" },
        "filename": {
            "label": "File",
            "type": "str",
            "default": "{PATH:videos}/MoDiff_{HASH:6}.mp4",
        },
        #"codec": { "type": "str", "options": ["libx264", "vp9"], "default": "libx264" },
        "quality": { "display": "slider", "type": "int", "min": 1, "max": 10, "default": 5 },
        "fps": { "label": "FPS", "type": "float", "default": 24, "min": 1, "max": 240, "step": 0.01 },
        "preview": { "display": "ui_video", "type": "url", "dataSource": "file" },
        "file": { "type": "video", "display": "output" },
        "width": { "display": "output", "type": "int" },
        "height": { "display": "output", "type": "int" },
        "frames": { "display": "output", "type": "int" },
    }

    def execute(self, **kwargs):
        import imageio
        import numpy as np
        from PIL import Image

        video = kwargs["video"]
        filename = kwargs.get("filename", "{PATH:videos}/MoDiff_{HASH:6}.mp4")
        quality = kwargs.get("quality", 5)
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
            if video_data is None:
                return None, 0, 0, 0

            parsed_filename = parse_filename(filename)
            Path(parsed_filename).parent.mkdir(parents=True, exist_ok=True)
            width, height, frame_count = 0, 0, 0

            if isinstance(video_data, str):
                reader = imageio.get_reader(video_data)
                meta = reader.get_meta_data()
                width, height = meta.get('size', (0, 0))
                try:
                    frame_count = reader.count_frames()
                except Exception:
                    frame_count = 0
                writer = imageio.get_writer(parsed_filename,
                                            fps=fps,
                                            quality=quality,
                                            codec='libx264',
                                            )
                for frame in reader:
                    writer.append_data(frame)
                reader.close()
                writer.close()

            elif isinstance(video_data, list):
                width, height, frame_count = frames_shape(video_data)

                writer = imageio.get_writer(parsed_filename,
                                            fps=fps,
                                            quality=quality,
                                            codec='libx264',
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

            return str(parsed_filename), width, height, frame_count

        if isinstance(video, list) and video and isinstance(video[0], list):
            files = []
            width = height = frames = 0
            for item in video:
                file, width, height, frames = save_video(item)
                files.append(file)
            return { "file": files, "width": width, "height": height, "frames": frames }

        file, width, height, frames = save_video(video)

        return {
            "file": file,
            "width": width,
            "height": height,
            "frames": frames,
        }
