# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
from os import path
import logging
logger = logging.getLogger('modiff')

from modiff.NodeBase import NodeBase
from spandrel import ModelLoader
from utils.torch_utils import ImageToTensor, TensorToImage
from utils.image import resize
from modiff.config import CONFIG

class Upscaler(NodeBase):
    def __init__(self, node_id=None):
        super().__init__(node_id)

        self._model_path = None
        self._model = None

    def execute(self, **kwargs):
        image = kwargs.get("image")
        model_id = kwargs.get("model_id", None)
        model_source = model_id.get('source', 'local') if isinstance(model_id, dict) else 'local'
        model_path = model_id.get('value', None) if isinstance(model_id, dict) else model_id
        downscale = kwargs.get("downscale", 1.0)
        tile_size = int(kwargs.get("tile_size", 256) or 0)
        tile_overlap = int(kwargs.get("tile_overlap", 32) or 0)
        device = kwargs.get("device")
        online_status = CONFIG.hf['online_status']

        if not model_path:
            raise ValueError("Model ID is required")

        if model_source == 'hub':
            from utils.huggingface import cached_file_path

            if model_path.endswith((".safetensors", ".pt", ".pth", ".ckpt", ".pkl", ".bin")):
                path_segments = model_path.split('/')
                if len(path_segments) < 3:
                    raise ValueError(f"Invalid model path: {model_path}")

                file = '/'.join(path_segments[2:])
                repo_id = '/'.join(path_segments[:2])
                model_path = cached_file_path(repo_id, file)
            else:
                raise ValueError(
                    "A Hub upscaler must include a pinned filename, for example "
                    "amd/realesrgan-x4plus/RealESRGAN_x4plus.pth. Install that file through Model Manager first."
                )

            if not model_path:
                if online_status == "Offline":
                    #logger.error(f"Model {model_path} not found in cache and online status is `offline`. Consider changing the online status or adding the model to the cache manually.")
                    raise FileNotFoundError(f"Model {model_path} not found in cache and online status is `offline`. Consider changing the online status or adding the model to the cache manually.")

                raise FileNotFoundError(
                    f"Upscaler {repo_id}/{file} is not installed. Install the pinned file through Model Manager first."
                )

        else:
            # check if path is absolute
            if not path.isabs(model_path):
                model_path = path.join(CONFIG.paths['models'], model_path)
            # check if file exist
            if not path.isfile(model_path):
                logger.error(f"Model {model_path} not found")
                return {"output": None}

        if not model_path:
            raise ValueError(f"Invalid model path: {model_path}")

        try:
            if model_path == self._model_path and self._model is not None:
                model = self._model
            else:
                model = ModelLoader().load_from_file(model_path).eval()
                self._model_path = model_path
                self._model = model
        except Exception as e:
            logger.error(f"Error loading Spandrel model {model_path}")
            raise e

        self.mm_add(model, priority=0)
        output = self.mm_exec(
            lambda: Upscaler.upscale(image, model, tile_size=tile_size, tile_overlap=tile_overlap),
            device,
            models=[model],
        )

        if downscale != 1.0:
            output = [resize(o, int(o.width * downscale), int(o.height * downscale), resample='LANCZOS') for o in output]

        return { "output": output }

    @staticmethod
    def _upscale_tensor_tiled(image, model, tile_size=256, tile_overlap=32):
        """Upscale one BCHW tensor without retaining full-resolution GPU features.

        ESRGAN-style networks can require tens of GiB of temporary activation
        memory for a 1024px input even though the model itself is small. Process
        overlapping input tiles and copy only each tile's non-overlap core into
        a CPU output tensor. This keeps the node model-agnostic while preserving
        seamless context around every tile boundary.
        """
        import torch

        if image.ndim == 3:
            image = image.unsqueeze(0)
        height, width = image.shape[-2:]
        tile_size = max(0, int(tile_size or 0))
        if tile_size == 0 or (height <= tile_size and width <= tile_size):
            return model(image.to(model.device)).to("cpu")

        overlap = max(0, min(int(tile_overlap or 0), tile_size // 2))
        output = None
        scale_y = scale_x = None
        for top in range(0, height, tile_size):
            bottom = min(top + tile_size, height)
            input_top = max(0, top - overlap)
            input_bottom = min(height, bottom + overlap)
            for left in range(0, width, tile_size):
                right = min(left + tile_size, width)
                input_left = max(0, left - overlap)
                input_right = min(width, right + overlap)
                tile = image[..., input_top:input_bottom, input_left:input_right].to(model.device)
                tile_output = model(tile).to("cpu")
                del tile

                current_scale_y = tile_output.shape[-2] // (input_bottom - input_top)
                current_scale_x = tile_output.shape[-1] // (input_right - input_left)
                if current_scale_y <= 0 or current_scale_x <= 0:
                    raise ValueError("Upscaler returned an invalid spatial scale.")
                if output is None:
                    scale_y, scale_x = current_scale_y, current_scale_x
                    output = torch.empty(
                        (tile_output.shape[0], tile_output.shape[1], height * scale_y, width * scale_x),
                        dtype=tile_output.dtype,
                        device="cpu",
                    )
                elif (current_scale_y, current_scale_x) != (scale_y, scale_x):
                    raise ValueError("Upscaler returned an inconsistent spatial scale between tiles.")

                crop_top = (top - input_top) * scale_y
                crop_left = (left - input_left) * scale_x
                crop_bottom = crop_top + (bottom - top) * scale_y
                crop_right = crop_left + (right - left) * scale_x
                output[..., top * scale_y:bottom * scale_y, left * scale_x:right * scale_x] = tile_output[
                    ..., crop_top:crop_bottom, crop_left:crop_right
                ]
                del tile_output

        if output is None:
            raise ValueError("Upscaler received an empty image tensor.")
        return output

    @staticmethod
    def upscale(image, model, tile_size=256, tile_overlap=32):
        image = ImageToTensor(image)
        image = image if isinstance(image, list) else [image]
        output = []

        for img in image:
            if img.ndim == 3:
                img = img.unsqueeze(0)
            # remove alpha channel
            if img.shape[1] == 4:
                img = img[:, :3]

            output.append(
                Upscaler._upscale_tensor_tiled(
                    img,
                    model,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )
            )
        del image

        if output:
            output = TensorToImage(output)

        return output
