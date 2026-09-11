# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
from modiff.upscaler_contracts import real_esrgan_x2_model_selection

MODULE_MAP = {
    "Upscaler": {
        "label": "Upscale with model",
        "category": "upscaler",
        "params": {
            "image": {
                "label": "Image or video frames",
                "type": ["image", "video"],
                "display": "input",
                "required": True,
            },
            "model_id": {
                "label": "Model",
                "display": "modelselect",
                "type": "string",
                "default": real_esrgan_x2_model_selection(),
                "fieldOptions": {
                    "noValidation": True,
                    "sources": ['hub', 'local'],
                    "filter": {
                        "hub": {},
                        "local": { "id": r"^upscalers/" }
                    },
                },
            },

            "downscale": { "label": "Downscale", "type": "float", "default": 1.0, "min": 0.1, "max": 1.0, "step": 0.01, "display": "slider", "description": "Post downscaling factor. After the image is upscaled, it is downscaled by this factor." },
            "tile_size": { "label": "Tile size", "type": "int", "default": 256, "min": 0, "max": 2048, "step": 32, "description": "Input tile size. Use 0 only when full-frame inference is known to fit." },
            "tile_overlap": { "label": "Tile overlap", "type": "int", "default": 32, "min": 0, "max": 256, "step": 8, "description": "Context overlap cropped from each tile boundary before CPU-side stitching." },
            "device": { "label": "Device", "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
            "output": { "label": "Upscaled frames", "type": ["image", "video"], "display": "output" },
        }
    },
}
