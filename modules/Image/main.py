from modiff.NodeBase import NodeBase
from PIL import Image
from modiff.config import CONFIG
from pathlib import Path
import logging
from utils.torch_utils import DEVICE_LIST, DEFAULT_DEVICE
from utils.paths import parse_filename
import hashlib
import json
import nanoid

logger = logging.getLogger('modiff')

def collapse_single(values):
    return values[0] if len(values) == 1 else values

def alpha_to_mask(image):
    if "A" in image.getbands():
        return image.getchannel("A").convert("L")
    return Image.new("L", image.size, 255)

class Load(NodeBase):
    """
    Load an image from a file
    """

    label = "Load Image"
    category = "image"
    resizable = True
    params = {
        "image": {
            "label": "Image",
            "display": "output",
            "type": "image",
        },
        'label': {
            "display": "ui_label",
            "value": "Load Image",
        },
        "file": {
            "label": False,
            "display": "filebrowser",
            "type": "str",
            "fieldOptions": {
                "fileTypes": ["image"],
                "multiple": True,
            },
        },
        "alpha_channel": {
            "label": "Alpha Channel",
            "type": "string",
            "options": ["ignore", "add alpha", "remove alpha"],
            "default": "ignore",
        },
        "width": { "display": "output", "type": "int" },
        "height": { "display": "output", "type": "int" },
        "mask": { "display": "output", "type": "image", "label": "Alpha mask" },
        "source_hash": { "display": "output", "type": "str", "label": "Source hash" },
    }

    def execute(self, **kwargs):
        file = kwargs["file"]
        images = []
        widths = []
        heights = []
        masks = []
        source_hashes = []

        file = file if isinstance(file, list) else [file]
        for f in file:
            if f is None or f == "":
                continue

            try:
                if f.startswith("http://") or f.startswith("https://"):
                    import requests
                    from io import BytesIO
                    response = requests.get(f)
                    response.raise_for_status()
                    content = response.content
                    image = Image.open(BytesIO(content))
                    source_hash = hashlib.sha256(content).hexdigest()
                else:
                    if not Path(f).is_absolute():
                        f = Path(CONFIG.paths['work_dir']) / f
                    if not Path(f).exists():
                        continue
                    content = Path(f).read_bytes()
                    image = Image.open(f)
                    source_hash = hashlib.sha256(content).hexdigest()

                image.load()
                mask = alpha_to_mask(image)
                image = image.copy()

                alpha_channel = kwargs.get("alpha_channel", "ignore")
                if alpha_channel == "add alpha" and image.mode != "RGBA":
                    image = image.convert("RGBA")
                elif alpha_channel == "remove alpha" and image.mode == "RGBA":
                    image = image.convert("RGB")
                images.append(image)
                widths.append(image.width)
                heights.append(image.height)
                masks.append(mask)
                source_hashes.append(source_hash)
            except Exception as e:
                logger.error(f"Error loading image {f}: {e}")
                continue

        if len(images) == 0:
            raise ValueError("Load Image could not load any selected image. Check that the file or URL exists and is a supported image.")

        return {
            "image": collapse_single(images),
            "width": collapse_single(widths),
            "height": collapse_single(heights),
            "mask": collapse_single(masks),
            "source_hash": collapse_single(source_hashes),
        }

class Save(NodeBase):
    """
    Save an image to a file
    """

    label = "Save Image"
    category = "image"
    style = { "minWidth": 360 }
    resizable = True
    params = {
        "image": {
            "label": "Image",
            "display": "input",
            "type": "image",
        },
        "filename": {
            "label": "File",
            "type": "str",
            "default": "{PATH:images}/MoDiff_{HASH:6}.webp",
        },
        "quality": {
            "label": "Quality",
            "type": "int",
            "display": "slider",
            "default": 100,
            "min": 1,
            "max": 100
        },
        "metadata": {
            "label": "PNG Metadata",
            "display": "textarea",
            "type": "text",
            "default": "",
            "description": "Optional JSON metadata to embed into PNG outputs.",
        },
        "output": {
            "label": "Filename",
            "display": "output",
            "type": "str",
        },
        "saved_image_data": {
            "label": "Saved Images",
            "display": "output",
            "type": "image",
            "hidden": True,
        },
        "saved_images": {
            "label": "Saved Images",
            "display": "ui_image",
            "type": "url",
            "dataSource": "saved_image_data",
        },
    }

    def execute(self, **kwargs):
        from pathlib import Path
        from PIL import Image

        image = kwargs.get("image")
        if not isinstance(image, list):
            image = [image]

        filename = kwargs.get("filename", "{PATH:images}/MoDiff_{HASH:6}.webp")
        extension = filename.split('.')[-1].lower()
        quality = kwargs.get("quality", 95)
        metadata = kwargs.get("metadata")
        output = []
        saved_images = []

        for index, img in enumerate(image):
            if img is None:
                continue
            parsed_filename = parse_filename(filename, index=index)
            parsed_filename = Path(parsed_filename)

            if not parsed_filename.is_absolute():
                parsed_filename = Path(CONFIG.paths['data']) / parsed_filename

            try:
                if not parsed_filename.parent.exists():
                    parsed_filename.parent.mkdir(parents=True)

                if extension in ['jpg', 'jpeg']:
                    #image = img.convert("RGB")
                    img.save(parsed_filename, quality=quality)
                elif extension == 'png':
                    #image = img.convert("RGBA")
                    pnginfo = None
                    if metadata:
                        from PIL.PngImagePlugin import PngInfo
                        pnginfo = PngInfo()
                        try:
                            parsed_metadata = json.loads(metadata) if isinstance(metadata, str) else metadata
                            if isinstance(parsed_metadata, dict):
                                for key, value in parsed_metadata.items():
                                    pnginfo.add_text(str(key), json.dumps(value) if not isinstance(value, str) else value)
                            else:
                                pnginfo.add_text("modiff_metadata", str(parsed_metadata))
                        except Exception:
                            pnginfo.add_text("modiff_metadata", str(metadata))
                    img.save(parsed_filename, pnginfo=pnginfo)
                elif extension == 'webp':
                    #image = img.convert("RGBA")
                    img.save(parsed_filename, quality=quality)
                else:
                    img.save(parsed_filename)
                output.append(str(parsed_filename))
                saved_images.append(img.copy())
            except Exception as e:
                logger.error(f"Error saving image {parsed_filename}: {e}")

        if len(output) == 0:
            output = None
        if len(output) == 1:
            output = output[0]

        return { "output": output, "saved_image_data": collapse_single(saved_images) if saved_images else None }

class Preview(NodeBase):
    """
    Preview an image
    """
    label = "Preview Image"
    category = "image"
    resizable = True
    params = {
        "vae": { "type": "pipeline", "display": "input", "label": "VAE" },
        "device": { "type": "string", "default": DEFAULT_DEVICE, "options": DEVICE_LIST },
        "image": { "type": ["image", "latent"], "display": "input", "onChange": {
            "action": "show",
            "data": { True: ["vae", "device"], False: [] },
            "condition": { "type": "latent" }
        }},
        "preview": { "display": "ui_image", "type": "url", "dataSource": "output" },
        "output": { "type": "image", "display": "output", "label": "All images" },
        "export": { "type": "str", "default": "0", "description": "Export the image at the given index. Leave empty to export all images." },
        "filtered": { "type": "image", "display": "output", "label": "Selected image" },
    }

    def execute(self, **kwargs):
        image = kwargs["image"]
        export = kwargs["export"]
        filtered = None
        if image is None:
            return {"output": None, "filtered": None}

        output = image

        # if image is an Image or an array of Images, pass it to the preview
        if not (isinstance(image, Image.Image) or (isinstance(image, list) and len(image) > 0 and isinstance(image[0], Image.Image))):
            from modules.Experiments.VAE import VAEDecode
            pipeline = kwargs["vae"]
            device = kwargs["device"]
            if pipeline is None:
                logger.error("VAE is required to decode latents")
                return {"output": None}
            pipeline = pipeline.vae if hasattr(pipeline, 'vae') else pipeline
            vae = VAEDecode()
            if isinstance(image, list) or isinstance(image, tuple):
                output = self.mm_exec(lambda: vae.decode(pipeline, image[0], image[1]), device, models=[pipeline])
            else:
                output = self.mm_exec(lambda: vae.decode(pipeline, image), device, models=[pipeline])

        filtered = output
        if export != "":
            try:
                indices = [int(i.strip()) for i in str(export).split(",") if i.strip()]
                if len(indices) > 0:
                    if isinstance(output, list):
                        filtered = [output[i] for i in indices if 0 <= i < len(output)]
                        if len(filtered) == 0:
                            filtered = None
                        elif len(filtered) == 1:
                            filtered = filtered[0]
                    else:
                        filtered = output if 0 in indices else None
            except Exception:
                pass

        return {"output": output, "filtered": filtered}

class Resize(NodeBase):
    """
    Resize an image or a list of images
    """

    label = "Resize"
    category = "image"
    params = {
        "image": { "type": "image", "display": "input" },
        "method": { "type": "string", "default": "stretch", "options": ["stretch", "cover", "contain", "fit", "pad"],
                   "description": """
                   The method to use to resize the image.
                   `Stretch` will stretch the image to the given width and height.
                   `Cover` will resize to the maximum dimension that fits the given size keeping the aspect ratio.
                   `Contain` will resize to the minimum dimension that fits the given size keeping the aspect ratio.
                   `Fit` will resize to the given size and crop the excess.
                   `Pad` will resize to the given size and pad the excess space with black.
                    """
        },
        "width": { "type": "int", "default": 1024 },
        "height": { "type": "int", "default": 1024 },
        "multiple_of": { "label": "Multiple of", "type": "int", "default": 1, "min": 1, "description": "Ensure the resulting width and height are multiples of this value." },
        "interpolation": { "default": "bicubic", "options": ["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"] },
        "output": { "label": "Image", "type": "image", "display": "output" },
    }

    def execute(self, **kwargs):
        from utils.image import resize, fit, cover, contain, pad

        image = kwargs["image"]
        method = kwargs["method"]
        width = kwargs["width"]
        height = kwargs["height"]
        interpolation = kwargs["interpolation"]
        multiple_of = kwargs["multiple_of"]

        if multiple_of > 1:
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of

        if method == "stretch":
            image = resize(image, width, height, interpolation)
        elif method == "cover":
            image = cover(image, width, height, interpolation)
        elif method == "contain":
            image = contain(image, width, height, interpolation)
        elif method == "fit":
            image = fit(image, width, height, interpolation)
        elif method == "pad":
            image = pad(image, width, height, interpolation)

        return {"output": image}


class Compare(NodeBase):
    label = "Image Compare"
    category = "image"
    resizable = True
    params = {
        "image_1": {
            "label": "Image From",
            "display": "input",
            "type": "image",
        },
        "image_2": {
            "label": "Image To",
            "display": "input",
            "type": "image",
        },
        "image_list": {
            "display": "output",
            "type": "image",
            "hidden": True
        },
        "compare": {
            "label": "Compare",
            "display": "ui_imagecompare",
            "type": "url",
            "dataSource": "image_list"
        },
    }

    def execute(self, **kwargs):
        image_1 = kwargs.get("image_1")
        image_2 = kwargs.get("image_2")
        image_1 = image_1[0] if isinstance(image_1, list) and len(image_1) > 0 else image_1
        image_2 = image_2[0] if isinstance(image_2, list) and len(image_2) > 0 else image_2

        return {
            "image_list": [image_1, image_2]
        }

class ApplyMask(NodeBase):
    label = "Apply Mask"
    category = "image"
    params = {
        "image": {
            "label": "Image",
            "display": "input",
            "type": "image",
        },
        "mask": {
            "label": "Mask",
            "display": "input",
            "type": "image",
        },
        "output": {
            "label": "Output",
            "display": "output",
            "type": "image",
        }
    }

    def execute(self, **kwargs):
        image = kwargs.get("image")
        mask = kwargs.get("mask")

        if image is None or mask is None:
            return {"output": None}

        image = [image] if not isinstance(image, list) else image
        mask = [mask] if not isinstance(mask, list) else mask

        if len(mask) < len(image):
            diff = len(image) - len(mask)
            mask = mask + [mask[-1]] * diff
        elif len(mask) > len(image):
            mask = mask[:len(image)]

        # Apply the mask to the image
        output = []
        for img, msk in zip(image, mask):
            img = img.convert("RGBA")
            msk = msk.convert("L")
            img.putalpha(msk)
            output.append(img)

        return {"output": output}

class Merge(NodeBase):
    label = "Merge Images"
    category = "image"
    params = {
        "image_from": {
            "label": "Image From",
            "display": "input",
            "type": "image",
        },
        "image_to": {
            "label": "Image To",
            "display": "input",
            "type": "image",
        },
        "mask": {
            "label": "Optional Mask",
            "display": "input",
            "type": "image",
            "description": "Optional mask to control the merging. If not provided, alpha compositing will be used."
        },
        "output": {
            "label": "Output",
            "display": "output",
            "type": "image",
        }
    }

    def execute(self, **kwargs):
        image_from = kwargs.get("image_from")
        image_to = kwargs.get("image_to")
        mask = kwargs.get("mask")

        if image_from is None or image_to is None:
            return {"output": None}

        image_from = [image_from] if not isinstance(image_from, list) else image_from
        image_to = [image_to] if not isinstance(image_to, list) else image_to
        if mask is not None:
            mask = [mask] if not isinstance(mask, list) else mask

        max_len = max(len(image_from), len(image_to), len(mask) if mask is not None else 0)

        def extend_list(lst, target_len):
            if len(lst) < target_len:
                return lst + [lst[-1]] * (target_len - len(lst))
            return lst[:target_len]

        image_from = extend_list(image_from, max_len)
        image_to = extend_list(image_to, max_len)
        if mask is not None:
            mask = extend_list(mask, max_len)
        else:
            mask = [None] * max_len

        output = []
        for img_from, img_to, msk in zip(image_from, image_to, mask):
            img_to = img_to.convert("RGBA")

            if img_from.size != img_to.size:
                img_from = img_from.resize(img_to.size, Image.Resampling.BICUBIC)
            img_from = img_from.convert("RGBA")

            if msk is not None:
                if msk.size != img_to.size:
                    msk = msk.resize(img_to.size, Image.Resampling.BICUBIC)
                msk = msk.convert("L")
                blended = Image.composite(img_from, img_to, msk)
            else:
                blended = Image.alpha_composite(img_to, img_from)

            output.append(blended)

        return {"output": output}
