import json

from PIL import Image

from modules.Primitive.main import DataViewer


def test_data_viewer_accepts_in_memory_pil_images_without_a_filename():
    image = Image.new("RGB", (8, 6), "red")

    result = DataViewer().execute(value=image)
    payload = json.loads(result["output"])

    assert payload == {
        "width": 8,
        "height": 6,
        "format": None,
        "mode": "RGB",
        "size": [8, 6],
        "filename": "",
        "exif": {},
    }
