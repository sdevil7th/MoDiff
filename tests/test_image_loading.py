"""Decoded file geometry, ordering and failures through normal image dispatch."""

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modules.Image.main import Load


@pytest.mark.parametrize('source_kind', ['local', 'web-import'])
def test_decoding_and_recorded_hash_use_the_same_read_bytes(source_kind, tmp_path, monkeypatch):
    path = tmp_path / 'changing.png'
    Image.new('RGB', (7, 5), 'red').save(path)
    original = path.read_bytes()
    replacement = BytesIO()
    Image.new('RGB', (3, 2), 'blue').save(replacement, format='PNG')
    read_bytes = Path.read_bytes

    def change_after_read(target):
        content = read_bytes(target)
        if target == path:
            target.write_bytes(replacement.getvalue())
        return content

    monkeypatch.setattr(Path, 'read_bytes', change_after_read)
    if source_kind == 'web-import':
        monkeypatch.setattr('modiff.media_import.import_web_media', lambda _url: path)
    source = str(path) if source_kind == 'local' else 'https://example.test/changing.png'
    node = Load('same-byte-source')
    result = node(file=source, alpha_channel='ignore')
    assert result['image'].size == (7, 5)
    assert result['image'].getpixel((0, 0)) == (255, 0, 0)
    assert result['source_hash'] == hashlib.sha256(original).hexdigest()
    fields = node._execution_input_record['fields']
    assert fields['source_hashes']['value'] == [result['source_hash']]
    assert fields['source_widths']['value'] == [7]
    assert fields['source_heights']['value'] == [5]


def test_consumed_image_hashes_keep_order_and_survive_an_unchanged_cache_hit(tmp_path):
    paths = [tmp_path / 'red.png', tmp_path / 'blue.png']
    for path, color in zip(paths, ('red', 'blue'), strict=True):
        Image.new('RGB', (3, 2), color).save(path)
    source = [str(path) for path in reversed(paths)]
    node = Load('ordered-consumed-hashes')
    node(file=source, alpha_channel='remove alpha')
    first = node._execution_input_record
    assert first['fields']['source_hashes']['value'] == [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in reversed(paths)
    ]
    assert first['fields']['alpha_channel']['value'] == 'remove alpha'
    node(file=source, alpha_channel='remove alpha')
    assert node._execution_input_record == first
    assert node._cache_reason == 'unchanged_inputs'


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_keeps_pixels_alpha_geometry_and_original_file_hash(orientation, tmp_path):
    pixels = np.array([
        [[255, 0, 0, 0], [0, 255, 0, 50], [0, 0, 255, 100]],
        [[255, 255, 0, 150], [0, 255, 255, 200], [255, 0, 255, 255]],
    ], dtype=np.uint8)
    exif = Image.Exif()
    exif[274] = orientation
    path = tmp_path / "oriented.png"
    Image.fromarray(pixels).save(path, exif=exif)
    original_file = path.read_bytes()
    expected = {
        1: pixels,
        2: pixels[:, ::-1],
        3: pixels[::-1, ::-1],
        4: pixels[::-1],
        5: pixels.transpose(1, 0, 2),
        6: np.rot90(pixels, -1),
        7: pixels.transpose(1, 0, 2)[::-1, ::-1],
        8: np.rot90(pixels),
    }[orientation]
    result = Load("oriented-image")(file=str(path), alpha_channel="ignore")
    assert np.array_equal(np.asarray(result["image"]), expected)
    assert np.array_equal(np.asarray(result["mask"]), expected[:, :, 3])
    assert (result["width"], result["height"]) == (expected.shape[1], expected.shape[0])
    assert result["source_hash"] == hashlib.sha256(original_file).hexdigest()
    assert path.read_bytes() == original_file


@pytest.mark.parametrize("format_name,extension", [
    ("PNG", "png"), ("JPEG", "jpeg"), ("WEBP", "webp"),
    ("BMP", "bmp"), ("GIF", "gif"), ("TIFF", "tiff"),
])
def test_common_image_formats_keep_source_geometry_and_hash(format_name, extension, tmp_path):
    path = tmp_path / ("source." + extension)
    Image.new("RGB", (7, 5), (10, 20, 30)).save(path, format=format_name)
    result = Load("format-image")(file=str(path), alpha_channel="ignore")
    assert result["image"].size == (7, 5)
    assert result["mask"].getextrema() == (255, 255)
    assert result["source_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_multiple_images_keep_authored_order_and_individual_geometry(tmp_path):
    paths = []
    for index, (size, color) in enumerate([((3, 5), "red"), ((7, 9), "blue")]):
        path = tmp_path / f"source-{index}.png"
        Image.new("RGB", size, color).save(path)
        paths.append(path)
    result = Load("ordered-images")(file=[str(path) for path in reversed(paths)], alpha_channel="ignore")
    assert result["width"] == [7, 3]
    assert result["height"] == [9, 5]
    assert [image.getpixel((0, 0)) for image in result["image"]] == [(0, 0, 255), (255, 0, 0)]
    assert result["source_hash"] == [hashlib.sha256(path.read_bytes()).hexdigest() for path in reversed(paths)]


@pytest.mark.parametrize("body", [b"", b"not an image", b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"])
def test_corrupt_input_has_an_actionable_failure_and_no_reusable_cache(body, tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(body)
    node = Load("corrupt-image")
    with pytest.raises(RuntimeError, match="Load Image could not load any selected image"):
        node(file=str(path), alpha_channel="ignore")
    assert not node._cache_valid
    assert path.read_bytes() == body


def test_pillow_decompression_bomb_refusal_is_preserved_without_large_allocation(tmp_path, monkeypatch):
    path = tmp_path / "over-limit.png"
    Image.new("RGB", (3, 3), "black").save(path)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 4)
    node = Load("over-limit-image")
    with pytest.raises(RuntimeError, match="Load Image could not load any selected image"):
        node(file=str(path), alpha_channel="ignore")
    assert not node._cache_valid
