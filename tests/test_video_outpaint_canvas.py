"""User-visible automatic video expansion, through normal node dispatch."""
from PIL import Image
import pytest

from modules.VideoConditioning import main as conditioning


def canvas_node():
    klass = getattr(conditioning, 'OutpaintCanvas', None)
    assert klass is not None, 'Automatic video canvas/mask preparation is missing'
    return klass('video-outpaint-canvas-regression')


def test_padding_keeps_source_pixels_and_generates_matching_white_borders():
    frames = [Image.new('RGB', (4, 3), color) for color in ('red', 'blue')]
    originals = [frame.tobytes() for frame in frames]
    result = canvas_node()(video=frames, left=2, right=3, top=1, bottom=2)
    assert result['width'] == 9 and result['height'] == 6
    assert result['frames'] == 2
    for index, (canvas, mask) in enumerate(zip(result['canvas'], result['mask'])):
        assert canvas.size == mask.size == (9, 6)
        assert canvas.crop((2, 1, 6, 4)).tobytes() == originals[index]
        assert mask.convert('L').crop((2, 1, 6, 4)).getextrema() == (0, 0)
        for point in [(0, 2), (8, 2), (4, 0), (4, 5)]:
            assert mask.convert('L').getpixel(point) == 255
    assert [frame.tobytes() for frame in frames] == originals


def test_zero_padding_is_a_pixel_preserving_preparation_with_no_generated_region():
    frame = Image.new('RGB', (4, 3), (17, 42, 89))
    result = canvas_node()(video=[frame], left=0, right=0, top=0, bottom=0)
    assert result['canvas'][0].tobytes() == frame.tobytes()
    assert result['mask'][0].convert('L').getextrema() == (0, 0)


def test_prepared_mode_preserves_canvas_and_alignment_ignores_border_controls(monkeypatch):
    from modules.Video import main as video
    frames = [Image.new('RGB', (4, 3), 'red'), Image.new('RGB', (4, 3), 'blue')]
    supplied = []
    def load(value, **kwargs):
        supplied.append(value)
        return {'video': [Image.new('RGB', (2, 2), 'white')]}
    monkeypatch.setattr(video, 'load_video_file', load)
    result = canvas_node()(video=frames, preparation='prepared', prepared_mask_file='masks/boundary.mp4', left=999)
    assert supplied == ['masks/boundary.mp4']
    assert all(actual is original for actual, original in zip(result['canvas'], frames))
    assert (result['width'], result['height'], result['frames']) == (4, 3, 2)
    assert all(mask.size == (4, 3) and mask.getextrema() == ((255, 255),) * 3 for mask in result['mask'])


def test_expand_mode_does_not_load_the_inactive_prepared_mask(monkeypatch):
    from modules.Video import main as video
    monkeypatch.setattr(video, 'load_video_file', lambda *args, **kwargs: pytest.fail('inactive mask read'))
    result = canvas_node()(video=[Image.new('RGB', (4, 3))], preparation='expand', prepared_mask_file='', left=2)
    assert result['width'] == 6


def test_preparation_mode_rejects_unknown_choice():
    with pytest.raises(ValueError, match='preparation.*prepared, expand'):
        canvas_node()(video=[Image.new('RGB', (4, 3))], preparation='guess')


@pytest.mark.parametrize('field', ['left', 'right', 'top', 'bottom'])
def test_negative_borders_are_rejected_by_ordinary_dispatch(field):
    with pytest.raises(RuntimeError, match=field) as caught:
        canvas_node()(video=[Image.new('RGB', (4, 3))], **{field: -1})
    assert isinstance(caught.value.__cause__, ValueError)


def test_mixed_source_geometry_is_rejected_before_constructing_partial_output():
    with pytest.raises(RuntimeError, match='same dimensions') as caught:
        canvas_node()(video=[Image.new('RGB', (4, 3)), Image.new('RGB', (5, 3))], left=2)
    assert isinstance(caught.value.__cause__, ValueError)


def test_registry_publishes_the_actual_canvas_and_mask_outputs():
    from modules import MODULE_MAP
    fields = MODULE_MAP['modules.VideoConditioning']['OutpaintCanvas']['params']
    for name in ('canvas', 'mask'):
        assert fields[name]['type'] == 'video'
        assert fields[name]['display'] == 'output'


def test_memory_pressure_rejects_before_allocating_a_canvas(monkeypatch):
    from modiff import hardware
    frame = Image.new('RGB', (4, 3))
    monkeypatch.setattr(hardware, 'system_memory_snapshot', lambda: {
        'available_bytes': 4 * 1024**3, 'total_bytes': 32 * 1024**3,
    })
    monkeypatch.setattr(Image, 'new', lambda *args, **kwargs: pytest.fail('allocated before memory admission'))
    with pytest.raises(RuntimeError, match='RAM reserve') as caught:
        canvas_node()(video=[frame], left=2)
    assert isinstance(caught.value.__cause__, MemoryError)


def test_mask_frames_are_independent_and_keep_frame_order():
    frames = [Image.new('RGB', (4, 3), color) for color in ('red', 'blue')]
    result = canvas_node()(video=frames, left=1)
    result['mask'][0].putpixel((0, 0), (0, 0, 0))
    assert result['mask'][1].getpixel((0, 0)) == (255, 255, 255)
    assert result['canvas'][0].getpixel((1, 0)) == (255, 0, 0)
    assert result['canvas'][1].getpixel((1, 0)) == (0, 0, 255)


def test_canvas_does_not_impose_a_generation_models_resolution_limit():
    result = canvas_node()(video=[Image.new('RGB', (16, 2))], left=4097)
    assert result['width'] == 4113
    assert result['canvas'][0].size == (4113, 2)


@pytest.mark.parametrize('video', [[], [Image.new('RGBA', (4, 3))]])
def test_unprepared_or_empty_sources_have_an_actionable_normal_dispatch_error(video):
    with pytest.raises(RuntimeError, match='source video|Normalize Video') as caught:
        canvas_node()(video=video)
    assert isinstance(caught.value.__cause__, ValueError)


def test_interruption_does_not_publish_a_partial_frame_collection(monkeypatch):
    node = canvas_node()

    def interrupt_after_first_frame(*args, **kwargs):
        node._interrupt = True

    monkeypatch.setattr(node, 'progress', interrupt_after_first_frame)
    with pytest.raises(RuntimeError, match='interrupted') as caught:
        node(video=[Image.new('RGB', (4, 3)) for _ in range(3)], left=1)
    assert isinstance(caught.value.__cause__, InterruptedError)
    assert node.output.get('canvas') is None
