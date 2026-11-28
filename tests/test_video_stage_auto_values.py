"""Read-only video controls must match their ordinary deterministic stages."""
from copy import deepcopy
import json

import pytest

from modiff.workflow_auto_values import DATA_ACTIONS, inspect_resource_value


def literal(value):
    return {"value": value}


def wire(source, port):
    return {"sourceId": source, "sourceKey": port}


def storyboard_graph():
    return {
        "board": {"module": "modules.VideoConditioning", "action": "Storyboard", "params": {
            "shots_json": literal(json.dumps([{"prompt": "first shot", "seed": 41, "duration_seconds": 5}, "second shot"])),
            "base_seed": literal(10)}},
        "shot": {"module": "modules.VideoConditioning", "action": "ShotInputs", "params": {
            "storyboard": wire("board", "storyboard"), "index": literal(0), "base_seed": wire("board", "seed"),
            "width": literal(768), "height": literal(512), "num_frames": literal(121), "fps": literal(16),
            "steps": literal(8), "guidance_scale": literal(1)}},
        "prompt": {"module": "modules.VideoConditioning", "action": "ShotPrompt", "params": {
            "storyboard": wire("board", "storyboard"), "index": literal(0), "prompt_override": literal("")}},
        "sample": {"module": "modules.DiffusersVideo", "action": "Denoise", "params": {
            "width": wire("shot", "width_out"), "height": wire("shot", "height_out"),
            "num_frames": wire("shot", "frames"), "num_inference_steps": wire("shot", "steps_out"),
            "seed": wire("shot", "seed"), "guidance_scale": wire("shot", "guidance"),
            "prompt": wire("prompt", "prompt")}},
    }


def test_stage_storyboard_helpers_have_explicit_data_only_contracts():
    for action in ["Storyboard", "ShotInputs", "ShotPrompt", "OutpaintCanvas", "AlignMask"]:
        assert ("modules.VideoConditioning", action) in DATA_ACTIONS


def test_storyboard_resource_values_are_pure_and_preserve_record_overrides(monkeypatch):
    from modules.VideoConditioning.main import Storyboard, ShotInputs, ShotPrompt
    for cls in [Storyboard, ShotInputs, ShotPrompt]:
        monkeypatch.setattr(cls, "execute", lambda *a, **k: pytest.fail("Inspection must not execute a node"))
    nodes = storyboard_graph()
    before = deepcopy(nodes)
    expected = {"width": 768, "height": 512, "num_frames": 81, "num_inference_steps": 8,
                "seed": 41, "guidance_scale": 1, "prompt": "first shot"}
    for field, value in expected.items():
        assert inspect_resource_value(nodes, "sample", field) == value
    assert nodes == before


def test_storyboard_inspection_uses_declared_numeric_widget_casts():
    from modules.VideoConditioning.main import evaluate_shot_inputs
    nodes = storyboard_graph()
    for name in ('index', 'width', 'height', 'num_frames', 'fps', 'steps', 'guidance_scale'):
        nodes['shot']['params'][name]['value'] = str(nodes['shot']['params'][name]['value'])
    assert inspect_resource_value(nodes, 'sample', 'num_frames') == 81
    assert inspect_resource_value(nodes, 'sample', 'width') == 768
    assert evaluate_shot_inputs({'storyboard': '["a"]', 'width': 768})['width_out'] == 768


def test_canvas_inspection_keeps_cycle_and_safe_integer_bounds():
    nodes = {
        'normalize': {'module': 'modules.VideoConditioning', 'action': 'Normalize', 'params': {
            'width': literal(9_007_199_254_740_991), 'height': literal(480), 'num_frames': literal(81)}},
        'canvas': {'module': 'modules.VideoConditioning', 'action': 'OutpaintCanvas', 'params': {
            'video': wire('normalize', 'output'), 'left': literal(1)}},
        'sample': {'module': 'modules.DiffusersVideo', 'action': 'Generate', 'params': {'width': wire('canvas', 'width')}},
    }
    with pytest.raises(ValueError):
        inspect_resource_value(nodes, 'sample', 'width')
    nodes['normalize']['params']['width'] = wire('canvas', 'width')
    with pytest.raises(ValueError, match='cycle'):
        inspect_resource_value(nodes, 'sample', 'width')


@pytest.mark.parametrize("field,value", [("index", 9), ("index", True), ("fps", float("inf")),
                                         ("width", 0), ("seed_override", True)])
def test_storyboard_inspection_keeps_native_validation(field, value):
    nodes = storyboard_graph()
    nodes["shot"]["params"][field] = literal(value)
    if field == "seed_override":
        nodes["shot"]["params"]["use_seed_override"] = literal(True)
    with pytest.raises(ValueError):
        inspect_resource_value(nodes, "sample", "num_frames")


def test_prepared_and_expanded_canvas_geometry_is_forecast_without_reading_media(monkeypatch):
    from modules.VideoConditioning.main import OutpaintCanvas, Normalize
    for cls in [OutpaintCanvas, Normalize]:
        monkeypatch.setattr(cls, "execute", lambda *a, **k: pytest.fail("Inspection must not decode or allocate media"))
    nodes = {
        "normalize": {"module": "modules.VideoConditioning", "action": "Normalize", "params": {
            "width": literal(320), "height": literal(480), "num_frames": literal(81)}},
        "canvas": {"module": "modules.VideoConditioning", "action": "OutpaintCanvas", "params": {
            "video": wire("normalize", "output"), "preparation": literal("expand"),
            "left": literal(256), "right": literal(256), "top": literal(0), "bottom": literal(0)}},
        "sample": {"module": "modules.DiffusersVideo", "action": "Generate", "params": {
            "width": wire("canvas", "width"), "height": wire("canvas", "height"),
            "num_frames": wire("canvas", "frames")}},
    }
    assert inspect_resource_value(nodes, "sample", "width") == 832
    assert inspect_resource_value(nodes, "sample", "height") == 480
    assert inspect_resource_value(nodes, "sample", "num_frames") == 81
    nodes["canvas"]["params"]["preparation"] = literal("prepared")
    assert inspect_resource_value(nodes, "sample", "width") == 320


def test_unknown_or_invalid_canvas_controls_do_not_gain_geometry_authority():
    nodes = {
        "source": {"module": "custom.unknown", "action": "Frames", "params": {}},
        "canvas": {"module": "modules.VideoConditioning", "action": "OutpaintCanvas", "params": {
            "video": wire("source", "video"), "preparation": literal("expand")}},
        "sample": {"module": "modules.DiffusersVideo", "action": "Generate", "params": {"width": wire("canvas", "width")}},
    }
    with pytest.raises(ValueError):
        inspect_resource_value(nodes, "sample", "width")
