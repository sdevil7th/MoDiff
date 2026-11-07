"""Exact public pre-stage connected signal remains usable without migration."""

from copy import deepcopy

import pytest

from image_stage_test_support import PRE_STAGE_SIGNALS
from test_image_stage_graph import runtime as graph_runtime, graph, run, node, connection
from modules.DiffusersImage import main as image

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

# Re-export the shared pytest fixture under its injected name.
runtime = graph_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_original_connected_whole_contract_signal_preserves_callback_and_graph(runtime, kind):
    server, source_path, mask_path, acquisitions, _ = runtime
    pipeline_class = "FluxFillPipeline" if kind == "fill" else "FluxControlPipeline"
    mode = "inpaint" if kind == "fill" else "control_image"
    action = "Inpaint" if kind == "fill" else "ControlGenerate"
    original = deepcopy(PRE_STAGE_SIGNALS[kind])
    assert not {"EncodeInputs", "Denoise"} & set(original["actions"])
    assert not {"EncodeInputs", "Denoise"} & set(original["connectionActions"])
    assert original["pipelineClass"] == pipeline_class and original["mode"] == mode
    current_node = getattr(image, action)("retained-signal")
    current_node.update_image_contract({"image_contract": deepcopy(original)}, "pipeline")
    # Compatibility allows only the exact historical action addition. Identity,
    # supported actions/ports and unrelated presentation still must agree.
    for changes in (
        {"library": "foreign"},
        {"mediaKind": "audio"},
        {"mode": "text_to_image"},
        {"pipelineClass": "FluxPipeline"},
        {"schemaVersion": 2},
        {"actions": {}},
        {"connectionActions": {}},
        {"fieldParams": {}},
    ):
        with pytest.raises(ValueError):
            current_node.update_image_contract({"image_contract": {**deepcopy(original), **changes}}, "pipeline")
    current = graph(kind, source_path, mask_path)
    del current["nodes"]["encode"]
    current["paths"][0].remove("encode")
    current["nodes"]["denoise"] = node(
        "modules.DiffusersImage",
        action,
        pipeline=connection("loader", "pipeline"),
        image_contract=deepcopy(original),
        prompt="sun",
        width=64,
        height=80,
        seed=5,
        num_inference_steps=2,
        guidance_scale=1.0,
        max_sequence_length=4,
        num_images_per_prompt=1,
        strength=1.0,
        output_type="latent",
        **(
            {"image": connection("source", "image"), "mask_image": connection("mask", "image")}
            if kind == "fill"
            else {"control_image": connection("source", "image")}
        ),
    )
    del current["nodes"]["decode"]["params"]["latent_binding"]
    current["nodes"]["decode"]["params"]["width"] = {"value": 64}
    current["nodes"]["decode"]["params"]["height"] = {"value": 80}
    run(server, current, "saved-original-connected-signal")
    assert server.node_cache["preview"].output["output"].size == (64, 80)
    assert len(acquisitions) == 1
