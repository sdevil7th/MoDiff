"""Preserve existing whole-operation latent graphs without automatic migration."""

import pytest

from test_image_stage_graph import runtime as graph_runtime, graph, run, node, connection

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

# Re-export the shared pytest fixture under its injected name.
runtime = graph_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


@pytest.mark.parametrize("kind", ["fill", "control"])
def test_whole_operation_legacy_raw_latent_decode_remains_valid(runtime, kind):
    server, source_path, mask_path, acquisitions, _ = runtime
    original = graph(kind, source_path, mask_path)
    del original["nodes"]["encode"]
    original["paths"][0].remove("encode")
    original["nodes"]["denoise"] = node(
        "modules.DiffusersImage",
        "Inpaint" if kind == "fill" else "ControlGenerate",
        pipeline=connection("loader", "pipeline"),
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
    del original["nodes"]["decode"]["params"]["latent_binding"]
    # Whole packed-latent operations use explicit raw-decode geometry.
    original["nodes"]["decode"]["params"]["width"] = {"value": 64}
    original["nodes"]["decode"]["params"]["height"] = {"value": 80}
    run(server, original, "saved-whole-latent")
    output = server.node_cache["preview"].output["output"]
    assert output.size == (64, 80)
    assert len(acquisitions) == 1
