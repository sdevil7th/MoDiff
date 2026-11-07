"""An existing reviewed compatible whole route must not gain a stage seal."""

from test_image_stage_graph import runtime as graph_runtime, graph, run, node, connection
from modules.DiffusersImage import main as image
from modiff.model_artifact_catalog import catalog_repository_pin
from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
import pytest

from image_stage_test_support import stage_offline_runtime as stage_offline_runtime

# Re-export the shared pytest fixture under its injected name.
runtime = graph_runtime

pytestmark = pytest.mark.usefixtures("stage_offline_runtime")


def test_reviewed_compatible_canny_whole_operation_load_remains_valid(runtime):
    server, source_path, mask_path, acquisitions, _ = runtime
    profile = DIFFUSERS_EXECUTION_PROFILES["flux-canny:direct"]
    repo = image.FLUX_CANNY_REPAIR_REPO
    assert repo in profile.compatible_repos
    pin = catalog_repository_pin(repo)
    original = graph("control", source_path, mask_path)
    original["nodes"]["loader"]["params"]["model_id"]["value"] = {"source": "hub", "value": repo}
    original["nodes"]["loader"]["params"]["revision"]["value"] = pin["revision"]
    for removed in ("encode", "decode"):
        del original["nodes"][removed]
        original["paths"][0].remove(removed)
    original["nodes"]["denoise"] = node(
        "modules.DiffusersImage",
        "ControlGenerate",
        pipeline=connection("loader", "pipeline"),
        control_image=connection("source", "image"),
        prompt="sun",
        width=64,
        height=80,
        seed=5,
        num_inference_steps=2,
        guidance_scale=1.0,
        max_sequence_length=4,
        num_images_per_prompt=1,
        output_type="pil",
    )
    original["nodes"]["preview"]["params"]["image"] = connection("denoise", "images")
    run(server, original, "saved-compatible-whole")
    assert server.node_cache["preview"].output["output"].size == (64, 80)
    assert acquisitions[0][1:] == (repo, pin["revision"])
    assert vars(server.node_cache["loader"].output["pipeline"]).get("_modiff_image_stage_owner") is None
    with pytest.raises(ValueError, match="incomplete|stale"):
        image.EncodeInputs("ungranted-compatible-stage")(
            pipeline=server.node_cache["loader"].output["pipeline"],
            prompt="sun",
            width=64,
            height=80,
            seed=5,
            max_sequence_length=4,
            control_image=server.node_cache["source"].output["image"],
        )
