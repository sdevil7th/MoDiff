"""Real small-VAE execution of edited unpack/decode/output stages, not image-quality proof."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest


def admissions():
    catalog = json.loads(gzip.decompress((Path(__file__).resolve().parents[1] / "modiff/registered_block_v2_catalog.v1.json.gz").read_bytes()))
    return [item["definition"] for item in catalog["entries"] if item["definition"]["source"].get("pipelineClass", "").startswith("Qwen")]


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Qwen optional runtime not active")
@pytest.mark.parametrize("admission", admissions(), ids=lambda item: item["definitionId"])
def test_replaced_unpack_decoder_and_output_blocks_execute_with_real_vae(admission):
    import torch
    from diffusers import AutoencoderKLQwenImage
    from diffusers.modular_pipelines import PipelineState
    from modiff.huggingface_node_library import reviewed_huggingface_node_library
    from modiff.modular_composition import build_reviewed_modular_composition_blocks
    from modules.ModularDiffusers import reviewed_blocks

    library = reviewed_huggingface_node_library()
    definition = next(item for item in library["definitions"] if item["id"] == admission["source"]["manifestDefinitionId"])
    classes = {"QwenImageAfterDenoiseStep", "QwenImageLayeredAfterDenoiseStep", "QwenImageDecoderStep", "QwenImageLayeredDecoderStep", "QwenImageProcessImagesOutputStep", "QwenImageInpaintProcessImagesOutputStep"}
    order = admission["graph"]["executionOrder"]
    stages = sorted([node for node in admission["graph"]["nodes"] if node.get("modularDiffusers", {}).get("blockClass") in classes], key=lambda node: order.index(node["nodeId"]))
    assert len(stages) >= 2
    operations = []
    for node in stages:
        metadata = node["modularDiffusers"]
        source = next(item for item in definition["blockPlacements"] if item["blockDefinitionId"] == metadata["blockDefinitionId"])
        operations.append({"kind": "replace", "path": metadata["placementPath"], "sourceDefinitionId": definition["id"],
            "sourceBlockDefinitionId": metadata["blockDefinitionId"], "sourcePath": source["path"], "sourceExecutionScope": "selected_workflow"})
    recipe = {"schemaVersion": 1, "diffusersRevision": library["diffusersRevision"], "pipelineClass": definition["pipelineClass"],
        "workflowId": definition["workflowId"], "definitionId": definition["id"], "blockContractHash": definition["blockContractHash"], "operations": operations}
    validated, blocks = build_reviewed_modular_composition_blocks(recipe)
    pipeline = blocks.init_pipeline()
    layered = "Layered" in definition["pipelineClass"]
    torch.manual_seed(42)
    vae = AutoencoderKLQwenImage(base_dim=32, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
        input_channels=4 if layered else 3, latents_mean=[0.0]*16, latents_std=[1.0]*16)
    pipeline.update_components(vae=vae)
    pipeline._modiff_composition_hash = validated["recipeHash"]
    state = PipelineState()
    for name, value in {"latents": torch.randn((1, 12 if layered else 4, 64)), "height": 32, "width": 32, "layers": 2, "output_type": "pil"}.items():
        state.set(name, value)
    issued = reviewed_blocks._issue_state(token=object(), pipeline_class=definition["pipelineClass"], workflow_id=definition["workflowId"],
        execution_scope="unpruned_pipeline", pipeline=pipeline, state=state, completed_path=("denoise",))
    for node in stages:
        metadata = node["modularDiffusers"]
        result = reviewed_blocks.ReviewedModularWorkflowStep().execute(pipeline_class=definition["pipelineClass"], workflow_id=definition["workflowId"],
            execution_scope="unpruned_pipeline", composition_recipe=recipe, placement_path=metadata["placementPath"],
            block_definition_id=metadata["blockDefinitionId"], block_class=metadata["blockClass"], block_contract_hash=metadata["blockContractHash"],
            execution_kind="step", state_in=issued)
        issued = result["state_out"]
    images = state.get("images")
    if layered:
        assert len(images) == 1 and len(images[0]) == 2
        images = images[0]
    else:
        assert len(images) == 1
    assert all(image.size == (32, 32) for image in images)
    if not layered:
        decoder = next(node for node in stages if node["modularDiffusers"]["blockClass"] == "QwenImageDecoderStep")
        metadata = decoder["modularDiffusers"]
        state.set("latents", torch.zeros((1, 2, 3)))
        with pytest.raises(ValueError, match=r"Modular block .*QwenImageDecoderStep.*4D or 5D.*Input shapes: latents=\(1, 2, 3\)"):
            reviewed_blocks.ReviewedModularWorkflowStep().execute(pipeline_class=definition["pipelineClass"], workflow_id=definition["workflowId"],
                execution_scope="unpruned_pipeline", composition_recipe=recipe, placement_path=metadata["placementPath"],
                block_definition_id=metadata["blockDefinitionId"], block_class=metadata["blockClass"], block_contract_hash=metadata["blockContractHash"],
                execution_kind="step", state_in=issued)
