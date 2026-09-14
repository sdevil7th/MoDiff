"""Replaced FLUX unpack/decode blocks execute with real small VAEs, no downloads."""

import importlib.util
import gzip
import json
from pathlib import Path

import pytest

from modiff.huggingface_node_library import reviewed_huggingface_node_library


def admissions():
    return [item for item in reviewed_huggingface_node_library()["definitions"]
            if item.get("pipelineClass", "").startswith("Flux")
            and item.get("definitionKind") == "modular_pipeline_workflow"]


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="Optional model runtime not active")
@pytest.mark.parametrize("definition", admissions(), ids=lambda item: item["id"])
def test_replaced_flux_decoder_matches_direct_upstream_pixels(definition):
    import numpy as np
    import torch
    from diffusers import AutoencoderKL, AutoencoderKLFlux2
    from diffusers.modular_pipelines import PipelineState
    from modiff.modular_composition import build_reviewed_modular_composition_blocks, _block_at_path
    from modules.ModularDiffusers import reviewed_blocks

    library = reviewed_huggingface_node_library()
    contracts = {item["id"]: item for item in library["blockDefinitions"]}
    catalog = json.loads(gzip.decompress((Path(__file__).resolve().parents[1] /
        "modiff/registered_block_v2_catalog.v1.json.gz").read_bytes()))
    compiled = next(item["definition"] for item in catalog["entries"]
                    if item["definition"]["source"]["manifestDefinitionId"] == definition["id"])
    stages = [{"path": node["modularDiffusers"]["placementPath"],
               "blockDefinitionId": node["modularDiffusers"]["blockDefinitionId"]}
              for node in compiled["graph"]["nodes"]
              if node.get("modularDiffusers", {}).get("blockClass") in
              {"FluxDecodeStep", "Flux2UnpackLatentsStep", "Flux2DecodeStep"}]
    stages.sort(key=lambda item: contracts[item["blockDefinitionId"]]["className"] != "Flux2UnpackLatentsStep")
    assert len(stages) == (2 if definition["pipelineClass"].startswith("Flux2") else 1)
    recipe = {"schemaVersion": 1, "diffusersRevision": library["diffusersRevision"],
        "pipelineClass": definition["pipelineClass"], "workflowId": definition["workflowId"],
        "definitionId": definition["id"], "blockContractHash": definition["blockContractHash"],
        "operations": [{"kind": "replace", "path": stage["path"], "sourceDefinitionId": definition["id"],
                        "sourceBlockDefinitionId": stage["blockDefinitionId"],
                        "sourcePath": next(p["path"] for p in definition["blockPlacements"]
                                           if p["blockDefinitionId"] == stage["blockDefinitionId"]),
                        "sourceExecutionScope": "selected_workflow"} for stage in stages]}
    validated, blocks = build_reviewed_modular_composition_blocks(recipe)
    pipeline = blocks.init_pipeline()
    torch.manual_seed(42)
    kwargs = dict(block_out_channels=(32, 32, 32, 32), layers_per_block=1, latent_channels=2)
    vae = (AutoencoderKLFlux2(**kwargs) if definition["pipelineClass"].startswith("Flux2") else
           AutoencoderKL(**kwargs, down_block_types=("DownEncoderBlock2D",)*4,
                         up_block_types=("UpDecoderBlock2D",)*4, shift_factor=0.0))
    vae.eval()
    pipeline.update_components(vae=vae)
    pipeline._modiff_composition_hash = validated["recipeHash"]
    initial = torch.randn((1, 4, 8))

    def run(adapter):
        state = PipelineState()
        for name, value in {"latents": initial.clone(), "height": 32, "width": 32, "output_type": "pil",
                           "latent_ids": torch.tensor([[[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,1,1,0]]])}.items():
            state.set(name, value)
        issued = reviewed_blocks._issue_state(token=object(), pipeline_class=definition["pipelineClass"],
            workflow_id=definition["workflowId"], execution_scope="unpruned_pipeline", pipeline=pipeline,
            state=state, completed_path=())
        for stage in stages:
            contract = contracts[stage["blockDefinitionId"]]
            if not adapter:
                _, state = _block_at_path(pipeline.blocks, tuple(stage["path"]))(pipeline, state)
                continue
            result = reviewed_blocks.ReviewedModularWorkflowStep().execute(
                pipeline_class=definition["pipelineClass"], workflow_id=definition["workflowId"],
                execution_scope="unpruned_pipeline", placement_path=stage["path"], composition_recipe=recipe,
                block_definition_id=contract["id"], block_class=contract["className"],
                block_contract_hash=contract["contentHash"], execution_kind="step", state_in=issued)
            issued = result["state_out"]
        images = state.get("images")
        assert len(images) == 1 and images[0].size == (32, 32)
        return np.asarray(images[0])

    np.testing.assert_array_equal(run(True), run(False))
