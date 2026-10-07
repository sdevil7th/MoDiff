"""The optional bundle projects existing handles; it never acquires models."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import diffusers
import pytest
import torch

from modules import MODULE_MAP
from modiff.component_bundle_contracts import QWEN_T2I_BUNDLE_INPUTS
from modiff.NodeBase import deep_equal
from modiff.operation_catalog import build_operation_catalog
from modules.ModularDiffusers.component_bundle import normalize_component_bundle_inputs
from modules.ModularDiffusers.denoise import Denoise
from modules.ModularDiffusers.embeddings import EncodePrompt
from modules.ModularDiffusers.latents import DecodeLatents
from modules.ModularDiffusers.loaders import annotate_modular_loader_outputs
from modules.ModularDiffusers.modular_utils import require_modiff_node_contract
from modules.ModularDiffusers.route_state import (
    ROUTE_STATE_INPUT, issue_normal_decode_route_state, issue_pipeline_instance_token, require_component_binding,
)


PIPELINE = "QwenImageModularPipeline"
STAGES = ((EncodePrompt, "text_encoder", "embeddings"), (Denoise, "denoise", "denoise"),
          (DecodeLatents, "decoder", "latents"))
MEMBERS = ("text_encoder", "tokenizer", "transformer", "scheduler", "vae")


class ExactManager:
    def __init__(self, suffix):
        self.objects = {f"{name}-{suffix}": SimpleNamespace(name=name) for name in MEMBERS}
        self.calls = []

    def get_components_by_ids(self, *, ids, return_dict_with_names):
        self.calls.append(tuple(ids))
        if return_dict_with_names:
            return {self.objects[key].name: self.objects[key] for key in ids if key in self.objects}
        return {key: self.objects[key] for key in ids if key in self.objects}


def bound_bundle(*, suffix="a", missing=(), pipeline=PIPELINE):
    members = {name: {"model_id": f"{name}-{suffix}"} for name in MEMBERS if name not in missing}
    token = issue_pipeline_instance_token(model_type=pipeline, repo_id="fixture/qwen", repo_source="hub", revision="a" * 40)
    outputs = {
        "pipeline_components": members,
        "text_encoders": {name: dict(members[name]) for name in ("text_encoder", "tokenizer") if name in members},
        "unet_out": {"model_id": f"transformer-{suffix}"},
        "scheduler": {"model_id": f"scheduler-{suffix}"}, "vae_out": {"model_id": f"vae-{suffix}"},
    }
    annotate_modular_loader_outputs(outputs, model_type=pipeline, repo_id="fixture/qwen", repo_source="hub",
                                   revision="a" * 40, trust_remote_code=False, pipeline_instance_token=token)
    return token, outputs, ExactManager(suffix)


@pytest.mark.parametrize("stage", QWEN_T2I_BUNDLE_INPUTS)
def test_role_projection_uses_exact_resident_handles_and_original_owner_without_mutation(stage):
    token, outputs, manager = bound_bundle()
    kwargs = {"pipeline_components": outputs["pipeline_components"], "prompt": "keep this value"}
    before = deepcopy(kwargs)
    projected = normalize_component_bundle_inputs(kwargs, node_type=stage, component_manager=manager)
    assert "pipeline_components" not in projected
    assert kwargs == before
    assert projected["prompt"] == "keep this value"
    expected_calls = []
    for field, names in QWEN_T2I_BUNDLE_INPUTS[stage].items():
        role = "denoiser" if field == "unet" else field
        assert require_component_binding(projected[field], label=field, expected_role=role) is token
        expected_calls.extend((f"{name}-a",) for name in names)
    assert manager.calls == expected_calls


@pytest.mark.parametrize("stage,member", [("text_encoder", "tokenizer"), ("denoise", "scheduler"), ("decoder", "vae")])
def test_partial_bundles_fail_before_model_initialization(stage, member):
    _, outputs, manager = bound_bundle(missing=(member,))
    with pytest.raises(ValueError, match=f"missing required member '{member}'"):
        normalize_component_bundle_inputs({"pipeline_components": outputs["pipeline_components"]},
                                          node_type=stage, component_manager=manager)


def test_bundle_rejects_unsealed_wrong_role_unknown_family_and_mutated_identity():
    _, outputs, manager = bound_bundle()
    bundle = outputs["pipeline_components"]
    unsealed = {key: value for key, value in bundle.items() if isinstance(key, str)}
    _, foreign, _ = bound_bundle(pipeline="FluxModularPipeline")
    cases = [(unsealed, "process-local"), (outputs["text_encoders"], "loader role"),
             (foreign["pipeline_components"], "not 'QwenImageModularPipeline'")]
    for payload, message in cases:
        with pytest.raises(ValueError, match=message):
            normalize_component_bundle_inputs({"pipeline_components": payload}, node_type="text_encoder",
                                              component_manager=manager)
    bundle["text_encoder"]["model_id"] = "replacement"
    with pytest.raises(ValueError, match="identity changed"):
        normalize_component_bundle_inputs({"pipeline_components": bundle}, node_type="text_encoder", component_manager=manager)


def test_override_keeps_same_owner_configuration_and_rejects_other_roles_generations_or_members():
    token, outputs, manager = bound_bundle()
    scheduler = outputs["scheduler"]
    scheduler["scheduler_kwargs"] = {"shift": 3.0}
    kwargs = {"pipeline_components": outputs["pipeline_components"], "scheduler": scheduler}
    assert normalize_component_bundle_inputs(kwargs, node_type="denoise", component_manager=manager)["scheduler"] is scheduler
    _, foreign, _ = bound_bundle()  # Same component IDs, different loader execution.
    for override, message in [(foreign["scheduler"], "different Models Loader execution"),
                              (outputs["vae_out"], "loader role")]:
        with pytest.raises(ValueError, match=message):
            normalize_component_bundle_inputs(dict(kwargs, scheduler=override), node_type="denoise", component_manager=manager)
    from modules.ModularDiffusers.route_state import bind_loader_outputs

    different = {key: value for key, value in scheduler.items() if isinstance(key, str)}
    different["model_id"] = "other-scheduler"
    bind_loader_outputs({"scheduler": different}, token)
    with pytest.raises(ValueError, match="exact managed members"):
        normalize_component_bundle_inputs(dict(kwargs, scheduler=different), node_type="denoise", component_manager=manager)


@pytest.mark.parametrize("field", ["image_latents", "image_latents_with_strength", "controlnet_bundle", "ip_adapter",
                                 ROUTE_STATE_INPUT, "mask", "masked_image_latents", "control_image_latents", "control_mode",
                                 "image_embeds", "image_condition_latents"])
def test_text_to_image_facade_does_not_grant_an_image_or_control_route(field):
    _, outputs, manager = bound_bundle()
    with pytest.raises(ValueError, match="only Qwen text-to-image"):
        normalize_component_bundle_inputs({"pipeline_components": outputs["pipeline_components"], field: {}},
                                          node_type="denoise", component_manager=manager)


@pytest.mark.parametrize("node_type,stage,module", STAGES)
def test_actual_node_rejects_evicted_members_before_block_resolution(node_type, stage, module):
    _, outputs, manager = bound_bundle()
    member = next(iter(QWEN_T2I_BUNDLE_INPUTS[stage].values()))[0]
    del manager.objects[f"{member}-a"]
    node = object.__new__(node_type)
    with (patch(f"modules.ModularDiffusers.{module}.components", manager),
          patch(f"modules.ModularDiffusers.{module}.require_modiff_node_contract") as resolve,
          pytest.raises(ValueError, match="exact managed component ID")):
        node.execute(pipeline_components=outputs["pipeline_components"])
    resolve.assert_not_called()


def cache_node(node_type, stage):
    node = object.__new__(node_type)
    node._pipeline_class = diffusers.QwenImageModularPipeline
    node._model_type = PIPELINE
    if stage == "denoise":
        blocks, config = require_modiff_node_contract(node._pipeline_class, stage)
        node._route_cache_node_input_names = tuple(config["input_names"])
        node._route_cache_block_input_names = tuple(blocks.input_names)
        node._route_cache_component_names = tuple(blocks.component_names)
        node._route_cache_model_input_names = tuple(config["model_input_names"])
    return node


@pytest.mark.parametrize("node_type,stage,module", STAGES)
def test_actual_cache_keeps_original_bundle_identity_and_revalidates_eviction(node_type, stage, module):
    token, outputs, manager = bound_bundle()
    node = cache_node(node_type, stage)
    params = {"pipeline_components": outputs["pipeline_components"]}
    if stage == "decoder":
        latents = torch.ones((1, 1, 2, 2))
        params.update(latents=latents)
        params[ROUTE_STATE_INPUT] = issue_normal_decode_route_state(binding=token, latents=latents)
    before = deepcopy(params)
    with patch(f"modules.ModularDiffusers.{module}.components", manager):
        assert node._cache_params_equal(params, params)
        assert node._cache_params_equal(params, params), "fresh role projections must not turn an identical bundle into a miss"
        assert deep_equal(params, before)
        if stage == "decoder":
            assert params["latents"] is latents
        _, newer, _ = bound_bundle()
        assert not node._cache_params_equal(params, dict(params, pipeline_components=newer["pipeline_components"]))
        member = next(iter(QWEN_T2I_BUNDLE_INPUTS[stage].values()))[0]
        del manager.objects[f"{member}-a"]
        with pytest.raises(ValueError, match="exact managed component ID"):
            node._cache_params_equal(params, params)


def test_decode_bundle_preserves_route_owner_guard_before_initialization_and_cache():
    _, outputs, manager = bound_bundle()
    foreign, _, _ = bound_bundle()
    latents = torch.ones((1, 1, 2, 2))
    params = {"pipeline_components": outputs["pipeline_components"], "latents": latents,
              ROUTE_STATE_INPUT: issue_normal_decode_route_state(binding=foreign, latents=latents)}
    node = cache_node(DecodeLatents, "decoder")
    with patch("modules.ModularDiffusers.latents.components", manager):
        with pytest.raises(ValueError, match="different ModelsLoader|different Models Loader|binding"):
            node._cache_params_equal(params, params)
        with pytest.raises(ValueError, match="different ModelsLoader|different Models Loader|binding"):
            node.execute(**params)


@pytest.mark.parametrize("node_type,stage,module", STAGES)
def test_actual_stage_execution_installs_only_the_projected_managed_components(node_type, stage, module):
    token, outputs, manager = bound_bundle()
    node = cache_node(node_type, stage)
    node.notify = Mock(side_effect=AssertionError("unexpected missing-input notification"))
    node.record_generation_inputs = Mock()
    node.progress = Mock()
    node._interrupt = False
    _, config = require_modiff_node_contract(node._pipeline_class, stage, resolve_blocks=False)
    config["output_names"] = {"text_encoder": ["embeddings"], "denoise": ["latents", "route_state_out"], "decoder": ["images"]}[stage]
    latents = torch.ones((1, 1, 2, 2))
    expected = {"text_encoder": ["text_encoder", "tokenizer"], "denoise": ["transformer", "scheduler"], "decoder": ["vae"]}[stage]
    pipeline = Mock()
    pipeline.component_names = expected
    pipeline._execution_device = torch.device("cpu")
    def install(**components):
        for name, component in components.items():
            setattr(pipeline, name, component)
    pipeline.update_components.side_effect = install
    pipeline.return_value = SimpleNamespace(get_by_kwargs=lambda _: {"prompt_embeds": "encoded"},
                                          get=lambda name: {"latents": latents, "images": "decoded"}.get(name))
    block_inputs = [name for name in config["input_names"] if name != ROUTE_STATE_INPUT] + ["prompt_embeds"]
    blocks = SimpleNamespace(component_names=expected, input_names=block_inputs, init_pipeline=Mock(return_value=pipeline))
    params = {"pipeline_components": outputs["pipeline_components"]}
    if stage == "text_encoder":
        params["prompt"] = "an exact prompt"
    elif stage == "denoise":
        params.update(embeddings={"prompt_embeds": "encoded"}, seed=7, num_inference_steps=1)
    else:
        params.update(latents=latents)
        params[ROUTE_STATE_INPUT] = issue_normal_decode_route_state(binding=token, latents=latents)
    with (patch(f"modules.ModularDiffusers.{module}.components", manager),
          patch(f"modules.ModularDiffusers.{module}.require_modiff_node_contract", return_value=(blocks, config)),
          patch("modules.ModularDiffusers.denoise.deepcopy", side_effect=lambda value: value)):
        result = node.execute(**params)
    assert result is not None
    installed = pipeline.update_components.call_args.kwargs
    assert set(installed) == set(expected)
    assert all(installed[name] is manager.objects[f"{name}-a"] for name in expected)
    assert "pipeline_components" not in pipeline.call_args.kwargs
    if stage == "denoise":
        from modules.ModularDiffusers.route_state import consume_decode_route_state
        consume_decode_route_state(result["route_state_out"], binding=token, model_type=PIPELINE, latents=result["latents"])


@pytest.mark.parametrize("node_type,stage,module", STAGES)
def test_bundle_signal_alone_refreshes_exact_native_fields_without_loading_blocks(node_type, stage, module):
    node = object.__new__(node_type)
    node._model_type = ""
    node._pipeline_class = None
    node.get_signal_value = Mock(side_effect=lambda field: PIPELINE if field == "pipeline_components" else None)
    node.send_node_definition = Mock()
    with patch("modules.ModularDiffusers.native_blocks.prepare_native_pipeline_blocks", side_effect=AssertionError("resolved blocks")):
        node.update_node({}, "pipeline_components")
    assert node._pipeline_class is diffusers.QwenImageModularPipeline
    params = node.send_node_definition.call_args.args[0]
    assert params["pipeline_components"]["hidden"] is False
    assert params["pipeline_components"]["signalCompatibility"]["role"] == "pipeline_components"
    if stage == "denoise":
        assert {value["target"] for value in params["pipeline_components"]["onSignal"] if isinstance(value, dict)} == {
            "guider", "controlnet_bundle",
        }


@pytest.mark.parametrize("stage", QWEN_T2I_BUNDLE_INPUTS)
def test_existing_unbundled_inputs_keep_their_existing_runtime_contract(stage):
    kwargs = {"unet": {"repo_id": "legacy"}, "prompt": "old graph"}
    assert normalize_component_bundle_inputs(kwargs, node_type=stage, component_manager=Mock()) is kwargs


def test_public_registry_dynamic_schema_and_actual_four_contract_fixture_match():
    from modiff.diffusers_profiles import public_execution_profiles
    from modiff.operation_catalog import resolve_operation

    fixture = json.loads((Path(__file__).parent / "fixtures/qwen_component_bundle_contract_v1.json").read_text())
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed a node")):
        contracts, _ = build_operation_catalog(MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {})
    rows = {(row["pipelineClass"], row["task"], row["operationId"]): row for row in contracts}
    assert len(fixture["contracts"]) == 4
    for contract in fixture["contracts"]:
        assert contract == rows[contract["pipelineClass"], contract["task"], contract["operationId"]]
        if contract["decomposition"] == "loader":
            assert MODULE_MAP["modules.ModularDiffusers"]["ModelsLoader"]["params"]["pipeline_components"]["connectionRole"] == "pipeline_components"
            text = next(port for port in contract["ports"] if port["name"] == "text_encoders")
            assert {member["name"] for member in text["semantics"]["members"]} == {"text_encoder", "tokenizer"}
            continue
        from modiff.component_bundle_contracts import with_qwen_t2i_bundle_contract
        assert with_qwen_t2i_bundle_contract(contract) == contract
        for port in contract["ports"]:
            if "suppliedBy" in port["semantics"]:
                assert set(port["semantics"]["suppliedBy"]["members"]) == {member["name"] for member in port["semantics"]["members"]}
        action = contract["nodeKey"].rsplit(".", 1)[1]
        declaration = MODULE_MAP["modules.ModularDiffusers"][action]["params"]["pipeline_components"]
        assert isinstance(declaration, dict), "the AST registry must resolve the schema factory"
        assert declaration["type"] == "diffusers_modular_pipeline_components"
        assert declaration["required"] is False
        _, config = require_modiff_node_contract(diffusers.QwenImageModularPipeline, contract["nodeType"], resolve_blocks=False)
        assert config["params"]["pipeline_components"]["hidden"] is False
        resolved = resolve_operation(MODULE_MAP, contracts, {
            "pipelineClass": PIPELINE, "task": "text_to_image", "operationId": contract["operationId"],
        })
        assert resolved["params"]["pipeline_components"]["type"] == declaration["type"]
    for contract in contracts:
        if any("suppliedBy" in port.get("semantics", {}) for port in contract["ports"]):
            assert (contract["pipelineClass"], contract["task"]) == (PIPELINE, "text_to_image")


@pytest.mark.parametrize("pipeline", [diffusers.QwenImageModularPipeline, diffusers.QwenImageLayeredModularPipeline,
                                      diffusers.FluxModularPipeline, diffusers.StableDiffusionXLModularPipeline])
@pytest.mark.parametrize("stage", QWEN_T2I_BUNDLE_INPUTS)
def test_resolved_stage_is_exact_authoritative_metadata_without_foreign_facade_fields(pipeline, stage):
    from modules.ModularDiffusers.modular_utils import get_model_type_metadata

    metadata = get_model_type_metadata(pipeline.__name__)["node_params"][stage]
    original = deepcopy(metadata)
    _, resolved = require_modiff_node_contract(pipeline, stage, resolve_blocks=False)
    assert resolved == metadata
    assert ("pipeline_components" in resolved["params"]) is (pipeline is diffusers.QwenImageModularPipeline)
    if pipeline is diffusers.QwenImageModularPipeline:
        assert "pipeline_components" not in resolved["model_input_names"]
        assert "pipeline_components" not in resolved["input_names"]
    resolved["params"].clear()
    assert get_model_type_metadata(pipeline.__name__)["node_params"][stage] == original


def test_authoritative_wrapper_transport_round_trip_preserves_block_dependencies():
    from modules.ModularDiffusers.modular_utils import QWEN_IMAGE_NODE_SPECS
    from modules.ModularDiffusers.pipeline_schema import MoDiffPipelineConfig

    config = MoDiffPipelineConfig(node_specs=QWEN_IMAGE_NODE_SPECS)
    # Built-in callbacks are trusted registry configuration. Custom JSON
    # sidecars intentionally reject these string callbacks at their boundary.
    document = json.loads(config.to_json_string())
    restored = MoDiffPipelineConfig.from_dict(document)
    assert restored.to_dict() == document
    prior_specs = deepcopy(QWEN_IMAGE_NODE_SPECS)
    for stage in QWEN_T2I_BUNDLE_INPUTS:
        prior_specs[stage].pop("wrapper_inputs")
    prior = MoDiffPipelineConfig(node_specs=prior_specs)
    for stage in QWEN_T2I_BUNDLE_INPUTS:
        node = deepcopy(config.node_params[stage])
        assert "pipeline_components" in node["params"]
        node["params"].pop("pipeline_components")
        assert node == prior.node_params[stage]


def test_only_reviewed_text_to_image_task_publishes_or_resolves_the_facade_input():
    from modiff.operation_catalog import resolve_operation

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    leaves = [contract for contract in contracts
              if contract["nodeKey"] in {f"modules.ModularDiffusers.{action}" for action in
                                         ("EncodePrompt", "Denoise", "DecodeLatents")}]
    assert any(contract["pipelineClass"] == PIPELINE and contract["task"] != "text_to_image" for contract in leaves)
    for contract in leaves:
        reviewed = (contract["pipelineClass"], contract["task"]) == (PIPELINE, "text_to_image")
        assert any(port["direction"] == "input" and port["name"] == "pipeline_components"
                   for port in contract["ports"]) is reviewed
        resolved = resolve_operation(MODULE_MAP, contracts, {key: contract[key] for key in
                                                           ("pipelineClass", "task", "operationId")})
        assert ("pipeline_components" in resolved["params"]) is reviewed


def test_opt_in_bundle_edges_preserve_ordinary_auto_owner_task_and_starter_raw_defaults():
    from modiff.operation_starters import resolve_operation_starter
    from modiff.workflow_task_identity import modular_graph_tasks, resource_consumers

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    starter = resolve_operation_starter(MODULE_MAP, contracts, {"pipelineClass": PIPELINE, "task": "text_to_image"})
    nodes = {row["operation"]["operationId"]: {"module": row["module"], "action": row["action"],
             "params": {field: {"value": param.get("value", param.get("default"))}
                        for field, param in row["params"].items() if param.get("display") != "output"}}
             for row in starter["nodes"]}
    loader = "diffusion.load_models"
    for edge in starter["edges"]:
        nodes[edge["target"]]["params"][edge["targetHandle"]] = {"sourceId": edge["source"], "sourceKey": edge["sourceHandle"]}
    assert not any(edge["targetHandle"] == "pipeline_components" for edge in starter["edges"])
    before = resource_consumers(nodes, loader, {loader})
    for operation, fields in [("diffusion.encode_prompt", ["text_encoders"]),
                              ("diffusion.denoise", ["unet", "scheduler"]), ("diffusion.decode_latents", ["vae"])]:
        for field in fields:
            nodes[operation]["params"].pop(field)
        nodes[operation]["params"]["pipeline_components"] = {"sourceId": loader, "sourceKey": "pipeline_components"}
    assert resource_consumers(nodes, loader, {loader}) == before
    assert modular_graph_tasks(nodes, loader, before, PIPELINE) == {"text_to_image"}
