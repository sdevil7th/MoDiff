import json

from modiff.execution_input_provenance import (
    apply_resolved_execution_inputs,
    build_resolved_execution_inputs,
    capture_generation_inputs,
)


def node(params=None):
    return {"module": "modules.Diffusers", "action": "Encode", "params": params or {}}


def test_adapter_alias_origins_follow_the_consumed_control_and_overrides():
    from modiff.execution_input_provenance import bind_generation_input_origins

    graph_node = node({'guidance_scale': {'sourceId': 'source', 'sourceKey': 'value'}})
    record = capture_generation_inputs('generate', graph_node, {'true_cfg_scale': 5})
    mapping = {'true_cfg_scale': 'guidance_scale'}
    bound = bind_generation_input_origins('generate', graph_node, record, source_fields=mapping)
    assert bound['fields']['true_cfg_scale'] == {
        'value': 5, 'source': 'connected', 'sourceNodeId': 'source', 'sourcePortId': 'value',
    }
    overridden = bind_generation_input_origins('generate', graph_node, record,
        {'guidance_scale': 5}, source_fields=mapping)
    assert overridden['fields']['true_cfg_scale'] == {'value': 5, 'source': 'override'}
    assert record['fields']['true_cfg_scale']['source'] == 'literal'


def test_capture_connected_values_not_fallbacks_and_never_walk_objects():
    class Opaque:
        def __str__(self):
            raise AssertionError("Runtime objects must not be stringified")

    source = node({"prompt": {"value": "fallback", "sourceId": "text", "sourceKey": "output"}})
    record = capture_generation_inputs("encode", source, {
        "prompt": "actual wired prompt", "seed": 42, "quant_config": Opaque(),
        "token": "secret", "hf_token": "secret", "components": Opaque(),
        "repo_id": {"source": "hub", "value": "Qwen/Qwen-Image-2512", "token": "secret"},
    }, {"seed": 42})
    assert record["fields"]["prompt"] == {
        "value": "actual wired prompt", "source": "connected", "sourceNodeId": "text", "sourcePortId": "output",
    }
    assert record["fields"]["seed"]["source"] == "override"
    assert record["fields"]["repo_id"]["value"] == "Qwen/Qwen-Image-2512"
    assert record["omittedFields"] == {"quant_config": "unsupported-value"}
    assert "secret" not in json.dumps(record)
    assert source["params"]["prompt"]["value"] == "fallback"


def test_capture_bounds_nonfinite_oversized_nested_and_unsafe_values():
    record = capture_generation_inputs("node", node(), {
        "prompt": "x" * 16385, "seed": 10**1000, "width": float("nan"),
        "height": object(), "negative_prompt": [["nested"]], "prompt_2": ["x"] * 17,
    })
    assert record["fields"] == {}
    assert len(record["omittedFields"]) == 6


def test_independent_redux_scales_remain_distinct_in_the_execution_record():
    record = capture_generation_inputs('redux', node(), {
        'prompt_embeds_scale': [1.0, 0.35], 'pooled_prompt_embeds_scale': [0.8, 0.2],
    })
    assert record['fields']['prompt_embeds_scale']['value'] == [1, 0.35]
    assert record['fields']['pooled_prompt_embeds_scale']['value'] == [0.8, 0.2]


def test_ordinary_loader_identity_survives_an_added_tensor_decoder():
    graph = {'nodes': {
        'loader': node(),
        'generate': node({'pipeline': {'sourceId': 'loader', 'sourceKey': 'pipeline'}}),
        'decode': node({'pipeline': {'sourceId': 'loader', 'sourceKey': 'pipeline'},
                        'latents': {'sourceId': 'generate', 'sourceKey': 'latents_out'}}),
        'preview': node({'image': {'sourceId': 'decode', 'sourceKey': 'images'}}),
    }}
    args = {'loader': {'pipeline_class': 'Flux2KleinKVPipeline', 'model_id': 'owner/model'},
            'generate': {'prompt': 'actual prompt', 'output_type': 'latent'},
            'decode': {'output_type': 'pil'}, 'preview': {}}
    records = {key: capture_generation_inputs(key, value, args[key]) for key, value in graph['nodes'].items()}
    receipt = build_resolved_execution_inputs(graph, records, task_id='latent-run', attempt_index=0, node_id='preview')
    output = apply_resolved_execution_inputs({'taskId': 'latent-run', 'nodeId': 'preview',
        'modelType': 'ZImageModularPipeline'}, receipt)
    assert output['modelType'] == 'Flux2KleinKVPipeline'
    assert output['prompt'] == 'actual prompt'
    assert 'outputType' in receipt['ambiguousFields']
    records['decode'] = capture_generation_inputs('decode', graph['nodes']['decode'], {'model_type': 'AnotherPipeline'})
    ambiguous = build_resolved_execution_inputs(graph, records, task_id='latent-run', attempt_index=0, node_id='preview')
    assert 'modelType' in ambiguous['ambiguousFields']
    assert 'modelType' not in apply_resolved_execution_inputs(output, ambiguous)


def test_builtin_json_producer_does_not_conflict_with_the_image_model_identity():
    graph = {'nodes': {
        'json': {'module': 'modules.Text', 'action': 'ProcessText', 'params': {}},
        'loader': {'module': 'modules.DiffusersImage', 'action': 'LoadPipeline', 'params': {}},
        'generate': node({'pipeline': {'sourceId': 'loader', 'sourceKey': 'pipeline'},
                          'control_mode': {'sourceId': 'json', 'sourceKey': 'output'}}),
    }}
    args = {'json': {'pipeline_class': 'BuiltinDataOperationV1'},
            'loader': {'pipeline_class': 'FluxControlNetPipeline'}, 'generate': {'control_mode': [0, 2]}}
    records = {key: capture_generation_inputs(key, value, args[key]) for key, value in graph['nodes'].items()}
    receipt = build_resolved_execution_inputs(graph, records, task_id='union', attempt_index=0, node_id='generate')
    assert receipt['summary']['modelType'] == 'FluxControlNetPipeline'
    assert receipt['summary']['dataOperationType'] == 'BuiltinDataOperationV1'
    assert receipt['summary']['controlMode'] == [0, 2]
    assert 'modelType' not in receipt['ambiguousFields']


def test_output_ancestry_and_ambiguity_are_not_guessed():
    graph = {"nodes": {
        "text": node(), "encode": node({"prompt": {"sourceId": "text", "sourceKey": "output"}}),
        "generate": node({"state": {"sourceId": "encode", "sourceKey": "state"}}),
        "preview": node({"images": {"sourceId": "generate", "sourceKey": "images"}}),
        "unrelated": node(),
    }}
    records = {key: capture_generation_inputs(key, value, {
        "prompt": "actual" if key == "encode" else "unrelated",
    } if key in {"encode", "unrelated"} else {"width": 1328} if key == "generate" else {}) for key, value in graph["nodes"].items()}
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=1, node_id="preview")
    assert receipt["summary"] == {"prompt": "actual", "width": 1328}
    assert "unrelated" not in [record["nodeId"] for record in receipt["nodes"]]
    output = apply_resolved_execution_inputs({
        "taskId": "run", "attemptIndex": 1, "nodeId": "preview", "prompt": "fallback", "width": 1024,
        "formSnapshot": {"prompt": "fallback"}, "promptSettingsHash": "stale",
    }, receipt)
    assert output["prompt"] == "actual" and output["width"] == 1328
    assert output["formSnapshot"]["prompt"] == "fallback"
    assert "promptSettingsHash" not in output
    assert not output["exactTemplateCompatible"]
    assert "resolvedExecutionInputs" not in apply_resolved_execution_inputs({"taskId": "different"}, receipt)
    records["generate"]["fields"]["prompt"] = {"value": "different upstream prompt", "source": "literal"}
    ambiguous = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=1, node_id="preview")
    assert ambiguous["ambiguousFields"] == ["prompt"]
    assert "prompt" not in ambiguous["summary"]
    assert "prompt" not in apply_resolved_execution_inputs(output, ambiguous)


def test_omitted_inputs_disable_fallback_claim_and_receipts_are_detached():
    graph = {"nodes": {"preview": node()}}
    records = {"preview": capture_generation_inputs("preview", node(), {"prompt": object()})}
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=0, node_id="preview")
    assert receipt["unavailableFields"] == ["prompt"]
    output = apply_resolved_execution_inputs({"taskId": "run", "nodeId": "preview", "prompt": "fallback"}, receipt)
    assert "prompt" not in output
    receipt["nodes"][0]["omittedFields"].clear()
    assert records["preview"]["omittedFields"] == {"prompt": "unsupported-value"}
    assert output["resolvedExecutionInputs"]["nodes"][0]["omittedFields"] == {"prompt": "unsupported-value"}


def test_json_equivalent_numbers_agree_but_booleans_do_not_and_mismatched_receipts_are_removed():
    graph = {"nodes": {"a": node(), "b": node({"input": {"sourceId": "a", "sourceKey": "result"}})}}
    records = {"a": capture_generation_inputs("a", graph["nodes"]["a"], {"guidance_scale": 4}),
               "b": capture_generation_inputs("b", graph["nodes"]["b"], {"guidance_scale": 4.0})}
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=0, node_id="b")
    assert receipt["summary"] == {"guidanceScale": 4}
    assert receipt["ambiguousFields"] == []
    records["a"]["fields"]["guidance_scale"]["value"] = True
    records["b"]["fields"]["guidance_scale"]["value"] = 1
    receipt = build_resolved_execution_inputs(graph, records, task_id="run", attempt_index=0, node_id="b")
    assert receipt["ambiguousFields"] == ["guidanceScale"]
    wrong = {"taskId": "other", "nodeId": "b", "resolvedExecutionInputs": receipt}
    assert "resolvedExecutionInputs" not in apply_resolved_execution_inputs(wrong, receipt)
