"""Service interfaces use reviewed dynamic fields, never client type hints."""

from copy import deepcopy
from unittest.mock import patch

import pytest

from modiff import service_package as service
from modules import MODULE_MAP


def graph(pipeline="ZImageModularPipeline"):
    def node(action, params):
        return {"module": "modules.ModularDiffusers", "action": action, "params": params}
    return {"nodes": {
        "load": node("ModelsLoader", {"model_type": {"value": pipeline}}),
        "encode": node("EncodePrompt", {
            "text_encoders": {"sourceId": "load", "sourceKey": "text_encoders"},
            "prompt": {"value": "Rain", "type": "int"},
            "invented": {"value": "not declared", "type": "string"},
        }),
        "denoise": node("Denoise", {
            "unet": {"sourceId": "load", "sourceKey": "unet"},
            "seed": {"value": 9}, "num_inference_steps": {"value": 8},
        }),
    }, "paths": [["load", "encode", "denoise"]]}


@pytest.mark.parametrize("pipeline", ["ZImageModularPipeline", "FluxModularPipeline", "QwenImageModularPipeline"])
def test_reviewed_modular_fields_are_discovered_without_constructing_nodes(pipeline):
    g = graph(pipeline)
    before = deepcopy(g)
    with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed node")):
        fields = service.inspect_graph(g, MODULE_MAP)["inputs"]
    assert {"nodeId": "encode", "field": "prompt", "type": "string"} in fields
    assert {"nodeId": "denoise", "field": "seed", "type": "int"} in fields
    assert not any(f["field"] in {"invented", "model_type"} for f in fields)
    assert g == before


@pytest.mark.parametrize("change", ["disconnected", "unknown", "custom", "connected_identity", "ambiguous"])
def test_unresolved_modular_ownership_never_trusts_graph_types(change):
    g = graph()
    if change == "disconnected":
        g["nodes"]["encode"]["params"]["text_encoders"] = {"value": None}
    elif change == "unknown":
        g["nodes"]["load"]["params"]["model_type"]["value"] = "UnknownPipeline"
    elif change == "custom":
        g["nodes"]["load"]["params"]["model_type"]["value"] = "CustomModularPipeline"
    elif change == "connected_identity":
        g["nodes"]["selector"] = {"module": "modules.Primitive", "action": "TextValue", "params": {"text": {"value": "ZImageModularPipeline"}}}
        g["nodes"]["load"]["params"]["model_type"] = {"value": "ZImageModularPipeline", "sourceId": "selector", "sourceKey": "output"}
        g["paths"] = [["selector", "load", "encode", "denoise"]]
    else:
        g["nodes"]["other"] = deepcopy(g["nodes"]["load"])
        g["nodes"]["encode"]["params"]["extra_model"] = {"sourceId": "other", "sourceKey": "text_encoders"}
        g["paths"] = [["load", "other", "encode", "denoise"]]
    assert not any(f["nodeId"] == "encode" for f in service.inspect_graph(g, MODULE_MAP)["inputs"])


def test_modular_service_roundtrip_keeps_dynamic_types_and_rejects_wrong_invocation_values():
    g = graph()
    g['nodes']['load']['params'].update({
        'repo_id': {'value': {'source': 'hub', 'value': 'Tongyi-MAI/Z-Image-Turbo'}},
        'revision': {'value': 'f332072aa78be7aecdf3ee76d5c247082da564a6'},
    })
    g['nodes']['denoise']['params']['embeddings'] = {'sourceId': 'encode', 'sourceKey': 'embeddings'}
    g['nodes']['decode'] = {'module': 'modules.ModularDiffusers', 'action': 'DecodeLatents', 'params': {
        'vae': {'sourceId': 'load', 'sourceKey': 'vae'},
        'latents': {'sourceId': 'denoise', 'sourceKey': 'latents'},
    }}
    g['nodes']['preview'] = {'module': 'modules.Image', 'action': 'Preview', 'params': {
        'image': {'sourceId': 'decode', 'sourceKey': 'images'},
        'preview': {'display': 'ui_image', 'sourceKey': 'output'},
    }}
    g['paths'][0].extend(['decode', 'preview'])
    interface = {'inputs': {'prompt': [{'nodeId': 'encode', 'field': 'prompt'}],
                            'seed': [{'nodeId': 'denoise', 'field': 'seed'}]},
                 'outputs': {'image': [{'nodeId': 'preview', 'field': 'preview'}]}}
    contract = {'backend': {}, 'packages': {}, 'customNodes': [], 'optionalProfiles': []}
    package = service.build_package(g, interface, registry=MODULE_MAP, contract=contract)
    values = {'prompt': 'A glass vessel', 'seed': 42}
    prepared = service.prepare_package(package, values, registry=MODULE_MAP, contract=contract, sid='test')
    assert prepared['nodes']['encode']['params']['prompt']['value'] == values['prompt']
    assert prepared['nodes']['denoise']['params']['seed']['value'] == 42
    assert package['graph']['nodes']['encode']['params']['prompt']['value'] is None
    for invalid in ({'prompt': ['a'], 'seed': 42}, {'prompt': 'a', 'seed': True}):
        with pytest.raises(ValueError, match='requires'):
            service.prepare_package(package, invalid, registry=MODULE_MAP, contract=contract, sid='test')
