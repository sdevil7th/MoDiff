import pytest
import torch

from modules.Tensor.main import SeededGenerator


@pytest.mark.parametrize("shared_consumer", [False, True])
def test_graph_attempt_shares_one_generator_across_terminal_paths(monkeypatch, tmp_path, shared_consumer):
    from copy import deepcopy
    from modiff.NodeBase import NodeBase
    from modiff.server import WebServer

    calls = []

    class Sample(NodeBase):
        __module__ = "modules.Test.main"

        def execute(self, generator):
            calls.append(self.node_id)
            return {"value": torch.rand(1, generator=generator).item()}

    class Sink(NodeBase):
        __module__ = "modules.Test.main"

        def execute(self, value_input):
            return {"value": value_input}

    registry = {
        "modules.Tensor": {"SeededGenerator": {"params": SeededGenerator.params}},
        "modules.Test": {
            "Sample": {"params": {"generator": {"type": "generator"}, "value": {"display": "output", "type": "float"}}},
            "Sink": {"params": {"value_input": {"type": "float"}, "value": {"display": "output", "type": "float"}}},
        },
    }
    # Exercise the normal registry and NodeBase contracts without model weights.
    monkeypatch.setattr("modiff.NodeBase._module_map", lambda: registry)
    app = WebServer(modules=registry, work_dir=str(tmp_path), data_dir=str(tmp_path))
    app.current_task = {"task_id": "generator-test", "progress": 0}
    app.interrupt_flag = False
    app._prepare_auto_runtime_for_graph = lambda _: None
    app._runtime_fingerprint = lambda: {"fingerprint": "unit"}
    app._runtime_measurement = lambda **_: {"elapsedSeconds": 0}
    app._record_auto_resource_success = lambda *a, **k: None
    app._record_optimization_observations = lambda *a, **k: []
    app.queue_message = lambda *a, **k: None
    app.execute_node = lambda id, node, sid: WebServer.execute_node(app, id, node, sid, quiet=True)
    app.node_cache = {"seed": SeededGenerator("seed"), "sample": Sample("sample")}
    nodes = {
        "seed": {"module": "modules.Tensor", "action": "SeededGenerator", "params": {"seed": {"value": 42}, "device": {"value": "cpu"}}},
        "sample": {"module": "modules.Test", "action": "Sample", "params": {"generator": {"sourceId": "seed", "sourceKey": "generator"}}},
    }
    if shared_consumer:
        for id in ("preview", "export"):
            app.node_cache[id] = Sink(id)
            nodes[id] = {"module": "modules.Test", "action": "Sink", "params": {"value_input": {"sourceId": "sample", "sourceKey": "value"}}}
        paths = [["seed", "sample", "preview"], ["seed", "sample", "export"]]
    else:
        app.node_cache["sample2"] = Sample("sample2")
        nodes["sample2"] = deepcopy(nodes["sample"])
        paths = [["seed", "sample"], ["seed", "sample2"]]
    g = {"sid": "test", "nodes": nodes, "paths": paths}
    expected = torch.rand(2, generator=torch.Generator().manual_seed(42)).tolist()
    generators = []
    for attempt in range(2):
        app.current_task["attempt_index"] = attempt
        app._execute_graph(deepcopy(g))
        generators.append(app.node_cache["seed"].output["generator"])
        assert calls.count("sample") == attempt + 1
        assert app.node_cache["sample"].output["value"] == expected[0]
        if shared_consumer:
            assert app.node_cache["preview"].output == app.node_cache["export"].output
        else:
            assert calls.count("sample2") == attempt + 1
            assert app.node_cache["sample2"].output["value"] == expected[1]
    assert generators[0] is not generators[1]


def test_seeded_generator_dispatch_reinitializes_mutable_state_for_repeat_runs():
    node = SeededGenerator('seeded-generator-test')
    first = node(seed=42, device='cpu')['generator']
    expected = torch.randn(8, generator=first)
    second = node(seed=42, device='cpu')['generator']
    assert second is not first
    torch.testing.assert_close(torch.randn(8, generator=second), expected, rtol=0, atol=0)
    third = node(seed=43, device='cpu')['generator']
    assert not torch.equal(torch.randn(8, generator=third), expected)


def test_seeded_generator_exposes_a_typed_port_accepted_by_ordinary_flux():
    from modules.DiffusersImage.call_inputs import normalize_call_inputs
    generator = SeededGenerator('generator-input-test')(seed=1, device='cpu')['generator']
    assert SeededGenerator.params['generator']['display'] == 'output'
    assert normalize_call_inputs('FluxPipeline', {'generator': generator})['generator'] is generator


def test_seeded_generator_rejects_unavailable_device_with_actionable_error():
    with pytest.raises(RuntimeError, match='choose an available device or cpu'):
        SeededGenerator('bad-generator-test')(seed=1, device='unavailable:0')


@pytest.mark.parametrize('seed', [True, False, -1, 1.5, float('inf'), {}, [], 'bad', 4294967296])
def test_seeded_generator_rejects_invalid_seed_before_dispatch_coercion(seed):
    with pytest.raises(ValueError, match='Seed must be an integer'):
        SeededGenerator('invalid-seed-test')(seed=seed, device='cpu')
