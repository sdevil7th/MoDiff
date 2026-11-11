"""Root-run actual ModelCPU PEFT transfer; synthetic weights, no fit grant."""
from copy import deepcopy
import json
import os
import weakref

import pytest
import torch

import test_flux3_native_lora as native_tests
import test_flux3_bfl_lora as bfl_tests
import test_modular_weight_formats as weight_format_tests
from test_flux3_lora_cuda import prediction
from test_flux3_lora_placement import placement_graph

pytestmark = pytest.mark.skipif(os.environ.get("MODIFF_RUN_FLUX3_TINY_CUDA") != "1",
                                reason="Explicit isolated root-run CUDA lane only.")
native_runtime = native_tests.runtime
artifacts = weight_format_tests.runtime
bfl_runtime = bfl_tests.runtime


def factor_tokens(runtime):
    model = runtime.server.node_cache["models"].loader.transformer
    return {name: {"objectId": id(value), "version": value._version,
                   "device": str(value.device), "dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in model.named_parameters() if ".lora_A." in name or ".lora_B." in name}


def hook_tokens(runtime):
    from diffusers.modular_pipelines.components_manager import CustomOffloadHook, UserCustomOffloadHook
    model = runtime.server.node_cache["models"].loader.transformer
    hook = vars(model).get("_hf_hook")
    assert type(hook) is CustomOffloadHook and hook.execution_device == torch.device("cuda:0")
    rows = [row for row in runtime.manager.model_hooks
            if type(row) is UserCustomOffloadHook and row.model is model and row.hook is hook]
    assert len(rows) == 1
    row = rows[0]
    assert runtime.manager.components[row.model_id] is model
    assert row.model_id in runtime.manager.collections["models"]
    return id(model), id(hook), id(row), row.model_id


@pytest.mark.parametrize("pair", native_tests.PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize("scheme", ["native", "nvfp4"])
def test_real_model_cpu_transfer_retains_unchanged_lora_owner(request, monkeypatch, tmp_path, pair, scheme):
    assert torch.cuda.is_available(), "No CPU device emulation can pass the transfer counterexample."
    runtime = request.getfixturevalue("native_runtime" if scheme == "native" else "bfl_runtime")
    if scheme == "native":
        # Same acquisition-only dtype adaptation as the maintained CUDA lane.
        from diffusers import ComponentSpec
        original = ComponentSpec.load

        def construct(spec, **kwargs):
            result = original(spec, **kwargs)
            dtype = kwargs.get("torch_dtype")
            if isinstance(result, torch.nn.Module) and isinstance(dtype, torch.dtype):
                result.to(dtype=dtype)
            return result

        monkeypatch.setattr(ComponentSpec, "load", construct)
    graph = placement_graph(pair, [native_tests.descriptor(tmp_path, original=True)],
                            mode="model_cpu", scheme=scheme)
    graph["nodes"]["models"]["params"].update(device={"value": "cuda:0"}, dtype={"value": "bfloat16"})
    node = runtime.run(graph)
    owner = weakref.ref(node.loader)
    assert node.loader.vae.dtype == torch.float32
    before = factor_tokens(runtime)
    assert before and {row["device"] for row in before.values()} == {"cpu"}
    hooks = hook_tokens(runtime)
    from modiff.native_disk_lora import disk_lora_binding_matches
    assert disk_lora_binding_matches(node, node._native_disk_lora_state["policy"], runtime.manager)
    del node
    expected = prediction(runtime)
    after = factor_tokens(runtime)
    assert {row["device"] for row in after.values()} == {"cuda:0"}
    assert {name: {key: value for key, value in row.items() if key != "device"} for name, row in before.items()} == {
        name: {key: value for key, value in row.items() if key != "device"} for name, row in after.items()}
    assert hook_tokens(runtime) == hooks
    node = runtime.server.node_cache["models"]
    binding_matches = disk_lora_binding_matches(node, node._native_disk_lora_state["policy"], runtime.manager)
    print("ACTUAL_MODEL_CPU_TRANSFER", json.dumps({"before": before, "after": after,
          "hookIdentityUnchanged": True, "bindingMatchesAfterOfficialForward": binding_matches}), flush=True)
    del node
    # The ordinary next graph dispatch must reuse the same actual manager owner.
    node = runtime.run(deepcopy(graph))
    assert node.loader is owner(), "Official CPU-to-CUDA transfer invalidated an unchanged typed LoRA owner."
    assert binding_matches, "The completed owner seal rejected only its official placement transition."
    assert hook_tokens(runtime) == hooks
    torch.testing.assert_close(prediction(runtime), expected, rtol=0, atol=0)
