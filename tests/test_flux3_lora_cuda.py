"""Explicit root-run tiny CUDA transfer lane; no pretrained/fit/UI grant.

Acquisition substitutes synthetic tiny fixtures. Official converter/PEFT/hooks,
real ModelsLoader/source/owner dispatch and actual CUDA computation execute.
Invoke only in the source-bound isolated native harness, never a live app.
"""
from copy import deepcopy
import os
import weakref

import pytest
import torch

import test_flux3_native_lora as native_tests
import test_flux3_bfl_lora as bfl_tests
import test_modular_weight_formats as weight_format_tests
from test_flux3_lora_placement import placement_graph

pytestmark = pytest.mark.skipif(os.environ.get("MODIFF_RUN_FLUX3_TINY_CUDA") != "1",
                                reason="Explicit isolated root-run CUDA lane only.")
native_runtime = native_tests.runtime
artifacts = weight_format_tests.runtime
bfl_runtime = bfl_tests.runtime


def prediction(runtime):
    generator = torch.Generator().manual_seed(31)
    value = dict(hidden_states=torch.randn(1, 8, 4, generator=generator).to("cuda:0", torch.bfloat16),
                 encoder_hidden_states=torch.randn(1, 8, 32, generator=generator).to("cuda:0", torch.bfloat16),
                 timestep=torch.tensor([.5], device="cuda:0"),
                 img_ids=torch.zeros(8, 4, device="cuda:0"), txt_ids=torch.zeros(8, 4, device="cuda:0"))
    model = runtime.server.node_cache["models"].loader.transformer
    if model.config.flow_map:
        value["target_timestep"] = torch.tensor([.25], device="cuda:0")
    result = model(**value).sample.detach().clone()
    torch.cuda.synchronize()
    assert result.device.type == "cuda" and result.dtype == torch.bfloat16 and torch.isfinite(result).all()
    return result


@pytest.mark.parametrize("pair", native_tests.PAIRS, ids=lambda row: row[0])
@pytest.mark.parametrize("scheme", ["native", "fp8r", "nvfp4"])
def test_selected_group_cuda_lora_transfer_packing_and_warm_owner(request, monkeypatch, tmp_path, pair, scheme):
    assert torch.cuda.is_available(), "Explicit CUDA lane must not pass through a CPU fallback."
    runtime = request.getfixturevalue("native_runtime" if scheme == "native" else "bfl_runtime")
    if scheme == "native":
        # The native fixture constructs random components rather than reading
        # trained files; apply the actual per-component dtype passed by loader.
        from diffusers import ComponentSpec
        original = ComponentSpec.load

        def construct(spec, **kwargs):
            result = original(spec, **kwargs)
            dtype = kwargs.get("torch_dtype")
            if isinstance(result, torch.nn.Module) and isinstance(dtype, torch.dtype):
                result.to(dtype=dtype)
            return result

        monkeypatch.setattr(ComponentSpec, "load", construct)
    source = placement_graph(pair, scheme=scheme)
    values = source["nodes"]["models"]["params"]
    values["device"] = {"value": "cuda:0"}
    values["dtype"] = {"value": "bfloat16"}
    baseline_node = runtime.run(source)
    assert baseline_node.loader.vae.dtype == torch.float32
    baseline = prediction(runtime)
    packed = bfl_tests.packed_state(baseline_node.loader.transformer) if scheme != "native" else None
    former = weakref.ref(baseline_node.loader)
    del baseline_node
    item = native_tests.descriptor(tmp_path, original=True)
    selected = deepcopy(source)
    selected["nodes"]["models"]["params"]["lora_list"] = {"value": [item]}
    node = runtime.run(selected)
    assert former() is None
    changed = prediction(runtime)
    assert not torch.allclose(changed, baseline)
    if packed is not None:
        bfl_tests.assert_packed_state(node.loader.transformer, packed)
    owner = weakref.ref(node.loader)
    assert runtime.run(deepcopy(selected)).loader is owner()
    torch.testing.assert_close(prediction(runtime), changed, rtol=0, atol=0)
    if packed is not None:
        bfl_tests.assert_packed_state(node.loader.transformer, packed)
    assert node.loader.vae.dtype == torch.float32
