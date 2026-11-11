"""Publisher PEFT metadata is interpreted by official code, never ignored."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from modiff.flux3_lora_contract import flux3_factor_header


def metadata():
    path = Path(__file__).parent / "fixtures/flux3-lora-peft-metadata.v1.json"
    return json.loads(path.read_text(encoding="utf-8"))["metadata"]


def artifact(tmp_path, values):
    config = json.loads(values["lora_adapter_metadata"])
    targets = config["transformer.target_modules"]
    state = {}
    for target in targets:
        prefix = "transformer.mode_blocks.image.0." + target
        state[prefix + ".lora_A.weight"] = torch.ones(16, 32)
        state[prefix + ".lora_B.weight"] = torch.ones(32, 16)
    path = tmp_path / "adapter.safetensors"
    save_file(state, path, metadata=values)
    return SimpleNamespace(load_directory=tmp_path, weight_name=path.name)


def test_actual_publisher_inactive_peft_metadata_is_not_blanket_refused(tmp_path):
    values = metadata()
    before = deepcopy(values)
    header = flux3_factor_header(artifact(tmp_path, values))
    assert len(header) == 12
    assert values == before


def test_official_pipeline_interprets_reviewed_peft_metadata_unchanged(tmp_path):
    import test_flux3_native_lora as native_tests
    native_tests.require_flux3_api()
    from diffusers import Flux3ModularPipeline

    model = native_tests.tiny_transformer(native_tests.PAIRS[0][0])
    values = metadata()
    config = json.loads(values["lora_adapter_metadata"])
    targets = config["transformer.target_modules"]
    state = {}
    generator = torch.Generator().manual_seed(41)
    for name, module in model.named_modules():
        if type(module) is torch.nn.Linear and any(name == key or name.endswith("." + key) for key in targets):
            state["transformer." + name + ".lora_A.weight"] = torch.randn(16, module.in_features, generator=generator) * .01
            state["transformer." + name + ".lora_B.weight"] = torch.randn(module.out_features, 16, generator=generator) * .01
    path = tmp_path / "adapter.safetensors"
    save_file(state, path, metadata=values)
    flux3_factor_header(SimpleNamespace(load_directory=tmp_path, weight_name=path.name))
    pipeline = Flux3ModularPipeline(workflow="text2image")
    pipeline.register_components(transformer=model)
    pipeline.load_lora_weights(str(tmp_path), weight_name=path.name, adapter_name="reviewed",
                               use_safetensors=True, local_files_only=True, low_cpu_mem_usage=False)
    actual = model.peft_config["reviewed"]
    assert actual.r == actual.lora_alpha == 16
    assert set(actual.target_modules) == set(targets)
    assert actual.use_dora is False and actual.use_rslora is False and actual.use_qalora is False
    assert actual.bias == "none" and actual.lora_bias is False
    assert model.active_adapters() == ["reviewed"]
    assert values == metadata()


@pytest.mark.parametrize("key,value", [
    ("transformer.use_dora", True), ("transformer.use_rslora", True),
    ("transformer.use_qalora", True), ("transformer.modules_to_save", ["embed_tokens"]),
    ("transformer.lora_bias", True), ("transformer.use_dora", 0),
    ("transformer.r", True), ("transformer.r", 8),
    ("transformer.lora_alpha", float("nan")), ("transformer.fan_in_fan_out", True),
    ("text_encoder.r", 16), ("transformer.unknown", None),
])
def test_metadata_cannot_enable_unreviewed_or_mismatched_behavior(tmp_path, key, value):
    values = metadata()
    config = json.loads(values["lora_adapter_metadata"])
    config[key] = value
    values["lora_adapter_metadata"] = json.dumps(config)
    with pytest.raises(ValueError):
        flux3_factor_header(artifact(tmp_path, values))
