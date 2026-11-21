"""Official model-local compatibility and cache boundaries; no trained weights/GPU."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from modiff import transformers_image_text_resources as resources
from modiff.prompt_enhancement import _declared_contract


def selection():
    contract = _declared_contract()
    return {'source': 'hub', 'value': contract['repository']}, contract['revision']


@pytest.fixture
def simulated_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, 'hip', 'unit-rocm')
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
        lambda _device: SimpleNamespace(gcnArchName='gfx1151:sramecc+:xnack-'))


@pytest.mark.parametrize('change', [
    {'dtype': 'float32'}, {'dtype': 'float16'}, {'quantization': 'bnb_4bit'},
    {'device': 'cpu:0'}, {'device': 'cuda:1'}, {'revision': 'a' * 40},
    {'selection': {'source': 'hub', 'value': 'other/model'}},
    {'architecture': 'gfx1100'}, {'architecture': ''}, {'hip': None},
])
def test_unaffected_precision_quantization_device_and_artifacts_keep_defaults(simulated_rocm, monkeypatch, change):
    selected, revision = selection()
    values = dict(selection=selected, revision=revision, dtype='bfloat16', device='cuda:0', quantization='none')
    if 'architecture' in change:
        monkeypatch.setattr(torch.cuda, 'get_device_properties',
            lambda _device: SimpleNamespace(gcnArchName=change['architecture']))
    elif 'hip' in change:
        monkeypatch.setattr(torch.version, 'hip', change['hip'])
    else:
        values.update(change)
    assert resources.image_text_vision_attention_policy(**values) is None


@pytest.fixture
def loaded_native_config_model(simulated_rocm, monkeypatch):
    """Tiny actual official class; the pretrained/device factories are unit substitutes."""
    import transformers
    from modules.HuggingFaceTransformers import main as nodes
    config = transformers.Qwen3_5Config(
        text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
            head_dim=32, layer_types=['full_attention']),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
            num_heads=1, out_hidden_size=32, num_position_embeddings=64),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        model = transformers.Qwen3_5ForConditionalGeneration(config).eval()
    model.set_attn_implementation('sdpa')
    parameter_dtypes = {name: tensor.dtype for name, tensor in model.named_parameters()}
    calls = []
    native_set = model.set_attn_implementation
    def set_attention(value):
        calls.append(deepcopy(value))
        return native_set(value)
    monkeypatch.setattr(model, 'set_attn_implementation', set_attention)
    monkeypatch.setattr(model, 'to', lambda _device: model)
    monkeypatch.setattr(nodes, '_normalized_device', lambda _value: 'cuda:0')
    monkeypatch.setattr('modiff.prompt_enhancement.validate_cached_selection', lambda *_args: None)
    monkeypatch.setattr(transformers.AutoProcessor, 'from_pretrained', lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, 'from_pretrained', lambda *_args, **_kwargs: model)
    selected, revision = selection()
    handle, receipt = nodes._load_model(selection_value=selected, revision_value=revision,
        dtype_value='bfloat16', device_value='cuda:0', task='image-video-to-text')
    return model, handle, receipt, calls, parameter_dtypes


def test_official_vision_only_policy_keeps_receipt_schema_root_text_and_tensor_dtypes(loaded_native_config_model):
    from modules.HuggingFaceTransformers import main as nodes
    model, handle, receipt, calls, dtypes = loaded_native_config_model
    selected, revision = selection()
    expected = resources.image_text_vision_attention_policy(selected, revision, 'bfloat16', 'cuda:0', 'none')
    assert set(receipt['runtime']) == {'transformersVersion', 'dtype', 'device', 'preprocessorClass',
        'modelClass', 'modelType', 'architectures', 'quantization'}
    assert calls == [{'vision_config': 'eager'}]
    assert resources._image_text_attention_state(model) == {key: expected[key] for key in ('root', 'text', 'vision')}
    assert {name: tensor.dtype for name, tensor in model.named_parameters()} == dtypes
    assert nodes._validated_handle(handle, task='image-video-to-text')[0] is model


@pytest.mark.parametrize('fault', ['vision', 'root', 'text'])
def test_stale_or_changed_cached_attention_refuses_before_generation(loaded_native_config_model, fault):
    from modules.HuggingFaceTransformers import main as nodes
    model, handle, receipt, _calls, _dtypes = loaded_native_config_model
    receipt = deepcopy(receipt)
    name = {'root': '', 'text': 'text_config', 'vision': 'vision_config'}[fault]
    model.set_attn_implementation({name: 'sdpa' if fault == 'vision' else 'eager'})
    receipt.pop('receiptDigest')
    handle['receipt'] = nodes._sealed_receipt(receipt)
    with pytest.raises(ValueError, match='Reload the model'):
        nodes._validated_handle(handle, task='image-video-to-text')


@pytest.mark.parametrize('dtype,quantization', [('float32', 'none'), ('bfloat16', 'bnb_4bit')])
def test_unaffected_cached_receipts_need_no_new_field_or_hardware_query(simulated_rocm, monkeypatch, dtype, quantization):
    def forbidden(*_args):
        pytest.fail('Unaffected cached handle queried GPU properties')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', forbidden)
    selected, revision = selection()
    resources.validate_image_text_vision_attention_policy(SimpleNamespace(),
        {'kind': 'hub', 'repository': selected['value'], 'revision': revision},
        {'dtype': dtype, 'device': 'cuda:0', 'quantization': {'mode': quantization}})


def test_non_native_or_changed_initial_attention_is_not_silently_rewritten(simulated_rocm):
    selected, revision = selection()
    policy = resources.image_text_vision_attention_policy(selected, revision, 'bfloat16', 'cuda:0', 'none')
    with pytest.raises(ValueError, match='native SDPA model'):
        resources.apply_image_text_vision_attention_policy(SimpleNamespace(), policy)
