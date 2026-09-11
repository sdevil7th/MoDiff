"""Every pinned ordinary FLUX call argument needs an explicit graph/owner classification.

This inventory complements real tensor tests; a classified field alone is not
execution or image-quality evidence. App-owned callbacks are not user code.
"""
import ast
import importlib.util
from pathlib import Path

import pytest


def upstream_signatures():
    root = Path(importlib.util.find_spec('diffusers').origin).parent / 'pipelines'
    result = {}
    for path in root.glob('flux*/pipeline_flux*.py'):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                for method in node.body:
                    if isinstance(method, ast.FunctionDef) and method.name == '__call__':
                        result[node.name] = {
                            arg.arg for arg in method.args.args + method.args.kwonlyargs
                        } - {'self'}
    return result


def test_all_pinned_flux_call_arguments_have_an_explicit_owner():
    from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS, image_pipeline_contract
    from modules.DiffusersImage.call_inputs import PIPELINE_CALL_INPUTS

    app_owned = {'return_dict', 'callback_on_step_end', 'callback_on_step_end_tensor_inputs'}
    primary = {'prompt', 'image', 'mask_image', 'control_image', 'num_inference_steps'}
    signatures = upstream_signatures()
    assert len(PIPELINE_CALL_INPUTS) == 17
    reviewed_exports = {
        'FluxPriorReduxPipeline' if name == 'FluxReduxPipeline' else name
        for name in PIPELINE_CALL_INPUTS
    }
    assert set(signatures) == reviewed_exports, (
        f'Pinned FLUX exports changed: missing={sorted(set(signatures) - reviewed_exports)}, '
        f'stale={sorted(reviewed_exports - set(signatures))}'
    )
    for name in PIPELINE_CALL_INPUTS:
        upstream = 'FluxPriorReduxPipeline' if name == 'FluxReduxPipeline' else name
        adapter = IMAGE_PIPELINE_ADAPTERS[name]
        fields = set().union(*(
            set(image_pipeline_contract(adapter, mode)['fieldParams']) for mode in adapter.modes
        ))
        aliases = {adapter.guidance_parameter, adapter.secondary_guidance_parameter,
                   adapter.conditioning_scale_parameter} - {None}
        missing = signatures[upstream] - fields - primary - aliases - app_owned
        assert not missing, f'{upstream} has unclassified public arguments: {sorted(missing)}'


def test_redux_secondary_prompt_and_supplied_embeddings_reach_the_prior_only():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from PIL import Image
    import torch
    from modules.DiffusersImage.main import FluxReduxPipelineBundle
    from modules.DiffusersImage.call_inputs import normalize_call_inputs

    image = Image.new('RGB', (32, 32))
    supplied, pooled = torch.ones(1, 3, 8), torch.ones(1, 8)
    processed, processed_pooled = torch.zeros(1, 4, 8), torch.zeros(1, 8)
    prior = Mock(return_value=SimpleNamespace(prompt_embeds=processed, pooled_prompt_embeds=processed_pooled))
    base = Mock(return_value=SimpleNamespace(images=[image]))
    bundle = FluxReduxPipelineBundle(prior, base)
    bundle(image=image, prompt='primary', prompt_2='secondary wording')
    assert prior.call_args.kwargs['prompt_2'] == 'secondary wording'
    assert 'prompt_2' not in base.call_args.kwargs
    options = normalize_call_inputs('FluxReduxPipeline', {
        'prompt_embeds': supplied, 'pooled_prompt_embeds': pooled,
    })
    bundle(image=image, prompt=None, **options)
    assert prior.call_args.kwargs['prompt_embeds'] is supplied
    assert prior.call_args.kwargs['pooled_prompt_embeds'] is pooled
    assert base.call_args.kwargs['prompt_embeds'] is processed
    assert base.call_args.kwargs['pooled_prompt_embeds'] is processed_pooled
    with pytest.raises(ValueError, match='connected together'):
        normalize_call_inputs('FluxReduxPipeline', {'prompt_embeds': supplied})


def test_redux_contract_revision_preserves_only_exact_historical_fields():
    from copy import deepcopy
    from modules.DiffusersImage.main import (
        IMAGE_PIPELINE_ADAPTERS, image_pipeline_contract, _compatible_image_contract,
    )

    for mode in ('edit_image', 'multi_image_reference_edit'):
        current = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS['FluxReduxPipeline'], mode)
        previous = deepcopy(current)
        for key in ('prompt_2', 'prompt_embeds', 'pooled_prompt_embeds'):
            previous['fieldParams'].pop(key)
        assert _compatible_image_contract(previous, current)
        broken = deepcopy(previous)
        broken['fieldParams'].pop('width')
        assert not _compatible_image_contract(broken, current)
    current = image_pipeline_contract(IMAGE_PIPELINE_ADAPTERS['FluxPipeline'], 'text_to_image')
    broken = deepcopy(current)
    for key in ('prompt_2', 'prompt_embeds', 'pooled_prompt_embeds'):
        broken['fieldParams'].pop(key)
    assert not _compatible_image_contract(broken, current)


def test_redux_bundle_matches_actual_upstream_prior_with_tiny_components():
    """Real SigLIP/Redux forward and fusion, not a mocked prior or weight download."""
    from unittest.mock import Mock
    pytest.importorskip('transformers')
    import torch
    from PIL import Image
    from diffusers import FluxPriorReduxPipeline
    from diffusers.pipelines.flux.modeling_flux import ReduxImageEncoder
    from transformers import CLIPTextConfig, CLIPTextModel, SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor
    from modules.DiffusersImage.main import FluxReduxPipelineBundle

    torch.manual_seed(812)
    prior = FluxPriorReduxPipeline(
        image_encoder=SiglipVisionModel(SiglipVisionConfig(hidden_size=8, intermediate_size=16,
            num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=4)),
        feature_extractor=SiglipImageProcessor(size={'height': 16, 'width': 16}),
        image_embedder=ReduxImageEncoder(redux_dim=8, txt_in_features=8),
        text_encoder=CLIPTextModel(CLIPTextConfig(vocab_size=16, hidden_size=8,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2)),
    )
    images = [Image.new('RGB', (16, 16), color) for color in ('red', 'blue')]
    options = dict(prompt_embeds=torch.randn(2, 3, 8), pooled_prompt_embeds=torch.randn(2, 8),
                   prompt_embeds_scale=[1.0, 0.35], pooled_prompt_embeds_scale=[0.8, 0.2])
    expected = prior(image=images, **{key: value.clone() if torch.is_tensor(value) else value
                                    for key, value in options.items()})
    base = Mock()
    FluxReduxPipelineBundle(prior, base)(image=images, prompt='saved text stays authored', **options)
    actual = base.call_args.kwargs
    torch.testing.assert_close(actual['prompt_embeds'], expected.prompt_embeds, rtol=0, atol=0)
    torch.testing.assert_close(actual['pooled_prompt_embeds'], expected.pooled_prompt_embeds, rtol=0, atol=0)
    assert actual['prompt_embeds'].shape == (1, 19, 8)
    assert 'prompt_2' not in actual
