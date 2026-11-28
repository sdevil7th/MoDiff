"""Reference assembly publishes the same conditional contract through the AST registry."""
from pathlib import Path
import json

import pytest


@pytest.mark.parametrize('action', ['WorkflowMiniMaxH3ReferenceAssembler', 'WorkflowJointAVReferenceAssembler'])
def test_reference_assembler_registry_keeps_validation_and_conditional_media(action):
    import modules
    import modules.ModularDiffusers as package
    from modules.ModularDiffusers import workflow_blocks

    parsed = modules.parse_module_file(str(Path(workflow_blocks.__file__)), package)[action]
    assert parsed['skipParamsCheck'] is True
    params = parsed['params']
    fixture = json.loads((Path(__file__).parent / 'fixtures/reference-input-requirements.json').read_bytes())
    for field, metadata in fixture.items():
        assert {key: params[field][key] for key in metadata} == metadata
    # Video soundtrack is optional: only audio reference mode requires audio.
    assert params['audio']['requiredWhen']['equals'] != 'video'


@pytest.mark.parametrize('action', ['WorkflowMiniMaxH3ReferenceAssembler', 'WorkflowJointAVReferenceAssembler'])
def test_normal_reference_dispatch_preserves_order_and_selected_input_validation(action, monkeypatch):
    from dataclasses import dataclass
    from PIL import Image
    from modules.ModularDiffusers import workflow_blocks

    class Reference:
        kind = 'image'

    @dataclass
    class ImageReference(Reference):
        image: object

    node = getattr(workflow_blocks, action)('reference-dispatch')
    monkeypatch.setattr(node, '_reference_types', lambda _: (Reference, ImageReference, None, None))
    assert node._skip_params_check is True
    original = ImageReference(Image.new('RGB', (8, 8)))
    added = Image.new('RGB', (8, 8), 'red')
    result = node(reference_kind='image', image=added, references_in=[original])['references']
    assert result[0] is original and result[1].image is added
    # These failures must remain at normal graph dispatch, not become success
    # because discovery now publishes the inherited skip flag correctly.
    for kind, kwargs, message in [
        ('image', {}, 'decoded PIL image'),
        ('video', {}, 'nonempty list'),
        ('audio', {}, 'decoded audio'),
        ('unsupported', {'image': added}, 'kind must be'),
    ]:
        with pytest.raises(RuntimeError, match=message):
            node(reference_kind=kind, **kwargs)


@pytest.mark.parametrize('pipeline', ['MiniMaxH3ModularPipeline', 'PikaH3FlexModularPipeline'])
def test_new_reference_starter_keeps_selected_image_required_for_media_completion(pipeline):
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    result = resolve_operation_starter(MODULE_MAP, contracts,
        {'pipelineClass': pipeline, 'task': 'reference_to_video_with_audio'})
    helper = next(n for n in result['nodes'] if n['action'] == 'WorkflowJointAVReferenceAssembler')
    required = {p['field'] for p in result['requiredInputs']
                if p['operationId'] == helper['operation']['operationId']}
    assert required == {'image'}
