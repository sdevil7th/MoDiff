"""Fresh operation schemas expose declared formats without executing a loader."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from modules import MODULE_MAP
from modiff.operation_catalog import build_operation_catalog, resolve_operation


@pytest.mark.parametrize('pipeline,task', [
    ('Flux3DistilledModularPipeline', 'text_to_image'),
    ('Flux3DistilledModularPipeline', 'edit_image'),
    ('Flux3ModularPipeline', 'text_to_image'),
    ('Flux3ModularPipeline', 'edit_image'),
    ('FluxModularPipeline', 'text_to_image'),
    ('QwenImageModularPipeline', 'text_to_image'),
])
def test_fresh_loader_format_control_uses_registered_choices_without_registry_mutation(pipeline, task):
    import diffusers
    from modules.ModularDiffusers.modular_utils import get_model_type_metadata
    from modiff.modular_weight_formats import declared_weight_formats

    if not hasattr(diffusers, pipeline):
        pytest.skip('This class requires the explicit reviewed source fixture or evaluation runtime.')
    original = deepcopy(MODULE_MAP)
    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    selection = {'pipelineClass': pipeline, 'task': task, 'operationId': 'diffusion.load_models'}
    expected = list(declared_weight_formats(pipeline, get_model_type_metadata(pipeline)))
    raw = original['modules.ModularDiffusers']['ModelsLoader']['params']['weight_format']
    with (
        patch('socket.socket.connect', side_effect=AssertionError('Network during authoring')),
        patch('modiff.NodeBase.NodeBase.__init__', side_effect=AssertionError('Constructed node')),
    ):
        resolved = resolve_operation(MODULE_MAP, contracts, selection)
        field = resolved['params']['weight_format']
        assert field['options'] == expected
        assert field['hidden'] is (len(expected) == 1)
        assert field['default'] == raw['default'] == 'native'
        assert ('value' in field) is ('value' in raw)
        assert field.get('value', field['default']) == raw.get('value', raw['default']) == 'native'
        assert resolved['values'].get('weight_format', 'native') == 'native'
        # A new draft owns its choices; editing one response cannot affect a
        # later draft, the registered raw node or an already saved definition.
        field['options'].append('unregistered')
        renewed = resolve_operation(MODULE_MAP, contracts, selection)
        assert renewed['params']['weight_format']['options'] == expected
    assert MODULE_MAP == original
