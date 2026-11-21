"""Real stage facade captures the actual sealed geometry and Control RNG seed."""
import pytest

from image_stage_test_support import stage_offline_runtime as _stage_offline_runtime

runtime = _stage_offline_runtime


@pytest.mark.parametrize('kind', ['fill', 'control'])
def test_actual_encode_facade_captures_used_geometry_and_only_control_seed(runtime, kind):
    from test_image_stage_numerical import selected_owner, values
    from modules.DiffusersImage.main import EncodeInputs
    owner, _strategy, _ = selected_owner(kind)
    given = {**values(count=2), 'width': 128, 'height': 128}
    node = EncodeInputs('actual-encoder-consumption')
    encoded = node(pipeline=owner, **given)['encoded_inputs']
    record = node._execution_input_record['fields']
    for field in ('width', 'height'):
        assert encoded.constraints[field] == given[field]
        assert encoded.consumed[field] == record[field]['value'] == given[field]
    assert encoded.consumed['effectiveBatch'] == 2
    assert record['num_images_per_prompt']['value'] == 2
    if kind == 'control':
        assert encoded.constraints['seed'] == encoded.consumed['seed'] == record['seed']['value'] == 5
    else:
        assert 'seed' not in encoded.constraints and 'seed' not in encoded.consumed
        assert record['seed']['value'] is None
