"""CPU-only preservation controls; synthetic tables are not native model proof."""
from copy import deepcopy
import json

import pytest

from modiff.prompt_enhancement_output import validate_target_layout_output
from modiff.prompt_enhancement_output import preservation_request, PRESERVATION_REQUEST_SUFFIX


LAYOUT = {'schemaVersion': 2, 'coordinateSpace': 'normalized_xyxy', 'regions': [{
    'id': 'authored_circle', 'description': 'Red circle.', 'action': 'relocate',
    'targetBox': [.2, .2, .8, .8], 'source': {'referenceIndex': 0,
        'box': [0, 0, 1, 1], 'expectedReferenceId': 'sha256:' + 'a' * 64}}]}
ROW = {'id': 'circle_1', 'from': 'ref_image_0', 'src_bbox': [0, 0, 1000, 1000],
       'tgt_bbox': [200, 200, 800, 800], 'desc': 'A single vivid red circle.'}


def validate(rows=None, caption='Move <circle_1> from <ref_image_0>.', layout=None):
    return validate_target_layout_output(caption + '\n' + json.dumps([ROW] if rows is None else rows),
        LAYOUT if layout is None else layout, mode='i2i_bbox', image_count=1)


def test_native_id_bijection_preserves_every_authored_field_except_description():
    original = deepcopy(LAYOUT)
    result = validate()
    expected = deepcopy(LAYOUT)
    expected['regions'][0]['description'] = ROW['desc']
    assert result['layoutConditioning'] == expected
    assert result['caption'] == 'Move <authored_circle> from <ref_image_0>.'
    assert result['rawCaption'] == 'Move <circle_1> from <ref_image_0>.'
    assert result['rawTable'] == [ROW] and LAYOUT == original


@pytest.mark.parametrize('patch', [
    {'src_bbox': [1, 0, 1000, 1000]}, {'tgt_bbox': [201, 200, 800, 800]},
    {'from': 'ref_image_1'}, {'tgt_bbox': None}, {'id': 'ref_image_0'},
    {'tgt_bbox_1stage': [200, 200, 800, 800]}, {'desc': ''},
])
def test_mutation_or_unsupported_native_operator_is_not_a_proposal(patch):
    with pytest.raises(ValueError):
        validate([{**ROW, **patch}])


def test_duplicate_signatures_do_not_guess_by_row_order_or_description():
    layout = deepcopy(LAYOUT)
    layout['regions'].append({**deepcopy(layout['regions'][0]), 'id': 'another_circle'})
    with pytest.raises(ValueError, match='ambiguous|bijection'):
        validate([ROW, {**ROW, 'id': 'circle_2'}],
                 caption='Move <circle_1> and <circle_2>.', layout=layout)


def test_exact_authored_ids_disambiguate_but_never_override_geometry():
    row = {**ROW, 'id': 'authored_circle'}
    assert validate([row], caption='Move <authored_circle>.')['layoutConditioning']['regions'][0]['source'] == LAYOUT['regions'][0]['source']
    with pytest.raises(ValueError, match='geometry|signature'):
        validate([{**row, 'tgt_bbox': [201, 200, 800, 800]}], caption='Move <authored_circle>.')


def test_unmapped_caption_id_and_non_grid_authority_reject_losslessly():
    with pytest.raises(ValueError):
        validate(caption='Move <circle_1> and <unknown>.')
    layout = deepcopy(LAYOUT)
    layout['regions'][0]['source']['box'][0] = .00001
    with pytest.raises(ValueError, match='never rounded'):
        validate(layout=layout)


def test_removal_preserves_null_target_action_and_exact_binding():
    layout = deepcopy(LAYOUT)
    layout['regions'][0].update(action='remove', targetBox=None)
    result = validate([{**ROW, 'tgt_bbox': None}], layout=layout)
    assert result['layoutConditioning']['regions'][0]['action'] == 'remove'
    assert result['layoutConditioning']['regions'][0]['targetBox'] is None
    assert result['layoutConditioning']['regions'][0]['source'] == LAYOUT['regions'][0]['source']


def test_preservation_request_keeps_raw_structured_conflict_and_finite_budget_guards():
    request = preservation_request('Move <authored_circle>.', LAYOUT, 1)
    assert request.endswith(PRESERVATION_REQUEST_SUFFIX)
    assert '"from":"ref_image_0"' in request
    assert LAYOUT['regions'][0]['source']['expectedReferenceId'] not in request
    with pytest.raises(ValueError, match='already contains regions'):
        preservation_request('Move <authored_circle>. ' + json.dumps([ROW]), LAYOUT, 1)
    with pytest.raises(ValueError, match='1 MiB'):
        preservation_request('x' * 1048550 + '<authored_circle>', LAYOUT, 1)
    with pytest.raises(ValueError, match='actual supplied image'):
        preservation_request('Move <authored_circle>.', LAYOUT, 0)
