"""CPU-only preservation controls; synthetic tables are not native model proof."""
from copy import deepcopy
import json

import pytest

from modiff.prompt_enhancement_output import validate_target_layout_output
from modiff.prompt_enhancement_output import preservation_request, preservation_system_prompt


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
    assert json.loads(request.rsplit('\n', 1)[1])[0]['id'] == 'authored_circle'
    assert '"from":"ref_image_0"' in request
    assert LAYOUT['regions'][0]['source']['expectedReferenceId'] not in request
    with pytest.raises(ValueError, match='already contains regions'):
        preservation_request('Move <authored_circle>. ' + json.dumps([ROW]), LAYOUT, 1)
    with pytest.raises(ValueError, match='1 MiB'):
        preservation_request('x' * 1048550 + '<authored_circle>', LAYOUT, 1)
    with pytest.raises(ValueError, match='actual supplied image'):
        preservation_request('Move <authored_circle>.', LAYOUT, 0)


@pytest.mark.parametrize('image_count', [0, 1])
def test_request_keeps_complete_authored_table_last_without_scenery_authority(image_count):
    layout = deepcopy(LAYOUT)
    if not image_count:
        layout['regions'][0].pop('source')
        layout['regions'][0]['action'] = 'place'
    before = deepcopy(layout)
    authored = '  Show <authored_circle> with unchanged scenery and warm lighting. {Keep this wording.}\n'
    request = preservation_request(authored, layout, image_count)
    assert request.startswith('Authored request (text):\n' + authored + '\n\n')
    tags = next(line for line in request.splitlines() if line.startswith('Allowed region tags (JSON array): '))
    assert json.loads(tags.split(': ', 1)[1]) == ['<authored_circle>']
    # A terminal JSON decode detects any instructions appended after the table.
    table = json.loads(request.rsplit('\n', 1)[1])
    assert len(table) == 1 and table[0]['id'] == 'authored_circle'
    assert table[0]['desc'] == layout['regions'][0]['description']
    if image_count:
        assert list(table[0]) == ['id', 'from', 'src_bbox', 'tgt_bbox', 'desc']
        assert table[0]['from'] == 'ref_image_0'
        assert table[0]['src_bbox'] == [0, 0, 1000, 1000]
        assert table[0]['tgt_bbox'] == [200, 200, 800, 800]
    else:
        assert list(table[0]) == ['id', 'bbox', 'desc']
        assert table[0]['bbox'] == [200, 200, 800, 800]
    assert LAYOUT['regions'][0]['source']['expectedReferenceId'] not in request
    assert layout == before


@pytest.mark.parametrize('mode,image_count', [('t2i_bbox', 0), ('i2i_bbox', 1)])
def test_mode_specific_worked_example_obeys_the_same_strict_geometry_parser(mode, image_count):
    system = preservation_system_prompt(mode)
    assert ('EDIT INSTRUCTION itself' in system) is bool(image_count)
    assert ('GENERATIVE CAPTION itself' in system) is (not image_count)
    assert 'untagged prose' in system and 'id/desc-only summary' in system
    assert system.count('Example output:\n') == 1
    supplied, output = system.split('Example authoritative table:\n', 1)[1].split('\nExample output:\n')
    rows = json.loads(supplied)
    layout = {'schemaVersion': 1, 'coordinateSpace': 'normalized_xyxy', 'regions': [{
        'id': 'sample', 'description': rows[0]['desc'], 'targetBox': [.1, .2, .4, .5]}]}
    proposal = validate_target_layout_output(output, layout, mode=mode, image_count=image_count)
    assert proposal['layoutConditioning']['regions'][0]['targetBox'] == layout['regions'][0]['targetBox']
    assert proposal['rawTable'][0]['id'] == 'sample'
    assert len(proposal['rawTable']) == 1  # Mentioned floor never gains a row.


def test_explicitly_authored_lighting_retains_its_row_and_allowed_tag():
    layout = deepcopy(LAYOUT)
    lighting = {'id': 'authored_lighting', 'description': 'Warm lighting.', 'action': 'place',
                'targetBox': [0, 0, 1, 1]}
    layout['regions'].append(lighting)
    before = deepcopy(layout)
    request = preservation_request('Move <authored_circle> under <authored_lighting>.', layout, 1)
    tags = next(line for line in request.splitlines() if line.startswith('Allowed region tags (JSON array): '))
    assert json.loads(tags.split(': ', 1)[1]) == ['<authored_circle>', '<authored_lighting>']
    rows = json.loads(request.rsplit('\n', 1)[1])
    assert rows[1] == {'id': 'authored_lighting', 'from': 'target', 'src_bbox': None,
                       'tgt_bbox': [0, 0, 1000, 1000], 'desc': lighting['description']}
    assert len(rows) == 2 and layout == before


@pytest.mark.parametrize('extra_rows', [False, True])
def test_id_description_summaries_with_extra_context_never_become_layout_authority(extra_rows):
    rows = [{'id': 'authored_circle', 'desc': 'A red circle.'}]
    if extra_rows:
        rows.extend({'id': name, 'desc': 'Unchanged context.'} for name in ('floor', 'room', 'lighting'))
    generated = 'Render the complete caption as: "Move <authored_circle>." ' + json.dumps(rows)
    with pytest.raises(ValueError, match='authored target IDs|unsupported table keys'):
        validate_target_layout_output(generated, LAYOUT, mode='i2i_bbox', image_count=1)


@pytest.mark.parametrize('image_count', [0, 1])
def test_structured_system_does_not_compete_with_the_authored_region_table(image_count):
    from modiff import prompt_enhancement as adapter

    class Processor:
        def apply_chat_template(self, messages, **_kwargs):
            self.messages = deepcopy(messages)
            return json.dumps(messages)

    _raw, prompts = adapter._json_file(adapter.SYSTEM_PROMPTS, 32 * 1024)
    layout = {'schemaVersion': 2, 'coordinateSpace': 'normalized_xyxy', 'regions': [{
        'id': 'fox', 'description': 'A fox.', 'action': 'place', 'targetBox': [.2, .2, .8, .8]}]}
    before = deepcopy(layout)
    processor = Processor()
    mode, _rendered, parsed = adapter.prepare_request(
        processor, prompts, 'A single <fox> with a blue scarf.', [object()] * image_count, layout)
    system = processor.messages[0]['content']
    assert 'class noun plus an index' not in system
    assert 'one row per' not in system
    assert 'Keep every supplied row and region ID verbatim' in system
    assert prompts[mode] not in system
    if image_count:
        assert 'attached in order as ref_image_0, ref_image_1, ...' in system
        assert 'target image does not exist yet' in system
        assert 'Copy the authored from values and source/target boxes exactly' in system
    else:
        assert 'attached in order' not in system
    assert processor.messages[1]['content'][-1]['text'] == preservation_request(
        'A single <fox> with a blue scarf.', layout, image_count)
    assert parsed == layout == before


@pytest.mark.parametrize('image_count,structured', [(0, False), (1, False), (0, True), (1, True)])
def test_rendered_system_and_shared_preallocation_generation_fingerprint_agree(image_count, structured):
    import hashlib
    from types import SimpleNamespace

    import torch
    from PIL import Image
    from modiff import prompt_enhancement as adapter

    class Processor:
        def apply_chat_template(self, messages, **_kwargs):
            self.messages = deepcopy(messages)
            return json.dumps(messages)

    _raw, prompts = adapter._json_file(adapter.SYSTEM_PROMPTS, 32 * 1024)
    layout = {'schemaVersion': 2, 'coordinateSpace': 'normalized_xyxy', 'regions': [{
        'id': 'fox', 'description': 'A fox.', 'action': 'place', 'targetBox': [.2, .2, .8, .8]}]} if structured else None
    images = [Image.new('RGB', (2, 2), 'red')] * image_count
    processor = Processor()
    mode, rendered, _parsed = adapter.prepare_request(processor, prompts, 'A single <fox>.', images, layout)
    batch = {'input_ids': torch.tensor([[1, 2]], dtype=torch.int64)}
    if image_count:
        batch.update(image_grid_thw=torch.tensor([[1, 1, 1]], dtype=torch.int64),
                     pixel_values=torch.zeros((1, 3), dtype=torch.float32))
    # Both normal preallocation and generation use this exact shared projection.
    fields, consumed = adapter._consumed_projection(
        {'id': 'test-adapter'}, prompts, mode, rendered, images, batch, 32, SimpleNamespace(check=lambda: None))
    actual = processor.messages[0]['content']
    assert fields['system_prompt_sha256'] == hashlib.sha256(actual.encode('utf-8')).hexdigest()
    assert consumed['fingerprints'] == fields
    assert (actual == prompts[mode]) is (not structured)
