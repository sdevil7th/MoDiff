"""Bounded target parser: actual node/processor, substituted generated text.

Only the generated suffix is deterministic test data. No trained output or
source-region authorization is claimed by these controls.
"""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest
import torch

import test_prompt_enhancement as stage_one
from modules.HuggingFaceTransformers import main as nodes

actual_pup = stage_one.actual_pup
LAYOUT = {'schemaVersion': 1, 'coordinateSpace': 'normalized_xyxy', 'regions': [
    {'id': 'cube_1', 'description': 'A red cube.', 'targetBox': [.1, .2, .4, .5]}]}
ROW = {'id': 'cube_1', 'bbox': [200, 100, 500, 400], 'desc': 'A red wooden cube.'}


def substitute_generated_suffix(monkeypatch, handle, suffix):
    def generated(**kwargs):
        encoded = handle['preprocessor'].tokenizer(suffix, return_tensors='pt', add_special_tokens=False)['input_ids']
        return torch.cat([kwargs['input_ids'], encoded], dim=1)
    monkeypatch.setattr(handle['model'], 'generate', generated)


def test_exact_authored_target_layout_needs_a_validated_preview_proposal(actual_pup, monkeypatch):
    handle, _ = actual_pup
    original = deepcopy(LAYOUT)
    suffix = 'A red wooden cube <cube_1>. ' + json.dumps([ROW], separators=(',', ':'))
    substitute_generated_suffix(monkeypatch, handle, suffix)
    node = nodes.GenerateImageVideoText('pup-structured-positive')
    node(model=handle, prompt='A cube <cube_1>.', operation='enhance_prompt',
         enhancement_max_new_tokens=256, layout_conditioning=LAYOUT)
    result = node.output['result']
    # A validated preview stays separate from authored state and cannot enter
    # the ordinary text output without a paired explicit Apply.
    assert result['layoutOutputValidated'] is True
    assert node.output['text'] is None  # No automatic caption/JSON feed into Encode Prompt.
    assert result['text'] == suffix  # Raw generated data remains diagnostic.
    proposal = result['structuredProposal']
    assert proposal['caption'] == 'A red wooden cube <cube_1>.'
    assert proposal['layoutConditioning']['regions'][0]['targetBox'] == LAYOUT['regions'][0]['targetBox']
    assert proposal['layoutConditioning']['regions'][0]['description'] == ROW['desc']
    assert result['applicationAllowed'] is True and result['applyRequired'] is True
    assert result['semanticFidelityVerified'] is False
    assert result['authoredPrompt'] == 'A cube <cube_1>.' and LAYOUT == original


def test_malformed_or_authored_geometry_changed_output_must_stay_non_applicable(actual_pup, monkeypatch):
    handle, _ = actual_pup
    rows = [
        [{**ROW, 'bbox': [201, 100, 500, 400]}],
        [{**ROW, 'id': 'other_1'}], [ROW, ROW],
        [{**ROW, 'bbox': [True, 100, 500, 400]}],
        [{**ROW, 'bbox': [200, 100, 200, 400]}],
        [{**ROW, 'unreviewed': 'property'}],
    ]
    for index, table in enumerate(rows):
        suffix = 'A cube <cube_1>. ' + json.dumps(table)
        substitute_generated_suffix(monkeypatch, handle, suffix)
        node = nodes.GenerateImageVideoText('pup-structured-negative-' + str(index))
        node(model=handle, prompt='A cube <cube_1>.', operation='enhance_prompt',
             enhancement_max_new_tokens=256, layout_conditioning=LAYOUT)
        result = node.output['result']
        assert result['applicationAllowed'] is False and node.output['text'] is None


def test_source_hash_or_reference_count_cannot_replace_encode_image_authority(actual_pup, monkeypatch):
    from PIL import Image
    handle, _ = actual_pup
    layout = {'schemaVersion': 2, 'coordinateSpace': 'normalized_xyxy', 'regions': [{
        'id': 'cube_1', 'description': 'A red cube.', 'action': 'relocate', 'targetBox': [.1, .2, .4, .5],
        'source': {'referenceIndex': 0, 'expectedReferenceId': 'sha256:' + '0' * 64, 'box': [.1, .2, .4, .5]}}]}
    generate = Mock(side_effect=AssertionError('Unbound source reached generation'))
    monkeypatch.setattr(handle['model'], 'generate', generate)
    node = nodes.GenerateImageVideoText('pup-source-negative')
    with pytest.raises(RuntimeError, match='reference-identity'):
        node(model=handle, prompt='Move <cube_1>.', operation='enhance_prompt',
             enhancement_max_new_tokens=256, layout_conditioning=layout,
             images=[Image.new('RGB', (256, 256), 'red')], reference_count=1,
             reference_source_sha256=['sha256:' + '0' * 64],
             reference_binding={'schemaVersion': 1, 'references': [{'referenceId': 'sha256:' + '0' * 64}]})
    assert not generate.called


from modiff.prompt_enhancement_output import validate_target_layout_output, require_authored_target_layout


def validate(text, *, mode='t2i_bbox', image_count=0, layout=None):
    return validate_target_layout_output(text, LAYOUT if layout is None else layout, mode=mode, image_count=image_count)


def output(row=None, caption='A cube <cube_1>.'):
    return caption + ' ' + json.dumps([ROW if row is None else row], ensure_ascii=True)


def test_pure_target_and_actual_image_mode_keep_authored_geometry_exact():
    original = deepcopy(LAYOUT)
    value = validate(output())
    assert value['layoutConditioning']['regions'][0]['targetBox'] == [.1, .2, .4, .5]
    assert value['layoutConditioning']['regions'][0]['description'] == ROW['desc']
    assert LAYOUT == original
    row = {'id': 'cube_1', 'from': 'target', 'src_bbox': None, 'tgt_bbox': ROW['bbox'], 'desc': ROW['desc']}
    result = validate(output(row, 'Keep <ref_image_1> and add <cube_1>.'), mode='i2i_bbox', image_count=2)
    assert result['rawTable'] == [row]
    with pytest.raises(ValueError, match='actual supplied image'):
        validate(output(row, 'Use <ref_image_2> and add <cube_1>.'), mode='i2i_bbox', image_count=2)


@pytest.mark.parametrize('mode,count', [('other', 0), ('t2i_bbox', 1), ('i2i_bbox', 0),
                                       ('i2i_bbox', 11), ('i2i_bbox', True), (None, 0), ('t2i_bbox', 0.0)])
def test_mode_and_count_require_actual_adapter_context(mode, count):
    with pytest.raises(ValueError, match='mode'):
        validate(output(), mode=mode, image_count=count)


@pytest.mark.parametrize('box', [[True, 100, 500, 400], [200.0, 100, 500, 400],
    [200, -1, 500, 400], [200, 100, 1001, 400], [200, 100, 200, 400], [200, 100, 500, 100],
    [200, 100, 500], None, [201, 100, 500, 400]])
def test_generated_box_domain_and_exact_authored_equality(box):
    with pytest.raises(ValueError, match='box|coordinates|geometry'):
        validate(output({**ROW, 'bbox': box}))


@pytest.mark.parametrize('caption', ['A cube.', 'A cube <cube_1> and another <cube_1>.',
    'A cube <cube_1> <other_1>.', 'A cube <<cube_1>>.', 'A cube <cube_1>. <ref_image_0>'])
def test_exact_target_tags_and_no_unbound_reference_mentions(caption):
    with pytest.raises(ValueError, match='tag|cite'):
        validate(output(caption=caption))


@pytest.mark.parametrize('rows', [[{**ROW, 'id': 'other_1'}], [ROW, ROW], [],
    [{**ROW, 'extra': 1}], [{key: value for key, value in ROW.items() if key != 'desc'}],
    [{**ROW, 'desc': ''}], [{**ROW, 'desc': 'x' * 4097}]])
def test_closed_rows_ids_and_description_limits(rows):
    with pytest.raises(ValueError):
        validate('A cube <cube_1>. ' + json.dumps(rows))


@pytest.mark.parametrize('text', [
    'A cube <cube_1>. [{"id":"cube_1","id":"cube_1","bbox":[200,100,500,400],"desc":"x"}]',
    'A cube <cube_1>. [{"id":"cube_1","bbox":[NaN,100,500,400],"desc":"x"}]',
    'A cube <cube_1>. [{"id":"cube_1","bbox":[200,100,500,400],"desc":"\ud800"}]',
    'A cube <cube_1>. [{"id":"cube_1","bbox":[200,100,500,400],"desc":"x"}',
    'A cube <cube_1>. [1] ' + output(), '```' + output() + '```', output() + ' trailing',
    'A cube <cube_1>. ' + '[' * 9 + '1' + ']' * 9,
])
def test_json_unicode_duplicate_finite_depth_and_trailing_grammar(text):
    with pytest.raises(ValueError):
        validate(text)


def test_non_unicode_caption_and_control_character_refused():
    for caption in ['A ' + chr(0xD800) + ' <cube_1>.', 'A cube\x00 <cube_1>.']:
        with pytest.raises(ValueError, match='Unicode|control'):
            validate(output(caption=caption))


def test_scanner_decodes_disjoint_candidates_with_finite_attempt_bound(monkeypatch):
    import modiff.prompt_enhancement_output as parser
    decoder = parser.json.JSONDecoder
    spans = []
    class CountedDecoder(decoder):
        def decode(self, value):
            spans.append(len(value))
            return super().decode(value)
    monkeypatch.setattr(parser.json, 'JSONDecoder', CountedDecoder)
    text = '[label] ' * 31 + output()
    validate(text)
    assert len(spans) == 32 and sum(spans) <= len(text)
    spans.clear()
    with pytest.raises(ValueError, match='candidates'):
        validate('[label] ' * 32 + output())
    assert len(spans) == 32
    with pytest.raises(ValueError, match='bounded|256 KiB'):
        validate('x' * (256 * 1024 + 1))


def test_quote_escape_and_brackets_in_caption_and_description_are_not_greedy():
    row = {**ROW, 'desc': 'A red cube with a "quoted" sign [not a table] and slash \\.'}
    value = validate(output(row, 'A sign reads "[OPEN]" on <cube_1>.'))
    assert value['rawTable'][0]['desc'] == row['desc']


def test_exact_decimal_grid_never_rounds_and_sources_remain_refused():
    for coordinate in [.10001, .30000000000000004]:
        layout = deepcopy(LAYOUT)
        layout['regions'][0]['targetBox'][0] = coordinate
        with pytest.raises(ValueError, match='never rounded'):
            require_authored_target_layout(layout)
    source = deepcopy(LAYOUT)
    source['regions'][0]['source'] = {'referenceIndex': 0, 'box': [.1, .2, .4, .5]}
    with pytest.raises(ValueError, match='source regions'):
        require_authored_target_layout(source)
    row = {'id': 'cube_1', 'from': 'ref_image_0', 'src_bbox': ROW['bbox'], 'tgt_bbox': ROW['bbox'], 'desc': 'A cube.'}
    with pytest.raises(ValueError, match='source mappings'):
        validate(output(row), mode='i2i_bbox', image_count=1)
    row['from'], row['src_bbox'], row['tgt_bbox_1stage'] = 'target', None, ROW['bbox']
    with pytest.raises(ValueError, match='first-pass'):
        validate(output(row), mode='i2i_bbox', image_count=1)


def test_nonrepresentable_authored_box_fails_before_template_and_neural_work(actual_pup, monkeypatch):
    handle, _ = actual_pup
    layout = deepcopy(LAYOUT)
    layout['regions'][0]['targetBox'][0] = .10001
    template = Mock(side_effect=AssertionError('Nonrepresentable layout reached template'))
    generation = Mock(side_effect=AssertionError('Nonrepresentable layout reached model'))
    monkeypatch.setattr(handle['preprocessor'], 'apply_chat_template', template)
    monkeypatch.setattr(handle['model'], 'generate', generation)
    node = nodes.GenerateImageVideoText('pup-grid-preflight')
    with pytest.raises(RuntimeError, match='never rounded'):
        node(model=handle, prompt='A cube <cube_1>.', operation='enhance_prompt',
             enhancement_max_new_tokens=256, layout_conditioning=layout)
    assert not template.called and not generation.called


def test_actual_i2i_target_only_proposal_never_reaches_generic_text_output(actual_pup, monkeypatch):
    from PIL import Image
    handle, _ = actual_pup
    row = {'id': 'cube_1', 'from': 'target', 'src_bbox': None, 'tgt_bbox': ROW['bbox'], 'desc': ROW['desc']}
    suffix = output(row, 'Keep <ref_image_1> and add <cube_1>.')
    substitute_generated_suffix(monkeypatch, handle, suffix)
    original = deepcopy(LAYOUT)
    node = nodes.GenerateImageVideoText('pup-i2i-target-only')
    node(model=handle, prompt='Add a cube <cube_1>.', operation='enhance_prompt',
         enhancement_max_new_tokens=256, layout_conditioning=LAYOUT,
         images=[Image.new('RGB', (256, 256), color) for color in ['red', 'blue']])
    result = node.output['result']
    assert result['systemPromptMode'] == 'i2i_bbox' and result['layoutOutputValidated'] is True
    assert result['applicationAllowed'] is True and result['applyRequired'] is True
    assert result['semanticFidelityVerified'] is False and node.output['text'] is None
    assert result['layoutConditioning'] == LAYOUT == original
    assert result['structuredProposal']['rawTable'] == [row]


def test_escaped_json_surrogate_and_multibyte_utf8_budget_are_checked_after_decode():
    with pytest.raises(ValueError, match='Unicode'):
        validate(output({**ROW, 'desc': chr(0xD800)}))
    with pytest.raises(ValueError, match='256 KiB'):
        validate(output(caption=chr(0x754C) * 90_000 + ' <cube_1>.'))


def test_proposal_json_budget_is_distinct_from_raw_utf8_budget():
    with pytest.raises(ValueError, match='proposal exceeds'):
        validate(output(caption=chr(0x754C) * 50_000 + ' <cube_1>.'))


def test_target_only_parser_does_not_enable_source_control_metadata():
    declaration = nodes.GenerateImageVideoText.params['layout_conditioning']
    assert declaration['hidden'] is True
    assert declaration['fieldOptions']['layoutConditioning']['sourceRegions'] is False
