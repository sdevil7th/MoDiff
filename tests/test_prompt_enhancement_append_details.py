"""One-call local composition policy; synthetic text is not trained-model proof."""
from copy import deepcopy
import json

import pytest

from modiff.prompt_enhancement_output import (
    APPEND_DETAILS_OPERATION, append_details_request, compose_append_details,
    validate_append_native_text,
)


LAYOUT = {'schemaVersion': 2, 'coordinateSpace': 'normalized_xyxy', 'regions': [
    {'id': 'fox', 'description': 'The authored red fox.', 'action': 'relocate',
     'targetBox': [.123456, .2, .8, .9], 'source': {'referenceIndex': 0,
     'box': [.00123, .2, .6, .7], 'expectedReferenceId': 'sha256:' + 'a' * 64}},
    {'id': 'shadow', 'description': 'The authored shadow.', 'action': 'remove',
     'targetBox': None, 'source': {'referenceIndex': 0, 'box': [.1, .1, .2, .2],
     'expectedReferenceId': 'sha256:' + 'a' * 64}}]}
AUTHORED = '  D\u00e9place <fox> depuis <ref_image_0>; remove <shadow>.  \n'


def test_composed_caption_keeps_exact_utf8_prefix_and_entire_raw_layout():
    original = deepcopy(LAYOUT)
    native = '  Fine red fur catches the soft light.\n'
    value = compose_append_details(AUTHORED, LAYOUT, native, image_count=1)
    assert set(value) == {'schemaVersion', 'kind', 'caption', 'layoutConditioning', 'nativeText'}
    assert value['schemaVersion'] == 3 and value['kind'] == 'caption-append-preserve-layout-v1'
    assert value['caption'].encode() == AUTHORED.encode() + b'\n\n' + native.encode()
    assert value['nativeText'] == native
    assert value['layoutConditioning'] == original == LAYOUT
    assert value['layoutConditioning'] is not LAYOUT
    value['layoutConditioning']['regions'][0]['description'] = 'changed elsewhere'
    assert LAYOUT == original


def test_model_request_has_only_tag_projected_wording_and_authored_descriptions():
    before = deepcopy(LAYOUT)
    request = append_details_request(AUTHORED, LAYOUT, 1)
    assert request.startswith(AUTHORED.replace('<fox>', 'fox').replace('<shadow>', 'shadow')
                              .replace('<ref_image_0>', 'ref_image_0'))
    assert 'fox: The authored red fox.' in request
    assert 'shadow: The authored shadow.' in request
    assert not any(value in request for value in ('<', '>', 'targetBox', 'referenceIndex',
                                                'expectedReferenceId', '.123456', 'relocate'))
    assert LAYOUT == before


@pytest.mark.parametrize('authored', [
    'Move <fox> <fox> and <shadow>.', 'Move <shadow>.', 'Move <fox> and <shadow> <unknown>.',
    'Move <<fox>> and <shadow>.', 'Move <fox and <shadow>.',
    'Move <fox> and <shadow> from <ref_image_1>.', 'Move <fox> and <shadow> from <ref_image_01>.',
])
def test_authored_tags_are_independently_validated(authored):
    with pytest.raises(ValueError):
        append_details_request(authored, LAYOUT, 1)


@pytest.mark.parametrize('native', [
    '', '   ', '\x00', '\ud800', 'An extra <fox> tag.', 'An unmatched > sign.',
    '```text\nA fox.\n```', '~~~text\nA fox.\n~~~',
    'A fox. [{"id":"fox","bbox":[0,0,1000,1000]}]', '[{"id":"fox","desc":"changed"}]',
    'A fox. []',
    'A [red] accent. [{"id":"fox","bbox":[0,0,1000,1000]}]',
    'A [red] accent. [{"desc":"extra table"}]',
    'A [red] accent. []', '[0, 1] then [{"desc":"extra table"}]',
    'An unmatched [red accent. [{"desc":"extra table"}]',
])
def test_native_suffix_rejects_tags_tables_fences_and_invalid_text(native):
    with pytest.raises(ValueError):
        validate_append_native_text(native)


def test_native_prose_punctuation_math_quoted_styles_and_whitespace_are_not_sanitized():
    value = '  A **soft** coat, a [red] accent, and the words "{hello}"; ratio [0, 1].\n'
    assert validate_append_native_text(value) == value


@pytest.mark.parametrize('native', ['x' * 131073, '\U0001f98a' * 65537, '\u00e9' * 50000])
def test_native_caption_and_ascii_proposal_budgets_are_independent(native):
    with pytest.raises(ValueError):
        compose_append_details(AUTHORED, LAYOUT, native, image_count=1)


def test_plain_t2i_publisher_is_selected_with_no_layout_authority_in_model_input():
    from modiff.prompt_enhancement import prepare_request
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            self.messages, self.kwargs = messages, kwargs
            return json.dumps(messages, ensure_ascii=False)
    processor = Processor()
    layout = {'schemaVersion': 1, 'coordinateSpace': 'normalized_xyxy', 'regions': [
        {'id': 'region_1', 'description': 'An authored blue cube.', 'targetBox': [.01, .02, .3, .4]}]}
    mode, rendered, _ = prepare_request(processor, {'t2i': 'PLAIN', 't2i_bbox': 'STRICT'},
        'A <region_1>.', [], layout, operation=APPEND_DETAILS_OPERATION)
    assert mode == 't2i' and processor.messages[0]['content'] == 'PLAIN'
    user = processor.messages[1]['content'][0]['text']
    assert 'region_1: An authored blue cube.' in user and 'targetBox' not in user
    assert '<region_1>' not in user and 'STRICT' not in rendered
    assert processor.kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
