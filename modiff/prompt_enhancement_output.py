"""Bounded target-layout proposals from one reviewed prompt-enhancement adapter.

This parser does not grant reference-frame authority or apply a graph edit.
"""
from copy import deepcopy
import json
import re
import unicodedata

from modiff.layout_conditioning import (
    MAX_LAYOUT_DESCRIPTION, MAX_STRUCTURED_PROMPT_BYTES, _contains_trailing_regions,
    _upstream_box, parse_layout_conditioning,
)

MAX_RESPONSE_BYTES = 256 * 1024
MAX_JSON_CANDIDATES = 32
MAX_JSON_DEPTH = 8
MAX_PROPOSAL_BYTES = 256 * 1024
PRESERVATION_REQUEST_SUFFIX = (
    'Authored request (text):\n{authored}\n\n'
    'Allowed region tags (JSON array): {tags}\n'
    'Return the enhanced output paragraph itself, then the COMPLETE authoritative '
    'table below, changing only desc. Keep every ID, source image, source box, '
    'target box, null value and removal/placement action. Do not return an id/desc '
    'summary. Scenery, lighting or atmosphere not already authored cannot acquire '
    'new rows or region tags; preserve and cite those explicitly supplied. These are authored '
    'constraints, not permission to infer reference bindings.\n'
    'Authoritative region table (JSON array; final input, not an example):\n{table}'
)
PRESERVATION_SYSTEM_PROMPT = (
    'Enhance the authored wording and region descriptions while preserving the supplied '
    'layout. The user supplies the complete, authoritative region table, not a request '
    'to plan or annotate a new layout.\n\n'
    'Keep every supplied row and region ID verbatim, in the same order and exact count. '
    'Preserve all keys and values except desc, including source-image labels, source '
    'boxes, target boxes and null values encoding placement or removal. Do not add '
    'background, scene, anchoring, preserved or inferred rows. Do not rename IDs, add '
    'first-pass estimate fields, infer reference bindings, move boxes or round coordinates. '
    'Existing source information is a constraint, not permission to create source authority.\n\n'
    'Return one English output paragraph followed by exactly one JSON array with the same table '
    'grammar. Cite every supplied region ID exactly once in its own angle brackets, '
    'using the ID verbatim. Reference-image tokens may cite only supplied reference '
    'images and are not additional table rows. Keep the requested subjects, counts, '
    'attributes, edits, style and quoted wording. Enhance visual specificity in the '
    'paragraph and desc text without introducing entities requiring additional rows. '
    'Mention scenery, lighting and atmosphere not already supplied as authored rows '
    'only as untagged prose; they do not acquire new rows or region tags. Preserve '
    'and cite every such region explicitly supplied in the authoritative table. '
    'Return no commentary, Markdown fences or additional output.\n\n'
)


def preservation_system_prompt(mode):
    """Replace automatic layout planning only for an authored preservation request."""
    if mode == 't2i_bbox':
        return (PRESERVATION_SYSTEM_PROMPT +
                'Write the GENERATIVE CAPTION itself, never instructions to render or quote another caption. '
                'Every row has exactly these keys in order: id, bbox, desc; bbox is unchanged. '
                'An id/desc-only summary is not this grammar. The following example illustrates '
                'format only; never copy its ID or geometry into the actual authored table.\n'
                'Example authoritative table:\n'
                '[{"id":"sample","bbox":[200,100,500,400],"desc":"A blue cube."}]\n'
                'Example output:\n'
                'A matte blue cube <sample> rests on a stone floor. '
                '[{"id":"sample","bbox":[200,100,500,400],"desc":"A matte blue cube."}]')
    if mode == 'i2i_bbox':
        return (PRESERVATION_SYSTEM_PROMPT + 'The source images are attached in order as '
                'ref_image_0, ref_image_1, ... . The edited target image does not exist yet. '
                'Copy the authored from values and source/target boxes exactly. '
                'Write the EDIT INSTRUCTION itself, never instructions to render or quote another caption. '
                'Every row has exactly these keys in order: id, from, src_bbox, tgt_bbox, desc; '
                'from, both boxes and their null values are unchanged. An id/desc-only summary '
                'is not this grammar. The following example illustrates format only; never '
                'copy its ID or geometry into the actual authored table.\n'
                'Example authoritative table:\n'
                '[{"id":"sample","from":"target","src_bbox":null,"tgt_bbox":[200,100,500,400],"desc":"A blue cube."}]\n'
                'Example output:\n'
                'Add a matte blue cube <sample> on the existing stone floor; keep everything else unchanged. '
                '[{"id":"sample","from":"target","src_bbox":null,"tgt_bbox":[200,100,500,400],"desc":"A matte blue cube."}]')
    raise ValueError('Authored-layout system instructions require a structured prompt mode.')


def _invalid(detail):
    return ValueError('Invalid generated target layout: ' + detail)


def _text(value, *, maximum, label):
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise _invalid(f'{label} must be bounded nonempty text.')
    try:
        value.encode('utf-8')
    except UnicodeError as error:
        raise _invalid(f'{label} must contain valid Unicode.') from error
    if any(unicodedata.category(char) == 'Cc' and char not in '\t\n\r' for char in value):
        raise _invalid(f'{label} contains unsupported control characters.')
    return value


def require_authored_target_layout(layout):
    """Preflight lossless integer-grid projection before spending model work."""
    parsed = parse_layout_conditioning(layout)
    if parsed is None or not parsed['regions']:
        raise _invalid('an authored target layout is required.')
    expected = {}
    for region in parsed['regions']:
        boxes = [region['targetBox']]
        if 'source' in region:
            boxes.append(region['source']['box'])
        for box in boxes:
            if box is not None and any(type(coordinate) is not int for coordinate in _upstream_box(box)):
                raise _invalid('authored boxes must map exactly to the integer 0..1000 grid; keep the authored layout or use Raw mode. Coordinates are never rounded.')
        expected[region['id']] = None if region['targetBox'] is None else _upstream_box(region['targetBox'])
    return parsed, expected


def _signature(region):
    source = region.get('source')
    return (None if source is None else source['referenceIndex'],
            None if source is None else tuple(_upstream_box(source['box'])),
            None if region['targetBox'] is None else tuple(_upstream_box(region['targetBox'])),
            region.get('action', 'relocate' if source else 'place'))


def preservation_request(authored, layout, image_count):
    """Project an immutable request, without claiming encoder/reference authority."""
    parsed, _expected = require_authored_target_layout(layout)
    _text(authored, maximum=MAX_STRUCTURED_PROMPT_BYTES, label='authored caption')
    if len(authored.encode('utf-8')) > MAX_STRUCTURED_PROMPT_BYTES:
        raise _invalid('structured prompt exceeds the 1 MiB UTF-8 limit.')
    if _contains_trailing_regions(authored):
        raise _invalid('caption already contains regions; use Raw mode or remove that region list.')
    rows = []
    for region in parsed['regions']:
        if f"<{region['id']}>" not in authored:
            raise _invalid('caption must explicitly mention every authored region ID.')
        source = region.get('source')
        if source is not None and source['referenceIndex'] >= image_count:
            raise _invalid('authored source index is outside the actual supplied image context.')
        target = None if region['targetBox'] is None else _upstream_box(region['targetBox'])
        row = {'id': region['id']}
        if image_count:
            row.update({'from': 'target' if source is None else f"ref_image_{source['referenceIndex']}",
                        'src_bbox': None if source is None else _upstream_box(source['box']), 'tgt_bbox': target})
        else:
            row['bbox'] = target
        row['desc'] = region['description']
        rows.append(row)
    # This legacy-named constant covers ALL static request-template bytes in
    # the existing v2 declaration hash; substitutions are authored data only.
    result = PRESERVATION_REQUEST_SUFFIX.format(
        authored=authored,
        tags=json.dumps([f"<{region['id']}>" for region in parsed['regions']], ensure_ascii=False),
        table=json.dumps(rows, ensure_ascii=False, allow_nan=False, separators=(',', ':')))
    if len(result.encode('utf-8')) > MAX_STRUCTURED_PROMPT_BYTES:
        raise _invalid('structured prompt exceeds the 1 MiB UTF-8 limit.')
    return result


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _invalid('duplicate JSON keys are not supported.')
        result[key] = value
    return result


def _nonfinite(_value):
    raise _invalid('JSON numbers must be finite.')


def _caption_table(text, *, require_caption=True):
    _text(text, maximum=MAX_RESPONSE_BYTES, label='output')
    if len(text.encode('utf-8')) > MAX_RESPONSE_BYTES:
        raise _invalid('output exceeds 256 KiB of UTF-8.')
    if '```' in text:
        raise _invalid('Markdown fences are not part of the caption/table grammar.')
    decoder = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_nonfinite)
    # Each candidate is disjoint. Every character is scanned once, and decoded
    # input across all attempts is at most the bounded original input size.
    start, stack, quoted, escaped, attempts = None, [], False, False, 0
    decoded = []
    end = len(text.rstrip())
    for index, char in enumerate(text):
        if start is None:
            if char == '[':
                attempts += 1
                if attempts > MAX_JSON_CANDIDATES:
                    raise _invalid('too many JSON candidates.')
                start, stack, quoted, escaped = index, ['['], False, False
            continue
        if quoted:
            if escaped:
                escaped = False
            elif char == '\\':
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char in '[{':
            stack.append(char)
            if len(stack) > MAX_JSON_DEPTH:
                raise _invalid('JSON nesting exceeds its depth limit.')
        elif char in ']}':
            if not stack or (stack[-1], char) not in (('[', ']'), ('{', '}')):
                raise _invalid('JSON delimiters do not match.')
            stack.pop()
            if not stack:
                try:
                    value = decoder.decode(text[start:index + 1])
                except (ValueError, RecursionError) as error:
                    if index + 1 == end:
                        raise _invalid('the trailing table is malformed: ' + str(error)) from error
                else:
                    decoded.append((start, index + 1, value))
                start = None
    if start is not None:
        raise _invalid('the JSON table is incomplete.')
    if len(decoded) != 1 or decoded[0][1] != end:
        raise _invalid('exactly one trailing JSON array is required.')
    start, _end, rows = decoded[0]
    caption = text[:start].rstrip()
    if require_caption:
        _text(caption, maximum=131072, label='caption')
    return caption, rows


def _box(value):
    if (type(value) is not list or len(value) != 4
            or any(type(item) is not int or not 0 <= item <= 1000 for item in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        raise _invalid('boxes require four integer y-first coordinates with positive area on 0..1000.')
    return value


def validate_target_layout_output(text, authored_layout, *, mode, image_count):
    """Change descriptions/caption only; never mint or alter reference authority."""
    if (type(mode) is not str or mode not in ('t2i_bbox', 'i2i_bbox')
            or type(image_count) is not int or not 0 <= image_count <= 10
            or (mode == 't2i_bbox') != (image_count == 0)):
        raise _invalid('mode must match the actual supplied image context.')
    parsed, expected = require_authored_target_layout(authored_layout)
    caption, rows = _caption_table(text)
    if type(rows) is not list or len(rows) != len(expected):
        raise _invalid('the table must contain exactly the authored target IDs.')
    required = {'id', 'bbox', 'desc'} if mode == 't2i_bbox' else {'id', 'from', 'src_bbox', 'tgt_bbox', 'desc'}
    descriptions, mapping, seen = {}, {}, set()
    for row in rows:
        if type(row) is not dict or set(row) != required:
            raise _invalid('unsupported table keys for the selected mode (first-pass estimate fields are not supported).')
        identity = row['id']
        if (type(identity) is not str or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', identity) is None
                or re.fullmatch(r'ref_image_[0-9]+', identity) or identity in seen):
            raise _invalid('unknown, duplicate or missing authored target ID.')
        seen.add(identity)
        if mode == 't2i_bbox':
            signature = (None, None, tuple(_box(row['bbox'])), 'place')
        else:
            origin = row['from']
            reference = re.fullmatch(r'ref_image_(0|[1-9][0-9]?)', origin) if type(origin) is str else None
            if origin == 'target' and row['src_bbox'] is None:
                source_index, source_box = None, None
            elif reference is not None and int(reference[1]) < image_count:
                source_index, source_box = int(reference[1]), tuple(_box(row['src_bbox']))
            else:
                raise _invalid('source mapping does not match an actual supplied image.')
            target = None if row['tgt_bbox'] is None else tuple(_box(row['tgt_bbox']))
            if source_index is None and target is None:
                raise _invalid('a target placement requires target geometry.')
            signature = (source_index, source_box, target,
                         'place' if source_index is None else 'remove' if target is None else 'relocate')
        candidates = [region for region in parsed['regions'] if _signature(region) == signature]
        # An authored ID is authoritative, including when equal geometry repeats.
        if identity in expected:
            candidates = [region for region in candidates if region['id'] == identity]
        if len(candidates) != 1 or candidates[0]['id'] in descriptions:
            raise _invalid('native rows need an unambiguous bijection with exact authored geometry/source/action; keep the authored layout unchanged.')
        authored_id = candidates[0]['id']
        mapping[identity] = authored_id
        descriptions[authored_id] = _text(row['desc'], maximum=MAX_LAYOUT_DESCRIPTION, label='description')
    tags = re.findall(r'<([^<>]*)>', caption)
    remainder = re.sub(r'<[^<>]*>', '', caption)
    if '<' in remainder or '>' in remainder:
        raise _invalid('caption contains incomplete or nested tags.')
    for identity in mapping:
        if tags.count(identity) != 1:
            raise _invalid('caption must cite each authored target ID exactly once.')
    for tag in tags:
        if tag in mapping:
            continue
        reference = re.fullmatch(r'ref_image_(0|[1-9][0-9]?)', tag)
        if reference is None or int(reference[1]) >= image_count:
            raise _invalid('caption tag does not match an authored target or actual supplied image.')
    proposal_layout = deepcopy(parsed)
    for region in proposal_layout['regions']:
        region['description'] = descriptions[region['id']]
    # Reuse the authored DTO's Unicode and canonical UTF-8 budget as well.
    proposal_layout = parse_layout_conditioning(proposal_layout)
    mapped_caption = re.sub(r'<([^<>]*)>', lambda match: '<' + mapping.get(match[1], match[1]) + '>', caption)
    result = {'schemaVersion': 1, 'caption': mapped_caption, 'layoutConditioning': proposal_layout, 'rawTable': deepcopy(rows)}
    if mapped_caption != caption or any('source' in region for region in parsed['regions']):
        result.update(schemaVersion=2, rawCaption=caption)
    if len(json.dumps(result, ensure_ascii=True, allow_nan=False).encode('ascii')) > MAX_PROPOSAL_BYTES:
        raise _invalid('the structured proposal exceeds its 256 KiB JSON budget.')
    return result


# New explicit composition policy. Existing strict generated-table parsing above
# remains unchanged, including its refusal of noncompliant native geometry.
APPEND_DETAILS_POLICY = 'append-details-preserve-layout-v1'
APPEND_DETAILS_OPERATION = 'enhance_prompt_append_details'
APPEND_REQUEST_TEMPLATE = (
    '{request}\n\nAuthored subject descriptions (wording context only):\n{descriptions}'
)


def _append_layout(layout, image_count):
    if type(image_count) is not int or not 0 <= image_count <= 10 or type(layout) is not dict:
        raise _invalid('append details requires an authored layout and actual ordered image count.')
    parsed = parse_layout_conditioning(layout)
    if parsed is None or not parsed['regions']:
        raise _invalid('append details requires a nonempty authored layout.')
    if any(region.get('source', {}).get('referenceIndex', -1) >= image_count
           for region in parsed['regions']):
        raise _invalid('authored source index is outside the actual supplied image context.')
    return parsed


def _append_tag_projection(value, parsed, image_count, *, require_targets):
    _text(value, maximum=131072, label='authored wording')
    if len(value.encode('utf-8')) > MAX_RESPONSE_BYTES:
        raise _invalid('authored wording exceeds its 256 KiB UTF-8 budget.')
    ids = {region['id'] for region in parsed['regions']}
    tags = re.findall(r'<([^<>]*)>', value)
    remainder = re.sub(r'<[^<>]*>', '', value)
    if '<' in remainder or '>' in remainder:
        raise _invalid('authored wording contains incomplete or nested tags.')
    if require_targets and any(tags.count(identity) != 1 for identity in ids):
        raise _invalid('authored caption must cite every region ID exactly once.')
    for tag in tags:
        reference = re.fullmatch(r'ref_image_(0|[1-9][0-9]?)', tag)
        if tag not in ids and (reference is None or int(reference[1]) >= image_count):
            raise _invalid('authored tag does not identify an authored region or actual reference.')
    return re.sub(r'<([^<>]*)>', lambda match: match[1], value)


def append_details_request(authored, layout, image_count):
    """Plain one-call wording context; no geometry or table becomes model input."""
    parsed = _append_layout(layout, image_count)
    request = _append_tag_projection(authored, parsed, image_count, require_targets=True)
    if _contains_trailing_regions(authored):
        raise _invalid('authored caption already contains a competing region table.')
    descriptions = '\n'.join(region['id'] + ': ' + _append_tag_projection(
        region['description'], parsed, image_count, require_targets=False)
        for region in parsed['regions'])
    rendered = APPEND_REQUEST_TEMPLATE.format(request=request, descriptions=descriptions)
    if len(rendered.encode('utf-8')) > MAX_STRUCTURED_PROMPT_BYTES:
        raise _invalid('append wording context exceeds the 1 MiB UTF-8 request budget.')
    return rendered


def validate_append_native_text(value):
    """Accept bounded native prose verbatim; never sanitize a model response."""
    _text(value, maximum=131072, label='native details')
    if len(value.encode('utf-8')) > MAX_RESPONSE_BYTES:
        raise _invalid('native details exceed their 256 KiB UTF-8 budget.')
    if '<' in value or '>' in value or '```' in value or '~~~' in value:
        raise _invalid('native details must not contain region tags or Markdown fences.')
    # Reuse the existing finite JSON scanner. Ordinary brackets, quoted wording,
    # styles and math are not rejected merely for resembling Markdown or JSON.
    if _contains_trailing_regions(value):
        raise _invalid('native details must not append a JSON table.')
    # Ordinary prose can contain an earlier non-JSON bracket or another array.
    # Check bounded suffix candidates independently instead of treating a failed
    # whole-response table parse as evidence that no trailing table exists.
    for attempt, match in enumerate(re.finditer(r'\[', value)):
        if attempt >= MAX_JSON_CANDIDATES:
            raise _invalid('native details contain too many JSON candidates.')
        try:
            _caption, rows = _caption_table(value[match.start():], require_caption=False)
        except ValueError:
            continue
        if type(rows) is list and (not rows or any(type(row) is dict for row in rows)):
            raise _invalid('native details must not append a JSON table.')
    return value


def compose_append_details(authored, authored_layout, native_text, *, image_count):
    """Compose text only; all authored layout bytes/values stay authoritative."""
    append_details_request(authored, authored_layout, image_count)
    validate_append_native_text(native_text)
    caption = authored + '\n\n' + native_text
    _text(caption, maximum=131072, label='composed caption')
    if len(caption.encode('utf-8')) > MAX_RESPONSE_BYTES:
        raise _invalid('composed caption exceeds its 256 KiB UTF-8 budget.')
    result = {'schemaVersion': 3, 'kind': 'caption-append-preserve-layout-v1',
              'caption': caption, 'layoutConditioning': deepcopy(authored_layout),
              'nativeText': native_text}
    if len(json.dumps(result, ensure_ascii=True, allow_nan=False).encode('ascii')) > MAX_PROPOSAL_BYTES:
        raise _invalid('the composed proposal exceeds its 256 KiB JSON budget.')
    return result
