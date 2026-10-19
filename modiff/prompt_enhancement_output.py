"""Bounded target-layout proposals from one reviewed prompt-enhancement adapter.

This parser does not grant reference-frame authority or apply a graph edit.
"""
from copy import deepcopy
import json
import re
import unicodedata

from modiff.layout_conditioning import MAX_LAYOUT_DESCRIPTION, _upstream_box, parse_layout_conditioning

MAX_RESPONSE_BYTES = 256 * 1024
MAX_JSON_CANDIDATES = 32
MAX_JSON_DEPTH = 8
MAX_PROPOSAL_BYTES = 256 * 1024


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
        if 'source' in region or region.get('action', 'place') != 'place':
            raise _invalid('source regions need the actual issued frame/owner context and remain unsupported.')
        box = _upstream_box(region['targetBox'])
        if any(type(coordinate) is not int for coordinate in box):
            raise _invalid('authored boxes must map exactly to the integer 0..1000 grid; keep the authored layout or use Raw mode. Coordinates are never rounded.')
        expected[region['id']] = box
    return parsed, expected


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _invalid('duplicate JSON keys are not supported.')
        result[key] = value
    return result


def _nonfinite(_value):
    raise _invalid('JSON numbers must be finite.')


def _caption_table(text):
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
    _text(caption, maximum=131072, label='caption')
    return caption, rows


def _box(value):
    if (type(value) is not list or len(value) != 4
            or any(type(item) is not int or not 0 <= item <= 1000 for item in value)
            or value[0] >= value[2] or value[1] >= value[3]):
        raise _invalid('boxes require four integer y-first coordinates with positive area on 0..1000.')
    return value


def validate_target_layout_output(text, authored_layout, *, mode, image_count):
    """Return a preview proposal only; source mappings and implicit Apply are refused."""
    if (type(mode) is not str or mode not in ('t2i_bbox', 'i2i_bbox')
            or type(image_count) is not int or not 0 <= image_count <= 10
            or (mode == 't2i_bbox') != (image_count == 0)):
        raise _invalid('mode must match the actual supplied image context.')
    parsed, expected = require_authored_target_layout(authored_layout)
    caption, rows = _caption_table(text)
    if type(rows) is not list or len(rows) != len(expected):
        raise _invalid('the table must contain exactly the authored target IDs.')
    required = {'id', 'bbox', 'desc'} if mode == 't2i_bbox' else {'id', 'from', 'src_bbox', 'tgt_bbox', 'desc'}
    descriptions = {}
    for row in rows:
        if type(row) is not dict or set(row) != required:
            raise _invalid('unsupported table keys for the selected mode (first-pass estimate fields are not supported).')
        identity = row['id']
        if type(identity) is not str or identity not in expected or identity in descriptions:
            raise _invalid('unknown, duplicate or missing authored target ID.')
        if mode == 'i2i_bbox' and (row['from'] != 'target' or row['src_bbox'] is not None):
            raise _invalid('source mappings remain unsupported without issued frame/owner authority.')
        box = _box(row['bbox' if mode == 't2i_bbox' else 'tgt_bbox'])
        if box != expected[identity]:
            raise _invalid('generated target geometry differs from the authored box; the authored layout is unchanged.')
        descriptions[identity] = _text(row['desc'], maximum=MAX_LAYOUT_DESCRIPTION, label='description')
    tags = re.findall(r'<([^<>]*)>', caption)
    remainder = re.sub(r'<[^<>]*>', '', caption)
    if '<' in remainder or '>' in remainder:
        raise _invalid('caption contains incomplete or nested tags.')
    for identity in expected:
        if tags.count(identity) != 1:
            raise _invalid('caption must cite each authored target ID exactly once.')
    for tag in tags:
        if tag in expected:
            continue
        reference = re.fullmatch(r'ref_image_(0|[1-9][0-9]?)', tag)
        if reference is None or int(reference[1]) >= image_count:
            raise _invalid('caption tag does not match an authored target or actual supplied image.')
    proposal_layout = deepcopy(parsed)
    for region in proposal_layout['regions']:
        region['description'] = descriptions[region['id']]
    # Reuse the authored DTO's Unicode and canonical UTF-8 budget as well.
    proposal_layout = parse_layout_conditioning(proposal_layout)
    result = {'schemaVersion': 1, 'caption': caption, 'layoutConditioning': proposal_layout, 'rawTable': deepcopy(rows)}
    if len(json.dumps(result, ensure_ascii=True, allow_nan=False).encode('ascii')) > MAX_PROPOSAL_BYTES:
        raise _invalid('the structured proposal exceeds its 256 KiB JSON budget.')
    return result
