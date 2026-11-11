"""Bounded authored regions and the reviewed caption/regions prompt projection.

This module performs no model loading. A selected backend adapter must explicitly
declare this wrapper input before calling the projection; a JSON value does not
grant model, reference-media, or execution authority.
"""
from __future__ import annotations

import json
import math
import re
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from typing import Literal, NotRequired, TypedDict

from modiff.block_definition_v2 import _stable_json


MAX_LAYOUT_REGIONS = 32
MAX_LAYOUT_REFERENCES = 8
MAX_BOUND_LAYOUT_REFERENCES = 10
MAX_LAYOUT_DESCRIPTION = 4096
MAX_LAYOUT_BYTES = 64 * 1024
MAX_STRUCTURED_PROMPT_BYTES = 1024 * 1024

LayoutBox = list[int | float]


class LayoutSource(TypedDict):
    referenceIndex: int
    box: LayoutBox


class LayoutRegion(TypedDict):
    id: str
    description: str
    targetBox: LayoutBox
    source: NotRequired[LayoutSource]


class BoundLayoutSource(LayoutSource):
    expectedReferenceId: str


class LayoutActionRegion(TypedDict):
    id: str
    description: str
    action: Literal["place", "relocate", "remove"]
    targetBox: LayoutBox | None
    source: NotRequired[BoundLayoutSource]


class LayoutConditioning(TypedDict):
    schemaVersion: Literal[1, 2]
    coordinateSpace: Literal["normalized_xyxy"]
    regions: list[LayoutRegion | LayoutActionRegion]


def _invalid(detail: str) -> ValueError:
    return ValueError(f"Invalid layout conditioning: {detail}")


def _record(value: object, required: set[str], optional: set[str] | None = None) -> dict:
    if (type(value) is not dict or len(value) > len(required) + len(optional or ())
            or any(type(key) is not str for key in value)):
        raise _invalid("expected a plain JSON object.")
    if not required.issubset(value) or not set(value).issubset(required | (optional or set())):
        raise _invalid("missing or unknown properties.")
    return value


def _utf8_size(value: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeError as error:
        raise _invalid("text must contain valid Unicode.") from error


def _box(value: object) -> LayoutBox:
    if type(value) is not list or len(value) != 4:
        raise _invalid("box coordinates must be four finite numbers between 0 and 1.")
    if any(type(item) not in {int, float} or not 0 <= item <= 1 or not math.isfinite(item) for item in value):
        raise _invalid("box coordinates must be four finite numbers between 0 and 1.")
    if value[0] >= value[2] or value[1] >= value[3]:
        raise _invalid("box corners must enclose a positive area.")
    return list(value)


def parse_layout_conditioning(value: object) -> LayoutConditioning | None:
    """Validate and copy authored JSON; reference binding is checked separately."""
    if value is None:
        return None
    item = _record(value, {"schemaVersion", "coordinateSpace", "regions"})
    if (type(item["schemaVersion"]) is not int or item["schemaVersion"] not in (1, 2)
            or type(item["coordinateSpace"]) is not str or item["coordinateSpace"] != "normalized_xyxy"):
        raise _invalid("unsupported schema or coordinate space.")
    if type(item["regions"]) is not list or len(item["regions"]) > MAX_LAYOUT_REGIONS:
        raise _invalid("at most 32 regions are supported.")
    version = item["schemaVersion"]
    regions = []
    identities: set[str] = set()
    for raw in item["regions"]:
        region = _record(raw, {"id", "description", "targetBox"} | ({"action"} if version == 2 else set()), {"source"})
        action = region.get("action")
        if version == 2 and (type(action) is not str or action not in ("place", "relocate", "remove")):
            raise _invalid("region action must be place, relocate, or remove.")
        if version == 2 and ((action == "place") == ("source" in region)):
            raise _invalid("place has no source; relocate and remove require a bound source.")
        identity = region["id"]
        if (type(identity) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", identity)
                or re.fullmatch(r"ref_image_[0-9]+", identity)):
            raise _invalid("region IDs must be ASCII identifiers and cannot use ref_image_N.")
        if identity in identities:
            raise _invalid("duplicate region IDs are not supported.")
        identities.add(identity)
        description = region["description"]
        if type(description) is not str or not description.strip() or len(description) > MAX_LAYOUT_DESCRIPTION:
            raise _invalid("region descriptions must contain 1–4096 characters.")
        _utf8_size(description)
        if action == "remove" and region["targetBox"] is not None:
            raise _invalid("remove requires targetBox null, not a zero-area target.")
        parsed = {"id": identity, "description": description,
                  "targetBox": None if action == "remove" else _box(region["targetBox"])}
        if version == 2:
            parsed["action"] = action
        if "source" in region:
            source = _record(region["source"], {"referenceIndex", "box"} | ({"expectedReferenceId"} if version == 2 else set()))
            index = source["referenceIndex"]
            maximum = MAX_BOUND_LAYOUT_REFERENCES if version == 2 else MAX_LAYOUT_REFERENCES
            if type(index) is not int or not 0 <= index < maximum:
                raise _invalid(f"source reference index must be an integer between 0 and {maximum - 1}.")
            parsed["source"] = {"referenceIndex": index, "box": _box(source["box"])}
            if version == 2:
                expected = source["expectedReferenceId"]
                if type(expected) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None:
                    raise _invalid("source requires its exact backend-issued reference ID.")
                parsed["source"]["expectedReferenceId"] = expected
        regions.append(parsed)
    result: LayoutConditioning = {"schemaVersion": version, "coordinateSpace": "normalized_xyxy", "regions": regions}
    # The existing V2 canonical writer uses ECMAScript number/string spelling.
    # Key order cannot change byte size. This matches the client's UTF-8 budget
    # for JSON.stringify instead of counting Python's differing 1.0/1e-07 forms.
    if _utf8_size(_stable_json(result)) > MAX_LAYOUT_BYTES:
        raise _invalid("structured layout exceeds 64 KiB of UTF-8 JSON.")
    return result


def _upstream_box(box: LayoutBox) -> list[int | float]:
    # Node execution shares a Python process. Another extension's Decimal
    # precision/traps must neither change coordinates nor fail this projection.
    context = Context(prec=64, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999,
                      capitals=1, clamp=0, flags=[], traps=[])
    with localcontext(context):
        scaled = [Decimal(str(value)) * 1000 for value in (box[1], box[0], box[3], box[2])]
        return [int(value) if value == value.to_integral_value() else float(value) for value in scaled]


def _contains_trailing_regions(prompt: str) -> bool:
    """Detect a competing region list only in explicitly structured mode."""
    decoder = json.JSONDecoder()
    end = len(prompt.rstrip())
    for index, match in enumerate(re.finditer(r"\[\s*\{", prompt)):
        if index >= MAX_LAYOUT_REGIONS:
            raise _invalid("caption contains too many JSON candidates; use Raw mode.")
        try:
            value, position = decoder.raw_decode(prompt, match.start())
        except (ValueError, RecursionError):
            continue
        if position == end and type(value) is list and any(
            type(region) is dict and "id" in region and any(key in region for key in ("bbox", "src_bbox", "tgt_bbox"))
            for region in value
        ):
            return True
    return False


def resolve_layout_prompt(
    raw_prompt: str | None,
    layout_conditioning: object | None,
    *,
    reference_count: int | None = None,
    bound_reference_ids: tuple[str, ...] | None = None,
) -> str | None:
    """Project declared regions into the reviewed caption/regions grammar.

    ``reference_count`` must come from real bound media supplied by the caller,
    never a client JSON claim. An encoder without that context must leave it
    unset, which refuses source-region mappings. Raw mode preserves its input.
    """
    if layout_conditioning is None:
        return raw_prompt
    parsed = parse_layout_conditioning(layout_conditioning)
    if type(raw_prompt) is not str:
        raise _invalid("structured layout requires a text caption.")
    if _utf8_size(raw_prompt) > MAX_STRUCTURED_PROMPT_BYTES:
        raise _invalid("structured prompt exceeds the 1 MiB UTF-8 limit.")
    maximum = MAX_BOUND_LAYOUT_REFERENCES if parsed["schemaVersion"] == 2 else MAX_LAYOUT_REFERENCES
    if reference_count is not None and (type(reference_count) is not int or not 0 <= reference_count <= maximum):
        raise _invalid(f"bound reference count must be an integer between 0 and {maximum}.")
    if bound_reference_ids is not None and (type(bound_reference_ids) is not tuple
            or len(bound_reference_ids) > MAX_BOUND_LAYOUT_REFERENCES
            or any(type(identity) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", identity) is None
                   for identity in bound_reference_ids)):
        raise _invalid("bound reference identities must come from the reference encoder.")
    assert parsed is not None
    if not parsed["regions"]:
        return raw_prompt
    if _contains_trailing_regions(raw_prompt):
        raise _invalid("caption already contains regions; use Raw mode or remove that region list.")
    upstream = []
    for region in parsed["regions"]:
        if f'<{region["id"]}>' not in raw_prompt:
            raise _invalid(f'caption must mention <{region["id"]}> explicitly.')
        if "source" in region:
            source = region["source"]
            if parsed["schemaVersion"] == 2:
                if bound_reference_ids is None or source["referenceIndex"] >= len(bound_reference_ids):
                    raise _invalid("source regions require actual bound reference media; rerun Encode Image.")
                if source["expectedReferenceId"] != bound_reference_ids[source["referenceIndex"]]:
                    raise _invalid("reference content, order, or preprocessing changed; explicitly rebind this source region.")
            elif reference_count is None:
                raise _invalid("source regions require actual bound reference media; use Raw mode for this encoder.")
            elif source["referenceIndex"] >= reference_count:
                raise _invalid("source region refers to a reference that is not bound.")
            upstream.append({"id": region["id"], "from": f'ref_image_{source["referenceIndex"]}',
                             "src_bbox": _upstream_box(source["box"]),
                             "tgt_bbox": None if region.get("action") == "remove" else _upstream_box(region["targetBox"]),
                             "desc": region["description"]})
        else:
            upstream.append({"id": region["id"], "bbox": _upstream_box(region["targetBox"]), "desc": region["description"]})
    result = raw_prompt + "\n" + json.dumps(upstream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if _utf8_size(result) > MAX_STRUCTURED_PROMPT_BYTES:
        raise _invalid("structured prompt exceeds the 1 MiB UTF-8 limit.")
    return result
