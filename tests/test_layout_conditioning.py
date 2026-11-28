"""Bounded structured prompt data, independently of a model runtime."""
from __future__ import annotations

import copy
from decimal import Inexact, localcontext
import json
from pathlib import Path

import pytest

from modiff.layout_conditioning import parse_layout_conditioning, resolve_layout_prompt


FIXTURE = json.loads((Path(__file__).parent / "fixtures/layout-conditioning.v1.json").read_text(encoding="utf-8"))


def layout():
    return copy.deepcopy(FIXTURE["valid"][1]["layout"])


@pytest.mark.parametrize("case", FIXTURE["valid"], ids=lambda case: case["name"])
def test_shared_valid_cases_preserve_authored_data(case):
    original = copy.deepcopy(case)
    assert resolve_layout_prompt(case["prompt"], case["layout"], reference_count=case.get("referenceCount")) == case["expectedPrompt"]
    assert case == original


@pytest.mark.parametrize("case", FIXTURE["invalid"], ids=lambda case: case["name"])
def test_shared_invalid_cases_are_not_silently_repaired(case):
    with pytest.raises(ValueError):
        parse_layout_conditioning(case["layout"])


def test_raw_mode_preserves_none_and_has_no_structured_side_effects():
    assert resolve_layout_prompt(None, None) is None
    assert resolve_layout_prompt("\n\t  original  \n", None) == "\n\t  original  \n"


def test_empty_structured_collection_keeps_the_caption_unchanged():
    value = layout()
    value["regions"] = []
    assert resolve_layout_prompt("  original\n", value) == "  original\n"


def test_region_source_does_not_authorize_its_own_reference_context():
    case = FIXTURE["valid"][2]
    with pytest.raises(ValueError, match="bound reference"):
        resolve_layout_prompt(case["prompt"], case["layout"])
    with pytest.raises(ValueError, match="reference"):
        resolve_layout_prompt(case["prompt"], case["layout"], reference_count=1)
    assert resolve_layout_prompt(case["prompt"], case["layout"], reference_count=2) == case["expectedPrompt"]


@pytest.mark.parametrize("count", [True, -1, 9, 1.5, "2"])
def test_reference_context_must_be_an_actual_bounded_count(count):
    with pytest.raises(ValueError):
        resolve_layout_prompt("<apple_1>", layout(), reference_count=count)


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [-0.1, 0, 1, 1], [0, 0, 1.1, 1], [0, 0, float("nan"), 1], [0, 0, float("inf"), 1], [0, 0, 1], ["0", 0, 1, 1], [0, 0, 10**1000, 1]])
def test_nonfinite_ambiguous_or_empty_boxes_fail_before_serialization(box):
    value = layout()
    value["regions"][0]["targetBox"] = box
    with pytest.raises(ValueError):
        parse_layout_conditioning(value)


def test_duplicate_ids_and_undeclared_keys_are_refused():
    value = layout()
    value["regions"].append(copy.deepcopy(value["regions"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        parse_layout_conditioning(value)
    value = layout()
    value["regions"][0]["trust_remote_code"] = True
    with pytest.raises(ValueError):
        parse_layout_conditioning(value)


def test_caption_tags_are_explicit_and_existing_region_syntax_is_not_merged():
    with pytest.raises(ValueError, match="<apple_1>"):
        resolve_layout_prompt("An apple.", layout())
    with pytest.raises(ValueError, match="Raw"):
        resolve_layout_prompt('<apple_1> [{"id":"old","bbox":[0,0,1000,1000],"desc":"old"}]', layout())


def test_utf8_budget_and_codepoint_limit_are_distinct():
    value = layout()
    value["regions"][0]["description"] = "🍎" * 4096
    parse_layout_conditioning(value)
    value["regions"][0]["description"] += "🍎"
    with pytest.raises(ValueError, match="4096"):
        parse_layout_conditioning(value)
    value = layout()
    value["regions"] = [dict(value["regions"][0], id=f"item_{index}", description="🍎" * 4096) for index in range(4)]
    with pytest.raises(ValueError, match="64 KiB"):
        parse_layout_conditioning(value)


@pytest.mark.parametrize("text", ["\ud800", "\udfff"])
def test_unpaired_surrogates_are_not_replaced(text):
    value = layout()
    value["regions"][0]["description"] = text
    with pytest.raises(ValueError, match="Unicode"):
        parse_layout_conditioning(value)


def test_order_escaping_and_fractional_coordinate_conversion_are_deterministic():
    value = layout()
    value["regions"][0]["description"] = '雪 "quoted"\ncaption'
    value["regions"][0]["targetBox"] = [0.000125, 0.2505, 0.87525, 0.7505]
    result = resolve_layout_prompt("<apple_1>\n", value)
    assert result.startswith("<apple_1>\n\n")
    encoded = json.loads(result.splitlines()[-1])
    assert encoded == [{"id": "apple_1", "bbox": [250.5, 0.125, 750.5, 875.25], "desc": '雪 "quoted"\ncaption'}]
    assert resolve_layout_prompt("<apple_1>\n", value) == result


def test_region_count_is_bounded_without_traversing_arbitrary_objects():
    value = layout()
    value["regions"] = [object()] * 33
    with pytest.raises(ValueError, match="32"):
        parse_layout_conditioning(value)


@pytest.mark.parametrize("precision", [2, 3, 5])
def test_projection_is_independent_of_ambient_decimal_context(precision):
    value = layout()
    value["regions"][0]["targetBox"] = [0.000125, 0.2505, 0.87525, 0.7505]
    with localcontext() as context:
        context.prec = precision
        context.traps[Inexact] = True
        before = context.copy()
        result = resolve_layout_prompt("<apple_1>", value)
        assert context.prec == before.prec and context.flags == before.flags and context.traps == before.traps
    assert json.loads(result.splitlines()[-1])[0]["bbox"] == [250.5, 0.125, 750.5, 875.25]


def test_json_discriminator_never_invokes_opaque_equality():
    calls = []

    class Opaque:
        def __eq__(self, other):
            calls.append(other)
            return True

    value = layout()
    value["coordinateSpace"] = Opaque()
    with pytest.raises(ValueError, match="schema|coordinate"):
        parse_layout_conditioning(value)
    assert not calls


V2 = json.loads((Path(__file__).parent / "fixtures/layout-conditioning.v2.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", V2["valid"], ids=lambda case: case["name"])
def test_version_two_actions_preserve_shared_projection(case):
    assert parse_layout_conditioning(case["layout"]) == case["layout"]
    assert resolve_layout_prompt(case["prompt"], case["layout"],
                                 bound_reference_ids=("sha256:" + "a" * 64,) * 10) == case["expectedPrompt"]


@pytest.mark.parametrize("case", V2["invalid"], ids=lambda case: case["name"])
def test_version_two_invalid_actions_are_not_repaired(case):
    with pytest.raises(ValueError):
        parse_layout_conditioning(case["layout"])


@pytest.mark.parametrize("index", [0, 7, 8, 9])
def test_version_two_source_requires_current_exact_identity_not_count(index):
    case = copy.deepcopy(V2["valid"][1])
    source = case["layout"]["regions"][0]["source"]
    source["referenceIndex"] = index
    for kwargs in ({}, {"reference_count": 10}, {"bound_reference_ids": ("sha256:" + "b" * 64,) * 10}):
        with pytest.raises(ValueError, match="bound|rebind"):
            resolve_layout_prompt(case["prompt"], case["layout"], **kwargs)
    resolve_layout_prompt(case["prompt"], case["layout"], bound_reference_ids=(source["expectedReferenceId"],) * 10)
    source["referenceIndex"] = 10
    with pytest.raises(ValueError):
        parse_layout_conditioning(case["layout"])
