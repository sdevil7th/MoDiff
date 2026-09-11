"""Native form strings must obey the exact upstream input, not a name whitelist."""

from typing import Optional, Union

import pytest

from modules.ModularDiffusers.reviewed_blocks import _runtime_input_value


@pytest.mark.parametrize("name,value,hint,expected", [
    ("controlnet_conditioning_scale", "0.00", Union[float, list[float]], 0.0),
    ("control_guidance_start", "0.20", Union[float, list[float]], 0.2),
    ("control_guidance_end", "1.00", Union[float, list[float]], 1.0),
    ("strength", "0.90", Optional[float], 0.9),
    ("layers", "4", int, 4),
    ("cfg_normalization", "false", bool, False),
    ("cfg_normalization", "true", bool, True),
    ("controlnet_conditioning_scale", [0.5, 1.0], Union[float, list[float]], [0.5, 1.0]),
    ("controlnet_conditioning_scale", "[0.5, 1.0]", Union[float, list[float]], [0.5, 1.0]),
])
def test_exact_numeric_and_boolean_contracts(name, value, hint, expected):
    assert _runtime_input_value(name, value, type_hint=hint) == expected


@pytest.mark.parametrize("value,hint", [
    (True, float), ("nan", float), ("inf", float), ("", float),
    ("garbage", float), ("4.5", int), (False, int), ("maybe", bool),
    ("[1, true]", Union[float, list[float]]),
])
def test_invalid_typed_values_fail_before_diffusers(value, hint):
    with pytest.raises(ValueError, match="sample"):
        _runtime_input_value("sample", value, type_hint=hint)


@pytest.mark.parametrize("hint", [str, Union[str, float], object, None])
def test_text_and_opaque_inputs_are_not_guessed(hint):
    assert _runtime_input_value("prompt", "001.00", type_hint=hint) == "001.00"


def test_typed_container_values_do_not_mutate_persisted_inputs():
    values = ["0.5", "1.0"]
    assert _runtime_input_value("scale", values, type_hint=list[float]) == [0.5, 1.0]
    assert values == ["0.5", "1.0"]


def test_exact_text_alternative_takes_precedence_over_legacy_field_name():
    assert _runtime_input_value("width", "auto", type_hint=Union[str, int]) == "auto"
    assert _runtime_input_value("guidance_scale", "001.00", type_hint=Union[str, float]) == "001.00"
