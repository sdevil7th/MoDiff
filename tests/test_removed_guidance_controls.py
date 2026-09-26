import copy
import json
from pathlib import Path

import pytest

from modiff.block_definition_v2 import (
    block_definition_content_hash_v2,
    validate_block_definition_v2,
    validate_block_instance_v2,
)
from tests.test_block_route_selection_v1 import route_selection_fixture


CASES = json.loads((Path(__file__).parent / "fixtures/block_removed_controls_v1.json").read_text())


def test_removed_controls_round_trip_without_execution_identity_changes():
    original = route_selection_fixture()
    for bindings in CASES["valid"]:
        instance = copy.deepcopy(original)
        instance["presentation"]["removedControlBindings"] = bindings
        assert validate_block_instance_v2(instance) == instance
        definition = copy.deepcopy(instance["definitionSnapshot"])
        definition["removedControlBindings"] = bindings
        assert validate_block_definition_v2(definition) == definition
        assert block_definition_content_hash_v2(definition) == definition["contentHash"]


def test_removed_controls_reject_malformed_and_unbounded_metadata():
    original = route_selection_fixture()
    invalid = CASES["invalid"] + [[{"nodeId": f"node-{i}", "fieldId": "scale"} for i in range(4097)]]
    for bindings in invalid:
        instance = copy.deepcopy(original)
        instance["presentation"]["removedControlBindings"] = bindings
        with pytest.raises(ValueError, match="removedControlBindings"):
            validate_block_instance_v2(instance)
        definition = copy.deepcopy(instance["definitionSnapshot"])
        definition["removedControlBindings"] = bindings
        with pytest.raises(ValueError, match="removedControlBindings"):
            validate_block_definition_v2(definition)
