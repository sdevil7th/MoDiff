from types import SimpleNamespace

import pytest

from modiff.modular_loop_bindings import bind_upstream_loop_inputs, validate_loop_bindings


class Member:
    inputs = [SimpleNamespace(name="value")]
    intermediate_outputs = [SimpleNamespace(name="value")]

    def __init__(self, increment=1, fail=False):
        self.increment = increment
        self.fail = fail

    def __call__(self, components, state, **kwargs):
        if self.fail:
            raise RuntimeError("incompatible tensor dimensions")
        state.value += self.increment
        return components, state


def fixture(binding=None):
    block = SimpleNamespace(sub_blocks={"first": Member(), "last": Member(10)})
    members = [{"path": ["loop", name], "iterationBindings": bindings}
               for name, bindings in [("first", {"value": binding} if binding else {}), ("last", {})]]
    return block, members


@pytest.mark.parametrize('tensor', [False, True])
def test_previous_values_are_snapshotted_before_earlier_producer_and_each_consumer(tensor):
    import torch
    observed = []

    class Producer(Member):
        def __call__(self, components, state, **kwargs):
            state.value += 10
            return components, state

    class Consumer(Member):
        inputs = [SimpleNamespace(name='previous_value')]
        intermediate_outputs = []

        def __call__(self, components, state, **kwargs):
            observed.append(int(state.previous_value))
            state.previous_value += 100  # Must not change another consumer's snapshot.
            return components, state

    block = SimpleNamespace(sub_blocks={'producer': Producer(), 'one': Consumer(), 'two': Consumer()})
    binding = {'kind': 'state', 'sourcePath': ['loop', 'producer'], 'output': 'value', 'timing': 'previous'}
    members = [{'path': ['loop', name], 'iterationBindings': {} if name == 'producer' else {'previous_value': binding}}
               for name in block.sub_blocks]
    originals = dict(block.sub_blocks)
    for _ in range(2):
        state = SimpleNamespace(value=torch.tensor(1.) if tensor else 1)
        with bind_upstream_loop_inputs(block, members, parent_path=['loop']):
            for _ in range(3):
                for member in block.sub_blocks.values():
                    _, state = member(None, state)
        assert block.sub_blocks == originals
    assert observed == [1, 1, 11, 11, 21, 21] * 2


def test_constant_applies_each_iteration_and_restores_actual_members():
    block, members = fixture({"kind": "constant", "value": 2})
    original = dict(block.sub_blocks)
    state = SimpleNamespace(value=100)
    with bind_upstream_loop_inputs(block, members, parent_path=["loop"]):
        results = []
        for index in range(3):
            for member in block.sub_blocks.values():
                _, state = member(None, state, i=index)
            results.append(state.value)
    assert results == [13, 13, 13]
    assert block.sub_blocks == original


def test_explicit_constant_supplies_entry_requirement_and_is_normalized_each_iteration():
    block, members = fixture({"kind": "constant", "value": "2"})
    block.sub_blocks["first"].required_inputs = ["value"]
    original = block.sub_blocks["first"]
    with bind_upstream_loop_inputs(block, members, parent_path=["loop"], coerce_input=lambda name, value, **kwargs: int(value)):
        bound = block.sub_blocks["first"]
        assert bound.inputs == []
        assert bound.required_inputs == []
        for _ in range(3):
            _, state = bound(None, SimpleNamespace())
            assert state.value == 3
    assert block.sub_blocks["first"] is original
    assert original.required_inputs == ["value"]


def test_carried_connection_uses_initial_state_then_previous_producer():
    block, members = fixture({"kind": "state", "sourcePath": ["loop", "last"], "output": "value", "timing": "previous"})
    state = SimpleNamespace(value=1)
    with bind_upstream_loop_inputs(block, members, parent_path=["loop"]):
        values = []
        for _ in range(3):
            for member in block.sub_blocks.values():
                _, state = member(None, state)
            values.append(state.value)
            state.value = -100  # Must not become the next iteration's producer.
    assert values == [12, 23, 34]


@pytest.mark.parametrize("patch,message", [
    ({"sourcePath": ["other", "last"]}, "same loop"),
    ({"output": "undeclared"}, "does not declare output"),
    ({"timing": "current"}, "must run before"),
    ({"timing": "maybe"}, "current or previous"),
])
def test_invalid_connections_have_member_and_input_diagnostics(patch, message):
    binding = {"kind": "state", "sourcePath": ["loop", "last"], "output": "value", "timing": "previous", **patch}
    block, members = fixture(binding)
    with pytest.raises(ValueError, match=f"loop/first, input value:.*{message}"):
        validate_loop_bindings(block, members, parent_path=["loop"])


def test_failed_member_has_exact_path_and_adapter_is_removed():
    block, members = fixture({"kind": "constant", "value": 2})
    block.sub_blocks["last"].fail = True
    original = dict(block.sub_blocks)
    with pytest.raises(ValueError, match="loop/last.*incompatible tensor dimensions"):
        with bind_upstream_loop_inputs(block, members, parent_path=["loop"]):
            for member in block.sub_blocks.values():
                member(None, SimpleNamespace(value=0))
    assert block.sub_blocks == original


def test_failed_bound_value_reports_its_actual_type_without_dumping_contents():
    block, members = fixture({"kind": "constant", "value": ["private-input-content"]})
    block.sub_blocks["first"].fail = True
    with pytest.raises(ValueError, match=r"loop/first.*Bound inputs: value=list\(length=1\)") as error:
        with bind_upstream_loop_inputs(block, members, parent_path=["loop"]):
            block.sub_blocks["first"](None, SimpleNamespace(value=0))
    assert "private-input-content" not in str(error.value)


def test_unmodified_loop_has_no_adapter_and_no_state_changes():
    block, members = fixture()
    original = dict(block.sub_blocks)
    with bind_upstream_loop_inputs(block, members, parent_path=["loop"]):
        assert block.sub_blocks == original


def test_missing_initial_carried_state_is_named():
    block, members = fixture({"kind": "state", "sourcePath": ["loop", "last"], "output": "value", "timing": "previous"})
    with pytest.raises(ValueError, match="initial state 'value' for iteration 0"):
        with bind_upstream_loop_inputs(block, members, parent_path=["loop"]):
            block.sub_blocks["first"](None, SimpleNamespace())


def test_memory_failure_retains_existing_resource_error_category():
    class OutOfMemoryError(RuntimeError):
        pass

    class OutOfMemoryMember(Member):
        def __call__(self, components, state, **kwargs):
            raise OutOfMemoryError("GPU allocation failed")

    block, members = fixture({"kind": "constant", "value": 2})
    block.sub_blocks["first"] = OutOfMemoryMember()
    original = dict(block.sub_blocks)
    with pytest.raises(OutOfMemoryError) as error:
        with bind_upstream_loop_inputs(block, members, parent_path=["loop"]):
            block.sub_blocks["first"](None, SimpleNamespace())
    assert "loop/first" in error.value.__notes__[0]
    assert block.sub_blocks == original
