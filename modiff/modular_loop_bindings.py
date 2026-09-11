"""Explicit iteration bindings delegated to the installed upstream loop.

There is no loop scheduler here: Diffusers still owns iteration, member order,
progress and state propagation. This adapter binds declared member inputs just
before that member is called, and is removed even when execution fails.
"""

from contextlib import contextmanager
from copy import deepcopy


def _bound_input_summary(state, fields):
    summaries = []
    for field in list(fields)[:8]:
        value = getattr(state, field, None)
        shape = getattr(value, "shape", None)
        detail = f"shape={tuple(shape)}" if shape is not None else (
            f"length={len(value)}" if isinstance(value, (list, tuple)) else ""
        )
        summaries.append(f"{field}={type(value).__name__}" + (f"({detail})" if detail else ""))
    return ", ".join(summaries)


def validate_loop_bindings(block, members, *, parent_path):
    """Validate lowered connections against actual upstream declarations."""
    names = list(block.sub_blocks)
    normalized = {}
    for item in members:
        name = item["path"][-1]
        member = block.sub_blocks[name]
        declared = {spec.name for spec in member.inputs}
        bindings = item.get("iterationBindings", {})
        if not isinstance(bindings, dict) or len(bindings) > 128:
            raise ValueError(f"Loop member {'/'.join(item['path'])}: iteration bindings must be a bounded object.")
        copied = {}
        for field, binding in bindings.items():
            label = f"Loop member {'/'.join(item['path'])}, input {field}"
            if field not in declared:
                raise ValueError(f"{label}: this input is not declared by {type(member).__name__}.")
            if not isinstance(binding, dict):
                raise ValueError(f"{label}: expected an explicit constant or iteration connection.")
            kind = binding.get("kind")
            if kind == "constant" and set(binding) == {"kind", "value"}:
                copied[field] = dict(binding)
                continue
            if kind != "state" or set(binding) != {"kind", "sourcePath", "output", "timing"}:
                raise ValueError(f"{label}: malformed iteration connection.")
            path = binding["sourcePath"]
            if not isinstance(path, list) or path[:-1] != list(parent_path) or not path or path[-1] not in names:
                raise ValueError(f"{label}: the producer must belong to this same loop.")
            producer = block.sub_blocks[path[-1]]
            output = binding["output"]
            if output not in {spec.name for spec in producer.intermediate_outputs}:
                raise ValueError(f"{label}: {type(producer).__name__} does not declare output {output!r}.")
            timing = binding["timing"]
            if timing not in {"current", "previous"}:
                raise ValueError(f"{label}: choose current or previous iteration explicitly.")
            if timing == "current" and names.index(path[-1]) >= names.index(name):
                raise ValueError(f"{label}: current-iteration producer must run before its consumer; use previous iteration for a carried value.")
            copied[field] = dict(binding)
        normalized[name] = copied
    return normalized


class _BoundLoopMember:
    def __init__(self, original, name, bindings, shared, parent_path, coerce_input):
        self.original = original
        self.name = name
        self.bindings = bindings
        self.shared = shared
        self.parent_path = parent_path
        self.coerce_input = coerce_input

    def _supplied(self, name):
        binding = self.bindings.get(name, {})
        return binding.get("kind") == "constant" or binding.get("timing") == "current"

    @property
    def inputs(self):
        # The upstream loop computes its entry requirements from these public
        # declarations. Explicit constants/current producers supply these fields
        # inside the iteration; previous values still need an initial seed.
        return [spec for spec in self.original.inputs if not self._supplied(spec.name)]

    @property
    def required_inputs(self):
        return [name for name in getattr(self.original, "required_inputs", ()) if not self._supplied(name)]

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __call__(self, components, state, **kwargs):
        shared = self.shared
        # The first member marks a new actual upstream iteration. Do not assume
        # a particular loop keyword (i, step_index, etc.) or invent a clock.
        if self.name == shared["first"]:
            shared["previous"] = shared["current"]
            shared["current"] = {}
            if shared["iteration"] == 0:
                # Snapshot before any member or input binding can mutate state.
                for key in shared["initial"]:
                    value = getattr(state, key[1], None)
                    if value is not None:
                        shared["previous"][key] = _snapshot_value(value)
        label = f"Loop member {'/'.join([*self.parent_path, self.name])} ({type(self.original).__name__})"
        for field, binding in self.bindings.items():
            if binding["kind"] == "constant":
                # A mutable user constant must not accumulate incidental writes
                # across iterations or executions.
                value = deepcopy(binding["value"])
                if self.coerce_input:
                    spec = next(spec for spec in self.original.inputs if spec.name == field)
                    value = self.coerce_input(field, value, type_hint=getattr(spec, "type_hint", None))
            else:
                key = (binding["sourcePath"][-1], binding["output"])
                bank = shared[binding["timing"]]
                if key not in bank:
                    if binding["timing"] == "previous" and shared["iteration"] == 0:
                        raise ValueError(f"{label}, input {field}: previous-iteration connection needs initial state {binding['output']!r} for iteration 0.")
                    else:
                        raise ValueError(f"{label}, input {field}: producer output {binding['output']!r} is missing in the {binding['timing']} iteration.")
                else:
                    # Consumers must not mutate the bank seen by their peers.
                    value = _snapshot_value(bank[key])
            setattr(state, field, value)
        bound_inputs = _bound_input_summary(state, self.bindings)
        try:
            components, state = self.original(components, state, **kwargs)
        except Exception as error:
            if isinstance(error, MemoryError) or any(cls.__name__ == "OutOfMemoryError" for cls in type(error).__mro__):
                # Preserve the exception category used by existing resource
                # diagnostics and recovery; do not relabel an OOM as bad wiring.
                error.add_note(label)
                raise
            detail = f" Bound inputs: {bound_inputs}." if bound_inputs else ""
            raise ValueError(f"{label}: {error}.{detail}") from error
        for output in shared["capture"].get(self.name, ()):
            value = getattr(state, output, None)
            if value is None:
                raise ValueError(f"{label}: declared connected output {output!r} was not produced.")
            # Only explicitly bound values are snapshotted. Cloning preserves
            # provenance when a later block mutates its input tensor in place.
            shared["current"][(self.name, output)] = _snapshot_value(value)
        if self.name == shared["last"]:
            shared["iteration"] += 1
        return components, state


def _snapshot_value(value):
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else deepcopy(value)


@contextmanager
def bind_upstream_loop_inputs(block, members, *, parent_path, coerce_input=None):
    if all("iterationBindings" not in item or item["iterationBindings"] == {} for item in members):
        # Preserve the existing descriptor-only execution path exactly. An
        # unedited loop does not need the new value-binding adapter at all.
        yield
        return
    bindings = validate_loop_bindings(block, members, parent_path=parent_path)
    if not any(bindings.values()):
        yield
        return
    original = list(block.sub_blocks.items())
    capture = {}
    initial = set()
    for fields in bindings.values():
        for binding in fields.values():
            if binding["kind"] == "state":
                capture.setdefault(binding["sourcePath"][-1], set()).add(binding["output"])
                if binding["timing"] == "previous":
                    initial.add((binding["sourcePath"][-1], binding["output"]))
    shared = {"first": original[0][0], "last": original[-1][0], "iteration": 0,
              "current": {}, "previous": {}, "capture": capture, "initial": initial}
    try:
        for name, member in original:
            block.sub_blocks[name] = _BoundLoopMember(member, name, bindings.get(name, {}), shared, parent_path, coerce_input)
        yield
    finally:
        for name, member in original:
            block.sub_blocks[name] = member
