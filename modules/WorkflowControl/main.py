"""Explicit boundary and collection nodes used by visual loop containers."""

import itertools
import json
from typing import Any

from modiff.NodeBase import NodeBase


def parse_shot_list(value: Any, *, maximum: int = 24) -> dict[str, Any]:
    """Normalize authored or locally generated shot JSON into a loop-ready list."""

    if isinstance(value, str):
        source = value.strip()
        if source.startswith("```"):
            lines = source.splitlines()
            source = "\n".join(lines[1:-1] if len(lines) > 2 else lines).strip()
        try:
            value = json.loads(source)
        except json.JSONDecodeError:
            starts = [index for index in (source.find("["), source.find("{")) if index >= 0]
            start = min(starts) if starts else -1
            end = max(source.rfind("]"), source.rfind("}"))
            if start < 0 or end <= start:
                raise ValueError("Shot list output must contain a JSON array or an object with a shots array.")
            try:
                value = json.loads(source[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"Shot list contains invalid JSON: {exc.msg}.") from exc
    if isinstance(value, dict):
        value = value.get("shots")
    if not isinstance(value, list) or not value:
        raise ValueError("Shot list must contain at least one shot object.")
    limit = max(1, min(100, int(maximum)))
    if len(value) > limit:
        raise ValueError(f"Shot list contains {len(value)} shots, above the configured maximum of {limit}.")

    shots = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"Shot {index + 1} must be a JSON object.")
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"Shot {index + 1} needs a non-empty prompt.")
        try:
            duration = float(raw.get("duration_seconds", raw.get("duration", 0)))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Shot {index + 1} duration must be a number of seconds.") from exc
        if duration <= 0 or duration > 120:
            raise ValueError(f"Shot {index + 1} duration must be greater than 0 and at most 120 seconds.")
        shot = {
            "index": index,
            "title": str(raw.get("title") or f"Shot {index + 1}").strip(),
            "prompt": prompt,
            "duration_seconds": duration,
            "transition": str(raw.get("transition") or ("cut" if index else "start")).strip(),
        }
        for name in ("audio_prompt", "reference_strategy", "notes"):
            if raw.get(name) not in (None, ""):
                shot[name] = str(raw[name]).strip()
        shots.append(shot)
    total = sum(shot["duration_seconds"] for shot in shots)
    return {"shots": shots, "shot_list_json": json.dumps(shots, ensure_ascii=False), "total_duration_seconds": total}


class AuthorShotList(NodeBase):
    """Validate an authored shot list and expose records suitable for a loop."""

    label = "Author Shot List"
    category = "Workflow Control"
    resizable = True
    params = {
        "shots_json": {
            "label": "Shots (JSON)",
            "display": "textarea",
            "type": "text",
            "default": '[{"title":"Opening","prompt":"Describe the opening action and camera movement.","duration_seconds":5}]',
            "description": "Each shot needs prompt and duration_seconds. Optional: title, transition, audio_prompt, reference_strategy, notes.",
        },
        "maximum_shots": {"label": "Maximum Shots", "type": "int", "default": 12, "min": 1, "max": 100},
        "shots": {"label": "Shots", "display": "output", "type": "collection"},
        "shot_list_json": {"label": "Shot List JSON", "display": "output", "type": "text"},
        "total_duration_seconds": {"label": "Approximate Duration", "display": "output", "type": "float"},
    }

    def execute(self, **kwargs):
        return parse_shot_list(kwargs.get("shots_json"), maximum=int(kwargs.get("maximum_shots") or 12))


class LoopInput(NodeBase):
    """Expose the initial value, then the carried value on later iterations."""

    label = "Loop Input"
    category = "Workflow Control"
    params = {
        "initial": {"label": "Initial Value", "display": "input", "type": "any"},
        "value": {"label": "Current Value", "display": "output", "type": "any"},
    }

    def execute(self, **kwargs):
        return {"value": kwargs.get("initial")}


class LoopIndex(NodeBase):
    """Expose zero- and one-based iteration values injected by the executor."""

    label = "Loop Index"
    category = "Workflow Control"
    params = {
        "index_value": {"label": "Index", "type": "int", "default": 0, "hidden": True},
        "iteration_count": {"label": "Iterations", "type": "int", "default": 1, "hidden": True},
        "index": {"label": "Index (0-based)", "display": "output", "type": "int"},
        "iteration": {"label": "Iteration (1-based)", "display": "output", "type": "int"},
        "is_first": {"label": "Is First", "display": "output", "type": "bool"},
        "is_last": {"label": "Is Last", "display": "output", "type": "bool"},
    }

    def execute(self, **kwargs):
        index = max(0, int(kwargs.get("index_value") or 0))
        count = max(1, int(kwargs.get("iteration_count") or 1))
        return {
            "index": index,
            "iteration": index + 1,
            "is_first": index == 0,
            "is_last": index == count - 1,
        }


class LoopItems(NodeBase):
    """Expose one item from a collection for collection-driven loops."""

    label = "Loop Items"
    category = "Workflow Control"
    params = {
        "collection": {"label": "Collection", "display": "input", "type": "any"},
        "item_index": {"label": "Item Index", "type": "int", "default": 0, "hidden": True},
        "item": {"label": "Item", "display": "output", "type": "any"},
        "index": {"label": "Index", "display": "output", "type": "int"},
        "count": {"label": "Count", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        collection = kwargs.get("collection")
        if isinstance(collection, dict):
            values = [{"key": key, "value": value} for key, value in collection.items()]
        elif isinstance(collection, (list, tuple)):
            values = list(collection)
        else:
            raise TypeError("Loop Items needs a list, tuple, or object collection.")
        index = int(kwargs.get("item_index") or 0)
        if index < 0 or index >= len(values):
            raise IndexError(f"Loop item index {index} is outside a collection of {len(values)} item(s).")
        return {"item": values[index], "index": index, "count": len(values)}


class LoopResult(NodeBase):
    """Mark the value to carry/collect and optionally stop the loop early."""

    label = "Loop Result"
    category = "Workflow Control"
    params = {
        "value_input": {"label": "Value", "display": "input", "type": "any"},
        "stop_input": {"label": "Stop", "display": "input", "type": "bool", "default": False},
        "value": {"label": "Last Value", "display": "output", "type": "any"},
        "collection": {"label": "Collected Values", "display": "output", "type": "any"},
        "stopped": {"label": "Stopped Early", "display": "output", "type": "bool"},
    }

    def execute(self, **kwargs):
        value: Any = kwargs.get("value_input")
        stopped = bool(kwargs.get("stop_input", False))
        return {"value": value, "collection": [value], "stopped": stopped}


class SeedSequence(NodeBase):
    label = "Seed Sequence"
    category = "Workflow Control"
    params = {
        "start": {"label": "Start Seed", "type": "int", "default": 0},
        "count": {"label": "Count", "type": "int", "default": 4, "min": 1, "max": 10000},
        "step": {"label": "Step", "type": "int", "default": 1},
        "seeds": {"label": "Seeds", "display": "output", "type": "any"},
    }

    def execute(self, **kwargs):
        start = int(kwargs.get("start") or 0)
        count = max(1, min(10000, int(kwargs.get("count") or 1)))
        step = int(kwargs.get("step") or 1)
        return {"seeds": [start + index * step for index in range(count)]}


class ParameterMatrix(NodeBase):
    label = "Parameter Matrix"
    category = "Workflow Control"
    resizable = True
    params = {
        "parameters": {
            "label": "Parameters (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "{}",
            "description": 'Example: {"steps": [20, 30], "guidance": [3.5, 5.0]}',
        },
        "max_combinations": {"label": "Maximum", "type": "int", "default": 256, "min": 1, "max": 10000},
        "mode": {"label": "Mode", "type": "string", "options": ["cartesian", "zip"], "default": "cartesian"},
        "combinations": {"label": "Combinations", "display": "output", "type": "any"},
        "count": {"label": "Count", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        raw = kwargs.get("parameters") or "{}"
        try:
            parameters = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            raise ValueError(f"Parameter Matrix needs valid JSON: {exc.msg}.") from exc
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("Parameter Matrix needs a non-empty JSON object.")
        keys = list(parameters)
        values = []
        total = 1
        for key in keys:
            choices = parameters[key]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Parameter {key!r} must contain a non-empty JSON array.")
            values.append(choices)
            total *= len(choices)
        mode = str(kwargs.get("mode") or "cartesian")
        if mode == "cartesian":
            combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        elif mode == "zip":
            lengths = {len(choices) for choices in values if len(choices) != 1}
            if len(lengths) > 1:
                raise ValueError("Zip mode needs arrays of the same length; one-item arrays may be broadcast.")
            count = max(map(len, values))
            combinations = [
                {key: choices[0] if len(choices) == 1 else choices[index] for key, choices in zip(keys, values)}
                for index in range(count)
            ]
        else:
            raise ValueError(f"Unsupported Parameter Matrix mode {mode!r}.")
        maximum = max(1, min(10000, int(kwargs.get("max_combinations") or 256)))
        if len(combinations) > maximum:
            raise ValueError(
                f"Parameter Matrix would create {len(combinations)} combinations, above the configured maximum of {maximum}."
            )
        return {"combinations": combinations, "count": len(combinations)}


class FanOut(NodeBase):
    """Create explicit ordered branches from one value and optional overrides."""

    label = "Fan Out"
    category = "Workflow Control"
    params = {
        "value": {"label": "Value", "display": "input", "type": "any"},
        "count": {"label": "Branches", "type": "int", "default": 2, "min": 1, "max": 10000},
        "branch_overrides": {
            "label": "Branch Overrides (JSON)",
            "display": "textarea",
            "type": "text",
            "default": "[]",
            "description": "Optional array of objects, one per branch.",
        },
        "branches": {"label": "Branches", "display": "output", "type": "collection"},
    }

    def execute(self, **kwargs):
        count = max(1, min(10000, int(kwargs.get("count") or 1)))
        raw = kwargs.get("branch_overrides") or "[]"
        try:
            overrides = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            raise ValueError(f"Fan Out needs valid branch override JSON: {exc.msg}.") from exc
        if not isinstance(overrides, list) or any(not isinstance(item, dict) for item in overrides):
            raise ValueError("Fan Out branch overrides must be a JSON array of objects.")
        if len(overrides) > count:
            raise ValueError("Fan Out has more override objects than configured branches.")
        return {
            "branches": [
                {
                    "index": index,
                    "value": kwargs.get("value"),
                    "overrides": dict(overrides[index]) if index < len(overrides) else {},
                }
                for index in range(count)
            ]
        }


class CollectionBatch(NodeBase):
    label = "Batch Collection"
    category = "Workflow Control"
    params = {
        "collection": {"label": "Collection", "display": "input", "type": "any"},
        "batch_size": {"label": "Batch Size", "type": "int", "default": 4, "min": 1, "max": 10000},
        "batches": {"label": "Batches", "display": "output", "type": "any"},
    }

    def execute(self, **kwargs):
        collection = kwargs.get("collection")
        if not isinstance(collection, (list, tuple)):
            raise TypeError("Batch Collection needs a list or tuple.")
        size = max(1, min(10000, int(kwargs.get("batch_size") or 1)))
        values = list(collection)
        return {"batches": [values[index : index + size] for index in range(0, len(values), size)]}


class CollectionFlatten(NodeBase):
    label = "Flatten Collection"
    category = "Workflow Control"
    params = {
        "collection": {"label": "Collection", "display": "input", "type": "any"},
        "flattened": {"label": "Flattened", "display": "output", "type": "any"},
    }

    def execute(self, **kwargs):
        collection = kwargs.get("collection")
        if not isinstance(collection, (list, tuple)):
            raise TypeError("Flatten Collection needs a list or tuple.")
        flattened = []
        for item in collection:
            flattened.extend(item if isinstance(item, (list, tuple)) else [item])
        return {"flattened": flattened}


class CollectionItem(NodeBase):
    """Select one ordered collection value with an explicit fallback policy."""

    label = "Get Collection Item"
    category = "Workflow Control"
    params = {
        "collection": {"label": "Collection", "display": "input", "type": "any"},
        "index": {"label": "Index", "display": "input", "type": "int", "default": 0},
        "out_of_range": {
            "label": "Out of Range",
            "type": "string",
            "options": ["error", "use_first", "use_last"],
            "default": "error",
        },
        "item": {"label": "Item", "display": "output", "type": "any"},
        "count": {"label": "Count", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        collection = kwargs.get("collection")
        if not isinstance(collection, (list, tuple)):
            raise TypeError("Get Collection Item needs a list or tuple.")
        values = list(collection)
        if not values:
            raise ValueError("Get Collection Item needs a non-empty collection.")
        index = int(kwargs.get("index") or 0)
        if not 0 <= index < len(values):
            policy = str(kwargs.get("out_of_range") or "error")
            if policy == "use_first":
                index = 0
            elif policy == "use_last":
                index = len(values) - 1
            elif policy == "error":
                raise IndexError(f"Collection index {index} is outside a collection of {len(values)} item(s).")
            else:
                raise ValueError(f"Unsupported collection fallback policy {policy!r}.")
        return {"item": values[index], "count": len(values)}


class GetField(NodeBase):
    """Read one dotted field from a loop item, job record, or preset."""

    label = "Get Record Field"
    category = "Workflow Control"
    params = {
        "record": {"label": "Record", "display": "input", "type": "any"},
        "field": {"label": "Field", "type": "string", "default": "value"},
        "default_value": {"label": "Default", "display": "input", "type": "any"},
        "value": {"label": "Value", "display": "output", "type": "any"},
        "found": {"label": "Found", "display": "output", "type": "bool"},
    }

    def execute(self, **kwargs):
        value = kwargs.get("record")
        found = True
        for part in [item for item in str(kwargs.get("field") or "").split(".") if item]:
            if isinstance(value, dict) and part in value:
                value = value[part]
            elif isinstance(value, (list, tuple)) and part.isdigit() and int(part) < len(value):
                value = value[int(part)]
            else:
                found = False
                value = kwargs.get("default_value")
                break
        return {"value": value, "found": found}


class ParameterPreset(NodeBase):
    """Create a named, serializable set of ordinary workflow parameters."""

    label = "Parameter Preset"
    category = "Workflow Control"
    resizable = True
    params = {
        "name": {"label": "Name", "type": "string", "default": "Preset"},
        "values": {"label": "Values (JSON)", "display": "textarea", "type": "text", "default": "{}"},
        "preset": {"label": "Preset", "display": "output", "type": "any"},
        "manifest": {"label": "Manifest", "display": "output", "type": "string"},
    }

    def execute(self, **kwargs):
        raw = kwargs.get("values") or "{}"
        try:
            values = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            raise ValueError(f"Parameter Preset needs valid JSON: {exc.msg}.") from exc
        if not isinstance(values, dict):
            raise ValueError("Parameter Preset values must be a JSON object.")
        preset = {
            "schema_version": 1,
            "name": str(kwargs.get("name") or "Preset"),
            "values": values,
        }
        return {"preset": preset, "manifest": json.dumps(preset, sort_keys=True)}
