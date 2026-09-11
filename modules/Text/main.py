# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import json
import math
import re

from modiff.NodeBase import NodeBase


BUILTIN_DATA_OPERATION_PIPELINE_CLASS = "BuiltinDataOperationV1"
MAX_DATA_OPERATION_TEXT_BYTES = 1_048_576
MAX_DATA_OPERATION_LINES = 100_000
MAX_DATA_OPERATION_JSON_DEPTH = 32
MAX_DATA_OPERATION_JSON_ITEMS = 100_000
MAX_SAFE_INTEGER = (1 << 53) - 1


def _bounded_text(value):
    if not isinstance(value, str):
        raise TypeError("Built-in data operations require text input.")
    if len(value.encode("utf-8")) > MAX_DATA_OPERATION_TEXT_BYTES:
        raise ValueError("Built-in data-operation text exceeds the 1 MiB UTF-8 limit.")
    return value


def _bounded_json(value):
    pending = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop()
        visited += 1
        if visited > MAX_DATA_OPERATION_JSON_ITEMS:
            raise ValueError("Converted JSON exceeds the bounded item count.")
        if depth > MAX_DATA_OPERATION_JSON_DEPTH:
            raise ValueError("Converted JSON exceeds the bounded nesting depth.")
        if item is None or isinstance(item, (str, bool, int)):
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Converted JSON may contain only finite numbers.")
            continue
        if isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
            continue
        if isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise ValueError("Converted JSON object keys must be strings.")
            pending.extend((child, depth + 1) for child in item.values())
            continue
        raise ValueError("Converted JSON contains an unsupported value type.")
    return value


def _convert_text(value, target_type):
    if target_type == "text":
        return value
    stripped = value.strip()
    if target_type == "integer":
        if not re.fullmatch(r"[+-]?(?:0|[1-9][0-9]*)", stripped):
            raise ValueError("Integer conversion requires a canonical base-10 integer.")
        converted = int(stripped)
        if abs(converted) > MAX_SAFE_INTEGER:
            raise ValueError("Integer conversion exceeds the exact 53-bit interchange range.")
        return converted
    if target_type == "float":
        try:
            converted = float(stripped)
        except ValueError as error:
            raise ValueError("Float conversion requires a finite decimal number.") from error
        if not math.isfinite(converted):
            raise ValueError("Float conversion requires a finite decimal number.")
        return converted
    if target_type == "boolean":
        normalized = stripped.lower()
        if normalized not in {"true", "false"}:
            raise ValueError("Boolean conversion accepts only true or false.")
        return normalized == "true"
    if target_type == "json":
        try:
            return _bounded_json(json.loads(value))
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError("JSON conversion requires one bounded JSON value.") from error
    raise ValueError("Unsupported built-in data conversion target.")

class TextToList(NodeBase):
    """
    Convert a text string into a list of strings
    """

    label = "Text to List"
    category = "text"
    resizable = True
    params = {
        "text": {"type": "string", "default": "", "display": "text"},
        "separator": {
            "type": "string",
            "options": {
                "\n": "\\n",
                ",": ",",
                " ": "[space]",
                "|": "|",
                ";": ";",
                ":": ":",
            },
            "default": ","
        },
        "output": {"type": "string", "display": "output"},
    }

    def execute(self, **kwargs):
        text = kwargs["text"]
        separator = kwargs["separator"]
        return [item.strip() for item in text.split(separator)]


class ProcessText(NodeBase):
    label = "Process Text/Data"
    category = "text"
    resizable = True
    params = {
        "source": {"label": "Source", "type": "string", "display": "textarea", "default": ""},
        "alternate_source": {
            "label": "Alternate Source",
            "type": "string",
            "display": "textarea",
            "default": "No alternate branch was selected.",
        },
        "pipeline_class": {
            "label": "Contract",
            "type": "string",
            "default": BUILTIN_DATA_OPERATION_PIPELINE_CLASS,
            "hidden": True,
        },
        "operation": {
            "label": "Operation",
            "type": "string",
            "options": {
                "text_select": "Select Line",
                "data_conversion": "Convert Data",
                "graph_utility": "Select Text Branch",
            },
            "default": "text_select",
        },
        "index": {"label": "Line Index", "type": "int", "default": 0, "min": -100_000, "max": 100_000},
        "selection_mode": {
            "label": "Index Behavior",
            "type": "string",
            "options": {"error": "Error", "clamp": "Clamp", "wrap": "Wrap"},
            "default": "error",
        },
        "ignore_empty_lines": {"label": "Ignore Empty Lines", "type": "bool", "default": False},
        "strip_line": {"label": "Trim Selected Line", "type": "bool", "default": True},
        "target_type": {
            "label": "Target Type",
            "type": "string",
            "options": {
                "text": "Text",
                "integer": "Integer",
                "float": "Float",
                "boolean": "Boolean",
                "json": "JSON",
            },
            "default": "json",
        },
        "condition": {"label": "Select Source", "type": "bool", "default": True},
        "output": {"label": "Output", "display": "output", "type": "any"},
        "selected_index": {"label": "Selected Index", "display": "output", "type": "int"},
        "item_count": {"label": "Item Count", "display": "output", "type": "int"},
    }

    def execute(self, **kwargs):
        return evaluate_data_operation(kwargs)


def evaluate_data_operation(kwargs):
    if kwargs.get("pipeline_class", BUILTIN_DATA_OPERATION_PIPELINE_CLASS) != BUILTIN_DATA_OPERATION_PIPELINE_CLASS:
        raise ValueError("Unsupported built-in data-operation contract.")
    source = _bounded_text(kwargs.get("source", ""))
    operation = kwargs.get("operation", "text_select")
    if operation == "data_conversion":
        return {
            "output": _convert_text(source, kwargs.get("target_type", "json")),
            "selected_index": -1,
            "item_count": 1,
        }
    if operation == "graph_utility":
        alternate_source = _bounded_text(kwargs.get("alternate_source", ""))
        condition = kwargs.get("condition", True)
        if not isinstance(condition, bool):
            raise ValueError("Text branch selection requires a boolean condition.")
        return {
            "output": source if condition else alternate_source,
            "selected_index": 0 if condition else 1,
            "item_count": 2,
        }
    if operation != "text_select":
        raise ValueError("Unsupported built-in data operation.")

    lines = source.splitlines()
    if kwargs.get("ignore_empty_lines", False):
        lines = [line for line in lines if line.strip()]
    if len(lines) > MAX_DATA_OPERATION_LINES:
        raise ValueError("Text selection exceeds the bounded line count.")
    if not lines:
        raise ValueError("Text selection requires at least one selectable line.")
    index = kwargs.get("index", 0)
    if isinstance(index, bool) or not isinstance(index, int) or not -MAX_DATA_OPERATION_LINES <= index <= MAX_DATA_OPERATION_LINES:
        raise ValueError("Line index must be a bounded integer.")
    mode = kwargs.get("selection_mode", "error")
    if mode == "wrap":
        selected_index = index % len(lines)
    elif mode == "clamp":
        selected_index = min(max(index, -len(lines)), len(lines) - 1)
        if selected_index < 0:
            selected_index += len(lines)
    elif mode == "error":
        if not -len(lines) <= index < len(lines):
            raise ValueError("Line index is outside the selected text range.")
        selected_index = index if index >= 0 else len(lines) + index
    else:
        raise ValueError("Unsupported line-selection behavior.")
    output = lines[selected_index]
    if kwargs.get("strip_line", True):
        output = output.strip()
    return {"output": output, "selected_index": selected_index, "item_count": len(lines)}
