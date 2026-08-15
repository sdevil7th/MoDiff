# Derived from cubiq/Mellon@5fd242921d13bff9fb03f4de405fdd39c2335e1f; modified by MoDiff.
import json
import os
import tempfile
from pathlib import Path

from PIL import Image

from modiff.NodeBase import NodeBase
from modiff.config import CONFIG
from utils.paths import parse_filename


MAX_DATA_EXPORT_BYTES = 1_048_576

def serialize_exif_value(data, max_length=100):
    """Recursively serialize EXIF data values to be JSON compatible"""
    if isinstance(data, bytes):
        # Convert bytes to hex string for readability
        return f"{data.hex()[:50]}{'...' if len(data) > 50 else ''}"
    elif isinstance(data, (int, float, str, bool, type(None))):
        # These types are JSON serializable
        return data
    elif isinstance(data, dict):
        # Recursively handle dictionaries
        return {str(k): serialize_exif_value(v, max_length) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        # Recursively handle lists and tuples
        return [serialize_exif_value(item, max_length) for item in data]
    else:
        # For other types, convert to string representation
        str_repr = str(data)
        if len(str_repr) > max_length:
            str_repr = str_repr[:max_length] + "..."
        return f"{str_repr}"

class DataViewer(NodeBase):
    label = "Data Viewer"
    category = "primitive"
    resizable = True
    params = {
        "value": {
            "label": "Data",
            "display": "input",
            "type": "any",
        },
        "preview": {
            "label": "Preview",
            "display": "ui_text",
            "dataSource": "output",
        },
        "output": {
            "label": "Output",
            "display": "output",
            "type": "string",
        }
    }

    def execute(self, **kwargs):
        from PIL.ExifTags import TAGS

        value = kwargs["value"]
        if isinstance(value, dict) or isinstance(value, list):
            value = json.dumps(value, indent=2)
        elif isinstance(value, Image.Image):
            # Extract EXIF data
            exif_data = {}
            if hasattr(value, '_getexif') and value._getexif() is not None:
                exif = value._getexif()
                for tag_id in exif:
                    tag = TAGS.get(tag_id, tag_id)
                    data = exif.get(tag_id)

                    # Use recursive serialization for all data types
                    exif_data[tag] = serialize_exif_value(data)

            value = json.dumps({
                "width": value.width,
                "height": value.height,
                "format": value.format,
                "mode": value.mode,
                "size": value.size,
                "filename": value.filename,
                "exif": exif_data
            }, indent=2)

        return {"output": str(value)}


class ExportData(NodeBase):
    """Persist one bounded JSON or text value under the app-owned data root."""

    label = "Export Data"
    category = "primitive"
    resizable = True
    params = {
        "value": {"label": "Data", "display": "input", "type": "any"},
        "filename": {
            "label": "File",
            "type": "str",
            "default": "{PATH:data}/exports/MoDiff_{HASH:6}.json",
        },
        "format": {
            "label": "Format",
            "type": "string",
            "options": ["json", "text"],
            "default": "json",
        },
        "file": {"label": "File", "display": "output", "type": "str"},
        "output": {"label": "Serialized Data", "display": "output", "type": "string"},
        "preview": {"label": "Preview", "display": "ui_text", "dataSource": "output"},
    }

    def execute(self, **kwargs):
        format_name = str(kwargs.get("format") or "json")
        if format_name not in {"json", "text"}:
            raise ValueError("Export Data format must be json or text.")

        value = kwargs.get("value")
        if format_name == "json":
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            try:
                body = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise ValueError("Export Data JSON values must be finite and serializable.") from error
        elif isinstance(value, str):
            body = value
        else:
            try:
                body = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise ValueError("Export Data text values must be finite and serializable.") from error

        encoded = (body + "\n").encode("utf-8")
        if len(encoded) > MAX_DATA_EXPORT_BYTES:
            raise ValueError(f"Export Data output exceeds the {MAX_DATA_EXPORT_BYTES}-byte limit.")

        default_filename = "{PATH:data}/exports/MoDiff_{HASH:6}.json"
        destination = Path(parse_filename(kwargs.get("filename") or default_filename))
        data_root = Path(CONFIG.paths["data"]).resolve()
        if not destination.is_absolute():
            destination = data_root / destination
        destination = destination.resolve(strict=False)
        try:
            destination.relative_to(data_root)
        except ValueError as error:
            raise ValueError("Export Data files must stay within the app data directory.") from error
        expected_suffix = ".json" if format_name == "json" else ".txt"
        if destination.suffix.lower() != expected_suffix:
            raise ValueError(f"Export Data {format_name} files must use the {expected_suffix} extension.")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return {"file": str(destination), "output": body}

class TextValue(NodeBase):
    label = "Text Value"
    category = "primitive"
    resizable = True
    params = {
        "text": {
            "label": "Text",
            "display": "text",
            "type": "string",
        },
        "output": {
            "label": "Output",
            "display": "output",
            "type": "string",
        }
    }

    def execute(self, **kwargs):
        return {"output": kwargs.get("text", "")}

class ToList(NodeBase):
    label = "Items to List"
    category = "primitive"
    params = {
        "item": { "type": "any", "display": "input", "spawn": True },
        "list": { "type": "any", "display": "output" },
    }

    def execute(self, **kwargs):
        return {"list": kwargs.get("item", [])}
