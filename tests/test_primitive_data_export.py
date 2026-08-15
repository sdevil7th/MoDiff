import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modiff.config import CONFIG
from modules.Primitive.main import ExportData, MAX_DATA_EXPORT_BYTES


class PrimitiveDataExportTests(unittest.TestCase):
    def test_json_export_is_bounded_deterministic_and_app_owned(self):
        with tempfile.TemporaryDirectory(prefix="modiff-data-export-") as temporary:
            data_root = Path(temporary) / "data"
            data_root.mkdir()
            output = data_root / "review" / "result.json"
            with patch.dict(CONFIG.paths, {"data": str(data_root)}):
                result = ExportData("data-export-json")(
                    value='{"z":2,"approved":false,"tags":["a","b"]}',
                    filename=str(output),
                    format="json",
                )

            self.assertEqual(result["file"], str(output))
            self.assertEqual(
                result["output"],
                json.dumps(
                    {"z": 2, "approved": False, "tags": ["a", "b"]},
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                ),
            )
            self.assertEqual(output.read_text(encoding="utf-8"), result["output"] + "\n")

    def test_text_and_json_string_outputs_remain_truthful(self):
        with tempfile.TemporaryDirectory(prefix="modiff-data-export-") as temporary:
            data_root = Path(temporary) / "data"
            data_root.mkdir()
            with patch.dict(CONFIG.paths, {"data": str(data_root)}):
                text = ExportData("data-export-text")(
                    value="selected branch",
                    filename=str(data_root / "selected.txt"),
                    format="text",
                )
                json_string = ExportData("data-export-json-string")(
                    value="selected branch",
                    filename=str(data_root / "selected.json"),
                    format="json",
                )

            self.assertEqual(text["output"], "selected branch")
            self.assertEqual(json.loads((data_root / "selected.json").read_text(encoding="utf-8")), "selected branch")

    def test_export_rejects_unsafe_unbounded_or_mismatched_outputs(self):
        with tempfile.TemporaryDirectory(prefix="modiff-data-export-") as temporary:
            data_root = Path(temporary) / "data"
            data_root.mkdir()
            with patch.dict(CONFIG.paths, {"data": str(data_root)}):
                node = ExportData()
                with self.assertRaisesRegex(ValueError, "within the app data"):
                    node(value={}, filename=str(Path(temporary) / "outside.json"), format="json")
                with self.assertRaisesRegex(ValueError, "must use the .json extension"):
                    node(value={}, filename=str(data_root / "wrong.txt"), format="json")
                with self.assertRaisesRegex(ValueError, "finite and serializable"):
                    node(value={"invalid": float("nan")}, filename=str(data_root / "nan.json"), format="json")
                with self.assertRaisesRegex(ValueError, "byte limit"):
                    node(
                        value="x" * MAX_DATA_EXPORT_BYTES,
                        filename=str(data_root / "large.txt"),
                        format="text",
                    )


if __name__ == "__main__":
    unittest.main()
