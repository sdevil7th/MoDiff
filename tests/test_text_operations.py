import unittest

from modules.Text.main import (
    BUILTIN_DATA_OPERATION_PIPELINE_CLASS,
    MAX_DATA_OPERATION_TEXT_BYTES,
    ProcessText,
)


class BuiltinDataOperationTests(unittest.TestCase):
    def setUp(self):
        self.node = ProcessText()

    def execute(self, **kwargs):
        values = {
            "source": "first\n second \nthird",
            "pipeline_class": BUILTIN_DATA_OPERATION_PIPELINE_CLASS,
            "operation": "text_select",
            "index": 0,
            "selection_mode": "error",
            "ignore_empty_lines": False,
            "strip_line": True,
            "target_type": "json",
        }
        values.update(kwargs)
        return self.node.execute(**values)

    def test_selects_lines_with_exact_error_clamp_wrap_and_negative_index_semantics(self):
        self.assertEqual(self.execute()["output"], "first")
        self.assertEqual(self.execute(index=-1)["output"], "third")
        self.assertEqual(self.execute(index=9, selection_mode="wrap")["selected_index"], 0)
        self.assertEqual(self.execute(index=9, selection_mode="clamp")["selected_index"], 2)
        self.assertEqual(self.execute(index=-9, selection_mode="clamp")["selected_index"], 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.execute(index=3)

    def test_empty_line_filtering_and_trimming_are_explicit(self):
        result = self.execute(source=" first \n\n second ", index=1, ignore_empty_lines=True)
        self.assertEqual(result, {"output": "second", "selected_index": 1, "item_count": 2})
        self.assertEqual(self.execute(index=1, strip_line=False)["output"], " second ")

    def test_converts_only_reviewed_finite_interchange_types(self):
        cases = {
            "text": (" hello ", " hello "),
            "integer": ("-42", -42),
            "float": ("1.25", 1.25),
            "boolean": ("false", False),
            "json": ('{"items":[1,true,null]}', {"items": [1, True, None]}),
        }
        for target_type, (source, expected) in cases.items():
            with self.subTest(target_type=target_type):
                result = self.execute(operation="data_conversion", source=source, target_type=target_type)
                self.assertEqual(result, {"output": expected, "selected_index": -1, "item_count": 1})

    def test_rejects_ambiguous_or_unbounded_conversions_before_output(self):
        invalid = (
            ("integer", "01", "canonical"),
            ("integer", str(1 << 53), "53-bit"),
            ("float", "nan", "finite"),
            ("boolean", "yes", "true or false"),
            ("json", "{broken", "bounded JSON"),
        )
        for target_type, source, message in invalid:
            with self.subTest(target_type=target_type, source=source):
                with self.assertRaisesRegex(ValueError, message):
                    self.execute(operation="data_conversion", source=source, target_type=target_type)
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            self.execute(source="x" * (MAX_DATA_OPERATION_TEXT_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "Unsupported built-in data-operation contract"):
            self.execute(pipeline_class="MutableContract")


if __name__ == "__main__":
    unittest.main()
