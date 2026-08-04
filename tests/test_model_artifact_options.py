import unittest

from modules.ModelArtifact.main import QuantizeDiffusersComponents


class ModelArtifactOptionTests(unittest.TestCase):
    def test_quantization_node_does_not_select_an_optional_backend_that_is_not_installed(self):
        self.assertEqual(QuantizeDiffusersComponents.params["quantization_mode"]["default"], "")

        node = QuantizeDiffusersComponents("missing-quantization-selection")
        with self.assertRaisesRegex(ValueError, "Select an installed quantization backend"):
            node.execute(model_id={"source": "hub", "value": "org/example"})


if __name__ == "__main__":
    unittest.main()
