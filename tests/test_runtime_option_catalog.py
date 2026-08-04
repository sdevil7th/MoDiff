import unittest
from unittest.mock import patch

import modules
from modiff.server import WebServer


class RuntimeOptionCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = WebServer(modules=modules.MODULE_MAP, work_dir=".", data_dir="data")

    def test_every_registered_option_has_a_descriptor_and_declared_default(self):
        malformed = []
        invalid_defaults = []
        # Catalog integrity must not depend on the CI host's accelerator or
        # optional packages. Runtime compatibility is covered separately by
        # test_runtime_status.py.
        with patch.object(self.server, "_runtime_choice_compatibility", return_value=(True, None)):
            for module, actions in self.server.modules.items():
                for action, values in actions.items():
                    params = self.server.describe_node_params(values.get("params", {}))
                    for field_name, field in params.items():
                        options = field.get("options") if isinstance(field, dict) else None
                        if not isinstance(options, (list, dict)):
                            continue
                        entries = list(options.values()) if isinstance(options, dict) else options
                        descriptors = [
                            entry
                            for entry in entries
                            if isinstance(entry, dict) and entry.get("schemaVersion") == 1
                        ]
                        for descriptor in descriptors:
                            if descriptor.get("value") == "[object Object]" or str(
                                descriptor.get("label", "")
                            ).startswith("['"):
                                malformed.append(f"{module}.{action}.{field_name}")

                        default = field.get("default")
                        defaults = default if isinstance(default, list) else [default]
                        for item in defaults:
                            if item in (None, "") or not descriptors:
                                continue
                            selected = next(
                                (descriptor for descriptor in descriptors if descriptor.get("value") == str(item)),
                                None,
                            )
                            if selected is None:
                                invalid_defaults.append(f"{module}.{action}.{field_name}={item}")

        self.assertEqual(malformed, [])
        self.assertEqual(invalid_defaults, [])


if __name__ == "__main__":
    unittest.main()
