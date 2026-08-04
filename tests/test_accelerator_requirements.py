import re
import unittest
from pathlib import Path

from modiff.runtime_profile import load_manifest


ROOT = Path(__file__).parents[1]
INDEX_OPTION = re.compile(r"(?:^|\s)--(?:extra-)?index-url(?:\s|=)")


class AcceleratorRequirementTests(unittest.TestCase):
    def test_package_index_options_are_standalone_and_match_the_manifest(self):
        manifest = load_manifest()
        indexed_profiles = {"nvidia-cuda", "intel-xpu", "cpu"}

        for name, profile in manifest["profiles"].items():
            with self.subTest(profile=name):
                requirements = (ROOT / profile["requirements"]).read_text(encoding="utf-8").splitlines()
                active_lines = [line.strip() for line in requirements if line.strip() and not line.lstrip().startswith("#")]

                for line in active_lines:
                    if INDEX_OPTION.search(line):
                        self.assertRegex(
                            line,
                            r"^--(?:extra-)?index-url(?:\s|=)",
                            "Package index options must be standalone requirements-file options",
                        )

                if name in indexed_profiles:
                    self.assertIn(f"--extra-index-url {profile['index']}", active_lines)


if __name__ == "__main__":
    unittest.main()
