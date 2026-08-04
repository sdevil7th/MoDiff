from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.lock_accelerator_wheels import parse_direct_wheel_requirements


class DirectWheelRequirementTests(unittest.TestCase):
    def _requirements(self, contents: str) -> Path:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "profile.txt"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_extracts_urls_without_existing_hashes(self):
        path = self._requirements(
            "# reviewed wheels\n"
            "https://example.test/torch.whl --hash=sha256:old\n"
            "https://example.test/vision.whl\n"
            "-e .[accelerator]\n"
        )

        urls, editable = parse_direct_wheel_requirements(path)

        self.assertEqual(
            urls,
            [
                "https://example.test/torch.whl",
                "https://example.test/vision.whl",
            ],
        )
        self.assertEqual(editable, "-e .[accelerator]")

    def test_rejects_index_based_profile_before_locking(self):
        path = self._requirements(
            "torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu\n-e .\n"
        )

        with self.assertRaisesRegex(ValueError, "not direct-wheel-only"):
            parse_direct_wheel_requirements(path)

    def test_rejects_profile_without_wheels(self):
        path = self._requirements("# unsupported placeholder\n-e .\n")

        with self.assertRaisesRegex(ValueError, "no direct HTTPS wheel URLs"):
            parse_direct_wheel_requirements(path)


if __name__ == "__main__":
    unittest.main()
