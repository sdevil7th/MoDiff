import unittest

from modiff.runtime_profile import load_manifest


class AcceleratorManifestTests(unittest.TestCase):
    def test_all_public_profiles_are_release_pinned(self):
        manifest = load_manifest()
        self.assertEqual(
            set(manifest["profiles"]),
            {"nvidia-cuda", "amd-rocm-linux", "amd-pytorch-windows", "intel-xpu", "apple-mps", "cpu"},
        )
        for name, profile in manifest["profiles"].items():
            with self.subTest(name=name):
                self.assertRegex(profile["torch"], r"^\d+\.\d+\.\d+(?:\+rocm\d+\.\d+(?:\.\d+)?)?$")
                self.assertTrue(profile["requirements"])
                self.assertTrue(profile["sources"])
                self.assertIn(profile["tier"], {"supported", "preview", "conditional", "unsupported"})
                if name == "amd-rocm-linux":
                    self.assertEqual(set(profile["wheel_hashes"]), {"torch", "torchvision", "torchaudio", "triton"})
                    self.assertTrue(all(len(value) == 64 for value in profile["wheel_hashes"].values()))
