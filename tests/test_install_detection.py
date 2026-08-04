import unittest
from unittest.mock import patch

from modiff.install import _amd_qualification, resolve_profile
from modiff.runtime_profile import load_manifest


class InstallDetectionTests(unittest.TestCase):
    def host(self, **changes):
        return {"os": "linux", "architecture": "x86_64", "wsl": False, "nvidia_usable": False, "amd_usable": False, "mps_candidate": False, **changes}

    def test_auto_profiles(self):
        self.assertEqual(resolve_profile("auto", self.host(nvidia_usable=True)), "nvidia-cuda")
        self.assertEqual(resolve_profile("auto", self.host(amd_usable=True)), "amd-rocm-linux")
        self.assertEqual(resolve_profile("auto", self.host()), "cpu")

    def test_hybrid_requires_selection(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            resolve_profile("auto", self.host(nvidia_usable=True, amd_usable=True))

    def test_wsl_is_not_guessed(self):
        self.assertEqual(resolve_profile("auto", self.host(wsl=True, nvidia_usable=True)), "cpu")

    def test_noninteractive_experimental_auto_uses_cpu_without_opt_in(self):
        host = self.host(amd_usable=True, os_version="26.04")
        self.assertEqual(resolve_profile("auto", host, non_interactive=True), "cpu")
        self.assertEqual(resolve_profile("auto", host, non_interactive=True, allow_experimental=True), "amd-rocm-linux")

    def test_qualified_strix_halo_host_is_supported(self):
        host = self.host(
            os_id="ubuntu", os_version="24.04.3", kernel="6.14.1018", amd_candidate=True,
            amd_architectures=["gfx1151"], kfd_present=True, kfd_accessible=True,
            render_nodes=["/dev/dri/renderD128"], groups=["video", "render"], rocminfo_returncode=0,
        )
        libraries = " ".join((
            "libamdhip64.so.7 libMIOpen.so.1 libhipblas.so.3 libhipblaslt.so.1 libhipfft.so.0 "
            "libhiprand.so.1 libhiprtc.so.7 libhipsolver.so.1 libhipsparse.so.4 libhipsparselt.so.0 "
            "librccl.so.1 librocblas.so.5 librocsolver.so.0 libroctracer64.so.4 libroctx64.so.4"
        ).split())
        with patch("modiff.install._command", return_value={"returncode": 0, "stdout": libraries, "stderr": ""}):
            tier, issues = _amd_qualification(host, load_manifest()["profiles"]["amd-rocm-linux"])
        self.assertEqual(tier, "supported")
        self.assertEqual(issues, [])

    def test_missing_permissions_are_guided_not_mutated(self):
        host = self.host(
            os_id="ubuntu", os_version="26.04", kernel="7.0.0", amd_candidate=True,
            amd_architectures=[], kfd_present=False, kfd_accessible=False,
            render_nodes=[], groups=[], rocminfo_returncode=1, rocminfo_error="permission denied",
        )
        with patch("modiff.install._command", return_value={"returncode": 0, "stdout": "", "stderr": ""}):
            tier, issues = _amd_qualification(host, load_manifest()["profiles"]["amd-rocm-linux"])
        self.assertEqual(tier, "experimental")
        group_issue = next(issue for issue in issues if issue["code"] == "gpu-groups-missing")
        self.assertIn("sudo usermod", group_issue["guided_command"])
