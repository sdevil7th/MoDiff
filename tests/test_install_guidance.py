import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modiff import install
from modiff.setup_catalog import PHASES, enrich_issue


class GuidedInstallerTests(unittest.TestCase):
    def test_structured_issue_contains_help_and_safe_action_metadata(self):
        issue = enrich_issue(
            "gpu-groups-missing", "groups required", blocking=True,
            command="sudo usermod -a -G video,render test",
            action={"id": "linux-add-gpu-groups", "argv": ["usermod", "-a", "-G", "video,render", "test"], "requires_admin": True},
            requires_reboot=True,
        )
        self.assertEqual(issue["status"], "blocked")
        self.assertTrue(issue["requires_admin"])
        self.assertTrue(issue["requires_reboot"])
        self.assertTrue(issue["verification"])
        self.assertTrue(issue["failure_help"])

    def test_system_action_allowlist_rejects_modified_commands(self):
        self.assertTrue(install._action_is_allowed({"id": "ubuntu-install-amdrocm-gfx1151", "argv": ["apt", "install", "-y", "amdrocm-gfx1151"]}))
        self.assertFalse(install._action_is_allowed({"id": "ubuntu-install-amdrocm-gfx1151", "argv": ["apt", "remove", "-y", "amdrocm-gfx1151"]}))
        self.assertFalse(install._action_is_allowed({"id": "unknown", "argv": ["sh", "-c", "anything"]}))

    def test_group_detection_is_safe_without_posix_grp(self):
        with patch.object(install, "grp", None):
            self.assertEqual(install._groups(), [])

    def test_malformed_journal_recovers_to_a_new_valid_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "install-state.json"
            journal.write_text("not json", encoding="utf-8")
            with patch.object(install, "JOURNAL_PATH", journal), patch.object(install, "MANAGED_ROOT", Path(temporary)):
                self.assertEqual(install._read_journal(), {})
                written = install._write_journal(status="running", current_phase="detect")
            self.assertEqual(written["schema_version"], 1)
            self.assertEqual(json.loads(journal.read_text(encoding="utf-8"))["status"], "running")

    def test_phase_record_updates_matching_setup_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "install-state.json"
            journal.write_text(json.dumps({"steps": [{"id": "toolchain", "phase": "toolchain", "status": "pending"}]}), encoding="utf-8")
            with patch.object(install, "JOURNAL_PATH", journal), patch.object(install, "MANAGED_ROOT", Path(temporary)):
                install._record_phase("toolchain")
            state = json.loads(journal.read_text(encoding="utf-8"))
            self.assertEqual(state["steps"][0]["status"], "complete")
            self.assertEqual(state["current_phase"], "toolchain")

    def test_staged_promotion_preserves_previous_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current, staged, previous = root / ".venv", root / ".venv.next", root / ".venv.previous"
            current.mkdir(); staged.mkdir()
            (current / "marker").write_text("old", encoding="utf-8")
            (staged / "marker").write_text("new", encoding="utf-8")
            with patch.object(install, "VENV", current), patch.object(install, "STAGED_VENV", staged), patch.object(install, "PREVIOUS_VENV", previous):
                install._promote_staged_environment()
            self.assertEqual((current / "marker").read_text(encoding="utf-8"), "new")
            self.assertEqual((previous / "marker").read_text(encoding="utf-8"), "old")

    def test_backend_only_skips_node_provisioning(self):
        with patch.object(install, "_ensure_node") as ensure_node:
            result = install._install_client(backend_only=True)
        self.assertEqual(result, {"status": "skipped", "reason": "--backend-only"})
        ensure_node.assert_not_called()

    def test_missing_sibling_client_skips_node_provisioning(self):
        with patch.object(install, "_client_path", return_value=None), patch.object(install, "_ensure_node") as ensure_node:
            result = install._install_client(backend_only=False)
        self.assertEqual(result, {"status": "skipped", "reason": "sibling client not found"})
        ensure_node.assert_not_called()

    def test_client_build_provisions_node_on_demand(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = root / "MoDiff-client"
            diagnostics = root / "diagnostics"
            client.mkdir()
            toolchains = {"node": "/tools/node", "npm": "/tools/npm", "node_version": "24.12.0"}
            completed = subprocess.CompletedProcess([], 0, stdout="ok", stderr="")
            with (
                patch.object(install, "_client_path", return_value=client),
                patch.object(install, "_ensure_node", return_value=toolchains) as ensure_node,
                patch.object(install, "DIAGNOSTICS_DIR", diagnostics),
                patch.object(install.subprocess, "run", return_value=completed) as run,
            ):
                result = install._install_client(backend_only=False)
        self.assertEqual(result, {"status": "complete", "path": str(client), "node": "24.12.0"})
        ensure_node.assert_called_once_with()
        self.assertEqual([call.args[0] for call in run.call_args_list], [["/tools/npm", "ci"], ["/tools/npm", "run", "build"]])

    def test_phase_contract_is_stable(self):
        self.assertEqual(PHASES, ["detect", "plan", "system-preparation", "toolchain", "backend", "client", "validation", "complete"])


if __name__ == "__main__":
    unittest.main()
