import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from modiff import install
from modiff.setup_catalog import PHASES, enrich_issue


class GuidedInstallerTests(unittest.TestCase):
    def _fake_torch(
        self,
        *,
        cuda=None,
        hip=None,
        mps_built=False,
        mps_available=False,
        execution_device="cpu:0",
        allow_mps_calls=False,
    ):
        module = types.ModuleType("torch")
        module.__version__ = "2.8.0"
        module.version = types.SimpleNamespace(cuda=cuda, hip=hip)
        module.float16 = object()
        module.float32 = object()

        class Tensor:
            def __mul__(self, _value):
                return self

            def __add__(self, _value):
                return self

            def cpu(self):
                return self

            def float(self):
                return self

            def tolist(self):
                return [3.0, 5.0]

        def tensor(_values, *, device, dtype):
            self.assertEqual(device, execution_device)
            self.assertIs(dtype, module.float32 if execution_device == "cpu:0" else module.float16)
            return Tensor()

        def unexpected_mps_call():
            self.fail("The CPU profile must not execute an MPS operation")

        module.tensor = tensor
        module.cuda = types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None, empty_cache=lambda: None)
        module.xpu = types.SimpleNamespace(is_available=lambda: False, synchronize=lambda: None, empty_cache=lambda: None)
        module.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_built=lambda: mps_built, is_available=lambda: mps_available)
        )
        mps_call = (lambda: None) if allow_mps_calls else unexpected_mps_call
        module.mps = types.SimpleNamespace(synchronize=mps_call, empty_cache=mps_call)
        return module

    def test_cpu_smoke_uses_cpu_when_the_pytorch_wheel_also_contains_mps(self):
        fake_torch = self._fake_torch(mps_built=True, mps_available=True)
        output = io.StringIO()

        with patch.dict(sys.modules, {"torch": fake_torch}), redirect_stdout(output):
            exec(install._smoke_script("cpu"), {})

        result = json.loads(output.getvalue())
        self.assertEqual(result["backend"], "cpu")
        self.assertEqual(result["device"], "cpu:0")

    def test_cpu_smoke_still_rejects_a_cuda_pytorch_wheel(self):
        fake_torch = self._fake_torch(cuda="12.8")

        with patch.dict(sys.modules, {"torch": fake_torch}), self.assertRaisesRegex(AssertionError, "cuda.*cpu"):
            exec(install._smoke_script("cpu"), {})

    def test_apple_mps_smoke_still_executes_on_mps(self):
        fake_torch = self._fake_torch(
            mps_built=True,
            mps_available=True,
            execution_device="mps:0",
            allow_mps_calls=True,
        )
        output = io.StringIO()

        with patch.dict(sys.modules, {"torch": fake_torch}), redirect_stdout(output):
            exec(install._smoke_script("apple-mps"), {})

        result = json.loads(output.getvalue())
        self.assertEqual(result["backend"], "mps")
        self.assertEqual(result["device"], "mps:0")

    def test_plan_render_is_safe_for_legacy_windows_console_encodings(self):
        host = {
            "os": "windows",
            "architecture": "x86_64",
            "candidates": ["nvidia"],
            "nvidia_usable": True,
            "amd_candidate": False,
            "intel_xpu_candidate": False,
            "mps_candidate": False,
            "wsl": False,
        }
        args = install.parser().parse_args(["--accelerator", "auto"])
        with patch.object(install, "detect_host", return_value=host):
            plan = install.build_plan(args)

        stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")
        with redirect_stdout(stream):
            install._render_plan(plan)
        stream.flush()

    def test_readme_documents_the_managed_cross_platform_install_contract(self):
        readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

        required_commands = (
            "git clone https://github.com/sdevil7th/MoDiff-client.git MoDiff-client",
            "./install.sh --accelerator auto",
            r".\install.ps1 -Accelerator auto",
            "./install.sh --accelerator auto --system-check",
            r".\install.ps1 -Accelerator auto -SystemCheck",
            "./install.sh --accelerator auto --resume",
            r".\install.ps1 -Accelerator auto -Resume",
            "./install.sh --accelerator auto --repair",
            r".\install.ps1 -Accelerator auto -Repair",
            "./install.sh --accelerator cpu --backend-only",
            r".\install.ps1 -Accelerator cpu -BackendOnly",
            "./run.sh",
            r".\run.ps1",
            "curl --fail http://127.0.0.1:8088/health",
        )
        for command in required_commands:
            with self.subTest(command=command):
                self.assertIn(command, readme)

        command_blocks = re.findall(r"```(?:bash|powershell|sh)?\n(.*?)```", readme, flags=re.DOTALL)
        unsupported = re.compile(r"(?m)^\s*uv\s+(?:sync|run)\b")
        self.assertFalse(
            any(unsupported.search(block) for block in command_blocks),
            "README command blocks must use the managed installer instead of uv sync/uv run",
        )

    def test_archive_member_destination_rejects_escaping_paths(self):
        unsafe_names = (
            "../escape",
            "nested/../../escape",
            "/absolute/path",
            r"C:\absolute\path",
            r"nested\..\escape",
        )
        with tempfile.TemporaryDirectory() as temporary:
            for member_name in unsafe_names:
                with self.subTest(member_name=member_name):
                    with self.assertRaisesRegex(RuntimeError, "Unsafe|escapes"):
                        install._archive_member_destination(Path(temporary), member_name)

    def test_archive_member_destination_accepts_nested_relative_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                install._archive_member_destination(root, "tool/bin/executable"),
                (root / "tool" / "bin" / "executable").resolve(),
            )

    def test_executable_project_dependency_pins_the_reviewed_diffusers_commit(self):
        project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        diffusers = next(item for item in project["project"]["dependencies"] if item.startswith("diffusers"))
        self.assertEqual(
            diffusers,
            "diffusers @ git+https://github.com/huggingface/diffusers.git@bb56997d4b7e87f0743f26a612f49ec4e7ce7213",
        )
        self.assertNotIn("diffusers", project["tool"]["uv"].get("sources", {}))

    def test_direct_hugging_face_hub_import_has_compatible_declared_dependency(self):
        project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = project["project"]["dependencies"]

        self.assertIn("huggingface-hub>=1.23.0,<2.0", dependencies)

    def test_opencv_is_optional_at_runtime_but_available_to_media_tests(self):
        root = Path(__file__).parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertFalse(any(item.startswith("opencv-python") for item in project["project"]["dependencies"]))
        self.assertEqual(
            project["project"]["optional-dependencies"]["gallery-media"],
            ["opencv-python-headless>=4.11.0"],
        )
        self.assertIn("opencv-python-headless>=4.11.0", (root / "requirements/test.txt").read_text(encoding="utf-8"))

    def test_every_managed_profile_installs_project_dependencies(self):
        root = Path(__file__).parents[1]
        manifest = install.load_manifest()

        for profile, specification in manifest["profiles"].items():
            with self.subTest(profile=profile):
                requirements = (root / specification["requirements"]).read_text(encoding="utf-8")
                self.assertRegex(requirements, r"(?m)^-e \.(?:\[[^]]+\])?$")

    def test_uv_project_commands_cannot_replace_the_managed_runtime(self):
        project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertIs(project["tool"]["uv"]["managed"], False)

    def test_all_launchers_validate_the_managed_profile_before_starting(self):
        root = Path(__file__).parents[1]
        linux_launcher = (root / "run.sh").read_text(encoding="utf-8")
        runtime_wrapper = (root / "scripts/with-runtime-env.sh").read_text(encoding="utf-8")
        windows_launcher = (root / "run.ps1").read_text(encoding="utf-8")

        self.assertIn("modiff.preflight --fail-on-error", linux_launcher)
        self.assertIn("scripts/with-runtime-env.sh", linux_launcher)
        self.assertNotIn("ROCM_LIBRARY_PATHS", linux_launcher)
        self.assertIn('RUNTIME_PROFILE" == "amd-rocm-linux"', runtime_wrapper)
        self.assertIn("/opt/rocm/core-*/lib", runtime_wrapper)
        self.assertIn('LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH', runtime_wrapper)
        bash = shutil.which("bash")
        bash_usable = False
        if bash:
            bash_usable = subprocess.run(
                [bash, "--version"],
                capture_output=True,
                check=False,
            ).returncode == 0
        if bash_usable:
            subprocess.run(
                [bash, "-n", str(root / "run.sh"), str(root / "scripts/with-runtime-env.sh")],
                check=True,
            )
        elif os.name != "nt":
            self.fail("A usable bash executable is required to validate the POSIX launchers.")
        self.assertIn("$preflightCode =", windows_launcher)
        self.assertIn("['execution_ready']", windows_launcher)
        self.assertNotIn("uv run", linux_launcher)
        self.assertNotRegex(linux_launcher, r"exec python(?:3)? main\.py")
        self.assertIn("Run ./install.sh before starting", linux_launcher)

    @unittest.skipIf(os.name == "nt", "POSIX executable bits are not available on Windows")
    def test_documented_posix_entrypoints_are_executable(self):
        root = Path(__file__).parents[1]

        for relative_path in ("install.sh", "run.sh", "scripts/with-runtime-env.sh"):
            with self.subTest(path=relative_path):
                self.assertTrue(
                    os.access(root / relative_path, os.X_OK),
                    f"{relative_path} must be executable because documentation invokes it directly",
                )

    def test_macos_optional_runtime_qualifier_is_manual_and_fail_closed(self):
        root = Path(__file__).parents[1]
        workflow = (root / ".github/workflows/qualify-optional-runtime-macos.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("workflow_dispatch:", workflow)
        self.assertNotIn("pull_request:", workflow)
        self.assertNotIn("push:", workflow)
        self.assertIn("runs-on: macos-14", workflow)
        self.assertIn('test "$(uname -m)" = "arm64"', workflow)
        self.assertIn('test "$(git diff --name-only)" = "pyproject.toml"', workflow)
        self.assertEqual(workflow.count('-  "peft>=0.17.0;'), 2)
        self.assertEqual(workflow.count('-  "transformers>=4.49.0;'), 2)
        self.assertIn("scripts/qualify_optional_runtime.py --preflight-only", workflow)
        self.assertIn("scripts/qualify_optional_runtime.py --consent", workflow)
        self.assertIn('assert value["status"] == "ready"', workflow)
        self.assertIn('assert value["status"] == "passed"', workflow)
        self.assertIn("prospective-base.diff", workflow)
        self.assertIn("actions/upload-artifact@bbbca2ddaa5d8feaa63e36b76fdaad77386f024f", workflow)

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

    def test_amd_windows_plan_fails_before_installing_an_unreviewed_runtime(self):
        host = {
            "os": "windows",
            "architecture": "x86_64",
            "candidates": ["amd"],
            "amd_candidate": True,
            "amd_usable": False,
        }
        args = install.parser().parse_args(["--accelerator", "amd", "--non-interactive"])
        with patch.object(install, "detect_host", return_value=host):
            plan = install.build_plan(args)

        self.assertEqual(plan["profile"], "amd-pytorch-windows")
        self.assertEqual(plan["support_tier"], "conditional")
        self.assertFalse(plan["execution_ready"])
        self.assertIn("amd-windows-install-review-required", [issue["code"] for issue in plan["issues"]])

    def test_intel_integrated_graphics_selects_the_managed_xpu_preview_profile(self):
        host = {
            "os": "windows",
            "architecture": "x86_64",
            "candidates": ["intel"],
            "nvidia_usable": False,
            "amd_candidate": False,
            "intel_xpu_candidate": True,
            "mps_candidate": False,
            "wsl": False,
        }
        args = install.parser().parse_args(["--accelerator", "auto", "--non-interactive"])
        with patch.object(install, "detect_host", return_value=host):
            plan = install.build_plan(args)

        self.assertEqual(plan["profile"], "intel-xpu")
        self.assertEqual(plan["support_tier"], "preview")
        self.assertTrue(plan["requirements_exist"])
        self.assertTrue(plan["execution_ready"])

    def test_windows_plan_reports_powershell_fallback_and_resume_commands(self):
        host = {
            "os": "windows",
            "architecture": "x86_64",
            "candidates": ["nvidia"],
            "nvidia_usable": True,
            "amd_candidate": False,
            "intel_xpu_candidate": False,
            "mps_candidate": False,
            "wsl": False,
        }
        args = install.parser().parse_args(["--accelerator", "auto"])
        with patch.object(install, "detect_host", return_value=host):
            plan = install.build_plan(args)

        self.assertEqual(
            plan["cpu_fallback_command"], r".\install.ps1 -Accelerator cpu"
        )
        self.assertEqual(
            plan["resume_command"], r".\install.ps1 -Accelerator auto -Resume"
        )

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

    def test_successful_journal_update_clears_stale_failure_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            journal = root / "install-state.json"
            journal.write_text(
                json.dumps({"status": "failed", "failure": "old failure"}),
                encoding="utf-8",
            )
            with (
                patch.object(install, "JOURNAL_PATH", journal),
                patch.object(install, "MANAGED_ROOT", root),
            ):
                written = install._write_journal(status="complete", current_phase="complete")

            self.assertNotIn("failure", written)
            self.assertNotIn("failure", json.loads(journal.read_text(encoding="utf-8")))
            self.assertFalse(journal.with_suffix(".json.tmp").exists())

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

    def test_missing_sibling_client_explains_required_layout(self):
        with patch.object(install, "_client_path", return_value=None), patch.object(install, "_ensure_node") as ensure_node:
            with self.assertRaisesRegex(RuntimeError, "Sibling MoDiff-client checkout not found"):
                install._install_client(backend_only=False)
        ensure_node.assert_not_called()

    def test_client_build_provisions_node_on_demand(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = root / "MoDiff-client"
            diagnostics = root / "diagnostics"
            web = root / "web"
            (client / "dist").mkdir(parents=True)
            (client / "dist" / "index.html").write_text("new client", encoding="utf-8")
            source = client / "src" / "studio" / "templateAssetSource.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({"mode": "local"}), encoding="utf-8")
            asset_script = client / "scripts" / "template-gallery-assets.py"
            asset_script.parent.mkdir(parents=True)
            asset_script.write_text("# asset installer\n", encoding="utf-8")
            python = root / "python"
            python.write_text("", encoding="utf-8")
            (web / "user").mkdir(parents=True)
            (web / "user" / "keep.txt").write_text("user asset", encoding="utf-8")
            (web / "stale.js").write_text("stale", encoding="utf-8")
            toolchains = {"node": "/tools/node", "npm": "/tools/npm", "node_version": "24.12.0"}
            completed = subprocess.CompletedProcess([], 0, stdout="ok", stderr="")
            with (
                patch.object(install, "_client_path", return_value=client),
                patch.object(install, "_ensure_node", return_value=toolchains) as ensure_node,
                patch.object(install, "DIAGNOSTICS_DIR", diagnostics),
                patch.object(install, "WEB_ROOT", web),
                patch.object(install.subprocess, "run", return_value=completed) as run,
            ):
                result = install._install_client(backend_only=False, python=python)
            self.assertEqual(
                result,
                {
                    "status": "complete",
                    "path": str(client),
                    "node": "24.12.0",
                    "web": str(web.resolve()),
                    "template_gallery": {"source": "local", "asset_mode": "local"},
                },
            )
            self.assertEqual((web / "index.html").read_text(encoding="utf-8"), "new client")
            self.assertEqual((web / "user" / "keep.txt").read_text(encoding="utf-8"), "user asset")
            self.assertFalse((web / "stale.js").exists())
            ensure_node.assert_called_once_with()
            npm = str(Path(toolchains["npm"]))
            self.assertEqual(
                [call.args[0] for call in run.call_args_list],
                [
                    [npm, "ci"],
                    [str(python), str(asset_script), "verify"],
                    [npm, "run", "build"],
                ],
            )

    def test_remote_gallery_is_downloaded_and_bundled_without_changing_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = root / "client"
            web = root / "web"
            diagnostics = root / "diagnostics"
            managed = root / ".modiff"
            gallery = client / "public" / "template-gallery"
            gallery.mkdir(parents=True)
            (gallery / "checkout-marker.txt").write_text("original", encoding="utf-8")
            source = client / "src" / "studio" / "templateAssetSource.json"
            source.parent.mkdir(parents=True)
            source.write_text(
                json.dumps(
                    {
                        "mode": "huggingface",
                        "repoId": "modiff-project/template-gallery",
                        "revision": "0" * 40,
                        "assetSetId": "sha256:canonical-json:" + "a" * 64,
                    }
                ),
                encoding="utf-8",
            )
            asset_script = client / "scripts" / "template-gallery-assets.py"
            asset_script.parent.mkdir(parents=True)
            asset_script.write_text("# asset installer\n", encoding="utf-8")
            python = root / "python"
            python.write_text("", encoding="utf-8")
            toolchains = {
                "node": "/tools/node",
                "npm": "/tools/npm",
                "node_version": "24.12.0",
            }

            def run_step(command, **kwargs):
                if "download" in command:
                    destination = Path(command[command.index("--destination") + 1])
                    downloaded = destination / "template-gallery"
                    downloaded.mkdir(parents=True)
                    (downloaded / "installed.webp").write_bytes(b"installed")
                elif command[-2:] == ["run", "build"]:
                    self.assertEqual(
                        kwargs["env"]["VITE_MODIFF_TEMPLATE_ASSET_MODE"], "local"
                    )
                    self.assertEqual(
                        (gallery / "installed.webp").read_bytes(), b"installed"
                    )
                    self.assertEqual(
                        (client / ".template-gallery.install-backup" / "checkout-marker.txt").read_text(
                            encoding="utf-8"
                        ),
                        "original",
                    )
                    self.assertFalse(
                        (client / "public" / ".template-gallery.install-backup").exists()
                    )
                    dist_gallery = client / "dist" / "template-gallery"
                    dist_gallery.mkdir(parents=True)
                    (client / "dist" / "index.html").write_text(
                        "client", encoding="utf-8"
                    )
                    (dist_gallery / "installed.webp").write_bytes(b"installed")
                return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

            with (
                patch.object(install, "_client_path", return_value=client),
                patch.object(install, "_ensure_node", return_value=toolchains),
                patch.object(install, "DIAGNOSTICS_DIR", diagnostics),
                patch.object(install, "MANAGED_ROOT", managed),
                patch.object(install, "WEB_ROOT", web),
                patch.object(install.subprocess, "run", side_effect=run_step),
            ):
                result = install._install_client(backend_only=False, python=python)

            self.assertEqual(
                result["template_gallery"]["source"], "huggingface"
            )
            self.assertEqual(result["template_gallery"]["asset_mode"], "local")
            self.assertEqual(
                (web / "template-gallery" / "installed.webp").read_bytes(),
                b"installed",
            )
            self.assertEqual(
                (gallery / "checkout-marker.txt").read_text(encoding="utf-8"),
                "original",
            )
            self.assertFalse((client / ".template-gallery.install-backup").exists())
            self.assertFalse((client / "dist" / ".template-gallery.install-backup").exists())

    def test_windows_npm_batch_launcher_uses_node_without_a_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            node_root = Path(temporary)
            npm = node_root / "npm.cmd"
            npm.write_text("batch", encoding="utf-8")
            npm_cli = node_root / "node_modules" / "npm" / "bin" / "npm-cli.js"
            npm_cli.parent.mkdir(parents=True)
            npm_cli.write_text("// npm", encoding="utf-8")

            command = install._npm_command(
                {"node": str(node_root / "node.exe"), "npm": str(npm)},
                "run",
                "build",
            )

        self.assertEqual(command, [str(node_root / "node.exe"), str(npm_cli), "run", "build"])

    def test_client_mirror_preserves_an_explicit_offline_gallery_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = root / "client"
            web = root / "web"
            (client / "dist" / "template-gallery").mkdir(parents=True)
            (client / "dist" / "index.html").write_text("client", encoding="utf-8")
            (client / "dist" / "template-gallery" / "video.mp4").write_bytes(b"media")

            install._mirror_client_dist(client, web)

            self.assertEqual((web / "template-gallery" / "video.mp4").read_bytes(), b"media")

    def test_client_mirror_accepts_remote_build_without_gallery_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            client = root / "client"
            web = root / "web"
            (client / "dist").mkdir(parents=True)
            (client / "dist" / "index.html").write_text("remote client", encoding="utf-8")
            (web / "template-gallery").mkdir(parents=True)
            (web / "template-gallery" / "stale.mp4").write_bytes(b"stale")

            install._mirror_client_dist(client, web)

            self.assertEqual((web / "index.html").read_text(encoding="utf-8"), "remote client")
            self.assertFalse((web / "template-gallery").exists())

    def test_phase_contract_is_stable(self):
        self.assertEqual(PHASES, ["detect", "plan", "system-preparation", "toolchain", "backend", "client", "validation", "complete"])

    def test_profile_package_policy_checks_required_and_prohibited_packages(self):
        with patch.object(
            install,
            "load_manifest",
            return_value={"profiles": {"test": {"required": ["required-package"], "prohibited": ["bad_package"]}}},
        ):
            script = install._profile_package_script("test")
        self.assertIn("required-package", script)
        self.assertIn("bad_package", script)
        self.assertIn("prohibited_installed", script)


if __name__ == "__main__":
    unittest.main()
