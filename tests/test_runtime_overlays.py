from dataclasses import replace
import base64
import csv
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

from modiff import optimization_packages
from modiff import runtime_overlays
from modiff import install as modiff_install
from modiff.optional_runtimes import TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID


class RuntimeOverlayArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.archive_root = self.root / "artifacts"
        self.archive_root.mkdir()
        self.site_packages = self.root / "site-packages"
        self.site_packages.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _with_complete_record(members, dist_info):
        record_name = f"{dist_info}/RECORD"
        files = [(name, body, *details) for name, body, *details in members if name != record_name]
        rows = []
        for name, body, *_details in files:
            digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=").decode("ascii")
            rows.append((name, f"sha256={digest}", str(len(body))))
        rows.append((record_name, "", ""))
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(rows)
        return [*files, (record_name, output.getvalue().encode("utf-8"))]

    @staticmethod
    def _default_members(
        *,
        dist_info="demo_pkg-1.0.0.dist-info",
        metadata_name="demo-pkg",
        metadata_version="1.0.0",
        entry_points=False,
    ):
        members = [
            ("demo_pkg/__init__.py", b"VALUE = 'reviewed'\n"),
            (
                f"{dist_info}/METADATA",
                (
                    "Metadata-Version: 2.1\n"
                    f"Name: {metadata_name}\n"
                    f"Version: {metadata_version}\n\n"
                ).encode("utf-8"),
            ),
            (
                f"{dist_info}/WHEEL",
                (
                    "Wheel-Version: 1.0\n"
                    "Generator: MoDiff test fixture\n"
                    "Root-Is-Purelib: true\n"
                    "Tag: py3-none-any\n\n"
                ).encode("utf-8"),
            ),
        ]
        if entry_points:
            members.append(
                (
                    f"{dist_info}/entry_points.txt",
                    b"[console_scripts]\ndemo-tool = demo_pkg:main\n",
                )
            )
        return RuntimeOverlayArtifactTests._with_complete_record(members, dist_info)

    @staticmethod
    def _write_member(wheel, name, body, *, mode=stat.S_IFREG | 0o644):
        info = zipfile.ZipInfo(name)
        info.create_system = 3
        info.external_attr = mode << 16
        wheel.writestr(info, body)

    def _create_locked_wheel(self, members=None):
        filename = "demo_pkg-1.0.0-py3-none-any.whl"
        return self._store_locked_wheel(
            distribution="demo-pkg",
            version="1.0.0",
            filename=filename,
            members=members or self._default_members(),
        )

    def _store_locked_wheel(self, *, distribution, version, filename, members):
        source = self.root / filename
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
            for member in members:
                if len(member) == 2:
                    name, body = member
                    mode = stat.S_IFREG | 0o644
                else:
                    name, body, mode = member
                self._write_member(wheel, name, body, mode=mode)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        artifact = {
            "distribution": distribution,
            "version": version,
            "filename": filename,
            "url": f"https://files.example.invalid/{filename}",
            "sha256": digest,
            "byteSize": source.stat().st_size,
            "platform": "any",
            "pythonTag": "py3",
            "machine": "any",
        }
        destination = self.archive_root / digest / filename
        destination.parent.mkdir()
        source.replace(destination)
        return artifact, destination

    def _create_distribution_wheel(self, distribution, version):
        wheel_name = distribution.replace("-", "_").replace(".", "_")
        dist_info = f"{wheel_name}-{version}.dist-info"
        members = [
            (f"{wheel_name}/__init__.py", f"NAME = {distribution!r}\n".encode("utf-8")),
            (
                f"{dist_info}/METADATA",
                (
                    "Metadata-Version: 2.1\n"
                    f"Name: {distribution}\n"
                    f"Version: {version}\n\n"
                ).encode("utf-8"),
            ),
            (
                f"{dist_info}/WHEEL",
                (
                    "Wheel-Version: 1.0\n"
                    "Generator: MoDiff closure fixture\n"
                    "Root-Is-Purelib: true\n"
                    "Tag: py3-none-any\n\n"
                ).encode("utf-8"),
            ),
        ]
        members = self._with_complete_record(members, dist_info)
        return self._store_locked_wheel(
            distribution=distribution,
            version=version,
            filename=f"{wheel_name}-{version}-py3-none-any.whl",
            members=members,
        )

    def _extract(self, archive):
        with zipfile.ZipFile(archive) as wheel:
            for info in wheel.infolist():
                if info.is_dir():
                    continue
                relative = Path(*PurePosixPath(info.filename).parts)
                destination = self.site_packages / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(wheel.read(info))

    def assert_invalid_wheel(self, members):
        artifact, _archive = self._create_locked_wheel(members)
        with self.assertRaises(RuntimeError):
            runtime_overlays.locked_artifact_file_seal([artifact], self.archive_root)

    def test_valid_locked_wheel_verifies_exact_archive_and_extraction(self):
        artifact, archive = self._create_locked_wheel()
        self._extract(archive)

        expected_seal = runtime_overlays.locked_artifact_file_seal(
            [artifact], self.archive_root
        )
        anchor = runtime_overlays.verify_artifact_anchored_overlay(
            self.site_packages, [artifact], self.archive_root
        )

        self.assertEqual(anchor["schemaVersion"], 1)
        self.assertEqual(anchor["artifacts"], [artifact])
        self.assertEqual(anchor["fileSeal"], expected_seal)
        self.assertRegex(anchor["digest"], r"^sha256:[0-9a-f]{64}$")

    def test_complete_candidate_closure_reanchors_retained_cache_before_imports(self):
        profile = optimization_packages.OPTIONAL_RUNTIME_PROFILES[
            TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
        ]
        expected_versions = {
            "transformers": "5.14.1",
            "peft": "0.20.0",
            "tokenizers": "0.22.2",
            "typer": "0.27.1",
            "annotated-doc": "0.0.5",
            "rich": "15.0.0",
            "markdown-it-py": "4.2.0",
            "mdurl": "0.1.2",
            "pygments": "2.20.0",
            "shellingham": "1.5.4",
        }
        self.assertEqual(
            {package.distribution: package.version for package in profile.packages},
            expected_versions,
        )

        artifacts = []
        archives = []
        individual_union = {}
        for distribution, version in expected_versions.items():
            artifact, archive = self._create_distribution_wheel(distribution, version)
            artifacts.append(artifact)
            archives.append(archive)
            self._extract(archive)
            individual_seal = runtime_overlays.locked_artifact_file_seal(
                [artifact], self.archive_root
            )
            self.assertTrue(individual_union.keys().isdisjoint(individual_seal))
            individual_union.update(individual_seal)

        closure_seal = runtime_overlays.locked_artifact_file_seal(
            artifacts, self.archive_root
        )
        self.assertEqual(closure_seal, dict(sorted(individual_union.items())))
        self.assertEqual(len(closure_seal), 4 * len(expected_versions))
        first_anchor = runtime_overlays.verify_artifact_anchored_overlay(
            self.site_packages, artifacts, self.archive_root
        )
        self.assertEqual(first_anchor["fileSeal"], closure_seal)

        lease = runtime_overlays.InstallLease(
            token="retained-cache",
            owner_kind="test",
            owner_id="closure",
            cancel_event=threading.Event(),
        )
        opener = mock.Mock()
        with mock.patch.object(runtime_overlays, "build_opener", return_value=opener):
            retained = runtime_overlays.cache_locked_artifacts(
                artifacts, self.archive_root, lease=lease
            )
        opener.open.assert_not_called()
        self.assertEqual(retained, archives)
        retained_anchor = runtime_overlays.verify_artifact_anchored_overlay(
            self.site_packages, artifacts, self.archive_root
        )
        self.assertEqual(retained_anchor, first_anchor)

        archives[-1].write_bytes(b"replacement archive")
        with (
            mock.patch.object(
                runtime_overlays.importlib,
                "import_module",
                side_effect=AssertionError("archive identity must fail before imports"),
            ),
            self.assertRaisesRegex(RuntimeError, "catalog identity"),
        ):
            runtime_overlays.verify_artifact_anchored_overlay(
                self.site_packages, artifacts, self.archive_root
            )

    def test_artifact_filenames_and_hashes_are_part_of_the_profile_spec_digest(self):
        profile = optimization_packages.OPTIONAL_RUNTIME_PROFILES[
            TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
        ]
        artifacts = []
        for package in profile.packages:
            artifact, _archive = self._create_distribution_wheel(
                package.distribution, package.version
            )
            artifacts.append(artifact)
        locked = replace(profile, artifact_locks=tuple(artifacts))
        self.assertEqual(locked.to_spec_dict()["artifactLocks"], artifacts)
        self.assertNotEqual(locked.spec_digest, profile.spec_digest)

        changed_artifacts = [dict(item) for item in artifacts]
        changed_artifacts[0]["sha256"] = "0" * 64
        changed = replace(locked, artifact_locks=tuple(changed_artifacts))
        self.assertNotEqual(changed.spec_digest, locked.spec_digest)
        self.assertNotEqual(
            changed.to_spec_dict()["artifactLocks"][0]["sha256"],
            locked.to_spec_dict()["artifactLocks"][0]["sha256"],
        )

    def test_optional_runtime_has_one_complete_wheel_closure_for_every_supported_target(self):
        from packaging import tags

        profile = optimization_packages.OPTIONAL_RUNTIME_PROFILES[
            TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
        ]
        targets = {
            ("linux", "x86_64"): "manylinux_2_17_x86_64",
            ("linux", "arm64"): "manylinux_2_17_aarch64",
            ("macos", "x86_64"): "macosx_10_12_x86_64",
            ("macos", "arm64"): "macosx_11_0_arm64",
            ("windows", "x86_64"): "win_amd64",
            ("windows", "arm64"): "win_arm64",
        }
        for (platform_name, machine), wheel_platform in targets.items():
            supported = set(
                tags.cpython_tags(
                    python_version=(3, 12),
                    abis=["cp312"],
                    platforms=[wheel_platform],
                )
            ) | set(
                tags.compatible_tags(
                    python_version=(3, 12),
                    interpreter="cp312",
                    platforms=[wheel_platform],
                )
            )
            with (
                self.subTest(platform=platform_name, machine=machine),
                mock.patch.object(optimization_packages, "_platform_name", return_value=platform_name),
                mock.patch.object(optimization_packages, "_machine_name", return_value=machine),
                mock.patch.object(tags, "sys_tags", return_value=iter(supported)),
            ):
                selected = optimization_packages._artifact_install_plan(profile)
            self.assertEqual(len(selected), 10)
            self.assertEqual(
                [item["distribution"] for item in selected],
                [package.distribution for package in profile.packages],
            )
            self.assertTrue(all(item["byteSize"] > 0 for item in selected))

        malformed = [dict(item) for item in profile.artifact_locks]
        for item in malformed:
            if item["platform"] == "windows" and item["machine"] == "x86_64":
                item["byteSize"] = 0
                break
        windows_tags = set(
            tags.cpython_tags(
                python_version=(3, 12), abis=["cp312"], platforms=["win_amd64"]
            )
        ) | set(
            tags.compatible_tags(
                python_version=(3, 12), interpreter="cp312", platforms=["win_amd64"]
            )
        )
        with (
            mock.patch.object(optimization_packages, "_platform_name", return_value="windows"),
            mock.patch.object(optimization_packages, "_machine_name", return_value="x86_64"),
            mock.patch.object(tags, "sys_tags", return_value=iter(windows_tags)),
            self.assertRaisesRegex(RuntimeError, "artifact lock is invalid"),
        ):
            optimization_packages._artifact_install_plan(replace(profile, artifact_locks=tuple(malformed)))

    def test_base_installer_records_the_exact_uv_executable_for_overlay_reuse(self):
        managed = self.root / "tool-managed"
        tool_root = managed / "tools" / "uv"
        tool_root.mkdir(parents=True)
        executable = tool_root / "uv.exe"
        executable.write_bytes(b"reviewed-uv-test-binary")
        digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        lock = {
            "url": "https://github.com/astral-sh/uv/releases/download/test/uv.zip",
            "archiveSha256": "a" * 64,
            "executable": "uv.exe",
            "executableSha256": digest,
        }
        with (
            mock.patch.object(modiff_install, "MANAGED_ROOT", managed),
            mock.patch.object(modiff_install, "UV_TOOL_LOCKS", {("windows", "x86_64"): lock}),
            mock.patch.object(modiff_install, "normalized_os", return_value="windows"),
            mock.patch.object(modiff_install, "normalized_arch", return_value="x86_64"),
        ):
            self.assertEqual(Path(modiff_install._ensure_uv()), executable)
        receipt = (tool_root / "receipt.json").read_text(encoding="utf-8")
        self.assertIn(digest, receipt)
        with (
            mock.patch.object(optimization_packages, "MANAGED_ROOT", managed),
            mock.patch.object(optimization_packages, "UV_TOOL_LOCKS", {("windows", "x86_64"): lock}),
            mock.patch.object(optimization_packages, "_platform_name", return_value="windows"),
            mock.patch.object(optimization_packages.platform, "machine", return_value="AMD64"),
        ):
            self.assertEqual(Path(optimization_packages._verified_uv_executable()), executable)
            executable.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "integrity check"):
                optimization_packages._verified_uv_executable()

    def test_normalization_removes_only_known_receipts_and_generated_scripts(self):
        artifact, archive = self._create_locked_wheel(
            self._default_members(entry_points=True)
        )
        with zipfile.ZipFile(archive) as wheel:
            reviewed_record = wheel.read("demo_pkg-1.0.0.dist-info/RECORD")
        self._extract(archive)
        dist_info = self.site_packages / "demo_pkg-1.0.0.dist-info"
        (dist_info / "RECORD").write_bytes(b"installer-rewritten-record\n")
        for receipt in ("INSTALLER", "direct_url.json", "REQUESTED"):
            (dist_info / receipt).write_bytes(b"installer generated\n")
        script = self.site_packages / "bin" / "demo-tool"
        script.parent.mkdir()
        script.write_bytes(b"#!/usr/bin/env python\n")

        runtime_overlays.normalize_locked_wheel_install(
            self.site_packages, [artifact], self.archive_root
        )

        self.assertEqual((dist_info / "RECORD").read_bytes(), reviewed_record)
        for receipt in ("INSTALLER", "direct_url.json", "REQUESTED"):
            self.assertFalse((dist_info / receipt).exists())
        self.assertFalse(script.exists())
        self.assertFalse(script.parent.exists())
        runtime_overlays.verify_artifact_anchored_overlay(
            self.site_packages, [artifact], self.archive_root
        )

    def test_tampered_extracted_file_is_rejected(self):
        artifact, archive = self._create_locked_wheel()
        self._extract(archive)
        (self.site_packages / "demo_pkg" / "__init__.py").write_bytes(b"tampered\n")

        with self.assertRaisesRegex(RuntimeError, "differs from its locked wheel"):
            runtime_overlays.verify_artifact_anchored_overlay(
                self.site_packages, [artifact], self.archive_root
            )

    def test_self_consistent_forged_installed_record_cannot_replace_archive_anchor(self):
        artifact, archive = self._create_locked_wheel()
        self._extract(archive)
        module = self.site_packages / "demo_pkg" / "__init__.py"
        module.write_bytes(b"MALICIOUS = True\n")
        record = self.site_packages / "demo_pkg-1.0.0.dist-info" / "RECORD"
        rows = []
        for path in sorted(self.site_packages.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.site_packages).as_posix()
            if path == record:
                rows.append(f"{relative},,")
                continue
            body = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=")
            rows.append(f"{relative},sha256={digest.decode('ascii')},{len(body)}")
        record.write_text("\n".join(rows) + "\n", encoding="utf-8")

        with (
            mock.patch.object(
                runtime_overlays.importlib,
                "import_module",
                side_effect=AssertionError("forged overlay must fail before imports"),
            ),
            self.assertRaisesRegex(RuntimeError, "differs from its locked wheel"),
        ):
            runtime_overlays.verify_artifact_anchored_overlay(
                self.site_packages, [artifact], self.archive_root
            )

    def test_replaced_locked_archive_is_rejected_before_extraction_trust(self):
        artifact, archive = self._create_locked_wheel()
        archive.write_bytes(b"not the reviewed wheel")

        with self.assertRaisesRegex(RuntimeError, "catalog identity"):
            runtime_overlays.locked_artifact_file_seal([artifact], self.archive_root)

    def test_missing_locked_archive_is_rejected(self):
        artifact, archive = self._create_locked_wheel()
        archive.unlink()

        with self.assertRaises((FileNotFoundError, RuntimeError)):
            runtime_overlays.locked_artifact_file_seal([artifact], self.archive_root)

    def test_extra_extracted_file_is_rejected(self):
        artifact, archive = self._create_locked_wheel()
        self._extract(archive)
        (self.site_packages / "demo_pkg" / "unreviewed.py").write_bytes(b"extra\n")

        with self.assertRaisesRegex(RuntimeError, "does not exactly match"):
            runtime_overlays.verify_artifact_anchored_overlay(
                self.site_packages, [artifact], self.archive_root
            )

    def test_extracted_pth_startup_hook_is_rejected(self):
        artifact, archive = self._create_locked_wheel()
        self._extract(archive)
        (self.site_packages / "unreviewed.pth").write_bytes(b"import unreviewed\n")

        with self.assertRaisesRegex(RuntimeError, "forbidden .pth"):
            runtime_overlays.verify_artifact_anchored_overlay(
                self.site_packages, [artifact], self.archive_root
            )

    def test_missing_or_duplicate_wheel_identity_documents_are_rejected(self):
        defaults = self._default_members()
        cases = {
            "missing WHEEL": [item for item in defaults if not item[0].endswith("/WHEEL")],
            "missing RECORD": [item for item in defaults if not item[0].endswith("/RECORD")],
            "duplicate WHEEL": defaults
            + [
                (
                    "duplicate-1.0.0.dist-info/WHEEL",
                    b"Wheel-Version: 1.0\nTag: py3-none-any\n\n",
                )
            ],
            "duplicate RECORD": defaults
            + [("duplicate-1.0.0.dist-info/RECORD", b"second-record\n")],
        }
        for label, members in cases.items():
            with self.subTest(label=label):
                self.assert_invalid_wheel(members)

    def test_wheel_record_must_cover_exact_files_hashes_and_sizes(self):
        defaults = self._default_members()
        record_name = "demo_pkg-1.0.0.dist-info/RECORD"
        record = next(body for name, body, *_details in defaults if name == record_name).decode("utf-8")
        cases = {
            "missing member": record.replace(next(line for line in record.splitlines(True) if line.startswith("demo_pkg/__init__.py,")), ""),
            "unknown member": record + "demo_pkg/unknown.py,sha256=AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA,1\n",
            "wrong digest": record.replace("sha256=", "sha256=A", 1),
            "wrong size": record.replace(",19\n", ",20\n", 1),
            "hashed self": record.replace(f"{record_name},,", f"{record_name},sha256=AAAA,1"),
            "duplicate path": record + next(line for line in record.splitlines(True) if line.startswith("demo_pkg/__init__.py,")),
        }
        for label, malformed in cases.items():
            with self.subTest(label=label):
                members = [
                    (name, malformed.encode("utf-8") if name == record_name else body, *details)
                    for name, body, *details in defaults
                ]
                self.assert_invalid_wheel(members)

    def test_dist_info_metadata_project_and_version_must_match_lock(self):
        cases = {
            "dist-info project": self._default_members(
                dist_info="other_pkg-1.0.0.dist-info"
            ),
            "dist-info version": self._default_members(
                dist_info="demo_pkg-2.0.0.dist-info"
            ),
            "metadata project": self._default_members(metadata_name="other-pkg"),
            "metadata version": self._default_members(metadata_version="2.0.0"),
        }
        for label, members in cases.items():
            with self.subTest(label=label):
                self.assert_invalid_wheel(members)

    def test_archive_pth_and_windows_unsafe_paths_are_rejected(self):
        cases = {
            "startup hook": "demo_pkg/unreviewed.pth",
            "alternate data stream": "demo_pkg/payload:stream",
            "trailing dot": "demo_pkg/payload.",
            "trailing space": "demo_pkg/payload ",
        }
        for label, unsafe_name in cases.items():
            with self.subTest(label=label):
                self.assert_invalid_wheel(
                    self._default_members() + [(unsafe_name, b"unreviewed\n")]
                )

    def test_unicode_normalization_collision_is_rejected(self):
        members = self._default_members() + [
            ("demo_pkg/caf\N{LATIN SMALL LETTER E WITH ACUTE}.py", b"first\n"),
            ("demo_pkg/cafe\N{COMBINING ACUTE ACCENT}.py", b"second\n"),
        ]
        self.assert_invalid_wheel(members)

    def test_symlink_archive_member_is_rejected(self):
        members = self._default_members() + [
            ("demo_pkg/link.py", b"outside.py", stat.S_IFLNK | 0o777)
        ]
        self.assert_invalid_wheel(members)

    def test_pre_cancelled_acquisition_does_not_open_a_url(self):
        artifact, archive = self._create_locked_wheel()
        archive.unlink()
        lease = runtime_overlays.InstallLease(
            token="test",
            owner_kind="test",
            owner_id="test",
            cancel_event=threading.Event(),
        )
        lease.cancel_event.set()
        opener = mock.Mock()

        with (
            mock.patch.object(runtime_overlays, "build_opener", return_value=opener),
            self.assertRaises(runtime_overlays.OverlayCancelled),
        ):
            runtime_overlays.cache_locked_artifacts(
                [artifact], self.archive_root, lease=lease
            )

        opener.open.assert_not_called()

    def test_acquisition_polls_cancellation_while_streaming(self):
        artifact, archive = self._create_locked_wheel()
        archive.unlink()
        lease = runtime_overlays.InstallLease(
            token="test",
            owner_kind="test",
            owner_id="test",
            cancel_event=threading.Event(),
        )

        class CancellingResponse:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def geturl(self):
                return artifact["url"]

            def read(self, _size):
                lease.cancel_event.set()
                return b"partial"

        opener = SimpleNamespace(open=mock.Mock(return_value=CancellingResponse()))
        with (
            mock.patch.object(runtime_overlays, "build_opener", return_value=opener),
            self.assertRaises(runtime_overlays.OverlayCancelled),
        ):
            runtime_overlays.cache_locked_artifacts(
                [artifact], self.archive_root, lease=lease
            )

        opener.open.assert_called_once()
        self.assertFalse(any(self.archive_root.rglob("*.part")))

    def test_os_lease_excludes_another_process_and_reacquires_after_release(self):
        managed = self.root / "lease-managed"
        lock_path = managed / "optimizations" / "install.lock"
        environment = {**os.environ, "MODIFF_MANAGED_ROOT": str(managed)}
        probe = """
from modiff.runtime_overlays import OverlayInstallBusy, release_install, reserve_install
try:
    lease = reserve_install('test', 'child')
except OverlayInstallBusy:
    print('busy')
else:
    print('acquired')
    release_install(lease)
"""
        with (
            mock.patch.object(runtime_overlays, "MANAGED_ROOT", managed),
            mock.patch.object(runtime_overlays, "INSTALL_LEASE_PATH", lock_path),
        ):
            lease = runtime_overlays.reserve_install("test", "parent")
            try:
                blocked = subprocess.run(
                    [sys.executable, "-c", probe],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )
            finally:
                runtime_overlays.release_install(lease)
            reacquired = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        self.assertEqual(blocked.returncode, 0, blocked.stderr)
        self.assertEqual(blocked.stdout.strip(), "busy")
        self.assertEqual(reacquired.returncode, 0, reacquired.stderr)
        self.assertEqual(reacquired.stdout.strip(), "acquired")

    def test_managed_promotion_uses_exact_directory_and_never_replaces_target(self):
        managed = self.root / "promotion-managed"
        staging = managed / "optimizations" / "staging"
        environments = managed / "optimizations" / "environments"
        source = staging / "runtime-1-cafebabe"
        destination = environments / source.name
        (source / "site-packages").mkdir(parents=True)
        environments.mkdir(parents=True)
        (source / "site-packages" / "proof.txt").write_text("reviewed", encoding="utf-8")
        with mock.patch.object(runtime_overlays, "MANAGED_ROOT", managed):
            runtime_overlays.promote_managed_directory(source, destination)
        self.assertFalse(source.exists())
        self.assertEqual(
            (destination / "site-packages" / "proof.txt").read_text(encoding="utf-8"),
            "reviewed",
        )

        replacement_source = staging / "runtime-2-deadbeef"
        replacement_source.mkdir()
        (replacement_source / "canary.txt").write_text("source", encoding="utf-8")
        occupied = environments / replacement_source.name
        occupied.mkdir()
        (occupied / "canary.txt").write_text("destination", encoding="utf-8")
        with (
            mock.patch.object(runtime_overlays, "MANAGED_ROOT", managed),
            self.assertRaises(runtime_overlays.OverlayStorageUnsafe),
        ):
            runtime_overlays.promote_managed_directory(replacement_source, occupied)
        self.assertEqual((replacement_source / "canary.txt").read_text(), "source")
        self.assertEqual((occupied / "canary.txt").read_text(), "destination")

    def test_managed_cleanup_quarantines_exact_tree_and_rejects_unsafe_target(self):
        managed = self.root / "cleanup-managed"
        environments = managed / "optimizations" / "environments"
        target = environments / "runtime-1-cafebabe"
        (target / "site-packages" / "nested").mkdir(parents=True)
        (target / "site-packages" / "nested" / "proof.txt").write_text(
            "reviewed",
            encoding="utf-8",
        )
        with mock.patch.object(runtime_overlays, "MANAGED_ROOT", managed):
            self.assertTrue(
                runtime_overlays.remove_managed_directory(target, parent=environments)
            )
            self.assertFalse(
                runtime_overlays.remove_managed_directory(target, parent=environments)
            )
        self.assertFalse(target.exists())
        self.assertEqual(list(environments.glob(".cleanup-*")), [])

        unsafe = environments / "runtime-2-deadbeef"
        unsafe.write_text("do-not-delete", encoding="utf-8")
        with (
            mock.patch.object(runtime_overlays, "MANAGED_ROOT", managed),
            self.assertRaises(runtime_overlays.OverlayStorageUnsafe),
        ):
            runtime_overlays.remove_managed_directory(unsafe, parent=environments)
        self.assertEqual(unsafe.read_text(encoding="utf-8"), "do-not-delete")

    def test_staging_identity_rejects_directory_replacement_before_promote_or_cleanup(self):
        managed = self.root / "identity-managed"
        staging = managed / "optimizations" / "staging"
        environments = managed / "optimizations" / "environments"
        staged = staging / "runtime-1-cafebabe"
        displaced = staging / "displaced"
        staged.mkdir(parents=True)
        environments.mkdir(parents=True)
        (staged / "proof.txt").write_text("validated", encoding="utf-8")
        with mock.patch.object(runtime_overlays, "MANAGED_ROOT", managed):
            identity = runtime_overlays.managed_directory_identity(staged)
            staged.replace(displaced)
            staged.mkdir()
            (staged / "proof.txt").write_text("replacement", encoding="utf-8")
            with self.assertRaises(runtime_overlays.OverlayStorageUnsafe):
                runtime_overlays.promote_managed_directory(
                    staged,
                    environments / staged.name,
                    expected_identity=identity,
                )
            with self.assertRaises(runtime_overlays.OverlayStorageUnsafe):
                runtime_overlays.remove_managed_directory(
                    staged,
                    parent=staging,
                    expected_identity=identity,
                )
        self.assertEqual((staged / "proof.txt").read_text(), "replacement")
        self.assertEqual((displaced / "proof.txt").read_text(), "validated")
        self.assertFalse((environments / staged.name).exists())

    def test_posix_promotion_requires_platform_exclusive_rename_flags(self):
        class Operation:
            def __init__(self):
                self.calls = []

            def __call__(self, *args):
                self.calls.append(args)
                return 0

        linux = Operation()
        macos = Operation()
        with (
            mock.patch("ctypes.CDLL", return_value=SimpleNamespace(renameat2=linux)),
            mock.patch.object(sys, "platform", "linux"),
        ):
            runtime_overlays._posix_rename_noreplace(3, "source", 4, "target")
        self.assertEqual(linux.calls[0][-1], 1)
        with (
            mock.patch("ctypes.CDLL", return_value=SimpleNamespace(renameatx_np=macos)),
            mock.patch.object(sys, "platform", "darwin"),
        ):
            runtime_overlays._posix_rename_noreplace(3, "source", 4, "target")
        self.assertEqual(macos.calls[0][-1], 0x00000004)
        with (
            mock.patch("ctypes.CDLL", return_value=SimpleNamespace()),
            mock.patch.object(sys, "platform", "linux"),
            self.assertRaises(runtime_overlays.OverlayStorageUnsafe),
        ):
            runtime_overlays._posix_rename_noreplace(3, "source", 4, "target")

    def test_cancelled_lease_cannot_enter_managed_promotion(self):
        lease = runtime_overlays.InstallLease(
            token="cancelled",
            owner_kind="test",
            owner_id="promotion",
            cancel_event=threading.Event(),
        )
        lease.cancel_event.set()
        with (
            mock.patch.object(runtime_overlays, "_ACTIVE_INSTALL", lease),
            mock.patch.object(runtime_overlays, "promote_managed_directory") as promote,
            self.assertRaises(runtime_overlays.OverlayCancelled),
        ):
            runtime_overlays.promote_staged_environment(
                lease,
                self.root / "staged",
                self.root / "destination",
            )
        promote.assert_not_called()

    def test_cancel_kills_benign_child_and_grandchild_before_escape(self):
        managed = self.root / "cancel-managed"
        lock_path = managed / "optimizations" / "install.lock"
        started = self.root / "cancel-started"
        escaped = self.root / "cancel-escaped"
        grandchild = (
            "import time; from pathlib import Path; "
            f"time.sleep(1.0); Path({str(escaped)!r}).write_text('escaped')"
        )
        child = (
            "import subprocess, sys, time; from pathlib import Path; "
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
            f"Path({str(started)!r}).write_text('started'); time.sleep(30)"
        )
        outcome = []
        with (
            mock.patch.object(runtime_overlays, "MANAGED_ROOT", managed),
            mock.patch.object(runtime_overlays, "INSTALL_LEASE_PATH", lock_path),
        ):
            lease = runtime_overlays.reserve_install("test", "cancel-tree")

            def run_tree():
                try:
                    runtime_overlays.run_cancellable_command(
                        [sys.executable, "-I", "-c", child],
                        environment=os.environ.copy(),
                        lease=lease,
                        timeout=30,
                        cwd=self.root,
                    )
                except runtime_overlays.OverlayCancelled:
                    outcome.append("cancelled")

            worker = threading.Thread(target=run_tree, daemon=True)
            worker.start()
            deadline = time.monotonic() + 10
            while not started.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(started.exists(), "benign child did not start")
            self.assertTrue(runtime_overlays.cancel_install(lease.token))
            worker.join(timeout=10)
            runtime_overlays.release_install(lease)
        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome, ["cancelled"])
        time.sleep(1.25)
        self.assertFalse(escaped.exists())

    @unittest.skipUnless(os.name == "nt", "Windows Job Object containment")
    def test_windows_job_contains_breakaway_descendants(self):
        managed = self.root / "breakaway-managed"
        lock_path = managed / "optimizations" / "install.lock"
        blocked = self.root / "breakaway-blocked"
        launched = self.root / "breakaway-launched"
        escaped = self.root / "breakaway-escaped"
        grandchild = (
            "import time; from pathlib import Path; "
            f"time.sleep(0.5); Path({str(escaped)!r}).write_text('escaped')"
        )
        child = (
            "import subprocess, sys; from pathlib import Path; "
            "flags=subprocess.CREATE_BREAKAWAY_FROM_JOB|subprocess.CREATE_NEW_PROCESS_GROUP; "
            "\ntry: subprocess.Popen([sys.executable, '-c', " + repr(grandchild) + "], creationflags=flags); "
            "Path(" + repr(str(launched)) + ").write_text('launched')"
            "\nexcept OSError: Path(" + repr(str(blocked)) + ").write_text('blocked')"
        )
        with (
            mock.patch.object(runtime_overlays, "MANAGED_ROOT", managed),
            mock.patch.object(runtime_overlays, "INSTALL_LEASE_PATH", lock_path),
        ):
            lease = runtime_overlays.reserve_install("test", "breakaway")
            try:
                result = runtime_overlays.run_cancellable_command(
                    [sys.executable, "-I", "-c", child],
                    environment=os.environ.copy(),
                    lease=lease,
                    timeout=10,
                    cwd=self.root,
                )
            finally:
                runtime_overlays.release_install(lease)
        self.assertEqual(result["returnCode"], 0, result["stderr"])
        self.assertTrue(blocked.exists() or launched.exists())
        time.sleep(0.75)
        self.assertFalse(escaped.exists())

    def test_parent_death_watchdog_retains_lease_until_tree_is_dead(self):
        managed = self.root / "watchdog-managed"
        lock_path = managed / "optimizations" / "install.lock"
        started = self.root / "watchdog-started"
        escaped = self.root / "watchdog-escaped"
        grandchild = (
            "import time; from pathlib import Path; "
            f"time.sleep(1.5); Path({str(escaped)!r}).write_text('escaped')"
        )
        child = (
            "import subprocess, sys, time; from pathlib import Path; "
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
            f"Path({str(started)!r}).write_text('started'); time.sleep(30)"
        )
        parent = f"""
import os, sys
from pathlib import Path
from modiff.runtime_overlays import release_install, reserve_install, run_cancellable_command
lease = reserve_install('test', 'orphan-parent')
try:
    run_cancellable_command(
        [sys.executable, '-I', '-c', {child!r}],
        environment=os.environ.copy(),
        lease=lease,
        timeout=30,
        cwd=Path({str(self.root)!r}),
    )
finally:
    release_install(lease)
"""
        environment = {**os.environ, "MODIFF_MANAGED_ROOT": str(managed)}
        process = subprocess.Popen(
            [sys.executable, "-c", parent],
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + 10
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(started.exists(), "watchdog child did not start")
        process.kill()
        process.wait(timeout=10)

        acquired = None
        with (
            mock.patch.object(runtime_overlays, "MANAGED_ROOT", managed),
            mock.patch.object(runtime_overlays, "INSTALL_LEASE_PATH", lock_path),
        ):
            deadline = time.monotonic() + 10
            while acquired is None and time.monotonic() < deadline:
                try:
                    acquired = runtime_overlays.reserve_install("test", "replacement")
                except runtime_overlays.OverlayInstallBusy:
                    time.sleep(0.05)
            self.assertIsNotNone(acquired, "watchdog did not release the OS lease")
            runtime_overlays.release_install(acquired)
        time.sleep(1.75)
        self.assertFalse(escaped.exists())

    def test_unqualified_candidate_rejects_before_artifact_selection_or_lease(self):
        profile = optimization_packages.OPTIONAL_RUNTIME_PROFILES[
            TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
        ]
        selector = mock.Mock(side_effect=AssertionError("selector must not run"))
        reserve = mock.Mock(side_effect=AssertionError("lease must not be reserved"))
        with (
            mock.patch.object(optimization_packages, "_artifact_install_plan", selector),
            mock.patch.object(optimization_packages, "reserve_install", reserve),
            mock.patch.object(
                optimization_packages,
                "current_base_binding",
                side_effect=AssertionError("base inspection must not run"),
            ),
            self.assertRaisesRegex(RuntimeError, "not qualified for installation"),
        ):
            optimization_packages.validate_optional_runtime_install_request(
                profile.id, profile.spec_digest, consent=True
            )

        selector.assert_not_called()
        reserve.assert_not_called()

    def test_flag_flip_with_incomplete_locks_still_rejects_before_lease(self):
        current = optimization_packages.OPTIONAL_RUNTIME_PROFILES[
            TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID
        ]
        future = replace(current, install_action_available=True, artifact_locks=())
        self.assertEqual(future.artifact_locks, ())
        selector = mock.Mock(wraps=optimization_packages._artifact_install_plan)
        reserve = mock.Mock(side_effect=AssertionError("lease must not be reserved"))
        installer = mock.Mock(side_effect=AssertionError("installer must not be selected"))
        with (
            mock.patch.object(
                optimization_packages,
                "OPTIONAL_RUNTIME_PROFILES",
                {future.id: future},
            ),
            mock.patch.object(optimization_packages, "_artifact_install_plan", selector),
            mock.patch.object(optimization_packages, "reserve_install", reserve),
            mock.patch.object(optimization_packages, "current_base_binding", return_value={}),
            mock.patch.object(optimization_packages, "_verified_uv_executable", installer),
            self.assertRaisesRegex(RuntimeError, "artifact lock is incomplete"),
        ):
            optimization_packages.validate_optional_runtime_install_request(
                future.id, future.spec_digest, consent=True
            )

        selector.assert_called_once_with(future)
        installer.assert_not_called()
        reserve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
