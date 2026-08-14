from copy import deepcopy
import base64
import csv
import hashlib
import io
import os
from pathlib import Path
import stat
import tarfile
import tempfile
import threading
import unittest
from unittest import mock
import zipfile

from modiff import optimization_packages, runtime_source_builds
from modiff.optional_runtimes import (
    OPTIONAL_RUNTIME_PROFILES,
    TRANSFORMERS_MAIN_COMMIT,
    TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID,
    TRANSFORMERS_MAIN_REVIEW_BASE_COMMIT,
    TRANSFORMERS_MAIN_REVIEWED_DELTA_PATHS,
    TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID,
    project_optional_runtime_qualification,
)
from modiff.runtime_overlays import InstallLease, locked_artifact_file_seal


class FakeSourceResponse:
    def __init__(self, body, *, url, content_length=None, on_read=None):
        self.body = body
        self.url = url
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self.on_read = on_read
        self.sent = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def geturl(self):
        return self.url

    def read(self, _size):
        if self.sent:
            return b""
        self.sent = True
        if self.on_read is not None:
            self.on_read()
        return self.body


class SourceBuildFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.cache = self.root / "artifacts"
        self.cache.mkdir()
        self.commit = "a" * 40
        self.archive_root = f"demo-{self.commit}"
        self.sequence = 0

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def lease():
        return InstallLease(
            token="source-build-test",
            owner_kind="test",
            owner_id="source-build",
            cancel_event=threading.Event(),
        )

    @staticmethod
    def _record_members(members, dist_info):
        record_name = f"{dist_info}/RECORD"
        rows = []
        for name, body in members:
            digest = base64.urlsafe_b64encode(hashlib.sha256(body).digest()).rstrip(b"=")
            rows.append((name, f"sha256={digest.decode('ascii')}", str(len(body))))
        rows.append((record_name, "", ""))
        output = io.StringIO(newline="")
        csv.writer(output, lineterminator="\n").writerows(rows)
        return [*members, (record_name, output.getvalue().encode("utf-8"))]

    def make_locked_wheel_with_startup_hook(self):
        distribution = "startup-demo"
        normalized = "startup_demo"
        version = "1.0.0"
        dist_info = f"{normalized}-{version}.dist-info"
        members = self._record_members(
            [
                (f"{normalized}/__init__.py", b"VALUE = 1\n"),
                ("startup-demo.pth", b"import startup_demo\n"),
                (
                    f"{dist_info}/METADATA",
                    b"Metadata-Version: 2.1\nName: startup-demo\nVersion: 1.0.0\n\n",
                ),
                (
                    f"{dist_info}/WHEEL",
                    b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
                ),
            ],
            dist_info,
        )
        filename = f"{normalized}-{version}-py3-none-any.whl"
        wheel_path = self.root / filename
        with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_STORED) as wheel:
            for name, body in members:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                wheel.writestr(info, body)
        digest = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
        destination = self.cache / digest / filename
        destination.parent.mkdir()
        wheel_path.replace(destination)
        return {
            "distribution": distribution,
            "version": version,
            "filename": filename,
            "url": f"https://files.pythonhosted.org/packages/reviewed/{filename}",
            "sha256": digest,
            "byteSize": destination.stat().st_size,
            "platform": "any",
            "pythonTag": "py3",
            "machine": "any",
        }

    def make_source_archive(self, members=None):
        members = members or [
            ("LICENSE", b"reviewed license\n", "file", ""),
            ("README.md", b"reviewed source\n", "file", ""),
            ("pyproject.toml", b"[build-system]\nrequires=['evil']\n", "file", ""),
            (
                "setup.py",
                b"raise RuntimeError('upstream build hooks must never execute')\n",
                "file",
                "",
            ),
            (
                "src/demo/__init__.py",
                b"VALUE = 'reviewed'\ndef main():\n    return VALUE\n",
                "file",
                "",
            ),
            ("src/demo/module.py", b"ANSWER = 42\n", "file", ""),
            ("ignored.txt", b"must not be extracted\n", "file", ""),
        ]
        self.sequence += 1
        archive = self.root / f"source-{self.sequence}.tar.gz"
        with tarfile.open(archive, "w:gz") as output:
            for relative, body, kind, linkname in members:
                name = f"{self.archive_root}/{relative}"
                info = tarfile.TarInfo(name)
                info.mtime = 1_700_000_000
                if kind == "file":
                    info.type = tarfile.REGTYPE
                    info.size = len(body)
                    output.addfile(info, io.BytesIO(body))
                elif kind == "symlink":
                    info.type = tarfile.SYMTYPE
                    info.linkname = linkname
                    output.addfile(info)
                elif kind == "hardlink":
                    info.type = tarfile.LNKTYPE
                    info.linkname = linkname
                    output.addfile(info)
                else:
                    raise AssertionError(kind)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        destination = self.cache / digest / f"demo-{self.commit}.tar.gz"
        destination.parent.mkdir(exist_ok=True)
        archive.replace(destination)
        return destination, {
            "kind": "github_commit_tarball",
            "repository": "owner/demo",
            "commit": self.commit,
            "archiveRoot": self.archive_root,
            "filename": f"demo-{self.commit}.tar.gz",
            "url": f"https://codeload.github.com/owner/demo/tar.gz/{self.commit}",
            "sha256": digest,
            "byteSize": destination.stat().st_size,
        }

    def make_contract(self, *, source_members=None, derive_output=False):
        source_path, source = self.make_source_archive(source_members)
        contract = {
            "schemaVersion": 1,
            "distribution": "demo-runtime",
            "version": "1.0.0",
            "sourceArtifact": source,
            "recipe": runtime_source_builds.SOURCE_BUILD_RECIPE,
            "pythonTag": "py3",
            "sourceDateEpoch": runtime_source_builds.FIXED_SOURCE_DATE_EPOCH,
            "sourceFiles": ["LICENSE", "README.md", "pyproject.toml", "setup.py"],
            "sourceTrees": ["src/demo"],
            "buildDependencies": [],
            "wheelMetadata": {
                "summary": "A reviewed fixture",
                "license": "Fixture License",
                "requiresPython": ">=3.10",
                "requiresDist": ["fixture-dependency>=1"],
                "consoleScripts": {"demo-runtime": "demo:main"},
                "packageRoot": "src",
                "licenseFile": "LICENSE",
            },
            "outputWheel": {
                "distribution": "demo-runtime",
                "version": "1.0.0",
                "filename": "demo_runtime-1.0.0-py3-none-any.whl",
                "sha256": "0" * 64,
                "byteSize": 1,
                "platform": "any",
                "pythonTag": "py3",
                "machine": "any",
            },
        }
        if derive_output:
            self.sequence += 1
            derivation = self.root / f"derivation-{self.sequence}"
            derivation.mkdir()
            extracted = derivation / "source"
            runtime_source_builds.extract_locked_source_tree(
                contract,
                source_path,
                extracted,
                lease=self.lease(),
            )
            output = derivation / contract["outputWheel"]["filename"]
            runtime_source_builds.assemble_locked_pure_python_wheel(
                contract,
                extracted,
                output,
                lease=self.lease(),
            )
            body = output.read_bytes()
            contract["outputWheel"]["sha256"] = hashlib.sha256(body).hexdigest()
            contract["outputWheel"]["byteSize"] = len(body)
        return contract, source_path


class SourceBuildContractTests(SourceBuildFixture):
    def test_exact_transformers_main_profile_is_immutable_and_linux_scoped(self):
        current = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_PEFT_RUNTIME_PROFILE_ID]
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID]
        source = runtime_source_builds.validate_source_build_contract(
            profile.source_builds[0]
        )

        self.assertEqual(current.packages[0].version, "5.14.1")
        self.assertNotIn("sourceBuilds", current.to_spec_dict())
        self.assertEqual(
            current.spec_digest,
            "sha256:7ac7fcf3b38718dbfc1d8c73e8836b9ee754c6801494ce0184f2c5d726525ab7",
        )
        self.assertEqual(profile.packages[0].version, "5.16.0.dev0")
        self.assertEqual(source["sourceArtifact"]["commit"], TRANSFORMERS_MAIN_COMMIT)
        self.assertEqual(
            source["sourceArtifact"]["sha256"],
            "206aaa32386db09202db21038f5610a7fb0f2817f013003ec3beb90aafbfc0d6",
        )
        self.assertEqual(source["sourceArtifact"]["byteSize"], 20_532_481)
        self.assertEqual(
            TRANSFORMERS_MAIN_REVIEW_BASE_COMMIT,
            "c1ff11866b3e2c473f92460ee0bf68d739921609",
        )
        self.assertEqual(
            TRANSFORMERS_MAIN_REVIEWED_DELTA_PATHS,
            (
                "docs/source/en/chat_templating_multimodal.md",
                "docs/source/en/image_processors.md",
                "docs/source/en/video_processors.md",
            ),
        )
        self.assertEqual(source["recipe"], "modiff_pure_python_wheel_v1")
        self.assertEqual(source["buildDependencies"], [])
        self.assertEqual(
            source["wheelMetadata"]["requiresDist"],
            [
                "huggingface-hub>=1.5.0,<2.0",
                "numpy>=1.17",
                "packaging>=20.0",
                "pyyaml>=5.1",
                "regex>=2025.10.22",
                "tokenizers>=0.22.0,<=0.23.0",
                "typer",
                "safetensors>=0.8.0",
                "tqdm>=4.60",
            ],
        )
        self.assertEqual(
            source["outputWheel"]["sha256"],
            "8a439d25595c6dde486cfbd5a6ed8158e0fe7554ec236491668425e11952898f",
        )
        self.assertEqual(source["outputWheel"]["byteSize"], 52_395_984)
        projected = profile.to_spec_dict()
        projected["sourceBuilds"][0]["sourceFiles"].append("unreviewed.py")
        self.assertNotIn(
            "unreviewed.py",
            profile.to_spec_dict()["sourceBuilds"][0]["sourceFiles"],
        )
        self.assertEqual(profile.contract_state, "qualified_platform_scoped")
        self.assertFalse(profile.cutover_ready)
        self.assertFalse(profile.install_action_available)
        self.assertFalse(profile.activation_available)
        qualified_targets = {
            (target.platform, target.machine)
            for target in profile.target_contracts
            if target.contract_state == "qualified"
            and target.cutover_ready
            and target.install_action_available
            and target.activation_available
        }
        self.assertEqual(qualified_targets, {("linux", "x86_64")})
        for target in profile.target_contracts:
            if (target.platform, target.machine) == ("linux", "x86_64"):
                continue
            self.assertEqual(target.contract_state, "candidate_unqualified")
            self.assertFalse(target.cutover_ready)
            self.assertFalse(target.install_action_available)
            self.assertFalse(target.activation_available)

        plan = optimization_packages._artifact_install_plan(profile)
        self.assertEqual(len(plan), len(profile.packages))
        self.assertNotIn("url", plan[0])
        install_urls = optimization_packages._artifact_install_urls(profile)
        self.assertEqual(len(install_urls), len(profile.packages) - 1)
        self.assertTrue(
            all(url.startswith("https://files.pythonhosted.org/") for url in install_urls)
        )

    def test_unqualified_main_target_rejects_before_binding_or_installer_resolution(self):
        import modiff.optional_runtimes as optional_runtimes

        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID]
        with (
            mock.patch.object(
                optional_runtimes,
                "optional_runtime_target",
                return_value=("macos", "arm64"),
            ),
            mock.patch.object(optimization_packages, "current_base_binding") as binding,
            mock.patch.object(optimization_packages, "_verified_uv_executable") as installer,
            self.assertRaisesRegex(RuntimeError, "not qualified for installation"),
        ):
            optimization_packages.validate_optional_runtime_install_request(
                profile.id,
                profile.spec_digest,
                consent=True,
            )
        binding.assert_not_called()
        installer.assert_not_called()

    def test_only_linux_x86_64_target_reaches_install_request_validation(self):
        import modiff.optional_runtimes as optional_runtimes

        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID]
        targets = (
            ("linux", "x86_64"),
            ("linux", "arm64"),
            ("macos", "x86_64"),
            ("macos", "arm64"),
            ("windows", "x86_64"),
            ("windows", "arm64"),
        )
        for target in targets:
            binding = mock.Mock(return_value={"reviewed": True})
            installer = mock.Mock(return_value="reviewed-uv")
            with (
                self.subTest(target=target),
                mock.patch.object(
                    optional_runtimes,
                    "optional_runtime_target",
                    return_value=target,
                ),
                mock.patch.object(
                    optimization_packages,
                    "current_base_binding",
                    binding,
                ),
                mock.patch.object(
                    optimization_packages,
                    "_verified_uv_executable",
                    installer,
                ),
            ):
                if target == ("linux", "x86_64"):
                    request = optimization_packages.validate_optional_runtime_install_request(
                        profile.id,
                        profile.spec_digest,
                        consent=True,
                    )
                    self.assertEqual(request["profile"], profile)
                    binding.assert_called_once()
                    installer.assert_called_once()
                    activation_spec = (
                        optimization_packages.validate_optional_runtime_activation_request(
                            profile.id,
                            profile.spec_digest,
                            consent=True,
                        )
                    )
                    self.assertEqual(activation_spec["id"], profile.id)
                else:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "not qualified for installation",
                    ):
                        optimization_packages.validate_optional_runtime_install_request(
                            profile.id,
                            profile.spec_digest,
                            consent=True,
                        )
                    binding.assert_not_called()
                    installer.assert_not_called()
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "not qualified for activation",
                    ):
                        optimization_packages.validate_optional_runtime_activation_request(
                            profile.id,
                            profile.spec_digest,
                            consent=True,
                        )

    def test_latest_main_relock_records_only_unselected_documentation_changes(self):
        profile = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID]
        source_build = profile.source_builds[0]

        self.assertNotEqual(TRANSFORMERS_MAIN_REVIEW_BASE_COMMIT, TRANSFORMERS_MAIN_COMMIT)
        self.assertEqual(len(TRANSFORMERS_MAIN_REVIEWED_DELTA_PATHS), 3)
        for path in TRANSFORMERS_MAIN_REVIEWED_DELTA_PATHS:
            self.assertTrue(path.startswith("docs/"))
            self.assertNotIn(path, source_build["sourceFiles"])
            self.assertFalse(
                any(
                    path == tree or path.startswith(f"{tree}/")
                    for tree in source_build["sourceTrees"]
                )
            )
        # The reviewed a597 advance did not change any selected package bytes,
        # so its normalized output remains the independently derived c1ff lock.
        self.assertEqual(
            source_build["outputWheel"]["sha256"],
            "8a439d25595c6dde486cfbd5a6ed8158e0fe7554ec236491668425e11952898f",
        )

    def test_in_memory_qualification_projection_exposes_complete_source_plan(self):
        candidate = OPTIONAL_RUNTIME_PROFILES[TRANSFORMERS_MAIN_PEFT_RUNTIME_PROFILE_ID]
        qualified = project_optional_runtime_qualification(candidate)
        with (
            mock.patch.object(
                optimization_packages,
                "OPTIONAL_RUNTIME_PROFILES",
                {candidate.id: qualified},
            ),
            mock.patch.object(
                optimization_packages,
                "current_base_binding",
                return_value={"reviewed": True},
            ),
            mock.patch.object(
                optimization_packages,
                "_verified_uv_executable",
                return_value="reviewed-uv",
            ),
        ):
            request = optimization_packages.validate_optional_runtime_install_request(
                candidate.id,
                qualified.spec_digest,
                consent=True,
            )
        self.assertEqual(request["profile"], qualified)
        self.assertEqual(len(request["sourceBuilds"]), 1)
        self.assertEqual(len(request["artifactLocks"]), len(candidate.packages))
        self.assertEqual(
            request["artifactLocks"][0], candidate.source_builds[0]["outputWheel"]
        )

    def test_contract_rejects_mutable_or_executable_supply_chain_fields(self):
        contract, _source = self.make_contract()
        cases = {}

        mutable_source = deepcopy(contract)
        mutable_source["sourceArtifact"]["url"] = (
            "https://github.com/owner/demo/archive/main.tar.gz"
        )
        cases["mutable source"] = mutable_source

        short_commit = deepcopy(contract)
        short_commit["sourceArtifact"]["commit"] = "a" * 39
        cases["short commit"] = short_commit

        noncanonical_commit = deepcopy(contract)
        noncanonical_commit["sourceArtifact"]["commit"] = "A" * 40
        cases["noncanonical commit"] = noncanonical_commit

        executable_dependency = deepcopy(contract)
        executable_dependency["buildDependencies"] = [{"distribution": "setuptools"}]
        cases["executable dependency"] = executable_dependency

        executable_recipe = deepcopy(contract)
        executable_recipe["recipe"] = "setuptools_bdist_wheel_v1"
        cases["executable recipe"] = executable_recipe

        missing_setup = deepcopy(contract)
        missing_setup["sourceFiles"].remove("setup.py")
        cases["missing reviewed setup"] = missing_setup

        header_injection = deepcopy(contract)
        header_injection["wheelMetadata"]["summary"] = "reviewed\nRequires-Dist: evil"
        cases["metadata injection"] = header_injection

        duplicate_requirement = deepcopy(contract)
        duplicate_requirement["wheelMetadata"]["requiresDist"] *= 2
        cases["duplicate requirement"] = duplicate_requirement

        escaping_package_root = deepcopy(contract)
        escaping_package_root["wheelMetadata"]["packageRoot"] = "other"
        cases["package root"] = escaping_package_root

        output_mismatch = deepcopy(contract)
        output_mismatch["outputWheel"]["distribution"] = "other"
        cases["output identity"] = output_mismatch

        extra_field = deepcopy(contract)
        extra_field["mutableRevision"] = "main"
        cases["unexpected field"] = extra_field

        for label, malformed in cases.items():
            with self.subTest(label=label), self.assertRaises(RuntimeError):
                runtime_source_builds.validate_source_build_contract(malformed)


class SourceArchiveSecurityTests(SourceBuildFixture):
    def test_extracts_only_reviewed_regular_inputs_with_deterministic_metadata(self):
        contract, source = self.make_contract()
        destination = self.root / "selected-source"
        receipt = runtime_source_builds.extract_locked_source_tree(
            contract,
            source,
            destination,
            lease=self.lease(),
        )

        self.assertEqual(receipt["selectedFileCount"], 6)
        self.assertRegex(receipt["sourceSealDigest"], r"^sha256:[0-9a-f]{64}$")
        self.assertFalse((destination / "ignored.txt").exists())
        for relative in (
            "LICENSE",
            "README.md",
            "pyproject.toml",
            "setup.py",
            "src/demo/__init__.py",
            "src/demo/module.py",
        ):
            path = destination / relative
            self.assertTrue(path.is_file())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
            self.assertEqual(int(path.stat().st_mtime), 315_532_800)

    def test_archive_rejects_traversal_collisions_links_and_incomplete_selection(self):
        defaults = [
            ("LICENSE", b"license\n", "file", ""),
            ("README.md", b"readme\n", "file", ""),
            ("pyproject.toml", b"build\n", "file", ""),
            ("setup.py", b"setup\n", "file", ""),
            ("src/demo/__init__.py", b"value\n", "file", ""),
        ]
        cases = {
            "traversal": [*defaults, ("../escape.py", b"escape\n", "file", "")],
            "case collision": [
                *defaults,
                ("src/demo/Value.py", b"one\n", "file", ""),
                ("src/demo/value.py", b"two\n", "file", ""),
            ],
            "selected symlink": [
                *defaults,
                ("src/demo/link.py", b"", "symlink", "__init__.py"),
            ],
            "escaping symlink": [
                *defaults,
                ("docs/link", b"", "symlink", "../../../outside"),
            ],
            "hardlink": [
                *defaults,
                ("docs/hard", b"", "hardlink", f"{self.archive_root}/README.md"),
            ],
            "incomplete": [item for item in defaults if item[0] != "README.md"],
        }
        for label, members in cases.items():
            with self.subTest(label=label):
                contract, source = self.make_contract(source_members=members)
                with self.assertRaises(RuntimeError):
                    runtime_source_builds.extract_locked_source_tree(
                        contract,
                        source,
                        self.root / f"selected-{label}",
                        lease=self.lease(),
                    )

    def test_archive_size_and_cancellation_bounds_fail_before_extraction(self):
        contract, source = self.make_contract()
        with (
            mock.patch.object(runtime_source_builds, "MAX_SOURCE_MEMBER_BYTES", 8),
            self.assertRaisesRegex(RuntimeError, "expansion bounds"),
        ):
            runtime_source_builds.extract_locked_source_tree(
                contract,
                source,
                self.root / "bounded-source",
                lease=self.lease(),
            )

        lease = self.lease()
        lease.cancel_event.set()
        with self.assertRaises(runtime_source_builds.OverlayCancelled):
            runtime_source_builds.extract_locked_source_tree(
                contract,
                source,
                self.root / "cancelled-source",
                lease=lease,
            )

    def test_cached_source_archive_is_rehashed_and_never_refetched(self):
        contract, source = self.make_contract()
        opener = mock.Mock()
        with mock.patch.object(runtime_source_builds, "build_opener", return_value=opener):
            retained = runtime_source_builds.cache_locked_source_archive(
                contract,
                self.cache,
                lease=self.lease(),
            )
        self.assertEqual(retained, source)
        opener.open.assert_not_called()

        source.write_bytes(b"replacement")
        with self.assertRaisesRegex(RuntimeError, "catalog identity"):
            runtime_source_builds.cache_locked_source_archive(
                contract,
                self.cache,
                lease=self.lease(),
            )

    def test_nonzero_directory_and_link_sizes_are_rejected_before_type_exit(self):
        contract, source = self.make_contract()

        class FakeArchive:
            def __init__(self, member):
                self.member = member

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def __iter__(self):
                return iter((self.member,))

        for member_type, message in (
            (tarfile.DIRTYPE, "directory declares file data"),
            (tarfile.SYMTYPE, "link declares file data"),
        ):
            member = tarfile.TarInfo(f"{self.archive_root}/ignored")
            member.type = member_type
            member.size = 1
            member.linkname = "README.md"
            with self.subTest(member_type=member_type), mock.patch.object(
                runtime_source_builds.tarfile,
                "open",
                return_value=FakeArchive(member),
            ), self.assertRaisesRegex(RuntimeError, message):
                runtime_source_builds.extract_locked_source_tree(
                    contract,
                    source,
                    self.root / f"nonzero-{member_type.decode('ascii')}",
                    lease=self.lease(),
                )

        member = tarfile.TarInfo(f"{self.archive_root}/ignored")
        member.type = tarfile.DIRTYPE
        member.size = 1
        with mock.patch.object(
            runtime_source_builds,
            "MAX_SOURCE_TOTAL_BYTES",
            0,
        ), mock.patch.object(
            runtime_source_builds.tarfile,
            "open",
            return_value=FakeArchive(member),
        ), self.assertRaisesRegex(RuntimeError, "expansion bounds"):
            runtime_source_builds.extract_locked_source_tree(
                contract,
                source,
                self.root / "nonzero-accounted-before-exit",
                lease=self.lease(),
            )

    def test_cold_source_download_disables_proxies_redirects_and_locks_length(self):
        contract, destination = self.make_contract()
        body = destination.read_bytes()
        destination.unlink()
        source = contract["sourceArtifact"]
        response = FakeSourceResponse(
            body,
            url=source["url"],
            content_length=len(body),
        )
        opener = mock.Mock()
        opener.open.return_value = response
        with mock.patch.object(
            runtime_source_builds,
            "build_opener",
            return_value=opener,
        ) as build:
            acquired = runtime_source_builds.cache_locked_source_archive(
                contract,
                self.cache,
                lease=self.lease(),
            )

        self.assertEqual(acquired, destination)
        self.assertEqual(acquired.read_bytes(), body)
        opener.open.assert_called_once()
        handlers = build.call_args.args
        proxy_handlers = [
            handler
            for handler in handlers
            if isinstance(handler, runtime_source_builds.ProxyHandler)
        ]
        self.assertEqual(len(proxy_handlers), 1)
        self.assertEqual(proxy_handlers[0].proxies, {})
        self.assertTrue(
            any(
                isinstance(handler, runtime_source_builds._RejectRedirects)
                for handler in handlers
            )
        )

    def test_cold_source_download_failures_leave_no_archive_or_partial(self):
        cases = ("redirect", "length", "truncated", "cancelled")
        for case in cases:
            with self.subTest(case=case):
                contract, destination = self.make_contract()
                body = destination.read_bytes()
                destination.unlink()
                source = contract["sourceArtifact"]
                lease = self.lease()
                response_body = body
                response_url = source["url"]
                content_length = len(body)
                on_read = None
                expected_error = RuntimeError
                if case == "redirect":
                    response_url = "https://example.invalid/unreviewed.tar.gz"
                elif case == "length":
                    content_length += 1
                elif case == "truncated":
                    response_body = body[:-1]
                    content_length = None
                else:
                    on_read = lease.cancel_event.set
                    expected_error = runtime_source_builds.OverlayCancelled
                response = FakeSourceResponse(
                    response_body,
                    url=response_url,
                    content_length=content_length,
                    on_read=on_read,
                )
                opener = mock.Mock()
                opener.open.return_value = response
                with mock.patch.object(
                    runtime_source_builds,
                    "build_opener",
                    return_value=opener,
                ), self.assertRaises(expected_error):
                    runtime_source_builds.cache_locked_source_archive(
                        contract,
                        self.cache,
                        lease=lease,
                    )
                self.assertFalse(destination.exists())
                self.assertEqual(
                    list(destination.parent.glob(f".{destination.name}.*.part")),
                    [],
                )

    def test_windows_reparse_markers_reject_archive_cache_and_workspace_paths(self):
        contract, source = self.make_contract(derive_output=True)
        with mock.patch.object(
            runtime_source_builds,
            "_is_reparse_point",
            return_value=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "artifact is unsafe"):
                runtime_source_builds._stream_sha256(
                    source,
                    maximum_bytes=runtime_source_builds.MAX_LOCKED_ARCHIVE_BYTES,
                    lease=self.lease(),
                )
            with self.assertRaisesRegex(RuntimeError, "cache root is unsafe"):
                runtime_source_builds.cache_locked_source_archive(
                    contract,
                    self.cache,
                    lease=self.lease(),
                )
            with self.assertRaisesRegex(RuntimeError, "workspace is unsafe"):
                runtime_source_builds.build_locked_source_wheel(
                    contract,
                    cache_root=self.cache,
                    work_root=self.root / "reparse-work",
                    lease=self.lease(),
                )


class SourceWheelAssemblyTests(SourceBuildFixture):
    def test_no_source_or_build_hook_can_execute(self):
        sentinel = self.root / "source-hook-executed"
        malicious_setup = (
            "from pathlib import Path\n"
            f"Path({str(sentinel)!r}).write_text('executed')\n"
            "raise RuntimeError('executed setup.py')\n"
        ).encode()
        members = [
            ("LICENSE", b"license\n", "file", ""),
            ("README.md", b"readme\n", "file", ""),
            (
                "pyproject.toml",
                b"[build-system]\nrequires=['malicious-backend']\nbuild-backend='malicious.build'\n",
                "file",
                "",
            ),
            ("setup.py", malicious_setup, "file", ""),
            ("src/demo/__init__.py", b"def main(): return 1\n", "file", ""),
        ]
        contract, _source = self.make_contract(
            source_members=members,
            derive_output=True,
        )
        with (
            mock.patch("runpy.run_path", side_effect=AssertionError("runpy used")),
            mock.patch("subprocess.Popen", side_effect=AssertionError("child used")),
            mock.patch("os.system", side_effect=AssertionError("shell used")),
            mock.patch("os.posix_spawn", side_effect=AssertionError("spawn used")),
        ):
            result = runtime_source_builds.build_locked_source_wheel(
                contract,
                cache_root=self.cache,
                work_root=self.root / "build-work",
                lease=self.lease(),
            )

        self.assertFalse(sentinel.exists())
        self.assertEqual(result.artifact, contract["outputWheel"])
        self.assertEqual(result.receipt["buildDependencyCount"], 0)
        self.assertEqual(
            result.receipt["buildRecipe"],
            runtime_source_builds.SOURCE_BUILD_RECIPE,
        )
        self.assertRegex(result.receipt["outputFileSealDigest"], r"^sha256:[0-9a-f]{64}$")
        with zipfile.ZipFile(result.path) as wheel:
            names = wheel.namelist()
            self.assertNotIn("setup.py", names)
            self.assertNotIn("pyproject.toml", names)
            metadata_name = "demo_runtime-1.0.0.dist-info/METADATA"
            metadata = wheel.read(metadata_name).decode("utf-8")
            self.assertIn("Requires-Dist: fixture-dependency>=1\n", metadata)

    def test_wheel_bytes_are_stable_across_source_mode_and_time_changes(self):
        contract, source_archive = self.make_contract()
        first_source = self.root / "first-source"
        second_source = self.root / "second-source"
        runtime_source_builds.extract_locked_source_tree(
            contract,
            source_archive,
            first_source,
            lease=self.lease(),
        )
        runtime_source_builds.extract_locked_source_tree(
            contract,
            source_archive,
            second_source,
            lease=self.lease(),
        )
        for path in second_source.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
                os.utime(path, (1_700_000_000, 1_700_000_000))
        first = self.root / "first.whl"
        second = self.root / "second.whl"
        runtime_source_builds.assemble_locked_pure_python_wheel(
            contract,
            first_source,
            first,
            lease=self.lease(),
        )
        runtime_source_builds.assemble_locked_pure_python_wheel(
            contract,
            second_source,
            second,
            lease=self.lease(),
        )

        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as wheel:
            infos = wheel.infolist()
            names = [info.filename for info in infos]
            self.assertEqual(names[:-1], sorted(names[:-1]))
            self.assertEqual(names[-1], "demo_runtime-1.0.0.dist-info/RECORD")
            for info in infos:
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(info.create_system, 3)
                self.assertEqual(
                    stat.S_IMODE((info.external_attr >> 16) & 0xFFFF),
                    0o644,
                )

    def test_assembled_wheel_passes_the_ordinary_file_seal_validator(self):
        contract, _source = self.make_contract(derive_output=True)
        result = runtime_source_builds.build_locked_source_wheel(
            contract,
            cache_root=self.cache,
            work_root=self.root / "sealed-work",
            lease=self.lease(),
        )
        seal = locked_artifact_file_seal(
            [contract["outputWheel"]],
            self.cache,
            lease=self.lease(),
        )
        self.assertIn("demo/__init__.py", seal)
        self.assertIn("demo_runtime-1.0.0.dist-info/METADATA", seal)
        self.assertEqual(
            hashlib.sha256(result.path.read_bytes()).hexdigest(),
            contract["outputWheel"]["sha256"],
        )

    def test_output_lock_drift_fails_closed(self):
        contract, _source = self.make_contract(derive_output=True)
        contract["outputWheel"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(
            RuntimeError,
            "differs from its reviewed output lock",
        ):
            runtime_source_builds.build_locked_source_wheel(
                contract,
                cache_root=self.cache,
                work_root=self.root / "drift-work",
                lease=self.lease(),
            )

    def test_package_startup_hooks_and_links_fail_closed(self):
        cases = {
            "pth": ("src/demo/activate.pth", b"import evil\n", "file", ""),
            "sitecustomize": (
                "src/demo/sitecustomize.py",
                b"raise RuntimeError\n",
                "file",
                "",
            ),
            "link": ("src/demo/link.py", b"", "symlink", "__init__.py"),
        }
        defaults = [
            ("LICENSE", b"license\n", "file", ""),
            ("README.md", b"readme\n", "file", ""),
            ("pyproject.toml", b"build\n", "file", ""),
            ("setup.py", b"raise RuntimeError\n", "file", ""),
            ("src/demo/__init__.py", b"value = 1\n", "file", ""),
        ]
        for label, malicious in cases.items():
            with self.subTest(label=label):
                contract, source = self.make_contract(
                    source_members=[*defaults, malicious]
                )
                extracted = self.root / f"startup-{label}"
                if label == "link":
                    with self.assertRaises(RuntimeError):
                        runtime_source_builds.extract_locked_source_tree(
                            contract,
                            source,
                            extracted,
                            lease=self.lease(),
                        )
                    continue
                runtime_source_builds.extract_locked_source_tree(
                    contract,
                    source,
                    extracted,
                    lease=self.lease(),
                )
                with self.assertRaisesRegex(RuntimeError, "startup hook"):
                    runtime_source_builds.assemble_locked_pure_python_wheel(
                        contract,
                        extracted,
                        self.root / f"startup-{label}.whl",
                        lease=self.lease(),
                    )

    def test_ordinary_wheel_policy_remains_strict(self):
        artifact = self.make_locked_wheel_with_startup_hook()
        with self.assertRaisesRegex(RuntimeError, "startup hook"):
            locked_artifact_file_seal([artifact], self.cache)


if __name__ == "__main__":
    unittest.main()
