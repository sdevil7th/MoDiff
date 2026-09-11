import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from huggingface_hub.errors import LocalEntryNotFoundError
from modiff.model_artifact_catalog import require_catalog_revision
from modiff.custom_modular_inspection import inspect_installed_custom_modular_contract

from modules.ModularDiffusers.custom_pipeline import (
    _BINDING_CACHE_LIMIT,
    CUSTOM_PIPELINE_IDENTITY_FIELD,
    CUSTOM_PIPELINE_MODEL_TYPE,
    CustomPipelineContractError,
    CustomPipelineExecutionIdentity,
    ReviewedComponentReference,
    _clear_custom_pipeline_binding_cache_for_tests,
    _resolve_installed_official_component_type,
    resolve_custom_pipeline_binding,
    resolve_custom_pipeline_identity,
)
from modules.ModularDiffusers.loaders import (
    AutoModelLoader,
    MODELS_LOADER_IDENTITY_OUTPUTS,
    ModelsLoader,
    _instantiate_reviewed_builtin_pipeline,
    _validate_reviewed_pipeline_index,
    annotate_modular_loader_outputs,
    load_components_strict,
)
from modules.ModularDiffusers.modular_utils import (
    DummyCustomPipeline,
    _get_registry_instance,
    get_model_type_metadata,
    pipeline_class_from_model_type,
    pipeline_class_from_runtime_inputs,
    require_modiff_node_contract,
)
from modules.ModularDiffusers.pipeline_schema import (
    MAX_MODIFF_PIPELINE_CONFIG_BYTES,
    MELLON_PIPELINE_CONFIG_FILENAME,
    MoDiffPipelineConfig,
    inspect_cached_hub_pipeline_sidecar,
)
from modules.ModularDiffusers.route_state import bind_standalone_component_output


requires_transformers = unittest.skipUnless(
    importlib.util.find_spec("transformers"),
    "requires the staged optional Transformers runtime",
)


def _config_bytes(label="Custom fixture", *, dtype="float16", steps=4):
    config = MoDiffPipelineConfig.from_dict(
        {
            "label": label,
            "default_dtype": dtype,
            "node_params": {
                "denoise": {
                    "block_name": "denoise",
                    "params": {
                        "unet": {"label": "Denoiser", "type": "diffusers_auto_model"},
                        "steps": {"label": "Steps", "type": "int", "default": steps},
                    },
                    "input_names": ["steps"],
                    "model_input_names": ["unet"],
                    "output_names": ["latents"],
                }
            },
        }
    )
    return config.to_json_string().encode("utf-8") + b"\n"


def _write_sidecar(directory, raw_bytes=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw_bytes = _config_bytes() if raw_bytes is None else raw_bytes
    (directory / MoDiffPipelineConfig.config_name).write_bytes(raw_bytes)
    index_path = directory / "modular_model_index.json"
    if not index_path.exists():
        index_path.write_text(
            json.dumps(
                {
                    "_class_name": "FluxModularPipeline",
                    "_blocks_class_name": "FluxAutoBlocks",
                }
            ),
            encoding="utf-8",
        )
    return raw_bytes


def _write_reviewed_flux_index(directory, *, repository="owner/pipeline", revision="a" * 40):
    from diffusers import FluxModularPipeline

    document = deepcopy(dict(FluxModularPipeline().config))
    document["_class_name"] = "FluxModularPipeline"
    for name, value in document.items():
        if not isinstance(value, tuple) or len(value) != 3 or not isinstance(value[2], dict):
            continue
        spec = dict(value[2])
        spec["pretrained_model_name_or_path"] = repository
        spec["revision"] = revision
        document[name] = [value[0], value[1], spec]
    raw_bytes = json.dumps(document, sort_keys=True).encode("utf-8")
    Path(directory, "modular_model_index.json").write_bytes(raw_bytes)
    return raw_bytes


class _SymlinkDirEntryProxy:
    """Deterministic DirEntry symlink view for Windows hosts without symlink rights."""

    def __init__(self, entry):
        self._entry = entry
        self.name = entry.name
        self.path = entry.path

    def is_symlink(self):
        return True

    def is_dir(self, *, follow_symlinks=True):
        return False

    def is_file(self, *, follow_symlinks=True):
        return self._entry.is_file(follow_symlinks=follow_symlinks)


@contextmanager
def _working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class VerifiedPipelineSidecarTests(unittest.TestCase):
    def test_inspector_translates_only_the_exact_cached_mellon_sidecar(self):
        revision = "b" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory, "cache", "snapshots", revision)
            snapshot.mkdir(parents=True)
            mellon_path = snapshot / MELLON_PIPELINE_CONFIG_FILENAME
            raw_bytes = _config_bytes("Official Mellon fixture")
            mellon_path.write_bytes(raw_bytes)

            def resolve(_repo_id, *, filename, revision, local_files_only):
                self.assertEqual(revision, "b" * 40)
                self.assertTrue(local_files_only)
                if filename == MoDiffPipelineConfig.config_name:
                    raise LocalEntryNotFoundError("MoDiff sidecar absent")
                self.assertEqual(filename, MELLON_PIPELINE_CONFIG_FILENAME)
                return str(mellon_path)

            with patch(
                "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
                side_effect=resolve,
            ) as hub_download:
                inspected = inspect_cached_hub_pipeline_sidecar("owner/pipeline", revision)

        self.assertEqual(hub_download.call_count, 2)
        self.assertEqual(inspected.filename, MELLON_PIPELINE_CONFIG_FILENAME)
        self.assertEqual(inspected.source_format, "mellon")
        self.assertEqual(inspected.raw_bytes, raw_bytes)
        self.assertEqual(inspected.sha256, hashlib.sha256(raw_bytes).hexdigest())
        self.assertEqual(inspected.config.label, "Official Mellon fixture")

    def test_mellon_inspection_preview_exposes_hierarchy_but_never_remote_code_execution(self):
        revision = "c" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory, "cache", "snapshots", revision)
            snapshot.mkdir(parents=True)
            mellon_path = snapshot / MELLON_PIPELINE_CONFIG_FILENAME
            mellon_path.write_bytes(_config_bytes("Preview fixture", steps=7))
            (snapshot / "prompt_expander.py").write_text(
                "raise RuntimeError('must never import')\n",
                encoding="utf-8",
            )

            def resolve(_repo_id, *, filename, revision, local_files_only):
                if filename == MoDiffPipelineConfig.config_name:
                    raise LocalEntryNotFoundError("MoDiff sidecar absent")
                return str(mellon_path)

            with patch(
                "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
                side_effect=resolve,
            ):
                preview = inspect_installed_custom_modular_contract("owner/pipeline", revision)

        self.assertTrue(preview["sidecar"]["translatedFromMellon"])
        self.assertEqual(preview["definition"]["blockCount"], 1)
        self.assertEqual(preview["definition"]["blocks"][0]["blockName"], "denoise")
        self.assertEqual(preview["admission"]["status"], "preview_only")
        self.assertFalse(preview["admission"]["executable"])
        self.assertTrue(preview["remoteCode"]["repositoryPythonPresent"])
        self.assertTrue(preview["remoteCode"]["requiredForExecution"])
        self.assertIsNone(preview["runtimeNode"])

    def test_only_modiff_sidecar_filename_is_accepted_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            (repository / "mellon_pipeline_config.json").write_bytes(_config_bytes())
            (repository / "pipeline_config.json").write_bytes(_config_bytes())
            with patch("modules.ModularDiffusers.pipeline_schema.hf_hub_download") as hub_download:
                with self.assertRaisesRegex(EnvironmentError, "modiff_pipeline_config.json"):
                    MoDiffPipelineConfig.load_verified(repository, source="local")
            hub_download.assert_not_called()

    def test_local_sidecar_hashes_the_exact_bounded_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_bytes = _write_sidecar(directory)

            verified = MoDiffPipelineConfig.load_verified(directory, source="local")

        self.assertEqual(verified.raw_bytes, raw_bytes)
        self.assertEqual(verified.sha256, hashlib.sha256(raw_bytes).hexdigest())
        self.assertEqual(verified.source, "local")
        self.assertIsNone(verified.revision)
        self.assertEqual(verified.repo_id, str(Path(directory).resolve()))

        with self.assertRaisesRegex(ValueError, "exceeds 4096"):
            MoDiffPipelineConfig.load_verified("x" * 4097, source="local")

    def test_sidecar_rejects_oversize_duplicate_keys_and_non_object_roots(self):
        invalid_documents = {
            "larger": b"{" + (b" " * MAX_MODIFF_PIPELINE_CONFIG_BYTES) + b"}",
            "Duplicate": b'{"node_params":{"denoise":{"block_name":"one","block_name":"two"}}}',
            "JSON object": b"[]",
            "Non-finite": b'{"default_dtype":NaN}',
        }
        for expected_message, raw_bytes in invalid_documents.items():
            with self.subTest(expected_message=expected_message), tempfile.TemporaryDirectory() as directory:
                _write_sidecar(directory, raw_bytes)
                with self.assertRaisesRegex(EnvironmentError, expected_message):
                    MoDiffPipelineConfig.load_verified(directory, source="local")

    def test_local_sidecar_rejects_revision_and_symlink_escape_without_hub_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory, "repository")
            outside = Path(directory, "outside.json")
            repository.mkdir()
            outside.write_bytes(_config_bytes())
            try:
                Path(repository, MoDiffPipelineConfig.config_name).symlink_to(outside)
                containment_context = nullcontext()
            except OSError:
                # Windows may deny symlink creation without Developer Mode.
                # Simulate the two canonical resolutions to exercise the same
                # containment decision deterministically in that environment.
                containment_context = patch.object(
                    Path,
                    "resolve",
                    side_effect=[repository.resolve(), outside.resolve()],
                )

            with patch("modules.ModularDiffusers.pipeline_schema.hf_hub_download") as hub_download:
                with containment_context:
                    with self.assertRaisesRegex(EnvironmentError, "contained"):
                        MoDiffPipelineConfig.load_verified(repository, source="local")
                with self.assertRaisesRegex(ValueError, "must not claim a Hub revision"):
                    MoDiffPipelineConfig.load_verified(repository, source="local", revision="a" * 40)
                hub_download.assert_not_called()

    def test_hub_source_uses_only_exact_cached_commit_and_never_local_shadow(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cached_snapshot = root / "cache" / "snapshots" / revision
            config_path = cached_snapshot / MoDiffPipelineConfig.config_name
            _write_sidecar(cached_snapshot)
            _write_sidecar(root / "owner" / "pipeline", _config_bytes("shadow"))

            with (
                _working_directory(root),
                patch(
                    "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
                    return_value=str(config_path),
                ) as hub_download,
            ):
                verified = MoDiffPipelineConfig.load_verified(
                    "owner/pipeline",
                    source="hub",
                    revision=revision,
                )

        hub_download.assert_called_once_with(
            "owner/pipeline",
            filename=MoDiffPipelineConfig.config_name,
            cache_dir=None,
            local_files_only=True,
            token=None,
            revision=revision,
        )
        self.assertEqual(verified.source, "hub")
        self.assertEqual(verified.repository_path, str(cached_snapshot.absolute()))
        self.assertEqual(verified.config.label, "Custom fixture")

    def test_hub_source_rejects_mutable_revision_and_missing_cache(self):
        with patch("modules.ModularDiffusers.pipeline_schema.hf_hub_download") as hub_download:
            with self.assertRaisesRegex(ValueError, "lowercase 40-character"):
                MoDiffPipelineConfig.load_verified("owner/pipeline", source="hub", revision="main")
            hub_download.assert_not_called()

        with patch(
            "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
            side_effect=LocalEntryNotFoundError("not cached"),
        ) as hub_download:
            with self.assertRaisesRegex(EnvironmentError, "Install that exact revision"):
                MoDiffPipelineConfig.load_verified("owner/pipeline", source="hub", revision="b" * 40)
            self.assertEqual(hub_download.call_count, 2)

    def test_sidecar_rejects_hostile_shapes_callbacks_and_excessive_nesting(self):
        valid = json.loads(_config_bytes())
        invalid_documents = (
            ("non-empty 'node_params'", {}),
            ("non-empty 'node_params'", {"node_params": []}),
            ("non-empty 'node_params'", {"node_params": "denoise"}),
            ("JSON object or null", {**valid, "node_params": {"denoise": []}}),
            ("denoise.params", {**valid, "node_params": {"denoise": {"params": []}}}),
            (
                "block_name",
                {
                    **valid,
                    "node_params": {"denoise": {**valid["node_params"]["denoise"], "block_name": ["denoise"]}},
                },
            ),
            (
                "input_names",
                {
                    **valid,
                    "node_params": {"denoise": {**valid["node_params"]["denoise"], "input_names": "steps"}},
                },
            ),
            ("at most 16 loader component", {**valid, "loader_component_outputs": "image_encoder"}),
            ("invalid or duplicate loader component", {**valid, "loader_component_outputs": [{}]}),
            (
                "invalid or duplicate loader component",
                {**valid, "loader_component_outputs": ["image_encoder", "image_encoder"]},
            ),
            ("at most 64 layer block", {**valid, "layer_block_options": "transformer_blocks"}),
            ("invalid or duplicate layer block", {**valid, "layer_block_options": [{}]}),
            (
                "invalid or duplicate layer block",
                {**valid, "layer_block_options": ["transformer_blocks", "transformer_blocks"]},
            ),
            ("at most 16 guider class", {**valid, "guider_options": "ClassifierFreeGuidance"}),
            ("invalid or duplicate guider class", {**valid, "guider_options": [{}]}),
            (
                "invalid or duplicate guider class",
                {**valid, "guider_options": ["ClassifierFreeGuidance", "ClassifierFreeGuidance"]},
            ),
            ("at most 32 scheduler class", {**valid, "scheduler_options": "EulerDiscreteScheduler"}),
            ("invalid or duplicate scheduler class", {**valid, "scheduler_options": [{}]}),
            (
                "invalid or duplicate scheduler class",
                {**valid, "scheduler_options": ["EulerDiscreteScheduler", "EulerDiscreteScheduler"]},
            ),
            (
                "at most 2 denoise image-latent dimension",
                {**valid, "denoise_image_latent_dimensions": "height"},
            ),
            (
                "invalid or duplicate denoise image-latent dimension",
                {**valid, "denoise_image_latent_dimensions": [{}]},
            ),
            (
                "invalid or duplicate denoise image-latent dimension",
                {**valid, "denoise_image_latent_dimensions": ["height", "height"]},
            ),
            (
                "invalid or duplicate denoise image-latent dimension",
                {**valid, "denoise_image_latent_dimensions": ["depth"]},
            ),
        )
        for expected_message, document in invalid_documents:
            with self.subTest(expected_message=expected_message), tempfile.TemporaryDirectory() as directory:
                _write_sidecar(directory, json.dumps(document).encode("utf-8"))
                with self.assertRaisesRegex(EnvironmentError, expected_message):
                    MoDiffPipelineConfig.load_verified(directory, source="local")

        for callback in (
            "set_filters",
            ["update_node"],
            [{"action": "show", "data": {"true": ["steps"]}}, "update_node"],
            {"action": "exec", "data": "set_filters"},
            {"action": "create", "data": {}},
        ):
            with self.subTest(callback=callback), tempfile.TemporaryDirectory() as directory:
                document = json.loads(_config_bytes())
                document["node_params"]["denoise"]["params"]["steps"]["onChange"] = callback
                _write_sidecar(directory, json.dumps(document).encode("utf-8"))
                with self.assertRaisesRegex(EnvironmentError, "callback|prohibited field action"):
                    MoDiffPipelineConfig.load_verified(directory, source="local")

        with tempfile.TemporaryDirectory() as directory:
            document = json.loads(_config_bytes())
            document["node_params"]["denoise"]["params"]["steps"]["onChange"] = {
                "false": ["steps"],
                "true": [],
            }
            _write_sidecar(directory, json.dumps(document).encode("utf-8"))
            self.assertEqual(
                MoDiffPipelineConfig.load_verified(directory, source="local").config.label,
                "Custom fixture",
            )

        for callback in (
            {"true": ["ghost"]},
            {"action": "value", "target": "modiff_pipeline_identity"},
            {"action": "signal", "target": "steps"},
        ):
            with self.subTest(callback=callback), tempfile.TemporaryDirectory() as directory:
                document = json.loads(_config_bytes())
                document["node_params"]["denoise"]["params"]["steps"]["onChange"] = callback
                _write_sidecar(directory, json.dumps(document).encode("utf-8"))
                with self.assertRaisesRegex(EnvironmentError, "unknown field target|input or output"):
                    MoDiffPipelineConfig.load_verified(directory, source="local")

        for reserved_name in ("__proto__", "prototype", "constructor"):
            with self.subTest(reserved_name=reserved_name), tempfile.TemporaryDirectory() as directory:
                document = json.loads(_config_bytes())
                document["node_params"]["denoise"]["params"][reserved_name] = {"type": "string"}
                _write_sidecar(directory, json.dumps(document).encode("utf-8"))
                with self.assertRaisesRegex(EnvironmentError, "invalid parameter name"):
                    MoDiffPipelineConfig.load_verified(directory, source="local")

        nested = "leaf"
        for _index in range(32):
            nested = {"nested": nested}
        document = json.loads(_config_bytes())
        document["future_metadata"] = nested
        with tempfile.TemporaryDirectory() as directory:
            _write_sidecar(directory, json.dumps(document).encode("utf-8"))
            with self.assertRaisesRegex(EnvironmentError, "structural depth"):
                MoDiffPipelineConfig.load_verified(directory, source="local")

    def test_manifest_detects_loader_metadata_and_python_drift_but_not_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            _write_sidecar(repository)
            (repository / "pipeline_a.py").write_text("class PipelineA: pass\n", encoding="utf-8")
            (repository / "pipeline_b.py").write_text("class PipelineB: pass\n", encoding="utf-8")
            config_path = repository / "config.json"
            config_path.write_text(
                json.dumps({"auto_map": {"AutoPipeline": "pipeline_a.PipelineA"}}), encoding="utf-8"
            )
            weights_path = repository / "weights.safetensors"
            weights_path.write_bytes(b"weights-a")

            original = MoDiffPipelineConfig.load_verified(repository, source="local")
            weights_path.write_bytes(b"weights-b")
            weights_changed = MoDiffPipelineConfig.load_verified(repository, source="local")
            self.assertEqual(
                weights_changed.executable_manifest_sha256,
                original.executable_manifest_sha256,
                "P0 manifest intentionally does not prove model-weight bytes; full artifact proof is P0.4.",
            )

            config_path.write_text(
                json.dumps({"auto_map": {"AutoPipeline": "pipeline_b.PipelineB"}}), encoding="utf-8"
            )
            config_changed = MoDiffPipelineConfig.load_verified(repository, source="local")
            self.assertNotEqual(config_changed.executable_manifest_sha256, original.executable_manifest_sha256)

            (repository / "pipeline_b.py").unlink()
            python_removed = MoDiffPipelineConfig.load_verified(repository, source="local")
            self.assertNotEqual(
                python_removed.executable_manifest_sha256,
                config_changed.executable_manifest_sha256,
            )

    def test_manifest_enforces_file_count_size_and_local_symlink_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            _write_sidecar(repository)
            (repository / "a.py").write_bytes(b"123456789")
            (repository / "b.py").write_bytes(b"pass\n")
            with patch("modules.ModularDiffusers.pipeline_schema.MAX_LOCAL_EXECUTABLE_MANIFEST_FILES", 1):
                with self.assertRaisesRegex(EnvironmentError, "file executable-manifest limit"):
                    MoDiffPipelineConfig.load_verified(repository, source="local")
            with patch("modules.ModularDiffusers.pipeline_schema.MAX_LOCAL_EXECUTABLE_FILE_BYTES", 8):
                with self.assertRaisesRegex(EnvironmentError, "exceeds the 8-byte limit"):
                    MoDiffPipelineConfig.load_verified(repository, source="local")

            real_scandir = os.scandir

            def linked_scandir(path):
                with real_scandir(path) as scanner:
                    entries = list(scanner)
                if Path(path) == repository:
                    return [_SymlinkDirEntryProxy(entry) if entry.name == "b.py" else entry for entry in entries]
                return entries

            with patch("modules.ModularDiffusers.pipeline_schema.os.scandir", side_effect=linked_scandir):
                with self.assertRaisesRegex(EnvironmentError, "does not allow linked file"):
                    MoDiffPipelineConfig.load_verified(repository, source="local")

    def test_hub_manifest_allows_only_this_repository_blob_symlinks(self):
        revision = "c" * 40
        with tempfile.TemporaryDirectory() as directory:
            repo_cache = Path(directory, "models--owner--pipeline").resolve()
            blobs = repo_cache / "blobs"
            snapshot = repo_cache / "snapshots" / revision
            blobs.mkdir(parents=True)
            snapshot.mkdir(parents=True)
            sidecar_blob = blobs / "sidecar"
            sidecar_blob.write_bytes(_config_bytes())
            python_blob = blobs / "python"
            python_blob.write_text("class CachedPipeline: pass\n", encoding="utf-8")
            sidecar_path = snapshot / MoDiffPipelineConfig.config_name
            python_path = snapshot / "pipeline.py"
            sidecar_path.write_bytes(sidecar_blob.read_bytes())
            python_path.write_bytes(python_blob.read_bytes())
            real_scandir = os.scandir
            real_resolve = Path.resolve
            real_is_symlink = Path.is_symlink

            def linked_scandir(path):
                with real_scandir(path) as scanner:
                    entries = list(scanner)
                if Path(path) == snapshot:
                    return [
                        _SymlinkDirEntryProxy(entry) if entry.name == "pipeline.py" else entry for entry in entries
                    ]
                return entries

            def linked_resolve(path, strict=False):
                candidate = Path(path)
                if candidate == sidecar_path:
                    return sidecar_blob
                if candidate == python_path:
                    return python_blob
                return real_resolve(candidate, strict=strict)

            def linked_is_symlink(path):
                candidate = Path(path)
                return candidate == sidecar_path or real_is_symlink(candidate)

            with (
                patch(
                    "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
                    return_value=str(sidecar_path),
                ),
                patch("modules.ModularDiffusers.pipeline_schema.os.scandir", side_effect=linked_scandir),
                patch.object(Path, "resolve", new=linked_resolve),
                patch.object(Path, "is_symlink", new=linked_is_symlink),
            ):
                verified = MoDiffPipelineConfig.load_verified("owner/pipeline", source="hub", revision=revision)
            self.assertRegex(verified.executable_manifest_sha256, r"^[0-9a-f]{64}$")

            outside = Path(directory, "outside.py").resolve()
            outside.write_text("raise RuntimeError\n", encoding="utf-8")

            def escaping_resolve(path, strict=False):
                candidate = Path(path)
                if candidate == sidecar_path:
                    return sidecar_blob
                if candidate == python_path:
                    return outside
                return real_resolve(candidate, strict=strict)

            with (
                patch(
                    "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
                    return_value=str(sidecar_path),
                ),
                patch("modules.ModularDiffusers.pipeline_schema.os.scandir", side_effect=linked_scandir),
                patch.object(Path, "resolve", new=escaping_resolve),
                patch.object(Path, "is_symlink", new=linked_is_symlink),
            ):
                with self.assertRaisesRegex(EnvironmentError, "blobs directory"):
                    MoDiffPipelineConfig.load_verified("owner/pipeline", source="hub", revision=revision)

    def test_hub_snapshot_rejects_linked_snapshot_ancestor(self):
        revision = "d" * 40
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory, "cache", "snapshots", revision)
            config_path = snapshot / MoDiffPipelineConfig.config_name
            _write_sidecar(snapshot)
            with (
                patch(
                    "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
                    return_value=str(config_path),
                ),
                patch(
                    "modules.ModularDiffusers.pipeline_schema._is_linked_directory",
                    side_effect=lambda path: path == snapshot,
                ),
            ):
                with self.assertRaisesRegex(EnvironmentError, "symlink or junction"):
                    MoDiffPipelineConfig.load_verified("owner/pipeline", source="hub", revision=revision)


class CustomPipelineBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.pipeline_a = root / "pipeline-a"
        self.pipeline_b = root / "pipeline-b"
        _write_sidecar(self.pipeline_a, _config_bytes("Pipeline A", steps=4))
        _write_sidecar(self.pipeline_b, _config_bytes("Pipeline B", steps=8))
        _clear_custom_pipeline_binding_cache_for_tests()

    def tearDown(self):
        _clear_custom_pipeline_binding_cache_for_tests()
        self.temporary_directory.cleanup()

    def _resolve(self, repository, *, expected_identity=None, allow_selector_change=False, trust=False):
        return resolve_custom_pipeline_binding(
            source="local",
            repo_id=str(repository),
            revision=None,
            trust_remote_code=trust,
            expected_identity=expected_identity,
            allow_selector_change=allow_selector_change,
        )

    @staticmethod
    def _component_payload(binding):
        return {
            "model_type": CUSTOM_PIPELINE_MODEL_TYPE,
            CUSTOM_PIPELINE_IDENTITY_FIELD: binding.identity.to_dict(),
        }

    def test_installed_official_alias_resolves_to_its_canonical_runtime_class(self):
        class UnifiedTokenizer:
            pass

        UnifiedTokenizer.__module__ = "transformers.models.qwen2.tokenization_qwen2"
        reference = ReviewedComponentReference(
            name="tokenizer",
            library="transformers",
            class_name="Qwen2TokenizerFast",
            repository="owner/component",
            revision="a" * 40,
            subfolder="tokenizer",
            variant=None,
        )
        with patch(
            "modules.ModularDiffusers.custom_pipeline.importlib.import_module",
            return_value=SimpleNamespace(Qwen2TokenizerFast=UnifiedTokenizer),
        ):
            self.assertIs(_resolve_installed_official_component_type(reference), UnifiedTokenizer)

        UnifiedTokenizer.__module__ = "attacker_package.payload"
        with (
            patch(
                "modules.ModularDiffusers.custom_pipeline.importlib.import_module",
                return_value=SimpleNamespace(Qwen2TokenizerFast=UnifiedTokenizer),
            ),
            self.assertRaisesRegex(CustomPipelineContractError, "installed official"),
        ):
            _resolve_installed_official_component_type(reference)

    def test_binding_is_callable_immutable_and_config_reads_are_isolated(self):
        binding = self._resolve(self.pipeline_a)
        first_config = binding.pipeline_config()
        first_config.node_params["denoise"]["params"]["steps"]["default"] = 99

        self.assertEqual(binding.pipeline_config().node_params["denoise"]["params"]["steps"]["default"], 4)
        self.assertEqual(binding.__name__, CUSTOM_PIPELINE_MODEL_TYPE)
        self.assertEqual(binding.execution_status, "reviewed_official_components")
        with patch("diffusers.ModularPipeline.from_pretrained") as loader:
            with self.assertRaisesRegex(ValueError, "mutable.*contract-preview only"):
                binding()
        loader.assert_not_called()

    def test_custom_a_standard_b_custom_a_has_no_global_registry_mutation(self):
        registry = _get_registry_instance()
        before = registry.get_all()
        binding_a = self._resolve(self.pipeline_a)

        recovered_a = pipeline_class_from_runtime_inputs(None, self._component_payload(binding_a))
        standard_b = pipeline_class_from_model_type("FluxModularPipeline")
        recovered_a_again = pipeline_class_from_runtime_inputs(None, self._component_payload(binding_a))

        self.assertIs(recovered_a, binding_a)
        self.assertEqual(standard_b.__name__, "FluxModularPipeline")
        self.assertIs(recovered_a_again, binding_a)
        self.assertEqual(registry.get_all(), before)
        self.assertNotIn("repo_id", DummyCustomPipeline.__dict__)

    def test_interleaved_custom_bindings_keep_distinct_contracts_and_identities(self):
        binding_a = self._resolve(self.pipeline_a)
        binding_b = self._resolve(self.pipeline_b)

        with ThreadPoolExecutor(max_workers=4) as executor:
            recovered = list(
                executor.map(
                    lambda binding: pipeline_class_from_runtime_inputs(None, self._component_payload(binding)),
                    [binding_a, binding_b, binding_a, binding_b],
                )
            )

        self.assertEqual(
            [item.identity for item in recovered],
            [
                binding_a.identity,
                binding_b.identity,
                binding_a.identity,
                binding_b.identity,
            ],
        )
        self.assertEqual(recovered[0].pipeline_config().label, "Pipeline A")
        self.assertEqual(recovered[1].pipeline_config().label, "Pipeline B")
        with self.assertRaisesRegex(ValueError, "different custom contract identity"):
            pipeline_class_from_runtime_inputs(binding_a, self._component_payload(binding_b))

    def test_restart_recovery_is_local_only_and_does_not_depend_on_cache_residency(self):
        original = self._resolve(self.pipeline_a)
        identity_value = original.identity.to_dict()
        _clear_custom_pipeline_binding_cache_for_tests()

        with patch("modules.ModularDiffusers.pipeline_schema.hf_hub_download") as hub_download:
            recovered = resolve_custom_pipeline_identity(identity_value)

        self.assertIsNot(recovered, original)
        self.assertEqual(recovered.identity, original.identity)
        self.assertEqual(recovered.pipeline_config().label, "Pipeline A")
        hub_download.assert_not_called()

    def test_same_selector_sidecar_drift_fails_until_explicit_refresh(self):
        original = self._resolve(self.pipeline_a)
        _write_sidecar(self.pipeline_a, _config_bytes("Pipeline A revised", steps=12))

        with self.assertRaisesRegex(ValueError, "no longer matches the persisted"):
            resolve_custom_pipeline_identity(original.identity.to_dict())
        with patch("diffusers.ModularPipeline.from_pretrained") as pipeline_loader:
            with self.assertRaisesRegex(ValueError, "no longer matches the persisted"):
                original()
            pipeline_loader.assert_not_called()

        refreshed = self._resolve(self.pipeline_a)
        self.assertNotEqual(refreshed.identity, original.identity)
        self.assertEqual(refreshed.pipeline_config().label, "Pipeline A revised")

    def test_executable_manifest_drift_rejects_old_identity_before_upstream(self):
        (self.pipeline_a / "pipeline_a.py").write_text("class PipelineA: pass\n", encoding="utf-8")
        (self.pipeline_a / "pipeline_b.py").write_text("class PipelineB: pass\n", encoding="utf-8")
        index_path = self.pipeline_a / "modular_model_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "_class_name": "FluxModularPipeline",
                    "_blocks_class_name": "FluxAutoBlocks",
                    "_metadata_note": "pipeline-a",
                }
            ),
            encoding="utf-8",
        )
        original = self._resolve(self.pipeline_a)
        index_path.write_text(
            json.dumps(
                {
                    "_class_name": "FluxModularPipeline",
                    "_blocks_class_name": "FluxAutoBlocks",
                    "_metadata_note": "pipeline-b",
                }
            ),
            encoding="utf-8",
        )

        with patch("diffusers.ModularPipeline.from_pretrained") as pipeline_loader:
            with self.assertRaisesRegex(ValueError, "executable metadata no longer matches"):
                resolve_custom_pipeline_identity(original.identity.to_dict())
            with self.assertRaisesRegex(ValueError, "executable metadata no longer matches"):
                original()
        pipeline_loader.assert_not_called()

    def test_cached_hub_python_drift_rejects_old_identity_before_upstream(self):
        revision = "e" * 40
        repo_cache = Path(self.temporary_directory.name, "models--owner--pipeline")
        snapshot = repo_cache / "snapshots" / revision
        sidecar_path = snapshot / MoDiffPipelineConfig.config_name
        _write_sidecar(snapshot)
        python_path = snapshot / "pipeline.py"
        python_path.write_text("class PipelineA: pass\n", encoding="utf-8")
        with patch(
            "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
            return_value=str(sidecar_path),
        ):
            original = resolve_custom_pipeline_binding(
                source="hub",
                repo_id="owner/pipeline",
                revision=revision,
                trust_remote_code=False,
            )

        python_path.write_text("class PipelineB: pass\n", encoding="utf-8")
        with (
            patch(
                "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
                return_value=str(sidecar_path),
            ),
            patch("diffusers.ModularPipeline.from_pretrained") as pipeline_loader,
        ):
            with self.assertRaisesRegex(ValueError, "executable metadata no longer matches"):
                resolve_custom_pipeline_identity(original.identity.to_dict())
        pipeline_loader.assert_not_called()

    def test_identity_parser_rejects_tampering_and_malformed_container_fields(self):
        identity = self._resolve(self.pipeline_a).identity.to_dict()
        invalid_fields = {
            "source": [],
            "repo_id": {},
            "revision": "a" * 40,
            "trust_remote_code": "false",
            "config_sha256": [],
            "executable_manifest_sha256": {},
            "execution_id": "sha256:" + ("0" * 64),
        }
        for field_name, invalid_value in invalid_fields.items():
            with self.subTest(field_name=field_name):
                malformed = deepcopy(identity)
                malformed[field_name] = invalid_value
                with self.assertRaises((TypeError, ValueError)):
                    CustomPipelineExecutionIdentity.from_value(malformed)

        malformed = {**identity, "unknown": True}
        with self.assertRaisesRegex(ValueError, "unknown unknown"):
            CustomPipelineExecutionIdentity.from_value(malformed)

        oversized_repository = deepcopy(identity)
        oversized_repository["repo_id"] = "x" * 4097
        with self.assertRaisesRegex(ValueError, "4096-character"):
            CustomPipelineExecutionIdentity.from_value(oversized_repository)

        remote_code = deepcopy(identity)
        remote_code["trust_remote_code"] = True
        with self.assertRaisesRegex(ValueError, "repository code is disabled"):
            CustomPipelineExecutionIdentity.from_value(remote_code)

    def test_runtime_rejects_missing_and_mixed_custom_identities(self):
        binding_a = self._resolve(self.pipeline_a)
        binding_b = self._resolve(self.pipeline_b)
        with self.assertRaisesRegex(ValueError, "backend-issued contract identity"):
            pipeline_class_from_runtime_inputs(None, {"model_type": CUSTOM_PIPELINE_MODEL_TYPE})
        with self.assertRaisesRegex(ValueError, "incompatible contract identities"):
            pipeline_class_from_runtime_inputs(
                None,
                self._component_payload(binding_a),
                self._component_payload(binding_b),
            )
        malformed = self._component_payload(binding_a)
        malformed[CUSTOM_PIPELINE_IDENTITY_FIELD]["source"] = []
        with self.assertRaisesRegex(ValueError, "source must be exactly"):
            pipeline_class_from_runtime_inputs(None, malformed)

    def test_concurrent_same_identity_resolution_converges_without_registry_exposure(self):
        initial_registry = _get_registry_instance().get_all()
        with ThreadPoolExecutor(max_workers=12) as executor:
            bindings = list(executor.map(lambda _index: self._resolve(self.pipeline_a), range(32)))

        self.assertTrue(all(binding is bindings[0] for binding in bindings))
        self.assertEqual(_get_registry_instance().get_all(), initial_registry)

    def test_binding_cache_is_bounded_and_eviction_does_not_break_recovery(self):
        first = self._resolve(self.pipeline_a)
        root = Path(self.temporary_directory.name)
        for index in range(_BINDING_CACHE_LIMIT):
            repository = root / f"eviction-{index}"
            _write_sidecar(repository, _config_bytes(f"Eviction {index}"))
            self._resolve(repository)

        recovered = resolve_custom_pipeline_identity(first.identity.to_dict())
        self.assertIsNot(recovered, first)
        self.assertEqual(recovered.identity, first.identity)

    def test_ui_contract_lookup_does_not_instantiate_custom_code_and_local_execution_requires_snapshot(self):
        binding = self._resolve(self.pipeline_a)
        runtime_block = object()
        fake_pipeline = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={"denoise": runtime_block}))

        with patch("diffusers.ModularPipeline.from_pretrained", return_value=fake_pipeline) as loader:
            blocks, node_config = require_modiff_node_contract(binding, "denoise", resolve_blocks=False)
            loader.assert_not_called()
            self.assertIsNone(blocks)
            self.assertEqual(node_config["params"]["steps"]["default"], 4)

            with self.assertRaisesRegex(ValueError, "mutable.*contract-preview only"):
                require_modiff_node_contract(binding, "denoise")

        loader.assert_not_called()

    def test_attacker_component_library_is_never_imported_by_binding(self):
        (self.pipeline_a / "modular_model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "FluxModularPipeline",
                    "_blocks_class_name": "FluxAutoBlocks",
                    "transformer": [
                        None,
                        None,
                        {
                            "type_hint": ["attacker_package", "Payload"],
                            "pretrained_model_name_or_path": "owner/payload",
                            "revision": "a" * 40,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        real_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name == "attacker_package":
                raise AssertionError("repository-controlled package import was reached")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=guarded_import),
            patch("diffusers.ModularPipeline.from_pretrained") as pipeline_loader,
        ):
            with self.assertRaisesRegex(ValueError, "approved official"):
                self._resolve(self.pipeline_a)
        pipeline_loader.assert_not_called()

    def test_trust_true_binding_is_rejected_before_sidecar_or_upstream_load(self):
        with (
            patch("modules.ModularDiffusers.custom_pipeline.PipelineConfig.load_verified") as sidecar_loader,
            patch("diffusers.ModularPipeline.from_pretrained") as pipeline_loader,
        ):
            with self.assertRaisesRegex(ValueError, "repository code is disabled"):
                resolve_custom_pipeline_binding(
                    source="hub",
                    repo_id="owner/pipeline",
                    revision="f" * 40,
                    trust_remote_code=True,
                )
        sidecar_loader.assert_not_called()
        pipeline_loader.assert_not_called()

    @requires_transformers
    def test_exact_hub_contract_constructs_installed_blocks_from_private_metadata_snapshot(self):
        revision = "a" * 40
        repo_cache = Path(self.temporary_directory.name, "models--owner--pipeline")
        snapshot = repo_cache / "snapshots" / revision
        sidecar_path = snapshot / MoDiffPipelineConfig.config_name
        _write_sidecar(snapshot)
        expected_index = _write_reviewed_flux_index(snapshot, revision=revision)

        with patch(
            "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
            return_value=str(sidecar_path),
        ):
            binding = resolve_custom_pipeline_binding(
                source="hub",
                repo_id="owner/pipeline",
                revision=revision,
                trust_remote_code=False,
            )
            pipeline = binding.instantiate()

        execution_snapshot = pipeline._modiff_custom_execution_snapshot
        self.assertNotEqual(execution_snapshot.path, snapshot)
        self.assertEqual(
            (execution_snapshot.path / "modular_model_index.json").read_bytes(),
            expected_index,
        )
        self.assertEqual(
            pipeline._modiff_custom_execution_identity,
            binding.identity.to_dict(),
        )
        self.assertTrue(pipeline.pretrained_component_names)
        for component_name in pipeline.pretrained_component_names:
            spec = pipeline.get_component_spec(component_name)
            self.assertEqual(spec.pretrained_model_name_or_path, "owner/pipeline")
            self.assertEqual(spec.revision, revision)
        snapshot_root = execution_snapshot.root
        execution_snapshot.cleanup()
        self.assertFalse(snapshot_root.exists())

    @requires_transformers
    def test_source_mutation_after_final_revalidation_cannot_change_constructed_metadata(self):
        revision = "b" * 40
        repo_cache = Path(self.temporary_directory.name, "models--owner--race")
        snapshot = repo_cache / "snapshots" / revision
        sidecar_path = snapshot / MoDiffPipelineConfig.config_name
        _write_sidecar(snapshot)
        expected_index = _write_reviewed_flux_index(
            snapshot,
            repository="owner/race",
            revision=revision,
        )
        index_path = snapshot / "modular_model_index.json"

        with patch(
            "modules.ModularDiffusers.pipeline_schema.hf_hub_download",
            return_value=str(sidecar_path),
        ):
            binding = resolve_custom_pipeline_binding(
                source="hub",
                repo_id="owner/race",
                revision=revision,
                trust_remote_code=False,
            )

            def mutate_after_revalidation(_identity):
                index_path.write_text('{"auto_map":{"ModularPipelineBlocks":"payload.Code"}}', encoding="utf-8")
                return binding

            with patch(
                "modules.ModularDiffusers.custom_pipeline.resolve_custom_pipeline_identity",
                side_effect=mutate_after_revalidation,
            ):
                pipeline = binding.instantiate()

        execution_snapshot = pipeline._modiff_custom_execution_snapshot
        self.assertEqual(
            (execution_snapshot.path / "modular_model_index.json").read_bytes(),
            expected_index,
        )
        self.assertNotIn("auto_map", dict(pipeline.config))
        execution_snapshot.cleanup()

    def test_unpinned_auxiliary_and_repository_python_fail_before_installed_dispatch(self):
        document = {
            "_class_name": "FluxModularPipeline",
            "_blocks_class_name": "FluxAutoBlocks",
            "transformer": [
                None,
                None,
                {
                    "type_hint": ["diffusers", "FluxTransformer2DModel"],
                    "pretrained_model_name_or_path": "owner/auxiliary",
                    "revision": None,
                },
            ],
        }
        (self.pipeline_a / "modular_model_index.json").write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lacks an exact") as unpinned:
            self._resolve(self.pipeline_a)
        self.assertEqual(unpinned.exception.modiff_error_code, "custom_pipeline_unpinned_auxiliary")

        document["auto_map"] = {"ModularPipelineBlocks": "pipeline.Payload"}
        (self.pipeline_a / "modular_model_index.json").write_text(json.dumps(document), encoding="utf-8")
        with patch("builtins.__import__") as importer:
            with self.assertRaisesRegex(ValueError, "task-scoped operator authorization") as remote_code:
                self._resolve(self.pipeline_a)
        self.assertEqual(remote_code.exception.modiff_error_code, "custom_pipeline_authorization_required")
        importer.assert_not_called()

    @requires_transformers
    def test_local_auxiliary_component_path_is_rejected(self):
        _write_reviewed_flux_index(self.pipeline_a)
        document = json.loads((self.pipeline_a / "modular_model_index.json").read_text(encoding="utf-8"))
        document["text_encoder"][2]["pretrained_model_name_or_path"] = "../../attacker-component"
        document["text_encoder"][2]["revision"] = "b" * 40
        (self.pipeline_a / "modular_model_index.json").write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(CustomPipelineContractError) as context:
            self._resolve(self.pipeline_a)
        self.assertEqual(context.exception.modiff_error_code, "custom_pipeline_unpinned_auxiliary")


class ModelsLoaderCustomIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repository = Path(self.temporary_directory.name, "pipeline")
        _write_sidecar(self.repository)
        _clear_custom_pipeline_binding_cache_for_tests()

    def tearDown(self):
        _clear_custom_pipeline_binding_cache_for_tests()
        self.temporary_directory.cleanup()

    def _values(self, identity=None):
        return {
            "model_type": CUSTOM_PIPELINE_MODEL_TYPE,
            "repo_id": {"source": "local", "value": str(self.repository)},
            "revision": "",
            "trust_remote_code": False,
            CUSTOM_PIPELINE_IDENTITY_FIELD: identity,
        }

    @staticmethod
    def _capture_node_messages(node):
        node.set_field_value = Mock()
        node.set_field_visibility = Mock()
        node.set_field_params = Mock()

    def test_loader_declares_object_identity_and_refresh_actions(self):
        self.assertEqual(ModelsLoader.params[CUSTOM_PIPELINE_IDENTITY_FIELD]["type"], "object")
        self.assertTrue(ModelsLoader.params[CUSTOM_PIPELINE_IDENTITY_FIELD]["hidden"])
        self.assertEqual(ModelsLoader.params["repo_id"]["onChange"], "refresh_pipeline_identity")
        self.assertEqual(ModelsLoader.params["revision"]["onChange"], "refresh_pipeline_identity")
        self.assertEqual(ModelsLoader.params["trust_remote_code"]["onChange"], "refresh_pipeline_identity")
        self.assertEqual(
            get_model_type_metadata(CUSTOM_PIPELINE_MODEL_TYPE)["execution_status"],
            "reviewed_official_components",
        )

    def test_field_action_persists_identity_and_publishes_structured_output_signals(self):
        node = ModelsLoader("identity-field-action")
        self._capture_node_messages(node)

        node.refresh_pipeline_identity(self._values(), {"key": "repo_id"})

        persisted = node.set_field_value.call_args.args[0][CUSTOM_PIPELINE_IDENTITY_FIELD]
        parsed = CustomPipelineExecutionIdentity.from_value(persisted)
        self.assertEqual(parsed.source, "local")
        signal_calls = {
            call.args[0]: call.args[1]["signal"]
            for call in node.set_field_params.call_args_list
            if "signal" in call.args[1]
        }
        self.assertEqual(set(signal_calls), set(MODELS_LOADER_IDENTITY_OUTPUTS))
        for signal in signal_calls.values():
            self.assertEqual(signal["direction"], "output")
            self.assertEqual(signal["origin"], CUSTOM_PIPELINE_IDENTITY_FIELD)
            self.assertEqual(signal["value"], persisted)

    def test_field_action_clears_stale_hub_revision_when_source_is_local(self):
        node = ModelsLoader("identity-local-revision-normalization")
        self._capture_node_messages(node)
        values = self._values()
        values["revision"] = "a" * 40

        node.refresh_pipeline_identity(values, {"key": "repo_id"})

        published_values = node.set_field_value.call_args.args[0]
        self.assertEqual(published_values["revision"], "")
        identity = CustomPipelineExecutionIdentity.from_value(published_values[CUSTOM_PIPELINE_IDENTITY_FIELD])
        self.assertIsNone(identity.revision)

    def test_trust_true_field_actions_cannot_mint_or_advertise_a_contract(self):
        for ref_key in ("trust_remote_code", "refresh_pipeline_identity_button"):
            with self.subTest(ref_key=ref_key):
                node = ModelsLoader(f"identity-trust-disabled-{ref_key}")
                self._capture_node_messages(node)
                values = self._values()
                values["trust_remote_code"] = True
                with patch("modules.ModularDiffusers.loaders.resolve_custom_pipeline_binding") as resolver:
                    with self.assertRaisesRegex(ValueError, "(?i)repository code is disabled"):
                        node.refresh_pipeline_identity(values, {"key": ref_key})
                resolver.assert_not_called()
                self.assertIsNone(node.set_field_value.call_args.args[0][CUSTOM_PIPELINE_IDENTITY_FIELD])
                signals = [
                    call.args[1]["signal"]["value"]
                    for call in node.set_field_params.call_args_list
                    if "signal" in call.args[1]
                ]
                self.assertEqual(signals, [""] * len(MODELS_LOADER_IDENTITY_OUTPUTS))

    def test_same_selector_drift_clears_signal_and_requires_explicit_refresh(self):
        node = ModelsLoader("identity-explicit-refresh")
        self._capture_node_messages(node)
        node.refresh_pipeline_identity(self._values(), {"key": "repo_id"})
        old_identity = node.set_field_value.call_args.args[0][CUSTOM_PIPELINE_IDENTITY_FIELD]
        node.set_field_value.reset_mock()
        node.set_field_params.reset_mock()
        _write_sidecar(self.repository, _config_bytes("reviewed replacement", steps=6))

        with self.assertRaisesRegex(ValueError, "explicitly refresh"):
            node.refresh_pipeline_identity(self._values(old_identity), {"key": "revision"})
        node.set_field_value.assert_not_called()
        stale_signals = [
            call.args[1]["signal"]["value"]
            for call in node.set_field_params.call_args_list
            if "signal" in call.args[1]
        ]
        self.assertEqual(stale_signals, [""] * len(MODELS_LOADER_IDENTITY_OUTPUTS))

        node.set_field_params.reset_mock()
        node.refresh_pipeline_identity(
            self._values(old_identity),
            {"key": "refresh_pipeline_identity_button"},
        )
        new_identity = node.set_field_value.call_args.args[0][CUSTOM_PIPELINE_IDENTITY_FIELD]
        self.assertNotEqual(new_identity, old_identity)

    def test_generation_guard_prevents_stale_field_action_publication(self):
        node = ModelsLoader("identity-generation")
        self._capture_node_messages(node)
        binding = resolve_custom_pipeline_binding(
            source="local",
            repo_id=str(self.repository),
            revision=None,
            trust_remote_code=False,
        )
        entered = threading.Event()
        release = threading.Event()

        def slow_resolve(**_kwargs):
            entered.set()
            release.wait(timeout=5)
            return binding

        with patch("modules.ModularDiffusers.loaders.resolve_custom_pipeline_binding", side_effect=slow_resolve):
            worker = threading.Thread(
                target=node.refresh_pipeline_identity,
                args=(self._values(), {"key": "repo_id"}),
            )
            worker.start()
            self.assertTrue(entered.wait(timeout=2))
            node.refresh_pipeline_identity(
                {"model_type": "FluxModularPipeline"},
                {"key": "model_type"},
            )
            release.set()
            worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(node.set_field_value.call_count, 1)
        self.assertIsNone(node.set_field_value.call_args.args[0][CUSTOM_PIPELINE_IDENTITY_FIELD])
        signals = [
            call.args[1]["signal"]["value"]
            for call in node.set_field_params.call_args_list
            if "signal" in call.args[1]
        ]
        self.assertEqual(signals, ["FluxModularPipeline"] * len(MODELS_LOADER_IDENTITY_OUTPUTS))

    def test_generation_guard_suppresses_failure_from_obsolete_field_action(self):
        node = ModelsLoader("identity-generation-failure")
        self._capture_node_messages(node)
        entered = threading.Event()
        release = threading.Event()
        worker_errors = []

        def slow_failure(**_kwargs):
            entered.set()
            release.wait(timeout=5)
            raise ValueError("obsolete custom selection failed")

        def run_obsolete_action():
            try:
                node.refresh_pipeline_identity(self._values(), {"key": "repo_id"})
            except Exception as error:  # pragma: no cover - asserted through worker_errors
                worker_errors.append(error)

        with patch("modules.ModularDiffusers.loaders.resolve_custom_pipeline_binding", side_effect=slow_failure):
            worker = threading.Thread(target=run_obsolete_action)
            worker.start()
            self.assertTrue(entered.wait(timeout=2))
            node.refresh_pipeline_identity(
                {"model_type": "FluxModularPipeline"},
                {"key": "model_type"},
            )
            release.set()
            worker.join(timeout=5)

        self.assertFalse(worker.is_alive())
        self.assertEqual(worker_errors, [])
        signals = [
            call.args[1]["signal"]["value"]
            for call in node.set_field_params.call_args_list
            if "signal" in call.args[1]
        ]
        self.assertEqual(signals, ["FluxModularPipeline"] * len(MODELS_LOADER_IDENTITY_OUTPUTS))

    def test_execute_requires_persisted_identity_before_modular_pipeline_load(self):
        node = ModelsLoader("identity-required")
        with patch("modules.ModularDiffusers.loaders.ModularPipeline.from_pretrained") as pipeline_loader:
            with self.assertRaisesRegex(ValueError, "backend-issued identity"):
                node.execute(
                    model_type=CUSTOM_PIPELINE_MODEL_TYPE,
                    repo_id={"source": "local", "value": str(self.repository)},
                    device="cpu",
                    dtype=torch.float32,
                    auto_offload=False,
                    offload_mode="none",
                    modiff_pipeline_identity=None,
                )
        pipeline_loader.assert_not_called()

    def test_execute_reverifies_hash_before_modular_pipeline_load(self):
        binding = resolve_custom_pipeline_binding(
            source="local",
            repo_id=str(self.repository),
            revision=None,
            trust_remote_code=False,
        )
        _write_sidecar(self.repository, _config_bytes("changed after issue"))
        node = ModelsLoader("identity-reverify")
        with patch("modules.ModularDiffusers.loaders.ModularPipeline.from_pretrained") as pipeline_loader:
            with self.assertRaisesRegex(ValueError, "no longer matches"):
                node.execute(
                    model_type=CUSTOM_PIPELINE_MODEL_TYPE,
                    repo_id={"source": "local", "value": str(self.repository)},
                    device="cpu",
                    dtype=torch.float32,
                    auto_offload=False,
                    offload_mode="none",
                    modiff_pipeline_identity=binding.identity.to_dict(),
                )
        pipeline_loader.assert_not_called()

    def test_execute_rejects_hand_edited_local_revision_before_pipeline_load(self):
        binding = resolve_custom_pipeline_binding(
            source="local",
            repo_id=str(self.repository),
            revision=None,
            trust_remote_code=False,
        )
        node = ModelsLoader("identity-local-revision-rejection")
        with patch("modules.ModularDiffusers.loaders.ModularPipeline.from_pretrained") as pipeline_loader:
            with self.assertRaisesRegex(ValueError, "must not claim a Hub revision"):
                node.execute(
                    model_type=CUSTOM_PIPELINE_MODEL_TYPE,
                    repo_id={"source": "local", "value": str(self.repository)},
                    device="cpu",
                    dtype=torch.float32,
                    auto_offload=False,
                    offload_mode="none",
                    revision="a" * 40,
                    modiff_pipeline_identity=binding.identity.to_dict(),
                )
        pipeline_loader.assert_not_called()

    def test_execute_custom_rejects_unapproved_library_before_any_upstream_constructor(self):
        (self.repository / "modular_model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "FluxModularPipeline",
                    "_blocks_class_name": "FluxAutoBlocks",
                    "transformer": [
                        None,
                        None,
                        {
                            "type_hint": ["attacker_package", "Payload"],
                            "pretrained_model_name_or_path": "owner/payload",
                            "revision": "a" * 40,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        real_import = __import__

        def guarded_import(name, *args, **kwargs):
            if name == "attacker_package":
                raise AssertionError("repository-controlled package import was reached")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=guarded_import),
            patch("modules.ModularDiffusers.loaders.ModularPipeline.from_pretrained") as pipeline_loader,
        ):
            with self.assertRaisesRegex(ValueError, "approved official"):
                resolve_custom_pipeline_binding(
                    source="local",
                    repo_id=str(self.repository),
                    revision=None,
                    trust_remote_code=False,
                )
        pipeline_loader.assert_not_called()

    def test_self_describing_outputs_copy_the_exact_custom_identity(self):
        identity = resolve_custom_pipeline_binding(
            source="local",
            repo_id=str(self.repository),
            revision=None,
            trust_remote_code=False,
        ).identity.to_dict()
        outputs = {"unet_out": {"model_id": "one"}, "text_encoders": {"text_encoder": {}}}

        annotate_modular_loader_outputs(
            outputs,
            repo_id=str(self.repository.resolve()),
            repo_source="local",
            model_type=CUSTOM_PIPELINE_MODEL_TYPE,
            revision=None,
            trust_remote_code=False,
            custom_identity=identity,
        )
        identity["repo_id"] = "tampered after publication"

        for value in outputs.values():
            recovered = CustomPipelineExecutionIdentity.from_value(value[CUSTOM_PIPELINE_IDENTITY_FIELD])
            self.assertEqual(recovered.repo_id, str(self.repository.resolve()))
            self.assertEqual(value["repo_source"], "local")

        annotate_modular_loader_outputs(
            outputs,
            repo_id="owner/standard",
            repo_source="hub",
            model_type="FluxModularPipeline",
            revision="a" * 40,
            trust_remote_code=False,
        )
        self.assertTrue(all(CUSTOM_PIPELINE_IDENTITY_FIELD not in value for value in outputs.values()))

    def test_standard_component_load_does_not_force_local_only(self):
        spec = SimpleNamespace(
            pretrained_model_name_or_path="owner/component",
            load=Mock(return_value=object()),
        )
        pipeline = SimpleNamespace(
            _component_specs={"transformer": spec},
            _pretrained_model_name_or_path="owner/component",
            register_components=Mock(),
        )
        diagnostics = {}

        load_components_strict(
            pipeline,
            ["transformer"],
            required_names={"transformer"},
            model_id="owner/component",
            dtype="float32",
            offload_mode="none",
            quant_config=None,
            diagnostics=diagnostics,
            component_load_kwargs={"torch_dtype": torch.float32},
        )

        spec.load.assert_called_once_with(torch_dtype=torch.float32)
        self.assertEqual(diagnostics["components_loaded"], ["transformer"])

    def test_auto_model_trust_false_preserves_existing_download_behavior(self):
        node = AutoModelLoader("auto-model-download-contract")
        node.diffusers_loading_progress = Mock(return_value=nullcontext())
        revision = "a" * 40

        class ApprovedTransformer:
            pass

        spec = SimpleNamespace(
            load_id="owner/component",
            load=Mock(side_effect=RuntimeError("stop after load kwargs")),
        )
        with (
            patch("modules.ModularDiffusers.loaders.ComponentSpec", return_value=spec) as component_spec,
            patch(
                "modules.ModularDiffusers.loaders._preflight_reviewed_diffusers_component",
                return_value=(
                    "hub",
                    "owner/component",
                    revision,
                    "transformer",
                    "ApprovedTransformer",
                    "f" * 64,
                ),
            ),
            patch(
                "modules.ModularDiffusers.loaders._resolve_reviewed_diffusers_component_class",
                return_value=ApprovedTransformer,
            ),
            patch("modules.ModularDiffusers.loaders.reusable_standalone_component", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after load kwargs"):
                node.execute(
                    model_type="transformer",
                    model_id={"source": "hub", "value": "owner/component"},
                    dtype=torch.float32,
                    trust_remote_code=False,
                )
        component_spec.assert_called_once_with(
            name="transformer",
            type_hint=ApprovedTransformer,
            pretrained_model_name_or_path="owner/component",
            subfolder="transformer",
            variant=None,
            revision=revision,
        )
        spec.load.assert_called_once_with(torch_dtype=torch.float32)

    def test_auto_model_selector_spoofing_fails_before_node_cache(self):
        node = AutoModelLoader("auto-model-selector-cache-guard")
        common = {
            "model_type": "transformer",
            "trust_remote_code": False,
            "revision": "a" * 40,
        }
        invalid_selectors = (
            "owner/component",
            {"source": "api", "value": "owner/component"},
            {"source": [], "value": "owner/component"},
        )
        with patch("modiff.NodeBase.NodeBase.__call__") as base_call:
            for selector in invalid_selectors:
                with self.subTest(selector=selector):
                    with self.assertRaises((TypeError, ValueError)):
                        node(model_id=selector, **common)
            with self.assertRaisesRegex(ValueError, "requires a component type"):
                node(
                    model_type=[],
                    model_id={"source": "hub", "value": "owner/component"},
                    **{key: value for key, value in common.items() if key != "model_type"},
                )
        base_call.assert_not_called()

    def test_auto_model_derived_config_identity_participates_in_node_cache(self):
        node = AutoModelLoader("auto-model-derived-cache-key")
        identity_a = ("hub", "owner/component", "a" * 40, "transformer", "ClassA", "1" * 64)
        identity_b = ("hub", "owner/component", "a" * 40, "transformer", "ClassA", "2" * 64)

        def publish(**kwargs):
            identity = kwargs["_reviewed_component_identity"]
            source, repo_id, revision, _subfolder, class_name, fingerprint = identity
            model = {
                "model_id": "transformer-resident-a" if fingerprint == "1" * 64 else "transformer-resident-b",
                "class_name": class_name,
                "class": "A" if fingerprint == "1" * 64 else "B",
                "repo_id": repo_id,
                "repo_source": source,
                "revision": revision,
                "trust_remote_code": False,
            }
            bind_standalone_component_output(
                model,
                issuer=node._standalone_component_issuer,
                component_kind="transformer",
                reviewed_identity=identity,
            )
            return {"model": model}

        node.execute = Mock(side_effect=publish)
        inputs = {
            "model_type": "transformer",
            "model_id": {"source": "hub", "value": "owner/component"},
            "dtype": "float32",
            "subfolder": "transformer",
            "variant": "",
            "trust_remote_code": False,
            "revision": "a" * 40,
            "device": "cpu",
            "auto_offload": False,
            "offload_mode": "none",
        }
        with (
            patch(
                "modules.ModularDiffusers.loaders._preflight_reviewed_diffusers_component",
                side_effect=(identity_a, identity_b, identity_b),
            ),
            patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True),
        ):
            first = node(**inputs)
            second = node(**inputs)
            third = node(**inputs)

        self.assertEqual(first["model"]["class"], "A")
        self.assertEqual(second["model"]["class"], "B")
        self.assertEqual(third["model"]["class"], "B")
        self.assertEqual(node.execute.call_count, 2)

    def test_auto_model_preflight_accepts_only_diffusers_category_config(self):
        from modules.ModularDiffusers.loaders import _preflight_reviewed_diffusers_component

        revision = "a" * 40
        valid_config = {"_class_name": "FluxTransformer2DModel"}
        with (
            patch(
                "modules.ModularDiffusers.loaders._load_reviewed_component_config",
                return_value=valid_config,
            ) as load_config,
            patch("modules.ModularDiffusers.loaders.importlib.import_module") as class_import,
        ):
            result = _preflight_reviewed_diffusers_component(
                "transformer",
                {"source": "hub", "value": "owner/component"},
                "transformer",
                revision,
            )
        self.assertEqual(
            result,
            (
                "hub",
                "owner/component",
                revision,
                "transformer",
                "FluxTransformer2DModel",
                hashlib.sha256(b'{"_class_name":"FluxTransformer2DModel"}').hexdigest(),
            ),
        )
        load_config.assert_called_once_with("hub", "owner/component", "transformer", revision)
        class_import.assert_not_called()

        hostile_configs = (
            {"_class_name": "FluxTransformer2DModel", "auto_map": {"x": "attacker.Payload"}},
            {"model_type": "attacker_transformer"},
            {"_class_name": "AutoencoderKL"},
            {"_class_name": "AttackerPipeline"},
        )
        for config in hostile_configs:
            with (
                self.subTest(config=config),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_component_config",
                    return_value=config,
                ),
            ):
                with self.assertRaises(ValueError):
                    _preflight_reviewed_diffusers_component(
                        "transformer",
                        {"source": "hub", "value": "owner/component"},
                        "transformer",
                        revision,
                    )

    def test_curated_qwen_controlnet_preflight_uses_the_catalog_commit_without_network(self):
        from modules.ModularDiffusers.loaders import _preflight_reviewed_diffusers_component

        revision = "b13036f066d6dee7c20513e263d3d673055e9de8"
        with patch(
            "modules.ModularDiffusers.loaders._load_reviewed_component_config",
            return_value={"_class_name": "QwenImageControlNetModel"},
        ) as load_config:
            result = _preflight_reviewed_diffusers_component(
                "controlnet",
                {"source": "hub", "value": "InstantX/Qwen-Image-ControlNet-Union"},
                "",
                "",
            )

        self.assertEqual(
            result[0:5],
            (
                "hub",
                "InstantX/Qwen-Image-ControlNet-Union",
                revision,
                None,
                "QwenImageControlNetModel",
            ),
        )
        load_config.assert_called_once_with(
            "hub",
            "InstantX/Qwen-Image-ControlNet-Union",
            None,
            revision,
        )

    def test_sdxl_union_preflight_uses_only_the_exact_catalog_class_override(self):
        from modules.ModularDiffusers.loaders import _preflight_reviewed_diffusers_component

        revision = "801a4a3fa3d4c936f4feea95b98607bc6726f80c"
        with patch(
            "modules.ModularDiffusers.loaders._load_reviewed_component_config",
            return_value={"_class_name": "ControlNetModel"},
        ) as load_config:
            result = _preflight_reviewed_diffusers_component(
                "controlnet",
                {"source": "hub", "value": "xinsir/controlnet-union-sdxl-1.0"},
                "",
                "",
                "ControlNetUnionModel",
            )

        self.assertEqual(
            result[:5],
            (
                "hub",
                "xinsir/controlnet-union-sdxl-1.0",
                revision,
                None,
                "ControlNetUnionModel",
            ),
        )
        load_config.assert_called_once_with(
            "hub",
            "xinsir/controlnet-union-sdxl-1.0",
            None,
            revision,
        )

    def test_component_class_override_fails_closed_outside_its_exact_catalog_pin(self):
        from modules.ModularDiffusers.loaders import _preflight_reviewed_diffusers_component

        with patch(
            "modules.ModularDiffusers.loaders._load_reviewed_component_config",
            return_value={"_class_name": "ControlNetModel"},
        ):
            with self.assertRaisesRegex(ValueError, "exact reviewed repository"):
                _preflight_reviewed_diffusers_component(
                    "controlnet",
                    {"source": "hub", "value": "owner/unreviewed-controlnet"},
                    "",
                    "a" * 40,
                    "ControlNetUnionModel",
                )
            with self.assertRaisesRegex(ValueError, "exact reviewed repository"):
                _preflight_reviewed_diffusers_component(
                    "controlnet",
                    {"source": "hub", "value": "xinsir/controlnet-union-sdxl-1.0"},
                    "",
                    "",
                    "AttackerControlNetModel",
                )

    def test_auto_model_local_root_mapping_cannot_select_an_installed_library(self):
        from diffusers import FluxTransformer2DModel
        from modules.ModularDiffusers.loaders import _preflight_reviewed_diffusers_component

        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory).resolve()
            component_directory = repository / "transformer"
            component_directory.mkdir()
            (repository / "model_index.json").write_text(
                json.dumps({"transformer": ["attacker_package", "Payload"]}),
                encoding="utf-8",
            )
            (component_directory / "config.json").write_text(
                json.dumps({"_class_name": "FluxTransformer2DModel"}),
                encoding="utf-8",
            )
            relative_repository = os.path.relpath(repository, Path.cwd())
            preflight = _preflight_reviewed_diffusers_component(
                "transformer",
                {"source": "local", "value": relative_repository},
                "transformer",
                None,
            )
            self.assertEqual(preflight[1], str(repository.resolve()))

            attacker_module = ModuleType("attacker_package")
            attacker_module.Payload = SimpleNamespace(from_pretrained=Mock())
            node = AutoModelLoader("auto-model-installed-library-guard")
            node.diffusers_loading_progress = Mock(return_value=nullcontext())
            with (
                patch.dict(sys.modules, {"attacker_package": attacker_module}),
                patch.object(
                    FluxTransformer2DModel,
                    "from_pretrained",
                    side_effect=RuntimeError("approved loader stop"),
                ) as approved_load,
                patch("modules.ModularDiffusers.loaders.reusable_standalone_component", return_value=None),
            ):
                with self.assertRaisesRegex(ValueError, "approved loader stop"):
                    node.execute(
                        model_type="transformer",
                        model_id={"source": "local", "value": relative_repository},
                        dtype=torch.float32,
                        trust_remote_code=False,
                        subfolder="transformer",
                        device="cpu",
                        auto_offload=False,
                        offload_mode="none",
                    )

            approved_load.assert_called_once()
            self.assertEqual(approved_load.call_args.args[0], str(repository.resolve()))
            attacker_module.Payload.from_pretrained.assert_not_called()

    def test_auto_model_hub_revision_and_subfolder_fail_closed_before_config_access(self):
        from modules.ModularDiffusers.loaders import _preflight_reviewed_diffusers_component

        cases = (
            (None, "transformer", "immutable"),
            ("A" * 40, "transformer", "lowercase"),
            ("a" * 40, "../transformer", "traversal"),
        )
        with patch("modules.ModularDiffusers.loaders._load_reviewed_component_config") as load_config:
            for revision, subfolder, message in cases:
                with self.subTest(revision=revision, subfolder=subfolder):
                    with self.assertRaisesRegex(ValueError, message):
                        _preflight_reviewed_diffusers_component(
                            "transformer",
                            {"source": "hub", "value": "owner/component"},
                            subfolder,
                            revision,
                        )
        load_config.assert_not_called()

    def test_remote_code_and_non_boolean_bypasses_fail_before_node_cache_or_loaders(self):
        models_loader = ModelsLoader("remote-code-cache-guard")
        auto_loader = AutoModelLoader("auto-remote-code-cache-guard")
        with patch("modiff.NodeBase.NodeBase.__call__") as base_call:
            for node, model_type in (
                (models_loader, "FluxModularPipeline"),
                (models_loader, CUSTOM_PIPELINE_MODEL_TYPE),
                (auto_loader, "transformer"),
            ):
                with self.subTest(node=node.__class__.__name__, model_type=model_type):
                    with self.assertRaisesRegex(ValueError, "(?i)repository code is disabled"):
                        node(model_type=model_type, trust_remote_code=True)
                    with self.assertRaisesRegex(TypeError, "JSON boolean"):
                        node(model_type=model_type, trust_remote_code="false")
            custom_identity = resolve_custom_pipeline_binding(
                source="local",
                repo_id=str(self.repository),
                revision=None,
                trust_remote_code=False,
            ).identity.to_dict()
            models_loader(
                model_type=CUSTOM_PIPELINE_MODEL_TYPE,
                trust_remote_code=False,
                modiff_pipeline_identity=custom_identity,
            )
            base_call.assert_called_once()

        with (
            patch("modules.ModularDiffusers.loaders.ComponentSpec") as component_spec,
            patch("modules.ModularDiffusers.loaders.ModularPipeline.from_pretrained") as pipeline_loader,
        ):
            with self.assertRaisesRegex(ValueError, "repository code is disabled"):
                auto_loader.execute(
                    model_type="transformer",
                    model_id={"source": "hub", "value": "owner/component"},
                    dtype=torch.float32,
                    trust_remote_code=True,
                )
            with self.assertRaisesRegex(TypeError, "JSON boolean"):
                auto_loader.execute(
                    model_type="transformer",
                    model_id={"source": "hub", "value": "owner/component"},
                    dtype=torch.float32,
                    trust_remote_code="false",
                )
            with self.assertRaisesRegex(ValueError, "repository code is disabled"):
                models_loader.execute(
                    model_type="FluxModularPipeline",
                    repo_id={"source": "hub", "value": "owner/pipeline"},
                    device="cpu",
                    dtype=torch.float32,
                    trust_remote_code=True,
                    auto_offload=False,
                    offload_mode="none",
                )
        component_spec.assert_not_called()
        pipeline_loader.assert_not_called()

    def test_every_registered_builtin_pipeline_has_an_exact_reviewed_execution_binding(self):
        registry = _get_registry_instance().get_all()
        for pipeline_class, config in registry.items():
            model_type = pipeline_class.__name__
            if model_type == CUSTOM_PIPELINE_MODEL_TYPE:
                continue
            with self.subTest(model_type=model_type):
                expected_revision = require_catalog_revision(config.default_repo, model_type=model_type)
                self.assertEqual(
                    ModelsLoader._reviewed_builtin_selection(
                        model_type=model_type,
                        repo_id={"source": "hub", "value": config.default_repo},
                        revision=None,
                    ),
                    ("hub", config.default_repo, expected_revision),
                )
                self.assertEqual(
                    ModelsLoader._reviewed_builtin_selection(
                        model_type=model_type,
                        repo_id={"source": "hub", "value": config.default_repo},
                        revision=expected_revision,
                    ),
                    ("hub", config.default_repo, expected_revision),
                )

        variants = (
            ("QwenImageModularPipeline", "Qwen/Qwen-Image"),
            ("WanModularPipeline", "Wan-AI/Wan2.1-T2V-14B-Diffusers"),
            ("WanImage2VideoModularPipeline", "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"),
            ("WanImage2VideoModularPipeline", "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"),
        )
        for model_type, repository in variants:
            revision = require_catalog_revision(repository, model_type=model_type)
            with self.subTest(model_type=model_type, repository=repository):
                self.assertEqual(
                    ModelsLoader._reviewed_builtin_selection(
                        model_type=model_type,
                        repo_id={"source": "hub", "value": repository},
                        revision=revision,
                    ),
                    ("hub", repository, revision),
                )
                self.assertEqual(
                    ModelsLoader._reviewed_builtin_selection(
                        model_type=model_type,
                        repo_id={"source": "hub", "value": repository},
                        revision=None,
                    ),
                    ("hub", repository, revision),
                )

    def test_qwen_same_pipeline_variant_resolves_atomically_and_rejects_edit_models(self):
        base_revision = "75e0b4be04f60ec59a75f475837eced720f823b6"
        self.assertEqual(
            ModelsLoader._effective_builtin_selector(
                model_type="QwenImageModularPipeline",
                repo_id={"source": "hub", "value": "Qwen/Qwen-Image-2512"},
                revision="25468b98e3276ca6700de15c6628e51b7de54a26",
                workflow_id="text2image",
                reviewed_variant="Qwen/Qwen-Image",
            ),
            ({"source": "hub", "value": "Qwen/Qwen-Image"}, base_revision),
        )
        with self.assertRaisesRegex(ValueError, "does not admit model variant"):
            ModelsLoader._effective_builtin_selector(
                model_type="QwenImageModularPipeline",
                repo_id={"source": "hub", "value": "Qwen/Qwen-Image-2512"},
                revision=None,
                workflow_id="text2image",
                reviewed_variant="Qwen/Qwen-Image-Edit-2511",
            )
        with self.assertRaisesRegex(ValueError, "does not admit model variant"):
            ModelsLoader._effective_builtin_selector(
                model_type="QwenImageModularPipeline",
                repo_id={"source": "hub", "value": "Qwen/Qwen-Image-2512"},
                revision=None,
                workflow_id="image2image",
                reviewed_variant="Qwen/Qwen-Image",
            )

    def test_builtin_pipeline_rejects_alternate_artifacts_before_node_cache_reuse(self):
        node = ModelsLoader("reviewed-builtin-cache-guard")
        node.params = {"model_type": "FluxModularPipeline"}
        node.output = {name: object() for name in MODELS_LOADER_IDENTITY_OUTPUTS}
        attacks = (
            (
                {"source": "local", "value": str(self.repository)},
                None,
                "reviewed immutable Hub artifact",
            ),
            ({"source": "hub", "value": "attacker/pipeline"}, None, "requires reviewed repository"),
            (
                {"source": "hub", "value": "black-forest-labs/FLUX.1-dev"},
                "f" * 40,
                "requires reviewed revision",
            ),
        )
        with (
            patch("modiff.NodeBase.NodeBase.__call__") as base_call,
            patch("modules.ModularDiffusers.loaders._validate_reviewed_pipeline_index") as index_validator,
            patch("modules.ModularDiffusers.loaders.ModularPipeline.from_pretrained") as pipeline_loader,
        ):
            for repo_id, revision, message in attacks:
                with self.subTest(repo_id=repo_id, revision=revision):
                    with self.assertRaisesRegex(ValueError, message):
                        node(
                            model_type="FluxModularPipeline",
                            repo_id=repo_id,
                            revision=revision,
                            trust_remote_code=False,
                        )
        base_call.assert_not_called()
        index_validator.assert_not_called()
        pipeline_loader.assert_not_called()

    @requires_transformers
    def test_wan_standard_indexes_accept_only_exact_reviewed_concrete_component_types(self):
        base_document = {
            "_class_name": "WanImageToVideoPipeline",
            "_diffusers_version": "0.34.0.dev0",
            "image_encoder": ["transformers", "CLIPVisionModelWithProjection"],
            "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5TokenizerFast"],
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        }
        cases = (
            (
                "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                "b184e23a8a16b20f108f727c902e769e873ffc73",
                ["transformers", "CLIPImageProcessor"],
            ),
            (
                "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
                "eb849f76dfa246545b65774a9e25943ee69b3fa3",
                ["transformers", "CLIPImageProcessor"],
            ),
            (
                "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
                "17c30769b1e0b5dcaa1799b117bf20a9c31f59d7",
                ["transformers", "CLIPProcessor"],
            ),
        )
        for repository, revision, processor_type in cases:
            document = {**base_document, "image_processor": processor_type}
            with (
                self.subTest(repository=repository),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("model_index.json", document),
                ),
            ):
                filename, validated = _validate_reviewed_pipeline_index(
                    "WanImage2VideoModularPipeline",
                    repository,
                    revision,
                )
                self.assertEqual(filename, "model_index.json")
                self.assertEqual(validated, document)

        t2v_document = {
            "_class_name": "WanPipeline",
            "_diffusers_version": "0.33.0.dev0",
            "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5TokenizerFast"],
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        }
        for repository, revision in (
            (
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "0fad780a534b6463e45facd96134c9f345acfa5b",
            ),
            (
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "38ec498cb3208fb688890f8cc7e94ede2cbd7f68",
            ),
        ):
            with (
                self.subTest(repository=repository),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("model_index.json", t2v_document),
                ),
            ):
                filename, validated = _validate_reviewed_pipeline_index(
                    "WanModularPipeline",
                    repository,
                    revision,
                )
                self.assertEqual((filename, validated), ("model_index.json", t2v_document))

        tampered = {**base_document, "image_processor": ["transformers", "AutoProcessor"]}
        with (
            patch(
                "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                return_value=("model_index.json", tampered),
            ),
            self.assertRaisesRegex(ValueError, "AutoProcessor"),
        ):
            _validate_reviewed_pipeline_index(
                "WanImage2VideoModularPipeline",
                "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
                "17c30769b1e0b5dcaa1799b117bf20a9c31f59d7",
            )

    @requires_transformers
    def test_wan_animate_2_modular_index_accepts_only_the_pinned_t5_tokenizer(self):
        component = lambda library, class_name: [
            library,
            class_name,
            {
                "pretrained_model_name_or_path": "Wan-AI/reviewed",
                "revision": None,
                "subfolder": "component",
                "type_hint": [library, class_name],
                "variant": None,
            },
        ]
        cases = (
            (
                "WanAnimate2ModularPipeline",
                "WanAnimate2Blocks",
                "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
                "7d48412d7b903ff3a89f4f5a960d99e1899605a1",
                "DPMSolverMultistepScheduler",
            ),
            (
                "WanAnimate2DistilledModularPipeline",
                "WanAnimate2DistilledBlocks",
                "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
                "59e4141466bcb1bf9733eca1bc78be6891c9fbdf",
                "FlowMatchEulerDiscreteScheduler",
            ),
        )
        for model_type, blocks_class, repository, revision, scheduler_class in cases:
            document = {
                "_class_name": model_type,
                "_blocks_class_name": blocks_class,
                "_diffusers_version": "0.36.0.dev0",
                "image_encoder": component("transformers", "CLIPVisionModel"),
                "scheduler": component("diffusers", scheduler_class),
                "text_encoder": component("transformers", "UMT5EncoderModel"),
                "tokenizer": component("transformers", "T5TokenizerFast"),
                "transformer": component("diffusers", "WanAnimate2Transformer3DModel"),
                "vae": component("diffusers", "AutoencoderKLWan"),
            }
            with (
                self.subTest(model_type=model_type),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("modular_model_index.json", document),
                ),
            ):
                filename, validated = _validate_reviewed_pipeline_index(model_type, repository, revision)
                self.assertEqual((filename, validated), ("modular_model_index.json", document))

            tampered = {
                **document,
                "tokenizer": component("transformers", "PreTrainedTokenizerFast"),
            }
            with (
                self.subTest(model_type=f"{model_type}-tampered"),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("modular_model_index.json", tampered),
                ),
                self.assertRaisesRegex(ValueError, "PreTrainedTokenizerFast"),
            ):
                _validate_reviewed_pipeline_index(model_type, repository, revision)

            wrong_scheduler = {
                **document,
                "scheduler": component(
                    "diffusers",
                    "FlowMatchEulerDiscreteScheduler"
                    if scheduler_class == "DPMSolverMultistepScheduler"
                    else "DPMSolverMultistepScheduler",
                ),
            }
            with (
                self.subTest(model_type=f"{model_type}-wrong-scheduler"),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("modular_model_index.json", wrong_scheduler),
                ),
                self.assertRaisesRegex(ValueError, "scheduler"),
            ):
                _validate_reviewed_pipeline_index(model_type, repository, revision)

    @requires_transformers
    def test_qwen_edit_standard_index_accepts_only_reviewed_unused_tokenizer(self):
        base_document = {
            "_class_name": "QwenImageEditPipeline",
            "_diffusers_version": "0.35.0.dev0",
            "processor": ["transformers", "Qwen2VLProcessor"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
            "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLQwenImage"],
        }
        with patch(
            "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
            return_value=("model_index.json", base_document),
        ):
            filename, validated = _validate_reviewed_pipeline_index(
                "QwenImageEditModularPipeline",
                "Qwen/Qwen-Image-Edit",
                "ac7f9318f633fc4b5778c59367c8128225f1e3de",
            )
        self.assertEqual((filename, validated), ("model_index.json", base_document))

        tampered = {**base_document, "tokenizer": ["transformers", "AutoTokenizer"]}
        with (
            patch(
                "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                return_value=("model_index.json", tampered),
            ),
            self.assertRaisesRegex(ValueError, "unexpected executable component 'tokenizer'"),
        ):
            _validate_reviewed_pipeline_index(
                "QwenImageEditModularPipeline",
                "Qwen/Qwen-Image-Edit",
                "ac7f9318f633fc4b5778c59367c8128225f1e3de",
            )

    @requires_transformers
    def test_flux2_accepts_only_the_reviewed_concrete_processor(self):
        document = {
            "_class_name": "Flux2Pipeline",
            "_diffusers_version": "0.36.0.dev0",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Mistral3ForConditionalGeneration"],
            "tokenizer": ["transformers", "PixtralProcessor"],
            "transformer": ["diffusers", "Flux2Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLFlux2"],
        }
        with patch(
            "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
            return_value=("model_index.json", document),
        ):
            self.assertEqual(
                _validate_reviewed_pipeline_index(
                    "Flux2ModularPipeline",
                    "black-forest-labs/FLUX.2-dev",
                    "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
                ),
                ("model_index.json", document),
            )
        for wrong_type in ("AutoTokenizer", "Qwen2TokenizerFast", "CLIPProcessor"):
            with (
                self.subTest(wrong_type=wrong_type),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("model_index.json", {
                        **document, "tokenizer": ["transformers", wrong_type],
                    }),
                ),
                self.assertRaisesRegex(ValueError, wrong_type),
            ):
                _validate_reviewed_pipeline_index(
                    "Flux2ModularPipeline",
                    "black-forest-labs/FLUX.2-dev",
                    "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c",
                )

    @requires_transformers
    def test_flux2_klein_accepts_only_the_pinned_transformers_v5_tokenizer_alias(self):
        document = {
            "_class_name": "Flux2KleinPipeline",
            "_diffusers_version": "0.37.0.dev0",
            "is_distilled": True,
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen3ForCausalLM"],
            "tokenizer": ["transformers", "Qwen2TokenizerFast"],
            "transformer": ["diffusers", "Flux2Transformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLFlux2"],
        }
        with patch(
            "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
            return_value=("model_index.json", document),
        ):
            filename, validated = _validate_reviewed_pipeline_index(
                "Flux2KleinModularPipeline",
                "black-forest-labs/FLUX.2-klein-4B",
                "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            )
        self.assertEqual((filename, validated), ("model_index.json", document))

        tampered = {**document, "tokenizer": ["transformers", "AutoTokenizer"]}
        with (
            patch(
                "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                return_value=("model_index.json", tampered),
            ),
            self.assertRaisesRegex(ValueError, "AutoTokenizer"),
        ):
            _validate_reviewed_pipeline_index(
                "Flux2KleinModularPipeline",
                "black-forest-labs/FLUX.2-klein-4B",
                "e7b7dc27f91deacad38e78976d1f2b499d76a294",
            )

    @requires_transformers
    def test_flux1_accepts_only_the_pinned_transformers_v5_t5_tokenizer_alias(self):
        base_document = {
            "_class_name": "FluxPipeline",
            "_diffusers_version": "0.30.0.dev0",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "T5EncoderModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "T5TokenizerFast"],
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
        cases = (
            (
                "FluxModularPipeline",
                "black-forest-labs/FLUX.1-dev",
                "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                "FluxPipeline",
            ),
            (
                "FluxKontextModularPipeline",
                "black-forest-labs/FLUX.1-Kontext-dev",
                "24e9dedc4ef646698dc8eb4e18ae2cec3c9fea0d",
                "FluxKontextPipeline",
            ),
        )
        for model_type, repository, revision, pipeline_class in cases:
            document = {**base_document, "_class_name": pipeline_class}
            with (
                self.subTest(model_type=model_type),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("model_index.json", document),
                ),
            ):
                filename, validated = _validate_reviewed_pipeline_index(model_type, repository, revision)
                self.assertEqual((filename, validated), ("model_index.json", document))

            tampered = {**document, "tokenizer_2": ["transformers", "AutoTokenizer"]}
            with (
                self.subTest(model_type=f"{model_type}-tampered"),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("model_index.json", tampered),
                ),
                self.assertRaisesRegex(ValueError, "AutoTokenizer"),
            ):
                _validate_reviewed_pipeline_index(model_type, repository, revision)

    @requires_transformers
    def test_wan_flf_loads_reviewed_image_only_processor_after_index_validation(self):
        from diffusers.pipelines.pipeline_loading_utils import _fetch_class_library_tuple

        document = {
            "_class_name": "WanImageToVideoPipeline",
            "_diffusers_version": "0.34.0.dev0",
            "image_processor": ["transformers", "CLIPProcessor"],
            "image_encoder": ["transformers", "CLIPVisionModelWithProjection"],
            "scheduler": ["diffusers", "UniPCMultistepScheduler"],
            "text_encoder": ["transformers", "UMT5EncoderModel"],
            "tokenizer": ["transformers", "T5TokenizerFast"],
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "vae": ["diffusers", "AutoencoderKLWan"],
        }
        pipeline = _instantiate_reviewed_builtin_pipeline(
            "WanImage2VideoModularPipeline",
            "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
            index_filename="model_index.json",
            index_document=document,
            components_manager=None,
            collection="wan-flf-load-contract",
        )

        self.assertEqual(document["image_processor"], ["transformers", "CLIPProcessor"])
        self.assertEqual(
            _fetch_class_library_tuple(pipeline.get_component_spec("image_processor").type_hint),
            ("transformers", "CLIPImageProcessor"),
        )

    @requires_transformers
    def test_builtin_pipeline_rejects_cache_mutated_component_library_before_upstream(self):
        document = {
            "_class_name": "FluxModularPipeline",
            "_blocks_class_name": "FluxAutoBlocks",
            "transformer": [
                None,
                None,
                {
                    "type_hint": ["attacker_package", "Payload"],
                    "pretrained_model_name_or_path": "attacker/payload",
                },
            ],
        }
        node = ModelsLoader("reviewed-builtin-index-guard")
        with (
            patch(
                "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                return_value=("modular_model_index.json", document),
            ),
            patch("modules.ModularDiffusers.loaders.ModularPipeline.from_pretrained") as pipeline_loader,
        ):
            with self.assertRaisesRegex(ValueError, "attacker_package"):
                node.execute(
                    model_type="FluxModularPipeline",
                    repo_id={"source": "hub", "value": "black-forest-labs/FLUX.1-dev"},
                    revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                    device="cpu",
                    dtype=torch.float32,
                    trust_remote_code=False,
                    auto_offload=False,
                    offload_mode="none",
                )
        pipeline_loader.assert_not_called()

    def test_builtin_reviewed_index_fingerprint_participates_in_node_cache(self):
        node = ModelsLoader("reviewed-builtin-derived-cache-key")
        revision = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
        base_selection = (
            "hub",
            "black-forest-labs/FLUX.1-dev",
            revision,
            "modular_model_index.json",
        )
        selection_a = (*base_selection, {"_class_name": "FluxModularPipeline", "guidance": 1})
        selection_b = (*base_selection, {"_class_name": "FluxModularPipeline", "guidance": 2})
        outputs_a = {name: {"version": "A"} for name in MODELS_LOADER_IDENTITY_OUTPUTS}
        outputs_b = {name: {"version": "B"} for name in MODELS_LOADER_IDENTITY_OUTPUTS}
        node.execute = Mock(side_effect=(outputs_a, outputs_b))
        inputs = {
            "model_type": "FluxModularPipeline",
            "repo_id": {"source": "hub", "value": "black-forest-labs/FLUX.1-dev"},
            "device": "cpu",
            "dtype": "float32",
            "unet": None,
            "vae": None,
            "lora_list": None,
            "trust_remote_code": False,
            "auto_offload": False,
            "offload_mode": "none",
            "quant_config": None,
            "revision": revision,
            "modiff_pipeline_identity": None,
            "refresh_pipeline_identity_button": False,
        }
        with (
            patch.object(
                node,
                "_preflight_reviewed_builtin_selection",
                side_effect=(selection_a, selection_b, selection_b),
            ),
            patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True),
        ):
            first = node(**inputs)
            second = node(**inputs)
            third = node(**inputs)

        self.assertEqual(first["scheduler"]["version"], "A")
        self.assertEqual(second["scheduler"]["version"], "B")
        self.assertEqual(third["scheduler"]["version"], "B")
        self.assertEqual(node.execute.call_count, 2)

    def test_same_family_variant_capture_records_selected_repository_revision_and_dtype(self):
        node = ModelsLoader("same-family-captured-inputs")
        revision = "75e0b4be04f60ec59a75f475837eced720f823b6"
        outputs = {name: {"selected": "base"} for name in MODELS_LOADER_IDENTITY_OUTPUTS}
        node.execute = Mock(return_value=outputs)
        inputs = {
            "model_type": "QwenImageModularPipeline",
            "repo_id": {"source": "hub", "value": "Qwen/Qwen-Image-2512"},
            "reviewed_variant": "Qwen/Qwen-Image", "workflow_id": "text2image",
            "revision": "25468b98e3276ca6700de15c6628e51b7de54a26",
            "device": "cpu", "dtype": "bfloat16", "auto_offload": False,
            "offload_mode": "none", "trust_remote_code": False,
        }
        with (
            patch.object(node, "_preflight_reviewed_builtin_selection", return_value=(
                "hub", "Qwen/Qwen-Image", revision, "modular_model_index.json",
                {"_class_name": "QwenImageModularPipeline"},
            )),
            patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True),
        ):
            node(**inputs)
            node(**inputs)
        self.assertEqual(node.execute.call_count, 1)
        fields = node._execution_input_record["fields"]
        self.assertEqual(fields["repo_id"]["value"], "Qwen/Qwen-Image")
        self.assertEqual(fields["revision"]["value"], revision)
        self.assertEqual(fields["dtype"]["value"], "bfloat16")
        self.assertEqual(inputs["repo_id"]["value"], "Qwen/Qwen-Image-2512")

    @requires_transformers
    def test_builtin_pipeline_rejects_cache_selected_blocks_before_construction(self):
        document = {
            "_class_name": "FluxModularPipeline",
            "_blocks_class_name": "StableDiffusionXLAutoBlocks",
            "controlnet": [
                None,
                None,
                {"type_hint": ["attacker_package", "Payload"]},
            ],
        }
        node = ModelsLoader("reviewed-builtin-blocks-guard")
        with (
            patch(
                "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                return_value=("modular_model_index.json", document),
            ),
            patch("modules.ModularDiffusers.loaders._instantiate_reviewed_builtin_pipeline") as constructor,
        ):
            with self.assertRaisesRegex(ValueError, "blocks class"):
                node.execute(
                    model_type="FluxModularPipeline",
                    repo_id={"source": "hub", "value": "black-forest-labs/FLUX.1-dev"},
                    revision="3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
                    device="cpu",
                    dtype=torch.float32,
                    trust_remote_code=False,
                    auto_offload=False,
                    offload_mode="none",
                )
        constructor.assert_not_called()

    @requires_transformers
    def test_registered_pipeline_component_contracts_accept_only_installed_expected_type_hints(self):
        from diffusers.pipelines.pipeline_loading_utils import _fetch_class_library_tuple
        from modules.ModularDiffusers.loaders import _validate_reviewed_pipeline_index

        registry = _get_registry_instance().get_all()
        for pipeline_class, config in registry.items():
            model_type = pipeline_class.__name__
            if model_type == CUSTOM_PIPELINE_MODEL_TYPE:
                continue
            installed_pipeline = pipeline_class()
            document = {
                "_class_name": model_type,
                "_blocks_class_name": installed_pipeline.config.get("_blocks_class_name"),
            }
            for component_name, component_spec in installed_pipeline._component_specs.items():
                document[component_name] = [
                    None,
                    None,
                    {"type_hint": list(_fetch_class_library_tuple(component_spec.type_hint))},
                ]
            with (
                self.subTest(model_type=model_type),
                patch(
                    "modules.ModularDiffusers.loaders._load_reviewed_pipeline_index",
                    return_value=("modular_model_index.json", document),
                ),
            ):
                _validate_reviewed_pipeline_index(
                    model_type,
                    config.default_repo,
                    require_catalog_revision(config.default_repo, model_type=model_type),
                )


if __name__ == "__main__":
    unittest.main()
