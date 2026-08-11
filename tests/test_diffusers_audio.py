import hashlib
import sys
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modules as module_registry  # noqa: E402
from modules.DiffusersAudio.main import (  # noqa: E402
    ACE_CONTINUATION_MAX_EXTENSION_SECONDS,
    ACE_MAX_DURATION_SECONDS,
    ACE_STEP_DEFAULT_REPO,
    AUDIO_PIPELINE_ADAPTERS,
    AUDIO_SAMPLE_RATE_OPTIONS,
    STABLE_AUDIO_DEFAULT_REPO,
    FuseAdapters,
    Generate,
    LoadAdapter,
    LoadPipeline,
    SetAdapters,
    _resolve_audio_model_selection,
    _resolve_audio_loader_revision,
    _preflight_audio_invocation,
    audio_to_numpy,
    audio_to_tensor,
    crop_tail,
    get_audio_pipeline_adapter,
)
from modiff.config import CONFIG  # noqa: E402
from modiff.diffusers_profiles import public_execution_profiles  # noqa: E402
from modiff.path_identifiers import resolve_runtime_input_path  # noqa: E402
from modiff.server import WebServer, to_bytes  # noqa: E402


class FakeAceStepPipeline:
    _modiff_audio_pipeline_class = "AceStepPipeline"
    _modiff_audio_mode = "text_to_audio"
    device = "cpu"
    sample_rate = 48000

    def __init__(self):
        self.call_kwargs = None

    def __call__(self, bpm=None, **kwargs):
        self.call_kwargs = {**kwargs, "bpm": bpm}
        return SimpleNamespace(audios=np.zeros((1, 480), dtype=np.float32))


class FakeSourceConditionedAceStepPipeline:
    _modiff_audio_pipeline_class = "AceStepPipeline"
    _modiff_audio_mode = "audio_variation"
    device = "cpu"
    sample_rate = 48000

    def __init__(self):
        self.call_kwargs = None

    def __call__(
        self,
        prompt=None,
        lyrics=None,
        audio_duration=None,
        task_type=None,
        src_audio=None,
        reference_audio=None,
        audio_cover_strength=None,
        attention_kwargs=None,
        repainting_start=None,
        repainting_end=None,
        **kwargs,
    ):
        self.call_kwargs = {
            **kwargs,
            "prompt": prompt,
            "lyrics": lyrics,
            "audio_duration": audio_duration,
            "task_type": task_type,
            "src_audio": src_audio,
            "reference_audio": reference_audio,
            "audio_cover_strength": audio_cover_strength,
            "attention_kwargs": attention_kwargs,
            "repainting_start": repainting_start,
            "repainting_end": repainting_end,
        }
        return SimpleNamespace(audios=np.zeros((1, 480), dtype=np.float32))


class DiffusersAudioGenerateTests(unittest.TestCase):
    def test_audio_adapters_declare_ordered_mode_task_and_input_contracts(self):
        ace = AUDIO_PIPELINE_ADAPTERS["AceStepPipeline"]
        stable = AUDIO_PIPELINE_ADAPTERS["StableAudioPipeline"]

        self.assertEqual(
            ace.modes,
            ("text_to_audio", "audio_variation", "audio_continuation", "audio_repaint"),
        )
        self.assertEqual(
            [contract.task_type for contract in ace.mode_contracts],
            ["text2music", "cover", "continuation", "repaint"],
        )
        self.assertEqual(
            [(contract.source_audio, contract.reference_audio) for contract in ace.mode_contracts],
            [
                ("forbidden", "forbidden"),
                ("required", "forbidden"),
                ("required", "forbidden"),
                ("required", "forbidden"),
            ],
        )
        self.assertTrue(
            all(contract.max_duration_seconds == ACE_MAX_DURATION_SECONDS for contract in ace.mode_contracts)
        )
        self.assertEqual(
            ace.contract_for_mode("audio_continuation").max_extension_seconds,
            ACE_CONTINUATION_MAX_EXTENSION_SECONDS,
        )
        self.assertEqual(ace.source_audio_channels, 2)
        self.assertTrue(ace.duplicate_mono_source)
        self.assertEqual(stable.modes, ("text_to_audio",))
        self.assertIsNone(stable.source_audio_channels)
        self.assertEqual(stable.mode_contracts[0].task_type, "text2audio")
        self.assertEqual(stable.mode_contracts[0].max_duration_seconds, 47)
        self.assertNotIn("extract", Generate.params["task_type"]["options"])
        self.assertNotIn("lego", Generate.params["task_type"]["options"])
        self.assertNotIn("complete", Generate.params["task_type"]["options"])

    def test_every_direct_audio_profile_has_an_exact_adapter_contract(self):
        profiles = [
            profile
            for profile in public_execution_profiles()
            if profile["backend_path"] == "modules.DiffusersAudio.LoadPipeline"
        ]
        self.assertTrue(profiles)
        for profile in profiles:
            with self.subTest(profile=profile["id"]):
                adapter = AUDIO_PIPELINE_ADAPTERS[profile["pipeline_class"]]
                self.assertEqual(tuple(profile["modes"]), adapter.modes)
                self.assertEqual(profile["default_repo"], adapter.default_repo)

    def test_adapter_identity_and_real_loader_inputs_are_strict(self):
        for invalid in (None, "", " AceStepPipeline", "AceStepPipeline ", False, 0, {}, []):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "registered Diffusers audio pipeline class is required"):
                    get_audio_pipeline_adapter(invalid)

        invalid_loader_values = (
            {"mode": "text_to_audio"},
            {
                "model_id": {"source": "hub", "value": STABLE_AUDIO_DEFAULT_REPO},
                "mode": "text_to_audio",
            },
            {"pipeline_class": None, "mode": "text_to_audio"},
            {"pipeline_class": False, "mode": "text_to_audio"},
            {"pipeline_class": "AceStepPipeline"},
            {"pipeline_class": "AceStepPipeline", "mode": None},
            {"pipeline_class": "AceStepPipeline", "mode": False},
            {"pipeline_class": "AceStepPipeline", "mode": 0},
            {"pipeline_class": "AceStepPipeline", "mode": []},
            {"pipeline_class": "AceStepPipeline", "mode": {}},
            {"pipeline_class": "AceStepPipeline", "mode": ""},
            {"pipeline_class": "AceStepPipeline", "mode": " text_to_audio"},
            {"pipeline_class": "AceStepPipeline", "mode": "text_to_audio "},
        )
        for values in invalid_loader_values:
            node = LoadPipeline("audio-loader-invalid")
            node.execute = Mock()
            with self.subTest(values=values):
                with self.assertRaisesRegex(
                    ValueError, "registered Diffusers audio (pipeline class|mode) is required"
                ):
                    node(**values)
            node.execute.assert_not_called()

        pipeline = SimpleNamespace()
        node = LoadPipeline("audio-loader-explicit")
        node.execute = Mock(return_value={"pipeline": pipeline, "resolved_artifact": ACE_STEP_DEFAULT_REPO})
        with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
            node(
                model_id={"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
                pipeline_class="AceStepPipeline",
                mode="text_to_audio",
            )
        self.assertEqual(node.execute.call_args.kwargs["pipeline_class"], "AceStepPipeline")
        self.assertEqual(
            node.execute.call_args.kwargs["revision"],
            "200ba991ae448051e14b0183157e35c2d27c9fb0",
        )
        self.assertEqual(pipeline._modiff_audio_pipeline_class, "AceStepPipeline")
        self.assertEqual(pipeline._modiff_audio_revision, "200ba991ae448051e14b0183157e35c2d27c9fb0")

    def test_hub_pipeline_revisions_fail_closed_before_nodebase_or_upstream(self):
        custom_revision = "0123456789abcdef0123456789abcdef01234567"
        custom_selection = {"source": "hub", "value": "organization/custom-audio"}
        base_values = {
            "model_id": custom_selection,
            "pipeline_class": "AceStepPipeline",
            "mode": "text_to_audio",
        }
        invalid_revisions = (
            None,
            "",
            "main",
            custom_revision.upper(),
            f" {custom_revision}",
            custom_revision[:-1],
            123,
            False,
        )
        node = LoadPipeline("strict-audio-hub-revision")
        node.execute = Mock(side_effect=AssertionError("upstream must not run"))
        for revision in invalid_revisions:
            with self.subTest(revision=revision):
                with self.assertRaisesRegex(ValueError, "immutable lowercase|exact lowercase"):
                    node(**base_values, revision=revision)
        node.execute.assert_not_called()

        curated_mismatch = LoadPipeline("strict-audio-curated-mismatch")
        curated_mismatch.execute = Mock(side_effect=AssertionError("upstream must not run"))
        with self.assertRaisesRegex(ValueError, "pinned to .* does not match"):
            curated_mismatch(
                model_id={"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
                pipeline_class="AceStepPipeline",
                mode="text_to_audio",
                revision="0000000000000000000000000000000000000000",
            )
        curated_mismatch.execute.assert_not_called()

        pipeline = SimpleNamespace()
        valid = LoadPipeline("strict-audio-custom-valid")
        valid.execute = Mock(return_value={"pipeline": pipeline, "resolved_artifact": custom_selection["value"]})
        with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
            valid(**base_values, revision=custom_revision)
        self.assertEqual(valid.execute.call_args.kwargs["revision"], custom_revision)
        self.assertEqual(pipeline._modiff_audio_revision, custom_revision)

    def test_case_insensitive_local_model_source_is_never_replaced_by_a_managed_default(self):
        adapter = AUDIO_PIPELINE_ADAPTERS["StableAudioPipeline"]
        with tempfile.TemporaryDirectory() as temporary:
            local_model = Path(temporary) / ACE_STEP_DEFAULT_REPO
            local_model.mkdir(parents=True)
            with chdir(temporary):
                for source in ("local", "LOCAL", "Local"):
                    selection = {"source": source, "value": ACE_STEP_DEFAULT_REPO}
                    with self.subTest(source=source):
                        self.assertEqual(
                            _resolve_audio_model_selection(adapter, selection),
                            {"source": "local", "value": str(local_model.resolve())},
                        )

                self.assertEqual(
                    _resolve_audio_model_selection(
                        adapter,
                        {"source": "local", "value": str(local_model)},
                    ),
                    {"source": "local", "value": str(local_model.resolve())},
                )

                for invalid in ("organization/not-a-local-model", str(Path(temporary) / "missing")):
                    with self.subTest(invalid=invalid):
                        with self.assertRaisesRegex(ValueError, "directory does not exist"):
                            _resolve_audio_model_selection(
                                adapter,
                                {"source": "local", "value": invalid},
                            )

                        node = LoadPipeline("missing-local-audio-boundary")
                        node.execute = Mock(side_effect=AssertionError("upstream must not run"))
                        with self.assertRaisesRegex(ValueError, "directory does not exist"):
                            node(
                                model_id={"source": "local", "value": invalid},
                                pipeline_class="AceStepPipeline",
                                mode="text_to_audio",
                                revision="main",
                            )
                        node.execute.assert_not_called()

    def test_model_selection_source_is_canonical_and_cannot_bypass_the_catalog_pin(self):
        adapter = AUDIO_PIPELINE_ADAPTERS["AceStepPipeline"]
        for source in ("hub", "HUB", "Hub"):
            with self.subTest(source=source):
                self.assertEqual(
                    _resolve_audio_model_selection(
                        adapter,
                        {"source": source, "value": ACE_STEP_DEFAULT_REPO},
                    ),
                    {"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
                )

        self.assertEqual(
            _resolve_audio_model_selection(
                adapter,
                {"source": "HUB", "value": ACE_STEP_DEFAULT_REPO.upper()},
            ),
            {"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
        )

        for source in (None, "", " hub", "hub ", "remote", False, 0, {}, []):
            node = LoadPipeline("audio-source-boundary")
            node.execute = Mock()
            with self.subTest(source=repr(source)):
                with self.assertRaisesRegex(ValueError, "source must be exactly hub or local"):
                    node(
                        model_id={"source": source, "value": ACE_STEP_DEFAULT_REPO},
                        pipeline_class="AceStepPipeline",
                        mode="text_to_audio",
                    )
            node.execute.assert_not_called()

    def test_hub_model_source_cannot_resolve_as_a_local_directory(self):
        revision = "0123456789abcdef0123456789abcdef01234567"
        node = LoadPipeline("audio-hub-local-path-boundary")
        node.execute = Mock(side_effect=AssertionError("upstream must not run"))

        with tempfile.TemporaryDirectory() as temporary:
            local_repo = Path(temporary) / "organization" / "local-audio"
            local_repo.mkdir(parents=True)
            with chdir(temporary):
                invalid_hub_values = (
                    "organization/local-audio",
                    str(local_repo),
                    local_repo.as_uri(),
                    "../local-audio",
                    "single-component",
                )
                for value in invalid_hub_values:
                    with self.subTest(value=value):
                        with self.assertRaisesRegex(ValueError, "namespace/repository|local filesystem"):
                            node(
                                model_id={"source": "hub", "value": value},
                                pipeline_class="AceStepPipeline",
                                mode="text_to_audio",
                                revision=revision,
                            )

        node.execute.assert_not_called()

    def test_default_hub_model_cannot_resolve_as_a_local_directory(self):
        adapter = AUDIO_PIPELINE_ADAPTERS["AceStepPipeline"]
        node = LoadPipeline("audio-default-hub-local-path-boundary")
        node.execute = Mock(side_effect=AssertionError("upstream must not run"))

        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / adapter.default_repo).mkdir(parents=True)
            with chdir(temporary):
                for selection in (None, "", {"source": "hub", "value": ""}):
                    with self.subTest(selection=selection):
                        with self.assertRaisesRegex(ValueError, "local filesystem"):
                            node(
                                model_id=selection,
                                pipeline_class="AceStepPipeline",
                                mode="text_to_audio",
                            )

        node.execute.assert_not_called()

    def test_local_audio_model_drops_any_hub_revision(self):
        pipeline = SimpleNamespace()
        node = LoadPipeline("canonical-local-audio-cache")
        node.execute = Mock(return_value={"pipeline": pipeline, "resolved_artifact": "local-audio"})

        with tempfile.TemporaryDirectory() as temporary:
            local_model = Path(temporary) / "models" / "local-audio"
            local_model.mkdir(parents=True)
            with chdir(temporary), patch("modiff.NodeBase.modelstore.is_local_cached", return_value=True):
                first = node(
                    model_id={"source": "local", "value": "models/local-audio"},
                    pipeline_class="AceStepPipeline",
                    mode="text_to_audio",
                    revision="main",
                )
                second = node(
                    model_id={"source": "local", "value": str(local_model)},
                    pipeline_class="AceStepPipeline",
                    mode="text_to_audio",
                    revision=None,
                )

        self.assertIs(first, second)
        node.execute.assert_called_once()
        self.assertEqual(
            node.execute.call_args.kwargs["model_id"],
            {"source": "local", "value": str(local_model.resolve())},
        )
        # NodeBase serializes the optional field default as an empty string;
        # the facade resolves it to None again immediately inside execute.
        self.assertEqual(node.execute.call_args.kwargs["revision"], "")
        self.assertIsNone(
            _resolve_audio_loader_revision(
                node.execute.call_args.kwargs["model_id"],
                str(local_model.resolve()),
                "main",
            )
        )
        self.assertIsNone(pipeline._modiff_audio_revision)

    def test_loader_actions_update_class_modes_repository_and_output_signal(self):
        node = LoadPipeline("audio-contract-action")
        node.set_field_params = Mock()
        node.set_field_value = Mock()

        node.update_audio_contract(
            {
                "pipeline_class": "StableAudioPipeline",
                "mode": "audio_variation",
                "model_id": {"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
            },
            {"key": "pipeline_class"},
        )

        mode_update = next(call.args[1] for call in node.set_field_params.call_args_list if call.args[0] == "mode")
        self.assertEqual(mode_update["options"], ["text_to_audio"])
        self.assertEqual(mode_update["default"], "text_to_audio")
        published = node.set_field_value.call_args.args[0]
        self.assertEqual(published["model_id"], {"source": "hub", "value": STABLE_AUDIO_DEFAULT_REPO})
        self.assertEqual(published["revision"], "f21265c1e2710b3bd2386596943f0007f55f802e")
        self.assertEqual(published["audio_contract"]["taskType"], "text2audio")
        pipeline_signal = next(
            call.args[1]["signal"] for call in node.set_field_params.call_args_list if call.args[0] == "pipeline"
        )
        self.assertEqual(pipeline_signal["value"], published["audio_contract"])

    def test_audio_model_action_couples_repository_and_revision_before_real_execution(self):
        stale_revision = "0" * 40
        replacement_revision = "1234567890abcdef1234567890abcdef12345678"
        replacement = {"source": "hub", "value": "organization/replacement-audio"}
        node = LoadPipeline("audio-model-identity-action")
        node._sid = "audio-browser-session"
        messages = []
        current_server = SimpleNamespace(
            _current_dynamic_message_identity_payload=lambda: {},
            queue_message=lambda message, sid=None: messages.append((message, sid)),
        )

        with patch("modiff.NodeBase._server", return_value=current_server):
            node.update_audio_contract(
                {
                    "pipeline_class": "AceStepPipeline",
                    "mode": "text_to_audio",
                    "model_id": replacement,
                    "revision": stale_revision,
                },
                {"key": "model_id"},
            )

        value_message = next(message for message, _sid in messages if message["type"] == "set_field_value")
        self.assertEqual(value_message["fields"]["model_id"], replacement)
        self.assertEqual(value_message["fields"]["revision"], "")
        self.assertEqual(
            next(sid for message, sid in messages if message["type"] == "set_field_value"),
            "audio-browser-session",
        )

        for ref_key in ("pipeline_class", "mode"):
            preserving = LoadPipeline(f"audio-custom-pin-{ref_key}")
            preserving.set_field_params = Mock()
            preserving.set_field_value = Mock()
            preserving.update_audio_contract(
                {
                    "pipeline_class": "AceStepPipeline",
                    "mode": "text_to_audio",
                    "model_id": replacement,
                    "revision": replacement_revision,
                },
                {"key": ref_key},
            )
            with self.subTest(ref_key=ref_key):
                self.assertNotIn("revision", preserving.set_field_value.call_args.args[0])

        upstream_calls = []

        class FakePipelineClass:
            @classmethod
            def from_pretrained(cls, repository, **kwargs):
                upstream_calls.append((repository, kwargs["revision"]))
                return SimpleNamespace()

        executing = LoadPipeline("audio-replacement-execution")
        executing.mm_add = Mock()
        with (
            patch("modules.DiffusersAudio.main.pipeline_class_from_name", return_value=FakePipelineClass),
            patch("modules.DiffusersAudio.main.local_files_only", return_value=True),
            patch("modules.DiffusersAudio.main.apply_pipeline_offload"),
        ):
            result = executing.execute(
                pipeline_class="AceStepPipeline",
                mode="text_to_audio",
                model_id=replacement,
                revision=replacement_revision,
            )

        self.assertEqual(upstream_calls, [(replacement["value"], replacement_revision)])
        self.assertEqual(result["pipeline"]._modiff_audio_repo, replacement["value"])
        self.assertEqual(result["pipeline"]._modiff_audio_revision, replacement_revision)

    def test_audio_model_action_publishes_catalog_pin_and_clears_local_revision(self):
        cataloged = LoadPipeline("audio-catalog-pin-action")
        cataloged.set_field_params = Mock()
        cataloged.set_field_value = Mock()
        cataloged.update_audio_contract(
            {
                "pipeline_class": "StableAudioPipeline",
                "mode": "text_to_audio",
                "model_id": {"source": "hub", "value": STABLE_AUDIO_DEFAULT_REPO},
                "revision": "0" * 40,
            },
            {"key": "model_id"},
        )
        self.assertEqual(
            cataloged.set_field_value.call_args.args[0]["revision"],
            "f21265c1e2710b3bd2386596943f0007f55f802e",
        )

        with tempfile.TemporaryDirectory() as temporary:
            local_model = Path(temporary) / "local-audio"
            local_model.mkdir()
            local = LoadPipeline("audio-local-revision-action")
            local.set_field_params = Mock()
            local.set_field_value = Mock()
            local.update_audio_contract(
                {
                    "pipeline_class": "AceStepPipeline",
                    "mode": "text_to_audio",
                    "model_id": {"source": "local", "value": str(local_model)},
                    "revision": "0" * 40,
                },
                {"key": "model_id"},
            )
        self.assertEqual(local.set_field_value.call_args.args[0]["revision"], "")

    def test_generate_contract_signal_sets_task_and_audio_input_form_contract(self):
        node = Generate("audio-generate-contract")
        node.set_field_params = Mock()
        contract = (
            AUDIO_PIPELINE_ADAPTERS["AceStepPipeline"]
            .contract_for_mode("audio_variation")
            .signal_value("AceStepPipeline", ACE_STEP_DEFAULT_REPO)
        )

        node.update_audio_contract({"audio_contract": contract}, {"key": "pipeline"})

        updates = {call.args[0]: call.args[1] for call in node.set_field_params.call_args_list}
        self.assertEqual(updates, contract["fieldParams"])
        self.assertEqual(updates["task_type"]["options"], ["cover"])
        self.assertEqual(updates["task_type"]["default"], "cover")
        self.assertTrue(updates["source_audio"]["required"])
        self.assertFalse(updates["source_audio"]["hidden"])
        self.assertFalse(updates["reference_audio"]["required"])
        self.assertTrue(updates["reference_audio"]["hidden"])
        self.assertFalse(updates["audio_cover_strength"]["hidden"])
        self.assertTrue(updates["repainting_start"]["hidden"])
        variation_duration_visibility = next(
            call.args[1]["hidden"]
            for call in node.set_field_params.call_args_list
            if call.args[0] == "audio_duration" and "hidden" in call.args[1]
        )
        self.assertFalse(variation_duration_visibility)

        node.set_field_params.reset_mock()
        tampered_contract = {
            **contract,
            "fieldParams": {
                **contract["fieldParams"],
                "lyrics": {"hidden": True},
            },
        }
        with self.assertRaisesRegex(ValueError, "stale or mismatched task contract"):
            node.update_audio_contract({"audio_contract": tampered_contract}, {"key": "pipeline"})
        node.set_field_params.assert_not_called()

        for mode in ("audio_continuation", "audio_repaint"):
            node.set_field_params.reset_mock()
            derived_contract = (
                AUDIO_PIPELINE_ADAPTERS["AceStepPipeline"]
                .contract_for_mode(mode)
                .signal_value("AceStepPipeline", ACE_STEP_DEFAULT_REPO)
            )
            node.update_audio_contract({"audio_contract": derived_contract}, {"key": "pipeline"})
            duration_visibility = next(
                call.args[1]["hidden"]
                for call in node.set_field_params.call_args_list
                if call.args[0] == "audio_duration" and "hidden" in call.args[1]
            )
            with self.subTest(mode=mode):
                self.assertTrue(duration_visibility)

        for adapter in AUDIO_PIPELINE_ADAPTERS.values():
            for mode_contract in adapter.mode_contracts:
                node.set_field_params.reset_mock()
                signal = mode_contract.signal_value(adapter.pipeline_class, adapter.default_repo)
                node.update_audio_contract({"audio_contract": signal}, {"key": "pipeline"})
                updates = {call.args[0]: call.args[1] for call in node.set_field_params.call_args_list}
                with self.subTest(pipeline=adapter.pipeline_class, mode=mode_contract.mode):
                    self.assertEqual(updates, signal["fieldParams"])

    def test_loader_mode_cache_hit_is_retagged_without_reloading(self):
        pipeline = SimpleNamespace()
        node = LoadPipeline("audio-mode-cache")
        node.execute = Mock(return_value={"pipeline": pipeline, "resolved_artifact": ACE_STEP_DEFAULT_REPO})

        with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
            node(
                model_id={"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
                pipeline_class="AceStepPipeline",
                mode="text_to_audio",
            )
            node(
                model_id={"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
                pipeline_class="AceStepPipeline",
                mode="audio_repaint",
            )

        self.assertEqual(node.execute.call_count, 1)
        self.assertEqual(pipeline._modiff_audio_pipeline_class, "AceStepPipeline")
        self.assertEqual(pipeline._modiff_audio_mode, "audio_repaint")
        self.assertEqual(pipeline._modiff_audio_repo, ACE_STEP_DEFAULT_REPO)

    def test_loader_mode_retag_invalidates_cached_audio_descendant_contract(self):
        class Pipeline(FakeSourceConditionedAceStepPipeline):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def __call__(
                self,
                src_audio=None,
                repainting_start=None,
                repainting_end=None,
                **kwargs,
            ):
                self.calls += 1
                self.call_kwargs = {
                    **kwargs,
                    "src_audio": src_audio,
                    "repainting_start": repainting_start,
                    "repainting_end": repainting_end,
                }
                return SimpleNamespace(audios=np.zeros((1, 960), dtype=np.float32))

        pipeline = Pipeline()
        loader = LoadPipeline("audio-cache-loader")
        loader.execute = Mock(return_value={"pipeline": pipeline, "resolved_artifact": ACE_STEP_DEFAULT_REPO})
        task = Generate("audio-cache-task")
        task.progress = lambda *args, **kwargs: None
        server = object.__new__(WebServer)
        server.modules = module_registry.MODULE_MAP
        server.node_cache = {"loader": loader, "task": task}
        server.current_task = None
        server.queue_message = lambda *args, **kwargs: None
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}
        graph_node = {
            "module": "modules.DiffusersAudio",
            "action": "Generate",
            "params": {
                "pipeline": {"sourceId": "loader", "sourceKey": "pipeline"},
                "source_audio": {"value": source},
                "extension_duration": {"value": 0.01},
                "sample_rate": {"value": 48000},
            },
        }
        loader_values = {
            "model_id": {"source": "hub", "value": ACE_STEP_DEFAULT_REPO},
            "pipeline_class": "AceStepPipeline",
        }

        with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
            loader(mode="audio_continuation", **loader_values)
            server.execute_node("task", graph_node, "test", quiet=True)
            loader(mode="audio_repaint", **loader_values)

        self.assertTrue(loader._has_changed)
        self.assertEqual(loader.execute.call_count, 1)
        with self.assertRaisesRegex(RuntimeError, "end strictly greater than the start"):
            server.execute_node("task", graph_node, "test", quiet=True)
        self.assertEqual(pipeline.calls, 1)

    def test_ace_mode_task_mismatch_and_hidden_stale_tasks_fail_explicitly(self):
        pipeline = FakeAceStepPipeline()
        node = Generate("ace-task-contract")

        with self.assertRaisesRegex(ValueError, "text_to_audio requires task text2music"):
            node.execute(pipeline=pipeline, task_type="cover")
        for stale_task in ("extract", "lego", "complete"):
            with self.subTest(task=stale_task):
                with self.assertRaisesRegex(ValueError, "recognized but hidden"):
                    node.execute(pipeline=pipeline, task_type=stale_task)

        self.assertIsNone(pipeline.call_kwargs)

    def test_task_may_derive_only_when_absent_and_supplied_malformed_values_fail(self):
        pipeline = FakeAceStepPipeline()
        node = Generate("audio-task-raw-contract")
        node.execute = Mock(return_value={})
        for invalid in (None, "", " text2music", "text2music ", False, 0, {}, []):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaisesRegex(ValueError, "exact nonblank supported task string"):
                    node(pipeline=pipeline, task_type=invalid)
        node.execute.assert_not_called()

        direct = Generate()
        direct.progress = lambda *args, **kwargs: None
        direct.execute(pipeline=pipeline, audio_duration=0.01, sample_rate=48000)
        self.assertEqual(pipeline.call_kwargs["task_type"], "text2music")

    def test_required_audio_inputs_are_enforced_per_mode(self):
        required_modes = ("audio_variation", "audio_continuation", "audio_repaint")
        for mode in required_modes:
            pipeline = FakeSourceConditionedAceStepPipeline()
            pipeline._modiff_audio_mode = mode
            task = {
                "audio_variation": "cover",
                "audio_continuation": "continuation",
                "audio_repaint": "repaint",
            }[mode]
            with self.subTest(mode=mode, condition="missing-source"):
                with self.assertRaisesRegex(ValueError, "requires source audio"):
                    Generate().execute(
                        pipeline=pipeline,
                        task_type=task,
                        repainting_start=0,
                        repainting_end=0.01,
                    )
            self.assertIsNone(pipeline.call_kwargs)

    def test_required_source_media_is_validated_before_upstream(self):
        invalid_sources = (
            {"samples": np.zeros((1, 0), dtype=np.float32), "sample_rate": 48000},
            {"samples": np.zeros((0, 480), dtype=np.float32), "sample_rate": 48000},
            {"samples": np.zeros((1, 1, 10), dtype=np.float32), "sample_rate": 48000},
            {"samples": np.asarray([[float("nan")]], dtype=np.float32), "sample_rate": 48000},
            {"samples": np.asarray([[float("inf")]], dtype=np.float32), "sample_rate": 48000},
            {"samples": np.asarray([[float("-inf")]], dtype=np.float32), "sample_rate": 48000},
            {"samples": np.zeros((1, 10), dtype=np.float32), "sample_rate": 0},
            {"samples": np.zeros((1, 10), dtype=np.float32), "sample_rate": -1},
            {"samples": np.zeros((1, 10), dtype=np.float32), "sample_rate": 48000.5},
            {"samples": np.zeros((1, 241), dtype=np.float32), "sample_rate": 1},
        )
        for source in invalid_sources:
            pipeline = FakeSourceConditionedAceStepPipeline()
            with self.subTest(shape=np.asarray(source["samples"]).shape, rate=source["sample_rate"]):
                with self.assertRaisesRegex(
                    ValueError,
                    "(channel and one audio frame|mono or multichannel|samples must all be finite|sample rate|at most 240)",
                ):
                    Generate().execute(
                        pipeline=pipeline,
                        task_type="cover",
                        source_audio=source,
                        audio_duration=0.01,
                    )
            self.assertIsNone(pipeline.call_kwargs)

    def test_source_audio_file_variants_use_the_shared_resolver_and_decoder_provenance(self):
        from scipy.io import wavfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            data = root / "data"
            work.mkdir()
            data.mkdir()
            source_path = work / "short-stereo.wav"
            wavfile.write(
                source_path,
                48000,
                np.asarray([[100, 1000], [200, 2000], [300, 3000]], dtype=np.int16),
            )
            sources = (
                "short-stereo.wav",
                {"samples": "short-stereo.wav", "channels": 2},
                {"audio": "short-stereo.wav", "channels": 2},
                {"array": "short-stereo.wav", "channels": 2},
                {"path": "short-stereo.wav", "channels": 2},
                {"file": "short-stereo.wav", "channels": 2},
            )

            with (
                patch.dict(CONFIG.paths, {"work_dir": str(work), "data": str(data)}),
                patch(
                    "modules.DiffusersAudio.main.resolve_runtime_input_path",
                    wraps=resolve_runtime_input_path,
                ) as resolver,
            ):
                for source in sources:
                    with self.subTest(source=source):
                        invocation = _preflight_audio_invocation(
                            FakeSourceConditionedAceStepPipeline(),
                            {
                                "task_type": "cover",
                                "source_audio": source,
                                "audio_duration": 0.01,
                            },
                        )
                        self.assertEqual(invocation.source.samples.shape, (2, 3))
                        resolver.assert_called_once()
                        resolver.reset_mock()

    def test_source_audio_paths_cannot_traverse_or_escape_managed_roots_before_decoder_or_torch(self):
        from scipy.io import wavfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            data = root / "data"
            work.mkdir()
            data.mkdir()
            outside = root / "outside.wav"
            wavfile.write(outside, 48000, np.zeros((3, 2), dtype=np.int16))
            invalid_sources = (
                "../outside.wav",
                str(outside),
                "@data/../outside.wav",
                {"samples": "../outside.wav", "channels": 2},
                {"audio": str(outside), "channels": 2},
                {"array": "@data/../outside.wav", "channels": 2},
                {"path": str(outside), "channels": 2},
                {"file": "../outside.wav", "channels": 2},
            )

            with (
                patch.dict(CONFIG.paths, {"work_dir": str(work), "data": str(data)}),
                patch.dict(sys.modules, {"torch": None}),
                patch("scipy.io.wavfile.read") as decoder,
            ):
                for source in invalid_sources:
                    pipeline = FakeSourceConditionedAceStepPipeline()
                    with self.subTest(source=source):
                        with self.assertRaisesRegex(
                            ValueError,
                            "path identifier is invalid|must stay inside the configured MoDiff work or data directory",
                        ):
                            Generate().execute(
                                pipeline=pipeline,
                                task_type="cover",
                                source_audio=source,
                                audio_duration=0.01,
                            )
                    self.assertIsNone(pipeline.call_kwargs)

            decoder.assert_not_called()

    def test_short_audio_orientation_uses_channels_metadata_and_rejects_guessing_before_torch(self):
        frame_first = np.asarray(
            [[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]],
            dtype=np.float32,
        )
        channel_first = frame_first.T
        square = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        accepted_sources = (
            ({"samples": frame_first, "sample_rate": 48000, "channels": 2}, channel_first),
            ({"samples": channel_first, "sample_rate": 48000, "channels": 2}, channel_first),
            (
                {
                    "samples": square,
                    "sample_layout": "frames_first",
                    "sample_rate": 48000,
                    "channels": 2,
                },
                square.T,
            ),
            (
                {
                    "samples": square,
                    "sample_layout": "channels_first",
                    "sample_rate": 48000,
                    "channels": 2,
                },
                square,
            ),
        )
        for source, expected in accepted_sources:
            with self.subTest(shape=source["samples"].shape):
                invocation = _preflight_audio_invocation(
                    FakeSourceConditionedAceStepPipeline(),
                    {
                        "task_type": "cover",
                        "source_audio": source,
                        "audio_duration": 0.01,
                    },
                )
                self.assertEqual(invocation.source.samples.shape, expected.shape)
                np.testing.assert_allclose(invocation.source.samples, expected)

        mono = _preflight_audio_invocation(
            FakeSourceConditionedAceStepPipeline(),
            {
                "task_type": "cover",
                "source_audio": {
                    "samples": frame_first[:, :1],
                    "sample_rate": 48000,
                    "channels": 1,
                },
                "audio_duration": 0.01,
            },
        ).source.samples
        self.assertEqual(mono.shape, (2, 3))
        np.testing.assert_array_equal(mono[0], mono[1])

        from modules.Audio.main import Load as LoadAudio
        from scipy.io import wavfile

        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary) / "work"
            data = Path(temporary) / "data"
            work.mkdir()
            data.mkdir()
            wavfile.write(
                work / "two-frame-stereo.wav",
                48000,
                np.asarray([[100, 1000], [200, 2000]], dtype=np.int16),
            )
            with patch.dict(CONFIG.paths, {"work_dir": str(work), "data": str(data)}):
                audio_node_payload = LoadAudio().execute(file="two-frame-stereo.wav")["audio"]
                invocation = _preflight_audio_invocation(
                    FakeSourceConditionedAceStepPipeline(),
                    {
                        "task_type": "cover",
                        "source_audio": audio_node_payload,
                        "audio_duration": 0.01,
                    },
                )
        self.assertEqual(audio_node_payload["sample_layout"], "frames_first")
        self.assertEqual(invocation.source.samples.shape, (2, 2))

        with patch.dict(sys.modules, {"torch": None}):
            for shape in ((2, 2), (3, 2), (4, 2), (3, 1)):
                pipeline = FakeSourceConditionedAceStepPipeline()
                with self.subTest(ambiguous_shape=shape):
                    with self.assertRaisesRegex(ValueError, "ambiguous channel orientation"):
                        Generate().execute(
                            pipeline=pipeline,
                            task_type="cover",
                            source_audio={
                                "samples": np.zeros(shape, dtype=np.float32),
                                "sample_rate": 48000,
                            },
                            audio_duration=0.01,
                        )
                self.assertIsNone(pipeline.call_kwargs)

            for source, message in (
                (
                    {
                        "samples": frame_first,
                        "sample_rate": 48000,
                        "channels": 2,
                        "sample_layout": "time_major",
                    },
                    "sample_layout metadata must be exactly",
                ),
                (
                    {
                        "samples": frame_first,
                        "sample_rate": 48000,
                        "channels": 2,
                        "sample_layout": "channels_first",
                    },
                    "identifies 3",
                ),
            ):
                pipeline = FakeSourceConditionedAceStepPipeline()
                with self.subTest(source=source):
                    with self.assertRaisesRegex(ValueError, message):
                        Generate().execute(
                            pipeline=pipeline,
                            task_type="cover",
                            source_audio=source,
                            audio_duration=0.01,
                        )
                self.assertIsNone(pipeline.call_kwargs)

            pipeline = FakeSourceConditionedAceStepPipeline()
            with self.assertRaisesRegex(ValueError, "exactly 2 channels.*received 3"):
                Generate().execute(
                    pipeline=pipeline,
                    task_type="cover",
                    source_audio={
                        "samples": np.zeros((3, 2), dtype=np.float32),
                        "sample_rate": 48000,
                        "channels": 3,
                    },
                    audio_duration=0.01,
                )
            self.assertIsNone(pipeline.call_kwargs)

    def test_ace_source_channel_adapter_duplicates_mono_and_rejects_ambiguous_multichannel(self):
        mono = {"samples": np.zeros((1, 480), dtype=np.float32), "sample_rate": 48000}
        cases = (
            ("audio_variation", "cover", "reference_audio", {"audio_duration": 0.01}),
            ("audio_continuation", "continuation", "src_audio", {"extension_duration": 0.01}),
            (
                "audio_repaint",
                "repaint",
                "src_audio",
                {"repainting_start": 0, "repainting_end": 0.01},
            ),
        )
        for mode, task, upstream_field, values in cases:
            pipeline = FakeSourceConditionedAceStepPipeline()
            pipeline._modiff_audio_mode = mode
            node = Generate(f"mono-{mode}")
            node.progress = lambda *args, **kwargs: None

            node.execute(
                pipeline=pipeline,
                task_type=task,
                source_audio=mono,
                sample_rate=48000,
                **values,
            )

            conditioned = pipeline.call_kwargs[upstream_field]
            with self.subTest(mode=mode):
                expected_samples = 960 if mode == "audio_continuation" else 480
                self.assertEqual(tuple(conditioned.shape), (2, expected_samples))
                np.testing.assert_array_equal(conditioned[0].cpu().numpy(), conditioned[1].cpu().numpy())

        for channels in (3, 6):
            pipeline = FakeSourceConditionedAceStepPipeline()
            source = {
                "samples": np.zeros((channels, 480), dtype=np.float32),
                "sample_rate": 48000,
            }
            with self.subTest(channels=channels):
                with self.assertRaisesRegex(ValueError, "exactly 2 channels.*multichannel downmixing"):
                    Generate().execute(
                        pipeline=pipeline,
                        task_type="cover",
                        source_audio=source,
                        audio_duration=0.01,
                    )
            self.assertIsNone(pipeline.call_kwargs)

    def test_variation_rejects_a_distinct_reference_instead_of_overriding_its_required_source(self):
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}
        reference = {"samples": np.ones((1, 480), dtype=np.float32), "sample_rate": 48000}
        self.assertIsNot(source, reference)
        pipeline = FakeSourceConditionedAceStepPipeline()

        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "audio_variation does not accept reference audio"):
                Generate().execute(
                    pipeline=pipeline,
                    task_type="cover",
                    source_audio=source,
                    reference_audio=reference,
                )

        self.assertIsNone(pipeline.call_kwargs)

    def test_forbidden_audio_inputs_are_enforced_per_mode(self):
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}
        forbidden_cases = (
            ("text_to_audio", "text2music", "source_audio"),
            ("text_to_audio", "text2music", "reference_audio"),
            ("audio_variation", "cover", "reference_audio"),
            ("audio_continuation", "continuation", "reference_audio"),
            ("audio_repaint", "repaint", "reference_audio"),
        )
        for mode, task, forbidden_key in forbidden_cases:
            pipeline = FakeSourceConditionedAceStepPipeline()
            pipeline._modiff_audio_mode = mode
            values = {"pipeline": pipeline, "task_type": task, forbidden_key: source}
            if mode != "text_to_audio":
                values["source_audio"] = source
            if mode == "audio_repaint":
                values.update(repainting_start=0, repainting_end=0.01)
            with self.subTest(mode=mode, forbidden=forbidden_key):
                with self.assertRaisesRegex(ValueError, "does not accept"):
                    Generate().execute(**values)
            self.assertIsNone(pipeline.call_kwargs)

        class StableAudioFixture:
            _modiff_audio_pipeline_class = "StableAudioPipeline"

            def __init__(self):
                self.call_kwargs = None

            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                return SimpleNamespace(audios=np.zeros((1, 1, 480), dtype=np.float32))

        for forbidden_key in ("source_audio", "reference_audio"):
            pipeline = StableAudioFixture()
            with self.subTest(pipeline="stable", forbidden=forbidden_key):
                with self.assertRaisesRegex(ValueError, "does not accept"):
                    Generate().execute(pipeline=pipeline, **{forbidden_key: source})
            self.assertIsNone(pipeline.call_kwargs)

    def test_stable_duration_bounds_are_preflighted(self):
        pipeline = SimpleNamespace(_modiff_audio_pipeline_class="StableAudioPipeline")
        for duration in (0, -1, float("nan"), float("inf"), float("-inf"), 47.01):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "greater than 0 and at most 47"):
                    Generate().execute(pipeline=pipeline, audio_duration=duration)

    def test_ace_duration_and_continuation_extension_bounds_are_preflighted(self):
        invalid_durations = (0, -1, float("nan"), float("inf"), float("-inf"), 240.01)
        for duration in invalid_durations:
            pipeline = FakeAceStepPipeline()
            with self.subTest(kind="duration", value=duration):
                with self.assertRaisesRegex(ValueError, "greater than 0 and at most 240"):
                    Generate().execute(pipeline=pipeline, audio_duration=duration)
            self.assertIsNone(pipeline.call_kwargs)

        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}
        invalid_extensions = (0, -1, float("nan"), float("inf"), float("-inf"), 180.01)
        for extension in invalid_extensions:
            pipeline = FakeSourceConditionedAceStepPipeline()
            pipeline._modiff_audio_mode = "audio_continuation"
            with self.subTest(kind="extension", value=extension):
                with self.assertRaisesRegex(ValueError, "continuation extension.*greater than 0 and at most 180"):
                    Generate().execute(
                        pipeline=pipeline,
                        task_type="continuation",
                        source_audio=source,
                        extension_duration=extension,
                    )
            self.assertIsNone(pipeline.call_kwargs)

        pipeline = FakeSourceConditionedAceStepPipeline()
        pipeline._modiff_audio_mode = "audio_continuation"
        long_source = {"samples": np.zeros((2, 61), dtype=np.float32), "sample_rate": 1}
        with self.assertRaisesRegex(ValueError, "source plus extension must be at most 240"):
            Generate().execute(
                pipeline=pipeline,
                task_type="continuation",
                source_audio=long_source,
                extension_duration=180,
            )
        self.assertIsNone(pipeline.call_kwargs)

    def test_continuation_and_repaint_derive_duration_from_validated_source(self):
        source = {"samples": np.zeros((2, 960), dtype=np.float32), "sample_rate": 48000}

        continuation = FakeSourceConditionedAceStepPipeline()
        continuation._modiff_audio_mode = "audio_continuation"
        continuation_node = Generate("audio-derived-continuation")
        continuation_node.progress = lambda *args, **kwargs: None
        continuation_node.execute(
            pipeline=continuation,
            task_type="continuation",
            source_audio=source,
            audio_duration=float("inf"),
            extension_duration=0.01,
            sample_rate=48000,
        )
        self.assertAlmostEqual(continuation.call_kwargs["audio_duration"], 0.03)
        self.assertAlmostEqual(continuation.call_kwargs["repainting_start"], 0.02)
        self.assertAlmostEqual(continuation.call_kwargs["repainting_end"], 0.03)

        class RepaintPipeline(FakeSourceConditionedAceStepPipeline):
            _modiff_audio_mode = "audio_repaint"

            def __call__(self, src_audio=None, **kwargs):
                self.call_kwargs = {**kwargs, "src_audio": src_audio}
                return SimpleNamespace(audios=np.zeros((1, 1, src_audio.shape[-1]), dtype=np.float32))

        repaint = RepaintPipeline()
        repaint_node = Generate("audio-derived-repaint")
        repaint_node.progress = lambda *args, **kwargs: None
        result = repaint_node.execute(
            pipeline=repaint,
            task_type="repaint",
            source_audio=source,
            audio_duration=-1,
            repainting_start=0.01,
            repainting_end=0.02,
            sample_rate=48000,
        )
        self.assertAlmostEqual(repaint.call_kwargs["audio_duration"], 0.02)
        self.assertAlmostEqual(result["duration_seconds"], 0.02)

    def test_valid_ace_modes_route_to_their_declared_upstream_tasks(self):
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}
        cases = (
            ("text_to_audio", "text2music", "text2music"),
            ("audio_variation", "cover", "cover"),
            ("audio_continuation", "continuation", "repaint"),
            ("audio_repaint", "repaint", "repaint"),
        )
        for mode, task, upstream_task in cases:
            pipeline = FakeSourceConditionedAceStepPipeline()
            pipeline._modiff_audio_mode = mode
            node = Generate(f"valid-{mode}")
            node.progress = lambda *args, **kwargs: None
            values = {
                "pipeline": pipeline,
                "task_type": task,
                "audio_duration": 0.01,
                "sample_rate": 48000,
            }
            if mode != "text_to_audio":
                values["source_audio"] = source
            if mode == "audio_continuation":
                values["extension_duration"] = 0.01
            if mode == "audio_repaint":
                values.update(repainting_start=0, repainting_end=0.01)

            node.execute(**values)

            with self.subTest(mode=mode):
                self.assertEqual(pipeline.call_kwargs["task_type"], upstream_task)
                if mode in {"audio_continuation", "audio_repaint"}:
                    self.assertIsNotNone(pipeline.call_kwargs["src_audio"])
                if mode == "audio_variation":
                    self.assertIsNotNone(pipeline.call_kwargs["reference_audio"])

    def test_repaint_interval_is_validated_before_generation(self):
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}
        for start, end, message in ((-0.1, 0.01, "at least 0"), (0.01, 0.01, "strictly greater")):
            pipeline = FakeSourceConditionedAceStepPipeline()
            pipeline._modiff_audio_mode = "audio_repaint"
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, message):
                    Generate().execute(
                        pipeline=pipeline,
                        task_type="repaint",
                        source_audio=source,
                        repainting_start=start,
                        repainting_end=end,
                    )
            self.assertIsNone(pipeline.call_kwargs)

        pipeline = FakeSourceConditionedAceStepPipeline()
        pipeline._modiff_audio_mode = "audio_repaint"
        with self.assertRaisesRegex(ValueError, "exceeds.*source duration"):
            Generate().execute(
                pipeline=pipeline,
                task_type="repaint",
                source_audio=source,
                repainting_start=0,
                repainting_end=1,
            )
        self.assertIsNone(pipeline.call_kwargs)

    def test_invalid_contract_fails_before_torch_import_or_upstream_call(self):
        pipeline = FakeSourceConditionedAceStepPipeline()
        pipeline._modiff_audio_mode = "audio_continuation"
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(ValueError, "requires source audio"):
                Generate().execute(pipeline=pipeline, task_type="continuation")
        self.assertIsNone(pipeline.call_kwargs)

    def test_untagged_pipeline_recovery_is_exact_and_only_safe_for_single_mode(self):
        AceStepPipeline = type("AceStepPipeline", (), {"__call__": lambda self, **kwargs: None})
        with self.assertRaisesRegex(ValueError, "ambiguous across modes"):
            Generate().execute(pipeline=AceStepPipeline())

        class StableAudioPipeline:
            device = "cpu"
            vae = SimpleNamespace(config={"sampling_rate": 44100})

            def __init__(self):
                self.call_kwargs = None

            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                return SimpleNamespace(audios=np.zeros((1, 1, 441), dtype=np.float32))

        pipeline = StableAudioPipeline()
        result = Generate().execute(
            pipeline=pipeline,
            audio_duration=0.01,
            sample_rate=44100,
            stable_audio_steps=2,
        )
        self.assertEqual(pipeline.call_kwargs["audio_end_in_s"], 0.01)
        self.assertEqual(result["sample_rate_out"], 44100)

        class StableAudioSubclass(StableAudioPipeline):
            pass

        with self.assertRaisesRegex(ValueError, "not an exact supported"):
            Generate().execute(pipeline=StableAudioSubclass(), audio_duration=0.01)

    def test_tagged_audio_pipeline_rejects_runtime_class_and_managed_repository_mismatches(self):
        StableAudioPipeline = type(
            "StableAudioPipeline",
            (),
            {
                "_modiff_audio_pipeline_class": "AceStepPipeline",
                "_modiff_audio_mode": "text_to_audio",
            },
        )
        with self.assertRaisesRegex(ValueError, "runtime class StableAudioPipeline.*tagged as AceStepPipeline"):
            Generate().execute(pipeline=StableAudioPipeline(), audio_duration=0.01)

        mismatched_repo = SimpleNamespace(
            _modiff_audio_pipeline_class="StableAudioPipeline",
            _modiff_audio_mode="text_to_audio",
            _modiff_audio_repo=ACE_STEP_DEFAULT_REPO,
        )
        with self.assertRaisesRegex(ValueError, "managed repository.*supports AceStepPipeline.*StableAudioPipeline"):
            Generate().execute(pipeline=mismatched_repo, audio_duration=0.01)

    def test_graph_contract_distinguishes_required_and_optional_audio_inputs(self):
        for node_class in (LoadAdapter, SetAdapters, FuseAdapters, Generate):
            with self.subTest(node=node_class.__name__):
                self.assertTrue(node_class.params["pipeline"]["required"])

        self.assertFalse(Generate.params["source_audio"]["required"])
        self.assertFalse(Generate.params["reference_audio"]["required"])
        default_overlay = AUDIO_PIPELINE_ADAPTERS["AceStepPipeline"].contract_for_mode(
            "text_to_audio"
        ).field_param_overlay()
        for field, params in default_overlay.items():
            for key in ("hidden", "required", "max"):
                if key in params:
                    with self.subTest(field=field, key=key):
                        value = Generate.params[field].get(key, False) if key == "hidden" else Generate.params[field][key]
                        self.assertEqual(value, params[key])
        self.assertTrue(Generate.params["lora_scale"]["hidden"])
        self.assertIn("per-call multiplier", Generate.params["lora_scale"]["description"])
        self.assertIn("ignored by ACE-Step", Generate.params["stable_audio_steps"]["description"])
        self.assertIn("ignored by ACE-Step", Generate.params["stable_audio_guidance"]["description"])
        self.assertIn("ignored by ACE-Step", Generate.params["num_waveforms"]["description"])
        self.assertTrue(LoadPipeline.params["pipeline_class"]["fieldOptions"]["noValidation"])
        self.assertTrue(LoadPipeline.params["mode"]["fieldOptions"]["noValidation"])
        self.assertTrue(Generate.params["task_type"]["fieldOptions"]["noValidation"])
        self.assertEqual(LoadAdapter.params["weight_name"]["default"], "adapter_model.safetensors")
        self.assertIn("literal lowercase .safetensors", LoadAdapter.params["weight_name"]["description"])

    def test_ace_step_lora_load_set_and_fuse_contracts(self):
        events = []

        class Pipeline:
            _modiff_audio_pipeline_class = "AceStepPipeline"

            def unload_lora_weights(self):
                events.append("unload")

            def load_lora_weights(self, path, **kwargs):
                events.append(("load", path, kwargs))

            def set_adapters(self, names, weights):
                events.append(("set", names, weights))

            def fuse_lora(self, **kwargs):
                events.append(("fuse", kwargs))

        pipeline = Pipeline()
        with tempfile.TemporaryDirectory() as temporary:
            adapter_file = Path(temporary) / "adapter_model.safetensors"
            adapter_file.write_bytes(b"audio-lora-fixture")
            LoadAdapter("audio-lora").execute(
                pipeline=pipeline,
                adapter_path={"source": "local", "value": str(adapter_file)},
                adapter_name="style",
                scale=0.6,
            )
        SetAdapters("audio-blend").execute(
            pipeline=pipeline,
            adapter_names="style",
            adapter_weights="0.4",
        )
        FuseAdapters("audio-fuse").execute(pipeline=pipeline, enabled=True, safe_fusing=True)

        self.assertEqual(events[0], "unload")
        self.assertEqual(events[1][0], "load")
        self.assertEqual(events[1][2]["weight_name"], "adapter_model.safetensors")
        self.assertTrue(events[1][2]["use_safetensors"])
        self.assertEqual(events[2], ("set", ["style"], [0.6]))
        self.assertEqual(events[3], ("set", ["style"], [0.4]))
        self.assertEqual(events[4], ("fuse", {"safe_fusing": True}))

    def test_audio_lora_requires_a_literal_lowercase_safetensors_filename_before_cache_or_mutation(self):
        class Pipeline:
            _modiff_audio_pipeline_class = "AceStepPipeline"

            def __init__(self):
                self.events = []

            def unload_lora_weights(self):
                self.events.append("unload")

            def load_lora_weights(self, *args, **kwargs):
                self.events.append(("load", args, kwargs))

        revision = "0123456789abcdef0123456789abcdef01234567"
        digest = "a" * 64
        invalid_names = (
            "weights.bin",
            "weights.BIN",
            "weights.SAFETENSORS",
            "weights.SafeTensors",
            "weights.safetensors.bin",
        )
        for weight_name in invalid_names:
            pipeline = Pipeline()
            with self.subTest(source="hub", weight_name=weight_name):
                with (
                    patch("utils.huggingface.cached_file_path") as cache_lookup,
                    self.assertRaisesRegex(ValueError, "literal lowercase \\.safetensors suffix"),
                ):
                    LoadAdapter().execute(
                        pipeline=pipeline,
                        adapter_path={"source": "hub", "value": "org/audio-lora"},
                        weight_name=weight_name,
                        revision=revision,
                        expected_sha256=digest,
                    )
                cache_lookup.assert_not_called()
            self.assertEqual(pipeline.events, [])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid_files = []
            for weight_name in invalid_names:
                path = root / weight_name
                path.write_bytes(b"unsafe-format-fixture")
                invalid_files.append(path)

            for path in invalid_files:
                pipeline = Pipeline()
                with self.subTest(source="local-file", weight_name=path.name):
                    with self.assertRaisesRegex(ValueError, "literal lowercase \\.safetensors suffix"):
                        LoadAdapter().execute(
                            pipeline=pipeline,
                            adapter_path={"source": "local", "value": str(path)},
                        )
                self.assertEqual(pipeline.events, [])

            local_folder = root / "folder"
            local_folder.mkdir()
            for weight_name in ("weights.bin", "weights.SAFETENSORS"):
                pipeline = Pipeline()
                with self.subTest(source="local-folder", weight_name=weight_name):
                    with self.assertRaisesRegex(ValueError, "literal lowercase \\.safetensors suffix"):
                        LoadAdapter().execute(
                            pipeline=pipeline,
                            adapter_path={"source": "local", "value": str(local_folder)},
                            weight_name=weight_name,
                        )
                self.assertEqual(pipeline.events, [])

            lowercase_file = root / "accepted.safetensors"
            lowercase_file.write_bytes(b"safe-format-fixture")
            valid_pipeline = Pipeline()
            LoadAdapter().execute(
                pipeline=valid_pipeline,
                adapter_path={"source": "local", "value": str(lowercase_file)},
                replace_existing=False,
            )
            self.assertEqual(valid_pipeline.events[0][0], "load")
            self.assertEqual(valid_pipeline.events[0][2]["weight_name"], "accepted.safetensors")
            self.assertTrue(valid_pipeline.events[0][2]["use_safetensors"])

            for cached_name in ("cached.bin", "cached.SAFETENSORS"):
                cached_path = root / cached_name
                cached_path.write_bytes(b"unsafe-cache-alias")
                pipeline = Pipeline()
                with self.subTest(source="hub-cache", cached_name=cached_name):
                    with (
                        patch("utils.huggingface.cached_file_path", return_value=str(cached_path)),
                        patch("utils.huggingface.resolve_managed_hf_cache_file") as resolve_cached,
                        self.assertRaisesRegex(ValueError, "literal lowercase \\.safetensors suffix"),
                    ):
                        LoadAdapter().execute(
                            pipeline=pipeline,
                            adapter_path={"source": "hub", "value": "org/audio-lora"},
                            weight_name="weights.safetensors",
                            revision=revision,
                            expected_sha256=digest,
                        )
                    resolve_cached.assert_not_called()
                self.assertEqual(pipeline.events, [])

            preflight = LoadAdapter("audio-lora-format-preflight")
            preflight.execute = Mock(side_effect=AssertionError("LoRA execution must not run"))
            for selection, weight_name in (
                ({"source": "hub", "value": "org/audio-lora"}, "weights.SAFETENSORS"),
                ({"source": "local", "value": str(root / "weights.SAFETENSORS")}, None),
            ):
                with self.subTest(source="nodebase-preflight", weight_name=weight_name):
                    values = {
                        "pipeline": Pipeline(),
                        "adapter_path": selection,
                        "revision": revision,
                        "expected_sha256": digest,
                    }
                    if weight_name is not None:
                        values["weight_name"] = weight_name
                    with self.assertRaisesRegex(ValueError, "literal lowercase \\.safetensors suffix"):
                        preflight(**values)
            preflight.execute.assert_not_called()

    def test_audio_lora_rejects_non_ace_pipeline(self):
        with self.assertRaisesRegex(ValueError, "AceStepPipeline"):
            LoadAdapter("wrong-audio-lora").execute(
                pipeline=SimpleNamespace(_modiff_audio_pipeline_class="StableAudioPipeline"),
                adapter_path={"source": "local", "value": "/models/audio-style"},
            )

    def test_audio_lora_source_and_local_path_boundaries_fail_before_pipeline_calls(self):
        class Pipeline:
            _modiff_audio_pipeline_class = "AceStepPipeline"

            def __init__(self):
                self.calls = []

            def load_lora_weights(self, *args, **kwargs):
                self.calls.append((args, kwargs))

        for source in (None, "", " hub", "hub ", "remote", False, 0, {}, []):
            pipeline = Pipeline()
            with self.subTest(source=repr(source)):
                with self.assertRaisesRegex(ValueError, "source must be exactly hub or local"):
                    LoadAdapter().execute(
                        pipeline=pipeline,
                        adapter_path={"source": source, "value": "org/not-installed"},
                        replace_existing=False,
                    )
            self.assertEqual(pipeline.calls, [])

        pipeline = Pipeline()
        for selection in (None, "", "   ", {"source": "local", "value": ""}, {"source": "hub", "value": " "}):
            with self.subTest(selection=repr(selection)):
                with self.assertRaisesRegex(ValueError, "Audio LoRA (selection|repository ID or local path)"):
                    LoadAdapter().execute(
                        pipeline=pipeline,
                        adapter_path=selection,
                        replace_existing=False,
                    )
            self.assertEqual(pipeline.calls, [])

        with self.assertRaisesRegex(FileNotFoundError, "path does not exist"):
            LoadAdapter().execute(
                pipeline=pipeline,
                adapter_path="org/not-installed",
                replace_existing=False,
            )
        self.assertEqual(pipeline.calls, [])

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            adapter_file = temporary_path / "installed.safetensors"
            adapter_file.write_bytes(b"installed-audio-lora")

            hub_pipeline = Pipeline()
            hub_revision = "0123456789abcdef0123456789abcdef01234567"
            expected_sha256 = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
            with (
                patch("utils.huggingface.cached_file_path", return_value=str(adapter_file)) as cached,
                patch(
                    "utils.huggingface.resolve_managed_hf_cache_file",
                    return_value=adapter_file,
                ) as resolve_cached,
            ):
                LoadAdapter().execute(
                    pipeline=hub_pipeline,
                    adapter_path={"source": "HUB", "value": "org/installed"},
                    weight_name="installed.safetensors",
                    revision=hub_revision,
                    expected_sha256=expected_sha256,
                    replace_existing=False,
                )
            cached.assert_called_once_with(
                "org/installed",
                "installed.safetensors",
                revision=hub_revision,
            )
            resolve_cached.assert_called_once_with(str(adapter_file))
            self.assertEqual(len(hub_pipeline.calls), 1)

            hash_mismatch_pipeline = Pipeline()
            with (
                patch("utils.huggingface.cached_file_path", return_value=str(adapter_file)),
                patch("utils.huggingface.resolve_managed_hf_cache_file", return_value=adapter_file),
            ):
                with self.assertRaisesRegex(ValueError, "pinned SHA-256 verification"):
                    LoadAdapter().execute(
                        pipeline=hash_mismatch_pipeline,
                        adapter_path={"source": "hub", "value": "org/installed"},
                        weight_name="installed.safetensors",
                        revision=hub_revision,
                        expected_sha256="0" * 64,
                        replace_existing=False,
                    )
            self.assertEqual(hash_mismatch_pipeline.calls, [])

            escaped_cache_pipeline = Pipeline()
            managed_cache = temporary_path / "managed-cache"
            managed_cache.mkdir()
            with (
                patch("utils.huggingface.cached_file_path", return_value=str(adapter_file)),
                patch.dict("utils.huggingface.CONFIG.hf", {"cache_dir": str(managed_cache)}),
            ):
                with self.assertRaisesRegex(ValueError, "outside the managed cache root"):
                    LoadAdapter().execute(
                        pipeline=escaped_cache_pipeline,
                        adapter_path={"source": "hub", "value": "org/installed"},
                        weight_name="installed.safetensors",
                        revision=hub_revision,
                        expected_sha256=expected_sha256,
                        replace_existing=False,
                    )
            self.assertEqual(escaped_cache_pipeline.calls, [])

            for revision, expected_hash in (
                (None, expected_sha256),
                ("main", expected_sha256),
                (hub_revision.upper(), expected_sha256),
                (hub_revision, ""),
                (hub_revision, "not-a-sha256"),
            ):
                rejected_pipeline = Pipeline()
                with self.subTest(hub_revision=revision, expected_hash=expected_hash):
                    with patch("utils.huggingface.cached_file_path") as rejected_cache:
                        with self.assertRaisesRegex(ValueError, "immutable lowercase|exact lowercase|SHA-256"):
                            LoadAdapter().execute(
                                pipeline=rejected_pipeline,
                                adapter_path={"source": "hub", "value": "org/installed"},
                                weight_name="installed.safetensors",
                                revision=revision,
                                expected_sha256=expected_hash,
                                replace_existing=False,
                            )
                    rejected_cache.assert_not_called()
                self.assertEqual(rejected_pipeline.calls, [])

            for weight_name in (
                "../installed.safetensors",
                "/installed.safetensors",
                "nested\\..\\installed.safetensors",
                "nested//installed.safetensors",
            ):
                rejected_pipeline = Pipeline()
                with self.subTest(hub_weight_name=weight_name):
                    with patch("utils.huggingface.cached_file_path") as rejected_cache:
                        with self.assertRaisesRegex(ValueError, "relative Hub file path without traversal"):
                            LoadAdapter().execute(
                                pipeline=rejected_pipeline,
                                adapter_path={"source": "hub", "value": "org/installed"},
                                weight_name=weight_name,
                                revision=hub_revision,
                                expected_sha256=expected_sha256,
                                replace_existing=False,
                            )
                    rejected_cache.assert_not_called()
                self.assertEqual(rejected_pipeline.calls, [])

            local_pipeline = Pipeline()
            LoadAdapter().execute(
                pipeline=local_pipeline,
                adapter_path=str(adapter_file),
                replace_existing=False,
            )
            self.assertEqual(len(local_pipeline.calls), 1)
            self.assertEqual(local_pipeline.calls[0][1]["weight_name"], "installed.safetensors")

            from utils.huggingface import cached_file_path

            with patch("utils.huggingface.try_to_load_from_cache", return_value=str(adapter_file)) as cache_lookup:
                self.assertEqual(
                    cached_file_path("org/installed", "installed.safetensors", revision=hub_revision),
                    str(adapter_file),
                )
            self.assertEqual(cache_lookup.call_args.kwargs["revision"], hub_revision)

            real_node_pipeline = Pipeline()
            real_node = LoadAdapter("audio-lora-real-node-local")
            real_node.execute = Mock(return_value={"output": real_node_pipeline})
            with chdir(temporary), patch("modiff.NodeBase.modelstore.is_local_cached", return_value=True):
                first = real_node(
                    pipeline=real_node_pipeline,
                    adapter_path=adapter_file.name,
                    replace_existing=False,
                )
                second = real_node(
                    pipeline=real_node_pipeline,
                    adapter_path=str(adapter_file),
                    replace_existing=False,
                )
            self.assertIs(first, second)
            real_node.execute.assert_called_once()
            self.assertEqual(
                real_node.execute.call_args.kwargs["adapter_path"],
                {"source": "local", "value": str(adapter_file)},
            )

            invalid_real_node = LoadAdapter("audio-lora-real-node-invalid")
            invalid_real_node.execute = Mock()
            for invalid in (None, "", {"source": "", "value": str(adapter_file)}):
                with self.subTest(real_node_selection=repr(invalid)):
                    with self.assertRaises(ValueError):
                        invalid_real_node(
                            pipeline=real_node_pipeline,
                            adapter_path=invalid,
                            replace_existing=False,
                        )
            invalid_real_node.execute.assert_not_called()

            adapter_folder = temporary_path / "folder"
            adapter_folder.mkdir()
            traversal_pipeline = Pipeline()
            with self.assertRaisesRegex(FileNotFoundError, "inside the selected folder"):
                LoadAdapter().execute(
                    pipeline=traversal_pipeline,
                    adapter_path={"source": "local", "value": str(adapter_folder)},
                    weight_name="../installed.safetensors",
                    replace_existing=False,
                )
            self.assertEqual(traversal_pipeline.calls, [])

    def test_audio_lora_identity_is_validated_before_real_nodebase_execution(self):
        pipeline = SimpleNamespace(_modiff_audio_pipeline_class="AceStepPipeline")
        selection = {"source": "hub", "value": "organization/custom-audio-lora"}
        revision = "0123456789abcdef0123456789abcdef01234567"
        digest = "a" * 64
        invalid = LoadAdapter("strict-audio-lora-identity")
        invalid.execute = Mock(side_effect=AssertionError("LoRA upstream must not run"))

        for values in (
            {"revision": None, "expected_sha256": digest},
            {"revision": "main", "expected_sha256": digest},
            {"revision": revision.upper(), "expected_sha256": digest},
            {"revision": revision, "expected_sha256": ""},
            {"revision": revision, "expected_sha256": "not-a-digest"},
        ):
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, "immutable lowercase|exact lowercase|SHA-256"):
                    invalid(
                        pipeline=pipeline,
                        adapter_path=selection,
                        replace_existing=False,
                        **values,
                    )
        invalid.execute.assert_not_called()

        valid = LoadAdapter("strict-audio-lora-valid")
        valid.execute = Mock(return_value={"output": pipeline})
        with patch("modiff.NodeBase.modelstore.is_hf_cached", return_value=True):
            valid(
                pipeline=pipeline,
                adapter_path=selection,
                revision=revision,
                expected_sha256=f"SHA256:{digest.upper()}",
                replace_existing=False,
            )
        self.assertEqual(valid.execute.call_args.kwargs["revision"], revision)
        self.assertEqual(valid.execute.call_args.kwargs["expected_sha256"], digest)

    def test_hub_audio_lora_cannot_resolve_as_a_local_directory(self):
        pipeline = SimpleNamespace(_modiff_audio_pipeline_class="AceStepPipeline")
        node = LoadAdapter("audio-lora-hub-local-path-boundary")
        node.execute = Mock(side_effect=AssertionError("LoRA upstream must not run"))
        revision = "0123456789abcdef0123456789abcdef01234567"

        with tempfile.TemporaryDirectory() as temporary:
            local_repo = Path(temporary) / "organization" / "local-audio-lora"
            local_repo.mkdir(parents=True)
            with chdir(temporary):
                with self.assertRaisesRegex(ValueError, "local filesystem"):
                    node(
                        pipeline=pipeline,
                        adapter_path={"source": "hub", "value": "organization/local-audio-lora"},
                        revision=revision,
                        expected_sha256="a" * 64,
                        replace_existing=False,
                    )

        node.execute.assert_not_called()

    def test_stable_audio_uses_native_generation_rate_and_requested_delivery_rate(self):
        class FakeStableAudio:
            _modiff_audio_pipeline_class = "StableAudioPipeline"
            device = "cpu"
            vae = SimpleNamespace(config={"sampling_rate": 44100})

            def __init__(self):
                self.call_kwargs = None

            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                return SimpleNamespace(audios=np.zeros((2, 2, 44100), dtype=np.float32))

        pipeline = FakeStableAudio()
        result = Generate().execute(
            pipeline=pipeline,
            prompt="Clear wooden impacts in a quiet room",
            negative_prompt="low quality",
            audio_duration=1,
            stable_audio_steps=120,
            stable_audio_guidance=0,
            num_waveforms=2,
            sample_rate=48000,
        )

        self.assertEqual(pipeline.call_kwargs["audio_end_in_s"], 1)
        self.assertEqual(pipeline.call_kwargs["num_inference_steps"], 120)
        self.assertEqual(pipeline.call_kwargs["guidance_scale"], 0)
        self.assertEqual(pipeline.call_kwargs["num_waveforms_per_prompt"], 2)
        self.assertEqual(result["sample_rate_out"], 48000)
        self.assertEqual(result["duration_seconds"], 1)
        self.assertEqual(result["audio"]["samples"].shape[-1], 48000)
        self.assertEqual(len(result["audio_variations"]), 2)
        self.assertTrue(all(item["samples"].shape == (2, 48000) for item in result["audio_variations"]))
        self.assertIs(result["audio"], result["audio_variations"][0])

    def test_explicit_zero_audio_controls_are_preserved_only_where_upstream_allows_them(self):
        pipeline = FakeSourceConditionedAceStepPipeline()
        node = Generate("ace-zero-values-test")
        node.progress = lambda *args, **kwargs: None
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}

        result = node.execute(
            pipeline=pipeline,
            task_type="cover",
            source_audio=source,
            prompt="A silent control fixture",
            audio_duration=0.01,
            guidance_scale=0,
            shift=0.1,
            lora_scale=0,
            audio_cover_strength=0,
        )

        self.assertEqual(pipeline.call_kwargs["guidance_scale"], 0)
        self.assertEqual(pipeline.call_kwargs["shift"], 0.1)
        self.assertEqual(pipeline.call_kwargs["attention_kwargs"]["scale"], 0)
        self.assertEqual(pipeline.call_kwargs["audio_cover_strength"], 0)
        self.assertEqual(result["audio_variations"], [result["audio"]])

    def test_audio_numeric_resource_bounds_fail_before_upstream(self):
        raw_node = Generate("audio-raw-integer-bounds")
        raw_node.execute = Mock()
        for field, value in (("num_inference_steps", 1.5), ("stable_audio_steps", True), ("num_waveforms", "1.5")):
            with self.subTest(raw_field=field, raw_value=value):
                with self.assertRaisesRegex(ValueError, "exact finite integer"):
                    raw_node(pipeline=FakeAceStepPipeline(), **{field: value})
        raw_node.execute.assert_not_called()

        ace_cases = (
            ({"num_inference_steps": 0}, "inference steps"),
            ({"num_inference_steps": 101}, "inference steps"),
            ({"guidance_scale": -0.1}, "guidance"),
            ({"guidance_scale": float("nan")}, "guidance"),
            ({"shift": 0}, "shift"),
            ({"shift": 10.1}, "shift"),
            ({"lora_scale": -0.1}, "LoRA call strength"),
            ({"lora_scale": float("inf")}, "LoRA call strength"),
            ({"bpm": 401}, "BPM"),
            ({"bpm": 170.5}, "BPM"),
            ({"seed": -1}, "seed"),
        )
        for values, message in ace_cases:
            pipeline = FakeAceStepPipeline()
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    Generate().execute(pipeline=pipeline, audio_duration=0.01, **values)
            self.assertIsNone(pipeline.call_kwargs)

        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}
        for strength in (-0.1, 1.1, float("nan")):
            pipeline = FakeSourceConditionedAceStepPipeline()
            with self.subTest(cover_strength=strength):
                with self.assertRaisesRegex(ValueError, "cover strength"):
                    Generate().execute(
                        pipeline=pipeline,
                        task_type="cover",
                        source_audio=source,
                        audio_duration=0.01,
                        audio_cover_strength=strength,
                    )
            self.assertIsNone(pipeline.call_kwargs)

        stable_cases = (
            ({"stable_audio_steps": 0}, "Stable Audio steps"),
            ({"stable_audio_steps": 301}, "Stable Audio steps"),
            ({"stable_audio_guidance": -0.1}, "Stable Audio guidance"),
            ({"stable_audio_guidance": float("inf")}, "Stable Audio guidance"),
            ({"num_waveforms": 0}, "Stable Audio variations"),
            ({"num_waveforms": 9}, "Stable Audio variations"),
        )
        for values, message in stable_cases:
            pipeline = SimpleNamespace(_modiff_audio_pipeline_class="StableAudioPipeline")
            with self.subTest(values=values):
                with self.assertRaisesRegex(ValueError, message):
                    Generate().execute(pipeline=pipeline, audio_duration=0.01, **values)

        self.assertEqual(Generate.params["shift"]["min"], 0.1)

    def test_audio_lora_scales_are_finite_and_bounded_before_mutation(self):
        class Pipeline:
            _modiff_audio_pipeline_class = "AceStepPipeline"

            def __init__(self):
                self.events = []

            def unload_lora_weights(self):
                self.events.append("unload")

            def load_lora_weights(self, *args, **kwargs):
                self.events.append("load")

            def set_adapters(self, *args, **kwargs):
                self.events.append("set")

        with tempfile.TemporaryDirectory() as temporary:
            adapter_file = Path(temporary) / "adapter_model.safetensors"
            adapter_file.write_bytes(b"bounded-audio-lora")
            for scale in (-0.1, 2.1, float("nan"), float("inf")):
                pipeline = Pipeline()
                with self.subTest(scale=scale):
                    with self.assertRaisesRegex(ValueError, "Audio LoRA strength"):
                        LoadAdapter().execute(
                            pipeline=pipeline,
                            adapter_path=str(adapter_file),
                            scale=scale,
                        )
                self.assertEqual(pipeline.events, [])

        pipeline = Pipeline()
        for weights in ("nan", "inf", "-0.1", "2.1"):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "finite values from 0 through 2"):
                    SetAdapters().execute(
                        pipeline=pipeline,
                        adapter_names="style",
                        adapter_weights=weights,
                    )
        self.assertEqual(pipeline.events, [])

    def test_unsigned_pcm_midpoint_normalizes_to_zero(self):
        samples, _sample_rate = audio_to_numpy(
            {"samples": np.asarray([0, 128, 255], dtype=np.uint8), "sample_rate": 8000}
        )

        np.testing.assert_allclose(samples[0], [-1.0, 0.0, 127 / 128], atol=1e-7)

    def test_audio_frame_tail_is_cropped_to_requested_duration(self):
        audio = {
            "samples": np.zeros((2, 24000 * 2), dtype=np.float32),
            "sample_rate": 24000,
            "duration_seconds": 2.0,
        }
        cropped = crop_tail(audio, 0.0, 1.0)
        self.assertEqual(cropped["samples"].shape, (2, 24000))
        self.assertEqual(cropped["duration_seconds"], 1.0)

    def test_shared_audio_contract_serializes_to_wav_bytes(self):
        encoded = to_bytes(
            "audio",
            {"samples": np.asarray([[0.0, 0.5, -0.5]], dtype=np.float32), "sample_rate": 24000},
        )
        self.assertEqual(encoded[:4], b"RIFF")
        self.assertEqual(encoded[8:12], b"WAVE")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audio.wav"
            output.write_bytes(encoded)
            self.assertEqual(to_bytes("audio", output), encoded)

    def test_loader_rejects_unsupported_mode_before_resolving_pipeline(self):
        node = LoadPipeline("ace-mode-test")
        with self.assertRaisesRegex(ValueError, "does not support video_to_video"):
            node.execute(pipeline_class="AceStepPipeline", mode="video_to_video")

    def test_no_offload_audio_pipeline_loads_directly_on_cuda(self):
        loaded = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        node = LoadPipeline("ace-direct-load-test")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        with (
            patch("modules.DiffusersAudio.main.pipeline_class_from_name", return_value=FakePipeline),
            patch("modules.DiffusersAudio.main.apply_pipeline_offload"),
        ):
            node.execute(
                model_id="org/ace-step",
                pipeline_class="AceStepPipeline",
                mode="text_to_audio",
                revision="0123456789abcdef0123456789abcdef01234567",
                device="cuda:0",
                auto_offload=False,
                offload_mode="none",
            )

        self.assertEqual(loaded["repo"], "org/ace-step")
        self.assertEqual(loaded["kwargs"]["device_map"], "cuda")
        self.assertEqual(loaded["kwargs"]["revision"], "0123456789abcdef0123456789abcdef01234567")

    def test_curated_audio_pipeline_uses_catalog_revision(self):
        loaded = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, repo, **kwargs):
                loaded.update({"repo": repo, "kwargs": kwargs})
                return cls()

        node = LoadPipeline("ace-revision-test")
        node.progress = lambda *args, **kwargs: None
        node.mm_add = lambda *args, **kwargs: None
        for source in ("hub", "HUB", "Hub"):
            with self.subTest(source=source):
                with (
                    patch("modules.DiffusersAudio.main.pipeline_class_from_name", return_value=FakePipeline),
                    patch("modules.DiffusersAudio.main.apply_pipeline_offload"),
                ):
                    node.execute(
                        model_id={"source": source, "value": ACE_STEP_DEFAULT_REPO},
                        pipeline_class="AceStepPipeline",
                        mode="text_to_audio",
                        device="cpu",
                        auto_offload=False,
                        offload_mode="none",
                    )

                self.assertEqual(
                    loaded["kwargs"]["revision"],
                    "200ba991ae448051e14b0183157e35c2d27c9fb0",
                )

    def test_xl_turbo_schema_uses_distilled_defaults(self):
        steps = Generate.params["num_inference_steps"]
        guidance = Generate.params["guidance_scale"]

        self.assertEqual(steps["default"], 8)
        self.assertEqual(guidance["default"], 1.0)
        self.assertIn("8 denoising steps", steps["description"])
        self.assertIn("guidance-distilled", guidance["description"])
        self.assertIn("above 1", guidance["description"])

    def test_generate_sample_rate_is_a_four_option_delivery_selector(self):
        self.assertEqual(
            Generate.params["sample_rate"]["options"],
            AUDIO_SAMPLE_RATE_OPTIONS,
        )

    def test_ace_output_is_resampled_to_requested_delivery_rate(self):
        pipeline = FakeAceStepPipeline()
        node = Generate("ace-sample-rate-test")
        node.progress = lambda *args, **kwargs: None

        result = node.execute(
            pipeline=pipeline,
            prompt="A short instrumental cue",
            audio_duration=0.01,
            sample_rate=96000,
        )

        self.assertEqual(result["sample_rate_out"], 96000)
        self.assertEqual(result["audio"]["sample_rate"], 96000)
        self.assertEqual(result["audio"]["samples"].shape[-1], 960)

    def test_execute_forwards_xl_turbo_defaults_when_values_are_omitted(self):
        pipeline = FakeAceStepPipeline()
        node = Generate("ace-defaults-test")
        node.progress = lambda *args, **kwargs: None

        node.execute(pipeline=pipeline, prompt="A short instrumental cue", audio_duration=1, sample_rate=48000)

        self.assertIsNotNone(pipeline.call_kwargs)
        self.assertEqual(pipeline.call_kwargs["num_inference_steps"], 8)
        self.assertEqual(pipeline.call_kwargs["guidance_scale"], 1.0)

    def test_execute_reports_indeterminate_progress_before_ace_pipeline_starts(self):
        events = []

        class ProgressAwarePipeline(FakeAceStepPipeline):
            def __call__(self, bpm=None, **kwargs):
                events.append(("pipeline",))
                return super().__call__(bpm=bpm, **kwargs)

        pipeline = ProgressAwarePipeline()
        node = Generate("ace-initial-progress-test")
        node.progress = lambda value, **metadata: events.append(("progress", value, metadata))

        node.execute(
            pipeline=pipeline,
            prompt="A short instrumental cue",
            audio_duration=1,
            num_inference_steps=8,
            sample_rate=48000,
        )

        self.assertEqual(events[0][0], "progress")
        self.assertEqual(events[0][1], -1)
        self.assertEqual(events[0][2]["phase"], "denoising")
        self.assertEqual(events[0][2]["message"], "Generating audio (text2music)")
        self.assertEqual(events[0][2]["current_step"], 0)
        self.assertEqual(events[0][2]["total_steps"], 8)
        self.assertEqual(events[1], ("pipeline",))

    def test_execute_normalizes_string_bpm_for_ace_metadata(self):
        pipeline = FakeAceStepPipeline()
        node = Generate("ace-bpm-test")
        node.progress = lambda *args, **kwargs: None

        node.execute(
            pipeline=pipeline,
            prompt="A short instrumental cue",
            audio_duration=1,
            sample_rate=48000,
            bpm="170",
        )

        self.assertEqual(pipeline.call_kwargs["bpm"], 170)
        self.assertIsInstance(pipeline.call_kwargs["bpm"], int)

    def test_cover_routes_source_track_to_reference_audio_without_optional_audio_code_modules(self):
        pipeline = FakeSourceConditionedAceStepPipeline()
        node = Generate("ace-cover-reference-test")
        node.progress = lambda *args, **kwargs: None
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}

        node.execute(
            pipeline=pipeline,
            task_type="cover",
            source_audio=source,
            prompt="A restrained acoustic variation",
            audio_duration=0.01,
        )

        self.assertIsNone(pipeline.call_kwargs["src_audio"])
        self.assertIsNotNone(pipeline.call_kwargs["reference_audio"])

    def test_repaint_routes_source_track_to_src_audio_without_implicit_timbre_reference(self):
        pipeline = FakeSourceConditionedAceStepPipeline()
        pipeline._modiff_audio_mode = "audio_repaint"
        node = Generate("ace-repaint-source-test")
        node.progress = lambda *args, **kwargs: None
        source = {"samples": np.zeros((2, 480), dtype=np.float32), "sample_rate": 48000}

        node.execute(
            pipeline=pipeline,
            task_type="repaint",
            source_audio=source,
            prompt="Repair the selected interval",
            audio_duration=0.01,
            repainting_start=0,
            repainting_end=0.01,
        )

        self.assertIsNotNone(pipeline.call_kwargs["src_audio"])
        self.assertIsNone(pipeline.call_kwargs["reference_audio"])

    def test_audio_input_is_resampled_to_the_pipeline_native_rate(self):
        source = {"samples": np.zeros((2, 44100), dtype=np.float32), "sample_rate": 44100}

        tensor = audio_to_tensor(source, device="cpu", target_sample_rate=48000)

        self.assertEqual(tuple(tensor.shape), (2, 48000))


if __name__ == "__main__":
    unittest.main()
