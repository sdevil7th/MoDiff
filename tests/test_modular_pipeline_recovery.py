import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from diffusers import QwenImageEditPlusModularPipeline

from modules.ModularDiffusers.modular_utils import (
    DummyCustomPipeline,
    pin_modular_component_revisions,
    pipeline_class_from_runtime_inputs,
    require_immutable_hub_revision,
)
from modules.ModularDiffusers.denoise import Denoise


class ModularPipelineRecoveryTests(unittest.TestCase):
    def test_modular_component_specs_require_reviewed_auxiliary_pins(self):
        primary = SimpleNamespace(
            pretrained_model_name_or_path="Tongyi-MAI/Z-Image-Turbo",
            revision=None,
        )
        known_auxiliary = SimpleNamespace(
            pretrained_model_name_or_path="lllyasviel/flux_redux_bfl",
            revision=None,
        )
        unknown_auxiliary = SimpleNamespace(
            pretrained_model_name_or_path="user/custom-component",
            revision=None,
        )
        pipeline = SimpleNamespace(
            _component_specs={
                "transformer": primary,
                "image_encoder": known_auxiliary,
                "custom": unknown_auxiliary,
            }
        )

        with self.assertRaisesRegex(ValueError, "40-character"):
            pin_modular_component_revisions(
                pipeline,
                "Tongyi-MAI/Z-Image-Turbo",
                "f332072aa78be7aecdf3ee76d5c247082da564a6",
            )

        self.assertIsNone(primary.revision)
        self.assertIsNone(known_auxiliary.revision)
        self.assertIsNone(unknown_auxiliary.revision)

    def test_modular_component_specs_accept_explicit_auxiliary_commit(self):
        explicit_revision = "c" * 40
        primary = SimpleNamespace(pretrained_model_name_or_path="owner/base", revision=None)
        auxiliary = SimpleNamespace(pretrained_model_name_or_path="owner/component", revision=explicit_revision)
        pipeline = SimpleNamespace(_component_specs={"transformer": primary, "auxiliary": auxiliary})

        applied = pin_modular_component_revisions(pipeline, "owner/base", "d" * 40)

        self.assertEqual(primary.revision, "d" * 40)
        self.assertEqual(auxiliary.revision, explicit_revision)
        self.assertEqual(applied, {"transformer": "d" * 40, "auxiliary": explicit_revision})

    def tearDown(self):
        DummyCustomPipeline.repo_id = None
        DummyCustomPipeline.revision = None
        DummyCustomPipeline.trust_remote_code = False

    def test_remote_code_requires_an_immutable_commit_revision(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            require_immutable_hub_revision("owner/custom-pipeline", "main", required=True)
        revision = "a" * 40
        self.assertEqual(
            require_immutable_hub_revision("owner/custom-pipeline", revision, required=True),
            revision,
        )

    def test_dummy_custom_pipeline_never_silently_enables_remote_code(self):
        DummyCustomPipeline.repo_id = "owner/custom-pipeline"
        with patch("diffusers.ModularPipeline.from_pretrained", return_value="pipeline") as loader:
            self.assertEqual(DummyCustomPipeline(), "pipeline")

        loader.assert_called_once_with(
            "owner/custom-pipeline",
            trust_remote_code=False,
            local_files_only=True,
        )

    def test_dummy_custom_pipeline_propagates_explicit_trust_and_revision(self):
        DummyCustomPipeline.repo_id = "owner/custom-pipeline"
        DummyCustomPipeline.trust_remote_code = True
        with patch("diffusers.ModularPipeline.from_pretrained") as loader:
            with self.assertRaisesRegex(ValueError, "40-character"):
                DummyCustomPipeline()
            loader.assert_not_called()

            DummyCustomPipeline.revision = "b" * 40
            loader.return_value = "pipeline"
            self.assertEqual(DummyCustomPipeline(), "pipeline")

        loader.assert_called_once_with(
            "owner/custom-pipeline",
            trust_remote_code=True,
            local_files_only=True,
            revision="b" * 40,
        )

    def test_dynamic_denoise_declares_its_stable_model_input_as_required(self):
        self.assertTrue(Denoise.params["unet"]["required"])

    def test_modular_denoise_honors_interrupt_at_step_boundary(self):
        node = Denoise("interrupt-probe")
        node._interrupt = True

        with self.assertRaisesRegex(InterruptedError, "interrupted by the user"):
            node._raise_if_interrupted()

    def test_modular_denoise_publishes_measurable_progress_before_the_first_step(self):
        node = Denoise("progress-probe")
        node.progress = Mock()

        node._publish_initial_denoise_progress(50)

        node.progress.assert_called_once_with(
            0,
            phase="denoising",
            message="Denoising 0/50",
            current_step=0,
            total_steps=50,
            elapsed_seconds=0.0,
            average_step_seconds=None,
            eta_seconds=None,
        )

    def test_preserves_pipeline_class_set_by_dynamic_signal(self):
        self.assertIs(
            pipeline_class_from_runtime_inputs(QwenImageEditPlusModularPipeline, {}),
            QwenImageEditPlusModularPipeline,
        )

    def test_recovers_pipeline_class_from_nested_loader_output(self):
        runtime_inputs = {
            "text_encoders": {
                "text_encoder": {"component": object()},
                "repo_id": "Qwen/Qwen-Image-Edit-2511",
                "model_type": "QwenImageEditPlusModularPipeline",
            }
        }
        self.assertIs(
            pipeline_class_from_runtime_inputs(None, runtime_inputs),
            QwenImageEditPlusModularPipeline,
        )

    def test_recovers_custom_pipeline_marker(self):
        self.assertIs(
            pipeline_class_from_runtime_inputs(
                None,
                {
                    "model_type": "DummyCustomPipeline",
                    "repo_id": "owner/custom-pipeline",
                    "revision": "c" * 40,
                    "trust_remote_code": True,
                },
            ),
            DummyCustomPipeline,
        )
        self.assertEqual(DummyCustomPipeline.repo_id, "owner/custom-pipeline")
        self.assertEqual(DummyCustomPipeline.revision, "c" * 40)
        self.assertTrue(DummyCustomPipeline.trust_remote_code)

    def test_custom_pipeline_recovery_rejects_missing_trust_metadata(self):
        with self.assertRaisesRegex(ValueError, "trust metadata"):
            pipeline_class_from_runtime_inputs(None, {"model_type": "DummyCustomPipeline"})

    def test_rejects_mixed_model_inputs_before_loading(self):
        with self.assertRaisesRegex(ValueError, "incompatible pipeline classes"):
            pipeline_class_from_runtime_inputs(
                None,
                {"model_type": "QwenImageEditPlusModularPipeline"},
                {"model_type": "FluxModularPipeline"},
            )

    def test_unknown_pipeline_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "Install a Diffusers version"):
            pipeline_class_from_runtime_inputs(None, {"model_type": "FuturePipeline"})


if __name__ == "__main__":
    unittest.main()
