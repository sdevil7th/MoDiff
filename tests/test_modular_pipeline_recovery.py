import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from diffusers import FluxModularPipeline, QwenImageEditPlusModularPipeline

from modules.ModularDiffusers.modular_utils import (
    DummyCustomPipeline,
    pin_modular_component_revisions,
    pipeline_class_from_runtime_inputs,
    require_immutable_hub_revision,
)
from modules.ModularDiffusers.denoise import Denoise
from modules.ModularDiffusers.embeddings import EncodePrompt


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

    def test_remote_code_requires_an_immutable_commit_revision(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            require_immutable_hub_revision("owner/custom-pipeline", "main", required=True)
        revision = "a" * 40
        self.assertEqual(
            require_immutable_hub_revision("owner/custom-pipeline", revision, required=True),
            revision,
        )

    def test_dummy_custom_pipeline_is_an_unbound_non_executable_registry_marker(self):
        with patch("diffusers.ModularPipeline.from_pretrained") as loader:
            with self.assertRaisesRegex(ValueError, "contract_only.*verified contract checksum"):
                DummyCustomPipeline()
            loader.assert_not_called()

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

    def test_selected_pipeline_class_must_match_connected_runtime_components(self):
        with self.assertRaisesRegex(
            ValueError,
            "configured for pipeline class 'QwenImageEditPlusModularPipeline'.*identify 'FluxModularPipeline'",
        ):
            pipeline_class_from_runtime_inputs(
                QwenImageEditPlusModularPipeline,
                {"model_type": "FluxModularPipeline"},
            )

    def test_matching_selected_and_runtime_pipeline_class_is_preserved(self):
        self.assertIs(
            pipeline_class_from_runtime_inputs(
                QwenImageEditPlusModularPipeline,
                {"model_type": "QwenImageEditPlusModularPipeline"},
            ),
            QwenImageEditPlusModularPipeline,
        )

    def test_denoise_synchronizes_model_type_after_runtime_recovery(self):
        node = Denoise("runtime-model-type")
        with patch(
            "modules.ModularDiffusers.denoise.require_modiff_node_contract",
            side_effect=RuntimeError("stop after recovery"),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after recovery"):
                node.execute(
                    unet={
                        "model_type": FluxModularPipeline.__name__,
                        "repo_id": "local/fixture",
                    }
                )

        self.assertIs(node._pipeline_class, FluxModularPipeline)
        self.assertEqual(node._model_type, FluxModularPipeline.__name__)

    def test_downstream_blocks_do_not_reload_repository_config_before_component_injection(self):
        node = EncodePrompt("reviewed-block-construction")
        pipeline = Mock()
        state = Mock()
        state.get_by_kwargs.return_value = {"prompt_embeds": "encoded"}
        pipeline.return_value = state
        blocks = Mock()
        blocks.component_names = ["text_encoder"]
        blocks.input_names = ["prompt"]

        def init_without_repository(*args, **kwargs):
            self.assertEqual(args, (), "a repository argument would make upstream reload model config")
            self.assertIn("components_manager", kwargs)
            return pipeline

        blocks.init_pipeline.side_effect = init_without_repository
        node_config = {
            "params": {},
            "model_input_names": ["text_encoders"],
            "input_names": ["prompt"],
            "output_names": ["embeddings"],
        }
        managed_component = object()

        with (
            patch(
                "modules.ModularDiffusers.embeddings.require_modiff_node_contract",
                return_value=(blocks, node_config),
            ),
            patch("modules.ModularDiffusers.embeddings.collect_model_ids", return_value=["text-encoder-id"]),
            patch(
                "modules.ModularDiffusers.embeddings.components.get_components_by_ids",
                return_value={"text_encoder": managed_component},
            ),
        ):
            outputs = node.execute(
                text_encoders={
                    "repo_id": "attacker/reloaded-config",
                    "model_type": FluxModularPipeline.__name__,
                },
                prompt="test prompt",
            )

        self.assertEqual(outputs, {"embeddings": {"prompt_embeds": "encoded"}})
        pipeline.update_components.assert_called_once_with(text_encoder=managed_component)

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

    def test_legacy_custom_pipeline_marker_is_not_treated_as_an_execution_identity(self):
        with self.assertRaisesRegex(ValueError, "backend-issued contract identity"):
            pipeline_class_from_runtime_inputs(
                None,
                {
                    "model_type": "DummyCustomPipeline",
                    "repo_id": "owner/custom-pipeline",
                    "revision": "c" * 40,
                    "trust_remote_code": True,
                },
            )

    def test_custom_pipeline_recovery_rejects_missing_trust_metadata(self):
        with self.assertRaisesRegex(ValueError, "backend-issued contract identity"):
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
