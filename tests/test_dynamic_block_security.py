import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modiff.modular_workflow_discovery import reviewed_modular_workflow_contract
from modules.ModularDiffusers.dynamic_node import DynamicBlockNode, _custom_node_contract
from modules.ModularDiffusers.pipeline_schema import MoDiffPipelineConfig


_REVISION = "a" * 40


def _custom_config(params=None):
    params = {} if params is None else params
    return MoDiffPipelineConfig.from_dict(
        {
            "label": "Dynamic fixture",
            "default_dtype": "float32",
            "node_params": {
                "custom": {
                    "params": params,
                    "model_input_names": [],
                    "input_names": [name for name in params if name in {"prompt", "image"}],
                    "output_names": [],
                    "label": "Dynamic fixture",
                }
            },
        }
    )


def _verified(config=None):
    return SimpleNamespace(
        config=_custom_config() if config is None else config,
        revision=_REVISION,
        repository_path="C:/hf-cache/models--owner--block/snapshots/" + _REVISION,
    )


def _binding(config=None):
    return SimpleNamespace(
        pipeline_config=lambda: _custom_config() if config is None else config,
        identity=SimpleNamespace(to_dict=lambda: {"schema": "fixture"}),
        execution_contract=SimpleNamespace(pipeline_class_name="FluxModularPipeline"),
    )


class DynamicBlockSecurityTests(unittest.TestCase):
    def test_reviewed_workflow_selector_exposes_only_tasks_the_sidecar_can_carry(self):
        config = MoDiffPipelineConfig.from_dict(
            {
                "label": "Text-only fixture",
                "node_params": {
                    "custom": {
                        "params": {
                            "prompt": {"type": "string"},
                            "image": {"type": "image", "display": "input"},
                            "out_images": {"type": "image", "display": "output"},
                            "out_processed_image": {"type": "image", "display": "output"},
                        },
                        "model_input_names": [],
                        "input_names": ["prompt"],
                        "output_names": ["out_images", "out_processed_image"],
                    }
                },
            }
        )
        contract = _custom_node_contract(
            config,
            reviewed_modular_workflow_contract("FluxModularPipeline"),
        )

        self.assertEqual(contract["params"]["workflow"]["options"], {"text_to_image": "Text To Image"})
        self.assertTrue(contract["params"]["image"]["hidden"])
        self.assertTrue(contract["params"]["out_processed_image"]["hidden"])

    def test_sidecar_pipeline_fields_must_exist_in_the_reviewed_upstream_contract(self):
        config = MoDiffPipelineConfig.from_dict(
            {
                "label": "Unknown-field fixture",
                "node_params": {
                    "custom": {
                        "params": {"attacker_input": {"type": "string"}},
                        "model_input_names": [],
                        "input_names": ["attacker_input"],
                        "output_names": [],
                    }
                },
            }
        )
        workflow_contract = reviewed_modular_workflow_contract("FluxModularPipeline")
        with self.assertRaisesRegex(ValueError, "absent from the reviewed upstream workflow contract"):
            _custom_node_contract(config, workflow_contract)

    def test_imported_trust_and_non_boolean_values_fail_before_any_loader(self):
        node = DynamicBlockNode("dynamic-imported-trust")

        cases = (
            (False, ValueError, "backend-issued reviewed identity"),
            (True, ValueError, "task-scoped operator authorization"),
            ("false", TypeError, "JSON boolean"),
            (1, TypeError, "JSON boolean"),
        )
        for trust_value, error_type, message in cases:
            with (
                self.subTest(trust_value=trust_value),
                patch.object(
                    node,
                    "_get_verified_custom_config",
                ) as verify_config,
                patch(
                    "modules.ModularDiffusers.dynamic_node.resolve_custom_pipeline_binding",
                ) as pipeline_loader,
            ):
                with self.assertRaisesRegex(error_type, message):
                    node.execute(
                        {"source": "hub", "value": "owner/custom-block"},
                        "cpu",
                        False,
                        trust_value,
                        offload_mode="none",
                        revision=_REVISION,
                    )

                verify_config.assert_not_called()
                pipeline_loader.assert_not_called()

    def test_unknown_reviewed_workflow_is_rejected_before_pipeline_construction(self):
        config = MoDiffPipelineConfig.from_dict(
            {
                "label": "Workflow fixture",
                "default_dtype": "float32",
                "node_params": {
                    "custom": {
                        "params": {
                            "prompt": {"type": "string"},
                            "out_images": {"type": "image", "display": "output"},
                        },
                        "model_input_names": [],
                        "input_names": ["prompt"],
                        "output_names": ["out_images"],
                    }
                },
            }
        )
        binding = _binding(config)
        binding.instantiate = MagicMock(side_effect=AssertionError("pipeline construction must not run"))
        node = DynamicBlockNode("dynamic-unknown-workflow")
        with (
            patch(
                "modules.ModularDiffusers.dynamic_node.CustomPipelineExecutionIdentity.from_value",
                return_value=SimpleNamespace(),
            ),
            patch(
                "modules.ModularDiffusers.dynamic_node.resolve_custom_pipeline_binding",
                return_value=binding,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "Unknown Modular workflow task"):
                node.execute(
                    {"source": "hub", "value": "owner/custom-block"},
                    "cpu",
                    False,
                    False,
                    offload_mode="none",
                    revision=_REVISION,
                    modiff_pipeline_identity={"schema": "fixture"},
                    workflow="attacker_workflow",
                    prompt="hello",
                )
        binding.instantiate.assert_not_called()

    def test_exact_revision_is_used_for_local_only_verified_sidecar_resolution(self):
        node = DynamicBlockNode("dynamic-sidecar-revision")
        verified = _verified()
        with (
            patch(
                "modules.ModularDiffusers.dynamic_node.PipelineConfig.load_verified",
                return_value=verified,
            ) as load_verified,
        ):
            result = node._get_verified_custom_config("owner/custom-block", _REVISION)

        self.assertIs(result, verified)
        load_verified.assert_called_once_with(
            "owner/custom-block",
            source="hub",
            revision=_REVISION,
        )

    def test_hostile_sidecar_actions_are_rejected_before_definition_publication(self):
        hostile_actions = (
            {"prompt": {"type": "string", "onChange": "update_node"}},
            {"prompt": {"type": "string", "onSignal": {"action": "exec", "data": "update_node"}}},
            {"prompt": {"type": "string", "onChange": {"action": "create", "data": {}}}},
        )

        for index, params in enumerate(hostile_actions):
            node = DynamicBlockNode(f"dynamic-hostile-action-{index}")
            node.send_node_definition_with_meta = MagicMock()
            with patch(
                "modules.ModularDiffusers.dynamic_node.resolve_custom_pipeline_binding",
                return_value=_binding(_custom_config(params)),
            ):
                with self.assertRaisesRegex(ValueError, "must not define"):
                    node.update_node(
                        {
                            "repo_id": "owner/custom-block",
                            "revision": _REVISION,
                            "trust_remote_code": False,
                        },
                        {"key": "load_block_button"},
                    )

            node.send_node_definition_with_meta.assert_not_called()

    def test_trusted_preview_is_rejected_before_sidecar_resolution(self):
        node = DynamicBlockNode("dynamic-trusted-preview")
        node.send_node_definition_with_meta = MagicMock()
        with patch("modules.ModularDiffusers.dynamic_node.resolve_custom_pipeline_binding") as verify_config:
            with self.assertRaisesRegex(ValueError, "Trust Remote Code off"):
                node.update_node(
                    {
                        "repo_id": "owner/custom-block",
                        "revision": _REVISION,
                        "trust_remote_code": True,
                    },
                    {"key": "load_block_button"},
                )
        verify_config.assert_not_called()
        node.send_node_definition_with_meta.assert_not_called()

    def test_declarative_sidecar_actions_remain_available(self):
        params = {
            "mode": {
                "type": "string",
                "onChange": {"image": ["image"], "text": ["prompt"]},
            },
            "image": {"display": "input", "type": "image"},
            "prompt": {"type": "string"},
            "identity": {
                "display": "input",
                "type": "object",
                "onSignal": {"action": "value", "target": "mode"},
            },
        }
        node = DynamicBlockNode("dynamic-declarative-actions")
        node.send_node_definition_with_meta = MagicMock()
        with (
            patch(
                "modules.ModularDiffusers.dynamic_node.resolve_custom_pipeline_binding",
                return_value=_binding(_custom_config(params)),
            ) as verify_config,
            patch("modules.ModularDiffusers.dynamic_node.PipelineConfig.load") as network_config_load,
        ):
            node.update_node(
                {
                    "repo_id": "owner/custom-block",
                    "revision": _REVISION,
                    "trust_remote_code": False,
                },
                {"key": "load_block_button"},
            )

        published_params = node.send_node_definition_with_meta.call_args.args[0]
        self.assertEqual(published_params["mode"], params["mode"])
        self.assertEqual(published_params["identity"], params["identity"])
        self.assertEqual(published_params["prompt"], {**params["prompt"], "hidden": False})
        self.assertEqual(published_params["image"], {**params["image"], "hidden": True})
        self.assertEqual(
            published_params["workflow"]["options"],
            {"text_to_image": "Text To Image", "image_to_image": "Image To Image"},
        )
        verify_config.assert_called_once_with(
            source="hub",
            repo_id="owner/custom-block",
            revision=_REVISION,
            trust_remote_code=False,
            expected_identity=None,
        )
        network_config_load.assert_not_called()

    def test_declarative_sidecar_cannot_target_unpublished_fields(self):
        hostile_targets = (
            {"mode": {"type": "string", "onChange": {"true": ["ghost"]}}},
            {
                "mode": {
                    "type": "string",
                    "onChange": {"action": "value", "target": "modiff_pipeline_identity"},
                }
            },
            {
                "mode": {"type": "string", "onChange": {"action": "signal", "target": "prompt"}},
                "prompt": {"type": "string"},
            },
        )
        for index, params in enumerate(hostile_targets):
            node = DynamicBlockNode(f"dynamic-hostile-target-{index}")
            node.send_node_definition_with_meta = MagicMock()
            with patch(
                "modules.ModularDiffusers.dynamic_node.resolve_custom_pipeline_binding",
                return_value=_binding(_custom_config(params)),
            ):
                with self.assertRaisesRegex(ValueError, "unknown contract field|input or output"):
                    node.update_node(
                        {
                            "repo_id": "owner/custom-block",
                            "revision": _REVISION,
                            "trust_remote_code": False,
                        },
                        {"key": "load_block_button"},
                    )
            node.send_node_definition_with_meta.assert_not_called()

    def test_prototype_sensitive_sidecar_field_names_are_rejected(self):
        for index, field_name in enumerate(("__proto__", "prototype", "constructor")):
            node = DynamicBlockNode(f"dynamic-prototype-field-{index}")
            node.send_node_definition_with_meta = MagicMock()
            with patch(
                "modules.ModularDiffusers.dynamic_node.resolve_custom_pipeline_binding",
                return_value=_binding(_custom_config({field_name: {"type": "string"}})),
            ):
                with self.assertRaisesRegex(ValueError, "fields must map"):
                    node.update_node(
                        {
                            "repo_id": "owner/custom-block",
                            "revision": _REVISION,
                            "trust_remote_code": False,
                        },
                        {"key": "load_block_button"},
                    )
            node.send_node_definition_with_meta.assert_not_called()


if __name__ == "__main__":
    unittest.main()
