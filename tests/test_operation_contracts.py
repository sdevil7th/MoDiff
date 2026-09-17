"""Stage discovery must describe registered adapters without loading a pipeline."""

from copy import deepcopy
from types import SimpleNamespace
import unittest

from modiff.operation_contracts import build_modular_operation_contracts


class Param:
    def __init__(self, name, type_):
        self.name = name
        self.type = type_

    def to_dict(self):
        return {"type": self.type}


def config():
    spec = {
        "inputs": [Param("embeddings", "embeddings"), Param("seed", "int")],
        "model_inputs": [Param("transformer", "diffusers_auto_model")],
        "outputs": [Param("embeddings", "embeddings"), Param("latents", "latents")],
        "required_inputs": ["embeddings"],
        "required_model_inputs": ["transformer"],
        "block_name": "denoise",
    }
    return SimpleNamespace(
        node_specs={"denoise": spec, "controlnet": None},
        node_params={
            "denoise": {
                "input_names": ["embeddings", "seed"],
                "model_input_names": ["transformer"],
                "output_names": ["out_embeddings", "latents"],
            }
        },
    )


MODULES = {"modules.ModularDiffusers": {"Denoise": {}}}


class OperationContractTests(unittest.TestCase):
    def test_same_operation_preserves_pipeline_scoped_port_meaning_and_saved_action(self):
        configs = {"FutureModularPipeline": config(), "AnotherModularPipeline": config()}
        contracts = build_modular_operation_contracts(configs, MODULES)
        self.assertEqual(len(contracts), 2)
        self.assertEqual({item["operationId"] for item in contracts}, {"diffusion.denoise"})
        self.assertEqual({item["nodeKey"] for item in contracts}, {"modules.ModularDiffusers.Denoise"})
        self.assertEqual([item["pipelineClass"] for item in contracts], sorted(configs))
        self.assertTrue(all(item["support"] == "declared" for item in contracts))
        ports = {port["name"]: port for port in contracts[0]["ports"]}
        self.assertEqual(ports["out_embeddings"]["semanticName"], "embeddings")
        self.assertEqual(ports["out_embeddings"]["direction"], "output")
        self.assertFalse(ports["out_embeddings"]["required"])
        self.assertTrue(ports["transformer"]["required"])
        self.assertEqual(ports["transformer"]["roles"], ["component"])
        self.assertFalse(ports["seed"]["required"])

    def test_unbound_custom_and_missing_actions_do_not_advertise_stage_support(self):
        self.assertEqual(build_modular_operation_contracts({"Custom": SimpleNamespace(node_specs=None)}, MODULES), [])
        self.assertEqual(build_modular_operation_contracts({"Pipeline": config()}, {}), [])

    def test_metadata_does_not_mutate_registry_or_share_returned_lists(self):
        source = {"Pipeline": config()}
        before = deepcopy(source["Pipeline"].node_specs["denoise"]["required_inputs"])
        first = build_modular_operation_contracts(source, MODULES)
        first[0]["ports"][0]["types"].append("wildcard")
        self.assertEqual(source["Pipeline"].node_specs["denoise"]["required_inputs"], before)
        self.assertNotIn("wildcard", build_modular_operation_contracts(source, MODULES)[0]["ports"][0]["types"])

    def test_bundle_is_not_mislabeled_as_an_upstream_block(self):
        source = config()
        source.node_specs["denoise"]["block_name"] = None
        result = build_modular_operation_contracts({"Pipeline": source}, MODULES)[0]
        self.assertEqual(result["decomposition"], "bundle")
        self.assertIsNone(result["blockName"])

    def test_invalid_declared_ports_fail_instead_of_advertising_partial_support(self):
        for mutation in ("unknown_required", "duplicate", "invalid_type"):
            source = config()
            spec = source.node_specs["denoise"]
            if mutation == "unknown_required":
                spec["required_inputs"].append("missing")
            elif mutation == "duplicate":
                spec["inputs"].append(Param("seed", "int"))
            else:
                spec["inputs"][0].type = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_modular_operation_contracts({"Pipeline": source}, MODULES)

    def test_one_conditioning_socket_can_supply_both_values_and_components(self):
        source = config()
        source.node_specs["denoise"]["model_inputs"].append(Param("embeddings", "embeddings"))
        source.node_params["denoise"]["model_input_names"].append("embeddings")
        ports = build_modular_operation_contracts({"Pipeline": source}, MODULES)[0]["ports"]
        inputs = [port for port in ports if port["name"] == "embeddings" and port["direction"] == "input"]
        self.assertEqual(len(inputs), 1)
        self.assertEqual(inputs[0]["roles"], ["value", "component"])
        self.assertTrue(inputs[0]["required"])


class PipelineOperationContractTests(unittest.TestCase):
    def test_pipeline_projection_preserves_task_visibility_and_pipeline_role(self):
        from modiff.operation_contracts import build_pipeline_operation_contract

        modules = {"modules.Example": {"Generate": {"params": {
            "pipeline": {"type": "image_pipeline", "display": "input", "required": True},
            "image": {"type": "image", "display": "input", "hidden": True},
            "width": {"type": "int", "default": 512},
            "images": {"type": "image", "display": "output"},
            "refresh": {"display": "button", "type": "bool"},
        }}}}
        before = deepcopy(modules)
        result = build_pipeline_operation_contract(
            modules, pipeline_class="NewPipeline", task="edit_image",
            operation_id="diffusion.generate_image", node_key="modules.Example.Generate",
            field_overrides={"image": {"hidden": False, "required": True}, "width": {"hidden": True}},
        )
        self.assertEqual(result["decomposition"], "pipeline")
        self.assertEqual(result["task"], "edit_image")
        ports = {port["name"]: port for port in result["ports"]}
        self.assertEqual(ports["pipeline"]["roles"], ["pipeline"])
        self.assertTrue(ports["image"]["required"])
        self.assertFalse(ports["image"]["hidden"])
        self.assertTrue(ports["width"]["hidden"])
        self.assertNotIn("refresh", ports)
        self.assertEqual(modules, before)

    def test_missing_actions_are_not_advertised_and_loader_outputs_keep_pipeline_role(self):
        from modiff.operation_contracts import build_pipeline_operation_contract

        arguments = dict(pipeline_class="NewPipeline", task="text_to_image",
                         operation_id="diffusion.load_models", node_key="modules.Example.Load", loader=True)
        self.assertIsNone(build_pipeline_operation_contract({}, **arguments))
        result = build_pipeline_operation_contract(
            {"modules.Example": {"Load": {"params": {
                "pipeline": {"type": "image_pipeline", "display": "output"},
            }}}}, **arguments,
        )
        self.assertEqual(result["decomposition"], "loader")
        self.assertEqual(result["ports"][0]["roles"], ["pipeline"])
        self.assertFalse(result["ports"][0]["required"])
