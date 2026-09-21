"""Ordinary stage drafts must use the reviewed runtime's actual wires."""

import unittest
from unittest.mock import patch

from modules import MODULE_MAP
from modiff.operation_catalog import build_operation_catalog


class OperationStarterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})

    def resolve(self, pipeline, task):
        from modiff.operation_starters import resolve_operation_starter

        return resolve_operation_starter(MODULE_MAP, self.contracts, {"pipelineClass": pipeline, "task": task})

    def test_exact_profile_selects_shared_pipeline_model_without_constructing_nodes(self):
        from modiff.operation_starters import resolve_operation_starter
        from modiff.diffusers_profiles import resolve_execution_profiles_for_loader

        for identity, repository in (
            ("flux-schnell:direct", "black-forest-labs/FLUX.1-schnell"),
            ("flux-krea:direct", "black-forest-labs/FLUX.1-Krea-dev"),
        ):
            with (
                self.subTest(profile=identity),
                patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed")),
            ):
                result = resolve_operation_starter(
                    MODULE_MAP,
                    self.contracts,
                    {
                        "pipelineClass": "FluxPipeline",
                        "task": "text_to_image",
                        "executionProfileId": identity,
                    },
                )
                loader = result["nodes"][0]
                self.assertEqual(loader["params"]["model_id"]["value"], {"source": "hub", "value": repository})
                self.assertRegex(loader["params"]["revision"]["value"], r"^[0-9a-f]{40}$")
                profiles, reason = resolve_execution_profiles_for_loader(
                    loader["module"], loader["action"], loader["values"]
                )
                self.assertIsNone(reason)
                self.assertEqual([p.id for p in profiles], [identity])
                self.assertNotIn("executionProfileId", result, "Keep the existing response envelope compatible")

    def test_profile_selection_rejects_unrelated_unknown_and_invalid_identities(self):
        from modiff.operation_starters import resolve_operation_starter

        for identity in ("ace-step-audio:direct", "unknown", [], None, " flux-dev:direct"):
            with self.subTest(profile=identity), self.assertRaises(ValueError):
                resolve_operation_starter(
                    MODULE_MAP,
                    self.contracts,
                    {
                        "pipelineClass": "FluxPipeline",
                        "task": "text_to_image",
                        "executionProfileId": identity,
                    },
                )

    def test_every_advertised_profile_task_binds_without_constructing_models(self):
        from modiff.operation_starters import resolve_operation_starter
        from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES, public_execution_profiles

        contracts, support = build_operation_catalog(
            MODULE_MAP, public_execution_profiles(), catalog_resolver=lambda: {}
        )
        checked = 0
        with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("constructed")):
            for pipeline in support:
                for task in pipeline["tasks"]:
                    if not task["operationIds"]:
                        continue
                    for identity in task["executionProfileIds"]:
                        with self.subTest(pipeline=pipeline["pipelineClass"], task=task["task"], profile=identity):
                            result = resolve_operation_starter(
                                MODULE_MAP,
                                contracts,
                                {
                                    "pipelineClass": pipeline["pipelineClass"],
                                    "task": task["task"],
                                    "executionProfileId": identity,
                                },
                            )
                            loader = result["nodes"][0]
                            field = "repo_id" if loader["action"] == "ModelsLoader" else "model_id"
                            if loader["operation"]["decomposition"] == "integrated":
                                from modiff.integrated_operation_contracts import integrated_operation_values

                                self.assertEqual(loader["values"], integrated_operation_values(
                                    loader["operation"], profile=DIFFUSERS_EXECUTION_PROFILES[identity]
                                ))
                                self.assertRegex(loader["values"][field]["revision"], r"^[0-9a-f]{40}$")
                            else:
                                self.assertEqual(
                                    loader["values"][field]["value"], DIFFUSERS_EXECUTION_PROFILES[identity].default_repo
                                )
                                self.assertRegex(loader["values"]["revision"], r"^[0-9a-f]{40}$")
                            checked += 1
        self.assertGreater(checked, 150)

    def test_four_stages_are_an_ordinary_draft_with_exact_state_and_component_wires(self):
        with patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed node")):
            result = self.resolve("AnimaModularPipeline", "text_to_image")
        self.assertEqual(len(result["nodes"]), 4)
        self.assertEqual(len(result["edges"]), 5)
        self.assertNotIn("receipt", result)
        self.assertTrue(all(n["operation"]["task"] == "text_to_image" for n in result["nodes"]))
        self.assertTrue(
            any(e["sourceHandle"] == "state_out" and e["targetHandle"] == "state_in" for e in result["edges"])
        )

    def test_task_switch_uses_upstream_rules_and_exposes_required_image(self):
        text = self.resolve("FluxModularPipeline", "text_to_image")
        image = self.resolve("FluxModularPipeline", "image_to_image")
        edit = self.resolve("FluxKontextModularPipeline", "edit_image")
        self.assertEqual(len(text["nodes"]), 4)
        self.assertEqual(len(image["nodes"]), 5)
        self.assertEqual(image["workflowId"], "image2image")
        self.assertEqual(edit["workflowId"], "image_conditioned")
        self.assertTrue(any(i["field"] == "image" for i in image["requiredInputs"]))
        self.assertTrue(any(e["sourceHandle"] == "out_width" and e["targetHandle"] == "width" for e in image["edges"]))
        with self.assertRaises(ValueError):
            self.resolve("FluxModularPipeline", "edit_image")

    def test_task_required_media_is_declared_on_each_operation_port(self):
        for pipeline, task in sorted({(c['pipelineClass'], c['task']) for c in self.contracts
                                      if c['task'] and c['nodeKey'].startswith('modules.ModularDiffusers.')}):
            result = self.resolve(pipeline, task)
            nodes = {n['operation']['operationId']: n for n in result['nodes']}
            for required in result['requiredInputs']:
                ports = nodes[required['operationId']]['operation']['ports']
                port = next(p for p in ports if p['name'] == required['field'] and p['direction'] == 'input')
                if port['semantics']['kind'] == 'media':
                    with self.subTest(pipeline=pipeline, task=task, field=required['field']):
                        self.assertTrue(port['required'])
        image = self.resolve('StableDiffusionXLModularPipeline', 'image_to_image')
        encode = next(n for n in image['nodes'] if n['operation']['operationId'] == 'diffusion.encode_image')
        self.assertFalse(next(p for p in encode['operation']['ports'] if p['name'] == 'mask_image')['required'])

    def test_required_component_contract_completes_denoise_vae_wiring(self):
        # The SDXL runtime requires a managed VAE even for text-to-image.
        # Admission's minimal component edges alone do not describe this input.
        result = self.resolve("StableDiffusionXLModularPipeline", "text_to_image")
        edge = {
            "source": "diffusion.load_models",
            "sourceHandle": "vae_out",
            "target": "diffusion.denoise",
            "targetHandle": "vae",
        }
        self.assertIn(edge, result["edges"])
        for task in ('image_to_image', 'inpaint'):
            with self.subTest(task=task):
                selected = self.resolve('StableDiffusionXLModularPipeline', task)
                self.assertIn(edge, selected['edges'])
                denoise = next(n['operation'] for n in selected['nodes']
                               if n['operation']['operationId'] == 'diffusion.denoise')
                vae = next(p for p in denoise['ports'] if p['name'] == 'vae')
                self.assertEqual([m['name'] for m in vae['semantics']['members']], ['vae'])
        image = self.resolve("StableDiffusionXLModularPipeline", "image_to_image")
        self.assertEqual(
            image["sharedInputs"],
            [
                {
                    "name": "seed",
                    "members": [
                        {"operationId": "diffusion.encode_image", "field": "seed"},
                        {"operationId": "diffusion.denoise", "field": "seed"},
                    ],
                }
            ],
        )
        self.assertEqual(result["sharedInputs"], [])
        # These adapters do not declare that denoiser component dependency.
        for pipeline in ("FluxModularPipeline", "QwenImageModularPipeline"):
            self.assertNotIn(edge, self.resolve(pipeline, "text_to_image")["edges"])

    def test_standard_fallback_is_a_load_and_call_and_keeps_audio_contract(self):
        result = self.resolve("StableAudioPipeline", "text_to_audio")
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["edges"][0]["targetHandle"], "pipeline")

    def test_standard_loader_starters_do_not_require_an_execution_recipe_override(self):
        # All ordinary loaders can use their own resource fields. A connected
        # starter must not ask Fix to add an optional recipe source.
        from modiff.operation_catalog import resolve_operation

        checked = set()
        for contract in self.contracts:
            if not contract.get("nodeKey", "").endswith(".LoadPipeline"):
                continue
            node = resolve_operation(
                MODULE_MAP,
                self.contracts,
                {
                    "pipelineClass": contract["pipelineClass"],
                    "task": contract["task"],
                    "operationId": contract["operationId"],
                },
            )
            if "execution_recipe" not in node["params"]:
                continue
            with self.subTest(pipeline=contract["pipelineClass"], task=contract["task"]):
                self.assertIs(node["params"]["execution_recipe"].get("required"), False)
            checked.add(node["module"])
        self.assertEqual(
            checked,
            {"modules.DiffusersImage", "modules.DiffusersVideo", "modules.DiffusersAudio", "modules.DiffusersThreeD"},
        )

    def test_unconditional_starters_keep_the_reviewed_resident_float32_recipe(self):
        # These upstream samplers do not share the text-to-image loader's
        # bfloat16/model-offload recipe. Check individual insertion as well as
        # the connected starter; existing generic schemas must stay unchanged.
        from copy import deepcopy
        from modiff.operation_catalog import resolve_operation

        original = deepcopy(MODULE_MAP["modules.DiffusersImage"]["LoadPipeline"])
        for pipeline in ("DDPMPipeline", "DDIMPipeline", "ConsistencyModelPipeline"):
            with self.subTest(pipeline=pipeline):
                draft = self.resolve(pipeline, "unconditional_image")
                loader = next(n for n in draft["nodes"] if n["action"] == "LoadPipeline")
                single = resolve_operation(
                    MODULE_MAP,
                    self.contracts,
                    {
                        "pipelineClass": pipeline,
                        "task": "unconditional_image",
                        "operationId": loader["operation"]["operationId"],
                    },
                )
                for node in (loader, single):
                    for field, expected in {"dtype": "float32", "auto_offload": False, "offload_mode": "none"}.items():
                        self.assertEqual(node["values"].get(field), expected)
                        self.assertEqual(node["params"][field]["value"], expected)
                    self.assertEqual(node["values"].get("conditioning_model_id"), "")
                    self.assertEqual(node["values"].get("conditioning_revision"), "")
        self.assertEqual(MODULE_MAP["modules.DiffusersImage"]["LoadPipeline"], original)

    def test_loader_defaults_keep_conditioned_models_and_other_task_recipes(self):
        from modules.DiffusersImage.main import IMAGE_PIPELINE_ADAPTERS

        for pipeline, task in (("FluxPipeline", "text_to_image"), ("FluxControlNetPipeline", "control_image")):
            with self.subTest(pipeline=pipeline):
                loader = next(n for n in self.resolve(pipeline, task)["nodes"] if n["action"] == "LoadPipeline")
                self.assertEqual(loader["values"]["dtype"], "bfloat16")
                self.assertEqual(loader["params"]["dtype"]["value"], "bfloat16")
                self.assertNotIn("offload_mode", loader["values"])
                auxiliary = IMAGE_PIPELINE_ADAPTERS[pipeline].default_conditioning_repo
                self.assertEqual(
                    loader["values"]["conditioning_model_id"],
                    {"source": "hub", "value": auxiliary} if auxiliary else "",
                )
                if auxiliary:
                    self.assertEqual(len(loader["values"]["conditioning_revision"]), 40)

    def test_all_task_drafts_have_existing_visible_endpoints_and_one_writer(self):
        with (
            patch("socket.socket.connect", side_effect=AssertionError("Network during authoring")),
            patch("modiff.NodeBase.NodeBase.__init__", side_effect=AssertionError("Constructed node")),
        ):
            for pipeline, task in sorted({(c["pipelineClass"], c["task"]) for c in self.contracts if c["task"]}):
                with self.subTest(pipeline=pipeline, task=task):
                    result = self.resolve(pipeline, task)
                    nodes = {n["operation"]["operationId"]: n for n in result["nodes"]}
                    targets = set()
                    for edge in result["edges"]:
                        self.assertIn(edge["sourceHandle"], nodes[edge["source"]]["params"])
                        self.assertIn(edge["targetHandle"], nodes[edge["target"]]["params"])
                        target = edge["target"], edge["targetHandle"]
                        self.assertNotIn(target, targets)
                        targets.add(target)

    def test_invalid_selection_is_rejected_without_mutation(self):
        from modiff.operation_starters import resolve_operation_starter

        for selection in (
            None,
            {},
            {"pipelineClass": "__proto__", "task": "text_to_image"},
            {"pipelineClass": "FluxModularPipeline", "task": None},
            {"pipelineClass": "FluxModularPipeline", "task": "text_to_image", "code": "x"},
        ):
            with self.assertRaises(ValueError):
                resolve_operation_starter(MODULE_MAP, self.contracts, selection)


def test_every_reviewed_starter_wire_retains_compatible_semantic_direction_and_component_names():
    from modules import MODULE_MAP
    from modiff.operation_catalog import build_operation_catalog
    from modiff.operation_starters import resolve_operation_starter

    contracts, _ = build_operation_catalog(MODULE_MAP, [], catalog_resolver=lambda: {})
    for pipeline, task in sorted({(c['pipelineClass'], c['task']) for c in contracts
                                  if c['task'] and c['nodeKey'].startswith('modules.ModularDiffusers.')}):
        starter = resolve_operation_starter(MODULE_MAP, contracts, {'pipelineClass': pipeline, 'task': task})
        nodes = {n['operation']['operationId']: n['operation'] for n in starter['nodes']}
        for edge in starter['edges']:
            left = next(p for p in nodes[edge['source']]['ports'] if p['name'] == edge['sourceHandle'] and p['direction'] == 'output')['semantics']
            right = next(p for p in nodes[edge['target']]['ports'] if p['name'] == edge['targetHandle'] and p['direction'] == 'input')['semantics']
            assert (left['kind'], left['scope']) == (right['kind'], right['scope']), (pipeline, task, edge)
            if right['state'] is not None:
                assert left['state'] == right['state'], (pipeline, task, edge)
            if left['kind'] == 'component' and left['members'] and right['members']:
                assert {m['name'] for m in right['members']} <= {m['name'] for m in left['members']}, (pipeline, task, edge)
