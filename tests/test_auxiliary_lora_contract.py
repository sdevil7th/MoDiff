import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from modiff.auxiliary_lora import (
    LORA_DESCRIPTOR_SCHEMA,
    build_lora_descriptor,
    controlled_lora_receipts_from_graph,
    lora_resource_requirement,
    native_lora_chain_ids,
    resolve_lora_descriptor,
)
from modules.ModularDiffusers.adapters import Lora
from modules.ModularDiffusers.loaders import apply_lora_scheduler_override, update_lora_adapters
from utils.huggingface import CONFIG


REVISION = "a" * 40


def _write_tiny_safetensors(path: Path, marker: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"lora.weight": np.asarray([marker], dtype=np.float32)}, str(path))


class MutationTrackingPipeline:
    def __init__(self):
        self.adapters = {"transformer": ["working"]}
        self.events = []
        self.scheduler = None

    def get_list_adapters(self):
        return self.adapters

    def unload_lora_weights(self):
        self.events.append("unload")
        self.adapters = {"transformer": []}

    def delete_adapters(self, name):
        self.events.append(("delete", name))

    def load_lora_weights(self, path, **kwargs):
        self.events.append(("load", path, kwargs))

    def set_adapters(self, names, weights):
        self.events.append(("set", names, weights))


def _local_descriptor(path: Path, *, node_id: str = "local", scale: float = 1.0):
    return Lora(node_id).execute(
        {"source": "local", "value": str(path)},
        scale,
        weight_name=path.name,
    )["lora"]


def _resign_descriptor(descriptor):
    payload = {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    descriptor["descriptor_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return descriptor


class AuxiliaryLoraContractTests(unittest.TestCase):
    def test_public_lora_node_exposes_named_ordered_adapter_inputs(self):
        from modules import MODULE_MAP

        params = MODULE_MAP["modules.ModularDiffusers"]["Lora"]["params"]
        self.assertEqual(params["adapter_name"]["type"], "string")
        self.assertEqual(params["adapter_name"]["default"], "")
        self.assertEqual(params["previous_loras"]["display"], "input")
        self.assertEqual(params["previous_loras"]["type"], params["lora"]["type"])

    def test_ordinary_node_dispatch_preserves_named_chain_and_invalidates_changed_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.safetensors", Path(directory) / "second.safetensors"
            _write_tiny_safetensors(first)
            _write_tiny_safetensors(second, marker=2)
            first_node, second_node = Lora("first"), Lora("second")

            def inputs(path, name, scale, previous=None):
                return {"model": {"source": "local", "value": str(path)},
                        "weight_name": path.name, "revision": "", "expected_sha256": "",
                        "scale": scale, "adapter_name": name, "previous_loras": previous,
                        "scheduler_class": "", "scheduler_config": "{}"}

            # The real dispatcher, validation and cache run. Only discovery of
            # these temporary local fixtures is suppressed to avoid server I/O.
            with patch("modiff.NodeBase.modelstore.is_local_cached", return_value=True):
                previous = first_node(**inputs(first, "first_style", "0.4"))["lora"]
                chain_inputs = inputs(second, "second_style", "0.8", previous)
                result = second_node(**chain_inputs)["lora"]
                self.assertEqual([item["adapter_name"] for item in result], ["first_style", "second_style"])
                self.assertEqual([item["scale"] for item in result], [0.4, 0.8])
                self.assertIs(second_node(**chain_inputs)["lora"], result)
                self.assertEqual(second_node._cache_reason, "unchanged_inputs")
                changed = second_node(**{**chain_inputs, "scale": "0.25"})["lora"]
                self.assertEqual([item["scale"] for item in changed], [0.4, 0.25])
                self.assertEqual(second_node._cache_reason, "inputs_changed")

    def test_native_lora_preserves_explicit_name_in_descriptor_and_graph_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight)
            selection = {"source": "local", "value": str(weight)}
            descriptor = Lora("native-style").execute(selection, 0.75, adapter_name="original_style")["lora"]
            self.assertEqual(descriptor["adapter_name"], "original_style")
            graph = {"nodes": {"native-style": {
                "module": "modules.ModularDiffusers", "action": "Lora",
                "params": {"model": {"value": selection}, "scale": {"value": 0.75},
                           "adapter_name": {"value": "original_style"}},
            }}, "paths": [["native-style"]]}
            receipt = controlled_lora_receipts_from_graph(graph)[0]
            self.assertEqual(receipt["adapterName"], "original_style")
            self.assertEqual(receipt["descriptorSha256"], descriptor["descriptor_sha256"])

    def test_native_lora_chain_preserves_order_scales_and_existing_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.safetensors", Path(directory) / "second.safetensors"
            _write_tiny_safetensors(first)
            _write_tiny_safetensors(second, marker=2)
            before = Lora("first").execute({"source": "local", "value": str(first)}, 0.4,
                                           adapter_name="first_style")["lora"]
            saved = copy.deepcopy(before)
            result = Lora("second").execute({"source": "local", "value": str(second)}, 0.8,
                                            adapter_name="second_style", previous_loras=before)["lora"]
            self.assertEqual([item["adapter_name"] for item in result], ["first_style", "second_style"])
            self.assertEqual([item["scale"] for item in result], [0.4, 0.8])
            self.assertEqual(before, saved)
            pipeline = MutationTrackingPipeline()
            update_lora_adapters(pipeline, result)
            self.assertEqual(pipeline.events[-1], ("set", ["first_style", "second_style"], [0.4, 0.8]))

    def test_native_graph_chain_receipts_and_resource_inspection_preserve_each_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = [Path(directory) / f"style-{index}.safetensors" for index in range(2)]
            for index, weight in enumerate(weights):
                _write_tiny_safetensors(weight, marker=index + 1)
            nodes = {str(index): {
                "module": "modules.ModularDiffusers", "action": "Lora", "params": {
                    "model": {"value": {"source": "local", "value": str(weight)}},
                    "adapter_name": {"value": f"style_{index}"}, "scale": {"value": (0.4, 0.8)[index]},
                },
            } for index, weight in enumerate(weights)}
            nodes["1"]["params"]["previous_loras"] = {"sourceId": "0", "sourceKey": "lora"}
            graph = {"nodes": nodes, "paths": [["0", "1"]]}
            self.assertEqual(native_lora_chain_ids(graph, "1"), ["0", "1"])
            receipts = controlled_lora_receipts_from_graph(graph)
            self.assertEqual([item["adapterName"] for item in receipts], ["style_0", "style_1"])
            self.assertEqual([item["scale"] for item in receipts], [0.4, 0.8])
            requirement = lora_resource_requirement("1", nodes["1"], graph=graph)
            self.assertEqual(requirement["adapterName"], "style_1")
            self.assertEqual(requirement["artifact"]["sha256"], hashlib.sha256(weights[1].read_bytes()).hexdigest())
            self.assertEqual(requirement["tensorCount"], 1)
            self.assertGreater(requirement["systemRamBytes"], 0)
            with self.assertRaisesRegex(ValueError, "exact artifact"):
                lora_resource_requirement("1", nodes["1"])
            nodes["1"]["params"]["adapter_name"]["value"] = "style_0"
            with self.assertRaisesRegex(ValueError, "unique adapter names"):
                controlled_lora_receipts_from_graph(graph)

    def test_native_graph_chain_rejects_unresolved_sources_cycles_and_literal_descriptors(self):
        base = {"nodes": {
            "first": {"module": "modules.ModularDiffusers", "action": "Lora", "params": {}},
            "last": {"module": "modules.ModularDiffusers", "action": "Lora", "params": {
                "previous_loras": {"sourceId": "first", "sourceKey": "lora"},
            }},
        }, "paths": [["first", "last"]]}
        mutations = [
            lambda graph: graph["nodes"]["first"].update(module="custom.unreviewed"),
            lambda graph: graph["nodes"]["last"]["params"]["previous_loras"].update(sourceKey="data"),
            lambda graph: graph["nodes"]["last"]["params"]["previous_loras"].update(sourceId="missing"),
            lambda graph: graph["nodes"]["first"]["params"].update(previous_loras={"sourceId": "last", "sourceKey": "lora"}),
            lambda graph: graph["nodes"]["last"]["params"].update(scale={"sourceId": "first", "sourceKey": "lora", "value": 1}),
            lambda graph: graph.update(paths=[["last"]]),
            lambda graph: graph["nodes"]["first"]["params"].update(previous_loras={"value": {"schema": LORA_DESCRIPTOR_SCHEMA}}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate.__code__.co_firstlineno):
                graph = copy.deepcopy(base)
                mutate(graph)
                with self.assertRaises(ValueError):
                    native_lora_chain_ids(graph, "last")
        nodes = {str(index): {"module": "modules.ModularDiffusers", "action": "Lora", "params": (
            {"previous_loras": {"sourceId": str(index - 1), "sourceKey": "lora"}} if index else {}
        )} for index in range(33)}
        with self.assertRaisesRegex(ValueError, "At most 32"):
            native_lora_chain_ids({"nodes": nodes, "paths": [list(nodes)]}, "32")

    def test_connected_adapter_controls_cannot_claim_stale_literal_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight)
            graph = {"nodes": {"adapter": {
                "module": "modules.DiffusersImage", "action": "LoadAdapter", "params": {
                    "adapter_path": {"value": {"source": "local", "value": str(weight)}},
                    "adapter_name": {"value": "style"}, "scale": {
                        "value": 0.5, "sourceId": "dynamic-control", "sourceKey": "value",
                    },
                    "pipeline": {"sourceId": "pipeline", "sourceKey": "pipeline"},
                },
            }}, "paths": [["adapter"]]}
            with self.assertRaisesRegex(ValueError, "scale must be resolved"):
                controlled_lora_receipts_from_graph(graph)
            graph["nodes"]["adapter"]["params"]["scale"] = {"value": 0.5}
            self.assertEqual(controlled_lora_receipts_from_graph(graph)[0]["scale"], 0.5)

    def test_native_lora_chain_revalidates_prior_bytes_names_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "first.safetensors", Path(directory) / "second.safetensors"
            _write_tiny_safetensors(first)
            _write_tiny_safetensors(second, marker=2)
            selection = {"source": "local", "value": str(second)}
            before = Lora("first").execute({"source": "local", "value": str(first)}, 1,
                                           adapter_name="first_style")["lora"]
            with self.assertRaisesRegex(ValueError, "unique adapter names"):
                Lora("second").execute(selection, 1, adapter_name="first_style", previous_loras=before)
            with self.assertRaisesRegex(ValueError, "At most 32"):
                Lora("second").execute(selection, 1, adapter_name="second_style", previous_loras=[before] * 32)
            tampered = copy.deepcopy(before)
            tampered["scale"] = 0.5
            with self.assertRaises(ValueError):
                Lora("second").execute(selection, 1, adapter_name="second_style", previous_loras=tampered)
            _write_tiny_safetensors(first, marker=3)
            with self.assertRaisesRegex(ValueError, "SHA-256|hash|changed"):
                Lora("second").execute(selection, 1, adapter_name="second_style", previous_loras=before)

    def test_generated_names_allow_versioned_weights_and_nested_node_ids_without_collisions(self):
        import torch

        with tempfile.TemporaryDirectory() as directory:
            names = []
            for filename in ("Lightning-V1.0.safetensors", "Lightning-V1_0.safetensors"):
                weight = Path(directory) / filename
                _write_tiny_safetensors(weight)
                node_id = "block-v2-node:4:root:7:adapter"
                descriptor = _local_descriptor(weight, node_id=node_id)
                graph = {
                    "nodes": {node_id: {
                        "module": "modules.ModularDiffusers", "action": "Lora",
                        "params": {
                            "model": {"value": {"source": "local", "value": str(weight)}},
                            "weight_name": {"value": filename}, "scale": {"value": 1.0},
                        },
                    }},
                    "paths": [[node_id]],
                }
                receipt = controlled_lora_receipts_from_graph(graph)[0]
                self.assertEqual(receipt["descriptorSha256"], descriptor["descriptor_sha256"])
                self.assertEqual(receipt["adapterName"], descriptor["adapter_name"])
                name = descriptor["adapter_name"]
                torch.nn.ModuleDict({name: torch.nn.Identity()})
                self.assertEqual(_local_descriptor(weight, node_id=node_id)["adapter_name"], name)
                names.append(name)
            self.assertNotEqual(names[0], names[1])

    def test_executable_graph_receipts_bind_order_and_ignore_disconnected_adapters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            disconnected = root / "disconnected.safetensors"
            _write_tiny_safetensors(first, 1)
            _write_tiny_safetensors(second, 2)
            _write_tiny_safetensors(disconnected, 3)

            def direct_node(path, name, scale, replace_existing):
                return {
                    "module": "modules.DiffusersImage",
                    "action": "LoadAdapter",
                    "params": {
                        "adapter_path": {"value": {"source": "local", "value": str(path)}},
                        "weight_name": {"value": path.name},
                        "adapter_name": {"value": name},
                        "scale": {"value": scale},
                        "replace_existing": {"value": replace_existing},
                    },
                }

            graph = {
                "nodes": {
                    "pipeline": {"module": "unit", "action": "Pipeline", "params": {}},
                    "first": direct_node(first, "first", 0.5, True),
                    "second": direct_node(second, "second", 0.25, False),
                    "disconnected": direct_node(disconnected, "unused", 1, True),
                    "generate": {"module": "unit", "action": "Generate", "params": {}},
                },
                "paths": [["pipeline", "first", "second", "generate"]],
            }
            receipts = controlled_lora_receipts_from_graph(graph)
            reversed_receipts = controlled_lora_receipts_from_graph(
                {**graph, "paths": [["pipeline", "second", "first", "generate"]]}
            )
            expected_first_digest = hashlib.sha256(first.read_bytes()).hexdigest()

        self.assertEqual([item["adapterName"] for item in receipts], ["first", "second"])
        self.assertEqual([item["replaceExisting"] for item in receipts], [True, False])
        self.assertEqual(receipts[0]["artifact"]["sha256"], expected_first_digest)
        self.assertNotIn(str(root), json.dumps(receipts))
        self.assertEqual([item["adapterName"] for item in reversed_receipts], ["second", "first"])

    def test_modular_graph_receipt_matches_the_runtime_descriptor_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight)
            graph = {
                "nodes": {
                    "adapter": {
                        "module": "modules.ModularDiffusers",
                        "action": "Lora",
                        "params": {
                            "model": {"value": {"source": "local", "value": str(weight)}},
                            "weight_name": {"value": weight.name},
                            "scale": {"value": 0.75},
                        },
                    }
                },
                "paths": [["adapter"]],
            }
            receipt = controlled_lora_receipts_from_graph(graph)[0]
            descriptor = Lora("adapter").execute(
                {"source": "local", "value": str(weight)},
                0.75,
                weight_name=weight.name,
            )["lora"]

        self.assertEqual(receipt["descriptorSha256"], descriptor["descriptor_sha256"])
        self.assertEqual(receipt["adapterName"], "style_adapter")
        self.assertIsNone(receipt["replaceExisting"])

    def test_empty_direct_image_adapter_is_the_same_noop_as_the_loader(self):
        graph = {
            "nodes": {
                "adapter": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadAdapter",
                    "params": {
                        "adapter_path": {"value": {"source": "hub", "value": ""}},
                    },
                }
            },
            "paths": [["adapter"]],
        }
        self.assertEqual(controlled_lora_receipts_from_graph(graph), [])

    def test_local_file_produces_one_versioned_content_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight)
            expected_digest = hashlib.sha256(weight.read_bytes()).hexdigest()

            descriptor = _local_descriptor(weight)
            resolved = resolve_lora_descriptor(descriptor)

        self.assertEqual(descriptor["schema"], LORA_DESCRIPTOR_SCHEMA)
        self.assertEqual(descriptor["artifact"]["source"], "local")
        self.assertEqual(descriptor["artifact"]["sha256"], expected_digest)
        self.assertEqual(resolved.weight_name, "style.safetensors")
        self.assertEqual(resolved.load_directory, weight.parent.resolve())

    def test_local_directory_requires_a_contained_literal_lowercase_safetensors_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "adapter.bin").write_bytes(b"bin")
            (root / "adapter.SAFETENSORS").write_bytes(b"upper")
            for weight_name in ("adapter.bin", "adapter.SAFETENSORS", "../outside.safetensors"):
                with self.subTest(weight_name=weight_name):
                    with self.assertRaisesRegex(ValueError, "lowercase .safetensors|stay inside"):
                        build_lora_descriptor(
                            selection={"source": "local", "value": str(root)},
                            weight_name=weight_name,
                            revision="",
                            expected_sha256="",
                            adapter_name="adapter",
                            scale=1,
                        )

    def test_raw_and_source_inferred_selections_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight)
            with self.assertRaisesRegex(TypeError, "explicitly provide source and value"):
                build_lora_descriptor(
                    selection=str(weight),
                    weight_name=weight.name,
                    revision="",
                    expected_sha256="",
                    adapter_name="adapter",
                    scale=1,
                )
            with self.assertRaisesRegex(ValueError, "exactly 'hub' or 'local'"):
                build_lora_descriptor(
                    selection={"source": "", "value": str(weight)},
                    weight_name=weight.name,
                    revision="",
                    expected_sha256="",
                    adapter_name="adapter",
                    scale=1,
                )

    def test_hub_identity_is_revision_aware_and_preserves_the_snapshot_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory).resolve()
            repo_root = cache_root / "models--example--style"
            alias = repo_root / "snapshots" / REVISION / "weights" / "style.safetensors"
            blob = repo_root / "blobs" / ("b" * 64)
            alias.parent.mkdir(parents=True)
            blob.parent.mkdir(parents=True)
            _write_tiny_safetensors(alias)
            blob.write_bytes(alias.read_bytes())
            digest = hashlib.sha256(alias.read_bytes()).hexdigest()

            with patch.dict(CONFIG.hf, {"cache_dir": str(cache_root)}):
                with patch("utils.huggingface.cached_file_path", return_value=str(alias)) as cached:
                    with patch("utils.huggingface.resolve_managed_hf_cache_file", return_value=blob):
                        descriptor = build_lora_descriptor(
                            selection={"source": "hub", "value": "example/style"},
                            weight_name="weights/style.safetensors",
                            revision=REVISION,
                            expected_sha256=digest,
                            adapter_name="style",
                            scale=0.5,
                        )
                        resolved = resolve_lora_descriptor(descriptor)

        self.assertEqual(resolved.load_directory, alias.parent.resolve())
        self.assertEqual(resolved.weight_name, "style.safetensors")
        self.assertNotEqual(resolved.load_directory, blob.parent)
        self.assertEqual(cached.call_args.kwargs["revision"], REVISION)

    def test_hub_requires_exact_repository_revision_hash_and_alias(self):
        cases = (
            ({"source": "hub", "value": "single-name"}, REVISION, "a" * 64, "style.safetensors", "namespace/repository"),
            ({"source": "hub", "value": "example/style"}, "main", "a" * 64, "style.safetensors", "40-character"),
            ({"source": "hub", "value": "example/style"}, REVISION.upper(), "a" * 64, "style.safetensors", "40-character"),
            ({"source": "hub", "value": "example/style"}, REVISION, "", "style.safetensors", "64 lowercase"),
            ({"source": "hub", "value": "example/style"}, REVISION, "a" * 64, "style.bin", "lowercase .safetensors"),
        )
        for selection, revision, digest, weight_name, message in cases:
            with self.subTest(revision=revision, weight_name=weight_name):
                with self.assertRaisesRegex(ValueError, message):
                    build_lora_descriptor(
                        selection=selection,
                        weight_name=weight_name,
                        revision=revision,
                        expected_sha256=digest,
                        adapter_name="style",
                        scale=1,
                    )

    def test_hub_cache_hit_must_use_the_exact_repository_snapshot_lexical_path(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory).resolve()
            wrong_alias = (
                cache_root
                / "models--other--style"
                / "snapshots"
                / REVISION
                / "style.safetensors"
            )
            wrong_alias.parent.mkdir(parents=True)
            wrong_alias.write_bytes(b"same bytes")
            digest = hashlib.sha256(wrong_alias.read_bytes()).hexdigest()
            with patch.dict(CONFIG.hf, {"cache_dir": str(cache_root)}):
                with patch("utils.huggingface.cached_file_path", return_value=str(wrong_alias)):
                    with self.assertRaisesRegex(ValueError, "exact repository snapshot path"):
                        build_lora_descriptor(
                            selection={"source": "hub", "value": "example/style"},
                            weight_name="style.safetensors",
                            revision=REVISION,
                            expected_sha256=digest,
                            adapter_name="style",
                            scale=1,
                        )

    def test_hub_resolved_blob_must_remain_in_the_selected_repository_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "hub"
            alias = (
                cache_root
                / "models--example--style"
                / "snapshots"
                / REVISION
                / "style.safetensors"
            )
            outside = root / "outside" / "blob"
            alias.parent.mkdir(parents=True)
            outside.parent.mkdir(parents=True)
            alias.write_bytes(b"same bytes")
            outside.write_bytes(b"same bytes")
            digest = hashlib.sha256(alias.read_bytes()).hexdigest()
            with patch.dict(CONFIG.hf, {"cache_dir": str(cache_root)}):
                with patch("utils.huggingface.cached_file_path", return_value=str(alias)):
                    with patch("utils.huggingface.resolve_managed_hf_cache_file", return_value=outside):
                        with self.assertRaisesRegex(ValueError, "outside its managed repository cache"):
                            build_lora_descriptor(
                                selection={"source": "hub", "value": "example/style"},
                                weight_name="style.safetensors",
                                revision=REVISION,
                                expected_sha256=digest,
                                adapter_name="style",
                                scale=1,
                            )

    def test_consumer_rejects_legacy_partial_extra_and_tampered_descriptors_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight)
            descriptor = _local_descriptor(weight)
            values = [
                "",
                [],
                {},
                {"lora_path": str(weight.parent), "weight_name": weight.name, "adapter_name": "legacy"},
                {key: value for key, value in descriptor.items() if key != "descriptor_sha256"},
                {**descriptor, "unexpected": True},
                {**descriptor, "scale": 0.25},
            ]
            for value in values:
                pipeline = MutationTrackingPipeline()
                with self.subTest(value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        update_lora_adapters(pipeline, value)
                    self.assertEqual(pipeline.events, [])

    def test_consumer_rehashes_a_to_b_file_mutation_before_any_pipeline_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight, 1)
            descriptor = _local_descriptor(weight)
            _write_tiny_safetensors(weight, 2)
            pipeline = MutationTrackingPipeline()

            with self.assertRaisesRegex(ValueError, "no longer matches"):
                update_lora_adapters(pipeline, descriptor)

        self.assertEqual(pipeline.events, [])

    def test_whole_list_is_validated_before_an_existing_adapter_is_unloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.safetensors"
            second = root / "second.safetensors"
            _write_tiny_safetensors(first, 1)
            _write_tiny_safetensors(second, 2)
            first_descriptor = _local_descriptor(first, node_id="first")
            second_descriptor = _local_descriptor(second, node_id="second")
            _write_tiny_safetensors(second, 3)
            pipeline = MutationTrackingPipeline()

            with self.assertRaisesRegex(ValueError, "no longer matches"):
                update_lora_adapters(pipeline, [first_descriptor, second_descriptor])

        self.assertEqual(pipeline.events, [])

    def test_malformed_or_empty_safetensors_fail_before_modular_pipeline_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            for label in ("malformed", "empty"):
                _write_tiny_safetensors(weight)
                descriptor = _local_descriptor(weight)
                if label == "malformed":
                    weight.write_bytes(b"not-a-safetensors-file")
                    message = "valid Safetensors"
                else:
                    save_file({}, str(weight))
                    message = "at least one tensor"
                descriptor["artifact"]["sha256"] = hashlib.sha256(weight.read_bytes()).hexdigest()
                _resign_descriptor(descriptor)
                pipeline = MutationTrackingPipeline()

                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, message):
                        update_lora_adapters(pipeline, descriptor)
                    self.assertEqual(pipeline.events, [])

    def test_scheduler_contract_rejects_dynamic_imports_duplicate_keys_and_oversized_json(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight)
            selection = {"source": "local", "value": str(weight)}
            cases = (
                ("os.system", "{}", "simple Diffusers export"),
                ("DiffusionPipeline", "{}", "reviewed scheduler contract"),
                ("SchedulerMixin", "{}", "reviewed scheduler contract"),
                ("EulerDiscreteScheduler", "{}", "reviewed scheduler contract"),
                ("FlowMatchEulerDiscreteScheduler", '{"shift":1,"shift":2}', "Duplicate scheduler JSON key"),
                ("FlowMatchEulerDiscreteScheduler", '{"value":"' + ("x" * 17000) + '"}', "byte limit"),
                (
                    "FlowMatchEulerDiscreteScheduler",
                    '{"pretrained_model_name_or_path":"attacker/scheduler"}',
                    "unsupported constructor parameters",
                ),
                (
                    "FlowMatchEulerDiscreteScheduler",
                    '{"return_unused_kwargs":true}',
                    "unsupported constructor parameters",
                ),
                (
                    "FlowMatchEulerDiscreteScheduler",
                    '{"nested":' + ("[" * 1100) + "0" + ("]" * 1100) + "}",
                    "nesting limit",
                ),
            )
            for scheduler_class, config, message in cases:
                with self.subTest(scheduler_class=scheduler_class, message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        build_lora_descriptor(
                            selection=selection,
                            weight_name=weight.name,
                            revision="",
                            expected_sha256="",
                            adapter_name="style",
                            scale=1,
                            scheduler_class=scheduler_class,
                            scheduler_config=config,
                        )

    def test_consumer_rejects_diffusers_config_controls_without_config_io_or_pipeline_mutation(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "lightning.safetensors"
            _write_tiny_safetensors(weight)
            descriptor = Lora("lightning").execute(
                {"source": "local", "value": str(weight)},
                1,
                weight_name=weight.name,
                scheduler_class="FlowMatchEulerDiscreteScheduler",
                scheduler_config={"base_shift": 1.25},
            )["lora"]

            for config_key, config_value in (
                ("pretrained_model_name_or_path", "attacker/scheduler"),
                ("return_unused_kwargs", True),
            ):
                tampered = copy.deepcopy(descriptor)
                tampered["scheduler"]["config"] = {config_key: config_value}
                _resign_descriptor(tampered)
                pipeline = MutationTrackingPipeline()
                pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
                pipeline.update_components = lambda **_components: pipeline.events.append("update")
                with self.subTest(config_key=config_key):
                    with patch.object(
                        FlowMatchEulerDiscreteScheduler,
                        "load_config",
                        side_effect=AssertionError("scheduler config I/O must stay unreachable"),
                    ) as load_config:
                        with self.assertRaisesRegex(ValueError, "unsupported constructor parameters"):
                            update_lora_adapters(pipeline, tampered)
                        load_config.assert_not_called()
                    self.assertEqual(pipeline.events, [])

    def test_flow_match_scheduler_resource_values_are_bounded_before_construction_or_mutation(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "lightning.safetensors"
            _write_tiny_safetensors(weight)
            with self.assertRaisesRegex(ValueError, "num_train_timesteps"):
                Lora("oversized-scheduler").execute(
                    {"source": "local", "value": str(weight)},
                    1,
                    weight_name=weight.name,
                    scheduler_class="FlowMatchEulerDiscreteScheduler",
                    scheduler_config={"num_train_timesteps": 1_000_000_000},
                )

            descriptor = Lora("bounded-scheduler").execute(
                {"source": "local", "value": str(weight)},
                1,
                weight_name=weight.name,
                scheduler_class="FlowMatchEulerDiscreteScheduler",
                scheduler_config={"base_shift": 1.25},
            )["lora"]
            pipeline = MutationTrackingPipeline()
            current_config = dict(FlowMatchEulerDiscreteScheduler().config)
            current_config["num_train_timesteps"] = 1_000_000_000
            pipeline.scheduler = type("CurrentScheduler", (), {"config": current_config})()
            pipeline.update_components = lambda **_components: pipeline.events.append("update")

            with patch(
                "diffusers.schedulers.scheduling_flow_match_euler_discrete.np.linspace",
                side_effect=AssertionError("oversized scheduler allocation must stay unreachable"),
            ) as linspace:
                with self.assertRaisesRegex(ValueError, "num_train_timesteps"):
                    update_lora_adapters(pipeline, descriptor)
                linspace.assert_not_called()

        self.assertEqual(pipeline.events, [])

    def test_hostile_descriptor_containers_are_bounded_before_copy_or_pipeline_mutation(self):
        class HostileDict(dict):
            def __deepcopy__(self, _memo):
                raise AssertionError("hostile descriptor must not reach deepcopy")

        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight)
            descriptor = _local_descriptor(weight)
            cases = []

            deeply_nested = copy.deepcopy(descriptor)
            nested = {}
            for _ in range(600):
                nested = {"nested": nested}
            deeply_nested["scheduler"] = {
                "class_name": "FlowMatchEulerDiscreteScheduler",
                "config": nested,
            }
            cases.append(("deep", deeply_nested, "nesting limit"))

            cyclic = copy.deepcopy(descriptor)
            cycle = {}
            cycle["cycle"] = cycle
            cyclic["scheduler"] = {
                "class_name": "FlowMatchEulerDiscreteScheduler",
                "config": cycle,
            }
            cases.append(("cycle", cyclic, "nesting limit|value limit"))

            custom = copy.deepcopy(descriptor)
            custom["scheduler"] = {
                "class_name": "FlowMatchEulerDiscreteScheduler",
                "config": HostileDict(base_shift=1.25),
            }
            cases.append(("custom", custom, "unsupported value type HostileDict"))

            oversized_top_level = {f"field_{index}": index for index in range(257)}
            cases.append(("oversized-top-level", oversized_top_level, "item object limit"))

            for label, value, message in cases:
                pipeline = MutationTrackingPipeline()
                with self.subTest(label=label):
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        update_lora_adapters(pipeline, value)
                    self.assertEqual(pipeline.events, [])

    def test_scheduler_wrong_return_type_fails_before_any_pipeline_mutation(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        class SchedulerPipeline(MutationTrackingPipeline):
            def __init__(self):
                super().__init__()
                self.scheduler = FlowMatchEulerDiscreteScheduler()

            def update_components(self, **components):
                self.events.append(("update_components", components))

        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "lightning.safetensors"
            _write_tiny_safetensors(weight)
            descriptor = Lora("lightning").execute(
                {"source": "local", "value": str(weight)},
                1,
                weight_name=weight.name,
                scheduler_class="FlowMatchEulerDiscreteScheduler",
                scheduler_config={"base_shift": 1.25},
            )["lora"]
            pipeline = SchedulerPipeline()

            with patch.object(
                FlowMatchEulerDiscreteScheduler,
                "from_config",
                return_value=(FlowMatchEulerDiscreteScheduler(), {}),
            ):
                with self.assertRaisesRegex(ValueError, "exact scheduler instance"):
                    update_lora_adapters(pipeline, descriptor)

        self.assertEqual(pipeline.events, [])

    def test_modular_scheduler_is_preconstructed_and_bound_to_the_same_pipeline(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        class SchedulerPipeline(MutationTrackingPipeline):
            def __init__(self):
                super().__init__()
                self.scheduler = FlowMatchEulerDiscreteScheduler()

            def update_components(self, **components):
                self.events.append(("update_components", components))
                self.scheduler = components["scheduler"]

        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "lightning.safetensors"
            _write_tiny_safetensors(weight)
            descriptor = Lora("lightning").execute(
                {"source": "local", "value": str(weight)},
                1,
                weight_name=weight.name,
                scheduler_class="FlowMatchEulerDiscreteScheduler",
                scheduler_config={"base_shift": 1.25},
            )["lora"]

            no_scheduler = MutationTrackingPipeline()
            with self.assertRaisesRegex(ValueError, "does not expose one"):
                update_lora_adapters(no_scheduler, descriptor)
            self.assertEqual(no_scheduler.events, [])

            pipeline = SchedulerPipeline()
            prepared = update_lora_adapters(pipeline, descriptor)
            other = SchedulerPipeline()
            with self.assertRaisesRegex(ValueError, "different pipeline"):
                apply_lora_scheduler_override(other, prepared=prepared)
            self.assertEqual(other.events, [])

            scheduler = apply_lora_scheduler_override(pipeline, prepared=prepared)

        load_event = next(event for event in pipeline.events if isinstance(event, tuple) and event[0] == "load")
        self.assertTrue(load_event[2]["use_safetensors"])
        self.assertEqual(load_event[2]["weight_name"], "lightning.safetensors")
        self.assertIs(pipeline.scheduler, scheduler)
        self.assertAlmostEqual(scheduler.config.base_shift, 1.25)

    def test_descriptor_digest_cannot_be_recomputed_around_an_invalid_artifact_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "style.safetensors"
            _write_tiny_safetensors(weight)
            descriptor = _local_descriptor(weight)
            tampered = copy.deepcopy(descriptor)
            tampered["artifact"]["repository"] = "example/style"
            _resign_descriptor(tampered)

            with self.assertRaisesRegex(ValueError, "Local LoRA artifact fields"):
                resolve_lora_descriptor(tampered)


if __name__ == "__main__":
    unittest.main()
