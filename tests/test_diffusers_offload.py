import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modiff.diffusers_offload import (
    OFFLOAD_MODE_GROUP_CPU,
    OFFLOAD_MODE_GROUP_DISK,
    OFFLOAD_MODE_MODEL_CPU,
    OFFLOAD_MODE_NONE,
    OFFLOAD_MODE_SEQUENTIAL_CPU,
    apply_component_group_offload,
    apply_pipeline_offload,
    normalize_offload_mode,
)
from modiff.diffusers_profiles import QWEN_IMAGE_2512_PREQUANTIZED_REPO, public_execution_profiles
from modules.ModularDiffusers.denoise import embeddings_are_missing, embeddings_missing_error
from modules.ModularDiffusers.embeddings import extract_prompt_embeddings
from modules.ModularDiffusers.loaders import normalize_quant_config_input
from modules.ModularDiffusers.loaders import RequiredComponentLoadError, load_components_strict
from modules.QwenImage.main import build_qwen_pipeline_quantization_config, coerce_pipeline_quantization_config


class FakePipelineState:
    def __init__(self, values=None, kwargs_values=None, raise_kwargs=False):
        self.values = values or {}
        self.kwargs_values = kwargs_values or {}
        self.raise_kwargs = raise_kwargs

    def get_by_kwargs(self, kwargs_type):
        if self.raise_kwargs:
            raise KeyError(kwargs_type)
        return self.kwargs_values.get(kwargs_type, {})

    def get(self, key, default=None):
        return self.values.get(key, default)


class FakePipeline:
    def __init__(self):
        self.calls = []
        self.transformer = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.Linear(2, 2),
        )

    def to(self, device):
        self.calls.append(("to", str(device)))

    def enable_model_cpu_offload(self, device=None):
        self.calls.append(("model_cpu", str(device)))

    def enable_sequential_cpu_offload(self, device=None):
        self.calls.append(("sequential_cpu", str(device)))


class FakeCudaRuntime:
    def __init__(self, *, free_bytes=14 * 1024 ** 3, total_bytes=16 * 1024 ** 3):
        self.free_bytes = free_bytes
        self.total_bytes = total_bytes
        self.fractions = []

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def mem_get_info(self, *_args):
        return self.free_bytes, self.total_bytes

    def set_per_process_memory_fraction(self, fraction, cuda_index):
        self.fractions.append((fraction, cuda_index))


class FakeTorchRuntime:
    def __init__(self, cuda=None):
        self.cuda = cuda or FakeCudaRuntime()


class FakeComponentSpec:
    default_creation_method = "from_pretrained"
    pretrained_model_name_or_path = "fake/repo"

    def __init__(self, name, error=None):
        self.name = name
        self.error = error

    def load(self, **kwargs):
        if self.error:
            raise self.error
        return {"name": self.name, "kwargs": kwargs}


class FakeStrictPipeline:
    _pretrained_model_name_or_path = "fake/repo"

    def __init__(self, specs):
        self._component_specs = specs
        self.registered = {}

    def register_components(self, **components):
        self.registered.update(components)
        for name, component in components.items():
            setattr(self, name, component)


class DiffusersOffloadSmokeTest(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree(Path("data") / "offload" / "diffusers" / "smoke-node", ignore_errors=True)

    def test_normalizes_legacy_auto_cpu(self):
        self.assertEqual(normalize_offload_mode("auto_cpu", auto_offload=True), OFFLOAD_MODE_MODEL_CPU)
        self.assertEqual(normalize_offload_mode(OFFLOAD_MODE_GROUP_CPU, auto_offload=True), OFFLOAD_MODE_GROUP_CPU)
        self.assertEqual(normalize_offload_mode(OFFLOAD_MODE_GROUP_CPU, auto_offload=False), OFFLOAD_MODE_NONE)

    def test_pipeline_model_and_sequential_cpu_offload(self):
        model_pipeline = FakePipeline()
        model_result = apply_pipeline_offload(
            model_pipeline,
            mode=OFFLOAD_MODE_MODEL_CPU,
            device="cuda:0",
            node_id="smoke-node",
        )
        self.assertTrue(model_result.applied)
        self.assertEqual(model_pipeline.calls, [("model_cpu", "cuda:0")])

        sequential_pipeline = FakePipeline()
        sequential_result = apply_pipeline_offload(
            sequential_pipeline,
            mode=OFFLOAD_MODE_SEQUENTIAL_CPU,
            device="cuda:0",
            node_id="smoke-node",
        )
        self.assertTrue(sequential_result.applied)
        self.assertEqual(sequential_pipeline.calls, [("sequential_cpu", "cuda:0")])

    def test_group_cpu_and_disk_component_offload(self):
        group_pipeline = FakePipeline()
        group_result = apply_component_group_offload(
            group_pipeline,
            component_names=["transformer"],
            device="cpu",
            mode=OFFLOAD_MODE_GROUP_CPU,
            node_id="smoke-node",
            scope="smoke",
        )
        self.assertTrue(group_result.applied)
        self.assertEqual(group_result.components, ["transformer"])

        disk_pipeline = FakePipeline()
        disk_result = apply_component_group_offload(
            disk_pipeline,
            component_names=["transformer"],
            device="cpu",
            mode=OFFLOAD_MODE_GROUP_DISK,
            node_id="smoke-node",
            scope="smoke",
        )
        self.assertTrue(disk_result.applied)
        self.assertTrue(disk_result.disk_path)
        self.assertTrue(Path(disk_result.disk_path).exists())

    def test_vae_group_offload_uses_leaf_hooks_for_direct_encode_decode(self):
        pipeline = FakePipeline()
        pipeline.vae = torch.nn.Sequential(torch.nn.Conv3d(3, 3, kernel_size=1))
        calls = []

        def capture_group_offload(module, **kwargs):
            calls.append((module, kwargs))

        with patch("diffusers.hooks.apply_group_offloading", capture_group_offload):
            result = apply_component_group_offload(
                pipeline,
                component_names=["vae", "transformer"],
                device="cuda:0",
                mode=OFFLOAD_MODE_GROUP_DISK,
                node_id="smoke-node",
                scope="direct-vae-entrypoints",
            )

        self.assertEqual(result.components, ["vae", "transformer"])
        self.assertEqual(calls[0][0], pipeline.vae)
        self.assertEqual(calls[0][1]["offload_type"], "leaf_level")
        self.assertIsNone(calls[0][1]["num_blocks_per_group"])
        self.assertEqual(calls[1][0], pipeline.transformer)
        self.assertEqual(calls[1][1]["offload_type"], "block_level")
        self.assertEqual(calls[1][1]["num_blocks_per_group"], 2)

    def test_quant_config_string_is_rejected_with_actionable_error(self):
        self.assertIsNone(normalize_quant_config_input(None))
        self.assertEqual(normalize_quant_config_input({"transformer": "config"}), {"transformer": "config"})

        with self.assertRaisesRegex(TypeError, "Quant Config must be connected"):
            normalize_quant_config_input("quantization_config")

        with self.assertRaisesRegex(ValueError, "must be connected"):
            coerce_pipeline_quantization_config("quantization_config")

    def test_public_execution_profiles_include_direct_qwen_fallback(self):
        profiles = {profile["id"]: profile for profile in public_execution_profiles()}
        qwen_profile = profiles["qwen-image:t2i-direct"]
        self.assertEqual(qwen_profile["backend_path"], "modules.QwenImage.LoadPipeline")
        self.assertEqual(qwen_profile["pipeline_class"], "QwenImagePipeline")
        self.assertEqual(qwen_profile["fallback_repo"], QWEN_IMAGE_2512_PREQUANTIZED_REPO)
        self.assertEqual(qwen_profile["default_quantized_components"], [])
        self.assertIn(OFFLOAD_MODE_MODEL_CPU, qwen_profile["retry_offload_modes"])
        self.assertIn(OFFLOAD_MODE_SEQUENTIAL_CPU, qwen_profile["retry_offload_modes"])
        self.assertIn(OFFLOAD_MODE_GROUP_DISK, qwen_profile["retry_offload_modes"])

    def test_qwen_pipeline_quant_config_supports_component_fallback(self):
        quant_config = build_qwen_pipeline_quantization_config(
            components=["transformer", "text_encoder"],
            quantization_mode="bnb_4bit",
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
            double_quant=True,
        )
        self.assertIsNotNone(quant_config)
        self.assertEqual(sorted(quant_config.quant_mapping.keys()), ["text_encoder", "transformer"])

        transformer_only = build_qwen_pipeline_quantization_config(
            components=["transformer"],
            quantization_mode="bnb_4bit",
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
            double_quant=True,
        )
        self.assertEqual(list(transformer_only.quant_mapping.keys()), ["transformer"])

    def test_strict_component_loading_raises_for_required_component_oom(self):
        original = RuntimeError("CUDA out of memory while loading text_encoder")
        pipeline = FakeStrictPipeline({
            "text_encoder": FakeComponentSpec("text_encoder", original),
            "scheduler": FakeComponentSpec("scheduler"),
        })
        diagnostics = {}

        with self.assertRaisesRegex(RequiredComponentLoadError, "text_encoder") as context:
            load_components_strict(
                pipeline,
                ["text_encoder", "scheduler"],
                required_names={"text_encoder", "scheduler"},
                model_id="Qwen/Qwen-Image-2512",
                dtype="bfloat16",
                offload_mode=OFFLOAD_MODE_GROUP_CPU,
                quant_config={"text_encoder": "bnb_4bit"},
                diagnostics=diagnostics,
                component_load_kwargs={
                    "torch_dtype": "bfloat16",
                    "quantization_config": {"text_encoder": "bnb_4bit"},
                },
            )

        self.assertIs(context.exception.__cause__, original)
        self.assertIn("CUDA out of memory", str(context.exception))
        self.assertEqual(diagnostics["components_failed"][0]["name"], "text_encoder")

    def test_strict_component_loading_keeps_optional_component_failure_diagnostic(self):
        pipeline = FakeStrictPipeline({
            "optional_encoder": FakeComponentSpec("optional_encoder", RuntimeError("optional failed")),
            "scheduler": FakeComponentSpec("scheduler"),
        })
        diagnostics = {}

        load_components_strict(
            pipeline,
            ["optional_encoder", "scheduler"],
            required_names={"scheduler"},
            model_id="fake/repo",
            dtype="bfloat16",
            offload_mode=OFFLOAD_MODE_MODEL_CPU,
            quant_config=None,
            diagnostics=diagnostics,
            component_load_kwargs={"torch_dtype": "bfloat16"},
        )

        self.assertEqual(diagnostics["components_failed"][0]["name"], "optional_encoder")
        self.assertIn("scheduler", diagnostics["components_loaded"])
        self.assertIn("scheduler", pipeline.registered)

    def test_prompt_embeddings_extract_from_kwargs_or_prompt_fields(self):
        mapped = FakePipelineState(kwargs_values={"denoiser_input_fields": {"prompt_embeds": "mapped"}})
        self.assertEqual(extract_prompt_embeddings(mapped), {"prompt_embeds": "mapped"})

        fallback = FakePipelineState(
            values={
                "prompt_embeds": "prompt",
                "prompt_embeds_mask": "mask",
                "negative_prompt_embeds": "negative",
                "negative_prompt_embeds_mask": "negative-mask",
            }
        )
        self.assertEqual(
            extract_prompt_embeddings(fallback),
            {
                "prompt_embeds": "prompt",
                "prompt_embeds_mask": "mask",
                "negative_prompt_embeds": "negative",
                "negative_prompt_embeds_mask": "negative-mask",
            },
        )

        fallback_after_key_error = FakePipelineState(values={"prompt_embeds": "prompt"}, raise_kwargs=True)
        self.assertEqual(extract_prompt_embeddings(fallback_after_key_error), {"prompt_embeds": "prompt"})

        with self.assertRaisesRegex(KeyError, "embeddings"):
            extract_prompt_embeddings(FakePipelineState())

    def test_denoise_embedding_guards_identify_missing_embeddings(self):
        self.assertTrue(embeddings_are_missing(None))
        self.assertTrue(embeddings_are_missing({}))
        self.assertFalse(embeddings_are_missing({"prompt_embeds": "ok"}))
        self.assertTrue(embeddings_missing_error(KeyError("embeddings")))
        self.assertFalse(embeddings_missing_error(KeyError("scheduler")))

    def test_server_classifies_edge_resolution_embedding_key_error(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        server.current_task = None
        server._cuda_memory_snapshot = lambda: None
        server._gpu_process_snapshot = lambda: None

        classification = WebServer._classify_exception(server, KeyError("embeddings"))
        self.assertEqual(classification["category"], "graph_incomplete")
        self.assertEqual(classification["error_code"], "missing_prompt_embeddings")

        payload = WebServer._exception_payload(server, KeyError("embeddings"))
        self.assertEqual(payload["category"], "graph_incomplete")
        self.assertEqual(payload["error_code"], "missing_prompt_embeddings")
        self.assertIn("Prompt embeddings are missing", payload["message"])

    def test_server_classifies_chained_loader_oom_before_missing_embeddings(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        oom = RuntimeError("CUDA out of memory while loading text_encoder")
        wrapped = RuntimeError("Error executing modules.ModularDiffusers.ModelsLoader")
        wrapped.__cause__ = oom

        classification = WebServer._classify_exception(server, wrapped)
        self.assertEqual(classification["category"], "oom")
        self.assertEqual(classification["error_code"], "cuda_oom")
        self.assertIn("text_encoder", classification["message"])

    def test_server_classifies_quantized_cuda_kernel_unsupported(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        error = RuntimeError(
            "CUDA error: CUBLAS_STATUS_NOT_SUPPORTED when calling "
            "cublasLtMatmulAlgoGetHeuristic"
        )
        classification = WebServer._classify_exception(server, error)
        self.assertEqual(classification["category"], "cuda_kernel")
        self.assertEqual(classification["error_code"], "cuda_kernel_unsupported")

    def test_server_reports_missing_connected_output_with_source_node(self):
        from modiff.server import MissingConnectedOutputError, WebServer

        class SourceNode:
            module_name = "modules.ModularDiffusers"
            class_name = "EncodePrompt"
            output = {"doc": "ok"}

        server = object.__new__(WebServer)
        server.node_cache = {"source": SourceNode()}

        with self.assertRaisesRegex(MissingConnectedOutputError, "embeddings") as context:
            WebServer._connected_output_value(
                server,
                target_node_id="denoise",
                target_node_name="modules.ModularDiffusers.Denoise",
                target_param="embeddings",
                source_node_id="source",
                source_key="embeddings",
            )

        classification = WebServer._classify_exception(server, context.exception)
        self.assertEqual(classification["error_code"], "missing_connected_output")

    def test_cuda_budget_is_advisory_by_default_and_resets_memory_fraction(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        fake_torch = FakeTorchRuntime()

        with patch("modiff.server.import_module", lambda name: fake_torch if name == "torch" else None):
            result = WebServer._apply_cuda_runtime_budget(
                server,
                {
                    "device": "cuda:0",
                    "cudaBudgetPolicy": "advisory",
                    "requestedCudaBudgetBytes": 3 * 1024 ** 3,
                },
            )

        self.assertFalse(result["applied"])
        self.assertEqual(result["cuda_budget_policy"], "advisory")
        self.assertEqual(fake_torch.cuda.fractions, [(1.0, 0)])

    def test_cuda_budget_can_still_be_enforced_explicitly(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        fake_torch = FakeTorchRuntime()

        with patch("modiff.server.import_module", lambda name: fake_torch if name == "torch" else None):
            result = WebServer._apply_cuda_runtime_budget(
                server,
                {
                    "device": "cuda:0",
                    "cudaBudgetPolicy": "enforced",
                    "requestedCudaReserveBytes": 1 * 1024 ** 3,
                    "requestedCudaBudgetBytes": 2 * 1024 ** 3,
                },
            )

        self.assertTrue(result["applied"])
        self.assertEqual(result["applied_budget_bytes"], 2 * 1024 ** 3)
        self.assertAlmostEqual(fake_torch.cuda.fractions[0][0], 0.125)
        self.assertEqual(fake_torch.cuda.fractions[0][1], 0)

    def test_direct_qwen_loader_participates_in_generic_resource_retry(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "qwen-loader": {
                    "module": "modules.QwenImage",
                    "action": "LoadPipeline",
                    "params": {
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                        "auto_offload": {"value": True},
                    },
                },
            },
        }

        updated = WebServer._apply_resource_retry_to_graph(server, graph, OFFLOAD_MODE_GROUP_DISK)

        self.assertEqual(updated, ["qwen-loader"])
        self.assertEqual(graph["nodes"]["qwen-loader"]["params"]["offload_mode"]["value"], OFFLOAD_MODE_GROUP_DISK)

    def test_execute_graph_retries_oom_with_next_offload_mode_and_diagnostics(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        server.current_task = {"task_id": "task-1"}
        server.interrupt_flag = False
        server.queue_message = lambda *args, **kwargs: None
        server._apply_cuda_runtime_budget = lambda runtime_hints: {}
        server._apply_deterministic_mode = lambda graph: None
        server._runtime_fingerprint = lambda: {"fingerprint": "fake"}
        server._release_runtime_caches_for_retry = lambda: {"released": {}, "errors": []}
        server._loader_diagnostics_snapshot = lambda: {"loader-node": {"normalized_offload_mode": "group_cpu"}}
        calls = {"count": 0}

        def execute_node(node_id, node, sid):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("CUDA out of memory while loading text_encoder")

        server.execute_node = execute_node
        graph = {
            "sid": "sid-1",
            "paths": [["loader-node"]],
            "runtimeHints": {
                "source": "studio",
                "device": "cuda:0",
                "offloadMode": OFFLOAD_MODE_GROUP_CPU,
                "resourceRetryModes": [OFFLOAD_MODE_GROUP_DISK],
            },
            "nodes": {
                "loader-node": {
                    "module": "modules.ModularDiffusers",
                    "action": "ModelsLoader",
                    "params": {
                        "offload_mode": {"value": OFFLOAD_MODE_GROUP_CPU},
                        "auto_offload": {"value": True},
                    },
                },
            },
        }

        result = WebServer.execute_graph(server, graph)
        self.assertIsNone(result)
        self.assertEqual(calls["count"], 2)
        self.assertEqual(graph["nodes"]["loader-node"]["params"]["offload_mode"]["value"], OFFLOAD_MODE_GROUP_DISK)


if __name__ == "__main__":
    unittest.main()
