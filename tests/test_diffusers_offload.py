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
    apply_model_offload,
    apply_pipeline_offload,
    configure_components_manager_offload,
    normalize_execution_device,
    normalize_offload_mode,
    reset_pipeline_device_map_for_runtime,
    supports_accelerator_cpu_offload,
)
from modiff.diffusers_profiles import (
    QWEN_IMAGE_2512_PREQUANTIZED_REPO,
    execution_profiles_for_execution,
    optional_runtime_profile_ids_for_execution,
    public_execution_profiles,
)
from modiff.model_artifact_catalog import catalog_revision
from modiff.optional_runtime_execution import optional_runtime_requirement_for_execution
from modiff.studio_execution_specs import studio_execution_spec_for_pair, studio_model_dependencies_for_pair
from modules.ModularDiffusers.denoise import embeddings_are_missing, embeddings_missing_error
from modules.ModularDiffusers.embeddings import extract_prompt_embeddings
from modules.ModularDiffusers.loaders import normalize_quant_config_input
from modules.ModularDiffusers.loaders import (
    AutoModelLoader,
    ModelsLoader,
    RequiredComponentLoadError,
    component_reuse_compatible,
    load_components_strict,
    place_pipeline_components,
    record_pipeline_component_runtime_policy,
    reusable_standalone_component,
    should_incrementally_group_offload,
)
from modules.DiffusersImage.main import build_qwen_pipeline_quantization_config, coerce_pipeline_quantization_config


def resource_plan_target(model_type, mode):
    profiles = execution_profiles_for_execution(model_type, mode)
    if len(profiles) != 1:
        raise AssertionError(f"Expected one execution profile for {model_type}:{mode}, got {len(profiles)}")
    profile = profiles[0]
    return {
        "autoResourceSchemaVersion": 2,
        "executionProfileId": profile.id,
        "modelType": model_type,
        "mode": mode,
        "loaderModule": profile.loader_module,
        "loaderAction": profile.loader_action,
        "executionPath": profile.execution_path,
        "pipelineClass": profile.pipeline_class,
    }


def auto_resource_plan_target(model_type, mode):
    target = {
        **resource_plan_target(model_type, mode),
        "modelDependencies": studio_model_dependencies_for_pair(model_type, mode),
        "optionalRuntimeProfileIds": list(
            optional_runtime_profile_ids_for_execution(model_type, mode)
        ),
        "optionalRuntimeRequirement": optional_runtime_requirement_for_execution(
            model_type,
            mode,
        ),
    }
    specification = studio_execution_spec_for_pair(model_type, mode)
    if specification is not None:
        target["studioExecutionSpecContract"] = {
            "schemaVersion": specification["schemaVersion"],
            "id": specification["id"],
            "contentHash": specification["contentHash"],
            "executionProfileId": specification["executionProfileId"],
        }
    return target


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


class FakeDeviceMappedPipeline(FakePipeline):
    hf_device_map = {"transformer": 0, "text_encoder": "cpu"}

    def reset_device_map(self):
        self.calls.append(("reset_device_map", None))
        self.hf_device_map = None


class FakeResidentDeviceMappedPipeline(FakeDeviceMappedPipeline):
    hf_device_map = {"transformer": 0, "text_encoder": "cuda:0"}


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
    def test_generic_component_filter_does_not_assign_an_uninstalled_repository(self):
        node = AutoModelLoader("generic-component-filter")

        with patch.object(node, "set_field_params") as set_field_params:
            node.set_filters({"model_type": "controlnet"}, None)

        model_call = next(call for call in set_field_params.call_args_list if call.args[0] == "model_id")
        params = model_call.args[1]
        filters = params["fieldOptions"]["filter"]
        class_names = filters["hub"]["className"]
        self.assertEqual(class_names, sorted(class_names))
        self.assertEqual(filters["local"]["className"], class_names)
        self.assertIn("ControlNetModel", class_names)
        self.assertIn("QwenImageControlNetModel", class_names)
        self.assertIn("FluxControlNetModel", class_names)
        self.assertIn("ZImageControlNetModel", class_names)
        self.assertNotIn("FluxPipeline", class_names)
        self.assertNotIn("AutoencoderKL", class_names)
        self.assertNotIn("value", params)
        self.assertNotIn("default", params)

        self.assertIn(
            "every Hub component",
            AutoModelLoader.params["revision"]["description"],
        )

    def test_generic_pipeline_filter_preserves_selection_until_user_chooses_an_installed_repository(self):
        node = ModelsLoader("generic-pipeline-filter")
        node.model_types_loaded = True

        with patch.object(node, "set_field_params") as set_field_params:
            node.set_filters({"model_type": "QwenImageModularPipeline"}, None)

        repo_call = next(call for call in set_field_params.call_args_list if call.args[0] == "repo_id")
        params = repo_call.args[1]
        self.assertEqual(
            params["fieldOptions"]["filter"]["hub"]["className"],
            ["QwenImageModularPipeline"],
        )
        self.assertNotIn("value", params)
        self.assertNotIn("default", params)

    def test_auto_model_loader_rejects_pipeline_class_before_weight_load(self):
        node = AutoModelLoader("invalid-component-loader")

        with patch("modules.ModularDiffusers.loaders.ComponentSpec.load") as load:
            with self.assertRaisesRegex(ValueError, "requires a component type"):
                node.execute(
                    model_type="QwenImageModularPipeline",
                    model_id={"source": "hub", "value": "InstantX/Qwen-Image-ControlNet-Union"},
                    dtype=torch.bfloat16,
                    trust_remote_code=False,
                    device="cuda:0",
                    auto_offload=True,
                    offload_mode=OFFLOAD_MODE_MODEL_CPU,
                )

        load.assert_not_called()

    def tearDown(self):
        shutil.rmtree(Path("data") / "offload" / "diffusers" / "smoke-node", ignore_errors=True)

    def test_normalizes_legacy_auto_cpu(self):
        self.assertEqual(normalize_offload_mode("auto_cpu", auto_offload=True), OFFLOAD_MODE_MODEL_CPU)
        self.assertEqual(normalize_offload_mode(OFFLOAD_MODE_GROUP_CPU, auto_offload=True), OFFLOAD_MODE_GROUP_CPU)
        self.assertEqual(normalize_offload_mode(OFFLOAD_MODE_GROUP_CPU, auto_offload=False), OFFLOAD_MODE_NONE)
        self.assertEqual(
            normalize_offload_mode(OFFLOAD_MODE_MODEL_CPU, auto_offload=True, device="cpu:0"),
            OFFLOAD_MODE_NONE,
        )
        self.assertEqual(
            normalize_offload_mode(OFFLOAD_MODE_GROUP_CPU, auto_offload=True, device="mps"),
            OFFLOAD_MODE_NONE,
        )
        self.assertEqual(
            normalize_offload_mode(OFFLOAD_MODE_MODEL_CPU, auto_offload=True, device="cuda"),
            OFFLOAD_MODE_MODEL_CPU,
        )
        self.assertEqual(str(normalize_execution_device("cuda")), "cuda:0")
        self.assertTrue(supports_accelerator_cpu_offload("cuda:1"))
        self.assertFalse(supports_accelerator_cpu_offload("cpu"))
        self.assertFalse(supports_accelerator_cpu_offload("mps"))

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

    def test_pipeline_cpu_and_mps_execution_never_install_cpu_offload_hooks(self):
        for device in ("cpu:0", "mps"):
            for mode in (
                OFFLOAD_MODE_MODEL_CPU,
                OFFLOAD_MODE_SEQUENTIAL_CPU,
                OFFLOAD_MODE_GROUP_CPU,
                OFFLOAD_MODE_GROUP_DISK,
            ):
                with self.subTest(device=device, mode=mode):
                    pipeline = FakePipeline()
                    result = apply_pipeline_offload(
                        pipeline,
                        mode=mode,
                        device=device,
                        node_id="smoke-node",
                    )

                    self.assertEqual(result.mode, OFFLOAD_MODE_NONE)
                    self.assertEqual(result.method, "to_device")
                    self.assertEqual(pipeline.calls, [("to", device)])

    def test_standalone_model_cpu_execution_never_installs_group_hooks(self):
        model = torch.nn.Linear(2, 2)
        with patch(
            "diffusers.hooks.apply_group_offloading",
            side_effect=AssertionError("CPU execution must not install a group-offload hook"),
        ):
            result = apply_model_offload(
                model,
                component_name="transformer",
                mode=OFFLOAD_MODE_MODEL_CPU,
                device="cpu:0",
                node_id="smoke-node",
            )

        self.assertEqual(result.mode, OFFLOAD_MODE_NONE)
        self.assertEqual(result.method, "to_device")
        self.assertEqual(model.weight.device.type, "cpu")

    def test_real_components_manager_skips_auto_offload_for_cpu(self):
        from diffusers import ComponentsManager

        manager = ComponentsManager()
        result = configure_components_manager_offload(
            manager,
            mode=OFFLOAD_MODE_MODEL_CPU,
            device="cpu:0",
        )

        self.assertEqual(result.mode, OFFLOAD_MODE_NONE)
        self.assertEqual(result.method, "components_manager_no_offload")
        self.assertFalse(manager._auto_offload_enabled)

    def test_models_loader_cpu_contract_reaches_model_load_without_enabling_auto_offload(self):
        class StopAtModelLoad(RuntimeError):
            pass

        class CpuComponentsManager:
            _auto_offload_enabled = False
            _auto_offload_device = None

            def enable_auto_cpu_offload(self, **_kwargs):
                raise NotImplementedError(
                    "`enable_auto_cpu_offload()` relies on the `mem_get_info()` method. "
                    "It's not implemented for cpu."
                )

            def disable_auto_cpu_offload(self):
                raise AssertionError("An inactive manager should not need disabling.")

            def _lookup_ids(self, **_kwargs):
                return []

            def remove_from_collection(self, *_args, **_kwargs):
                return None

        manager = CpuComponentsManager()
        node = ModelsLoader("cpu-loader")
        with (
            patch("modules.ModularDiffusers.loaders.components", manager),
            patch(
                "modules.ModularDiffusers.loaders._validate_reviewed_pipeline_index",
                return_value=("model_index.json", {"_class_name": "QwenImagePipeline"}),
            ),
            patch(
                "modules.ModularDiffusers.loaders._instantiate_reviewed_builtin_pipeline",
                side_effect=StopAtModelLoad("model load reached"),
            ),
        ):
            with self.assertRaisesRegex(StopAtModelLoad, "model load reached"):
                node.execute(
                    model_type="QwenImageModularPipeline",
                    repo_id={"source": "hub", "value": "Qwen/Qwen-Image-2512"},
                    device="cpu:0",
                    dtype=torch.float32,
                    auto_offload=True,
                    offload_mode=OFFLOAD_MODE_MODEL_CPU,
                )

        self.assertEqual(node._loader_diagnostics["normalized_offload_mode"], OFFLOAD_MODE_NONE)

    def test_resident_modular_components_report_each_accelerator_copy(self):
        class RecordingModule(torch.nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name
                self.devices = []

            def to(self, device):
                self.devices.append(device)
                return self

        text_encoder = RecordingModule("text_encoder")
        transformer = RecordingModule("transformer")
        pipeline = type(
            "ResidentPipeline",
            (),
            {
                "components": {
                    "tokenizer": object(),
                    "text_encoder": text_encoder,
                    "transformer": transformer,
                }
            },
        )()
        progress = []

        names = place_pipeline_components(
            pipeline,
            "cuda:0",
            lambda name, index, total: progress.append((name, index, total)),
        )

        self.assertEqual(names, ["text_encoder", "transformer"])
        self.assertEqual(progress, [("text_encoder", 1, 2), ("transformer", 2, 2)])
        self.assertEqual(text_encoder.devices, ["cuda:0"])
        self.assertEqual(transformer.devices, ["cuda:0"])

    def test_standalone_component_reuse_requires_matching_identity_dtype_and_runtime_policy(self):
        class ResidentManager:
            def __init__(self, model):
                self.model = model

            def _lookup_ids(self, **_kwargs):
                return {"controlnet_1"}

            def get_one(self, *, component_id):
                self.last_component_id = component_id
                return self.model

        model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        model._diffusers_load_id = "repo/controlnet|null|null|null"
        model._modiff_offload_mode = OFFLOAD_MODE_NONE
        model._modiff_execution_device = "cpu"
        manager = ResidentManager(model)

        reused = reusable_standalone_component(
            manager,
            name="controlnet",
            load_id=model._diffusers_load_id,
            dtype=torch.bfloat16,
            offload_mode=OFFLOAD_MODE_NONE,
            device="cpu",
        )
        self.assertEqual(reused, ("controlnet_1", model))
        self.assertEqual(manager.last_component_id, "controlnet_1")

        self.assertIsNone(
            reusable_standalone_component(
                manager,
                name="controlnet",
                load_id=model._diffusers_load_id,
                dtype=torch.float16,
                offload_mode=OFFLOAD_MODE_NONE,
                device="cpu",
            )
        )
        self.assertIsNone(
            reusable_standalone_component(
                manager,
                name="controlnet",
                load_id=model._diffusers_load_id,
                dtype=torch.bfloat16,
                offload_mode=OFFLOAD_MODE_MODEL_CPU,
                device="cpu",
            )
        )

    def test_shared_component_reuse_requires_matching_offload_device_and_disk_owner(self):
        model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        model._modiff_offload_mode = OFFLOAD_MODE_GROUP_CPU
        model._modiff_execution_device = "cuda:0"
        model._modiff_offload_node_id = None

        self.assertTrue(
            component_reuse_compatible(
                model,
                dtype=torch.bfloat16,
                requested_quantization=None,
                offload_mode=OFFLOAD_MODE_GROUP_CPU,
                device="cuda:0",
                node_id="loader-a",
            )
        )
        self.assertFalse(
            component_reuse_compatible(
                model,
                dtype=torch.bfloat16,
                requested_quantization=None,
                offload_mode=OFFLOAD_MODE_GROUP_DISK,
                device="cuda:0",
                node_id="loader-a",
            )
        )
        self.assertFalse(
            component_reuse_compatible(
                model,
                dtype=torch.bfloat16,
                requested_quantization=None,
                offload_mode=OFFLOAD_MODE_GROUP_CPU,
                device="cuda:1",
                node_id="loader-a",
            )
        )

        pipeline = type("SharedPipeline", (), {"components": {"transformer": model}})()
        record_pipeline_component_runtime_policy(
            pipeline,
            offload_mode=OFFLOAD_MODE_GROUP_DISK,
            device="cuda:0",
            node_id="loader-a",
        )
        self.assertTrue(
            component_reuse_compatible(
                model,
                dtype=torch.bfloat16,
                requested_quantization=None,
                offload_mode=OFFLOAD_MODE_GROUP_DISK,
                device="cuda:0",
                node_id="loader-a",
            )
        )
        self.assertFalse(
            component_reuse_compatible(
                model,
                dtype=torch.bfloat16,
                requested_quantization=None,
                offload_mode=OFFLOAD_MODE_GROUP_DISK,
                device="cuda:0",
                node_id="loader-b",
            )
        )

    def test_device_map_is_reset_before_runtime_offload(self):
        pipeline = FakeDeviceMappedPipeline()

        result = apply_pipeline_offload(
            pipeline,
            mode=OFFLOAD_MODE_MODEL_CPU,
            device="cuda:0",
            node_id="smoke-node",
        )

        self.assertTrue(result.applied)
        self.assertEqual(
            pipeline.calls,
            [("reset_device_map", None), ("model_cpu", "cuda:0")],
        )

    def test_resident_device_map_is_preserved_without_offload(self):
        pipeline = FakeResidentDeviceMappedPipeline()

        result = apply_pipeline_offload(
            pipeline,
            mode=OFFLOAD_MODE_NONE,
            device="cuda:0",
            node_id="smoke-node",
        )

        self.assertTrue(result.applied)
        self.assertEqual(result.method, "preserve_device_map")
        self.assertEqual(pipeline.calls, [])
        self.assertEqual(pipeline.hf_device_map, {"transformer": 0, "text_encoder": "cuda:0"})

    def test_explicit_cuda_device_map_is_preserved_without_offload(self):
        pipeline = FakePipeline()
        pipeline.hf_device_map = "cuda"

        result = apply_pipeline_offload(
            pipeline,
            mode=OFFLOAD_MODE_NONE,
            device="cuda:0",
            node_id="loader",
        )

        self.assertEqual(result.method, "preserve_device_map")
        self.assertEqual(pipeline.calls, [])
        self.assertEqual(pipeline.hf_device_map, "cuda")

    def test_device_map_without_reset_support_fails_before_movement(self):
        pipeline = FakePipeline()
        pipeline.hf_device_map = {"transformer": 0}

        with self.assertRaisesRegex(RuntimeError, "cannot reset that placement"):
            reset_pipeline_device_map_for_runtime(pipeline)

        self.assertEqual(pipeline.calls, [])

    def test_cpu_execution_bypasses_group_cpu_and_disk_component_hooks(self):
        group_pipeline = FakePipeline()
        with patch(
            "diffusers.hooks.apply_group_offloading",
            side_effect=AssertionError("CPU execution must not install a group-offload hook"),
        ):
            group_result = apply_component_group_offload(
                group_pipeline,
                component_names=["transformer"],
                device="cpu",
                mode=OFFLOAD_MODE_GROUP_CPU,
                node_id="smoke-node",
                scope="smoke",
            )
        self.assertTrue(group_result.applied)
        self.assertEqual(group_result.mode, OFFLOAD_MODE_NONE)
        self.assertEqual(group_result.method, "to_device")
        self.assertEqual(group_result.components, ["transformer"])

        disk_pipeline = FakePipeline()
        with patch(
            "diffusers.hooks.apply_group_offloading",
            side_effect=AssertionError("CPU execution must not install a group-offload hook"),
        ):
            disk_result = apply_component_group_offload(
                disk_pipeline,
                component_names=["transformer"],
                device="cpu",
                mode=OFFLOAD_MODE_GROUP_DISK,
                node_id="smoke-node",
                scope="smoke",
            )
        self.assertTrue(disk_result.applied)
        self.assertEqual(disk_result.mode, OFFLOAD_MODE_NONE)
        self.assertEqual(disk_result.method, "to_device")
        self.assertIsNone(disk_result.disk_path)

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
        self.assertEqual(qwen_profile["backend_path"], "modules.DiffusersImage.LoadPipeline")
        self.assertEqual(qwen_profile["pipeline_class"], "QwenImagePipeline")
        self.assertEqual(qwen_profile["fallback_repo"], QWEN_IMAGE_2512_PREQUANTIZED_REPO)
        self.assertEqual(qwen_profile["default_quantized_components"], [])
        self.assertIn(OFFLOAD_MODE_MODEL_CPU, qwen_profile["retry_offload_modes"])
        self.assertIn(OFFLOAD_MODE_SEQUENTIAL_CPU, qwen_profile["retry_offload_modes"])
        self.assertIn(OFFLOAD_MODE_GROUP_DISK, qwen_profile["retry_offload_modes"])

    def test_qwen_pipeline_quant_config_supports_component_fallback(self):
        from importlib.util import find_spec

        if find_spec("bitsandbytes") is None:
            with self.assertRaisesRegex(RuntimeError, "BitsAndBytes 4-bit quantization is not installed"):
                build_qwen_pipeline_quantization_config(
                    components=["transformer"],
                    quantization_mode="bnb_4bit",
                    compute_dtype=torch.bfloat16,
                    quant_type="nf4",
                    double_quant=True,
                )
            return
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

    def test_incremental_group_offload_is_selected_by_quantized_components_not_pipeline_name(self):
        self.assertTrue(
            should_incrementally_group_offload(
                use_group_offload=True,
                quant_config={"transformer": "bnb_4bit", "text_encoder": "bnb_4bit"},
            )
        )
        self.assertFalse(
            should_incrementally_group_offload(
                use_group_offload=False,
                quant_config={"transformer": "bnb_4bit"},
            )
        )
        self.assertFalse(
            should_incrementally_group_offload(
                use_group_offload=True,
                quant_config={"vae": "bnb_4bit"},
            )
        )

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

    def test_generic_qwen_loader_participates_in_resource_retry(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "qwen-loader": {
                    "module": "modules.DiffusersImage",
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

    def test_native_auto_plan_applies_direct_cuda_device_map_to_image_loader(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "recipe": {
                    "module": "modules.DiffusersRuntime",
                    "action": "DiffusersExecutionRecipe",
                    "params": {
                        "device_map": {"value": "none"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
                "qwen-loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": {"source": "hub", "value": "old-model"}},
                        "revision": {"value": "a" * 40},
                        "pipeline_class": {"value": "QwenImagePipeline"},
                        "device_map": {"value": "none"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                        "auto_offload": {"value": True},
                        "execution_recipe": {"sourceId": "recipe", "sourceKey": "execution_recipe"},
                    },
                },
            },
        }

        updated = WebServer._apply_resource_retry_plan_to_graph(
            server,
            graph,
            {
                **resource_plan_target("QwenImageModularPipeline", "text_to_image"),
                "modelRepo": "Qwen/Qwen-Image-2512",
                "offloadMode": OFFLOAD_MODE_NONE,
                "deviceMap": "cuda",
            },
        )

        params = graph["nodes"]["qwen-loader"]["params"]
        self.assertEqual(updated, ["recipe", "qwen-loader"])
        self.assertEqual(params["device_map"]["value"], "cuda")
        self.assertEqual(params["offload_mode"]["value"], OFFLOAD_MODE_NONE)
        self.assertFalse(params["auto_offload"]["value"])
        self.assertEqual(params["model_id"]["value"]["value"], "Qwen/Qwen-Image-2512")
        self.assertEqual(params["revision"]["value"], catalog_revision("Qwen/Qwen-Image-2512"))
        self.assertEqual(graph["nodes"]["recipe"]["params"]["device_map"]["value"], "cuda")
        self.assertEqual(graph["nodes"]["recipe"]["params"]["offload_mode"]["value"], OFFLOAD_MODE_NONE)

    def test_structured_audio_plan_does_not_rewrite_independent_video_loader(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "audio-recipe": {
                    "module": "modules.DiffusersRuntime",
                    "action": "DiffusersExecutionRecipe",
                    "params": {
                        "device_map": {"value": "none"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
                "audio-loader": {
                    "module": "modules.DiffusersAudio",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": {"source": "hub", "value": "old-audio"}},
                        "revision": {"value": "a" * 40},
                        "pipeline_class": {"value": "AceStepPipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                        "auto_offload": {"value": True},
                        "execution_recipe": {
                            "sourceId": "audio-recipe",
                            "sourceKey": "execution_recipe",
                        },
                    },
                },
                "audio-generate": {
                    "module": "modules.DiffusersAudio",
                    "action": "Generate",
                    "params": {"audio_duration": {"value": 12}, "num_inference_steps": {"value": 4}},
                },
                "video-loader": {
                    "module": "modules.DiffusersVideo",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": {"source": "hub", "value": "Lightricks/LTX-Video"}},
                        "pipeline_class": {"value": "LTXConditionPipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                        "auto_offload": {"value": True},
                    },
                },
            },
        }

        updated = WebServer._apply_resource_retry_plan_to_graph(
            server,
            graph,
            {
                **resource_plan_target("AceStepAudioPipeline", "text_to_audio"),
                "modelRepo": "ACE-Step/acestep-v15-xl-turbo-diffusers",
                "offloadMode": OFFLOAD_MODE_NONE,
                "deviceMap": "cuda",
                "generation": {"audioDuration": 24, "steps": 8},
            },
        )

        self.assertEqual(updated, ["audio-recipe", "audio-loader"])
        self.assertEqual(graph["nodes"]["audio-recipe"]["params"]["device_map"]["value"], "cuda")
        self.assertEqual(graph["nodes"]["audio-recipe"]["params"]["offload_mode"]["value"], OFFLOAD_MODE_NONE)
        self.assertEqual(
            graph["nodes"]["audio-loader"]["params"]["model_id"]["value"]["value"],
            "ACE-Step/acestep-v15-xl-turbo-diffusers",
        )
        self.assertEqual(
            graph["nodes"]["audio-loader"]["params"]["revision"]["value"],
            catalog_revision("ACE-Step/acestep-v15-xl-turbo-diffusers"),
        )
        # Auto retry plans own runtime configuration only; creative/generation
        # controls remain exactly as the user configured them.
        self.assertEqual(graph["nodes"]["audio-generate"]["params"]["audio_duration"]["value"], 12)
        self.assertEqual(
            graph["nodes"]["video-loader"]["params"]["model_id"]["value"]["value"],
            "Lightricks/LTX-Video",
        )
        self.assertEqual(
            graph["nodes"]["video-loader"]["params"]["pipeline_class"]["value"],
            "LTXConditionPipeline",
        )

    def test_auto_repo_mutation_uses_custom_plan_revision_and_changes_identity_atomically(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        revision = "b" * 40
        graph = {
            "nodes": {
                "loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": {"source": "local", "value": "old/repo"}},
                        "revision": {"value": "a" * 40},
                        "pipeline_class": {"value": "FluxPipeline"},
                    },
                },
            },
        }

        updated = WebServer._apply_resource_retry_plan_to_graph(
            server,
            graph,
            {
                **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
                "modelRepo": "custom/new-repo",
                "artifactRevision": revision,
            },
        )

        params = graph["nodes"]["loader"]["params"]
        self.assertEqual(updated, ["loader"])
        self.assertEqual(params["model_id"]["value"], {"source": "hub", "value": "custom/new-repo"})
        self.assertEqual(params["revision"]["value"], revision)

    def test_z_image_auto_plan_updates_the_client_shaped_direct_loader(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        target_repo = "Tongyi-MAI/Z-Image-Turbo"
        graph = {
            "paths": [["z-image-loader", "independent-modular-loader"]],
            "nodes": {
                "z-image-loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": {"source": "hub", "value": "custom/old-repo"}},
                        "revision": {"value": "a" * 40},
                        "pipeline_class": {"value": "ZImagePipeline"},
                    },
                },
                "independent-modular-loader": {
                    "module": "modules.ModularDiffusers",
                    "action": "ModelsLoader",
                    "params": {
                        "repo_id": {"value": {"source": "hub", "value": "custom/modular-repo"}},
                        "revision": {"value": "c" * 40},
                        "model_type": {"value": "ZImageModularPipeline"},
                    },
                },
            },
        }

        updated = WebServer._apply_resource_retry_plan_to_graph(
            server,
            graph,
            {
                **resource_plan_target("ZImageModularPipeline", "text_to_image"),
                "modelRepo": target_repo,
            },
        )

        params = graph["nodes"]["z-image-loader"]["params"]
        self.assertEqual(updated, ["z-image-loader"])
        self.assertEqual(params["model_id"]["value"], {"source": "hub", "value": target_repo})
        self.assertEqual(params["revision"]["value"], catalog_revision(target_repo))
        self.assertEqual(
            graph["nodes"]["independent-modular-loader"]["params"]["repo_id"]["value"]["value"],
            "custom/modular-repo",
        )

    def test_auto_repo_mutation_rejects_stale_pinned_revision_without_partial_change(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        old_selection = {"source": "hub", "value": "custom/old-repo"}
        stale_revision = "a" * 40
        graph = {
            "runtimeHints": {
                "autoFieldOverrides": [
                    {"schemaVersion": 1, "nodeId": "loader", "fieldKey": "revision"},
                ],
            },
            "nodes": {
                "loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": dict(old_selection)},
                        "revision": {"value": stale_revision},
                        "pipeline_class": {"value": "QwenImagePipeline"},
                    },
                },
            },
        }

        with self.assertRaisesRegex(RuntimeError, "revision override is pinned"):
            WebServer._apply_resource_retry_plan_to_graph(
                server,
                graph,
                {
                    **resource_plan_target("QwenImageModularPipeline", "text_to_image"),
                    "modelRepo": "Qwen/Qwen-Image-2512",
                },
            )

        self.assertEqual(graph["nodes"]["loader"]["params"]["model_id"]["value"], old_selection)
        self.assertEqual(graph["nodes"]["loader"]["params"]["revision"]["value"], stale_revision)

    def test_auto_repo_mutation_error_does_not_echo_untrusted_graph_identity(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        marker = "GRAPH_SECRET_MARKER_" + "x" * 1024
        graph = {
            "runtimeHints": {
                "autoFieldOverrides": [
                    {"schemaVersion": 1, "nodeId": "loader", "fieldKey": "revision"},
                ],
            },
            "nodes": {
                "loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": {"source": "hub", "value": marker}},
                        "revision": {"value": "a" * 40},
                        "pipeline_class": {"value": "QwenImagePipeline"},
                    },
                },
            },
        }

        with self.assertRaises(RuntimeError) as raised:
            WebServer._apply_resource_retry_plan_to_graph(
                server,
                graph,
                {
                    **resource_plan_target("QwenImageModularPipeline", "text_to_image"),
                    "modelRepo": "Qwen/Qwen-Image-2512",
                },
            )

        self.assertNotIn("GRAPH_SECRET_MARKER_", str(raised.exception))
        self.assertLess(len(str(raised.exception)), 256)
        self.assertEqual(raised.exception.modiff_error_code, "auto_resource_target_mismatch")

    def test_auto_repo_mutation_preserves_a_pinned_repo_and_its_revision(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        selection = {"source": "hub", "value": "custom/pinned-repo"}
        revision = "c" * 40
        graph = {
            "runtimeHints": {
                "autoFieldOverrides": [
                    {"schemaVersion": 1, "nodeId": "loader", "fieldKey": "model_id"},
                ],
            },
            "nodes": {
                "loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": dict(selection)},
                        "revision": {"value": revision},
                        "pipeline_class": {"value": "FluxPipeline"},
                    },
                },
            },
        }

        updated = WebServer._apply_resource_retry_plan_to_graph(
            server,
            graph,
            {
                **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
                "modelRepo": "custom/new-repo",
                "artifactRevision": "d" * 40,
            },
        )

        self.assertEqual(updated, [])
        self.assertEqual(graph["nodes"]["loader"]["params"]["model_id"]["value"], selection)
        self.assertEqual(graph["nodes"]["loader"]["params"]["revision"]["value"], revision)

    def test_auto_repo_mutation_preserves_pinned_revision_when_repository_is_unchanged(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        repository = "custom/same-repo"
        pinned_revision = "c" * 40
        graph = {
            "runtimeHints": {
                "autoFieldOverrides": [
                    {"schemaVersion": 1, "nodeId": "loader", "fieldKey": "revision"},
                ],
            },
            "nodes": {
                "loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": {"source": "hub", "value": repository}},
                        "revision": {"value": pinned_revision},
                        "pipeline_class": {"value": "FluxPipeline"},
                    },
                },
            },
        }

        updated = WebServer._apply_resource_retry_plan_to_graph(
            server,
            graph,
            {
                **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
                "modelRepo": repository,
                "artifactRevision": "d" * 40,
            },
        )

        self.assertEqual(updated, [])
        self.assertEqual(graph["nodes"]["loader"]["params"]["revision"]["value"], pinned_revision)

    def test_auto_repo_mutation_fails_closed_when_loader_has_no_revision_contract(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "legacy-loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": {"source": "hub", "value": "custom/old-repo"}},
                        "pipeline_class": {"value": "QwenImagePipeline"},
                    },
                },
            },
        }

        with self.assertRaisesRegex(RuntimeError, "without a revision field"):
            WebServer._apply_resource_retry_plan_to_graph(
                server,
                graph,
                {
                    **resource_plan_target("QwenImageModularPipeline", "text_to_image"),
                    "modelRepo": "Qwen/Qwen-Image-2512",
                },
            )
        self.assertEqual(
            graph["nodes"]["legacy-loader"]["params"]["model_id"]["value"]["value"],
            "custom/old-repo",
        )

    def test_auto_repo_mutation_rejects_plan_revision_that_disagrees_with_catalog(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "model_id": {"value": {"source": "hub", "value": "custom/old-repo"}},
                        "revision": {"value": "a" * 40},
                        "pipeline_class": {"value": "QwenImagePipeline"},
                    },
                },
            },
        }

        with self.assertRaisesRegex(RuntimeError, "does not match the reviewed catalog commit"):
            WebServer._apply_resource_retry_plan_to_graph(
                server,
                graph,
                {
                    **resource_plan_target("QwenImageModularPipeline", "text_to_image"),
                    "modelRepo": "Qwen/Qwen-Image-2512",
                    "artifactRevision": "b" * 40,
                },
            )
        self.assertEqual(
            graph["nodes"]["loader"]["params"]["model_id"]["value"]["value"],
            "custom/old-repo",
        )

    def test_exact_modular_plans_update_only_the_matching_models_loader(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        cases = (
            ("QwenImageModularPipeline", "control_image"),
            ("QwenImageEditModularPipeline", "edit_image"),
            ("QwenImageEditPlusModularPipeline", "edit_image"),
            ("QwenImageLayeredModularPipeline", "layer_decomposition"),
        )
        for model_type, mode in cases:
            with self.subTest(model_type=model_type, mode=mode):
                profile = execution_profiles_for_execution(model_type, mode)[0]
                graph = {
                    "nodes": {
                        "target": {
                            "module": "modules.ModularDiffusers",
                            "action": "ModelsLoader",
                            "params": {
                                "model_type": {"value": model_type},
                                "repo_id": {"value": {"source": "hub", "value": "custom/old"}},
                                "revision": {"value": "a" * 40},
                                "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                            },
                        },
                        "other-modular": {
                            "module": "modules.ModularDiffusers",
                            "action": "ModelsLoader",
                            "params": {
                                "model_type": {"value": "ZImageModularPipeline"},
                                "repo_id": {"value": {"source": "hub", "value": "custom/other"}},
                                "revision": {"value": "c" * 40},
                                "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                            },
                        },
                        "direct-image": {
                            "module": "modules.DiffusersImage",
                            "action": "LoadPipeline",
                            "params": {
                                "pipeline_class": {"value": "QwenImagePipeline"},
                                "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                            },
                        },
                    },
                }

                updated = WebServer._apply_resource_retry_plan_to_graph(
                    server,
                    graph,
                    {
                        **resource_plan_target(model_type, mode),
                        "modelRepo": profile.default_repo,
                        "offloadMode": OFFLOAD_MODE_NONE,
                    },
                )

                self.assertEqual(updated, ["target"])
                self.assertEqual(
                    graph["nodes"]["target"]["params"]["repo_id"]["value"]["value"],
                    profile.default_repo,
                )
                self.assertEqual(
                    graph["nodes"]["target"]["params"]["revision"]["value"],
                    catalog_revision(profile.default_repo),
                )
                self.assertEqual(
                    graph["nodes"]["other-modular"]["params"]["repo_id"]["value"]["value"],
                    "custom/other",
                )
                self.assertEqual(
                    graph["nodes"]["direct-image"]["params"]["offload_mode"]["value"],
                    OFFLOAD_MODE_MODEL_CPU,
                )

    def test_same_module_action_different_pipeline_class_is_not_repurposed(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "flux": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "pipeline_class": {"value": "FluxPipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
                "qwen": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "pipeline_class": {"value": "QwenImagePipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
            },
        }

        updated = WebServer._apply_resource_retry_plan_to_graph(
            server,
            graph,
            {
                **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
                "offloadMode": OFFLOAD_MODE_GROUP_DISK,
            },
        )

        self.assertEqual(updated, ["flux"])
        self.assertEqual(
            graph["nodes"]["flux"]["params"]["offload_mode"]["value"],
            OFFLOAD_MODE_GROUP_DISK,
        )
        self.assertEqual(
            graph["nodes"]["qwen"]["params"]["offload_mode"]["value"],
            OFFLOAD_MODE_MODEL_CPU,
        )
        self.assertEqual(graph["nodes"]["qwen"]["params"]["pipeline_class"]["value"], "QwenImagePipeline")

    def test_plan_with_zero_exact_loader_identities_fails_closed(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "video": {
                    "module": "modules.DiffusersVideo",
                    "action": "LoadPipeline",
                    "params": {
                        "pipeline_class": {"value": "LTXConditionPipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
            },
        }

        with self.assertRaisesRegex(RuntimeError, "matched zero exact loader identities") as raised:
            WebServer._apply_resource_retry_plan_to_graph(
                server,
                graph,
                {
                    **resource_plan_target("AceStepAudioPipeline", "text_to_audio"),
                    "offloadMode": OFFLOAD_MODE_GROUP_DISK,
                },
            )

        self.assertEqual(raised.exception.modiff_error_code, "auto_resource_target_mismatch")
        self.assertEqual(
            graph["nodes"]["video"]["params"]["offload_mode"]["value"],
            OFFLOAD_MODE_MODEL_CPU,
        )

    def test_exact_idempotent_loader_plan_is_valid_but_target_only_plan_is_not(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "flux": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "pipeline_class": {"value": "FluxPipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_GROUP_DISK},
                        "auto_offload": {"value": True},
                    },
                },
            },
        }
        target = resource_plan_target("FluxSchnellPipeline", "text_to_image")

        self.assertEqual(
            WebServer._apply_resource_retry_plan_to_graph(
                server,
                graph,
                {**target, "offloadMode": OFFLOAD_MODE_GROUP_DISK},
            ),
            [],
        )
        with self.assertRaisesRegex(RuntimeError, "none exposes an applicable plan field"):
            WebServer._apply_resource_retry_plan_to_graph(server, graph, target)

    def test_disconnected_matching_loader_does_not_satisfy_plan_target(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "paths": [["qwen"]],
            "nodes": {
                "disconnected-flux": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "pipeline_class": {"value": "FluxPipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
                "qwen": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "pipeline_class": {"value": "QwenImagePipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
            },
        }

        with self.assertRaisesRegex(RuntimeError, "matched zero exact loader identities"):
            WebServer._apply_resource_retry_plan_to_graph(
                server,
                graph,
                {
                    **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
                    "offloadMode": OFFLOAD_MODE_GROUP_DISK,
                },
            )
        self.assertEqual(
            graph["nodes"]["disconnected-flux"]["params"]["offload_mode"]["value"],
            OFFLOAD_MODE_MODEL_CPU,
        )

    def test_disconnected_duplicate_identity_is_not_mutated_with_executable_target(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "paths": [["active-flux"]],
            "nodes": {
                "active-flux": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "pipeline_class": {"value": "FluxPipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
                "disconnected-flux": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "pipeline_class": {"value": "FluxPipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
            },
        }

        updated = WebServer._apply_resource_retry_plan_to_graph(
            server,
            graph,
            {
                **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
                "offloadMode": OFFLOAD_MODE_GROUP_DISK,
            },
        )

        self.assertEqual(updated, ["active-flux"])
        self.assertEqual(
            graph["nodes"]["disconnected-flux"]["params"]["offload_mode"]["value"],
            OFFLOAD_MODE_MODEL_CPU,
        )

    def test_wan_modular_plan_cannot_be_class_or_path_routed(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "nodes": {
                "wan": {
                    "module": "modules.ModularDiffusers",
                    "action": "ModelsLoader",
                    "params": {
                        "model_type": {"value": "WanModularPipeline"},
                        "offload_mode": {"value": OFFLOAD_MODE_MODEL_CPU},
                    },
                },
            },
        }
        plan = {
            "modelType": "WanModularPipeline",
            "mode": "text_to_video",
            "loaderModule": "modules.ModularDiffusers",
            "loaderAction": "ModelsLoader",
            "executionPath": "modular-diffusers",
            "pipelineClass": "WanModularPipeline",
            "offloadMode": OFFLOAD_MODE_GROUP_DISK,
        }

        with self.assertRaisesRegex(RuntimeError, "does not resolve to one exact execution profile"):
            WebServer._apply_resource_retry_plan_to_graph(server, graph, plan)
        self.assertEqual(
            graph["nodes"]["wan"]["params"]["offload_mode"]["value"],
            OFFLOAD_MODE_MODEL_CPU,
        )

    def test_retry_plan_sanitizer_preserves_exact_candidate_and_loader_target(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        plan = {
            **resource_plan_target("QwenImageEditPlusModularPipeline", "edit_image"),
            "candidateId": "qwen-edit-plus-retry",
            "id": "qwen-edit-plus-retry",
            "modelType": "QwenImageEditPlusModularPipeline",
            "mode": "edit_image",
            "offloadMode": OFFLOAD_MODE_GROUP_DISK,
        }

        sanitized = WebServer._sanitize_retry_plan_for_hints(server, plan)

        for key in (
            "candidateId",
            "id",
            "modelType",
            "mode",
            "loaderModule",
            "loaderAction",
            "executionPath",
            "pipelineClass",
        ):
            self.assertEqual(sanitized[key], plan[key])

    def test_auto_retry_preserves_pinned_fields_and_requires_an_unpinned_change(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        graph = {
            "runtimeHints": {
                "autoFieldOverrides": [
                    {
                        "schemaVersion": 1,
                        "nodeId": "loader",
                        "fieldKey": "dtype",
                        "value": "float16",
                    },
                    {
                        "schemaVersion": 1,
                        "nodeId": "loader",
                        "fieldKey": "offload_mode",
                        "value": OFFLOAD_MODE_NONE,
                    },
                ],
            },
            "nodes": {
                "loader": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "dtype": {"value": "float16"},
                        "offload_mode": {"value": OFFLOAD_MODE_NONE},
                        "pipeline_class": {"value": "FluxPipeline"},
                    },
                },
            },
        }
        plan = {
            **resource_plan_target("FluxSchnellPipeline", "text_to_image"),
            "dtype": "bfloat16",
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "onCategories": ["oom"],
        }

        updated = WebServer._apply_resource_retry_plan_to_graph(server, graph, plan)
        retry_index, skipped = WebServer._next_applicable_retry_plan_index(
            server,
            graph,
            [plan],
            -1,
            {"category": "oom", "error_code": "cuda_oom"},
        )

        self.assertEqual(updated, [])
        self.assertIsNone(retry_index)
        self.assertEqual(skipped, [0])
        self.assertEqual(graph["nodes"]["loader"]["params"]["dtype"]["value"], "float16")
        self.assertEqual(graph["nodes"]["loader"]["params"]["offload_mode"]["value"], OFFLOAD_MODE_NONE)

    def test_execute_graph_retries_oom_with_next_offload_mode_and_diagnostics(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        server.current_task = {"task_id": "task-1"}
        server.interrupt_flag = False
        messages = []
        server.queue_message = messages.append
        server._apply_cuda_runtime_budget = lambda runtime_hints: {}
        deterministic_calls = []

        def apply_deterministic(_graph):
            deterministic_calls.append(len(deterministic_calls) + 1)
            return {"enabled": True, "seed": 17, "application": deterministic_calls[-1]}

        server._apply_deterministic_mode = apply_deterministic
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
            "deterministicMode": {"enabled": True, "strict": False, "seed": 17},
            "runtimeHints": {
                **resource_plan_target("ZImageModularPipeline", "text_to_image"),
                "source": "studio",
                "resourceMode": "expert",
                "device": "cuda:0",
                "offloadMode": OFFLOAD_MODE_GROUP_CPU,
                "resourceRetryModes": [OFFLOAD_MODE_GROUP_DISK],
            },
            "nodes": {
                "loader-node": {
                    "module": "modules.DiffusersImage",
                    "action": "LoadPipeline",
                    "params": {
                        "offload_mode": {"value": OFFLOAD_MODE_GROUP_CPU},
                        "auto_offload": {"value": True},
                        "pipeline_class": {"value": "ZImagePipeline"},
                    },
                },
            },
        }

        with patch(
            "modiff.server.graph_optional_runtime_requirement",
            return_value={
                "schemaVersion": 1,
                "delivery": "optional_overlay",
                "requiredNow": True,
                "profileIds": ["huggingface-transformers-peft-5.14.1-0.20.0"],
                "executionProfileIds": ["z-image:auto"],
                "state": "active",
                "reason": "optional_runtime_active",
            },
        ):
            result = WebServer.execute_graph(server, graph)
        self.assertIsNone(result)
        self.assertEqual(calls["count"], 2)
        self.assertEqual(deterministic_calls, [1, 2])
        completed = next(message for message in messages if message.get("type") == "graph_completed")
        self.assertEqual(completed["deterministicMode"]["seed"], 17)
        self.assertEqual(completed["deterministicMode"]["application"], 2)
        self.assertEqual(graph["nodes"]["loader-node"]["params"]["offload_mode"]["value"], OFFLOAD_MODE_GROUP_DISK)

    def test_auto_retry_modes_come_from_exact_profile_and_expert_modes_remain_explicit(self):
        from modiff.server import WebServer

        server = object.__new__(WebServer)
        auto_hints = {
            **auto_resource_plan_target("QwenImageModularPipeline", "text_to_image"),
            "resourceMode": "auto",
            "offloadMode": OFFLOAD_MODE_MODEL_CPU,
            "resourceRetryModes": [OFFLOAD_MODE_GROUP_CPU],
        }

        self.assertEqual(
            WebServer._resource_retry_modes(server, auto_hints),
            [OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        )
        self.assertEqual(
            [plan["offloadMode"] for plan in WebServer._coerce_retry_plan_list(server, auto_hints)],
            [OFFLOAD_MODE_SEQUENTIAL_CPU, OFFLOAD_MODE_GROUP_DISK],
        )
        self.assertEqual(
            WebServer._resource_retry_modes(
                server,
                {
                    **auto_resource_plan_target("ZImageModularPipeline", "text_to_image"),
                    "resourceMode": "auto",
                    "offloadMode": OFFLOAD_MODE_GROUP_CPU,
                },
            ),
            [OFFLOAD_MODE_GROUP_DISK],
        )
        self.assertEqual(
            WebServer._resource_retry_modes(
                server,
                {
                    "resourceMode": "expert",
                    "offloadMode": OFFLOAD_MODE_MODEL_CPU,
                    "resourceRetryModes": [OFFLOAD_MODE_GROUP_CPU],
                },
            ),
            [OFFLOAD_MODE_GROUP_CPU],
        )


if __name__ == "__main__":
    unittest.main()
