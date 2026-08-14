import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

from modules import MODULE_MAP
from modules.DiffusersRuntime.main import (
    ATTENTION_BACKENDS,
    ApplyPipelineRuntimeConfig,
    LoadPrequantizedDiffusersComponent,
    PipelineQuantizationConfigV2,
    apply_attention_backend,
    build_quantization_config_v2,
    build_execution_recipe,
    build_runtime_capabilities,
    configure_channels_last,
    configure_denoiser_cache,
    configure_layerwise_casting,
    configure_regional_compile,
    configure_vae_memory,
    estimate_pipeline_memory,
    execution_recipe_summary,
    loader_runtime_options,
    plan_execution_recipes,
    release_pipeline_memory,
    summarize_safetensors_files,
)


class FakeAttentionComponent:
    def __init__(self):
        self.backends = []
        self.is_cache_enabled = False
        self.cache_configs = []
        self.compile_calls = []

    def set_attention_backend(self, backend):
        self.backends.append(backend)

    def enable_cache(self, config):
        self.cache_configs.append(config)
        self.is_cache_enabled = True

    def disable_cache(self):
        self.is_cache_enabled = False

    def compile_repeated_blocks(self, **kwargs):
        self.compile_calls.append(kwargs)


class FakeVAE:
    def __init__(self):
        self.calls = []

    def enable_slicing(self):
        self.calls.append("enable_slicing")

    def disable_slicing(self):
        self.calls.append("disable_slicing")

    def enable_tiling(self):
        self.calls.append("enable_tiling")

    def disable_tiling(self):
        self.calls.append("disable_tiling")


class FakePipeline:
    def __init__(self):
        self.transformer = FakeAttentionComponent()
        self.vae = FakeVAE()
        self.moves = []
        self.freed_hooks = 0

    def to(self, device):
        self.moves.append(device)
        return self

    def maybe_free_model_hooks(self):
        self.freed_hooks += 1


class FakePipelineQuantizationConfig:
    def __init__(self, *, quant_mapping):
        self.quant_mapping = quant_mapping


class DiffusersRuntimeTests(unittest.TestCase):
    def test_gguf_loader_pins_cataloged_artifact_and_base_config_independently(self):
        calls = {}

        class FakeComponent:
            @classmethod
            def from_single_file(cls, path, **kwargs):
                calls["component"] = (path, kwargs)
                return object()

        with tempfile.NamedTemporaryFile(suffix=".gguf") as artifact_file:
            node = LoadPrequantizedDiffusersComponent("gguf-revision-probe")
            node.progress = lambda *args, **kwargs: None
            with (
                patch("diffusers.FluxTransformer2DModel", FakeComponent),
                patch("diffusers.GGUFQuantizationConfig", return_value=object()),
                patch("huggingface_hub.hf_hub_download", return_value=artifact_file.name) as download,
            ):
                result = node.execute(
                    artifact={"source": "hub", "value": "city96/FLUX.1-schnell-gguf"},
                    filename="flux1-schnell-Q4_0.gguf",
                    component_class="FluxTransformer2DModel",
                    config_model="black-forest-labs/FLUX.1-schnell",
                )

        self.assertEqual(
            download.call_args.kwargs["revision"],
            "f495746ed9c5efcf4661f53ef05401dceadc17d2",
        )
        self.assertEqual(
            calls["component"][1]["config_revision"],
            "741f7c3ce8b383c54771c7003378a50191e9efe9",
        )
        self.assertIn("@f495746ed9c5efcf4661f53ef05401dceadc17d2:", result["resolved_artifact"])

    def test_runtime_nodes_are_registered(self):
        runtime = MODULE_MAP["modules.DiffusersRuntime"]
        self.assertIn("PipelineQuantizationConfigV2", runtime)
        self.assertIn("ApplyPipelineRuntimeConfig", runtime)
        self.assertIn("DiffusersComponentInventory", runtime)
        self.assertIn("DiffusersExecutionRecipe", runtime)
        self.assertIn("HardwareCapabilityProbe", runtime)
        self.assertIn("PipelineMemoryEstimate", runtime)
        self.assertIn("ExecutionRecipePlanner", runtime)
        self.assertIn("ReleasePipelineMemory", runtime)
        self.assertEqual(runtime["ApplyPipelineRuntimeConfig"]["category"], "Diffusers Runtime")

    def test_quantization_v2_builds_component_specific_configs_and_exclusions(self):
        calls = []

        def fake_config(backend, dtype, excluded):
            calls.append((backend, dtype, excluded))
            return {"backend": backend, "excluded": excluded}

        with (
            patch("modules.DiffusersRuntime.main.quant_config_for", side_effect=fake_config),
            patch(
                "diffusers.quantizers.PipelineQuantizationConfig",
                FakePipelineQuantizationConfig,
            ),
        ):
            config, summary = build_quantization_config_v2(
                backend="bnb_4bit",
                components=["transformer", "text_encoder"],
                dtype="bf16",
                excluded_modules="proj_out, norm_out",
                component_overrides={
                    "text_encoder": {
                        "backend": "quanto_float8",
                        "excluded_modules": ["final_layer_norm"],
                    }
                },
            )

        self.assertEqual(
            calls,
            [
                ("bnb_4bit", "bf16", ["proj_out", "norm_out"]),
                ("quanto_float8", "bf16", ["final_layer_norm"]),
            ],
        )
        self.assertEqual(set(config.quant_mapping), {"transformer", "text_encoder"})
        self.assertEqual(summary["text_encoder"]["backend"], "quanto_float8")

    def test_quantization_v2_rejects_unknown_components(self):
        with self.assertRaisesRegex(ValueError, "Unknown quantized components"):
            build_quantization_config_v2(
                backend="bnb_4bit",
                components=["transformer"],
                dtype="bf16",
                component_overrides={"mystery_encoder": "bnb_8bit"},
            )

    def test_quantization_v2_supports_dual_expert_video_transformers(self):
        with (
            patch(
                "modules.DiffusersRuntime.main.quant_config_for",
                side_effect=lambda backend, dtype, excluded: {"backend": backend, "dtype": dtype},
            ),
            patch(
                "diffusers.quantizers.PipelineQuantizationConfig",
                FakePipelineQuantizationConfig,
            ),
        ):
            config, summary = build_quantization_config_v2(
                backend="torchao_int8_weight_only",
                components=["transformer", "transformer_2"],
                dtype="bf16",
            )

        self.assertEqual(set(config.quant_mapping), {"transformer", "transformer_2"})
        self.assertEqual(summary["transformer_2"]["backend"], "torchao_int8_weight_only")

    def test_component_inventory_summarizes_weight_floors_without_loading_tensors(self):
        summary = summarize_safetensors_files(
            [
                {
                    "name": "transformer/model-00001.safetensors",
                    "source_bytes": 120,
                    "parameter_count": {"BF16": 40, "F32": 5},
                },
                {
                    "name": "transformer/model-00002.safetensors",
                    "source_bytes": 80,
                    "parameter_count": {"BF16": 20},
                },
                {
                    "name": "vae/model.safetensors",
                    "source_bytes": 30,
                    "parameter_count": {"F32": 3},
                },
            ]
        )

        self.assertEqual(summary["total_source_bytes"], 230)
        self.assertEqual(summary["total_parameter_count"], 68)
        self.assertEqual(summary["total_weight_bytes"], 152)
        self.assertEqual(summary["largest_shard"]["name"], "transformer/model-00001.safetensors")
        transformer = next(item for item in summary["components"] if item["component"] == "transformer")
        self.assertEqual(transformer["parameter_count"], 65)
        self.assertEqual(transformer["weight_bytes"], 140)

    def test_attention_backend_applies_only_to_compatible_components(self):
        pipeline = FakePipeline()
        result = apply_attention_backend(pipeline, "sage")

        self.assertEqual(pipeline.transformer.backends, ["sage"])
        self.assertEqual(result["applied"], ["transformer"])

    def test_obsolete_and_mutable_hub_aiter_backends_fail_closed(self):
        self.assertNotIn("aiter", ATTENTION_BACKENDS)
        self.assertNotIn("aiter_fa2_hub", ATTENTION_BACKENDS)
        for backend in ("aiter", "aiter_fa2_hub"):
            with self.subTest(backend=backend):
                pipeline = FakePipeline()
                with self.assertRaisesRegex(ValueError, "Unsupported attention backend"):
                    apply_attention_backend(pipeline, backend)
                with self.assertRaisesRegex(ValueError, "Unsupported attention backend"):
                    build_execution_recipe(attention_backend=backend)
                self.assertEqual(pipeline.transformer.backends, [])

    def test_attention_auto_preserves_diffusers_default(self):
        pipeline = FakePipeline()
        result = apply_attention_backend(pipeline, "auto")

        self.assertEqual(pipeline.transformer.backends, [])
        self.assertTrue(result["default_selection"])

    def test_attention_backend_configures_both_dual_expert_transformers(self):
        pipeline = FakePipeline()
        pipeline.transformer_2 = FakeAttentionComponent()

        result = apply_attention_backend(pipeline, "native")

        self.assertEqual(pipeline.transformer.backends, ["native"])
        self.assertEqual(pipeline.transformer_2.backends, ["native"])
        self.assertEqual(result["applied"], ["transformer", "transformer_2"])

    def test_execution_recipe_rejects_runtime_quantization_with_split_offload(self):
        quant = FakePipelineQuantizationConfig(quant_mapping={"transformer": object()})
        with self.assertRaisesRegex(ValueError, "Create optimized copy"):
            build_execution_recipe(
                quantization_config=quant,
                device_map="manual",
                device_map_overrides='{"transformer": 0, "text_encoder_2": "cpu"}',
                max_memory='{"0": "16GiB", "cpu": "48GiB"}',
                offload_mode="group_cpu",
                device="cuda:0",
            )

        recipe = build_execution_recipe(
            device_map="manual",
            device_map_overrides='{"transformer": 0, "text_encoder_2": "cpu"}',
            max_memory='{"0": "16GiB", "cpu": "48GiB"}',
            offload_mode="group_cpu",
            device="cuda:0",
            attention_backend="native",
            attention_components="transformer, controlnet",
            vae_slicing=True,
            vae_tiling=False,
        )

        self.assertEqual(recipe["max_memory"], {0: "16GiB", "cpu": "48GiB"})
        self.assertEqual(recipe["device_map"], {"transformer": 0, "text_encoder_2": "cpu"})
        self.assertEqual(recipe["attention_components"], ["transformer", "controlnet"])
        summary = execution_recipe_summary(recipe)
        self.assertNotIn("quantization_config", summary)
        self.assertEqual(summary["quantized_components"], [])

    def test_manual_device_map_requires_component_placements(self):
        with self.assertRaisesRegex(ValueError, "needs at least one component placement entry"):
            build_execution_recipe(device_map="manual", device_map_overrides="{}")

    def test_loader_runtime_options_preserve_legacy_controls_and_connected_recipe_override(self):
        legacy_recipe, legacy_device, legacy_offload, legacy_load = loader_runtime_options(
            {"device": "cpu", "auto_offload": False, "offload_mode": "group_cpu"},
            default_device="cuda:0",
            default_offload_mode="model_cpu",
        )
        self.assertEqual(legacy_recipe, {})
        self.assertEqual(legacy_device, "cpu")
        self.assertEqual(legacy_offload, "none")
        self.assertEqual(legacy_load, {})

        quant = object()
        recipe = {
            "device": "cuda:1",
            "offload_mode": "group_cpu",
            "quantization_config": quant,
            "device_map": "balanced",
            "max_memory": {0: "20GiB", "cpu": "64GiB"},
        }
        with self.assertRaisesRegex(ValueError, "full GPU residency"):
            loader_runtime_options(
                {"execution_recipe": recipe, "auto_offload": False},
                default_device="cuda:0",
                default_offload_mode="model_cpu",
            )

    def test_complete_pipeline_no_offload_streams_directly_to_target_accelerator(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_ENABLE_PARALLEL_LOADING", None)
            recipe, device, offload, load = loader_runtime_options(
                {
                    "execution_recipe": {
                        "device": "cuda:0",
                        "offload_mode": "none",
                        "device_map": "none",
                    }
                },
                default_device="cuda:0",
                default_offload_mode="model_cpu",
                direct_device_load=True,
            )
            self.assertEqual(os.environ["HF_ENABLE_PARALLEL_LOADING"], "YES")

        self.assertEqual(recipe["device_map"], "none")
        self.assertEqual(device, "cuda:0")
        self.assertEqual(offload, "none")
        self.assertEqual(load["device_map"], "cuda")

    def test_direct_device_loading_respects_explicit_parallel_loading_opt_out(self):
        with patch.dict(os.environ, {"HF_ENABLE_PARALLEL_LOADING": "false"}):
            _, _, offload, load = loader_runtime_options(
                {
                    "execution_recipe": {
                        "device": "cuda:0",
                        "offload_mode": "none",
                        "device_map": "cuda",
                    }
                },
                default_device="cuda:0",
                default_offload_mode="model_cpu",
                direct_device_load=True,
            )
            self.assertEqual(os.environ["HF_ENABLE_PARALLEL_LOADING"], "false")
        self.assertEqual(offload, "none")
        self.assertEqual(load["device_map"], "cuda")

    def test_direct_device_loading_never_overrides_offload_or_explicit_map(self):
        _, _, offload, load = loader_runtime_options(
            {
                "execution_recipe": {
                    "device": "cuda:0",
                    "offload_mode": "model_cpu",
                    "device_map": "none",
                }
            },
            default_device="cuda:0",
            default_offload_mode="model_cpu",
            direct_device_load=True,
        )
        self.assertEqual(offload, "model_cpu")
        self.assertNotIn("device_map", load)

        _, _, offload, load = loader_runtime_options(
            {
                "execution_recipe": {
                    "device": "cuda:1",
                    "offload_mode": "none",
                    "device_map": "balanced",
                }
            },
            default_device="cuda:0",
            default_offload_mode="model_cpu",
            direct_device_load=True,
        )
        self.assertEqual(offload, "none")
        self.assertEqual(load["device_map"], "balanced")

        _, _, offload, load = loader_runtime_options(
            {
                "execution_recipe": {
                    "device": "cuda:1",
                    "offload_mode": "none",
                    "device_map": "none",
                }
            },
            default_device="cuda:0",
            default_offload_mode="model_cpu",
            direct_device_load=True,
        )
        self.assertEqual(offload, "none")
        self.assertNotIn("device_map", load)

    def test_capability_probe_distinguishes_rocm_attention_from_nvidia_only_backends(self):
        class FakeCuda:
            @staticmethod
            def get_device_capability(_index):
                return (9, 0)

            @staticmethod
            def is_bf16_supported():
                return True

        fake_torch = types.SimpleNamespace(
            __version__="test-rocm",
            version=types.SimpleNamespace(hip="7.2", cuda=None),
            cuda=FakeCuda(),
            compile=lambda fn: fn,
        )
        hardware = {"devices": [{"type": "cuda", "device": "cuda:0"}]}
        available = {"aiter", "torchao", "bitsandbytes"}
        result = build_runtime_capabilities(
            hardware,
            torch_module=fake_torch,
            package_available=lambda name: name in available,
        )

        self.assertEqual(result["vendor"], "amd")
        self.assertNotIn("aiter", result["attention_backends"])
        self.assertNotIn("aiter_fa2_hub", result["attention_backends"])
        self.assertFalse(result["attention_backends"]["flash"]["available"])
        self.assertFalse(result["quantization_backends"]["torchao_float8"]["available"])

    def test_capability_probe_excludes_obsolete_rocm_attention_backend(self):
        class FakeCuda:
            @staticmethod
            def get_device_capability(_index):
                return (9, 0)

            @staticmethod
            def is_bf16_supported():
                return True

        fake_torch = types.SimpleNamespace(
            __version__="test-rocm",
            version=types.SimpleNamespace(hip="7.2", cuda=None),
            cuda=FakeCuda(),
            compile=lambda fn: fn,
        )
        result = build_runtime_capabilities(
            {"devices": [{"type": "cuda", "device": "cuda:0"}]},
            torch_module=fake_torch,
            package_available=lambda _name: False,
        )

        self.assertNotIn("aiter", result["attention_backends"])
        self.assertNotIn("aiter_fa2_hub", result["attention_backends"])
        self.assertEqual(
            result["attention_backends"]["sage"]["reason"],
            "SageAttention package is not installed",
        )

    def test_capability_probe_treats_missing_parent_package_as_unavailable(self):
        fake_torch = types.SimpleNamespace(
            __version__="test-cpu",
            version=types.SimpleNamespace(hip=None, cuda=None),
            compile=lambda fn: fn,
        )

        def missing_parent(name):
            if name == "optimum.quanto":
                raise ModuleNotFoundError("No module named 'optimum'")
            return None

        with patch("modules.DiffusersRuntime.main.importlib.util.find_spec", side_effect=missing_parent):
            result = build_runtime_capabilities(
                {"devices": [{"type": "cpu", "device": "cpu:0"}]},
                torch_module=fake_torch,
            )

        self.assertFalse(result["quantization_backends"]["quanto_float8"]["available"])

    def test_capability_probe_requires_nvidia_89_for_torchao_fp8(self):
        class FakeCuda:
            @staticmethod
            def get_device_capability(_index):
                return (8, 9)

            @staticmethod
            def is_bf16_supported():
                return True

        fake_torch = types.SimpleNamespace(
            __version__="test-cuda",
            version=types.SimpleNamespace(hip=None, cuda="12.8"),
            cuda=FakeCuda(),
            compile=lambda fn: fn,
        )
        result = build_runtime_capabilities(
            {"devices": [{"type": "cuda", "device": "cuda:0"}]},
            torch_module=fake_torch,
            package_available=lambda name: name == "torchao",
        )

        self.assertEqual(result["vendor"], "nvidia")
        self.assertTrue(result["quantization_backends"]["torchao_float8"]["available"])
        self.assertTrue(result["compile"]["available"])

    def test_capability_probe_reports_xpu_without_claiming_cuda_kernels(self):
        fake_torch = types.SimpleNamespace(
            __version__="test-xpu",
            version=types.SimpleNamespace(hip=None, cuda=None),
            xpu=types.SimpleNamespace(is_bf16_supported=lambda: True),
            compile=lambda fn: fn,
        )
        result = build_runtime_capabilities(
            {"devices": [{"type": "xpu", "device": "xpu:0"}]},
            torch_module=fake_torch,
            package_available=lambda _name: True,
        )

        self.assertEqual(result["vendor"], "intel")
        self.assertTrue(result["dtypes"]["bfloat16"])
        self.assertFalse(result["attention_backends"]["_native_flash"]["available"])
        self.assertFalse(result["quantization_backends"]["bnb_4bit"]["available"])

    def test_memory_estimate_reports_floors_without_fabricating_peak_vram(self):
        inventory = {
            "total_weight_bytes": 260,
            "components": [
                {
                    "component": "transformer",
                    "parameter_count": 100,
                    "weight_bytes": 200,
                    "dtype_counts": {"BF16": 100},
                },
                {
                    "component": "vae",
                    "parameter_count": 30,
                    "weight_bytes": 60,
                    "dtype_counts": {"BF16": 30},
                },
            ],
        }
        result = estimate_pipeline_memory(
            inventory,
            width=128,
            height=64,
            frames=9,
            batch_size=2,
            latent_channels=4,
            spatial_compression=8,
            temporal_compression=4,
            dtype_bytes=2,
            quantization_summary={"transformer": {"backend": "bnb_4bit"}},
            offload_mode="group_cpu",
        )

        self.assertEqual(result["weights"]["unquantized_weight_floor_bytes"], 260)
        self.assertEqual(result["weights"]["idealized_weight_floor_bytes"], 110)
        self.assertEqual(result["weights"]["accelerator_resident_weight_proxy_bytes"], 60)
        self.assertEqual(result["shape"]["latent_tensor_bytes"], 6144)
        self.assertIsNone(result["peak_accelerator_memory_bytes"])
        self.assertIn("unavailable", result["confidence"]["peak_accelerator_memory"])

    def test_recipe_planner_ranks_preference_without_claiming_runtime_proof(self):
        capabilities = {
            "backend": "cuda",
            "device": "cuda:0",
            "dtypes": {"float16": True, "bfloat16": True},
            "memory": {"accelerator_free_bytes": 20_000},
            "quantization_backends": {
                "bnb_8bit": {"available": True},
                "bnb_4bit": {"available": True},
            },
        }
        inventory = {
            "total_weight_bytes": 10_000,
            "components": [
                {"component": "transformer", "weight_bytes": 8_000},
                {"component": "vae", "weight_bytes": 2_000},
            ],
        }
        plan = plan_execution_recipes(capabilities, inventory, preference="balanced")

        selected = plan["candidates"][0]
        self.assertEqual(selected["profile"], "balanced")
        self.assertEqual(selected["quantization_backend"], "none")
        self.assertEqual(selected["proof_status"], "required")
        self.assertFalse(selected["regional_compile"])
        self.assertEqual(selected["denoiser_cache"], "none")

    def test_recipe_planner_does_not_invent_quantizable_components(self):
        plan = plan_execution_recipes(
            {
                "backend": "cpu",
                "device": "cpu:0",
                "dtypes": {"float32": True},
                "memory": {},
                "quantization_backends": {"quanto_float8": {"available": True}},
            },
            {
                "total_weight_bytes": 100,
                "components": [{"component": "root", "weight_bytes": 100}],
            },
            preference="low_memory",
        )

        self.assertEqual(plan["candidates"][0]["quantization_backend"], "none")

    def test_vae_memory_configuration_can_enable_and_disable_features(self):
        pipeline = FakePipeline()
        enabled = configure_vae_memory(pipeline, slicing=True, tiling=True)
        disabled = configure_vae_memory(pipeline, slicing=False, tiling=False)

        self.assertEqual(
            pipeline.vae.calls,
            ["enable_slicing", "enable_tiling", "disable_slicing", "disable_tiling"],
        )
        self.assertEqual(len(enabled["applied"]), 2)
        self.assertEqual(len(disabled["applied"]), 2)

    def test_first_block_cache_is_explicit_and_reversible(self):
        pipeline = FakePipeline()
        enabled = configure_denoiser_cache(pipeline, strategy="first_block", threshold=0.08)
        disabled = configure_denoiser_cache(pipeline, strategy="none")

        self.assertEqual(enabled["applied"], ["transformer"])
        self.assertAlmostEqual(pipeline.transformer.cache_configs[0].threshold, 0.08)
        self.assertEqual(disabled["disabled"], ["transformer"])
        self.assertFalse(pipeline.transformer.is_cache_enabled)

    def test_magcache_options_build_current_diffusers_config(self):
        pipeline = FakePipeline()
        result = configure_denoiser_cache(
            pipeline,
            strategy="magcache",
            options={
                "threshold": 0.07,
                "max_skip_steps": 2,
                "retention_ratio": 0.25,
                "num_inference_steps": 20,
                "calibrate": True,
            },
        )

        config = pipeline.transformer.cache_configs[0]
        self.assertEqual(result["config"], "MagCacheConfig")
        self.assertAlmostEqual(config.threshold, 0.07)
        self.assertEqual(config.max_skip_steps, 2)
        self.assertEqual(config.num_inference_steps, 20)

    def test_magcache_requires_calibration_or_model_specific_ratios(self):
        with self.assertRaisesRegex(ValueError, "calibrate=true"):
            configure_denoiser_cache(FakePipeline(), strategy="magcache")

    def test_regional_compile_uses_diffusers_repeated_block_api(self):
        pipeline = FakePipeline()
        result = configure_regional_compile(
            pipeline,
            enabled=True,
            components="transformer",
            backend="inductor",
            mode="reduce-overhead",
            fullgraph=False,
            dynamic=True,
        )

        self.assertEqual(result["applied"], ["transformer"])
        self.assertEqual(
            pipeline.transformer.compile_calls,
            [
                {
                    "backend": "inductor",
                    "mode": "reduce-overhead",
                    "fullgraph": False,
                    "dynamic": True,
                }
            ],
        )

    def test_layerwise_casting_is_explicit_and_idempotent(self):
        pipeline = FakePipeline()
        with patch("diffusers.hooks.apply_layerwise_casting") as apply:
            first = configure_layerwise_casting(
                pipeline,
                enabled=True,
                components="transformer",
                storage_dtype="float8_e4m3fn",
                compute_dtype="bfloat16",
            )
            second = configure_layerwise_casting(
                pipeline,
                enabled=True,
                components="transformer",
                storage_dtype="float8_e4m3fn",
                compute_dtype="bfloat16",
            )

        self.assertEqual(first["applied"], ["transformer"])
        self.assertEqual(second["alreadyApplied"], ["transformer"])
        apply.assert_called_once()

    def test_channels_last_only_mutates_selected_components_once(self):
        class FakeConvolutionalComponent:
            def __init__(self):
                self.memory_formats = []

            def to(self, **kwargs):
                self.memory_formats.append(kwargs["memory_format"])
                return self

        pipeline = types.SimpleNamespace(unet=FakeConvolutionalComponent())
        first = configure_channels_last(pipeline, enabled=True, components="unet")
        second = configure_channels_last(pipeline, enabled=True, components="unet")

        self.assertEqual(first["applied"], ["unet"])
        self.assertEqual(second["alreadyApplied"], ["unet"])
        self.assertEqual(len(pipeline.unet.memory_formats), 1)

    def test_execution_recipe_preserves_opt_in_layout_and_casting(self):
        recipe = build_execution_recipe(
            layerwise_casting=True,
            layerwise_casting_components="transformer",
            layerwise_storage_dtype="float8_e4m3fn",
            layerwise_compute_dtype="bfloat16",
            channels_last=True,
            channels_last_components="unet,vae",
        )

        self.assertTrue(recipe["layerwise_casting"])
        self.assertEqual(recipe["layerwise_casting_components"], ["transformer"])
        self.assertTrue(recipe["channels_last"])
        self.assertEqual(recipe["channels_last_components"], ["unet", "vae"])

    def test_pipeline_release_disables_cache_and_moves_weights_to_cpu(self):
        pipeline = FakePipeline()
        pipeline.transformer.is_cache_enabled = True
        with patch("torch.cuda.is_available", return_value=False):
            report = release_pipeline_memory(pipeline, move_to_cpu=True, disable_cache=True)

        self.assertFalse(pipeline.transformer.is_cache_enabled)
        self.assertEqual(pipeline.freed_hooks, 1)
        self.assertEqual(pipeline.moves, ["cpu"])
        self.assertIn("pipeline_to_cpu", report["released"])

    def test_runtime_node_returns_same_pipeline_and_user_readable_summary(self):
        pipeline = FakePipeline()
        result = ApplyPipelineRuntimeConfig("runtime-test").execute(
            pipeline=pipeline,
            attention_backend="native",
            vae_slicing=True,
            vae_tiling=False,
        )

        self.assertIs(result["configured_pipeline"], pipeline)
        summary = json.loads(result["summary"])
        self.assertEqual(summary["attention"]["applied"], ["transformer"])
        self.assertIn("vae", summary)

    def test_quantization_node_returns_pipeline_config(self):
        with (
            patch("modules.DiffusersRuntime.main.str_to_dtype", return_value="bf16"),
            patch("modules.DiffusersRuntime.main.quant_config_for", return_value={"config": True}),
            patch(
                "diffusers.quantizers.PipelineQuantizationConfig",
                FakePipelineQuantizationConfig,
            ),
        ):
            result = PipelineQuantizationConfigV2("quant-test").execute(
                backend="bnb_4bit",
                components=["transformer"],
                component_overrides="{}",
            )

        self.assertEqual(set(result["quantization_config"].quant_mapping), {"transformer"})
        self.assertEqual(json.loads(result["summary"])["transformer"]["backend"], "bnb_4bit")

    def test_disabled_quantization_node_keeps_connected_flow_executable(self):
        result = PipelineQuantizationConfigV2("quant-disabled").execute(
            backend="none",
            components=["transformer"],
            component_overrides="{}",
        )

        self.assertEqual(result["quantization_config"]["backend"], "none")
        self.assertTrue(result["quantization_config"]["disabled"])
        recipe = build_execution_recipe(quantization_config=result["quantization_config"])
        self.assertIsNone(recipe["quantization_config"])


if __name__ == "__main__":
    unittest.main()
