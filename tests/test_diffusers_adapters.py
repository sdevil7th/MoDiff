import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from modiff.auxiliary_lora import build_lora_descriptor
from modules.DiffusersAdapters.main import (
    LoRAComparisonJobs,
    LoRAFuseUnfuse,
    LoRAHotswap,
    LoRAInspectValidate,
    LoRAMergeArtifact,
    LoRAUnloadReset,
    apply_lora_mix,
)


class FakePipeline:
    def __init__(self):
        self.loaded = []
        self.active = None
        self.adapters = {"transformer": ["existing"]}
        self.fused = []
        self.deleted = []
        self.unloaded = 0
        self.transformer = FakeMergeComponent()

    def get_list_adapters(self):
        return self.adapters

    def load_lora_weights(self, path, **kwargs):
        self.loaded.append((path, kwargs))
        name = kwargs["adapter_name"]
        if name not in self.adapters["transformer"]:
            self.adapters["transformer"].append(name)

    def set_adapters(self, names, weights):
        self.active = (names, weights)

    def fuse_lora(self, **kwargs):
        self.fused.append(kwargs)

    def unfuse_lora(self, **kwargs):
        self.fused.append({"unfuse": kwargs})

    def delete_adapters(self, name):
        self.deleted.append(name)

    def unload_lora_weights(self):
        self.unloaded += 1


class FakeMergeComponent:
    def __init__(self):
        self.merges = []
        self.saves = []

    def add_weighted_adapter(self, names, weights, merged_name, **kwargs):
        self.merges.append((names, weights, merged_name, kwargs))

    def save_pretrained(self, directory, **kwargs):
        self.saves.append((directory, kwargs))


def adapter(directory, name, scale=1.0, *, scheduler_class="", scheduler_config=None):
    path = Path(directory) / f"{name}.safetensors"
    if not path.exists():
        save_file({"lora.weight": np.asarray([1.0], dtype=np.float32)}, str(path))
    return build_lora_descriptor(
        selection={"source": "local", "value": str(path)},
        weight_name=path.name,
        revision="",
        expected_sha256="",
        adapter_name=name,
        scale=scale,
        scheduler_class=scheduler_class,
        scheduler_config=scheduler_config or {},
    )


def resign_descriptor(descriptor):
    payload = {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    descriptor["descriptor_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return descriptor


class DiffusersAdapterTests(unittest.TestCase):
    def test_inspection_consumes_the_same_explicit_versioned_descriptor(self):
        self.assertEqual(LoRAInspectValidate.params["adapter"]["type"], "custom_lora")
        self.assertTrue(LoRAInspectValidate.params["adapter"]["required"])
        self.assertNotIn("weight_name", LoRAInspectValidate.params)
        with self.assertRaisesRegex(TypeError, "versioned descriptor"):
            LoRAInspectValidate().execute(adapter="/path/inferred/from/existence")

    def test_stack_loads_missing_adapters_and_activates_independent_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = FakePipeline()
            result = apply_lora_mix(
                pipeline,
                [adapter(directory, "existing", 0.25), adapter(directory, "style", 0.75)],
            )

        self.assertEqual([item[1]["adapter_name"] for item in pipeline.loaded], ["existing", "style"])
        self.assertEqual(pipeline.deleted, ["existing"])
        self.assertTrue(all(item[1]["use_safetensors"] for item in pipeline.loaded))
        self.assertTrue(all(item[1]["weight_name"].endswith(".safetensors") for item in pipeline.loaded))
        self.assertEqual(pipeline.active, (["existing", "style"], [0.25, 0.75]))
        self.assertEqual(result["adapter_names"], ["existing", "style"])

    def test_hotswap_requires_an_existing_slot_and_uses_in_place_api(self):
        with tempfile.TemporaryDirectory() as directory:
            pipeline = FakePipeline()
            LoRAHotswap().execute(
                pipeline=pipeline,
                replacement=adapter(directory, "replacement", 0.6),
                slot_name="existing",
            )

        self.assertTrue(pipeline.loaded[0][1]["hotswap"])
        self.assertTrue(pipeline.loaded[0][1]["use_safetensors"])
        self.assertEqual(pipeline.loaded[0][1]["weight_name"], "replacement.safetensors")
        self.assertEqual(pipeline.loaded[0][1]["adapter_name"], "existing")
        self.assertEqual(pipeline.active, (["existing"], [0.6]))

    def test_stack_and_hotswap_reject_partial_or_tampered_descriptors_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = adapter(directory, "style")
            tampered = {**valid, "scale": 0.25}
            legacy = {
                "lora_path": directory,
                "weight_name": "style.safetensors",
                "adapter_name": "style",
            }
            for operation, value in (
                ("stack-legacy", legacy),
                ("stack-tampered", tampered),
                ("hotswap-legacy", legacy),
                ("hotswap-tampered", tampered),
            ):
                pipeline = FakePipeline()
                with self.subTest(operation=operation):
                    with self.assertRaises((TypeError, ValueError)):
                        if operation.startswith("stack"):
                            apply_lora_mix(pipeline, value)
                        else:
                            LoRAHotswap().execute(
                                pipeline=pipeline,
                                replacement=value,
                                slot_name="existing",
                            )
                    self.assertEqual(pipeline.loaded, [])
                    self.assertEqual(pipeline.deleted, [])
                    self.assertIsNone(pipeline.active)

    def test_stack_revalidates_the_whole_list_before_deleting_an_existing_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            first = adapter(directory, "existing")
            second = adapter(directory, "style")
            (Path(directory) / "style.safetensors").write_bytes(b"mutated-style")
            pipeline = FakePipeline()

            with self.assertRaisesRegex(ValueError, "no longer matches"):
                apply_lora_mix(pipeline, [first, second])

        self.assertEqual(pipeline.loaded, [])
        self.assertEqual(pipeline.deleted, [])
        self.assertIsNone(pipeline.active)

    def test_stack_rejects_malformed_or_empty_safetensors_before_replacing_an_existing_slot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "existing.safetensors"
            for label in ("malformed", "empty"):
                save_file({"lora.weight": np.asarray([1.0], dtype=np.float32)}, str(path))
                descriptor = adapter(directory, "existing")
                if label == "malformed":
                    path.write_bytes(b"not-a-safetensors-file")
                    message = "valid Safetensors"
                else:
                    save_file({}, str(path))
                    message = "at least one tensor"
                descriptor["artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                resign_descriptor(descriptor)
                pipeline = FakePipeline()

                with self.subTest(label=label):
                    with self.assertRaisesRegex(ValueError, message):
                        apply_lora_mix(pipeline, descriptor)
                    self.assertEqual(pipeline.loaded, [])
                    self.assertEqual(pipeline.deleted, [])
                    self.assertIsNone(pipeline.active)

    def test_stack_and_hotswap_fail_closed_on_scheduler_bearing_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            value = adapter(
                directory,
                "lightning",
                scheduler_class="FlowMatchEulerDiscreteScheduler",
                scheduler_config={"base_shift": 1.0},
            )
            for operation in ("stack", "hotswap"):
                pipeline = FakePipeline()
                with self.subTest(operation=operation):
                    with self.assertRaisesRegex(ValueError, "scheduler-bearing"):
                        if operation == "stack":
                            apply_lora_mix(pipeline, value)
                        else:
                            LoRAHotswap().execute(
                                pipeline=pipeline,
                                replacement=value,
                                slot_name="existing",
                            )
                    self.assertEqual(pipeline.loaded, [])
                    self.assertEqual(pipeline.deleted, [])
                    self.assertIsNone(pipeline.active)

    def test_fuse_reset_and_comparison_jobs_preserve_explicit_user_choices(self):
        pipeline = FakePipeline()
        LoRAFuseUnfuse().execute(
            pipeline=pipeline,
            operation="fuse",
            adapter_names="existing,style",
            components="transformer",
            scale=0.8,
            safe_fusing=True,
        )
        LoRAUnloadReset().execute(pipeline=pipeline, adapter_names="style")
        jobs = LoRAComparisonJobs().execute(prompt="portrait", mixes='[{"existing": 1.0}, {"style": 0.7}]', seed=42)

        self.assertEqual(pipeline.fused[0]["adapter_names"], ["existing", "style"])
        self.assertEqual(pipeline.deleted, ["style"])
        self.assertEqual(jobs["jobs"][1]["mix"], {"style": 0.7})
        self.assertEqual(jobs["jobs"][1]["seed"], 42)

    def test_merge_artifact_uses_peft_method_and_writes_provenance(self):
        pipeline = FakePipeline()
        pipeline.adapters = {"transformer": ["existing", "style"]}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "merged"
            result = LoRAMergeArtifact().execute(
                pipeline=pipeline,
                component="transformer",
                adapter_names="existing,style",
                weights="1.0,0.7",
                merge_method="dare_ties",
                density=0.4,
                merged_name="final_mix",
                output_directory=str(destination),
            )

            merge = pipeline.transformer.merges[0]
            self.assertEqual(merge[0], ["existing", "style"])
            self.assertEqual(merge[3], {"combination_type": "dare_ties", "density": 0.4})
            self.assertTrue((destination / "modiff_merge_manifest.json").is_file())
            self.assertEqual(result["artifact_path"], str(destination))


if __name__ == "__main__":
    unittest.main()
