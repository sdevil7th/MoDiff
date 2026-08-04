import unittest
import tempfile
from pathlib import Path

from modules.DiffusersAdapters.main import (
    LoRAComparisonJobs,
    LoRAFuseUnfuse,
    LoRAHotswap,
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


def adapter(name, scale=1.0):
    return {"lora_path": f"/{name}", "weight_name": f"{name}.safetensors", "adapter_name": name, "scale": scale}


class DiffusersAdapterTests(unittest.TestCase):
    def test_stack_loads_missing_adapters_and_activates_independent_weights(self):
        pipeline = FakePipeline()
        result = apply_lora_mix(pipeline, [adapter("existing", 0.25), adapter("style", 0.75)])

        self.assertEqual([item[1]["adapter_name"] for item in pipeline.loaded], ["style"])
        self.assertEqual(pipeline.active, (["existing", "style"], [0.25, 0.75]))
        self.assertEqual(result["adapter_names"], ["existing", "style"])

    def test_hotswap_requires_an_existing_slot_and_uses_in_place_api(self):
        pipeline = FakePipeline()
        LoRAHotswap().execute(pipeline=pipeline, replacement=adapter("replacement", 0.6), slot_name="existing")

        self.assertTrue(pipeline.loaded[0][1]["hotswap"])
        self.assertEqual(pipeline.loaded[0][1]["adapter_name"], "existing")
        self.assertEqual(pipeline.active, (["existing"], [0.6]))

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
