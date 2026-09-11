import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modiff.auto_resource import auto_resource_history_key
from modiff.controlled_artifacts import (
    controlled_artifact_receipts_from_graph,
    resolve_upscaler_artifact,
)
from utils.huggingface import CONFIG


UPSCALER_REVISION = "42efb9c3eeed1f5c0c8a626cf5f7f4481dfbb094"
ACE_REVISION = "200ba991ae448051e14b0183157e35c2d27c9fb0"
LTX_REVISION = "7c64400e1861cc0d7b98d570a1926d5408ec60cd"
SD15_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
SD15_CONTROLNET_REVISION = "115a470d547982438f70198e353a921996e2e819"


def _node(module, action, **params):
    return {
        "module": module,
        "action": action,
        "params": {key: {"value": value} for key, value in params.items()},
    }


class ControlledArtifactReceiptTests(unittest.TestCase):
    def test_conditioned_image_pipeline_receipt_binds_the_auxiliary_component(self):
        graph = {
            "nodes": {
                "pipeline": _node(
                    "modules.DiffusersImage",
                    "LoadPipeline",
                    model_id={"source": "hub", "value": "stable-diffusion-v1-5/stable-diffusion-v1-5"},
                    revision=SD15_REVISION,
                    pipeline_class="StableDiffusionControlNetPipeline",
                    mode="control_image",
                    conditioning_kind="controlnet",
                    conditioning_model_id={
                        "source": "hub",
                        "value": "lllyasviel/control_v11p_sd15_canny",
                    },
                    conditioning_revision=SD15_CONTROLNET_REVISION,
                ),
                "preview": _node("modules.Image", "Preview"),
            },
            "paths": [["pipeline", "preview"]],
        }

        receipts = controlled_artifact_receipts_from_graph(
            graph,
            primary_candidate={
                "loaderModule": "modules.DiffusersImage",
                "loaderAction": "LoadPipeline",
                "pipelineClass": "StableDiffusionControlNetPipeline",
            },
        )

        self.assertEqual(len(receipts), 1)
        receipt = receipts[0]
        self.assertEqual(receipt["kind"], "diffusers_conditioning_component")
        self.assertEqual(receipt["artifact"]["revision"], SD15_CONTROLNET_REVISION)
        self.assertEqual(receipt["componentClass"], "ControlNetModel")
        self.assertEqual(receipt["componentParameter"], "controlnet")
        self.assertTrue(receipt["safeSerializationRequired"])

    def test_executable_upscaler_is_rehashed_and_disconnected_copy_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_root = Path(directory)
            weight = (
                cache_root
                / "models--nateraw--real-esrgan"
                / "snapshots"
                / UPSCALER_REVISION
                / "RealESRGAN_x2plus.pth"
            )
            weight.parent.mkdir(parents=True)
            weight.write_bytes(b"reviewed-upscaler")
            digest = hashlib.sha256(weight.read_bytes()).hexdigest()
            selection = {
                "source": "hub",
                "value": "nateraw/real-esrgan/RealESRGAN_x2plus.pth",
                "revision": UPSCALER_REVISION,
                "sha256": digest,
                "byteSize": weight.stat().st_size,
            }
            graph = {
                "nodes": {
                    "pipeline": _node(
                        "modules.DiffusersVideo",
                        "LoadPipeline",
                        model_id={"source": "hub", "value": "Lightricks/LTX-Video-0.9.8-13B-distilled"},
                        revision=LTX_REVISION,
                        pipeline_class="LTXConditionPipeline",
                    ),
                    "upscale": _node("modules.Spandrel", "Upscaler", model_id=selection),
                    "disconnected": _node("modules.Spandrel", "Upscaler", model_id=selection),
                    "export": _node("modules.Video", "Export", fps=16),
                },
                "paths": [["pipeline", "upscale", "export"]],
            }
            primary = {
                "loaderModule": "modules.DiffusersVideo",
                "loaderAction": "LoadPipeline",
                "pipelineClass": "LTXConditionPipeline",
            }
            with patch.dict(CONFIG.hf, {"cache_dir": str(cache_root)}):
                receipts = controlled_artifact_receipts_from_graph(graph, primary_candidate=primary)

        self.assertEqual(len(receipts), 1)
        receipt = receipts[0]
        self.assertEqual(receipt["kind"], "spandrel_upscaler")
        self.assertEqual(receipt["artifact"]["revision"], UPSCALER_REVISION)
        self.assertEqual(receipt["artifact"]["sha256"], digest)
        self.assertNotIn(str(cache_root), json.dumps(receipt))

    def test_local_upscaler_receipt_never_publishes_its_absolute_root(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "custom.pth"
            weight.write_bytes(b"local-upscaler")
            resolved = resolve_upscaler_artifact({"source": "local", "value": str(weight)})

        self.assertEqual(resolved.receipt["artifact"]["source"], "local")
        self.assertEqual(resolved.receipt["artifact"]["weightName"], "custom.pth")
        self.assertNotIn(directory, json.dumps(resolved.receipt))

    def test_declared_upscaler_digest_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / "tampered.pth"
            weight.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "declared SHA-256"):
                resolve_upscaler_artifact(
                    {"source": "local", "value": str(weight), "sha256": "0" * 64}
                )

    def test_soundtrack_and_lyric_receipts_exclude_the_primary_pipeline(self):
        video_primary = _node(
            "modules.DiffusersVideo",
            "LoadPipeline",
            model_id={"source": "hub", "value": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"},
            revision="b8fff7315c768468a5333511427288870b2e9635",
            pipeline_class="WanTI2VPipeline",
        )
        soundtrack = _node(
            "modules.DiffusersAudio",
            "LoadPipeline",
            model_id={"source": "hub", "value": "ACE-Step/acestep-v15-xl-turbo-diffusers"},
            revision=ACE_REVISION,
            pipeline_class="AceStepPipeline",
        )
        soundtrack_graph = {
            "nodes": {"video": video_primary, "audio": soundtrack, "mux": _node("modules.Video", "ExportWithAudio")},
            "paths": [["video", "audio", "mux"]],
        }
        soundtrack_receipts = controlled_artifact_receipts_from_graph(
            soundtrack_graph,
            primary_candidate={
                "loaderModule": "modules.DiffusersVideo",
                "loaderAction": "LoadPipeline",
                "pipelineClass": "WanTI2VPipeline",
            },
        )

        audio_primary = copy.deepcopy(soundtrack)
        lyric_video = _node(
            "modules.DiffusersVideo",
            "LoadPipeline",
            model_id={"source": "hub", "value": "Lightricks/LTX-Video-0.9.8-13B-distilled"},
            pipeline_class="LTXConditionPipeline",
        )
        lyric_graph = {
            "nodes": {"audio": audio_primary, "video": lyric_video, "mux": _node("modules.Video", "ExportWithAudio")},
            "paths": [["audio", "video", "mux"]],
        }
        lyric_receipts = controlled_artifact_receipts_from_graph(
            lyric_graph,
            primary_candidate={
                "loaderModule": "modules.DiffusersAudio",
                "loaderAction": "LoadPipeline",
                "pipelineClass": "AceStepPipeline",
            },
        )

        self.assertEqual([item["kind"] for item in soundtrack_receipts], ["diffusers_pipeline"])
        self.assertEqual(soundtrack_receipts[0]["artifact"]["revision"], ACE_REVISION)
        self.assertEqual(soundtrack_receipts[0]["pipelineClass"], "AceStepPipeline")
        self.assertEqual([item["kind"] for item in lyric_receipts], ["diffusers_pipeline"])
        self.assertEqual(lyric_receipts[0]["artifact"]["revision"], LTX_REVISION)
        self.assertEqual(lyric_receipts[0]["pipelineClass"], "LTXConditionPipeline")

    def test_base_only_pipeline_and_disconnected_auxiliary_pipeline_add_no_receipt(self):
        primary = _node(
            "modules.DiffusersVideo",
            "LoadPipeline",
            model_id={"source": "hub", "value": "Lightricks/LTX-Video-0.9.8-13B-distilled"},
            revision=LTX_REVISION,
            pipeline_class="LTXConditionPipeline",
        )
        disconnected = _node(
            "modules.DiffusersAudio",
            "LoadPipeline",
            model_id={"source": "hub", "value": "ACE-Step/acestep-v15-xl-turbo-diffusers"},
            revision=ACE_REVISION,
            pipeline_class="AceStepPipeline",
        )
        graph = {"nodes": {"primary": primary, "unused": disconnected}, "paths": [["primary"]]}
        self.assertEqual(
            controlled_artifact_receipts_from_graph(
                graph,
                primary_candidate={
                    "loaderModule": "modules.DiffusersVideo",
                    "loaderAction": "LoadPipeline",
                    "pipelineClass": "LTXConditionPipeline",
                },
            ),
            [],
        )

    def test_pipeline_identity_changes_auto_history_key_and_malformed_digest_is_rejected(self):
        graph = {
            "nodes": {
                "video": _node(
                    "modules.DiffusersVideo",
                    "LoadPipeline",
                    model_id={"source": "hub", "value": "Lightricks/LTX-Video-0.9.8-13B-distilled"},
                    pipeline_class="LTXConditionPipeline",
                ),
                "audio": _node(
                    "modules.DiffusersAudio",
                    "LoadPipeline",
                    model_id={"source": "hub", "value": "ACE-Step/acestep-v15-xl-turbo-diffusers"},
                    revision=ACE_REVISION,
                    pipeline_class="AceStepPipeline",
                ),
            },
            "paths": [["video", "audio"]],
        }
        primary = {
            "loaderModule": "modules.DiffusersVideo",
            "loaderAction": "LoadPipeline",
            "pipelineClass": "LTXConditionPipeline",
        }
        receipt = controlled_artifact_receipts_from_graph(graph, primary_candidate=primary)[0]
        candidate = {
            "id": "unit",
            "modelType": "LTXVideoPipeline",
            "mode": "text_to_video",
            "controlledArtifacts": [receipt],
        }
        changed = copy.deepcopy(candidate)
        changed["controlledArtifacts"][0]["pipelineClass"] = "StableAudioPipeline"
        self.assertNotEqual(auto_resource_history_key(candidate), auto_resource_history_key(changed))

        malformed = copy.deepcopy(candidate)
        malformed["controlledArtifacts"][0]["descriptorSha256"] = "0" * 64
        # Malformed receipts normalize differently from the exact candidate;
        # they cannot reuse the exact artifact history identity.
        self.assertNotEqual(auto_resource_history_key(candidate), auto_resource_history_key(malformed))


if __name__ == "__main__":
    unittest.main()
