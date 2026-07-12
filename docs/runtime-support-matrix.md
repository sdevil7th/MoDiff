# Runtime support matrix

| Profile | Tier | Automated proof | Physical proof |
|---|---|---|---|
| NVIDIA CUDA (Windows/Ubuntu x64) | Supported | Manifest, resolver, CPU-host contract | Required for release |
| Apple MPS (Apple Silicon) | Supported installer | Manifest and contract | Required per model |
| AMD ROCm Linux, Ubuntu 24.04.3, gfx1150/gfx1151 | Supported stack | Detector fixtures | MoDiff model proof required |
| AMD ROCm Linux, Ubuntu 26.04 | Experimental | Detector fixtures | Local tensor and model proof required |
| AMD PyTorch Windows | Preview | Detector fixtures | Supported Ryzen/Radeon required |
| CPU | Supported | Install and tensor smoke | Reference host required |

“Supported” describes installation/runtime qualification. `/model_capabilities` remains the source of per-model qualification.
