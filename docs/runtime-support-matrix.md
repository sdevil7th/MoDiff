# Runtime support matrix

| Profile | Tier | Automated proof | Physical proof |
|---|---|---|---|
| NVIDIA CUDA (Windows/Ubuntu x64) | Supported | Manifest, resolver, CPU-host contract | Required for release |
| Apple MPS (Apple Silicon) | Supported installer | Manifest and contract | Required per model |
| Intel XPU (Linux/Windows x64) | Preview | Manifest, installer, and XPU tensor contract | Required per model/device/driver recipe |
| AMD ROCm Linux, Ubuntu 24.04.3, gfx1150/gfx1151 | Supported stack | Detector fixtures | MoDiff model proof required |
| AMD ROCm Linux, Ubuntu 26.04 | Experimental | Detector fixtures | Local tensor and model proof required |
| AMD Instinct MI300X, Ubuntu 24.04, gfx942 | Preview | Separate SDK profile, resolver and installer/runtime contracts | Cloud device tensor and frontend model proof still required |
| AMD PyTorch Windows | Conditional, install blocked | Official-platform manifest and explicit guidance | Complete MoDiff SDK wheel lock and physical model proof required |
| CPU | Supported | Install and tensor smoke | Reference host required |

“Supported” describes installation/runtime qualification, not model performance. `/model_capabilities` remains the source of graph capability. Exact model, dtype, placement, optimization, driver, and hardware qualification is receipt-specific; an unqualified recipe may be runnable with a warning but must not be described as optimized.
