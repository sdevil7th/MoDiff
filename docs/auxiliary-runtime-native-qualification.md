# Auxiliary runtime installation checks

The auxiliary-only profile definitions were checked on Linux x86-64 with
CPython 3.12 on 2026-10-06. These checks support installation and activation
actions on that target. They do not qualify an Auto route, image generation,
accelerator execution, model performance, Windows, or macOS.

The native application environment contained Torch `2.14.1+cpu`, Transformers
`5.18.0`, PEFT `0.21.2`, and Diffusers `0.41.0.dev0` from project-pinned commit
`fbf49e7f35857f76bc57b177e26f12b03687c668`. No mandatory model library was staged
in an auxiliary overlay.

For each enabled profile, the existing production installer ran in a temporary
operator data directory and managed root, with the current profile's exact spec
digest. It acquired the existing reviewed wheel locks, verified artifact hashes,
installed without dependencies, completed fresh-process validation against the
native base, and promoted the sealed overlay. After recording activation, a
fresh isolated (`python -I`) child invoked the ordinary startup overlay loader.
Torch, Transformers, and PEFT import origins were checked to remain in the native
application environment. The child performed the package checks below. The
selection was then rolled back and temporary operator data removed. No running
backend was restarted, model weights downloaded, or user data modified.

| Profile | Exact staged packages | Fresh CPU check | Result |
| --- | --- | --- | --- |
| GGUF | `gguf==0.19.0` | `GGUFReader` import and Diffusers `GGUFQuantizationConfig(compute_dtype=torch.float32)` construction | Passed |
| Quanto | `optimum-quanto==0.2.7`, `ninja==1.13.0` | Quantize a `Sequential(Linear(4, 4))` child to `qint8`, freeze it, and execute a finite `[1, 4]` output through the resulting `QLinear` | Passed |
| Gallery media | `opencv-python-headless==5.0.0.93`, `av==18.1.0` | OpenCV Canny on an 8×8 array; PyAV in-memory Matroska/FFV1 encode/decode of one grayscale frame with exact pixel round trip | Passed |
| bitsandbytes | `bitsandbytes==0.50.0` | Current native-base CUDA binary execution was unavailable | Actions remain closed |

The checked identities are:

| Historical workflow profile ID | Current spec digest |
| --- | --- |
| `huggingface-transformers-main-96fe6dce-peft-0.20.0-quanto-0.2.7` | `sha256:b5cf3eaff6041fd898066eeebbc5606d21b6b60426f0d26c2237ef45e394c9cd` |
| `huggingface-transformers-main-96fe6dce-peft-0.20.0-gguf-0.19.0` | `sha256:a92395e7b3a652a2badbeb346d336de96819dc83e5698801825feef4aecf52ec` |
| `gallery-media-opencv-5.0.0.93-pyav-18.1.0` | `sha256:960fdf8febf502c1bafe8ac0ba0e0f06c8fba56622c5c3af41cd31e42fbcc4e7` |
| `huggingface-transformers-main-96fe6dce-peft-0.20.0-bitsandbytes-0.50.0` | `sha256:f70be9681e65f59bd428851ed296944d6ff9a8a099eab4f41ec3e105514da393` |

The IDs retain saved workflow references. Their old composite spec digests and
installation receipts do not authorize the new definitions. Exact artifact
seals, native base version/origin bindings, explicit installation, and startup
validation remain required. Other targets stay closed pending their own checks.
Future changes to these specs need fresh installation evidence; this snapshot
does not transfer qualification to a new library version or environment.

The focused metadata regression is
`python -m pytest -q tests/test_auxiliary_runtime_profiles.py` (14 passed). It
checks staged wheel scope, unchanged reviewed auxiliary artifact identities,
required project dependency floors, changed composite spec digests, and the
target action boundaries. The CPU computations above were separate production
installer checks, not mocked unit tests.
