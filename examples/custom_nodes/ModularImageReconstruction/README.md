# VAE Image Reconstruction

Stage this directory through Developer → Add local source, inspect it and enable
the exact code. It is also a valid source layout for a pinned Hub Modular block.
It deliberately omits `model_input_names`, as some published Mellon sidecars do.
MoDiff derives a Models socket after approval from `expected_components`.

Connect Load Models → Pipeline Components to Models, a decoded image to Image,
and expose Amount as a control (0 to 1) on a containing Block through Configure
Interface. Send the Image output to Preview Image. The block reconstructs the image using the connected `AutoencoderKL` and
blends it with the source. The loader continues to own weights, precision and
offload policy; this source installs no dependencies and downloads no models.

Use an image with dimensions divisible by the VAE's spatial scale. The block uses
deterministic posterior mode and preserves the normal VAE forward hooks. It supports
compatible AutoencoderKL implementations, not every VAE class or modality.
When the VAE requests `force_upcast`, half-precision weights are temporarily used
in float32 and restored afterward, including on failure. This temporary memory
cost still needs to fit the selected resource policy.
