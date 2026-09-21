# Transformers depth workflows

In Developer → Workflows → Depth estimation, select an installed depth model
and preview the connected workflow. Load Models and Predict Map use the same
ordinary graph, image input, output preview and Saved Block controls as other
workflows. The underlying task nodes are Load Depth Estimation Model and Predict
Depth; the two reviewed Depth Anything V2 variants share these nodes.

The loader uses the official Hugging Face Transformers AutoImageProcessor and
AutoModelForDepthEstimation APIs. It requires the existing reviewed optional
Transformers runtime, an immutable Hub commit or an explicitly selected local
model, local files, native classes and safetensors. Browsing or opening a workflow
does not install a runtime, download weights or authorize repository Python.
The execution profiles declare the same optional-runtime boundary as the existing
Transformers text and vision nodes. Models remain in the existing graph cache.

## Inputs and outputs

Predict Depth takes one image. Processing Resolution `0` preserves the selected
processor's configured resize target; a positive value overrides that target.
The reviewed DPT processor preserves its configured aspect-ratio and patch-size
rules. The backend checks projected geometry before preprocessing, then verifies
actual tensor dimensions before model execution. Images with excessive aspect
ratios can require a connected Resize node. Match Input Resolution uses the
processor's official depth postprocessing to return the source dimensions.

- **Prediction Map** follows MoDiff's existing float32 NHWC relative-depth
  contract, with near `0` and far `1`.
- **Depth Preview** is an RGB grayscale visualization of that normalized map.
- **Native Depth** retains the model's floating-point values after optional
  spatial interpolation, as a CPU float32 HW tensor. It is not min/max normalized.
- **Depth Details** records the immutable model receipt, processed/output shapes,
  native range, depth polarity and normalization semantics. A constant map is
  explicitly identified.

Normalized preview intensity is not distance in metres. Relative inverse-depth
and model-declared metric depth use different polarities. The reviewed Depth
Anything configuration supplies the default; other models must specify their
documented polarity explicitly. A declaration of metric depth does not establish
measurement accuracy. Negative values introduced by upstream bicubic spatial
interpolation are retained in the native output rather than silently clipped.

## Reviewed adapter scope

The nodes are task-generic. The initial bounded preprocessing adapter uses the
native DPTImageProcessor contract; another AutoImageProcessor needs a reviewed
geometry/postprocessing adapter before execution. Selecting an arbitrary AutoModel
class alone cannot establish processor geometry, depth polarity or output units.
The two curated profiles pin the relative Small and Metric Outdoor Small models.
Auto resource qualification and Gallery eligibility remain separate from this
integration and require their own execution evidence.

The source contracts are the official
[Transformers depth documentation](https://huggingface.co/docs/transformers/model_doc/depth_anything)
and the reviewed runtime's `models/dpt/image_processing_dpt.py`,
`models/depth_anything/modeling_depth_anything.py` and
`models/auto/modeling_auto.py`. The executable package provenance and versions
remain in `modiff/optional_runtimes.py`; no additional model library is introduced.

When connecting the preview to a control model, use that model's documented
intensity convention. A compatible Image socket alone does not establish depth
polarity; an ordinary Invert processor can reverse the normalized preview.
