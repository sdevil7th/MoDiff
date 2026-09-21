"""Bounded adapter for the official DPT image-processor depth contract.

AutoModel handles model selection. Processor geometry and depth polarity are
separate backend contracts; normalized previews never claim metric distances.
"""
from math import isfinite


def depth_geometry(processor, image, resolution):
    """Bound the reviewed processor's resize before it allocates pixel tensors."""
    if type(processor).__name__ != "DPTImageProcessor":
        raise ValueError("This depth processor has no reviewed bounded preprocessing contract.")
    size = processor.size if resolution == 0 else {"height": resolution, "width": resolution}
    if not isinstance(size, dict):
        from transformers.image_utils import SizeDict

        if not isinstance(size, SizeDict):
            raise ValueError("Depth preprocessing requires the native processor size contract.")
        size = dict(size)
    if set(size) != {"height", "width"}:
        raise ValueError("Depth preprocessing requires explicit height and width.")
    if any(type(v) is not int or not 1 <= v <= 2048 for v in size.values()):
        raise ValueError("Depth preprocessing size must be between 1 and 2048 pixels.")
    multiple = processor.ensure_multiple_of
    if type(multiple) is not int or not 1 <= multiple <= 128:
        raise ValueError("Depth preprocessing has an invalid patch multiple.")
    if processor.do_pad or not processor.do_resize:
        raise ValueError("Depth preprocessing requires the reviewed resize without padding.")
    width, height = image.size
    scale_h, scale_w = size["height"] / height, size["width"] / width
    if processor.keep_aspect_ratio:
        scale_h = scale_w if abs(1 - scale_w) < abs(1 - scale_h) else scale_h
        scale_w = scale_h
    processed_h = round(scale_h * height / multiple) * multiple
    processed_w = round(scale_w * width / multiple) * multiple
    if not (1 <= processed_h <= 4096 and 1 <= processed_w <= 4096 and processed_h * processed_w <= 4_194_304):
        raise ValueError("Depth preprocessing exceeds the bounded image geometry; resize the source image first.")
    return size, (1, 3, processed_h, processed_w)


def depth_convention(model, selected):
    if selected in {"near_is_larger", "far_is_larger"}:
        return selected, "operator_selected"
    if selected != "model_default":
        raise ValueError("Depth convention must be model_default, near_is_larger, or far_is_larger.")
    config = getattr(model, "config", None)
    # The AutoModel contract does not define a universal depth polarity. Infer
    # only the reviewed native configuration; other models can state it explicitly.
    if getattr(config, "model_type", None) == "depth_anything":
        kind = getattr(config, "depth_estimation_type", None)
        if kind == "relative":
            return "near_is_larger", "relative_inverse_depth"
        if kind == "metric":
            maximum = getattr(config, "max_depth", None)
            if type(maximum) in (int, float) and isfinite(maximum) and maximum > 0:
                return "far_is_larger", "model_declared_metric_depth"
    raise ValueError("This model does not declare a reviewed depth convention; choose its documented depth polarity.")


def predict_depth(model, processor, image, *, receipt, resolution, match_input, convention):
    import torch
    from modules.DiffusersImage.main import normalize_prediction_map

    size, expected_shape = depth_geometry(processor, image, resolution)
    polarity, semantics = depth_convention(model, convention)
    batch = processor(images=[image], size=size, return_tensors="pt")
    if not hasattr(batch, "items") or set(batch) != {"pixel_values"}:
        raise RuntimeError("Depth preprocessing returned an unexpected tensor schema.")
    pixels = batch["pixel_values"]
    if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != expected_shape or not pixels.is_floating_point():
        raise RuntimeError("Depth preprocessing returned an unexpected pixel tensor.")
    if not bool(torch.isfinite(pixels).all()):
        raise RuntimeError("Depth preprocessing returned nonfinite pixel values.")
    runtime = receipt["runtime"]
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[runtime["dtype"]]
    with torch.inference_mode():
        outputs = model(pixel_values=pixels.to(device=runtime["device"], dtype=dtype))
        depth = getattr(outputs, "predicted_depth", None)
        if not isinstance(depth, torch.Tensor) or tuple(depth.shape) != (1, *expected_shape[-2:]):
            raise RuntimeError("Depth model returned an unexpected prediction shape.")
        if not depth.is_floating_point() or not bool(torch.isfinite(depth).all()):
            raise RuntimeError("Depth model returned nonfinite or nonnumeric predictions.")
        # The processor's official postprocessing owns spatial interpolation.
        processed = processor.post_process_depth_estimation(
            outputs, target_sizes=[(image.height, image.width)] if match_input else None
        )
        if not isinstance(processed, list) or len(processed) != 1 or set(processed[0]) != {"predicted_depth"}:
            raise RuntimeError("Depth postprocessing returned an unexpected result schema.")
        native = processed[0]["predicted_depth"]
        expected_output = (image.height, image.width) if match_input else expected_shape[-2:]
        if not isinstance(native, torch.Tensor) or tuple(native.shape) != expected_output or not native.is_floating_point():
            raise RuntimeError("Depth postprocessing returned an unexpected output shape.")
        native = native.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if not bool(torch.isfinite(native).all()):
            raise RuntimeError("Depth postprocessing returned nonfinite values.")
        minimum, maximum = float(native.min()), float(native.max())
        span = maximum - minimum
        normalized = (native - minimum) / span if span > 0 else torch.zeros_like(native)
        if polarity == "near_is_larger" and span > 0:
            normalized = 1 - normalized
        prediction_map, previews = normalize_prediction_map(normalized, kind="depth")
    return {
        "prediction_map": prediction_map,
        "preview_images": previews,
        "native_depth": native,
        "width_out": expected_output[1],
        "height_out": expected_output[0],
        "result": {
            "schemaVersion": 1,
            "task": "depth-estimation",
            "modelReceipt": receipt,
            "processedShape": list(expected_shape),
            "nativeDepth": {
                "layout": "HW", "dtype": "float32", "shape": list(expected_output),
                "semantics": semantics, "polarity": polarity, "valueRange": [minimum, maximum],
                "resizedToInput": match_input,
            },
            "preview": {"semantics": "relative_depth", "nearValue": 0.0, "farValue": 1.0, "constant": span == 0},
        },
    }
