"""Prompt examples for newly resolved operations, never saved graph migration.

Creator-derived examples are deliberately short paraphrases. Numeric settings
continue to come from the reviewed execution profile, independently of examples.
"""

_TASK_EXAMPLES = {
    "text_to_image": "A red ceramic teapot on a wooden table beside a window, soft morning light, detailed photograph.",
    "image_to_image": "A watercolor painting of the scene in the input image, preserving its composition and main subjects.",
    "edit_image": "Change the background to a sunlit garden. Keep the main subject, its pose and its appearance unchanged.",
    "multi_image_reference_edit": "Place the subjects from the reference images together in a sunlit garden, preserving their appearances.",
    "inpaint": "A small vase of yellow flowers in the masked area, matching the surrounding lighting and perspective.",
    "outpaint": "Continue the scene naturally beyond the original border, matching its lighting, perspective and style.",
    "control_image": "A detailed photograph following the structure of the control image, with soft natural lighting.",
    "text_to_audio": "Gentle rain falling on leaves, distant thunder, a quiet natural atmosphere without speech or music.",
    "audio_continuation": "Continue the recording with the same atmosphere, rhythm and instrumentation.",
    "audio_variation": "A variation of the input recording with the same style and mood.",
    "audio_repaint": "Blend the regenerated section naturally with the surrounding audio.",
}


def seed_operation_example(node, repository=None):
    """Called only while creating a resolved node schema, before authoring begins."""
    operation = node.get("operation", {})
    task = operation.get("task")
    example = _TASK_EXAMPLES.get(task)
    field = node.get("params", {}).get("prompt")
    if (
        not example or not field or field.get("hidden") or field.get("display") == "output"
        or ((field.get("value") or field.get("default")) and not field.get("fieldOptions", {}).get("exampleAttribution"))
    ):
        return
    if repository is None and field.get("fieldOptions", {}).get("exampleAttribution"):
        return
    source = None
    if operation.get("pipelineClass") == "QwenImage21Pipeline" and task in {"text_to_image", "edit_image"}:
        source = "https://github.com/QwenLM/Qwen-Image-2.1"
        if task == "text_to_image":
            example = 'A glowing shop sign reading "HELLO", seen on a rainy evening with reflections across the street.'
        elif task == "edit_image":
            example = "Replace the background with a beach at sunset, preserving the subject."
    elif repository == "black-forest-labs/FLUX.1-schnell" and task == "text_to_image":
        source = "https://huggingface.co/black-forest-labs/FLUX.1-schnell"
        example = 'A friendly cat presenting a handwritten sign reading "Welcome", detailed fur and soft daylight.'
    elif repository == "Tongyi-MAI/Z-Image-Turbo" and task == "text_to_image":
        source = "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo"
        example = (
            "A portrait of a woman wearing embroidered crimson traditional clothing and a gold hair ornament, "
            "holding a painted fan. A softly illuminated pagoda and colorful evening lights form the background."
        )
    options = dict(field.get("fieldOptions") or {})
    options["exampleAttribution"] = "Adapted from creator guidance" if source else "MoDiff task example"
    if source:
        options["exampleSource"] = source
    else:
        options.pop("exampleSource", None)
    field.update(value=example, fieldOptions=options)
    node.setdefault("values", {})["prompt"] = example
