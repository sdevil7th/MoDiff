"""H3 Image argument binding for the existing official image workflow nodes."""

from PIL import Image

from .workflow_runtime import continuation_generator_from_seed, workflow_stage_progress


def execute_image_stage(node, values):
    if getattr(node, '_interrupt', False):
        raise InterruptedError('Image workflow interrupted before the current stage.')
    # Imports stay inside execution so discovery remains model-free.
    from .workflow_blocks import (
        _bounded_integer, _issue_workflow_state, _require_workflow_state, _reviewed_preceding_stage,
    )

    pipeline_class = values['pipeline_class']
    workflow_id = values['workflow_id']
    for field in ('num_inference_steps', 'guidance_scale', 'strength', 'num_images_per_prompt',
                  'max_sequence_length', 'negative_prompt'):
        if values.get(field) is not None:
            raise ValueError(f'H3 Image does not accept {field}; use its declared stage controls.')
    token, pipeline = node._prepare_pipeline(
        pipeline_components=values.get('pipeline_components'), pipeline_class=pipeline_class,
        workflow_id=workflow_id, block_path=values.get('block_path'),
    )
    previous = _reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key)
    state = None
    if previous is not None:
        state = _require_workflow_state(values.get('state_in'), token=token, pipeline_class=pipeline_class,
                                        workflow_id=workflow_id, completed_stage=previous)
    arguments = {}
    if node.stage == 'before_encode':
        images = values.get('image')
        images = [images] if isinstance(images, Image.Image) else images
        if (not isinstance(images, list) or not 1 <= len(images) <= 8
                or any(not isinstance(image, Image.Image) for image in images)):
            raise ValueError('Reference preparation requires one through eight ordered decoded images.')
        arguments['image'] = images
    elif node.stage == 'text_encoder':
        prompt = values.get('prompt')
        if type(prompt) is not str or not prompt.strip():
            raise ValueError('Image encoding requires a nonblank prompt.')
        arguments['prompt'] = prompt
    elif node.stage == 'denoise':
        for field in ('width', 'height'):
            arguments[field] = _bounded_integer(values.get(field, 2048), label=field, minimum=32, maximum=8192)
        if arguments['width'] * arguments['height'] > 16777216:
            raise ValueError('H3 Image output area must not exceed 16,777,216 pixels.')
        arguments['min_generation_area'] = _bounded_integer(
            values.get('min_generation_area', 4194304), label='Minimum generation area',
            minimum=1024, maximum=16777216,
        )
    if node.stage in {'vae_encoder', 'denoise'}:
        arguments['generator'] = continuation_generator_from_seed(values.get('seed', 0), pipeline, state)
    consumed = {key: value for key, value in arguments.items() if key not in {'generator', 'image'}}
    consumed['pipeline_class'] = pipeline_class
    if 'generator' in arguments:
        consumed['seed'] = arguments['generator'].initial_seed()
    node.record_generation_inputs(consumed)
    # The pinned public blocks getter returns a copy; instrument the tree the
    # ordinary ModularPipeline call actually executes, then restore its methods.
    with workflow_stage_progress(node, pipeline._blocks):
        state = pipeline(**arguments, **({'state': state} if state is not None else {}))
    if getattr(node, '_interrupt', False):
        raise InterruptedError('Image workflow interrupted after the current stage.')
    return {'state_out': _issue_workflow_state(token=token, pipeline_class=pipeline_class,
            workflow_id=workflow_id, completed_stage=node.stage, state=state)}
