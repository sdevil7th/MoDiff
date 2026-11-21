"""H3 Image argument binding for the existing official image workflow nodes."""

from PIL import Image

from .workflow_runtime import continuation_generator_from_seed, workflow_stage_progress


def _denoise_arguments(values):
    from .workflow_blocks import _bounded_integer

    # These are graph integer-representation bounds, not model size presets.
    # Upstream rounds its own canvas and resizes back to the requested output.
    arguments = {name: _bounded_integer(values.get(name, 2048), label=name,
                                       minimum=1, maximum=2**53 - 1)
                 for name in ('width', 'height')}
    # Unlike requested output area, this setting has an explicit upstream
    # bound in MiniMaxH3ImagePrepareCanvasStep.
    arguments['min_generation_area'] = _bounded_integer(
        values.get('min_generation_area', 4194304), label='Minimum generation area',
        minimum=1024, maximum=16777216,
    )
    _guard_output_allocation(arguments['width'], arguments['height'])
    return arguments


def _canvas_size(pipeline, arguments):
    # Use the selected official provider's geometry, including its actual VAE
    # and transformer compression/patch configuration. Do not duplicate it.
    from diffusers.modular_pipelines.minimax_h3_image.before_denoise import get_image_canvas_size
    return get_image_canvas_size(arguments['height'], arguments['width'],
                                 arguments['min_generation_area'], pipeline.canvas_multiple)


def _guard_output_allocation(width, height):
    from modiff.hardware import system_memory_snapshot

    # Refuse an output whose RGB buffer alone cannot fit. This necessary lower
    # bound does not estimate activations or qualify an otherwise admitted run.
    available = system_memory_snapshot().get('available_bytes')
    required = width * height * 3
    if type(available) is not int or available <= 0:
        raise ValueError('Cannot check available system memory for the requested image output.')
    if required > available:
        raise ValueError(f'The requested RGB output alone needs at least {required} bytes; '
                         f'available system memory is {available} bytes. '
                         'This is a machine allocation check, not a model resolution limit.')


def execute_image_stage(node, values):
    if getattr(node, '_interrupt', False):
        raise InterruptedError('Image workflow interrupted before the current stage.')
    # Imports stay inside execution so discovery remains model-free.
    from .workflow_blocks import (
        _issue_workflow_state, _require_workflow_state, _reviewed_preceding_stage,
    )

    pipeline_class = values['pipeline_class']
    workflow_id = values['workflow_id']
    for field in ('num_inference_steps', 'guidance_scale', 'strength', 'num_images_per_prompt',
                  'max_sequence_length', 'negative_prompt'):
        if values.get(field) is not None:
            raise ValueError(f'H3 Image does not accept {field}; use its declared stage controls.')
    # Validate before a pipeline can acquire components or allocate tensors.
    arguments = _denoise_arguments(values) if node.stage == 'denoise' else {}
    if node.stage == 'before_encode':
        images = values.get('image')
        images = [images] if isinstance(images, Image.Image) else images
        if (not isinstance(images, list) or not images
                or any(not isinstance(image, Image.Image) for image in images)):
            raise ValueError('Reference preparation requires a nonempty ordered list of decoded images.')
        arguments['image'] = images
    elif node.stage == 'text_encoder':
        prompt = values.get('prompt')
        if type(prompt) is not str or not prompt.strip():
            raise ValueError('Image encoding requires a nonblank prompt.')
        arguments['prompt'] = prompt
    token, pipeline = node._prepare_pipeline(
        pipeline_components=values.get('pipeline_components'), pipeline_class=pipeline_class,
        workflow_id=workflow_id, block_path=values.get('block_path'),
    )
    if node.stage == 'denoise':
        canvas_height, canvas_width = _canvas_size(pipeline, arguments)
        _guard_output_allocation(canvas_width, canvas_height)
    previous = _reviewed_preceding_stage(pipeline_class, workflow_id, node.action_key)
    state = None
    if previous is not None:
        state = _require_workflow_state(values.get('state_in'), token=token, pipeline_class=pipeline_class,
                                        workflow_id=workflow_id, completed_stage=previous)
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
