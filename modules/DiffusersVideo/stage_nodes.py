"""Ordinary video stage nodes sharing the existing registered video adapters."""
from copy import deepcopy

from modiff.NodeBase import NodeBase
from modiff.video_stage_contract import ENCODE_FIELDS, DENOISE_FIELDS
from . import stage_execution as core
from .main import Generate, _adapter_signal


def _params(names):
    params = {key: deepcopy(value) for key, value in Generate.params.items() if key in names}
    params['mode']['onChange'] = 'update_stage_contract'
    params['pipeline']['onSignal'] = [
        {'action': 'value', 'target': 'video_contract'},
        {'action': 'exec', 'data': 'update_stage_contract'},
    ]
    return params


def _update_stage_contract(node, values, ref):
    signal = values.get('video_contract')
    if not isinstance(signal, dict):
        raise ValueError('The video loader did not publish its current contract.')
    from .main import get_video_pipeline_adapter, get_video_mode_field_contract
    adapter = get_video_pipeline_adapter(signal.get('pipelineClass'))
    if signal != _adapter_signal(adapter) or adapter.pipeline_class not in core.PROFILE_CLASSES.values():
        raise ValueError('The selected video pipeline has no separate-stage adapter.')
    mode = values.get('mode')
    if mode not in adapter.modes:
        raise ValueError('Select an explicit supported video task.')
    for name, field in get_video_mode_field_contract(adapter, mode).field_param_overlay().items():
        if name in node.params:
            node.set_field_params(name, field)
    node.set_field_params('mode', {'options': list(adapter.modes)})


class EncodeInputs(NodeBase):
    """Encode text and retain owned source media for the official video call."""
    label = 'Encode Video Inputs'
    category = 'Diffusers Video'
    resizable = True
    params = {
        **_params(ENCODE_FIELDS),
        'encoded_inputs': {'label': 'Encoded inputs', 'type': 'video_conditioning', 'display': 'output'},
    }
    update_stage_contract = _update_stage_contract

    def __call__(self, **kwargs):
        core.require_owner(kwargs['pipeline'])
        previous = self.output.get('encoded_inputs')
        if previous is not None:
            try:
                core.require_encoding(kwargs['pipeline'], previous, kwargs.get('mode'), kwargs.get('guidance_scale'))
            except ValueError:
                self.invalidate_cache()
        return super().__call__(**kwargs)

    def execute(self, **kwargs):
        self.progress(-1, phase='encoding', message='Encoding video inputs')
        if self._interrupt:
            raise InterruptedError('Video encoding interrupted.')
        self._active_pipeline = kwargs['pipeline']
        try:
            result = core.encode(kwargs['pipeline'], kwargs)
        finally:
            self._active_pipeline = None
        if self._interrupt:
            raise InterruptedError('Video encoding interrupted.')
        self.record_generation_inputs(dict(result.consumed))
        return {'encoded_inputs': result}


class Denoise(Generate):
    """Run the existing official video sampler with matching owned encoding."""
    label = 'Denoise Video'
    category = 'Diffusers Video'
    resizable = True
    params = {
        **_params(DENOISE_FIELDS),
        'frame_rate': {**deepcopy(Generate.params['frame_rate']), 'type': 'float', 'step': .01},
        'encoded_inputs': {'label': 'Encoded inputs', 'type': 'video_conditioning', 'display': 'input', 'required': True},
        'latents_out': {'label': 'Video latents', 'type': 'tensor', 'display': 'output'},
        'latent_binding': {'label': 'Latent binding', 'type': 'video_latent_binding', 'display': 'output'},
        **{key: deepcopy(Generate.params[key]) for key in ('width_out', 'height_out', 'frames_out')},
    }
    update_stage_contract = _update_stage_contract

    def __call__(self, **kwargs):
        core.require_encoding(kwargs['pipeline'], kwargs.get('encoded_inputs'),
                              kwargs.get('mode'), kwargs.get('guidance_scale'))
        return super().__call__(**kwargs)

    def _invoke_pipeline(self, pipeline, call_kwargs):
        encoding = self._video_encoding
        from modiff.execution_input_provenance import FIELD_NAMES
        if encoding.owner.pipeline_class == 'LTXConditionPipeline':
            call_kwargs['frame_rate'] = self._video_frame_rate
        call_kwargs.update(encoding.embeddings)
        call_kwargs.update(prompt=None, negative_prompt=None, output_type='latent')
        result = pipeline(**call_kwargs)
        # Text belongs to Encode Inputs, and latent transport is fixed here.
        # Recording their placeholder None/latent values as user settings would
        # contradict the actual prompt and final decoded output in the summary.
        self._video_consumed = {key: value for key, value in call_kwargs.items()
            if key in FIELD_NAMES and key not in {'prompt', 'negative_prompt', 'output_type'}}
        self._video_consumed['executed_task'] = encoding.mode
        self._video_consumed['seed'] = int(call_kwargs['generator'].initial_seed())
        # The ordinary Wan adapter restores its temporary scheduler in finally.
        # Seal against the loader's owner only after that restoration, without
        # weakening component identity checks during future stage consumption.
        self._video_sample = (result.frames,
            (call_kwargs['width'], call_kwargs['height'], call_kwargs['num_frames']), call_kwargs['generator'])
        return result

    def execute(self, **kwargs):
        pipeline = kwargs['pipeline']
        encoding = core.require_encoding(pipeline, kwargs.get('encoded_inputs'),
                                          kwargs.get('mode'), kwargs.get('guidance_scale'))
        values = {**kwargs, 'prompt': None, 'negative_prompt': None, 'output_type': 'latent',
                  'max_sequence_length': encoding.max_sequence_length}
        chunks = encoding.media['video_chunks']
        values['video'] = [frame for chunk in chunks for frame in chunk] if chunks is not None else None
        references = encoding.media['reference_images']
        values['reference_images'] = list(references) if references is not None else None
        self._video_encoding, self._video_sample = encoding, None
        self._video_consumed = None
        self._video_frame_rate = float(values.get('frame_rate', 25))
        try:
            if self._interrupt:
                raise InterruptedError('Video denoising interrupted.')
            self._execute_with_adapter(**values)
            if self._interrupt:
                raise InterruptedError('Video denoising interrupted.')
            sample = self._video_sample
            if sample is None:
                raise ValueError('The selected video adapter did not produce stage latents.')
            binding = core.issue_latent(pipeline, encoding, *sample)
            consumed = self._video_consumed
        finally:
            self._video_encoding, self._video_sample = None, None
            self._video_consumed = None
        self.record_generation_inputs(consumed)
        width, height, frames = binding.geometry
        return {'latents_out': binding.tensor, 'latent_binding': binding,
                'width_out': width, 'height_out': height, 'frames_out': frames}


class DecodeLatents(NodeBase):
    """Decode the selected video owner's actual latents through its official VAE."""
    label = 'Decode Video Latents'
    category = 'Diffusers Video'
    resizable = True
    params = {
        'pipeline': {**deepcopy(Generate.params['pipeline']), 'onSignal': []},
        'output_type': {'label': 'Output type', 'type': 'string', 'default': 'pil', 'options': ['pil']},
        'latents': {'label': 'Video latents', 'type': 'tensor', 'display': 'input', 'required': True},
        'latent_binding': {'label': 'Latent binding', 'type': 'video_latent_binding', 'display': 'input', 'required': True},
        **{key: deepcopy(Generate.params[key]) for key in ('video_out', 'width_out', 'height_out', 'frames_out')},
    }
    def __call__(self, **kwargs):
        core.require_latent(kwargs['pipeline'], kwargs.get('latent_binding'), kwargs.get('latents'))
        return super().__call__(**kwargs)

    def execute(self, **kwargs):
        if kwargs.get('output_type', 'pil') != 'pil':
            raise ValueError('Separate video output currently requires PIL frames; use the whole operation for other formats.')
        core.require_latent(kwargs['pipeline'], kwargs.get('latent_binding'), kwargs.get('latents'))
        self.progress(-1, phase='decoding', message='Decoding video frames')
        if self._interrupt:
            raise InterruptedError('Video decoding interrupted.')
        self._active_pipeline = kwargs['pipeline']
        try:
            result = core.decode(kwargs['pipeline'], kwargs['latent_binding'])
        finally:
            self._active_pipeline = None
        if self._interrupt:
            raise InterruptedError('Video decoding interrupted.')
        self.record_generation_inputs({'output_type': 'pil', 'executed_task': kwargs['latent_binding'].encoding.mode})
        return result
