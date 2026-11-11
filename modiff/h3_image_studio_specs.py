"""Reviewed H3 Image selections for the ordinary generic image operations."""

from copy import deepcopy

from modiff.model_artifact_catalog import read_model_artifact_catalog
from modiff.optional_runtimes import DIFFUSERS_H3_IMAGE_RUNTIME_PROFILE_ID


PIPELINE = 'MiniMaxH3ImageModularPipeline'
REPOSITORY = 'MiniMaxAI/MiniMax-H3-Image'
PROFILE = 'minimax-h3-image:official-provider:v1'


def h3_image_studio_specifications():
    pin, = [row for row in read_model_artifact_catalog()['downloadInventories'] if row['repo'] == REPOSITORY]
    profile = {
        'id': PROFILE, 'model_type': PIPELINE, 'pipeline_class': PIPELINE,
        'modes': ('text_to_image', 'edit_image', 'multi_image_reference_edit'),
        'loader_module': 'modules.ModularDiffusers', 'loader_action': 'ModelsLoader',
        'execution_path': 'modular-diffusers', 'default_repo': REPOSITORY, 'fallback_repo': None,
        'quantizable_components': (), 'default_quantized_components': (),
        'supported_offload_modes': ('none', 'group_disk'), 'retry_offload_modes': (),
        'max_low_memory_side': None, 'max_low_memory_steps': None,
        'live_proof': False, 'compatible_repos': (), 'public': True,
        'required_diffusers_provider': DIFFUSERS_H3_IMAGE_RUNTIME_PROFILE_ID,
        'optional_runtime_profiles': (DIFFUSERS_H3_IMAGE_RUNTIME_PROFILE_ID,),
        'optional_runtime_delivery': 'optional_overlay',
    }
    capability = {
        'modelType': PIPELINE, 'label': 'MiniMax H3 Image', 'displayName': 'MiniMax H3 Image',
        'family': 'MiniMax Image', 'supportTier': 'experimental',
        'qualificationStatus': 'runtime-required', 'qualifiedModes': [],
        'defaultRepo': REPOSITORY, 'downloadFiles': [row['path'] for row in pin['files']],
        'artifactLabel': 'Pinned official Modular image checkpoint',
        'license': 'MiniMax H3 Image Community License Agreement', 'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 2048, 'height': 2048, 'aspectRatio': '1:1'},
        'recommendedSteps': 6, 'recommendedGuidance': 1.0,
        'guidanceLabel': 'Fixed six-head sampler', 'supportsNegativePrompt': False,
        'supportsImageInput': True, 'supportsMultiImage': True, 'supportsMask': False,
        'supportsControlImage': False, 'supportsLayers': False, 'supportsLora': False,
        'outputKind': 'image', 'offloadSupport': {'default': 'group_disk', 'lowVram': 'group_disk',
            'emergency': 'group_disk', 'modes': list(profile['supported_offload_modes'])},
        'lowVram': None, 'modes': list(profile['modes']),
        'modeRequirements': {task: {'requiredImages': ['referenceImages']}
                             for task in ('edit_image', 'multi_image_reference_edit')},
        'executionStatus': 'expert_only', 'revisionCandidates': [pin['revision']],
        'autoEligible': False, 'templateEligible': False, 'galleryEligible': False, 'liveProof': False,
        'notes': ['The official sampler owns six output heads and the sigma schedule.',
                  'Requested output geometry retains the upstream four-megapixel generation canvas.',
                  'Ordered references retain separate vision and VAE resize policies.'],
    }
    result = {}
    for task in profile['modes']:
        stages = [('prompt', 'WorkflowTextEncode'), ('denoise', 'WorkflowImageDenoise'),
                  ('decode', 'WorkflowDecodeImage')]
        if task != 'text_to_image':
            stages = [('beforeEncode', 'WorkflowPrepareImages'), stages[0],
                      ('imageEncode', 'WorkflowImageEncode'), *stages[1:]]
        roles = (('models', 'modules.ModularDiffusers.ModelsLoader', -720, -80),) + tuple(
            (role, 'modules.ModularDiffusers.' + action, index * 360 - 360, -80)
            for index, (role, action) in enumerate(stages)
        ) + (('preview', 'modules.Image.Preview', len(stages) * 360, -80),)
        if task != 'text_to_image':
            roles += (('loadImage', 'modules.Image.Load', -720, 320),)
        edges = tuple(('models', 'pipeline_components', role, 'pipeline_components') for role, _ in stages)
        edges += tuple((first[0], 'state_out', second[0], 'state_in') for first, second in zip(stages, stages[1:]))
        edges += (('decode', 'images', 'preview', 'image'),)
        bindings = (
            ('models', 'model_type', 'pipelineClass'), ('models', 'repo_id', 'artifact'),
            ('models', 'revision', 'defaultRevision'), ('models', 'workflow_id', 'workflowId'),
            ('models', 'dtype', 'dtype'), ('models', 'device', 'device'),
            ('models', 'auto_offload', 'autoOffload'), ('models', 'offload_mode', 'offloadMode'),
            ('models', 'trust_remote_code', 'false'), ('prompt', 'prompt', 'prompt'),
            ('denoise', 'width', 'width'), ('denoise', 'height', 'height'), ('denoise', 'seed', 'seed'),
        )
        for role, _ in stages:
            bindings += ((role, 'pipeline_class', 'pipelineClass'), (role, 'workflow_id', 'workflowId'))
        if task != 'text_to_image':
            bindings += (('loadImage', 'file', 'referenceImages'), ('imageEncode', 'seed', 'seed'))
            edges += (('loadImage', 'image', 'beforeEncode', 'image'),)
        result[f'minimax-h3-image:modular-{task.replace("_", "-")}:v1'] = {
            'modelType': PIPELINE, 'mode': task, 'profile': deepcopy(profile),
            'capability': deepcopy(capability), 'roles': roles, 'edges': edges, 'bindings': bindings,
        }
    return result


def image_stage_field_overrides(action, workflow):
    """Publish only the controls consumed by the exact H3 stage."""
    if action == 'workflow_text_encoder':
        return {'state_in': {'hidden': workflow == 'text2image', 'required': workflow != 'text2image'},
                'negative_prompt': {'hidden': True}, 'max_sequence_length': {'hidden': True}}
    if action == 'workflow_image_encoder':
        return {name: {'hidden': True, 'required': False} for name in ('image', 'width', 'height')}
    if action == 'workflow_image_denoise':
        return {**{name: {'hidden': True} for name in (
            'guidance_scale', 'num_inference_steps', 'num_images_per_prompt', 'strength')},
            'min_generation_area': {'hidden': False},
            **{name: {'default': 2048, 'min': 32, 'max': 8192, 'step': 32} for name in ('width', 'height')}}
    return {}


def image_stage_excluded_fields(action):
    return {
        'workflow_text_encoder': ('negative_prompt', 'max_sequence_length'),
        'workflow_image_encoder': ('image', 'width', 'height'),
        'workflow_image_denoise': ('guidance_scale', 'num_inference_steps', 'num_images_per_prompt', 'strength'),
    }.get(action, ())
