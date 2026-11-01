"""Source-bound Expert graph declarations, independent of runtime activation."""
from copy import deepcopy

from modiff.diffusers_offload_modes import OFFLOAD_MODE_GROUP_DISK, OFFLOAD_MODE_NONE
from modiff.joint_av_model_adapters import PIKA_PIPELINE, PIKA_REPOSITORY, PIKA_REVISION
from modiff.supplemental_modular_contracts import PIKA_RUNTIME_PROFILE, PIKA_WORKFLOW_TASKS


PIKA_EXECUTION_PROFILE = 'pika-h3-flex:official-provider:v1'
_ROLES = (
    ('models', 'modules.ModularDiffusers.ModelsLoader', -760, -80),
    ('prompt', 'modules.ModularDiffusers.WorkflowJointAVTextEncode', -400, -80),
    ('denoise', 'modules.ModularDiffusers.WorkflowJointAVDenoise', -40, -80),
    ('decode', 'modules.ModularDiffusers.WorkflowJointAVDecode', 320, -80),
    ('videoExport', 'modules.Video.ExportWithAudio', 680, -80),
)
_CONDITIONED_ROLES = (
    ('models', 'modules.ModularDiffusers.ModelsLoader', -1080, -80),
    ('beforeEncode', 'modules.ModularDiffusers.WorkflowJointAVBeforeEncode', -720, 200),
    ('prompt', 'modules.ModularDiffusers.WorkflowJointAVTextEncode', -360, -80),
    ('imageEncode', 'modules.ModularDiffusers.WorkflowJointAVVaeEncode', 0, 200),
    ('denoise', 'modules.ModularDiffusers.WorkflowJointAVDenoise', 360, -80),
    ('decode', 'modules.ModularDiffusers.WorkflowJointAVDecode', 720, -80),
    ('videoExport', 'modules.Video.ExportWithAudio', 1080, -80),
)
_BINDINGS = (
    ('models', 'model_type', 'pipelineClass'), ('models', 'repo_id', 'artifact'),
    ('models', 'revision', 'defaultRevision'),
    ('models', 'workflow_id', 'workflowId'), ('models', 'dtype', 'dtype'),
    ('models', 'device', 'device'), ('models', 'auto_offload', 'autoOffload'),
    ('models', 'offload_mode', 'offloadMode'), ('models', 'trust_remote_code', 'false'),
    ('prompt', 'pipeline_class', 'pipelineClass'), ('prompt', 'workflow_id', 'workflowId'),
    ('prompt', 'block_path', 'workflowTextEncoderBlock'), ('prompt', 'prompt', 'prompt'),
    ('denoise', 'pipeline_class', 'pipelineClass'), ('denoise', 'workflow_id', 'workflowId'),
    ('denoise', 'block_path', 'workflowDenoiseBlock'), ('denoise', 'seed', 'seed'),
    ('decode', 'pipeline_class', 'pipelineClass'), ('decode', 'workflow_id', 'workflowId'),
    ('decode', 'block_path', 'workflowDecodeBlock'), ('videoExport', 'fps', 'fps'),
)


def pika_studio_specifications():
    """One exact provider and sampler; no model fit or publication promotion."""
    profile = {
        'id': PIKA_EXECUTION_PROFILE, 'model_type': PIKA_PIPELINE,
        'modes': tuple(PIKA_WORKFLOW_TASKS.values()),
        'loader_module': 'modules.ModularDiffusers', 'loader_action': 'ModelsLoader',
        'execution_path': 'modular-diffusers', 'pipeline_class': PIKA_PIPELINE,
        'default_repo': PIKA_REPOSITORY, 'fallback_repo': None,
        'quantizable_components': (), 'default_quantized_components': (),
        'supported_offload_modes': (OFFLOAD_MODE_NONE, OFFLOAD_MODE_GROUP_DISK),
        'retry_offload_modes': (), 'max_low_memory_side': None, 'max_low_memory_steps': None,
        'live_proof': False, 'compatible_repos': (), 'public': True,
        'required_diffusers_provider': PIKA_RUNTIME_PROFILE,
        'optional_runtime_profiles': (PIKA_RUNTIME_PROFILE,), 'optional_runtime_delivery': 'optional_overlay',
    }
    capability = {
        'modelType': PIKA_PIPELINE, 'label': 'Pika H3 Flex', 'displayName': 'Pika H3 Flex',
        'family': 'Joint video/audio', 'supportTier': 'experimental',
        'qualificationStatus': 'runtime-required', 'qualifiedModes': [],
        'defaultRepo': PIKA_REPOSITORY, 'downloadFiles': [],
        'artifactLabel': 'Pinned task-selected transformer and required scale-one model adapter',
        'license': 'MiniMax-H3 Community License Agreement', 'defaultDtype': 'bfloat16',
        'defaultSize': {'width': 832, 'height': 480, 'aspectRatio': '26:15'},
        'recommendedSteps': 6, 'recommendedGuidance': 1.0,
        'guidanceLabel': 'Package-owned sampler; no guidance control',
        'supportsNegativePrompt': False, 'supportsImageInput': True, 'supportsVideoInput': True,
        'supportsAudioInput': True, 'supportsMultiImage': True, 'supportsMask': False,
        'supportsControlImage': False, 'supportsLayers': False, 'supportsLora': False,
        'outputKind': 'video', 'outputMedia': ['video', 'audio'],
        'recommendedFrames': 124, 'recommendedFps': 24,
        'offloadSupport': {'default': OFFLOAD_MODE_GROUP_DISK, 'lowVram': OFFLOAD_MODE_GROUP_DISK,
            'emergency': OFFLOAD_MODE_GROUP_DISK, 'modes': list(profile['supported_offload_modes'])},
        'lowVram': None, 'modes': list(profile['modes']),
        'modeRequirements': {
            PIKA_WORKFLOW_TASKS['fl2va']: {'note': 'Requires at least one decoded first or last keyframe.'},
            PIKA_WORKFLOW_TASKS['ref2va']: {'note': 'Requires ordered image/video/audio references; audio-only lists are invalid.'},
        },
        'executionStatus': 'expert_only', 'revisionCandidates': [PIKA_REVISION],
        'autoEligible': False, 'templateEligible': False, 'galleryEligible': False, 'liveProof': False,
        'notes': [
            'Explicitly install and verify the exact reviewed official video provider before execution.',
            'The selected task requires its immutable model adapter at scale one; optional style adapters are not admitted.',
            'Six-step schedules belong to the official sampler. Steps, guidance and negative-prompt overrides are rejected.',
            'Group Disk keeps the small float32 audio VAE resident; native model fit, quality and lifecycle remain separate acceptance checks.',
            'Acquisition must use the finite task-specific artifact closure, never the complete repository.',
        ],
    }
    result = {}
    for workflow, task in PIKA_WORKFLOW_TASKS.items():
        roles = _ROLES if workflow == 't2va' else _CONDITIONED_ROLES
        stages = [role for role, *_ in roles if role not in {'models', 'videoExport'}]
        edges = tuple(('models', 'pipeline_components', role, 'pipeline_components') for role in stages)
        edges += tuple((first, 'state_out', second, 'state_in') for first, second in zip(stages, stages[1:]))
        edges += (('decode', 'video', 'videoExport', 'video'), ('decode', 'audio', 'videoExport', 'audio'))
        if workflow == 't2va':
            bindings = _BINDINGS + tuple(('denoise', field, source) for field, source in (
                ('width', 'width'), ('height', 'height'), ('num_frames', 'numFrames')))
        else:
            bindings = _BINDINGS + (
                ('beforeEncode', 'pipeline_class', 'pipelineClass'), ('beforeEncode', 'workflow_id', 'workflowId'),
                ('beforeEncode', 'block_path', 'workflowBeforeEncodeBlock'),
                ('beforeEncode', 'width', 'width'), ('beforeEncode', 'height', 'height'),
                ('beforeEncode', 'num_frames', 'requiredNumFrames' if workflow == 'ref2va' else 'numFrames'),
                ('imageEncode', 'pipeline_class', 'pipelineClass'), ('imageEncode', 'workflow_id', 'workflowId'),
                ('imageEncode', 'block_path', 'workflowVaeEncoderBlock'),
            ) + ((('beforeEncode', 'references', 'references'),) if workflow == 'ref2va' else (
                ('beforeEncode', 'image', 'referenceImages'), ('beforeEncode', 'last_image', 'lastImage')))
        result[f'pika-h3-flex:modular-{task.replace("_", "-")}:v1'] = {
            'modelType': PIKA_PIPELINE, 'mode': task, 'profile': deepcopy(profile),
            'capability': deepcopy(capability), 'roles': roles, 'edges': edges, 'bindings': bindings,
        }
    return result
