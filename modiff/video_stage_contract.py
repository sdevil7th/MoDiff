"""Pure authoring declarations for exact ordinary video stage selections.

No model imports, node construction, runtime activation or qualification.
"""
PROFILE_CLASSES = {
    'ltx-video:direct': 'LTXConditionPipeline',
    'wan-text-to-video:direct': 'WanPipeline',
    'wan-video-to-video:direct': 'WanVideoToVideoPipeline',
    'wan-22-image-to-video:direct': 'WanImageToVideoPipeline',
    'wan-vace:direct': 'WanVACEPipeline',
    'wan-22-ti2v-5b:direct': 'WanTI2VPipeline',
    'wan-22-ti2v-5b:image-direct:v1': 'WanTI2VImageToVideoPipeline',
}
ENCODE_FIELDS = frozenset({'pipeline', 'video_contract', 'mode', 'prompt', 'negative_prompt',
    'reference_images', 'last_image', 'video', 'mask', 'guidance_scale', 'max_sequence_length'})
DENOISE_FIELDS = frozenset({'pipeline', 'video_contract', 'mode', 'width', 'height', 'num_frames',
    'frame_rate', 'seed', 'num_inference_steps', 'guidance_scale', 'strength', 'denoise_strength',
    'reference_strength', 'conditioning_scale', 'scheduler_flow_shift', 'guidance_scale_2',
    'use_guidance_scale_2', 'secondary_guidance_scale', 'attention_kwargs_json'})


def video_stage_spec(original):
    """Project one retained exact whole recipe onto its actual stage nodes."""
    if (not isinstance(original, dict)
            or PROFILE_CLASSES.get(original.get('executionProfileId')) != original.get('pipelineClass')):
        raise ValueError('No declared stages for this exact video source profile.')
    roles = original['roles']
    if (sum(role == 'wanGenerate' and key == 'modules.DiffusersVideo.Generate' for role, key, *_ in roles) != 1
            or sum(role == 'wanPipeline' and key == 'modules.DiffusersVideo.LoadPipeline' for role, key, *_ in roles) != 1):
        raise ValueError('The retained video source roles disagree with their stage adapter.')
    # VACE's long-source adapter decodes each window to provide the next
    # continuity anchor. Its genuine boundary is encoded inputs -> rendered
    # frames, not one invented latent tensor for the whole clip.
    rendered = original['pipelineClass'] == 'WanVACEPipeline'
    expanded = []
    for role, key, x, y in roles:
        if role == 'wanGenerate':
            expanded.append(('videoEncode', 'modules.DiffusersVideo.EncodeInputs', x, y))
            expanded.extend((('denoise', 'modules.DiffusersVideo.Generate', x + 370, y),) if rendered else (
                ('denoise', 'modules.DiffusersVideo.Denoise', x + 370, y),
                ('decode', 'modules.DiffusersVideo.DecodeLatents', x + 740, y)))
        else:
            expanded.append((role, key, x, y))
    edges = []
    for source, output, target, field in original['edges']:
        if source == 'wanPipeline' and target == 'wanGenerate' and field == 'pipeline':
            edges.extend((source, output, role, field) for role in (
                ('videoEncode', 'denoise') if rendered else ('videoEncode', 'denoise', 'decode')))
            continue
        if source == 'wanGenerate':
            source = 'denoise' if rendered else 'decode'
        if target == 'wanGenerate':
            if field not in ENCODE_FIELDS | DENOISE_FIELDS:
                raise ValueError('The retained video source has an undeclared stage input.')
            target = 'videoEncode' if field in ENCODE_FIELDS else 'denoise'
        edges.append((source, output, target, field))
    edges.append(('videoEncode', 'encoded_inputs', 'denoise', 'encoded_inputs'))
    if not rendered:
        edges.extend((('denoise', 'latents_out', 'decode', 'latents'),
                      ('denoise', 'latent_binding', 'decode', 'latent_binding')))
    bindings = []
    for role, field, source in original['bindings']:
        if role != 'wanGenerate':
            bindings.append((role, field, 'defaultRevision'
                             if role == 'wanPipeline' and field == 'revision' else
                             'attentionBackend'
                             if role == 'diffusersRecipe' and field == 'attention_backend' else source))
        elif field == 'mode':
            # The resolved stage operations already own their exact selected
            # task. A form binding must not override that immutable identity.
            continue
        elif rendered and field == 'frame_rate':
            # VACE has no frame-rate call argument. Delivery owns the same
            # authored FPS through videoExport; do not expose an unused integer
            # duplicate that truncates the export's legitimate float value.
            continue
        elif field == 'output_type':
            bindings.append(('denoise' if rendered else 'decode', field, source))
        else:
            destinations = [('videoEncode', ENCODE_FIELDS), ('denoise', DENOISE_FIELDS)]
            found = [name for name, fields in destinations if field in fields]
            if not found:
                raise ValueError(f'The retained video source has an undeclared stage control: {field}.')
            bindings.extend((name, field, source) for name in found)
    if rendered and original['mode'] == 'video_inpaint':
        # The curated masked-edit recipe exposes its optional identity guide.
        # It still enters the official VACE encoder as ordinary source media.
        expanded.append(('loadImage', 'modules.Image.Load', -520, 780))
        edges.append(('loadImage', 'image', 'videoEncode', 'reference_images'))
        bindings.extend((('loadImage', 'file', 'referenceImages'), ('loadImage', 'alpha_channel', 'alphaMode')))
    return {**original, 'roles': tuple(expanded), 'edges': tuple(edges), 'bindings': tuple(bindings)}


def declared_backend_video_stages(profile, task, artifact, spec):
    """Static eligibility for the exact declaration, never live support."""
    from modiff.model_artifact_catalog import catalog_repository_pin
    if (not isinstance(profile, dict) or not isinstance(artifact, dict) or not isinstance(spec, dict)
            or PROFILE_CLASSES.get(profile.get('id')) != profile.get('pipeline_class')
            or profile.get('execution_path') != (
                'direct-wan-vace' if profile.get('pipeline_class') == 'WanVACEPipeline' else 'direct-diffusers-video')
            or profile.get('loader_module') != 'modules.DiffusersVideo' or profile.get('loader_action') != 'LoadPipeline'
            or task not in profile.get('modes', ()) or spec.get('mode') != task
            or spec.get('executionProfileId') != profile.get('id')
            or spec.get('pipelineClass') != profile.get('pipeline_class')):
        return False
    pin = catalog_repository_pin(profile.get('default_repo'), model_type=profile.get('model_type'))
    if not pin or artifact != {'repository': pin['repo'], 'revision': pin['revision']}:
        return False
    try:
        video_stage_spec(spec)
    except (TypeError, KeyError, ValueError):
        return False
    return True
