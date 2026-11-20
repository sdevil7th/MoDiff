"""Finite storage forecasts for reviewed native video stages and composition.

Inspection follows declared values/edges only. It neither decodes media nor
executes a supplier, and is not a measured activation peak or hardware claim.
"""
from __future__ import annotations

import math


def native_video_storage_geometry(nodes, source_id, source_port, planned, *, literal, number):
    source = nodes.get(source_id, {})
    key = (source.get('module'), source.get('action'), source_port)
    native = key in {('modules.DiffusersVideo', 'DecodeLatents', 'video_out'),
                     ('modules.DiffusersVideo', 'Generate', 'video_out')}
    composition = key == ('modules.Video', 'Concatenate', 'output')
    if not native and not composition:
        return None
    if native and source.get('action') == 'DecodeLatents':
        denoise = nodes.get(source.get('params', {}).get('latents', {}).get('sourceId'), {})
        owner = nodes.get(denoise.get('params', {}).get('pipeline', {}).get('sourceId'), {})
        # Preserve the original closed single-shot LTX contract and identities.
        # Its explicit ShotInputs-backed compositions are a separate forecast.
        has_shot_controls = any(
            (nodes.get(denoise.get('params', {}).get(field, {}).get('sourceId'), {}).get('module'),
             nodes.get(denoise.get('params', {}).get(field, {}).get('sourceId'), {}).get('action'),
             denoise.get('params', {}).get(field, {}).get('sourceKey'))
            == ('modules.VideoConditioning', 'ShotInputs', port)
            for field, port in [('num_frames', 'frames'), ('seed', 'seed'), ('frame_rate', 'fps_out')])
        if (owner.get('params', {}).get('pipeline_class', {}).get('value') == 'LTXConditionPipeline'
                and not has_shot_controls):
            return None

    def positive(node_id, field):
        raw = literal(nodes, node_id, field)
        value = number(raw)
        if (isinstance(raw, bool) or value is None or not math.isfinite(value)
                or value <= 0 or not float(value).is_integer()):
            raise ValueError(f'{node_id}.{field} needs an inspected positive whole video dimension/count.')
        return int(value)

    def prepared_geometry(encode_id):
        media = nodes[encode_id]['params'].get('video', {})
        supplier = nodes.get(media.get('sourceId'), {})
        identity = (supplier.get('module'), supplier.get('action'), media.get('sourceKey'))
        if identity == ('modules.VideoConditioning', 'Normalize', 'output'):
            return tuple(positive(media['sourceId'], f) for f in ('width', 'height', 'num_frames'))
        if identity == ('modules.VideoConditioning', 'OutpaintCanvas', 'canvas'):
            # Inspect the same output scalars that ordinary Run rechecks before
            # allocating the consumer; no frame collection is manufactured.
            from modiff.workflow_auto_values import inspect_video_canvas_geometry
            result = inspect_video_canvas_geometry(nodes, media['sourceId'], literal=literal)
            return tuple(result[field] for field in ('width', 'height', 'frames'))
        raise ValueError('Native video upscale needs an exact prepared Normalize/Canvas source geometry.')

    def supplied(param):
        return bool(param.get('sourceId') or param.get('sourceKey')) or param.get('value') is not None

    def direct_vace_text_geometry(params, node_id):
        # Saved whole-pipeline Wan VACE text graphs encode their prompts inside
        # Diffusers. External embeddings, conditioning or latents can change
        # the geometry or effective batch and have no direct storage proof.
        external = ('video', 'mask', 'reference_images', 'prompt_embeds', 'negative_prompt_embeds',
                    'latents', 'latents_in', 'latent_binding')
        if any(supplied(params.get(field, {})) for field in external):
            raise ValueError('Direct Wan VACE storage requires text-only inputs and generated latents.')
        prompt = params.get('prompt', {})
        negative = params.get('negative_prompt', {})
        if (any(param.get('sourceId') or param.get('sourceKey') for param in (prompt, negative))
                or not isinstance(prompt.get('value'), str) or not prompt['value'].strip()
                or negative.get('value') is not None and not isinstance(negative['value'], str)):
            raise ValueError('Direct Wan VACE storage requires a single literal prompt and optional negative prompt.')
        if any(params.get(field, {}).get('sourceId') or params.get(field, {}).get('sourceKey')
               for field in ('width', 'height', 'num_frames')):
            raise ValueError('Direct Wan VACE storage requires literal video geometry.')
        width, height, frames = (positive(node_id, field) for field in ('width', 'height', 'num_frames'))
        # The pinned VAE scales spatially by 8 and temporally by 4, with a
        # 2x2 transformer patch. Reject shapes the pipeline would round.
        if max(width, height) > 2048 or width % 16 or height % 16 or frames > 241 or (frames - 1) % 4:
            raise ValueError('Direct Wan VACE storage requires aligned native dimensions and frame count.')
        return width, height, frames

    seen = set()

    def inspect(node_id, port):
        if node_id in seen or len(seen) >= 256:
            raise ValueError('Native video composition is cyclic or exceeds its bounded source inspection.')
        seen.add(node_id)
        current = nodes.get(node_id, {})
        identity = (current.get('module'), current.get('action'), port)
        if identity == ('modules.Video', 'Concatenate', 'output'):
            fields = current['params']
            if set(fields) - {'clips', 'transition_seconds', 'fps'}:
                raise ValueError('Native composition has an unreviewed execution control.')
            for field, minimum, maximum in [('fps', 0, 120), ('transition_seconds', 0, 5)]:
                value = literal(nodes, node_id, field)
                if (type(value) not in (int, float) or not math.isfinite(value)
                        or not minimum <= value <= maximum or field == 'fps' and value == 0):
                    raise ValueError('Native composition needs finite declared FPS and crossfade controls.')
            collection = fields.get('clips', {})
            collection_id = collection.get('sourceId')
            items = nodes.get(collection_id, {})
            if (items.get('module'), items.get('action'), collection.get('sourceKey')) != ('modules.Primitive', 'ToList', 'list'):
                raise ValueError('Native composition requires its explicit ordered shot collection.')
            if not 2 <= len(items['params']) <= 64 or any(name != 'item' and not name.startswith('item>>>') for name in items['params']):
                raise ValueError('Native composition requires two to 64 declared shot inputs.')
            shots = []
            for item in items['params'].values():
                if not item.get('sourceId') or not item.get('sourceKey') or item.get('value') is not None:
                    raise ValueError('Native composition cannot borrow literal or unknown clip geometry.')
                shots.append(inspect(item['sourceId'], item['sourceKey']))
            if len({(s['width'], s['height']) for s in shots}) != 1:
                raise ValueError('Native composition upscale requires consistent declared shot geometry.')
            return {'width': shots[0]['width'], 'height': shots[0]['height'],
                    'frames': sum(s['frames'] for s in shots), 'shotCount': sum(s['shotCount'] for s in shots)}

        rendered = identity == ('modules.DiffusersVideo', 'Generate', 'video_out')
        decoded = identity == ('modules.DiffusersVideo', 'DecodeLatents', 'video_out')
        if not rendered and not decoded:
            raise ValueError('Native video upscale requires a declared native decoder or encoded render output.')
        params = current.get('params', {})
        if decoded:
            if set(params) != {'pipeline', 'output_type', 'latents', 'latent_binding'} or literal(nodes, node_id, 'output_type') != 'pil':
                raise ValueError('Native video upscale requires the unchanged PIL decoder and binding.')
            latent = params['latents']
            denoise_id = latent.get('sourceId')
            denoise = nodes.get(denoise_id, {})
            if ((denoise.get('module'), denoise.get('action'), latent.get('sourceKey')) != ('modules.DiffusersVideo', 'Denoise', 'latents_out')
                    or (params['latent_binding'].get('sourceId'), params['latent_binding'].get('sourceKey')) != (denoise_id, 'latent_binding')):
                raise ValueError('Native video decoder has a different sampler/latent binding.')
        else:
            denoise_id, denoise = node_id, current
            if literal(nodes, node_id, 'output_type') not in (None, 'pil'):
                raise ValueError('Encoded render upscale requires PIL video frames.')
        dp = denoise['params']
        pipeline = dp.get('pipeline', {})
        loader_id = pipeline.get('sourceId')
        loader = nodes.get(loader_id, {})
        pipeline_class = literal(nodes, loader_id, 'pipeline_class') if loader else None
        encode = dp.get('encoded_inputs', {})
        encode_id = encode.get('sourceId')
        direct_vace_text = (rendered and pipeline_class == 'WanVACEPipeline'
                            and literal(nodes, denoise_id, 'mode') == 'text_to_video' and not supplied(encode))
        if not direct_vace_text and (
                (nodes.get(encode_id, {}).get('module'), nodes.get(encode_id, {}).get('action'), encode.get('sourceKey'))
                != ('modules.DiffusersVideo', 'EncodeInputs', 'encoded_inputs')):
            raise ValueError('Native video upscale requires the matching real input encoder.')
        stages = {node_id, denoise_id} if direct_vace_text else {node_id, denoise_id, encode_id}
        owners = [o for o in planned if o['nodeId'] == loader_id and stages <= set(o['consumers'])]
        if len(owners) != 1:
            raise ValueError('Native video upscale has no single accepted generating owner.')
        owner = owners[0]
        for stage_id in stages:
            binding = nodes[stage_id]['params'].get('pipeline', {})
            if (binding.get('sourceId'), binding.get('sourceKey')) != (loader_id, 'pipeline'):
                raise ValueError('Native video stages must share their exact pipeline owner.')
        reviewed = {
            'LTXConditionPipeline': ('LTXVideoPipeline', 'Lightricks/LTX-Video-0.9.8-13B-distilled', '7c64400e1861cc0d7b98d570a1926d5408ec60cd'),
            'WanPipeline': ('WanVideoPipeline', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers', '0fad780a534b6463e45facd96134c9f345acfa5b'),
            'WanVideoToVideoPipeline': ('WanVideoPipeline', 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers', '0fad780a534b6463e45facd96134c9f345acfa5b'),
            'WanVACEPipeline': ('WanVACEPipeline', 'Wan-AI/Wan2.1-VACE-1.3B-diffusers', 'ec4d2cb062b548996b179d493fdd05340de702a1'),
        }
        target = reviewed.get(pipeline_class)
        if ((loader.get('module'), loader.get('action'), pipeline.get('sourceKey')) != ('modules.DiffusersVideo', 'LoadPipeline', 'pipeline')
                or target is None or (owner['modelType'], owner['repository']) != target[:2]
                or literal(nodes, loader_id, 'model_id') != {'source': 'hub', 'value': target[1]}
                or literal(nodes, loader_id, 'revision') != target[2]
                or rendered != (pipeline_class == 'WanVACEPipeline')):
            raise ValueError('Native video storage differs from its reviewed immutable generating profile.')
        for stage_id in [denoise_id] if direct_vace_text else [encode_id, denoise_id]:
            if literal(nodes, stage_id, 'mode') != owner['mode']:
                raise ValueError('Native video stages have different selected tasks.')
        for field in ['num_videos_per_prompt', 'num_images_per_prompt', 'batch_size']:
            value = literal(nodes, denoise_id, field)
            if value is not None and (type(value) is not int or value != 1):
                raise ValueError('Native video upscale has no multi-video batch storage contract.')
        if direct_vace_text:
            width, height, frames = direct_vace_text_geometry(dp, denoise_id)
        else:
            if any(supplied(dp.get(field, {})) for field in ('latents', 'latents_in', 'latent_binding')):
                raise ValueError('Externally supplied latents have no native video upscale geometry authority.')
            width, height, frames = (positive(denoise_id, field) for field in ('width', 'height', 'num_frames'))
        if owner['mode'] in {'video_to_video', 'video_color_edit', 'control_to_video', 'video_outpaint'}:
            if prepared_geometry(encode_id) != (width, height, frames):
                raise ValueError('Native video source geometry differs from the forecast sampler geometry.')
        geometry = {'width': width, 'height': height, 'frames': frames, 'shotCount': 1}
        if direct_vace_text:
            geometry['sourceKind'] = 'direct-wan-vace-text-to-video'
        return geometry

    return inspect(source_id, source_port)
