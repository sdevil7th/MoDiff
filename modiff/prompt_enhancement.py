"""Selected prompt-enhancement adapter; ordinary Transformers nodes own execution.

Outputs are proposals. This adapter never edits an authored prompt, loads another
model, authorizes remote code, releases owners or grants resource qualification.
"""
from copy import deepcopy
import hashlib
import json
import re
import sys
from pathlib import Path
import time

DESCRIPTOR = Path(__file__).resolve().parents[1] / 'data/prompt-enhancement-adapters.v1.json'
SYSTEM_PROMPTS = Path(__file__).resolve().parents[1] / 'data/prompt-enhancement-system-prompts.v1.json'


def _json_file(path, maximum):
    with path.open('rb') as handle:
        raw = handle.read(maximum + 1)
    if not 0 < len(raw) <= maximum:
        raise ValueError('Prompt enhancement declaration exceeds its bounded size.')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Prompt enhancement declaration contains a duplicate key.')
            result[key] = value
        return result
    def invalid(_value):
        raise ValueError('Prompt enhancement requires finite JSON.')
    return raw, json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def _declared_contract(*, with_bytes=False):
    """Read the bounded, non-executable selected adapter declaration."""
    declaration, contract = _json_file(DESCRIPTOR, 16 * 1024)
    required = {'schemaVersion', 'id', 'status', 'repository', 'revision', 'transformersVersion', 'modelClass',
                'processorClass', 'maxNewTokens', 'maxReferences', 'systemPromptsSha256', 'chatTemplateSha256',
                'modelWeight', 'public', 'autoEligible', 'templateEligible', 'liveProof'}
    if (type(contract) is not dict or set(contract) != required
            or type(contract.get('maxNewTokens')) is not int or contract['maxNewTokens'] != 3584
            or contract.get('maxReferences') != 10
            or any(contract.get(field) is not False for field in ('public', 'autoEligible', 'templateEligible', 'liveProof'))):
        raise ValueError('The prompt enhancement declaration is malformed or exceeds its reviewed scope.')
    string_patterns = {
        'id': r'[a-z][a-z0-9-]{0,127}',
        'repository': r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',
        'revision': r'[0-9a-f]{40}', 'transformersVersion': r'[0-9]+\.[0-9]+\.[0-9]+',
        'modelClass': r'[A-Za-z_][A-Za-z0-9_]{0,127}',
        'processorClass': r'[A-Za-z_][A-Za-z0-9_]{0,127}',
        'systemPromptsSha256': r'[0-9a-f]{64}', 'chatTemplateSha256': r'[0-9a-f]{64}',
    }
    weight = contract['modelWeight']
    if (type(contract['schemaVersion']) is not int or contract['schemaVersion'] != 1
            or type(contract['maxReferences']) is not int or contract['maxReferences'] != 10
            or any(type(contract[key]) is not str or not re.fullmatch(pattern, contract[key])
                   for key, pattern in string_patterns.items())
            or type(weight) is not dict or set(weight) != {'path', 'byteSize', 'sha256'}
            or weight.get('path') != 'model.safetensors'
            or type(weight.get('byteSize')) is not int or not 0 < weight['byteSize'] < 2**53
            or type(weight.get('sha256')) is not str or not re.fullmatch(r'[0-9a-f]{64}', weight['sha256'])):
        raise ValueError('The prompt enhancement declaration or model receipt is malformed.')
    return (contract, declaration) if with_bytes else contract


def selected_contract(model, processor, receipt):
    contract, prompts, _declaration = _selected_contract(model, processor, receipt)
    return contract, prompts


def _selected_contract(model, processor, receipt):
    """Require the exact immutable declaration and genuine official API types."""
    contract, declaration = _declared_contract(with_bytes=True)
    if type(receipt) is not dict:
        raise ValueError('The prompt enhancement declaration or model receipt is malformed.')
    source = receipt.get('source', {})
    if (contract.get('schemaVersion') != 1 or contract.get('status') != 'contract_only'
            or source != {'kind': 'hub', 'repository': contract['repository'], 'revision': contract['revision']}):
        raise ValueError('Prompt enhancement requires its reviewed immutable model selection.')
    import transformers
    model_class = getattr(transformers, contract['modelClass'], None)
    processor_class = getattr(transformers, contract['processorClass'], None)
    if not isinstance(model_class, type) or not isinstance(processor_class, type):
        raise ValueError('The reviewed prompt enhancement API export is unavailable; inspect the declared runtime.')
    runtime = receipt.get('runtime', {})
    if type(runtime) is not dict:
        raise ValueError('Prompt enhancement requires its reviewed native Transformers API receipt.')
    quantization = runtime.get('quantization')
    if quantization != {'mode': 'none'}:
        from modiff.transformers_image_text_resources import image_text_nf4_policy, validate_image_text_nf4_model
        expected = image_text_nf4_policy({'source': 'hub', 'value': contract['repository']},
            contract['revision'], runtime.get('dtype'), runtime.get('device'))
        if quantization != expected:
            raise ValueError('Prompt enhancement quantization differs from its reviewed native recipe.')
    if (str(transformers.__version__) != contract['transformersVersion']
            or type(model) is not model_class
            or type(processor) is not processor_class
            or runtime.get('modelClass') != contract['modelClass']
            or runtime.get('preprocessorClass') != contract['processorClass']
            or type(getattr(processor, 'chat_template', None)) is not str
            or hashlib.sha256(processor.chat_template.encode()).hexdigest() != contract['chatTemplateSha256']):
        raise ValueError('Prompt enhancement requires the reviewed native Transformers model and processor APIs.')
    if quantization != {'mode': 'none'}:
        validate_image_text_nf4_model(model, expected)
    raw, prompts = _json_file(SYSTEM_PROMPTS, 32 * 1024)
    if (hashlib.sha256(raw).hexdigest() != contract['systemPromptsSha256']
            or type(prompts) is not dict or set(prompts) != {'t2i', 'i2i', 't2i_bbox', 'i2i_bbox'}
            or any(type(value) is not str or not value for value in prompts.values())):
        raise ValueError('The reviewed prompt-enhancement system prompts have changed.')
    return contract, prompts, declaration


def validate_cached_selection(selection, revision, task):
    """Validate selected PUP assets only on explicit Load, before factories.

    Metadata hashes are local observations; the large model is checked for
    exact size/containment only. Its catalog SHA remains publisher provenance.
    """
    # This adapter-local selector does not make unrelated generic loads depend
    # on the private declaration. The declaration must still bind the selection.
    repository = 'black-forest-labs/flux-3-image-klein-pup'
    if task != 'image-video-to-text' or selection != {'source': 'hub', 'value': repository}:
        return None
    contract = _declared_contract()
    from modiff.model_artifact_catalog import catalog_download_inventory, catalog_revision
    from modiff.studio_execution_specs import reviewed_repository_download_files
    from modiff.hf_cache_layout import resolve_snapshot_cache_file
    from utils.huggingface import exact_cached_snapshot_path, _selected_snapshot_file_issues
    if (contract['status'] != 'contract_only' or contract['repository'] != repository
            or revision != contract['revision']
            or catalog_revision(repository) != revision):
        raise ValueError('Prompt enhancement Load requires its reviewed immutable model selection.')
    names = ('chat_template.jinja', 'config.json', 'generation_config.json', 'model.safetensors',
             'preprocessor_config.json', 'processor_config.json', 'system_prompts.json',
             'tokenizer.json', 'tokenizer_config.json')
    selected = reviewed_repository_download_files(repository)
    inventory = catalog_download_inventory(repository, revision, list(names))
    if (selected is None or sorted(selected) != list(names) or inventory is None
            or inventory.get('selection') != 'native-transformers-prompt-enhancement'):
        raise ValueError('Prompt enhancement selected-file inventory is unavailable or has changed.')
    entries = {entry['path']: entry for entry in inventory['files']}
    if (len(entries) != 9 or entries['model.safetensors']['byteSize'] != contract['modelWeight']['byteSize']
            or entries['model.safetensors'].get('sha256') != contract['modelWeight']['sha256']
            or entries['chat_template.jinja'].get('sha256') != contract['chatTemplateSha256']
            or entries['system_prompts.json'].get('sha256') != contract['systemPromptsSha256']):
        raise ValueError('Prompt enhancement selected-file inventory differs from its adapter declaration.')
    metadata = [entry for name, entry in entries.items() if name != 'model.safetensors']
    if (any(type(entry['byteSize']) is not int or not 0 < entry['byteSize'] <= 32 * 1024**2
            or type(entry.get('sha256')) is not str or not re.fullmatch(r'[0-9a-f]{64}', entry['sha256'])
            for entry in metadata) or sum(entry['byteSize'] for entry in metadata) > 32 * 1024**2):
        raise ValueError('Prompt enhancement metadata exceeds its reviewed bounded inventory.')
    snapshot = exact_cached_snapshot_path(repository, revision, marker_file='config.json')
    cache_root = snapshot.parent.parent.parent
    rows = [{'name': name, 'size': entries[name]['byteSize']} for name in names]
    plan = {'revision': revision, 'snapshot_commit': revision, 'snapshot_path': str(snapshot),
            'files': rows, 'validation_files': rows, 'total_file_count': 9,
            'total_bytes': inventory['exactBytes'], 'selection_limited': True}
    issues = _selected_snapshot_file_issues(repository, str(cache_root), plan)
    if issues:
        raise ValueError('Install/Repair the exact prompt enhancement selected-file snapshot before Load: ' + ' '.join(issues[:3]))
    for entry in metadata:
        path = resolve_snapshot_cache_file(snapshot / entry['path'], snapshot=snapshot,
                                           cache_root=cache_root, repository=repository)
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as handle:
            while chunk := handle.read(min(1024 * 1024, entry['byteSize'] - count + 1)):
                count += len(chunk)
                if count > entry['byteSize']:
                    raise ValueError('Prompt enhancement metadata changed while checking its bounded size.')
                digest.update(chunk)
        if count != entry['byteSize'] or digest.hexdigest() != entry['sha256']:
            raise ValueError('Install/Repair prompt enhancement metadata before Load: ' + entry['path'])
    return snapshot


def prepare_request(processor, prompts, authored, images, layout):
    """Render actual ordered media and authored target regions without mutation."""
    from modules.HuggingFaceTransformers import main as nodes
    from modiff.layout_conditioning import parse_layout_conditioning, resolve_layout_prompt
    nodes._bounded_string(authored, field='authored prompt')
    parsed = parse_layout_conditioning(layout)
    if parsed is not None and any('source' in region for region in parsed['regions']):
        raise ValueError('Prompt enhancement source regions need a bound reference-identity context; keep the authored layout unchanged.')
    if parsed is not None and parsed['regions']:
        from modiff.prompt_enhancement_output import require_authored_target_layout
        require_authored_target_layout(parsed)
    if len(images) > 10:
        raise ValueError('The selected prompt enhancement adapter accepts at most ten ordered images.')
    mode = ('i2i' if images else 't2i') + ('_bbox' if parsed and parsed['regions'] else '')
    user = resolve_layout_prompt(authored, parsed) if parsed is not None else authored
    content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': user}]
    messages = [{'role': 'system', 'content': prompts[mode]}, {'role': 'user', 'content': content}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    nodes._bounded_string(rendered, field='rendered prompt', maximum_characters=nodes.MAX_RENDERED_PROMPT_CHARACTERS,
                          maximum_bytes=nodes.MAX_PROMPT_UTF8_BYTES * 2)
    return mode, rendered, parsed


class _OwnedGeneration:
    def __init__(self, node):
        self.node, self.server, self.task, self.graph = node, None, None, None
        self.clock = time.monotonic
        self.started, self.last = self.clock(), None
        from modiff.native_component_lora import _GRAPH_SCOPE
        from modiff.NodeBase import _server
        scope = _GRAPH_SCOPE.get()
        server = _server()
        task = getattr(server, 'current_task', None)
        graph = scope[0] if scope else None
        args = task.get('args') if type(task) is dict else None
        graph_nodes = graph.get('nodes') if type(graph) is dict else None
        source = graph_nodes.get(node.node_id) if type(graph_nodes) is dict else None
        if (type(task) is dict and isinstance(args, (list, tuple)) and args and args[0] is graph
                and type(task.get('task_id')) is str and task['task_id']
                and type(source) is dict and task.get('current_node') == node.node_id
                and (node._sid is None or type(node._sid) is str)
                and (source.get('module'), source.get('action')) == ('modules.HuggingFaceTransformers', 'GenerateImageVideoText')
                and graph.get('sid') == task.get('sid') == node._sid
                and getattr(server, 'node_cache', {}).get(node.node_id) is node):
            self.server, self.task, self.graph = server, task, graph
            self.identity = (task.get('task_id'), task.get('attempt_index'), node._sid, source)

    def check(self):
        if self.task is None:
            return
        task_id, attempt, sid, source = self.identity
        task = self.server.current_task
        args = task.get('args') if type(task) is dict else None
        graph_nodes = self.graph.get('nodes')
        if (task is not self.task or not isinstance(args, (list, tuple)) or not args or args[0] is not self.graph
                or task.get('task_id') != task_id or task.get('attempt_index') != attempt
                or task.get('sid') != sid or self.node._sid != sid or self.graph.get('sid') != sid
                or task.get('current_node') != self.node.node_id
                or self.server.node_cache.get(self.node.node_id) is not self.node
                or type(graph_nodes) is not dict or graph_nodes.get(self.node.node_id) is not source
                or (source.get('module'), source.get('action')) != ('modules.HuggingFaceTransformers', 'GenerateImageVideoText')):
            raise InterruptedError('Prompt enhancement no longer belongs to the current task and node.')
        if task.get('interrupt_requested'):
            raise InterruptedError('Prompt enhancement interrupted by the current task.')

    def progress(self, count, maximum, *, immediate=False):
        self.check()
        now = self.clock()
        if self.task is not None and self.node._sid and (immediate or self.last is None or now - self.last >= .25):
            self.node.progress(-1, phase='encoding', message=f'Generating prompt proposal: {count}/{maximum} new tokens',
                               elapsed_seconds=now - self.started, current_step=count, total_steps=maximum)
            self.last = now
        self.check()


def _fingerprint(parts, observer):
    digest = hashlib.sha256()
    for part in parts:
        observer.check()
        for offset in range(0, len(part), 1024 * 1024):
            observer.check()
            digest.update(part[offset:offset + 1024 * 1024])
    observer.check()
    return 'sha256:' + digest.hexdigest()


def _tensor_fingerprint(tensor, observer):
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != 'cpu' or not tensor.is_contiguous():
        raise ValueError('Prompt enhancement requires bounded contiguous CPU processor tensors before placement.')
    header = json.dumps({'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
                         'byteOrder': sys.byteorder}, sort_keys=True, separators=(',', ':')).encode() + b'\n'
    flat = tensor.detach().reshape(-1).view(torch.uint8)
    # Read bounded slices, not a second complete pixel tensor or its string repr.
    def chunks():
        yield header
        for start in range(0, flat.numel(), 1024 * 1024):
            observer.check()
            yield flat[start:start + 1024 * 1024].numpy().tobytes()
    return _fingerprint(chunks(), observer)


def _consumed_projection(contract, prompts, mode, rendered, images, batch, limit, observer):
    import torch
    sources = []
    source_hashes = []
    for index, image in enumerate(images):
        observer.check()
        header = json.dumps({'mode': image.mode, 'width': image.width, 'height': image.height},
                            sort_keys=True, separators=(',', ':')).encode() + b'\n'
        def rows():
            yield header
            height = max(1, (1024 * 1024) // (image.width * 3))
            for y in range(0, image.height, height):
                observer.check()
                yield image.crop((0, y, image.width, min(y + height, image.height))).tobytes()
        fingerprint = _fingerprint(rows(), observer)
        source_hashes.append(fingerprint)
        sources.append({'index': index, 'mode': image.mode, 'width': image.width,
                        'height': image.height, 'sha256': fingerprint})
    fields = {
        'operation': 'enhance_prompt', 'enhancement_max_new_tokens': limit,
        'prompt_adapter_id': contract['id'], 'system_prompt_mode': mode,
        'system_prompt_sha256': hashlib.sha256(prompts[mode].encode()).hexdigest(),
        'rendered_prompt_sha256': 'sha256:' + hashlib.sha256(rendered.encode()).hexdigest(),
        'rendered_prompt_utf8_bytes': len(rendered.encode()),
        'rendered_input_ids_sha256': _tensor_fingerprint(batch['input_ids'], observer),
        'reference_source_sha256': source_hashes,
    }
    grids = []
    if images:
        grid, pixels = batch.get('image_grid_thw'), batch.get('pixel_values')
        if (not isinstance(grid, torch.Tensor) or grid.dtype not in (torch.int32, torch.int64)
                or list(grid.shape) != [len(images), 3] or grid.device.type != 'cpu'
                or not bool((grid > 0).all()) or not isinstance(pixels, torch.Tensor)
                or pixels.ndim != 2 or pixels.shape[0] != int(grid.prod(dim=1).sum())):
            raise ValueError('Prompt enhancement processor grid must describe the actual ordered image patches.')
        grids = grid.tolist()
        for name, tensor in [('pixel', pixels), ('grid', grid)]:
            fields[f'processed_{name}_sha256'] = _tensor_fingerprint(tensor, observer)
            fields[f'processed_{name}_shape'] = list(tensor.shape)
    elif batch.get('pixel_values') is not None or batch.get('image_grid_thw') is not None:
        raise ValueError('Text-only prompt enhancement unexpectedly produced image tensors.')
    consumed = {'schemaVersion': 1, 'renderedPrompt': rendered, 'references': sources,
                'processedImageGrid': grids, 'fingerprints': dict(fields)}
    if len(json.dumps(consumed, ensure_ascii=True).encode()) > 256 * 1024:
        raise ValueError('Prompt enhancement consumed-input projection exceeds its bounded size.')
    return fields, consumed


def preflight_processor_inputs(processor, kwargs, observer):
    """Explicit Run-only CPU preflight; no model factory or shared batch cache."""
    import transformers
    from modules.HuggingFaceTransformers import main as nodes
    from modiff.prompt_enhancement_summary import authored_request
    from modiff.transformers_image_text_resources import enforce_image_text_input_bounds
    contract = _declared_contract()
    processor_class = getattr(transformers, contract['processorClass'], None)
    if (str(transformers.__version__) != contract['transformersVersion']
            or not isinstance(processor_class, type) or type(processor) is not processor_class
            or type(getattr(processor, 'chat_template', None)) is not str
            or hashlib.sha256(processor.chat_template.encode()).hexdigest() != contract['chatTemplateSha256']):
        raise ValueError('Prompt enhancement preallocation requires its immutable official processor contract.')
    raw, prompts = _json_file(SYSTEM_PROMPTS, 32 * 1024)
    if hashlib.sha256(raw).hexdigest() != contract['systemPromptsSha256']:
        raise ValueError('Prompt enhancement system prompts changed before preallocation.')
    authored = kwargs.get('prompt')
    authored_request(authored, kwargs.get('layout_conditioning'))
    if kwargs.get('video') is not None or kwargs.get('use_chat_template', True) is not True:
        raise ValueError('Prompt enhancement preallocation requires its image/text chat contract.')
    images, _pixels = nodes._normalized_optional_images(kwargs.get('images'))
    limits = enforce_image_text_input_bounds(kwargs, images)
    observer.check()
    mode, rendered, _layout = prepare_request(processor, prompts, authored, images, kwargs.get('layout_conditioning'))
    limit = nodes._bounded_int(kwargs.get('enhancement_max_new_tokens'), field='enhancement_max_new_tokens',
        default=contract['maxNewTokens'], minimum=1, maximum=contract['maxNewTokens'])
    controls = nodes._generation_controls({**kwargs, 'max_new_tokens': limit}, maximum_new_tokens=contract['maxNewTokens'])
    if controls['do_sample'] is not False or controls['num_beams'] != 1 or controls['repetition_penalty'] != 1.0:
        raise ValueError('Prompt enhancement preallocation requires its exact greedy recipe.')
    processor_kwargs = {'text': rendered, 'return_tensors': 'pt', **({'images': images} if images else {})}
    processed = processor(**processor_kwargs)
    observer.check()
    if not hasattr(processed, 'values') or any(getattr(value, 'device', None) is not None
            and value.device.type != 'cpu' for value in processed.values()):
        raise ValueError('Prompt enhancement preallocation must remain on CPU.')
    batch = nodes._batch_to_device(processed, device='cpu')
    limits = enforce_image_text_input_bounds(kwargs, images, batch,
        patch_size=getattr(processor.image_processor, 'patch_size', None))
    fields, _consumed = _consumed_projection(contract, prompts, mode, rendered, images, batch, limit, observer)
    observer.check()
    return fields, limits


def generate(node, model, processor, receipt, kwargs):
    """Run the reviewed adapter through official generate, retaining a proposal."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from modules.HuggingFaceTransformers import main as nodes
    observer = _OwnedGeneration(node)
    observer.check()
    contract, prompts, declaration = _selected_contract(model, processor, receipt)
    from modiff.prompt_enhancement_summary import BoundSummary, authored_request
    authored = kwargs.get('prompt')
    authored_request(authored, kwargs.get('layout_conditioning'))
    summary = (BoundSummary(observer, declaration, contract, authored, kwargs.get('layout_conditioning'))
               if observer.task is not None else None)
    if kwargs.get('video') is not None:
        raise ValueError('Prompt enhancement accepts images and text only; video remains on the standard operation.')
    if kwargs.get('use_chat_template', True) is not True:
        raise ValueError('The selected prompt enhancement adapter requires its reviewed chat template.')
    images, pixels = nodes._normalized_optional_images(kwargs.get('images'))
    from modiff.transformers_image_text_resources import enforce_image_text_input_bounds
    enforce_image_text_input_bounds(kwargs, images)
    observer.check()
    mode, rendered, layout = prepare_request(processor, prompts, authored, images, kwargs.get('layout_conditioning'))
    limit = nodes._bounded_int(kwargs.get('enhancement_max_new_tokens'), field='enhancement_max_new_tokens',
                               default=contract['maxNewTokens'], minimum=1, maximum=contract['maxNewTokens'])
    controls = nodes._generation_controls({**kwargs, 'max_new_tokens': limit}, maximum_new_tokens=contract['maxNewTokens'])
    if controls['do_sample'] is not False or controls['num_beams'] != 1 or controls['repetition_penalty'] != 1.0:
        raise ValueError('The reviewed prompt enhancement recipe requires greedy decoding, one beam and no repetition penalty.')
    observer.check()
    observer.progress(0, limit, immediate=True)
    processor_kwargs = {'text': rendered, 'return_tensors': 'pt'}
    if images:
        processor_kwargs['images'] = images
    observer.check()
    processed = processor(**processor_kwargs)
    observer.check()
    if not hasattr(processed, 'values') or any(getattr(value, 'device', None) is not None
            and value.device.type != 'cpu' for value in processed.values()):
        raise ValueError('Prompt enhancement processor results must remain on CPU until input capture.')
    batch = nodes._batch_to_device(processed, device='cpu')
    input_limits = enforce_image_text_input_bounds(kwargs, images, batch,
        patch_size=getattr(processor.image_processor, 'patch_size', None))
    fields, consumed = _consumed_projection(contract, prompts, mode, rendered, images, batch, limit, observer)
    if summary is not None:
        summary.observe_inputs(consumed, controls)
    from modiff.ancillary_owner_lifecycle import check_preallocation_inputs
    check_preallocation_inputs(node, processor, fields, input_limits)
    observer.check()
    batch = nodes._batch_to_device(batch, device=receipt['runtime']['device'])
    input_ids = batch['input_ids']
    input_tokens = nodes._shape(input_ids)[1]
    observer.check()
    class OwnedStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **_kwargs):
            observer.progress(max(0, int(input_ids.shape[1]) - input_tokens), limit)
            return False
    controls['stopping_criteria'] = StoppingCriteriaList([OwnedStop()])
    node.record_generation_inputs({**fields, **{key: value for key, value in kwargs.items() if key not in fields},
                                   'prompt': authored})
    def native_generate():
        from modiff.ancillary_owner_lifecycle import before_image_text_generation
        before_image_text_generation(model, batch, controls)
        return model.generate(**batch, **controls)
    with torch.inference_mode():
        sequences = nodes._output_sequences(node.mm_exec(
            native_generate, receipt['runtime']['device'], models=[model]))
    observer.check()
    output_tokens = nodes._shape(sequences)[1]
    if output_tokens < input_tokens or not nodes._token_prefix_matches(sequences, input_ids, input_tokens):
        raise RuntimeError('Prompt enhancement did not preserve its input token prefix; no prompt can be applied.')
    count = output_tokens - input_tokens
    if not 0 < count <= limit:
        raise RuntimeError('Prompt enhancement returned an empty or excessive generated token sequence.')
    text = nodes._decode(processor, sequences[0, input_tokens:])
    if not text:
        raise RuntimeError('Prompt enhancement returned an empty proposal.')
    observer.progress(count, limit, immediate=True)
    result = {'schemaVersion': 1, 'task': 'prompt-enhancement', 'text': text,
              'authoredPrompt': authored, 'applyRequired': True, 'applicationAllowed': not mode.endswith('_bbox'),
              'semanticFidelityVerified': False,
              'inputTokens': input_tokens, 'generatedTokens': count, 'inputImageCount': len(images),
              'mediaPixels': pixels, 'systemPromptMode': mode,
              'systemPromptSha256': hashlib.sha256(prompts[mode].encode()).hexdigest(),
              'adapter': contract['id'], 'modelReceipt': receipt, 'consumedInput': consumed,
              'layoutConditioning': deepcopy(layout),
              'layoutOutputValidated': False if mode.endswith('_bbox') else None,
              'finishReason': 'length' if count == limit else 'stop'}
    # Structured generated output stays diagnostic until a separately reviewed
    # parser binds every returned box/id/source to the authored layout.
    if mode.endswith('_bbox'):
        from modiff.prompt_enhancement_output import validate_target_layout_output
        try:
            proposal = validate_target_layout_output(text, layout, mode=mode, image_count=len(images))
        except ValueError as error:
            result['applicationBlockedReason'] = str(error)
        else:
            result.update(layoutOutputValidated=True, applicationAllowed=True, structuredProposal=proposal)
        # Even validated structured content needs a paired atomic UI Apply.
        # Never auto-feed a caption/JSON suffix into Encode Prompt's own layout.
        return {'text': None, 'result': result, 'proposal_summary': summary.finish(result, controls) if summary else None}
    return {'text': text, 'result': result, 'proposal_summary': summary.finish(result, controls) if summary else None}
