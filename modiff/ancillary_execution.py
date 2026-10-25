"""Private graph-scope declarations; no admission, release or Apply authority."""
from copy import deepcopy
import hashlib
import json
import math
import re

PRIVATE_PARENTS = {
    f'{variant}:{mode}:prompt-preview-authoring:v1': {
        'specId': f'{variant}:modular:{mode}:v1', 'task': task,
        'profileId': f'{variant}:modular', 'pipelineClass': pipeline,
    }
    for variant, pipeline in (('flux3-klein', 'Flux3DistilledModularPipeline'),
                              ('flux3-klein-base', 'Flux3ModularPipeline'))
    for mode, task in (('text-to-image', 'text_to_image'), ('edit-image', 'edit_image'))
}
# A new opt-in identity keeps every legacy plain declaration/hash unchanged.
for _variant in ('flux3-klein', 'flux3-klein-base'):
    for _mode in ('text-to-image', 'edit-image'):
        PRIVATE_PARENTS[f'{_variant}:{_mode}:prompt-preview-authoring:v2'] = {
            **PRIVATE_PARENTS[f'{_variant}:{_mode}:prompt-preview-authoring:v1'],
            'layoutPolicy': 'preserve-authored-v1',
        }
PRIVATE_PARENTS['flux3-klein:text-to-image:prompt-preview-fp32-authoring:v1'] = {
    **PRIVATE_PARENTS['flux3-klein:text-to-image:prompt-preview-authoring:v1'],
    'providerRecipe': 'fp32-none',
}
PRIVATE_PARENTS['flux3-klein:text-to-image:prompt-preview-fp32-authoring:v2'] = {
    **PRIVATE_PARENTS['flux3-klein:text-to-image:prompt-preview-fp32-authoring:v1'],
    'layoutPolicy': 'preserve-authored-v1',
}
# Explicit composition identities never change the ten existing declarations.
PRIVATE_PARENTS['flux3-klein:edit-image:prompt-preview-authoring:v3'] = {
    **PRIVATE_PARENTS['flux3-klein:edit-image:prompt-preview-authoring:v1'],
    'layoutPolicy': 'append-details-preserve-layout-v1',
}
PRIVATE_PARENTS['flux3-klein:text-to-image:prompt-preview-fp32-authoring:v3'] = {
    **PRIVATE_PARENTS['flux3-klein:text-to-image:prompt-preview-fp32-authoring:v1'],
    'layoutPolicy': 'append-details-preserve-layout-v1',
}
# Base opts into the same generic composition without changing older recipes.
PRIVATE_PARENTS['flux3-klein-base:edit-image:prompt-preview-authoring:v3'] = {
    **PRIVATE_PARENTS['flux3-klein-base:edit-image:prompt-preview-authoring:v1'],
    'layoutPolicy': 'append-details-preserve-layout-v1',
}
PRIVATE_PARENTS['flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v3'] = {
    **PRIVATE_PARENTS['flux3-klein-base:text-to-image:prompt-preview-authoring:v1'],
    'providerRecipe': 'fp32-none',
    'layoutPolicy': 'append-details-preserve-layout-v1',
}
# Raw wording uses a distinct fixed-precision parent; all fourteen older DTOs
# retain their exact data, order and hashes.
PRIVATE_PARENTS['flux3-klein-base:text-to-image:prompt-preview-fp32-authoring:v1'] = {
    **PRIVATE_PARENTS['flux3-klein-base:text-to-image:prompt-preview-authoring:v1'],
    'providerRecipe': 'fp32-none',
}
PARENT_KEYS = ('authoringRecipeId', 'baseSpecId', 'baseSpecContentHash',
               'executionProfileId', 'pipelineClass', 'task')
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:-]{0,127}')
FIELD = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,63}')


class AncillaryExecutionError(ValueError):
    pass


def invalid(message):
    return AncillaryExecutionError('Invalid ancillary execution purpose: ' + message)


def record(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise invalid('missing or unknown keys.')
    return value


def text(value, maximum=512, pattern=None):
    if type(value) is not str or not value:
        raise invalid('identity must be nonempty text.')
    try:
        raw = value.encode('utf-8')
    except UnicodeError as error:
        raise invalid('identity requires valid Unicode.') from error
    if len(raw) > maximum or any(ord(char) < 32 for char in value) or pattern and not pattern.fullmatch(value):
        raise invalid('identity exceeds its exact contract.')
    return value


def bounded(value, maximum):
    entries = 0

    def visit(item, depth):
        nonlocal entries
        entries += 1
        if entries > 8192 or depth > 16:
            raise invalid('structured selection exceeds its limit.')
        kind = type(item)
        if kind is str:
            try:
                if len(item) > maximum or len(item.encode('utf-8')) > maximum:
                    raise invalid('structured text exceeds its limit.')
            except UnicodeError as error:
                raise invalid('structured text requires valid Unicode.') from error
        elif item is None or kind is bool:
            return
        elif kind is int:
            if abs(item) > 2**53 - 1:
                raise invalid('integer exceeds the JSON contract.')
        elif kind is float:
            if not math.isfinite(item):
                raise invalid('nonfinite number.')
        elif kind is list:
            for child in item:
                visit(child, depth + 1)
        elif kind is dict:
            if len(item) > 2048:
                raise invalid('mapping exceeds its limit.')
            for key, child in item.items():
                if type(key) is not str:
                    raise invalid('mapping keys must be text.')
                visit(key, depth + 1)
                visit(child, depth + 1)
        else:
            raise invalid('plain JSON values are required.')
    visit(value, 0)
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(raw) > maximum:
        raise invalid('serialized selection exceeds its limit.')


def parse_execution_purpose(value):
    """Parse only a claimed scope; never execute a supplier or infer authority."""
    bounded(value, 128 * 1024)
    record(value, ('schemaVersion', 'kind', 'bindings', 'target'))
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1 or value['kind'] not in ('workflow', 'explicit_preview'):
        raise invalid('unsupported purpose.')
    bindings = value['bindings']
    if type(bindings) is not list or not 1 <= len(bindings) <= 4 or value['kind'] == 'explicit_preview' and len(bindings) != 1:
        raise invalid('wrong number of selected bindings.')
    seen = set()
    for binding in bindings:
        record(binding, ('contractId', 'contentHash', 'parent', 'parentRoleNodes', 'roleNodes'))
        identity = text(binding['contractId'], 128, ID)
        text(binding['contentHash'], 64, re.compile(r'studio-ancillary-v1-[0-9a-f]{8}'))
        if identity in seen:
            raise invalid('duplicate ancillary binding.')
        seen.add(identity)
        parent = record(binding['parent'], (*PARENT_KEYS, 'authoringContentHash'))
        for key in PARENT_KEYS:
            text(parent[key], 128, ID)
        text(parent['baseSpecContentHash'], 64, re.compile(r'studio-spec-v1-[0-9a-f]{8}'))
        text(parent['authoringContentHash'], 64, re.compile(r'studio-authoring-v1-[0-9a-f]{8}'))
        identities = set()
        for name in ('parentRoleNodes', 'roleNodes'):
            mapping = binding[name]
            if type(mapping) is not dict or not 1 <= len(mapping) <= 2048:
                raise invalid('role map is missing or excessive.')
            for role, node_id in mapping.items():
                text(role, 128, ID)
                text(node_id)
                if node_id in identities:
                    raise invalid('role IDs collide.')
                identities.add(node_id)
    if value['kind'] == 'workflow':
        if value['target'] is not None:
            raise invalid('ordinary workflow has no Preview target.')
    else:
        target = record(value['target'], ('nodeId', 'outputField'))
        text(target['nodeId'])
        text(target['outputField'], 64, FIELD)
    return deepcopy(value)


def adapter_binding():
    from modiff.prompt_enhancement import _declared_contract
    try:
        adapter, raw = _declared_contract(with_bytes=True)
    except (OSError, ValueError) as error:
        raise invalid('selected adapter descriptor is unavailable or malformed.') from error
    record(adapter, ('schemaVersion', 'id', 'status', 'repository', 'revision', 'transformersVersion',
        'modelClass', 'processorClass', 'maxNewTokens', 'maxReferences', 'systemPromptsSha256',
        'chatTemplateSha256', 'modelWeight', 'public', 'autoEligible', 'templateEligible', 'liveProof'))
    if (type(adapter['schemaVersion']) is not int or adapter['schemaVersion'] != 1
            or adapter['status'] != 'contract_only' or type(adapter['maxNewTokens']) is not int
            or adapter['maxNewTokens'] != 3584 or type(adapter['maxReferences']) is not int
            or adapter['maxReferences'] != 10 or any(adapter[key] is not False
                for key in ('public', 'autoEligible', 'templateEligible', 'liveProof'))):
        raise invalid('adapter is not its current private contract.')
    text(adapter['id'], 128, re.compile(r'[a-z][a-z0-9-]{0,127}'))
    text(adapter['repository'], 256, re.compile(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+'))
    text(adapter['revision'], 40, re.compile(r'[0-9a-f]{40}'))
    return adapter, {'id': adapter['id'], 'declarationSha256': 'sha256:' + hashlib.sha256(raw).hexdigest()}


def adapter_verification(adapter, binding, runtime_inputs, request_inputs, layout_policy=None):
    """Bounded independent expectations, never copied from proposal output.

    Hashes retain their native domains: the file/chat/declaration digests use
    sha256:, while per-mode prompt digests match consumedInput's bare hex field.
    Device, dtype and quantization are choices of the actual provider literals.
    """
    from modiff.prompt_enhancement import SYSTEM_PROMPTS, _json_file
    from modiff.prompt_enhancement_output import PRESERVATION_REQUEST_SUFFIX, APPEND_REQUEST_TEMPLATE
    from modiff.transformers_image_text_resources import image_text_architecture_policy, image_text_nf4_policy
    raw, prompts = _json_file(SYSTEM_PROMPTS, 32 * 1024)
    if (hashlib.sha256(raw).hexdigest() != adapter['systemPromptsSha256']
            or type(prompts) is not dict or set(prompts) != {'t2i', 'i2i', 't2i_bbox', 'i2i_bbox'}
            or any(type(value) is not str or not value for value in prompts.values())):
        raise invalid('reviewed system-prompt verification metadata changed.')
    identifier = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,255}')
    for name in ('modelClass', 'processorClass'):
        text(adapter[name], 256, identifier)
    architecture = image_text_architecture_policy(adapter)
    text(architecture['modelType'], 256, identifier)
    if not 1 <= len(architecture['architectures']) <= 32:
        raise invalid('reviewed architecture verification metadata exceeds its limit.')
    for name in architecture['architectures']:
        text(name, 256, identifier)
    result = {'schemaVersion': 1, 'adapter': deepcopy(binding),
        'runtimeInputs': deepcopy(runtime_inputs), 'requestInputs': deepcopy(request_inputs),
        'supportedPromptModes': ['t2i', 'i2i'],
        'systemPromptsSha256': 'sha256:' + adapter['systemPromptsSha256'],
        'systemPromptSha256ByMode': {mode: hashlib.sha256(value.encode('utf-8')).hexdigest()
                                   for mode, value in prompts.items()},
        'chatTemplateSha256': 'sha256:' + adapter['chatTemplateSha256'],
        'modelReceipt': {'schemaVersion': 1, 'library': 'transformers', 'task': 'image-video-to-text',
            'source': {'kind': 'hub', 'repository': adapter['repository'], 'revision': adapter['revision']},
            'loader': {'preprocessorAutoClass': 'AutoProcessor', 'modelAutoClass': 'AutoModelForImageTextToText'},
            'security': {'localFilesOnly': True, 'trustRemoteCode': False,
                         'safetensorsOnly': True, 'hostedInference': False},
            'runtime': {'transformersVersion': adapter['transformersVersion'],
                'preprocessorClass': adapter['processorClass'], 'modelClass': adapter['modelClass'], **architecture}},
        'quantizationRecipes': [{'mode': 'none'}, image_text_nf4_policy(
            {'source': 'hub', 'value': adapter['repository']}, adapter['revision'], 'bfloat16', 'cuda:0')]}
    if layout_policy == 'append-details-preserve-layout-v1':
        result['layoutRequestSuffixSha256'] = 'sha256:' + hashlib.sha256(APPEND_REQUEST_TEMPLATE.encode('utf-8')).hexdigest()
    elif layout_policy is not None:
        if layout_policy != 'preserve-authored-v1':
            raise invalid('unsupported authored-layout preservation policy.')
        from modiff.prompt_enhancement import request_system_prompt
        result.update(supportedPromptModes=['t2i', 'i2i', 't2i_bbox', 'i2i_bbox'],
            systemPromptSha256ByMode={mode: hashlib.sha256(request_system_prompt(prompts, mode).encode('utf-8')).hexdigest()
                                     for mode in prompts},
            layoutRequestSuffixSha256='sha256:' + hashlib.sha256(PRESERVATION_REQUEST_SUFFIX.encode('utf-8')).hexdigest())
    return result


def params_for(modules, key):
    module, action = key.rsplit('.', 1)
    definition = modules.get(module, {}).get(action) if type(modules) is dict else None
    params = definition.get('params') if type(definition) is dict else None
    if type(params) is not dict:
        raise invalid('registered role is unavailable.')
    return params


def field(params, name, *, output=False):
    item = params.get(name)
    if type(item) is not dict or (item.get('display') == 'output') != output:
        raise invalid('declared field or direction is unavailable.')
    return item


def selected_input_bindings(role, params, names):
    """Bind generic selected controls to their actual registered input fields."""
    if len(set(names.values())) != len(names):
        raise invalid('selected input-control bindings are ambiguous.')
    expected = {'modelField': 'string', 'revisionField': 'string', 'dtypeField': 'string',
        'deviceField': 'string', 'quantizationField': 'string', 'operationField': 'string',
        'maxNewTokensField': 'int', 'minNewTokensField': 'int', 'maxInputTokensField': 'int',
        'maxImagePixelsField': 'int', 'field': 'int'}
    for label, name in names.items():
        spec = field(params, name)
        if label not in expected or spec.get('type') != expected[label]:
            raise invalid('selected input-control binding differs from its registered scalar type.')
        if spec.get('display') in ('input', 'output', 'signal') or str(spec.get('display', '')).startswith('ui_'):
            raise invalid('selected input-control binding requires a literal registered control.')
    return {'role': role, **names}


def selected_field_overrides(proposal):
    """Existing selected recipe presentation, without registry/default rewrites."""
    from modules.HuggingFaceTransformers.main import prompt_enhancement_operation_schema
    fields, values = prompt_enhancement_operation_schema()
    if fields.get('enhancement_max_new_tokens') != {'hidden': False} or values != {'operation': 'enhance_prompt'}:
        raise invalid('selected enhancement presentation differs from its declared schema.')
    result = [('proposal', 'enhancement_max_new_tokens', deepcopy(fields['enhancement_max_new_tokens'])),
              ('proposal', 'max_new_tokens', {'hidden': True})]
    for _role, name, patch in result:
        selected_input_bindings('proposal', proposal, {'field': name})
        if type(patch) is not dict or set(patch) != {'hidden'} or type(patch['hidden']) is not bool:
            raise invalid('selected field presentation permits only an exact hidden boolean.')
    return result


def ancillary_operations_for_parent(modules, payload, selection, role_params):
    """Build one optional, hash-covered declaration on an exact private parent."""
    from modiff.studio_execution_specs import _hash_string, _stable_json
    expected = PRIVATE_PARENTS.get(payload.get('id'))
    if expected is None or (payload['baseSpecId'], selection['pipelineClass'], selection['task'], selection['executionProfileId']) != (
            expected['specId'], expected['pipelineClass'], expected['task'], expected['profileId']):
        raise invalid('wrong selected private parent.')
    field(role_params['prompt'], 'prompt')
    field(role_params['prompt'], 'layout_conditioning')
    provider = params_for(modules, 'modules.HuggingFaceTransformers.LoadImageTextToTextModel')
    proposal = params_for(modules, 'modules.HuggingFaceTransformers.GenerateImageVideoText')
    if field(provider, 'model', output=True).get('type') != field(proposal, 'model').get('type'):
        raise invalid('provider model ports are incompatible.')
    for name in ('prompt', 'layout_conditioning', 'images', 'operation'):
        field(proposal, name)
    if field(proposal, 'proposal_summary', output=True).get('type') != 'string':
        raise invalid('proposal requires its static string output.')
    reference = None
    if expected['task'] == 'edit_image':
        if field(role_params['loadImage'], 'image', output=True).get('type') != proposal['images'].get('type'):
            raise invalid('parent image port is incompatible.')
        reference = {'role': 'loadImage', 'field': 'image'}
    _adapter, binding = adapter_binding()
    runtime_inputs = selected_input_bindings('provider', provider, {
        'modelField': 'model_id', 'revisionField': 'revision', 'dtypeField': 'dtype',
        'deviceField': 'device', 'quantizationField': 'quantization_mode'})
    request_inputs = selected_input_bindings('proposal', proposal, {
        'operationField': 'operation', 'maxNewTokensField': 'enhancement_max_new_tokens',
        'minNewTokensField': 'min_new_tokens', 'maxInputTokensField': 'max_input_tokens',
        'maxImagePixelsField': 'max_image_pixels'})
    parent = {'authoringRecipeId': payload['id'], 'baseSpecId': payload['baseSpecId'],
        'baseSpecContentHash': payload['baseSpecContentHash'], 'executionProfileId': expected['profileId'],
        'pipelineClass': expected['pipelineClass'], 'task': expected['task']}
    result = {'schemaVersion': 1, 'id': payload['id'] + ':preview', 'parent': parent,
        'executionPurpose': 'explicit_preview',
        'roles': [('provider', 'modules.HuggingFaceTransformers.LoadImageTextToTextModel', -720, -400),
                  ('proposal', 'modules.HuggingFaceTransformers.GenerateImageVideoText', -360, -400)],
        'edges': [('provider', 'model', 'proposal', 'model')],
        'bindings': {'owner': {'role': 'prompt', 'promptField': 'prompt', 'layoutField': 'layout_conditioning', 'referenceSource': reference},
            'request': {'role': 'proposal', 'promptField': 'prompt', 'layoutField': 'layout_conditioning', 'imagesField': 'images'},
            'output': {'role': 'proposal', 'field': 'proposal_summary'}},
        'initialValues': [('provider', runtime_inputs['modelField'], {'source': 'hub', 'value': _adapter['repository']}),
            ('provider', runtime_inputs['revisionField'], _adapter['revision']),
            ('proposal', request_inputs['operationField'],
             'enhance_prompt_append_details' if expected.get('layoutPolicy') == 'append-details-preserve-layout-v1' else 'enhance_prompt')],
        'fieldOverrides': selected_field_overrides(proposal), 'adapter': binding,
        'verification': adapter_verification(_adapter, binding, runtime_inputs, request_inputs, expected.get('layoutPolicy')),
        'sourceRegions': False}
    if 'layoutPolicy' in expected:
        result['layoutPolicy'] = expected['layoutPolicy']
    if expected['task'] == 'text_to_image':
        fp32 = expected.get('providerRecipe') == 'fp32-none'
        result['initialValues'].extend([
            (runtime_inputs['role'], runtime_inputs['dtypeField'], 'float32' if fp32 else 'bfloat16'),
            (runtime_inputs['role'], runtime_inputs['deviceField'], 'cuda:0'),
            (runtime_inputs['role'], runtime_inputs['quantizationField'], 'none' if fp32 else 'bnb_4bit'),
            (request_inputs['role'], request_inputs['maxInputTokensField'], 4096),
        ])
    result['contentHash'] = 'studio-ancillary-v1-' + _hash_string(_stable_json(result))
    result = json.loads(json.dumps(result, allow_nan=False))
    bounded(result, 32 * 1024)
    return [result]


def resolve_parent(modules, identity):
    from modiff.studio_execution_specs import resolve_studio_operation_authoring_recipe
    definition = PRIVATE_PARENTS.get(identity)
    if definition is None:
        raise invalid('parent authoring recipe is not privately declared.')
    return resolve_studio_operation_authoring_recipe(modules, {
        'pipelineClass': definition['pipelineClass'], 'task': definition['task'],
        'executionProfileId': definition['profileId'], 'authoringRecipeId': identity}, {'nodes': [], 'edges': []})


def native_parent_association(graph, binding, modules):
    """Bind current native stage context; successful dispatch grants ownership.

    Resolve only the existing native Block interface metadata. This does not
    lower, execute or prune a graph, and client authorities are not consulted.
    """
    from modiff.block_definition_v2 import validate_block_instance_v2
    snapshot = graph.get('runtimeHints', {}).get('workflowSnapshot')
    roots = snapshot.get('nodes') if type(snapshot) is dict else None
    edges = snapshot.get('edges', []) if type(snapshot) is dict else None
    if (type(roots) is not list or len(roots) > 2048 or type(edges) is not list
            or len(edges) > 32768):
        return None
    roles = binding['parentRoleNodes']
    parent = resolve_parent(modules, binding['parent']['authoringRecipeId'])
    declared = {role: tuple(key.rsplit('.', 1)) for role, key, _x, _y in parent['roles']}
    if set(roles) != set(declared) or len(set(roles.values())) != len(roles):
        return None
    documents, blocks, semantic, root_for = {}, {}, {}, {}

    def projected(root_id, node_id):
        return (f"block-v2-node:{len(root_id.encode('utf-16-le')) // 2}:{root_id}:"
                f"{len(node_id.encode('utf-16-le')) // 2}:{node_id}")

    for root in roots:
        if (type(root) is not dict or type(root.get('id')) is not str
                or root['id'] in documents or type(root.get('data')) is not dict):
            return None
        root_id, data = root['id'], root['data']
        documents[root_id] = root
        if type(data.get('blockInstanceV2')) is not dict:
            continue
        try:
            instance = validate_block_instance_v2(data['blockInstanceV2'])
        except (ValueError, TypeError):
            return None
        if instance['instanceId'] != root_id:
            return None
        blocks[root_id] = instance
        for node in instance['effectiveGraph']['nodes']:
            node_id = projected(root_id, node['nodeId'])
            if node_id in semantic:
                return None
            semantic[node_id], root_for[node_id] = node, root_id
    if set(documents) & set(semantic):
        return None  # An ordinary ID cannot impersonate a projected native leaf.
    selected_roots, ordinary = set(), []
    for role, node_id in roles.items():
        if node_id in semantic:
            data = semantic[node_id]['data']
            selected_roots.add(root_for[node_id])
        else:
            node = documents.get(node_id)
            if (type(node) is not dict or node_id in blocks or node.get('type') != 'custom'):
                return None
            data = node['data']
            ordinary.append({'id': node_id, 'module': data.get('module'), 'action': data.get('action')})
        if (data.get('module'), data.get('action')) != declared[role]:
            return None
    if roles.get('models') not in semantic:
        return None  # No all-flat or arbitrary custom-owner association.

    def endpoints(root_id, handle, output):
        if type(handle) is not str:
            return None
        if root_id not in blocks:
            return [(root_id, handle)] if root_id in documents else None
        instance = blocks[root_id]
        ports = instance['effectiveInterface']['boundary']['outputs' if output else 'inputs']
        matched = [port for port in ports if port['portId'] == handle]
        if len(matched) != 1:
            return None
        bindings = [matched[0]['binding'], *matched[0].get('mirrorBindings', [])]
        result = [(projected(root_id, item['nodeId']), item['fieldOrPortId']) for item in bindings]
        if len(set(result)) != len(result) or any(node_id not in semantic for node_id, _field in result):
            return None
        return result

    selected_ids, wires = set(roles.values()), set()
    contract = next((item for item in parent['ancillaryOperations']
        if item['id'] == binding['contractId'] and item['contentHash'] == binding['contentHash']), None)
    if contract is None:
        return None
    children = binding['roleNodes']
    child_keys = {role: tuple(key.rsplit('.', 1)) for role, key, _x, _y in contract['roles']}
    if (set(children) != set(child_keys) or len(set(children.values())) != len(children)
            or set(children.values()) & selected_ids):
        return None
    reference = contract['bindings']['owner']['referenceSource']
    request = contract['bindings']['request']
    child_edges, media_edges, child_ids = set(), set(), set()
    if reference is not None:
        # Actual Enable adopts these declared leaves into Inputs. Its public
        # image boundaries legitimately feed both image encoding and the
        # optional proposal; no other incident child wire is permitted.
        child_ids = set(children.values())
        child_edges = {(children[source], output, children[target], input_)
                       for source, output, target, input_ in contract['edges']}
        media_edges = {(roles[reference['role']], reference['field'],
                        children[request['role']], request['imagesField'])}
        for role, node_id in children.items():
            if node_id in semantic:
                data = semantic[node_id]['data']
                selected_roots.add(root_for[node_id])
            else:
                node = documents.get(node_id)
                if type(node) is not dict or node_id in blocks or node.get('type') != 'custom':
                    return None
                data = node['data']
                ordinary.append({'id': node_id, 'module': data.get('module'), 'action': data.get('action')})
            if (data.get('module'), data.get('action')) != child_keys[role]:
                return None
    lora_nodes = {node_id for node_id, node in semantic.items()
                  if (node['data'].get('module'), node['data'].get('action')) == ('modules.ModularDiffusers', 'Lora')}
    lora_nodes.update(node_id for node_id, node in documents.items()
        if node_id not in blocks and node.get('type') == 'custom'
        and (node['data'].get('module'), node['data'].get('action')) == ('modules.ModularDiffusers', 'Lora'))
    addressed = selected_ids | lora_nodes | child_ids
    containers = selected_roots | {root_for[node_id] for node_id in lora_nodes if node_id in root_for}
    for root_id in containers:
        for edge in blocks[root_id]['effectiveGraph']['edges']:
            wire = (projected(root_id, edge['sourceNodeId']), edge['sourcePortId'],
                    projected(root_id, edge['targetNodeId']), edge['targetPortId'])
            if wire[0] in addressed or wire[2] in addressed:
                if wire in wires:
                    return None
                wires.add(wire)
    for edge in edges:
        if type(edge) is not dict:
            return None
        if edge.get('source') not in containers | addressed and edge.get('target') not in containers | addressed:
            continue
        sources = endpoints(edge.get('source'), edge.get('sourceHandle'), True)
        targets = endpoints(edge.get('target'), edge.get('targetHandle'), False)
        if sources is None or targets is None:
            return None
        for source, output in sources:
            for target, input_ in targets:
                if source in addressed or target in addressed:
                    wire = (source, output, target, input_)
                    if wire in wires:
                        return None
                    wires.add(wire)
    if child_ids and {wire for wire in wires if wire[0] in child_ids or wire[2] in child_ids} != child_edges | media_edges:
        return None
    expected = {(roles[source], output, roles[target], input_)
                for source, output, target, input_ in parent['edges']}
    if {wire for wire in wires if wire[0] in selected_ids and wire[2] in selected_ids} != expected:
        return None
    support_wires, support_ids = set(), set()
    pending = []
    for wire in wires:
        source, output, target, input_ = wire
        if source in selected_ids or target in selected_ids:
            if wire in expected or wire in media_edges:
                continue
            if (source not in lora_nodes or output != 'lora'
                    or target != roles['models'] or input_ != 'lora_list'):
                return None
            support_wires.add(wire)
            pending.append(source)
    while pending:
        node_id = pending.pop()
        if node_id in support_ids or len(support_ids) >= 32:
            return None
        support_ids.add(node_id)
        incoming = [wire for wire in wires if wire[2] == node_id]
        if len(incoming) > 1:
            return None
        for wire in incoming:
            source, output, _target, input_ = wire
            if source not in lora_nodes or output != 'lora' or input_ != 'previous_loras':
                return None
            support_wires.add(wire)
            pending.append(source)
    joined = selected_ids | support_ids
    if {wire for wire in wires if wire[0] in joined or wire[2] in joined} != expected | support_wires | media_edges:
        return None
    selected_roots.update(root_for[node_id] for node_id in support_ids if node_id in root_for)
    for node_id in sorted(support_ids - set(root_for)):
        ordinary.append({'id': node_id, 'module': 'modules.ModularDiffusers', 'action': 'Lora'})
    # When this is the actual image dispatch, independently join every declared
    # parent-to-parent wire to the flat graph that the existing executor receives.
    # On Preview these owners are intentionally absent; their successful dispatch
    # origin and exact cache inputs must separately match this structural digest.
    nodes = graph.get('nodes', {})
    purpose = graph.get('runtimeHints', {}).get('executionPurpose', {})
    if purpose.get('kind') == 'explicit_preview':
        present = {roles[reference['role']]} if reference is not None else set()
        if set(nodes) != set(children.values()) | present or selected_ids & set(nodes) != present:
            return None
        if any((nodes[node_id].get('module'), nodes[node_id].get('action')) != child_keys[role]
               for role, node_id in children.items()):
            return None
        if any((nodes[node_id].get('module'), nodes[node_id].get('action')) != declared[role]
               for role, node_id in roles.items() if node_id in present):
            return None
        preview_edges = {(children[source], output, children[target], input_)
                         for source, output, target, input_ in contract['edges']} | media_edges
        actual = {(param.get('sourceId'), param.get('sourceKey'), node_id, name)
                  for node_id, node in nodes.items() for name, param in node['params'].items()
                  if param.get('sourceId') is not None}
        if actual != preview_edges:
            return None
    elif purpose.get('kind') == 'workflow':
        if not selected_ids <= set(nodes) or not support_ids <= set(nodes) or any(
                (nodes[node_id].get('module'), nodes[node_id].get('action')) != ('modules.ModularDiffusers', 'Lora')
                for node_id in support_ids):
            return None
        actual = {(param.get('sourceId'), param.get('sourceKey'), node_id, name)
                  for node_id in joined for name, param in nodes[node_id]['params'].items()
                  if param.get('sourceId') is not None}
        if actual != expected | support_wires:
            return None
    else:
        return None
    contexts = []
    for root_id in sorted(selected_roots):
        instance = blocks[root_id]
        structural_nodes = [{'nodeId': node['nodeId'], 'nodeType': node['nodeType'],
            'parentNodeId': node.get('parentNodeId'), 'semanticRole': node.get('semanticRole'),
            'module': node['data'].get('module'), 'action': node['data'].get('action')}
            for node in instance['effectiveGraph']['nodes']]
        interface, route = instance['effectiveInterface'], instance.get('routeSelection', {})
        contexts.append({'instanceId': root_id, 'definitionRef': instance['definitionRef'],
            'nodes': sorted(structural_nodes, key=lambda node: node['nodeId']),
            'edges': sorted(instance['effectiveGraph']['edges'], key=lambda edge: edge['edgeId']),
            'interface': {'boundary': interface['boundary'],
                'controls': [{key: value for key, value in control.items() if key != 'defaultValue'}
                             for control in interface['controls']]},
            'route': {key: route[key] for key in ('routeSetId', 'selectedRouteKey') if key in route}})
    payload = {'roots': contexts, 'ordinary': sorted(ordinary, key=lambda node: node['id']),
        'edges': sorted(expected | support_wires | child_edges | media_edges), 'parent': binding['parent'], 'parentRoleNodes': roles,
        'roleNodes': binding['roleNodes'], 'contractId': binding['contractId'], 'contentHash': binding['contentHash']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def literal(node, name):
    item = node.get('params', {}).get(name)
    if type(item) is not dict or item.get('sourceId') is not None:
        raise invalid('an exact authored literal is required for ' + name + '.')
    return item.get('value')


def check_node(nodes, node_id, key, modules, declared=None):
    node = nodes.get(node_id)
    if type(node) is not dict or (node.get('module'), node.get('action')) != tuple(key.rsplit('.', 1)) or type(node.get('params')) is not dict:
        raise invalid('role differs from its executable node.')
    params = declared if declared is not None else params_for(modules, key)
    extra = set(node['params']) - set(params)
    generic = params_for(modules, key) if extra else {}
    # Native export retains unset generic sockets even when a selected spec
    # omits them. Only an exact inert socket header is harmless: False/0/None
    # values and even empty source fields remain submitted inputs, not absence.
    inert = {name for name in extra if type(generic.get(name)) is dict
             and generic[name].get('display') == 'input'
             and generic[name].get('required') is not True
             and generic[name].get('value') is None
             and generic[name].get('default') is None
             and node['params'][name] == {'display': 'input'}}
    # Ordinary legacy starters include this hidden no-op default. Exact
    # registered specs deliberately omit the unbound field to preserve immutable
    # Block hashes. This single compatibility case accepts no policy override,
    # source, coercion or other scalar default; the selected loader stays native.
    legacy = set()
    name = 'inpaint_compatibility'
    if (declared is not None and key == 'modules.ModularDiffusers.ModelsLoader' and name in extra
            and type(generic.get(name)) is dict and generic[name].get('hidden') is True
            and generic[name].get('default') == 'native'
            and node['params'][name] == {'value': 'native'}):
        legacy.add(name)
    if extra - inert - legacy or any(type(item) is not dict for item in node['params'].values()):
        raise invalid('node has undeclared inputs or an invalid API field shape.')
    if any(type(params.get(name, generic.get(name))) is not dict
           or params.get(name, generic.get(name)).get('display') == 'output'
           for name in node['params']):
        raise invalid('a static output cannot be submitted as an executable input.')
    return node


def check_edge(nodes, roles, edge):
    source, output, target, input_ = edge
    item = nodes[roles[target]]['params'].get(input_)
    if type(item) is not dict or item.get('sourceId') != roles[source] or item.get('sourceKey') != output:
        raise invalid('declared edge differs from the graph.')


def _validate_execution_purpose(graph, modules):
    """Pure assertion on a selected graph; never prune or release anything."""
    hints = graph.get('runtimeHints') if type(graph) is dict else None
    if type(hints) is not dict or 'executionPurpose' not in hints:
        return None
    selection = parse_execution_purpose(hints['executionPurpose'])
    from modiff.workflow_auto_resource import graph_material
    from modiff.studio_execution_specs import STUDIO_EXECUTION_SPEC_DEFINITIONS, _public_spec, _execution_spec_role_params
    nodes = graph_material(graph)['nodes']
    descriptor, _binding = adapter_binding()
    selected = set()
    for binding in selection['bindings']:
        claim = binding['parent']
        parent = resolve_parent(modules, claim['authoringRecipeId'])
        matches = [item for item in parent['ancillaryOperations'] if item['id'] == binding['contractId']]
        if len(matches) != 1:
            raise invalid('unknown ancillary declaration.')
        contract = matches[0]
        if contract['adapter'] != _binding:
            raise invalid('selected adapter declaration changed during validation.')
        if claim != {**contract['parent'], 'authoringContentHash': parent['contentHash']} or binding['contentHash'] != contract['contentHash']:
            raise invalid('stale or mixed selected parent/ancillary identity.')
        owners, children = binding['parentRoleNodes'], binding['roleNodes']
        if set(owners) != {row[0] for row in parent['roles']} or set(children) != {row[0] for row in contract['roles']}:
            raise invalid('role map does not cover its exact declaration.')
        if selected.intersection(children.values()):
            raise invalid('selected ancillary IDs collide.')
        selected.update(children.values())
        if selection['kind'] == 'workflow':
            if set(children.values()).intersection(nodes):
                raise invalid('ordinary Run includes a Preview-only role.')
            public = _public_spec(parent['baseSpecId'], STUDIO_EXECUTION_SPEC_DEFINITIONS[parent['baseSpecId']])
            for role, key, _x, _y in parent['roles']:
                schema = _execution_spec_role_params(public, key, {'params': params_for(modules, key)})
                check_node(nodes, owners[role], key, modules, schema)
            for edge in parent['edges']:
                check_edge(nodes, owners, edge)
            for role, name, source in parent['bindings']:
                if name not in nodes[owners[role]]['params']:
                    raise invalid('parent binding is missing.')
                if source == 'pipelineClass' and literal(nodes[owners[role]], name) != claim['pipelineClass']:
                    raise invalid('parent pipeline selection differs from its exact authoring recipe.')
            continue
        output = contract['bindings']['output']
        if selection['target'] != {'nodeId': children[output['role']], 'outputField': output['field']}:
            raise invalid('target is not the genuine proposal leaf.')
        if graph.get('loops'):
            raise invalid('this Preview declaration has no loop body.')
        for role, key, _x, _y in contract['roles']:
            check_node(nodes, children[role], key, modules)
        for edge in contract['edges']:
            check_edge(nodes, children, edge)
        provider, proposal = nodes[children['provider']], nodes[children['proposal']]
        if (any(param.get('sourceId') is not None for param in provider['params'].values())
                or any(param.get('sourceId') is not None for name, param in proposal['params'].items()
                       if name not in {'model', 'images'})):
            raise invalid('Preview provider and generation controls require authored preallocation literals; only the declared model/images edges may be connected.')
        if literal(provider, 'model_id') != {'source': 'hub', 'value': descriptor['repository']} or literal(provider, 'revision') != descriptor['revision']:
            raise invalid('provider differs from the immutable adapter artifact.')
        if 'pipeline_class' in provider['params'] and literal(provider, 'pipeline_class') != 'AutoModelForImageTextToText':
            raise invalid('wrong official provider factory.')
        if PRIVATE_PARENTS[claim['authoringRecipeId']].get('providerRecipe') == 'fp32-none':
            for role, name, expected_value in contract['initialValues']:
                actual_value = literal(nodes[children[role]], name)
                if type(actual_value) is not type(expected_value) or actual_value != expected_value:
                    raise invalid('declared FP32 Preview input selection changed.')
        expected_operation = ('enhance_prompt_append_details'
            if contract.get('layoutPolicy') == 'append-details-preserve-layout-v1' else 'enhance_prompt')
        if literal(proposal, 'operation') != expected_operation:
            raise invalid('proposal operation is not selected.')
        video = proposal['params'].get('video', {})
        if video.get('sourceId') is not None or video.get('value') is not None:
            raise invalid('Preview accepts images/text only.')
        prompt = literal(proposal, 'prompt')
        if type(prompt) is not str or not prompt.strip() or len(prompt) > 65536 or len(prompt.encode('utf-8')) > 262144:
            raise invalid('authored prompt exceeds its raw bound.')
        from modiff.layout_conditioning import parse_layout_conditioning
        layout = parse_layout_conditioning(literal(proposal, 'layout_conditioning')) if 'layout_conditioning' in proposal['params'] else None
        if layout is not None:
            if contract.get('layoutPolicy') not in ('preserve-authored-v1', 'append-details-preserve-layout-v1'):
                raise invalid('the selected Preview supports plain prompt enhancement only; preserving an authored layout is not supported. Select the explicit structured authoring recipe.')
            if layout['regions'] and contract.get('layoutPolicy') == 'preserve-authored-v1':
                from modiff.prompt_enhancement_output import require_authored_target_layout
                require_authored_target_layout(layout)
        expected_nodes = set(children.values())
        reference = contract['bindings']['owner']['referenceSource']
        images = proposal['params'].get('images', {})
        if reference is None:
            if images.get('sourceId') is not None or images.get('value') is not None:
                raise invalid('text Preview has no image source.')
        else:
            source_id = owners[reference['role']]
            source = check_node(nodes, source_id, 'modules.Image.Load', modules)
            if images.get('sourceId') != source_id or images.get('sourceKey') != reference['field']:
                raise invalid('reference is not the real parent media endpoint.')
            files = literal(source, 'file')
            files = files if type(files) is list else [files]
            if not 1 <= len(files) <= 10 or any(type(item) is not str or not item or len(item.encode('utf-8')) > 4096 for item in files):
                raise invalid('one to ten ordered literal media selections are required.')
            expected_nodes.add(source_id)
        if layout is not None and any(region.get('source', {}).get('referenceIndex', -1) >= (
                0 if reference is None else len(files)) for region in layout['regions']):
            raise invalid('authored source index is outside the declared ordered media context.')
        if contract.get('layoutPolicy') == 'append-details-preserve-layout-v1':
            from modiff.prompt_enhancement_output import append_details_request
            append_details_request(prompt, literal(proposal, 'layout_conditioning'),
                                   0 if reference is None else len(files))
        if set(nodes) != expected_nodes:
            raise invalid('Preview includes an image owner or unrelated executable node.')
    if adapter_binding()[1] != _binding:
        raise invalid('selected adapter declaration changed during validation.')
    return selection


def validate_execution_purpose(graph, modules):
    try:
        return _validate_execution_purpose(graph, modules)
    except AncillaryExecutionError:
        raise
    except (ValueError, TypeError) as error:
        raise invalid(str(error)) from error
