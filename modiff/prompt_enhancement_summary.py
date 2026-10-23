"""Bounded diagnostic producer transport, without recipe or Apply authority."""
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
import stat
import sys
from types import ModuleType
import weakref
import hashlib
import json
import math
import re
import unicodedata

from modiff.block_definition_v2 import _stable_json

MAX_BYTES = 2 * 1024**2
DIAGNOSTIC_MAX_BYTES = 1048576
DIAGNOSTIC_MAX_CHARACTERS = 262144
NODE_MODULE = 'modules.HuggingFaceTransformers'
NODE_ACTION = 'GenerateImageVideoText'


def text(value, *, maximum=512, characters=None, nullable=False):
    if value is None and nullable:
        return None
    if type(value) is not str or not value or characters is not None and len(value) > characters:
        raise ValueError('Prompt proposal text identity is missing or exceeds its limit.')
    try:
        encoded = value.encode('utf-8')
    except UnicodeEncodeError as error:
        raise ValueError('Prompt proposal requires valid Unicode.') from error
    if len(encoded) > maximum or any(unicodedata.category(char) == 'Cc' and char not in '\t\r\n' for char in value):
        raise ValueError('Prompt proposal text exceeds its byte limit or contains unsupported controls.')
    return value


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError('Prompt proposal integer observation is outside its declared bounds.')
    return value


def closed(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('Prompt proposal contains a missing or unknown field.')
    return value


def bounded_json(value, maximum=MAX_BYTES):
    # Bound traversal and aggregate string bytes before allocating serialized
    # JSON. All values must already be detached built-in JSON primitives.
    entries, string_bytes = 0, 0
    def walk(item, depth):
        nonlocal entries, string_bytes
        entries += 1
        if entries > 8192 or depth > 16:
            raise ValueError('Prompt proposal JSON exceeds its structural limit.')
        kind = type(item)
        if kind is str:
            if len(item) > maximum:
                raise ValueError('Prompt proposal JSON exceeds its string limit.')
            text(item, maximum=maximum) if item else None
            string_bytes += len(item.encode('utf-8'))
            if string_bytes > maximum:
                raise ValueError('Prompt proposal JSON exceeds its aggregate byte limit.')
        elif item is None or kind is bool:
            return
        elif kind is int:
            integer(item, -(2**53 - 1), 2**53 - 1)
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError('Prompt proposal requires finite JSON numbers.')
        elif kind is list:
            for child in item:
                walk(child, depth + 1)
        elif kind is dict:
            if len(item) > 32:
                raise ValueError('Prompt proposal JSON has too many object keys.')
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError('Prompt proposal JSON keys must be text.')
                walk(key, depth + 1)
                walk(child, depth + 1)
        else:
            raise ValueError('Prompt proposal cannot transport an opaque runtime value.')
    walk(value, 0)
    raw = json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if len(raw.encode('utf-8')) > maximum:
        raise ValueError('Prompt proposal JSON exceeds its serialized byte limit.')
    return raw


def digest_bytes(raw):
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def diagnostic_output(result):
    """Exact rejected structured suffix only; never an applicable text output.

    An excessive or non-Unicode suffix remains omitted, never truncated. This
    diagnostic must not turn an otherwise completed rejected Preview into an
    execution failure.
    """
    if ((not result['systemPromptMode'].endswith('_bbox')
             and result.get('operation') != 'enhance_prompt_append_details')
            or result['applicationAllowed'] is not False):
        return None
    value = result['text']
    if type(value) is not str or not value or len(value) > DIAGNOSTIC_MAX_CHARACTERS:
        return None
    try:
        return value if len(value.encode('utf-8')) <= DIAGNOSTIC_MAX_BYTES else None
    except UnicodeEncodeError:
        return None


def _diagnostic_digest(value):
    if value is None:
        return None
    if (type(value) is not str or not value or len(value) > DIAGNOSTIC_MAX_CHARACTERS
            or len(value.encode('utf-8')) > DIAGNOSTIC_MAX_BYTES):
        raise ValueError('The native proposal diagnostic exceeds its exact string contract.')
    return digest_bytes(value.encode('utf-8'))


def canonical_digest(value):
    bounded_json(value)
    return digest_bytes(_stable_json(value).encode('utf-8'))


def authored_request(prompt, layout):
    # Deliberately measure/hash authored bytes, not _bounded_string's stripped
    # return. Standard caption semantics are outside this selected adapter.
    text(prompt, maximum=262144, characters=65536)
    if not prompt.strip():
        raise ValueError('Prompt enhancement requires a nonempty authored prompt.')
    if layout is None:
        return digest_bytes(prompt.encode('utf-8')), 'none', None
    if type(layout) is not dict:
        raise ValueError('Prompt enhancement layout remains an object or None.')
    from modiff.layout_conditioning import parse_layout_conditioning
    parse_layout_conditioning(layout)
    return digest_bytes(prompt.encode('utf-8')), 'object', canonical_digest({'kind': 'object', 'value': layout})


def _producer(observer):
    observer.check()
    from modiff.native_component_lora import _GRAPH_SCOPE
    from modiff.workflow_auto_resource import workflow_graph_hash
    from modules.HuggingFaceTransformers.main import GenerateImageVideoText
    server, task, graph, node = observer.server, observer.task, observer.graph, observer.node
    if type(node) is not GenerateImageVideoText or (node.module_name, node.class_name) != (NODE_MODULE, NODE_ACTION):
        raise ValueError('Prompt proposal requires the exact registered native action.')
    modules = getattr(server, 'modules', None)
    module = dict.get(modules, NODE_MODULE) if type(modules) is dict else None
    declaration = dict.get(module, NODE_ACTION) if type(module) is dict else None
    params = dict.get(declaration, 'params') if type(declaration) is dict else None
    field = dict.get(params, 'proposal_summary') if type(params) is dict else None
    if (type(field) is not dict or field != GenerateImageVideoText.params['proposal_summary']
            or field.get('type') != 'string' or field.get('display') != 'output'):
        raise ValueError('Prompt proposal requires its current registered native string output.')
    source = getattr(server, 'backend_source_identity', None)
    fingerprint = source.get('fingerprint') if type(source) is dict else None
    scope = _GRAPH_SCOPE.get()
    if (type(fingerprint) is not str or not re.fullmatch(r'sha256:backend-source-v1:[0-9a-f]{64}', fingerprint)
            or not scope or scope[0] is not graph or scope[1] != fingerprint):
        raise ValueError('Prompt proposal requires the actual worker and current source scope.')
    hints = task.get('runtimeHints')
    if hints is None:
        hints = {}
    if type(hints) is not dict:
        raise ValueError('Prompt proposal UI correlation must be an object.')
    ui = {key: text(hints.get(key), nullable=True) for key in ('clientRunId', 'runInputHash', 'workflowTabId')}
    for key in ('workflowCanvasEpoch', 'workflowFormEpoch'):
        value = hints.get(key)
        ui[key] = integer(value, 0, 2**53 - 1) if value is not None else None
    producer = {'taskId': text(task.get('task_id'), maximum=128),
        'attemptIndex': integer(task.get('attempt_index'), 0, 1000),
        'node': {'id': text(node.node_id), 'module': NODE_MODULE, 'action': NODE_ACTION},
        'sid': text(node._sid, nullable=True), 'workerInstance': text(getattr(server, 'instance', None), maximum=128),
        'backendSourceFingerprint': fingerprint, 'graphHash': workflow_graph_hash(graph),
        'recipe': None, 'uiContext': ui}
    bounded_json(producer, 4096)
    return producer


def _controls(value):
    closed(value, ('max_new_tokens', 'min_new_tokens', 'do_sample', 'num_beams',
                   'repetition_penalty', 'return_dict_in_generate', 'use_chat_template'))
    maximum = integer(value['max_new_tokens'], 1, 3584)
    integer(value['min_new_tokens'], 0, maximum)
    if (value['do_sample'] is not False or type(value['num_beams']) is not int or value['num_beams'] != 1
            or type(value['repetition_penalty']) not in (int, float) or value['repetition_penalty'] != 1.0
            or value['return_dict_in_generate'] is not False or value['use_chat_template'] is not True):
        raise ValueError('Prompt proposal controls differ from the actual reviewed greedy recipe.')


def _model_receipt(value, contract):
    from modules.HuggingFaceTransformers.main import SECURITY_CONTRACT, _receipt_digest_is_valid, _canonical_json
    closed(value, ('schemaVersion', 'library', 'task', 'source', 'loader', 'security', 'runtime', 'receiptDigest'))
    if (type(value['schemaVersion']) is not int or value['schemaVersion'] != 1
            or value['library'] != 'transformers' or value['task'] != 'image-video-to-text'
            or value['source'] != {'kind': 'hub', 'repository': contract['repository'], 'revision': contract['revision']}
            or value['loader'] != {'preprocessorAutoClass': 'AutoProcessor', 'modelAutoClass': 'AutoModelForImageTextToText'}
            or value['security'] != SECURITY_CONTRACT or not _receipt_digest_is_valid(value)):
        raise ValueError('Prompt proposal model receipt differs from its native selected contract.')
    security = closed(value['security'], SECURITY_CONTRACT)
    if any(security[key] is not expected for key, expected in SECURITY_CONTRACT.items()):
        raise ValueError('Prompt proposal security observations require exact native booleans.')
    runtime = closed(value['runtime'], ('transformersVersion', 'dtype', 'device', 'preprocessorClass',
        'modelClass', 'modelType', 'architectures', 'quantization'))
    quantization = runtime['quantization']
    if quantization != {'mode': 'none'}:
        from modiff.transformers_image_text_resources import image_text_nf4_policy
        expected = image_text_nf4_policy({'source': 'hub', 'value': contract['repository']},
            contract['revision'], runtime['dtype'], runtime['device'])
        if quantization != expected:
            raise ValueError('Prompt proposal quantization receipt differs from its exact native recipe.')
    if (runtime['transformersVersion'] != contract['transformersVersion']
            or runtime['modelClass'] != contract['modelClass'] or runtime['preprocessorClass'] != contract['processorClass']
            or runtime['dtype'] not in ('float32', 'float16', 'bfloat16')
            or type(runtime['device']) is not str or not re.fullmatch(r'(?:cpu|cuda|mps|xpu)(?::[0-9]{1,5})?', runtime['device'])
            ):
        raise ValueError('Prompt proposal runtime receipt differs from its selected native contract.')
    names = runtime['architectures']
    if type(names) is not list or len(names) > 32:
        raise ValueError('Prompt proposal architecture observations are malformed.')
    for name in names + ([runtime['modelType']] if runtime['modelType'] is not None else []):
        if type(name) is not str or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_-]{0,255}', name):
            raise ValueError('Prompt proposal class/type observation is malformed.')
    _canonical_json(value)  # Retain the existing ASCII receipt guard and domain.
    bounded_json(value, 16384)


def _consumed(value, controls, contract, mode, *, operation='enhance_prompt'):
    from modules.HuggingFaceTransformers.main import MAX_MEDIA_SIDE, MAX_MEDIA_PIXELS, MAX_TOTAL_MEDIA_PIXELS
    closed(value, ('schemaVersion', 'renderedPrompt', 'references', 'processedImageGrid', 'fingerprints'))
    if type(value['schemaVersion']) is not int or value['schemaVersion'] != 1:
        raise ValueError('Prompt proposal consumed schema is invalid.')
    rendered = text(value['renderedPrompt'], maximum=524288, characters=131072)
    references, grid, fields = value['references'], value['processedImageGrid'], value['fingerprints']
    if type(references) is not list or not 0 <= len(references) <= 10:
        raise ValueError('Prompt proposal references exceed the selected count.')
    if mode.startswith('t2i') != (len(references) == 0):
        raise ValueError('Prompt proposal mode differs from its actual reference count.')
    keys = {'operation', 'enhancement_max_new_tokens', 'prompt_adapter_id', 'system_prompt_mode',
            'system_prompt_sha256', 'rendered_prompt_sha256', 'rendered_prompt_utf8_bytes',
            'rendered_input_ids_sha256', 'reference_source_sha256'}
    if references:
        keys.update(('processed_pixel_sha256', 'processed_grid_sha256', 'processed_pixel_shape', 'processed_grid_shape'))
    closed(fields, keys)
    if (operation not in ('enhance_prompt', 'enhance_prompt_append_details')
            or fields['operation'] != operation or fields['enhancement_max_new_tokens'] != controls['max_new_tokens']
            or type(fields['enhancement_max_new_tokens']) is not int or fields['prompt_adapter_id'] != contract['id']
            or fields['system_prompt_mode'] != mode or fields['rendered_prompt_sha256'] != digest_bytes(rendered.encode('utf-8'))
            or type(fields['rendered_prompt_utf8_bytes']) is not int or fields['rendered_prompt_utf8_bytes'] != len(rendered.encode('utf-8'))
            or type(fields['system_prompt_sha256']) is not str or not re.fullmatch(r'[0-9a-f]{64}', fields['system_prompt_sha256'])):
        raise ValueError('Prompt proposal consumed fingerprints are inconsistent.')
    hashes = fields['reference_source_sha256']
    if type(hashes) is not list or len(hashes) != len(references):
        raise ValueError('Prompt proposal ordered reference hashes are inconsistent.')
    for key in ('rendered_input_ids_sha256', *(['processed_pixel_sha256', 'processed_grid_sha256'] if references else [])):
        if type(fields[key]) is not str or not re.fullmatch(r'sha256:[0-9a-f]{64}', fields[key]):
            raise ValueError('Prompt proposal tensor fingerprint is malformed.')
    total = 0
    for index, reference in enumerate(references):
        closed(reference, ('index', 'mode', 'width', 'height', 'sha256'))
        if (type(reference['index']) is not int or reference['index'] != index or reference['mode'] != 'RGB'
                or type(reference['sha256']) is not str or not re.fullmatch(r'sha256:[0-9a-f]{64}', reference['sha256'])
                or hashes[index] != reference['sha256']):
            raise ValueError('Prompt proposal reference order or fingerprint is inconsistent.')
        area = integer(reference['width'], 1, MAX_MEDIA_SIDE) * integer(reference['height'], 1, MAX_MEDIA_SIDE)
        if area > MAX_MEDIA_PIXELS:
            raise ValueError('Prompt proposal reference geometry exceeds the existing media limit.')
        total += area
    if total > MAX_TOTAL_MEDIA_PIXELS or type(grid) is not list or len(grid) != len(references):
        raise ValueError('Prompt proposal media/grid observations are inconsistent.')
    if references:
        patches = 0
        for row in grid:
            if type(row) is not list or len(row) != 3:
                raise ValueError('Prompt proposal processed image grid is malformed.')
            patches += math.prod(integer(cell, 1, 268435456) for cell in row)
        shape = fields['processed_pixel_shape']
        if (type(shape) is not list or len(shape) != 2
                or any(type(item) is not int or item <= 0 for item in shape)
                or shape[0] != patches or math.prod(shape) > 268435456
                or fields['processed_grid_shape'] != [len(references), 3]
                or any(type(item) is not int for item in fields['processed_grid_shape'])):
            raise ValueError('Prompt proposal processed tensor shapes are inconsistent.')
    if len(json.dumps(value, ensure_ascii=True).encode()) > 256 * 1024:
        raise ValueError('Prompt proposal exceeds the existing consumed projection guard.')
    bounded_json(value, 256 * 1024)


class BoundSummary:
    """Ephemeral bound context; only detached primitive output is published."""

    def __init__(self, observer, declaration, contract, prompt, layout, *, operation='enhance_prompt'):
        self.observer, self.declaration, self.contract = observer, declaration, contract
        self.producer = _producer(observer)
        self.prompt_hash, self.layout_kind, self.layout_hash = authored_request(prompt, layout)
        self.layout = deepcopy(layout)
        self.authored_prompt, self.operation = prompt, operation

    def check(self):
        if _producer(self.observer) != self.producer:
            raise InterruptedError('Prompt proposal producer/source graph changed before publication.')
        from modiff import prompt_enhancement as adapter
        try:
            raw, _ = adapter._json_file(adapter.DESCRIPTOR, 16 * 1024)
        except (OSError, ValueError) as error:
            raise ValueError('Prompt proposal declaration changed before publication.') from error
        if raw != self.declaration:
            raise ValueError('Prompt proposal declaration changed before publication.')

    def observe_inputs(self, consumed, controls):
        observe_inputs(self, consumed, controls)

    def finish(self, result, controls):
        self.check()
        consumed = result['consumedInput']
        actual_controls = {key: controls[key] for key in ('max_new_tokens', 'min_new_tokens', 'do_sample',
                           'num_beams', 'repetition_penalty', 'return_dict_in_generate')}
        actual_controls['use_chat_template'] = True
        _controls(actual_controls)
        mode = result['systemPromptMode']
        if mode not in ('t2i', 'i2i', 't2i_bbox', 'i2i_bbox'):
            raise ValueError('Prompt proposal mode is unsupported.')
        _consumed(consumed, actual_controls, self.contract, mode, operation=self.operation)
        _model_receipt(result['modelReceipt'], self.contract)
        integer(result['inputTokens'], 1, 32768)
        count = integer(result['generatedTokens'], 1, actual_controls['max_new_tokens'])
        expected_finish = 'length' if count == actual_controls['max_new_tokens'] else 'stop'
        if (result['finishReason'] != expected_finish or result['applyRequired'] is not True
                or result['semanticFidelityVerified'] is not False):
            raise ValueError('Prompt proposal completion observations are inconsistent.')
        structured, reason = result.get('structuredProposal'), result.get('applicationBlockedReason')
        if self.operation == 'enhance_prompt_append_details':
            from modiff.prompt_enhancement_output import compose_append_details
            if mode not in ('t2i', 'i2i') or result.get('operation') != self.operation:
                raise ValueError('Append-details operation and actual publisher mode changed.')
            proposal_text = None
            if result['layoutOutputValidated'] is True and result['applicationAllowed'] is True:
                if type(structured) is not dict or reason is not None:
                    raise ValueError('Composed proposal observations are inconsistent.')
                closed(structured, ('schemaVersion', 'kind', 'caption', 'layoutConditioning', 'nativeText'))
                expected = compose_append_details(self.authored_prompt, self.layout, result['text'],
                                                  image_count=len(consumed['references']))
                if structured != expected:
                    raise ValueError('Composed native text or exact authored prefix/layout changed after validation.')
            elif result['layoutOutputValidated'] is False and result['applicationAllowed'] is False and structured is None:
                text(reason, maximum=16384, characters=4096)
            else:
                raise ValueError('Composed proposal flags are inconsistent.')
        elif mode.endswith('_bbox'):
            proposal_text = None
            if result['layoutOutputValidated'] is True and result['applicationAllowed'] is True:
                from modiff.prompt_enhancement_output import validate_target_layout_output
                if type(structured) is not dict or reason is not None:
                    raise ValueError('Prompt proposal structured observations are inconsistent.')
                closed(structured, ('schemaVersion', 'caption', 'layoutConditioning', 'rawTable') +
                       (('rawCaption',) if structured.get('schemaVersion') == 2 else ()))
                rendered = structured.get('rawCaption', structured['caption']) + '\n' + bounded_json(structured['rawTable'], 256 * 1024)
                if validate_target_layout_output(rendered, self.layout, mode=mode, image_count=len(consumed['references'])) != structured:
                    raise ValueError('Prompt proposal structured layout changed after validation.')
                if len(json.dumps(structured, ensure_ascii=True).encode()) > 256 * 1024:
                    raise ValueError('Prompt proposal exceeds the existing structured guard.')
                bounded_json(structured, 256 * 1024)
            elif result['layoutOutputValidated'] is False and result['applicationAllowed'] is False and structured is None:
                text(reason, maximum=16384, characters=4096)
            else:
                raise ValueError('Prompt proposal structured flags are inconsistent.')
        else:
            if structured is not None or reason is not None or result['layoutOutputValidated'] is not None or result['applicationAllowed'] is not True:
                raise ValueError('Prompt proposal plain flags are inconsistent.')
            proposal_text = text(result['text'], maximum=1048576, characters=262144)
        request = {'authoredPromptSha256': self.prompt_hash, 'authoredLayoutKind': self.layout_kind,
            'authoredLayoutSha256': self.layout_hash, 'controls': actual_controls,
            'consumedInputSha256': canonical_digest(consumed)}
        adapter = {'id': self.contract['id'], 'declarationSha256': digest_bytes(self.declaration)}
        proposal = {'mode': mode, 'text': proposal_text, 'structuredProposal': structured,
            'finishReason': result['finishReason'], 'inputTokens': result['inputTokens'], 'generatedTokens': count,
            'applyRequired': True, 'applicationAllowed': result['applicationAllowed'],
            'layoutOutputValidated': result['layoutOutputValidated'], 'semanticFidelityVerified': False,
            'applicationBlockedReason': reason}
        bounded_json(request, 4096)
        bounded_json(adapter, 1024)
        summary = bounded_json({'schemaVersion': 1, 'kind': 'prompt-enhancement-proposal',
            'producer': self.producer, 'adapter': adapter, 'request': request,
            'modelReceipt': result['modelReceipt'], 'consumedInput': consumed, 'proposal': proposal})
        self.check()
        finish_pending(self, summary, consumed, request, diagnostic_output(result))
        return summary


OBSERVATION_MAX_BYTES = 32768
MAX_CACHE_OBSERVATIONS = 64


@dataclass(frozen=True)
class _PendingObservation:
    producer_json: str
    request_json: str
    consumed_json: str
    summary_sha256: str
    source_json: str
    diagnostic_sha256: str | None = None


@dataclass(frozen=True)
class _CacheObservation:
    pending: _PendingObservation
    native_node: object  # weakref only; never a node/model/observer.
    output_token: object  # weakref to native-held opaque output identity only.
    input_seal: str | None
    code_seal: str | None
    sources: frozenset
    dispatch_json: str
    detached: bool = False


def consumed_observation(consumed):
    """Capture actual pre-generation normalized RGB/tensor hashes and shape.

    Call only after the existing _consumed validation. The rendered text and
    tensor/media values are deliberately absent. Existing reference_source_sha256
    retains its RGB-header-and-pixel domain; it is not an encoded-file digest.
    """
    return bounded_json({key: consumed[key] for key in
        ('schemaVersion', 'references', 'processedImageGrid', 'fingerprints')}, OBSERVATION_MAX_BYTES)


def observe_inputs(bound_summary, consumed, controls):
    """New BoundSummary method, called before device transfer/model.generate."""
    bound_summary.check()
    mode = consumed['fingerprints']['system_prompt_mode']
    actual_controls = {key: controls[key] for key in ('max_new_tokens', 'min_new_tokens',
        'do_sample', 'num_beams', 'repetition_penalty', 'return_dict_in_generate')}
    actual_controls['use_chat_template'] = True
    _controls(actual_controls)
    _consumed(consumed, actual_controls, bound_summary.contract, mode, operation=bound_summary.operation)
    if getattr(bound_summary, '_pre_generation_observation', None) is not None:
        raise ValueError('The pre-generation input observation was already consumed.')
    request = {'authoredPromptSha256': bound_summary.prompt_hash,
        'authoredLayoutKind': bound_summary.layout_kind, 'authoredLayoutSha256': bound_summary.layout_hash,
        'controls': actual_controls}
    # Immutable JSON strings detach all aliases before inference can mutate data.
    source = _source_snapshot()
    if json.loads(source)['backend'] != bound_summary.producer['backendSourceFingerprint']:
        raise ValueError('The executable source changed after worker startup.')
    bound_summary._pre_generation_observation = (
        bounded_json(request, 4096), consumed_observation(consumed), source)


def finish_pending(bound_summary, summary, consumed, request, diagnostic=None):
    """End of existing BoundSummary.finish, after its final source check."""
    before = getattr(bound_summary, '_pre_generation_observation', None)
    projected_request = {key: request[key] for key in ('authoredPromptSha256',
        'authoredLayoutKind', 'authoredLayoutSha256', 'controls')}
    if before != (bounded_json(projected_request, 4096), consumed_observation(consumed), _source_snapshot()):
        raise ValueError('The normalized input changed after pre-generation capture.')
    bound_summary.observer.node._prompt_observation_pending = _PendingObservation(
        bounded_json(bound_summary.producer, 4096), before[0], before[1], digest_bytes(summary.encode('utf-8')), before[2],
        _diagnostic_digest(diagnostic))


class _OutputIncarnation:
    """Owned by the cache holder, observed weakly; never retained by the map.

    The exact native output mapping is not copied. Holding its identity here
    prevents recyclable-id authorization. Removing the actual cache holder
    destroys this token even if the weak observation record remains live.
    """
    __slots__ = ('output', '__weakref__')

    def __init__(self, output):
        self.output = output


def _source_snapshot():
    """Pure, bounded fresh executable and static metadata source observation."""
    from modiff.backend_source_identity import backend_source_identity
    from modiff import prompt_enhancement as adapter
    root = Path(__file__).resolve().parents[1]
    backend_before = backend_source_identity()['fingerprint']
    data = root / 'data'
    for parent in (data, *data.parents):
        if (getattr(parent.lstat(), 'st_file_attributes', 0) & 0x400
                or stat.S_ISLNK(parent.lstat().st_mode)):
            raise ValueError('The observed static metadata ancestry is not plain.')
    def census():
        result, total_bytes = {}, 0
        for entry_count, path in enumerate(data.iterdir(), 1):
            if entry_count > 1024:
                raise ValueError('The observed static directory exceeds its finite entry bound.')
            if path.suffix != '.json':
                continue
            info = path.lstat()
            if (getattr(info, 'st_file_attributes', 0) & 0x400 or not stat.S_ISREG(info.st_mode)
                    or not 0 < info.st_size <= 8 * 1024**2):
                raise ValueError('The observed static JSON entry exceeds its plain file contract.')
            if len(result) >= 256 or total_bytes + info.st_size > 64 * 1024**2:
                raise ValueError('The observed static JSON inventory exceeds its finite bound.')
            result[path.name] = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                info.st_ctime_ns, getattr(info, 'st_file_attributes', 0))
            total_bytes += info.st_size
        return result
    before, hashes = census(), {}
    for name, identity in before.items():
        digest, count = hashlib.sha256(), 0
        with (data / name).open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                count += len(chunk)
                if count > identity[2]:
                    raise ValueError('Static JSON changed while observing its source.')
                digest.update(chunk)
        if count != identity[2]:
            raise ValueError('Static JSON changed while observing its source.')
        hashes[name] = digest.hexdigest()
    if before != census():
        raise ValueError('Static JSON identities changed during source observation.')
    contract, descriptor = adapter._declared_contract(with_bytes=True)
    # Reuse the exact normal immutable cache resolver/body hashes. This also
    # verifies the external Hub system_prompts.json against its declared SHA;
    # a source data census alone cannot attest external processor metadata.
    snapshot = adapter.validate_cached_selection({'source': 'hub', 'value': contract['repository']},
        contract['revision'], 'image-video-to-text')
    if snapshot is None:
        raise ValueError('The immutable native processor metadata is no longer available.')
    prompts, _ = adapter._json_file(adapter.SYSTEM_PROMPTS, 32768)
    if hashlib.sha256(prompts).hexdigest() != contract['systemPromptsSha256']:
        raise ValueError('The current system prompts differ from their immutable declaration.')
    if backend_source_identity()['fingerprint'] != backend_before:
        raise ValueError('Executable source changed during source observation.')
    return bounded_json({'backend': backend_before,
        'metadata': [[name, hashes[name]] for name in sorted(hashes)], 'adapterSha256': digest_bytes(descriptor),
        'systemPromptsSha256': digest_bytes(prompts)}, OBSERVATION_MAX_BYTES)


def _source_current(server, record):
    producer = json.loads(record.pending.producer_json)
    current = _source_snapshot()
    if (producer['workerInstance'] != server.instance or producer['backendSourceFingerprint'] !=
            server.backend_source_identity['fingerprint']
            or json.loads(current)['backend'] != producer['backendSourceFingerprint']
            or record.pending.source_json != current):
        raise ValueError('The cache observation belongs to a different worker or changed source/metadata.')


def prune_cache_observations(server, node_ids=()):
    """Existing cache mutation boundaries only; never called by a GET."""
    records = getattr(server, '_prompt_summary_observations', None)
    if type(records) is not dict:
        return
    selected = set(node_ids)
    for key, record in tuple(records.items()):
        cached = server.node_cache.get(key)
        if (key in selected or type(record) is not _CacheObservation
                or cached is None or record.native_node() is not cached
                or record.output_token() is None
                or vars(cached).get('_prompt_output_incarnation') is not record.output_token()):
            records.pop(key, None)
    while len(records) > MAX_CACHE_OBSERVATIONS:
        records.pop(next(iter(records)))


def _valid_cache(server, node_id, record):
    require_native_cache_output(server, node_id)
    GenerateImageVideoText = _native_action()
    from modiff.node_cache_identity import implementation_identity
    from modiff.ancillary_owner_lifecycle import _digest
    _source_current(server, record)
    producer = json.loads(record.pending.producer_json)
    dispatch = json.loads(record.dispatch_json)
    if producer['node']['id'] != node_id or dispatch['node']['id'] != node_id:
        raise ValueError('The native cache observation cannot be renamed or aliased.')
    node = server.node_cache.get(node_id)
    token = record.output_token()
    if (node is None or record.native_node() is not node or token is None
            or vars(node).get('_prompt_output_incarnation') is not token
            or token.output is not vars(node).get('output')):
        raise ValueError('The observed native cache incarnation was replaced.')
    if record.detached:
        from modiff.workflow_auto_lifecycle import _RetainedPromptProposal
        if (type(node) is not _RetainedPromptProposal or vars(node).get('params') != {}
                or vars(node).get('_mm_models') != []
                or (vars(node).get('module_name'), vars(node).get('class_name')) != (NODE_MODULE, NODE_ACTION)):
            raise ValueError('The server-minted detached cache identity changed.')
    elif (type(node) is not GenerateImageVideoText or record.native_node() is not node
            or node._cache_valid is not True or node._cache_invalidated is not False
            or _digest(node._cache_input_snapshot) != record.input_seal
            or _digest(node._cache_implementation) != record.code_seal
            or _digest(implementation_identity(node)) != record.code_seal
            or type(node._cache_input_sources) is not frozenset
            or node._cache_input_sources != record.sources):
        raise ValueError('The exact native input/code/source cache seal changed.')
    value = node.output.get('proposal_summary')
    if type(value) is not str or digest_bytes(value.encode('utf-8')) != record.pending.summary_sha256:
        raise ValueError('The exact observed UTF-8 proposal string changed.')
    return node, value


def seal_dispatch_observation(server, node_id):
    """Normal execute_node success hook; never used by a cache GET.

    The existing actual _OwnedGeneration observer and _producer validate the
    current server/task/graph/native registry. A cache hit updates only dispatch
    B; original pre-output producer A is never reconstructed from summary data.
    """
    GenerateImageVideoText = _native_action()
    from modiff.ancillary_owner_lifecycle import _digest
    from modiff.native_component_lora import _GRAPH_SCOPE
    node = server.node_cache.get(node_id)
    if GenerateImageVideoText is None or type(node) is not GenerateImageVideoText:
        return
    prune_cache_observations(server)
    records = getattr(server, '_prompt_summary_observations', {})
    value = node.output.get('proposal_summary')
    if value is None:
        records.pop(node_id, None)
        vars(node).pop('_prompt_observation_pending', None)
        vars(node).pop('_prompt_output_incarnation', None)
        return
    scope = _GRAPH_SCOPE.get()
    if not scope or type(scope[0]) is not dict:
        raise ValueError('The native proposal dispatch has no current graph scope.')
    context = _actual_producer(server, server.current_task, scope[0], node)
    dispatch = {key: value for key, value in context.items() if key != 'recipe'}
    if node._has_changed is not True and node._has_changed is not False:
        raise ValueError('A dispatch must expose an exact native cache status.')
    dispatch['status'] = 'succeeded' if node._has_changed is True else 'cached'
    pending = vars(node).pop('_prompt_observation_pending', None)
    if node._has_changed is True:
        if (type(pending) is not _PendingObservation or json.loads(pending.producer_json) != context
                or digest_bytes(value.encode('utf-8')) != pending.summary_sha256
                or _diagnostic_digest(node.output.get('proposal_diagnostic')) != pending.diagnostic_sha256):
            raise ValueError('A computed proposal lacks its actual pre-output observation.')
        token = _OutputIncarnation(node.output)
        node._prompt_output_incarnation = token
        record = _CacheObservation(pending, weakref.ref(node), weakref.ref(token),
            _digest(node._cache_input_snapshot), _digest(node._cache_implementation),
            frozenset(node._cache_input_sources), bounded_json(dispatch, 4096))
    else:
        record = records.get(node_id)
        if type(record) is not _CacheObservation:
            return  # Historical unobserved cache stays diagnostic, never minted on GET.
        _valid_cache(server, node_id, record)
        if json.loads(record.pending.producer_json)['graphHash'] != dispatch['graphHash']:
            raise ValueError('The original proposal and current cached dispatch have different semantic graphs.')
        record = replace(record, dispatch_json=bounded_json(dispatch, 4096))
    records[node_id] = record
    server._prompt_summary_observations = records
    prune_cache_observations(server)


def bind_terminal_detached_observation(server, node_id, receipt, record):
    """Only after exact scoped terminal release succeeds, with no replay.

    The existing release helper already proves that this retained namespace was
    produced from the exact native string output. Failed/surviving release never
    rebinds; the original raw diagnostic summary remains independent.
    """
    if node_id not in receipt['retainedOutputNodes']:
        raise ValueError('The terminal release did not retain its observed proposal.')
    records = getattr(server, '_prompt_summary_observations', {})
    if type(record) is not _CacheObservation:
        return
    _source_current(server, record)
    from modiff.workflow_auto_lifecycle import _RetainedPromptProposal
    node = server.node_cache.get(node_id)
    if (type(node) is not _RetainedPromptProposal or vars(node).get('params') != {}
            or vars(node).get('_mm_models') != []
            or digest_bytes(node.output['proposal_summary'].encode('utf-8')) != record.pending.summary_sha256):
        raise ValueError('The retained terminal material differs from its observed output.')
    token = _OutputIncarnation(node.output)
    node._prompt_output_incarnation = token
    records[node_id] = replace(record, native_node=weakref.ref(node), output_token=weakref.ref(token), detached=True)
    server._prompt_summary_observations = records
    prune_cache_observations(server)


def read_cache_observation(server, node_id, *, field='proposal_summary'):
    """Pure existing-cache response path: no mint/update/model/processor work."""
    lock = getattr(server, '_node_cache_lock', None)
    if lock is not None and lock.locked():
        raise ValueError('Native cache is executing; wait for the actual task to finish.')
    record = getattr(server, '_prompt_summary_observations', {}).get(node_id)
    if type(record) is not _CacheObservation:
        raise ValueError('No actual native cache observation is available; explicitly recompute Preview.')
    node, value = _valid_cache(server, node_id, record)
    observation = {'schemaVersion': 1, 'summarySha256': record.pending.summary_sha256,
        'producer': json.loads(record.pending.producer_json), 'dispatch': json.loads(record.dispatch_json),
        'request': json.loads(record.pending.request_json), 'consumedInput': json.loads(record.pending.consumed_json)}
    if field == 'proposal_diagnostic':
        value = require_native_cache_output(server, node_id, field=field)
        if (record.pending.diagnostic_sha256 is None
                or _diagnostic_digest(value) != record.pending.diagnostic_sha256):
            raise ValueError('The exact observed UTF-8 proposal diagnostic changed.')
        observation['diagnosticSha256'] = record.pending.diagnostic_sha256
    elif field != 'proposal_summary':
        raise ValueError('Unsupported native proposal observation field.')
    bounded_json(observation, OBSERVATION_MAX_BYTES)
    return {'value': value, 'observation': observation}


def _native_action():
    # Static lookup only: GET must never import a neural runtime or invoke a
    # custom extension's properties while deciding whether it is servable.
    loaded = sys.modules.get(NODE_MODULE + '.main')
    action = vars(loaded).get(NODE_ACTION) if type(loaded) is ModuleType else None
    return action if type(action) is type else None


def require_native_cache_output(server, node_id, *, field='proposal_summary'):
    """Validate exact preloaded native/retained string before value access."""
    from modiff.workflow_auto_lifecycle import _RetainedPromptProposal
    node = server.node_cache.get(node_id)
    if node is None:
        raise LookupError('The proposal node is not present in cache.')
    action = _native_action()
    if action is None or type(node) not in (action, _RetainedPromptProposal):
        raise ValueError('Metadata is available only for the exact native proposal string.')
    state = vars(node)
    if (state.get('module_name'), state.get('class_name')) != (NODE_MODULE, NODE_ACTION):
        raise ValueError('The native proposal cache identity changed.')
    modules = server.modules
    module = dict.get(modules, NODE_MODULE) if type(modules) is dict else None
    declaration = dict.get(module, NODE_ACTION) if type(module) is dict else None
    params = dict.get(declaration, 'params') if type(declaration) is dict else None
    if field not in ('proposal_summary', 'proposal_diagnostic'):
        raise ValueError('Unsupported native proposal observation field.')
    field_name = field
    field = dict.get(params, field_name) if type(params) is dict else None
    expected = vars(action).get('params')
    expected = dict.get(expected, field_name) if type(expected) is dict else None
    if (type(field) is not dict or field != expected or field.get('type') != 'string'
            or field.get('display') != 'output'):
        raise ValueError('The current registered native string output contract changed.')
    output = state.get('output')
    if type(output) is not dict or field_name not in output or output[field_name] is None:
        raise LookupError('The native proposal output is empty.')
    value = output[field_name]
    if field_name == 'proposal_diagnostic':
        _diagnostic_digest(value)
        return value
    if type(value) is not str or not value or len(value.encode('utf-8')) > MAX_BYTES:
        raise ValueError('The native proposal output exceeds its exact string contract.')
    return value


def _actual_producer(server, task, graph, node):
    from modiff.prompt_enhancement import _OwnedGeneration
    observer = _OwnedGeneration(node)
    observer.check()
    if observer.server is not server or observer.task is not task or observer.graph is not graph:
        raise ValueError('The current dispatch is not the actual native producer scope.')
    return _producer(observer)


def terminal_cache_observation(server, node_id):
    """Save only the immutable weak observation before actual scoped release."""
    record = getattr(server, '_prompt_summary_observations', {}).get(node_id)
    if type(record) is not _CacheObservation:
        return None
    _valid_cache(server, node_id, record)
    return record
