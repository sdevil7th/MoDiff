"""Bounded evidence of resolved node call arguments, not inferred model defaults.

Only explicitly named generation fields enter local output history. Never walk
arbitrary objects (model components, tensors, media or credentials). An output's
receipt contains only records in its actual submitted graph ancestry.
"""

import math
import json
import hashlib
from copy import deepcopy


FIELD_NAMES = {
    "prompt": "prompt", "prompt_2": "prompt2", "prompt_3": "prompt3",
    "negative_prompt": "negativePrompt", "negative_prompt_2": "negativePrompt2",
    "seed": "seed", "width": "width", "height": "height",
    "num_inference_steps": "steps", "steps": "steps",
    "guidance_scale": "guidanceScale", "true_cfg_scale": "trueCfgScale",
    "max_sequence_length": "maxSequenceLength", "num_images_per_prompt": "imagesPerPrompt",
    "num_frames": "numFrames", "fps": "fps", "frame_rate": "fps",
    "audio_duration": "audioDuration", "sample_rate": "sampleRate",
    "strength": "strength", "repo_id": "repo", "model_id": "repo",
    "revision": "revision", "dtype": "dtype", "device": "device",
    "model_type": "modelType", "pipeline_class": "modelType",
    "auto_offload": "autoOffload", "offload_mode": "offloadMode",
    "weight_loading_mode": "weightLoadingMode",
    "attention_backend": "attentionBackend",
    "vae_slicing": "vaeSlicing", "enable_vae_slicing": "vaeSlicing",
    "vae_tiling": "vaeTiling", "enable_vae_tiling": "vaeTiling",
    "output_type": "outputType", "quant_config": "quantConfig",
    "conditioning_scale": "conditioningScale", "control_mode": "controlMode",
    "control_guidance_start": "controlGuidanceStart", "control_guidance_end": "controlGuidanceEnd",
    "prompt_embeds_scale": "reduxPromptEmbedsScale",
    "pooled_prompt_embeds_scale": "reduxPooledPromptEmbedsScale",
    "processing_resolution": "processingResolution",
    "match_input_resolution": "matchInputResolution", "depth_convention": "depthConvention",
    "use_kv_cache": "attentionContextReuse",
}
NODE_FIELD_NAMES = {
    ('Image', 'Load'): {
        **FIELD_NAMES, 'alpha_channel': 'imageAlphaChannel',
        'source_hashes': 'imageSourceHashes', 'source_widths': 'imageSourceWidths',
        'source_heights': 'imageSourceHeights', 'source_modes': 'imageSourceModes',
    },
    ('Image', 'Resize'): {
        **FIELD_NAMES, 'method': 'imageResizeMethod', 'interpolation': 'imageInterpolation',
        'multiple_of': 'imageSizeMultiple',
    },
    ('ImageOperations', 'ResizeImage'): {
        **FIELD_NAMES, 'fit_mode': 'imageFitMode', 'resampling': 'imageResampling',
    },
    ('ImageOperations', 'CropImage'): {**FIELD_NAMES, 'x': 'imageCropLeft', 'y': 'imageCropTop'},
    ('ImageOperations', 'MaskComposite'): {
        **FIELD_NAMES, 'mask_channel': 'imageMaskChannel', 'invert_mask': 'imageMaskInverted',
    },
    ('Image', 'Save'): {**FIELD_NAMES, 'filename': 'imageSaveFilename', 'quality': 'imageSaveQuality'},
    ('ModularDiffusers', 'ModelsLoader'): {**FIELD_NAMES, 'weight_format': 'weightFormat'},
    ('DiffusersImage', 'LoadPipeline'): {**FIELD_NAMES, 'mode': 'executionMode'},
    **{('DiffusersImage', action): {**FIELD_NAMES, 'executed_task': 'executedTask'}
       for action in ('Generate', 'Edit', 'Inpaint', 'ControlGenerate', 'ControlEdit', 'ControlInpaint',
                      'LayerDecompose', 'UnconditionalGenerate', 'PredictMap',
                      'EncodeInputs', 'Denoise', 'DecodeLatents')},
    **{('HuggingFaceTransformers', action): {**FIELD_NAMES, 'executed_task': 'executedTask'}
       for action in ('PredictDepth',)},
    ('HuggingFaceTransformers', 'GenerateImageVideoText'): {
        **FIELD_NAMES, 'executed_task': 'executedTask',
        'operation': 'textOperation', 'enhancement_max_new_tokens': 'enhancementMaxNewTokens',
        'prompt_adapter_id': 'promptAdapterId', 'system_prompt_mode': 'systemPromptMode',
        'system_prompt_sha256': 'systemPromptSha256',
        'rendered_prompt_sha256': 'renderedPromptSha256', 'rendered_prompt_utf8_bytes': 'renderedPromptUtf8Bytes',
        'rendered_input_ids_sha256': 'renderedInputIdsSha256', 'reference_source_sha256': 'referenceSourceSha256',
        'processed_pixel_sha256': 'processedPixelSha256', 'processed_pixel_shape': 'processedPixelShape',
        'processed_grid_sha256': 'processedGridSha256', 'processed_grid_shape': 'processedGridShape',
    },
    ('ImageOperations', 'ProcessImage'): {**FIELD_NAMES, 'executed_task': 'executedTask'},
    ('ModularDiffusers', 'Guider'): {
        **FIELD_NAMES,
        'guider': 'guiderType', 'enabled': 'guidanceEnabled',
        'use_original_formulation': 'guidanceOriginalFormulation',
        'guidance_rescale': 'guidanceRescale', 'start': 'guidanceStart', 'stop': 'guidanceStop',
    },
    ('DiffusersImage', 'ControlComponent'): {
        'model_id': 'controlComponentRepo', 'revision': 'controlComponentRevision',
        'shared_conditions': 'controlComponentSharedConditions',
    },
    ('DiffusersImage', 'ImagePromptAdapter'): {
        'adapter_model': 'ipAdapterRepo', 'revision': 'ipAdapterRevision',
        'weight_name': 'ipAdapterWeight', 'expected_sha256': 'ipAdapterSha256',
        'image_encoder_model': 'ipAdapterEncoderRepo',
        'image_encoder_revision': 'ipAdapterEncoderRevision',
        'image_encoder_sha256': 'ipAdapterEncoderSha256',
        'scale': 'ipAdapterScale', 'layer_scales': 'ipAdapterLayerScales',
    },
}


def _field_names(node):
    module = str(node.get('module', '')).removeprefix('modules.')
    # The JSON/data processor is an executed operation, not another model.
    # Keep its exact identity in the receipt without making the actual image
    # loader ambiguous. Other model-backed text processors retain modelType.
    if ((module, node.get('action')) == ('Text', 'ProcessText')
            and node.get('fields', {}).get('pipeline_class', {}).get('value') == 'BuiltinDataOperationV1'):
        return {**FIELD_NAMES, 'pipeline_class': 'dataOperationType'}
    return NODE_FIELD_NAMES.get((module, node.get('action')), FIELD_NAMES)

MAX_STRING = 16_384
MAX_ITEMS = 16
MAX_NODES = 256
MAX_RECORD_BYTES = 65_536
MAX_RECEIPT_BYTES = 262_144
# These fields already have top-level StudioOutput display slots. Other captured
# values remain available in the receipt without inventing new form semantics.
OUTPUT_FIELDS = {"prompt", "negativePrompt", "seed", "width", "height", "steps", "guidanceScale", "repo", "modelType"}

# Additional evidence only. Do not widen the v1 summary or imply that every
# authored control reached the library. Each supported adapter accounts for its
# own bounded literals at the actual call/configuration boundary.
ACCOUNTING_FIELDS = {
    ('DiffusersImage', 'LoadPipeline'): frozenset('''model_id pipeline_class mode revision
        conditioning_kind conditioning_model_id conditioning_revision dtype device quantization_mode
        quantized_components device_map auto_offload offload_mode enable_vae_slicing
        enable_vae_tiling low_cpu_mem_usage'''.split()),
    ('DiffusersRuntime', 'PipelineQuantizationConfigV2'): frozenset(
        'backend components dtype excluded_modules component_overrides'.split()),
    ('DiffusersRuntime', 'DiffusersExecutionRecipe'): frozenset('''device_map max_memory
        device_map_overrides offload_mode device attention_backend attention_components vae_slicing
        vae_tiling layerwise_casting layerwise_casting_components layerwise_storage_dtype
        layerwise_compute_dtype channels_last channels_last_components regional_compile
        compile_components compile_backend compile_mode compile_fullgraph compile_dynamic
        denoiser_cache cache_threshold cache_options'''.split()),
    ('DiffusersImage', 'UnconditionalGenerate'): frozenset(
        'image_contract batch_size seed num_inference_steps eta class_label output_type'.split()),
    ('Image', 'Preview'): frozenset({'device', 'export'}),
}
ACCOUNTING_PORTS = {
    ('DiffusersImage', 'LoadPipeline'): ({'execution_recipe': 'execution-recipe-v1'}, {}),
    ('DiffusersRuntime', 'PipelineQuantizationConfigV2'): ({}, {'quantization_config': 'disabled-quantization-v1'}),
    ('DiffusersRuntime', 'DiffusersExecutionRecipe'): (
        {'quantization_config': 'disabled-quantization-v1'}, {'execution_recipe': 'execution-recipe-v1'}),
    ('DiffusersImage', 'UnconditionalGenerate'): ({}, {}),
    ('Image', 'Preview'): ({}, {}),
}
ACCOUNTING_OBJECT_INPUTS = {
    ('DiffusersImage', 'LoadPipeline'): frozenset({'execution_recipe'}),
    ('DiffusersRuntime', 'PipelineQuantizationConfigV2'): frozenset(),
    ('DiffusersRuntime', 'DiffusersExecutionRecipe'): frozenset({'quantization_config'}),
    ('DiffusersImage', 'UnconditionalGenerate'): frozenset({'pipeline'}),
    ('Image', 'Preview'): frozenset({'image', 'vae'}),
}
ACCOUNTING_REASONS = frozenset({'adapter-inactive', 'feature-disabled', 'connected-recipe',
    'loader-normalization', 'presentation-contract', 'already-decoded-image', 'unsupported-contract'})


def _accounting_scope(node):
    return str(node.get('module', '')).removeprefix('modules.'), node.get('action')


def _accounting_digest(kind, value):
    return hashlib.sha256((kind + '\n' + json.dumps(value, sort_keys=True, separators=(',', ':'),
                                                   ensure_ascii=True, allow_nan=False)).encode('ascii')).hexdigest()


def _valid_projection(projection, kind):
    if type(projection) is not dict or set(projection) != {'kind', 'values', 'sha256'} or projection['kind'] != kind:
        return False
    values = projection['values']
    if kind == 'disabled-quantization-v1':
        if type(values) is not dict or values != {'backend': 'none', 'disabled': True}:
            return False
    elif kind == 'execution-recipe-v1':
        names = ACCOUNTING_FIELDS[('DiffusersRuntime', 'DiffusersExecutionRecipe')]
        expected = names - {'device_map_overrides'} | {'schema_version', 'device_map_strategy', 'quantization'}
        if type(values) is not dict or set(values) != expected or values.get('quantization') != 'none':
            return False
        if any(_bounded_value(value)[1] for value in values.values()):
            return False
    else:
        return False
    return projection['sha256'] == _accounting_digest(kind, values)


def disabled_quantization_projection(value):
    """Only the existing concrete disabled marker/normalized absence is known."""
    if value is not None and (type(value) is not dict or value != {
        'schema_version': 1, 'backend': 'none', 'disabled': True,
    }):
        return None
    values = {'backend': 'none', 'disabled': True}
    return {'kind': 'disabled-quantization-v1', 'values': values,
            'sha256': _accounting_digest('disabled-quantization-v1', values)}


def execution_recipe_projection(recipe):
    """Fixed plain projection, never a traversal of a quantizer/model object.

    Nonempty mapping configurations and enabled quantizers remain unsupported
    evidence in this first bounded contract; ordinary execution is unchanged.
    """
    fields = ACCOUNTING_FIELDS[('DiffusersRuntime', 'DiffusersExecutionRecipe')]
    expected = fields - {'device_map_overrides'} | {'schema_version', 'quantization_config', 'device_map_strategy'}
    if type(recipe) is not dict or set(recipe) != expected or recipe.get('schema_version') != 1:
        return None
    if disabled_quantization_projection(recipe['quantization_config']) is None:
        return None
    values = {'quantization': 'none'}
    for key in sorted(expected - {'quantization_config'}):
        value = recipe[key]
        if key in {'max_memory', 'cache_options'}:
            if type(value) is not dict or value:
                return None
            value = '{}'
        safe, reason = _bounded_value(value)
        if reason:
            return None
        values[key] = safe
    return {'kind': 'execution-recipe-v1', 'values': values,
            'sha256': _accounting_digest('execution-recipe-v1', values)}


def capture_field_accounting(node, inputs, *, consumed=None, inactive=None, superseded=None,
                             metadata=None, input_projections=None, output_projections=None):
    """Account explicitly for known inputs without altering v1 consumed fields.

    ``value`` is the argument actually consumed, or the inactive input;
    ``inputValue`` retains its normalized authored value for a supersession.
    Metadata accepts only an exact adapter-owned expected contract and stores
    its digest, never arbitrary nested input. Unknowns cannot become evidence.
    """
    scope = _accounting_scope(node)
    allowed = ACCOUNTING_FIELDS.get(scope)
    if allowed is None:
        return None
    consumed, inactive = consumed or {}, inactive or {}
    superseded, metadata = superseded or {}, metadata or {}
    fields, omitted, unaccounted = {}, {}, []
    for key in sorted(set(inputs) & allowed):
        choices = sum(key in group for group in (consumed, inactive, superseded, metadata))
        if choices != 1:
            unaccounted.append(key)
            continue
        original = inputs[key]
        if key in metadata:
            expected = metadata[key]
            if key != 'image_contract' or type(original) is not dict or original != expected:
                omitted[key] = 'metadata-contract-mismatch'
                continue
            fields[key] = {'disposition': 'metadata', 'reason': 'presentation-contract',
                           'sha256': _accounting_digest('image-contract-v1', expected)}
            continue
        if key in {'model_id', 'conditioning_model_id'} and type(original) is dict:
            original = original.get('value') if original.get('source') in {'hf', 'hub', 'local'} else original
        original, reason = _bounded_value(original)
        if reason:
            omitted[key] = reason
            continue
        disposition = 'consumed' if key in consumed else 'inactive' if key in inactive else 'superseded'
        value = consumed[key] if key in consumed else original if key in inactive else superseded[key][0]
        if key in {'model_id', 'conditioning_model_id'} and type(value) is dict:
            value = value.get('value') if value.get('source') in {'hf', 'hub', 'local'} else value
        value, reason = _bounded_value(value)
        if reason:
            omitted[key] = reason
            continue
        entry = {'disposition': disposition, 'inputValue': original, 'value': value}
        if disposition != 'consumed':
            reason = inactive[key] if key in inactive else superseded[key][1]
            if reason not in ACCOUNTING_REASONS:
                omitted[key] = 'unsupported-disposition'
                continue
            entry['reason'] = reason
        fields[key] = entry
    result = {'schemaVersion': 1, 'fields': fields, 'inputProjections': {}, 'outputProjections': {},
              'unaccountedFields': unaccounted, 'omittedFields': omitted}
    for label, supplied, expected in zip(('inputProjections', 'outputProjections'),
            (input_projections or {}, output_projections or {}), ACCOUNTING_PORTS[scope]):
        for key in expected.keys() - supplied.keys():
            omitted[key] = 'missing-projection'
        for key, projection in supplied.items():
            if key not in expected or not _valid_projection(projection, expected[key]):
                omitted[key] = 'unsupported-projection'
                continue
            result[label][key] = deepcopy(projection)
    result['complete'] = not omitted and not unaccounted
    if len(json.dumps(result, ensure_ascii=True).encode('ascii')) > MAX_RECORD_BYTES // 2:
        return {'schemaVersion': 1, 'fields': {}, 'inputProjections': {}, 'outputProjections': {},
                'unaccountedFields': [], 'omittedFields': {'accounting': 'record-size-limit'}, 'complete': False}
    return result


def _bounded_value(value):
    if value is None or type(value) is bool:
        return value, None
    if type(value) in (int, float):
        if abs(value) <= 2**53 - 1 and math.isfinite(value):
            # JSON/JavaScript has one numeric domain; 3 and 3.0 are compatible
            # evidence, while the boolean path above remains distinct.
            return int(value) if int(value) == value else value, None
        return None, "non-finite-or-unsafe-number"
    if type(value) is str:
        return (value, None) if len(value) <= MAX_STRING else (None, "oversize-string")
    if type(value) in (tuple, list):
        if len(value) > MAX_ITEMS:
            return None, "oversize-list"
        values = []
        for item in value:
            if type(item) in (tuple, list, dict):
                return None, "unsupported-value"
            safe, reason = _bounded_value(item)
            if reason:
                return None, reason
            values.append(safe)
        return values, None
    return None, "unsupported-value"


def capture_generation_inputs(node_id, node, args, overrides=None):
    """Snapshot allowlisted resolved arguments immediately before dispatch."""
    fields, omitted = {}, {}
    remaining_bytes = MAX_RECORD_BYTES
    params = node.get("params") or {}
    field_names = _field_names(node)
    for key, value in args.items():
        if key not in field_names:
            continue
        # Model selectors use a small repository selector envelope. Deliberately
        # do not recurse into arbitrary dictionaries or stringify runtime objects.
        if key in {"repo_id", "model_id", "adapter_model", "image_encoder_model"} and type(value) is dict:
            value = value.get("value") if value.get("source") in {"hf", "hub", "local"} else value
        safe, reason = _bounded_value(value)
        if not reason:
            size = len(json.dumps(safe, ensure_ascii=True).encode("ascii"))
            if size > remaining_bytes:
                reason = "record-size-limit"
            else:
                remaining_bytes -= size
        if reason:
            omitted[key] = reason
            continue
        param = params.get(key) or {}
        origin = {"source": "literal"}
        if overrides and key in overrides:
            origin = {"source": "override"}
        elif param.get("sourceId") and param.get("sourceKey"):
            origin = {
                "source": "connected", "sourceNodeId": str(param["sourceId"]),
                "sourcePortId": str(param["sourceKey"]),
            }
        fields[key] = {"value": safe, **origin}
    return {
        "nodeId": str(node_id), "module": node["module"], "action": node["action"],
        "fields": fields, "omittedFields": omitted,
    }


def bind_generation_input_origins(node_id, node, record, overrides=None, *, source_fields=None):
    """Attach this graph invocation's wiring to the successful call snapshot.

    NodeBase snapshots after normalization/post-processing. Exact adapters may
    replace that snapshot at their own consumed-input boundary. Cache hits keep
    the snapshot of the call that produced the reused output, not raw form text.
    """
    if not isinstance(record, dict):
        return None
    result = deepcopy(record)
    # Budget existing v1/accounting evidence before the newer optional section.
    applied_adapters = result.pop('appliedAdapters', None)
    result.update(nodeId=str(node_id), module=node["module"], action=node["action"])
    params = node.get("params") or {}
    for key, field in result["fields"].items():
        # Backend adapters can rename a consumed argument (for example a
        # generic guidance control to true_cfg_scale). Keep its original wire.
        source = (source_fields or {}).get(key, key)
        param = params.get(source) or {}
        origin = {"source": "literal"}
        if overrides and source in overrides:
            origin = {"source": "override"}
        elif param.get("sourceId") and param.get("sourceKey"):
            origin = {"source": "connected", "sourceNodeId": str(param["sourceId"]),
                      "sourcePortId": str(param["sourceKey"])}
        result["fields"][key] = {"value": field["value"], **origin}
    accounting = result.get('fieldAccounting')
    if accounting is not None:
        # Inspect only parameter names/wires. Never copy an unknown value into
        # the record, even if execution filtered that parameter out.
        literals = {key for key, param in params.items() if isinstance(param, dict)
                    and 'value' in param and not param.get('sourceId')}
        scope = _accounting_scope(node)
        allowed = ACCOUNTING_FIELDS.get(scope, frozenset()) | ACCOUNTING_OBJECT_INPUTS.get(scope, frozenset())
        unknown = {key for key, param in params.items() if isinstance(param, dict)
                   and (param.get('sourceId') or 'value' in param) and key not in allowed}
        accounting['unaccountedFields'] = sorted(set(accounting['unaccountedFields']) |
                                                 (literals - accounting['fields'].keys()) | unknown)
        for key, field in accounting['fields'].items():
            param = params.get(key) or {}
            field.pop('sourceNodeId', None)
            field.pop('sourcePortId', None)
            field['source'] = 'override' if overrides and key in overrides else 'literal'
            if field['source'] != 'override' and param.get('sourceId') and param.get('sourceKey'):
                field.update(source='connected', sourceNodeId=str(param['sourceId']), sourcePortId=str(param['sourceKey']))
        for key, projection in accounting['inputProjections'].items():
            param = params.get(key) or {}
            projection.pop('sourceNodeId', None)
            projection.pop('sourcePortId', None)
            if overrides and key in overrides:
                projection['source'] = 'override'
            elif param.get('sourceId') and param.get('sourceKey'):
                projection.update(source='connected', sourceNodeId=str(param['sourceId']), sourcePortId=str(param['sourceKey']))
            else:
                projection['source'] = 'literal'
        accounting['complete'] = not accounting['omittedFields'] and not accounting['unaccountedFields']
        if len(json.dumps(result, ensure_ascii=True).encode('ascii')) > MAX_RECORD_BYTES:
            result['fieldAccounting'] = {'schemaVersion': 1, 'fields': {}, 'inputProjections': {},
                'outputProjections': {}, 'unaccountedFields': [],
                'omittedFields': {'accounting': 'record-size-limit'}, 'complete': False}
    if applied_adapters is not None:
        result['appliedAdapters'] = applied_adapters
    return result


def build_resolved_execution_inputs(graph, records, *, task_id, attempt_index, node_id):
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, dict) or node_id not in nodes:
        return None
    ancestors, pending = set(), [node_id]
    while pending:
        current = pending.pop()
        if current in ancestors or current not in nodes:
            continue
        ancestors.add(current)
        for param in (nodes[current].get("params") or {}).values():
            if isinstance(param, dict) and param.get("sourceId") and param.get("sourceKey"):
                pending.append(str(param["sourceId"]))
    captured = [deepcopy(records[key]) for key in sorted(ancestors) if key in records]
    if not captured:
        return None
    incomplete = len(captured) > MAX_NODES
    bounded, accounting_records, applied_records = [], [], []
    remaining_bytes = MAX_RECEIPT_BYTES
    for record in captured[:MAX_NODES]:
        # Reserve the legacy ancestry first. Optional accounting must never
        # evict a previously fitting v1 record or clear its summary/task proof.
        accounting = record.pop('fieldAccounting', None)
        applied = record.pop('appliedAdapters', None)
        size = len(json.dumps(record, ensure_ascii=True).encode("ascii"))
        if size > remaining_bytes:
            incomplete = True
            break
        bounded.append(record)
        accounting_records.append(accounting)
        applied_records.append(applied)
        remaining_bytes -= size
    captured = bounded
    accounting_truncated = False
    for record, accounting in zip(captured, accounting_records):
        if accounting is None:
            continue
        base_size = len(json.dumps(record, ensure_ascii=True).encode('ascii'))
        size = len(json.dumps({**record, 'fieldAccounting': accounting}, ensure_ascii=True).encode('ascii')) - base_size
        if size > remaining_bytes:
            accounting_truncated = True
            continue
        record['fieldAccounting'] = accounting
        remaining_bytes -= size
    applied_truncated = False
    for record, applied in zip(captured, applied_records):
        if applied is None:
            continue
        applied_truncated = applied_truncated or applied.get('status') == 'truncated'
        base_size = len(json.dumps(record, ensure_ascii=True).encode('ascii'))
        total_size = len(json.dumps({**record, 'appliedAdapters': applied}, ensure_ascii=True).encode('ascii'))
        if total_size > MAX_RECORD_BYTES:
            applied = {'schemaVersion': 1, 'status': 'truncated', 'reason': 'record-size-limit'}
            applied_truncated = True
            total_size = len(json.dumps({**record, 'appliedAdapters': applied}, ensure_ascii=True).encode('ascii'))
        size = total_size - base_size
        if total_size > MAX_RECORD_BYTES or size > remaining_bytes:
            applied_truncated = True
            continue
        record['appliedAdapters'] = applied
        remaining_bytes -= size
    candidates, unavailable = {}, set()
    for record in captured:
        field_names = _field_names(record)
        for key, field in record["fields"].items():
            canonical = field_names[key]
            values = candidates.setdefault(canonical, [])
            if not any(type(existing) is type(field["value"]) and existing == field["value"] for existing in values):
                values.append(field["value"])
        unavailable.update(field_names[key] for key in record["omittedFields"])
    # Missing records can be unexecuted branches. Never manufacture their values.
    missing = sorted(ancestors - records.keys())
    summary = {key: values[0] for key, values in candidates.items() if len(values) == 1 and key not in unavailable}
    ambiguous = sorted(key for key, values in candidates.items() if len(values) > 1)
    if incomplete:
        unavailable.update(candidates)
        summary = {}
    from modiff.workflow_task_identity import graph_task_receipts

    graph_tasks = (
        graph_task_receipts({key: nodes[key] for key in ancestors}, {r["nodeId"]: r for r in captured})
        if not incomplete and not missing else []
    )
    return {
        **({"graphTasks": graph_tasks} if graph_tasks else {}),
        **({'fieldAccountingStatus': {'schemaVersion': 1, 'truncated': accounting_truncated}}
           if any(item is not None for item in accounting_records) else {}),
        **({'appliedAdaptersStatus': {'schemaVersion': 1, 'truncated': applied_truncated}}
           if any(item is not None for item in applied_records) else {}),
        "schemaVersion": 1, "source": "backend-execution", "taskId": str(task_id),
        "attemptIndex": int(attempt_index or 0), "nodeId": str(node_id),
        "nodes": captured, "summary": summary, "ambiguousFields": ambiguous,
        "unavailableFields": sorted(unavailable), "uncapturedNodeIds": missing,
        "truncated": incomplete,
    }


def apply_resolved_execution_inputs(output, receipt):
    """A trusted, identity-matching receipt takes precedence over form fallbacks."""
    output.pop("resolvedExecutionInputs", None)
    if not isinstance(receipt, dict) or (
        receipt.get("schemaVersion") != 1 or receipt.get("source") != "backend-execution"
        or receipt.get("taskId") != str(output.get("taskId"))
        or receipt.get("attemptIndex") != int(output.get("attemptIndex") or 0)
        or receipt.get("nodeId") != str(output.get("nodeId"))
    ):
        return output
    output["resolvedExecutionInputs"] = deepcopy(receipt)
    summary = receipt["summary"]
    tasks = {item['task'] for item in receipt.get('graphTasks', [])}
    if len(tasks) == 1 and None not in tasks:
        output['mode'] = next(iter(tasks))
    for key in OUTPUT_FIELDS:
        if receipt["truncated"] or key in receipt["ambiguousFields"] or key in receipt["unavailableFields"]:
            output.pop(key, None)
        elif key in summary:
            value = summary[key]
            # Arrays of prompts belong in per-node evidence, not in a single
            # string field that would silently present a different prompt.
            if value is None or isinstance(value, list):
                output.pop(key, None)
            else:
                output[key] = value
    # Old hashes/compatibility receipts describe the form, not resolved wires.
    if receipt["nodes"]:
        output.pop("promptSettingsHash", None)
        output["exactTemplateCompatible"] = False
        for key in ("provenance", "backendProvenance"):
            if isinstance(output.get(key), dict):
                output[key] = {**output[key], "exactTemplateCompatible": False}
                output[key].pop("promptSettingsHash", None)
    return output
