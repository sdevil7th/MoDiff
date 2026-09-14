"""Bounded inspection of built-in data controls; runtime preparation markers."""
from __future__ import annotations
from contextvars import ContextVar

RUNTIME_VALUES = ContextVar('workflow_auto_values', default={})
DATA_MODULES = {'modules.Primitive', 'modules.Text', 'modules.Image', 'modules.ImageOperations', 'modules.Audio', 'modules.Video'}


class DeferredResourceValue(ValueError):
    def __init__(self, node_id, field, source_id, source_key, nodes):
        required, pending = set(), [source_id]
        while pending:
            key = pending.pop()
            if key in required:
                continue
            source = nodes[key]
            if source['module'] not in DATA_MODULES:
                raise ValueError(f'{node_id}.{field} is computed by {source_id}; its supplier needs a reviewed data-only preparation contract.')
            required.add(key)
            pending.extend(p['sourceId'] for p in source['params'].values() if p.get('sourceId'))
        self.node_ids = required
        self.target = {'nodeId': node_id, 'field': field, 'sourceId': source_id, 'sourceKey': source_key}
        super().__init__(f'{node_id}.{field} will be resolved by data preparation when Run is pressed.')


def inspect_resource_value(nodes, node_id, field, visited=frozenset()):
    key = (node_id, field)
    if key in visited or len(visited) > 128:
        raise ValueError(f'{node_id}.{field} contains a resource input cycle or exceeds 128 control links.')
    param = nodes[node_id]['params'].get(field, {})
    if not param.get('sourceId'):
        return param.get('value')
    source_id, source_field = param['sourceId'], param.get('sourceKey')
    if (source_id, source_field) in RUNTIME_VALUES.get():
        return RUNTIME_VALUES.get()[(source_id, source_field)]
    source = nodes[source_id]
    def value(name):
        return inspect_resource_value(nodes, source_id, name, visited | {key})
    if source['module'] == 'modules.Primitive':
        if source['action'] in {'String', 'Integer', 'Float', 'Boolean'}:
            return value('value')
        if source['action'] == 'TextValue' and source_field == 'output':
            return value('text') or ''
        if source['action'] == 'ToList' and source_field == 'list':
            return [value(name) for name in source['params'] if name == 'item' or name.startswith('item>>>')]
    if source['module'] == 'modules.Text' and source['action'] == 'ProcessText':
        from modules.Text.main import evaluate_data_operation
        args = {name: value(name) for name, param in source['params'].items() if param.get('display') != 'output' and name not in {'output', 'selected_index', 'item_count'}}
        try:
            outputs = evaluate_data_operation(args)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{source_id}: {error}') from error
        if source_field not in outputs:
            raise ValueError(f'{source_id} has no data output {source_field}.')
        return outputs[source_field]
    raise DeferredResourceValue(node_id, field, source_id, source_field, nodes)
