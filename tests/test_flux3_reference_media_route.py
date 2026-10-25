"""Pure route-resolution regression: no registry, model, device or artifact import.

The actual role-selection statements and resource-consumer walker are extracted
from maintained source. Artifact acquisition is deliberately outside this test;
the selected context keeps those checks on its ordinary execution path.
"""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'modiff/flux3_lora_contract.py'


def resolve(graph, specification, *, with_selectors=False):
    tree = ast.parse(CONTRACT.read_text(encoding='utf-8'))
    selected = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'selected_flux3_context')
    start = next(index for index, node in enumerate(selected.body)
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == 'nodes' for target in node.targets))
    end = next(index for index, node in enumerate(selected.body)
               if isinstance(node, ast.If) and ast.unparse(node.test) == "'guider' in roles")
    selectors = [deepcopy(node) for node in selected.body
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == 'selectors' for target in node.targets)
                 or isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                    and node.target.id == 'selectors']
    body = deepcopy(selected.body[start:end]) + selectors + [ast.Return(
        ast.Tuple([ast.Name('roles', ast.Load()), ast.Name('selectors', ast.Load())], ast.Load()))]
    function = ast.FunctionDef(name='route', args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=name) for name in ('graph', 'owner_id', 'specification')],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=body, decorator_list=[])
    helpers = [deepcopy(node) for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == '_reference_image_source']
    resource = ast.parse((ROOT / 'modiff/workflow_task_identity.py').read_text(encoding='utf-8'))
    resource_nodes = [deepcopy(node) for node in resource.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'resource_consumers'
                      or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                         and target.id == '_RESOURCE_LINK' for target in node.targets)]
    namespace = {'re': re}
    module = ast.fix_missing_locations(ast.Module(body=resource_nodes + helpers + [function], type_ignores=[]))
    exec(compile(module, '<actual-pure-flux3-route>', 'exec'), namespace)
    result = namespace['route'](graph, 'models', specification)
    return result if with_selectors else result[0]


def source_graph():
    roles = {
        'models': 'modules.ModularDiffusers.ModelsLoader',
        'loadImage': 'modules.Image.Load',
        'imageEncode': 'modules.ModularDiffusers.ImageEncode',
        'prompt': 'modules.ModularDiffusers.EncodePrompt',
        'denoise': 'modules.ModularDiffusers.Denoise',
        'decode': 'modules.ModularDiffusers.DecodeLatents',
        'preview': 'modules.Image.Preview',
    }
    edges = [
        ('models', 'text_encoders', 'prompt', 'text_encoders'),
        ('models', 'unet_out', 'denoise', 'unet'),
        ('models', 'scheduler', 'denoise', 'scheduler'),
        ('models', 'vae_out', 'imageEncode', 'vae'),
        ('models', 'vae_out', 'decode', 'vae'),
        ('loadImage', 'image', 'imageEncode', 'image'),
        ('imageEncode', 'image_cond_latents', 'denoise', 'image_cond_latents'),
        ('imageEncode', 'reference_binding', 'prompt', 'reference_binding'),
        ('imageEncode', 'reference_binding', 'denoise', 'reference_binding'),
        ('prompt', 'embeddings', 'denoise', 'embeddings'),
        ('denoise', 'latents', 'decode', 'latents'),
        ('decode', 'images', 'preview', 'image'),
    ]
    nodes = {role: {'module': key.rsplit('.', 1)[0], 'action': key.rsplit('.', 1)[1], 'params': {}}
             for role, key in roles.items()}
    for source, output, target, field in edges:
        nodes[target]['params'][field] = {'sourceId': source, 'sourceKey': output}
    return {'nodes': nodes, 'paths': [list(nodes)]}, {'roles': list(roles.items()), 'edges': edges}


def add_resize(graph, identity='resize', source='loadImage', output='image'):
    graph['nodes'][identity] = {'module': 'modules.ImageOperations', 'action': 'ProcessImage', 'params': {
        'pipeline_class': {'value': 'BuiltinImageOperationV1'},
        'operation': {'value': 'image_upscale'}, 'resize_width': {'value': 256},
        'resize_height': {'value': 256}, 'resize_fit_mode': {'value': 'contain'},
        'resize_resampling': {'value': 'nearest'}, 'mask': {'display': 'input'},
        'image': {'sourceId': source, 'sourceKey': output},
    }}
    graph['paths'][0].append(identity)
    graph['nodes']['imageEncode']['params']['image'] = {'sourceId': identity, 'sourceKey': 'output'}


class ReferenceMediaRouteTests(unittest.TestCase):
    def test_original_direct_load_control(self):
        graph, spec = source_graph()
        self.assertEqual(resolve(graph, spec), {role: role for role, _key in spec['roles']})
        self.assertEqual(resolve(graph, spec, with_selectors=True)[1],
                         tuple(sorted((role, key, role) for role, key in spec['roles'])))

    def test_resize_between_load_and_encoder_preserves_all_core_roles(self):
        graph, spec = source_graph()
        add_resize(graph)
        self.assertEqual(resolve(graph, spec), {role: role for role, _key in spec['roles']})

    def test_two_resizes_and_dimension_edit_do_not_replace_the_media_leaf(self):
        graph, spec = source_graph()
        add_resize(graph)
        add_resize(graph, 'resize2', 'resize', 'output')
        before = resolve(graph, spec, with_selectors=True)
        graph['nodes']['resize2']['params']['resize_width']['value'] = 512
        graph['nodes']['resize2']['params']['resize_height']['value'] = 512
        self.assertEqual(resolve(graph, spec, with_selectors=True), before)
        self.assertEqual(before[1][-2:], (
            ('reference-media:0', 'modules.ImageOperations.ProcessImage', 'resize2'),
            ('reference-media:1', 'modules.ImageOperations.ProcessImage', 'resize')))

    def test_preprocessor_topology_changes_the_route_selector_identity(self):
        graph, spec = source_graph()
        add_resize(graph)
        before = resolve(graph, spec, with_selectors=True)[1]
        add_resize(graph, 'resize2', 'resize', 'output')
        after = resolve(graph, spec, with_selectors=True)[1]
        self.assertNotEqual(before, after)
        graph['nodes']['resize2']['params']['image'] = {'sourceId': 'loadImage', 'sourceKey': 'image'}
        graph['nodes']['resize']['params']['image'] = {'sourceId': 'resize2', 'sourceKey': 'output'}
        graph['nodes']['imageEncode']['params']['image'] = {'sourceId': 'resize', 'sourceKey': 'output'}
        self.assertNotEqual(resolve(graph, spec, with_selectors=True)[1], after)

    def test_unreviewed_suppliers_and_selectors_are_rejected(self):
        for mutation in ('custom', 'model', 'operation', 'pipeline', 'dynamic_operation',
                         'scalar', 'mask', 'mask_value', 'output', 'load_mask', 'missing', 'cycle'):
            with self.subTest(mutation=mutation):
                graph, spec = source_graph()
                add_resize(graph)
                node = graph['nodes']['resize']
                params = node['params']
                if mutation == 'custom':
                    node['module'] = 'custom.unreviewed'
                elif mutation == 'model':
                    node['module'], node['action'] = 'modules.ModularDiffusers', 'DecodeLatents'
                elif mutation == 'operation':
                    params['operation']['value'] = 'image_filter'
                elif mutation == 'pipeline':
                    params['pipeline_class']['value'] = 'UnknownPipeline'
                elif mutation == 'dynamic_operation':
                    params['operation'] = {'sourceId': 'unknown', 'sourceKey': 'value'}
                elif mutation in ('scalar', 'mask'):
                    params['resize_width' if mutation == 'scalar' else 'mask'] = {
                        'sourceId': 'unknown', 'sourceKey': 'value'}
                elif mutation == 'mask_value':
                    params['mask']['value'] = 'unreviewed-mask'
                elif mutation == 'output':
                    graph['nodes']['imageEncode']['params']['image']['sourceKey'] = 'image'
                elif mutation == 'load_mask':
                    params['image']['sourceKey'] = 'mask'
                elif mutation == 'missing':
                    params['image']['sourceId'] = 'missing'
                else:
                    params['image'] = {'sourceId': 'resize', 'sourceKey': 'output'}
                with self.assertRaises(ValueError):
                    resolve(graph, spec)

    def test_preprocessors_must_be_executable_and_outside_loops(self):
        for mutation in ('missing_path', 'loop', 'depth'):
            with self.subTest(mutation=mutation):
                graph, spec = source_graph()
                add_resize(graph)
                if mutation == 'missing_path':
                    graph['paths'][0].remove('resize')
                elif mutation == 'loop':
                    graph['loops'] = [{'bodyNodeIds': ['resize']}]
                else:
                    previous = 'resize'
                    for index in range(64):
                        identity = f'resize-{index}'
                        add_resize(graph, identity, previous, 'output')
                        previous = identity
                with self.assertRaises(ValueError):
                    resolve(graph, spec)

    def test_core_owner_material_unknown_consumer_and_loop_guards_remain(self):
        for mutation in ('owner', 'material', 'binding', 'duplicate', 'unknown', 'path', 'loop'):
            with self.subTest(mutation=mutation):
                graph, spec = source_graph()
                add_resize(graph)
                if mutation == 'owner':
                    graph['nodes']['imageEncode']['params']['vae']['sourceId'] = 'foreign-owner'
                elif mutation == 'material':
                    graph['nodes']['denoise']['params']['embeddings']['sourceKey'] = 'wrong'
                elif mutation == 'binding':
                    graph['nodes']['denoise']['params']['reference_binding']['sourceId'] = 'prompt'
                elif mutation == 'duplicate':
                    graph['nodes']['duplicate'] = deepcopy(graph['nodes']['prompt'])
                elif mutation == 'unknown':
                    graph['nodes']['unknown'] = {'module': 'custom', 'action': 'Unknown',
                        'params': {'model': {'sourceId': 'models', 'sourceKey': 'unet_out'}}}
                elif mutation == 'path':
                    graph['paths'][0].remove('denoise')
                else:
                    graph['loops'] = [{'bodyNodeIds': ['denoise']}]
                with self.assertRaises(ValueError):
                    resolve(graph, spec)


class FullSelectedContextTests(unittest.TestCase):
    """Run the whole context issuer; substitute only artifact acquisition."""

    def graph(self, model_type, weight_format='native'):
        from modiff.native_component_lora import reviewed_flux3_component_declaration
        from modiff.studio_execution_specs import studio_execution_spec_for_pair
        declaration = reviewed_flux3_component_declaration(model_type, 'edit_image')
        specification = studio_execution_spec_for_pair(
            model_type, 'edit_image', execution_profile_id=declaration.profile_id)
        self.assertNotIn('requiredDiffusersProvider', specification)
        self.assertEqual(specification['executionProfileId'], declaration.profile_id)
        nodes = {role: {'module': key.rsplit('.', 1)[0], 'action': key.rsplit('.', 1)[1], 'params': {}}
                 for role, key, *_ in specification['roles']}
        for source, output, target, field in specification['edges']:
            nodes[target]['params'][field] = {'sourceId': source, 'sourceKey': output}
        values = dict(model_type=model_type, execution_profile_id=declaration.profile_id,
                      repo_id={'source': 'hub', 'value': declaration.repository},
                      reviewed_variant=declaration.repository, revision=declaration.revision,
                      workflow_id='image_conditioned', dtype='bfloat16', device='cuda:0',
                      offload_mode='none', auto_offload=False, weight_loading_mode='cpu',
                      trust_remote_code=False, weight_format=weight_format)
        nodes['models']['params'] = {key: {'value': value} for key, value in values.items()}
        if 'guider' in nodes:
            nodes['guider']['params']['model_type'] = {'value': model_type}
        return {'nodes': nodes, 'paths': [list(nodes)]}, specification, declaration

    def issue(self, graph, specification, declaration, calls, source='unit-source'):
        from modiff.diffusers_profiles import DIFFUSERS_EXECUTION_PROFILES
        from modiff.flux3_lora_contract import selected_flux3_context
        loader = ModuleType('modules.ModularDiffusers.loaders')

        def acquire(**kwargs):
            self.assertEqual(kwargs, dict(model_type=declaration.pipeline_class,
                repo_id={'source': 'hub', 'value': declaration.repository}, revision=declaration.revision))
            calls.append(kwargs)
            return 'hub', declaration.repository, declaration.revision, 'unit-index.json', {'unit': 'index'}

        loader.ModelsLoader = type('ModelsLoader', (), {'_preflight_reviewed_builtin_selection': staticmethod(acquire)})
        loader._reviewed_json_fingerprint = lambda value: hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        with patch.dict(sys.modules, {'modules.ModularDiffusers.loaders': loader}):
            return selected_flux3_context(graph, source, 'models',
                DIFFUSERS_EXECUTION_PROFILES[specification['executionProfileId']], declaration)

    def test_full_context_retains_real_artifact_placement_and_resize_identity(self):
        from modiff.modular_weight_formats import selected_weight_format_contract
        from modiff.native_component_lora import ComponentLoraContext
        heavy_before = {name for name in sys.modules if name.split('.')[0] in ('torch', 'diffusers', 'transformers')}
        for model in ('Flux3ModularPipeline', 'Flux3DistilledModularPipeline'):
            for weight_format in ('native', 'fp8r'):
                with self.subTest(model=model, weight_format=weight_format):
                    graph, specification, declaration = self.graph(model, weight_format)
                    calls = []
                    direct = self.issue(graph, specification, declaration, calls)
                    self.assertIs(type(direct), ComponentLoraContext)
                    self.assertEqual(direct.stage_selectors, tuple(sorted(
                        (role, key, role) for role, key, *_ in specification['roles'])))
                    self.assertEqual(direct.declaration, declaration)
                    self.assertEqual((direct.source_fingerprint, direct.dtype, direct.device),
                                     ('unit-source', 'bfloat16', 'cuda:0'))
                    self.assertEqual((direct.specification_id, direct.specification_hash),
                                     (specification['id'], specification['contentHash']))
                    self.assertIsNone(direct.placement)
                    self.assertEqual(direct.weight_format_contract, selected_weight_format_contract(
                        model_type=model, repository=declaration.repository,
                        revision=declaration.revision, weight_format=weight_format))
                    add_resize(graph)
                    resized = self.issue(graph, specification, declaration, calls)
                    self.assertNotEqual(resized.signature, direct.signature)
                    graph['nodes']['resize']['params']['resize_width']['value'] = 512
                    graph['nodes']['resize']['params']['resize_height']['value'] = 512
                    self.assertEqual(self.issue(graph, specification, declaration, calls).signature, resized.signature)
                    self.assertEqual(len(calls), 3)
        self.assertEqual({name for name in sys.modules
                          if name.split('.')[0] in ('torch', 'diffusers', 'transformers')}, heavy_before)

    def test_full_context_rejects_source_and_core_faults_before_artifact_acquisition(self):
        for fault in ('revision', 'repository', 'variant', 'workflow', 'placement', 'override',
                      'owner', 'binding', 'processor', 'path', 'loop', 'source'):
            with self.subTest(fault=fault):
                graph, specification, declaration = self.graph('Flux3DistilledModularPipeline')
                add_resize(graph)
                params = graph['nodes']['models']['params']
                if fault in ('revision', 'variant', 'workflow', 'placement'):
                    field, value = {'revision': ('revision', 'main'), 'variant': ('reviewed_variant', 'other/model'),
                                    'workflow': ('workflow_id', 'text2image'),
                                    'placement': ('offload_mode', 'unreviewed')}[fault]
                    params[field] = {'value': value}
                elif fault == 'repository':
                    params['repo_id'] = {'value': {'source': 'local', 'value': 'other/model'}}
                elif fault == 'override':
                    params['unet'] = {'sourceId': 'foreign', 'sourceKey': 'unet_out'}
                elif fault == 'owner':
                    graph['nodes']['imageEncode']['params']['vae']['sourceId'] = 'foreign'
                elif fault == 'binding':
                    graph['nodes']['denoise']['params']['reference_binding']['sourceId'] = 'prompt'
                elif fault == 'processor':
                    graph['nodes']['resize']['params']['operation']['value'] = 'unreviewed'
                elif fault == 'path':
                    graph['paths'][0].remove('resize')
                elif fault == 'loop':
                    graph['loops'] = [{'bodyNodeIds': ['resize']}]
                calls = []
                with self.assertRaises(ValueError):
                    self.issue(graph, specification, declaration, calls, source='' if fault == 'source' else 'unit-source')
                self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
