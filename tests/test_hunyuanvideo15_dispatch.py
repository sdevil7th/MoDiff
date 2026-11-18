"""Use the ordinary dispatcher and official stages with tiny CPU components."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from modiff.hunyuan_video15_artifacts import HUNYUAN_720P_I2V_REPOSITORY, HUNYUAN_720P_I2V_REVISION


MODEL = 'HunyuanVideo15ModularPipeline'
WORKFLOW = 'image2video'
MODULE = 'modules.ModularDiffusers'


def node(action, **values):
    return {'module': MODULE, 'action': action,
            'params': {key: {'value': value} for key, value in values.items()}}


def connect(target, field, source, output):
    target['params'][field] = {'sourceId': source, 'sourceKey': output}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from diffusers import AutoencoderKLHunyuanVideo15, FlowMatchEulerDiscreteScheduler
    from diffusers import HunyuanVideo15Transformer3DModel
    from diffusers.guiders import ClassifierFreeGuidance
    from diffusers.modular_pipelines import ComponentsManager
    from diffusers.modular_pipelines.hunyuan_video1_5.encoders import HunyuanVideo15TextEncoderStep
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec
    from transformers import ByT5Tokenizer, Qwen2Tokenizer, Qwen2_5_VLTextModel, SiglipImageProcessor
    from transformers import SiglipVisionConfig, SiglipVisionModel, T5Config, T5EncoderModel
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

    import modules
    import modules.ModularDiffusers as modular
    from modules.ModularDiffusers import loaders, workflow_blocks
    from modiff import server as server_module
    from modiff.config import CONFIG

    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    monkeypatch.delenv('MODIFF_SUPERVISOR_QUEUE_STATE', raising=False)
    monkeypatch.delenv('MODIFF_WORKER_SUPERVISED', raising=False)
    for field in ('data', 'images', 'videos', 'audio', 'models', 'upscalers', 'temp'):
        monkeypatch.setitem(CONFIG.paths, field, str(tmp_path / field))
    monkeypatch.setitem(CONFIG.paths, 'work_dir', str(tmp_path))
    monkeypatch.setitem(CONFIG.hf, 'cache_dir', str(tmp_path / 'hub'))
    monkeypatch.setitem(CONFIG.hf, 'online_status', 'Offline')

    manager = ComponentsManager()
    for owner in (modular, loaders, workflow_blocks):
        monkeypatch.setattr(owner, 'components', manager)
    monkeypatch.setattr(server_module.memory_manager, 'cache', {})
    # Preserve the immutable selection and the real ModelsLoader validation.
    index = Path(__file__).parent / 'fixtures/hunyuanvideo15_720p_i2v_index.v1.json'
    snapshot = tmp_path / 'hub/models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-720p_i2v/snapshots'
    snapshot = snapshot / HUNYUAN_720P_I2V_REVISION
    snapshot.mkdir(parents=True)
    (snapshot / 'model_index.json').write_bytes(index.read_bytes())
    observed = {}

    def encode_pixels(pixels):
        # Inspect the real official VAE preparation at full selected pixel
        # geometry without executing the 720p convolutional model on CPU.
        observed['pixels'] = tuple(pixels.shape)
        height, width = pixels.shape[-2:]
        return SimpleNamespace(latents=pixels.new_zeros((1, 32, 1, height // 16, width // 16)))

    def predict_noise(hidden_states, **kwargs):
        observed.setdefault('denoise', []).append((tuple(hidden_states.shape), kwargs['timestep_r']))
        return (torch.zeros_like(hidden_states[:, :32]),)

    def synthetic_load(spec, **options):
        assert spec.pretrained_model_name_or_path == HUNYUAN_720P_I2V_REPOSITORY
        assert spec.revision == HUNYUAN_720P_I2V_REVISION
        assert options['trust_remote_code'] is False
        match spec.name:
            case 'text_encoder':
                value = Qwen2_5_VLTextModel(Qwen2_5_VLTextConfig(
                    vocab_size=16, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=2, bos_token_id=1, eos_token_id=2,
                ))
            case 'tokenizer':
                value = Qwen2Tokenizer(vocab={'<|endoftext|>': 0, 'a': 1}, merges=[])
            case 'text_encoder_2':
                value = T5EncoderModel(T5Config(vocab_size=16, d_model=8, d_ff=16, num_layers=1,
                                               num_heads=2, d_kv=4))
            case 'tokenizer_2':
                value = ByT5Tokenizer()
            case 'image_encoder':
                value = SiglipVisionModel(SiglipVisionConfig(hidden_size=8, intermediate_size=16,
                    num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8))
            case 'feature_extractor':
                value = SiglipImageProcessor(size={'height': 16, 'width': 16})
            case 'scheduler':
                value = FlowMatchEulerDiscreteScheduler()
            case 'guider':
                value = ClassifierFreeGuidance(guidance_scale=6.0)
            case 'transformer':
                value = HunyuanVideo15Transformer3DModel(
                    num_attention_heads=1, attention_head_dim=16, num_layers=1, num_refiner_layers=1,
                    text_embed_dim=8, text_embed_2_dim=8, image_embed_dim=8, rope_axes_dim=(4, 6, 6),
                    target_size=720, use_meanflow=False,
                )
                monkeypatch.setattr(value, 'forward', predict_noise)
            case 'vae':
                value = AutoencoderKLHunyuanVideo15(block_out_channels=(32,) * 5, layers_per_block=1)
                monkeypatch.setattr(value, 'encode', encode_pixels)
            case _:
                pytest.fail('The loader asked for an unexpected pretrained component: ' + spec.name)
        assert isinstance(value, spec.type_hint)
        value._diffusers_load_id = spec.load_id
        return value.eval() if isinstance(value, torch.nn.Module) else value

    def encode_prompt(components, *, prompt, device, dtype, batch_size, num_videos_per_prompt):
        # Check the real stage-read transformer before substituting the text
        # embeddings. The dual neural encoder itself is outside this probe.
        observed.setdefault('text', []).append((components.transformer, dtype, prompt))
        shape = (batch_size * num_videos_per_prompt, 2)
        embeds = torch.ones((*shape, 8), dtype=dtype, device=device)
        mask = torch.ones(shape, dtype=dtype, device=device)
        return embeds, mask, embeds, mask

    monkeypatch.setattr(ComponentSpec, 'load', synthetic_load)
    monkeypatch.setattr(HunyuanVideo15TextEncoderStep, 'encode_prompt', staticmethod(encode_prompt))
    server = server_module.WebServer(modules=modules.MODULE_MAP, work_dir=str(tmp_path),
                                     data_dir=str(tmp_path / 'data'))
    monkeypatch.setattr(server_module, 'server', server)
    monkeypatch.setattr(server, 'queue_message', lambda *args, **kwargs: None)
    source = tmp_path / 'image.png'
    Image.new('RGB', (640, 360), (160, 180, 200)).save(source)
    yield server, source, observed
    server.node_cache.clear()


def test_native_720p_i2v_dispatch_preserves_widget_geometry_and_loader_components(runtime):
    server, source, observed = runtime
    nodes = {
        'model': node('ModelsLoader', model_type=MODEL,
                      repo_id={'source': 'hub', 'value': HUNYUAN_720P_I2V_REPOSITORY},
                      revision=HUNYUAN_720P_I2V_REVISION, workflow_id=WORKFLOW,
                      device='cpu', dtype='float32', auto_offload=False, offload_mode='none'),
        'source': {'module': 'modules.Image', 'action': 'Load', 'params': {'file': {'value': str(source)}}},
        'text': node('WorkflowHunyuanVideo15TextEncode', pipeline_class=MODEL, workflow_id=WORKFLOW,
                     block_path='text_encoder', prompt='A sunny forest', num_videos_per_prompt='1'),
        'vae': node('WorkflowHunyuanVideo15VaeEncode', pipeline_class=MODEL, workflow_id=WORKFLOW,
                    block_path='vae_encoder', width='1280', height='720'),
        'image': node('WorkflowHunyuanVideo15ImageEncode', pipeline_class=MODEL, workflow_id=WORKFLOW,
                      block_path='image_encoder'),
        'denoise': node('WorkflowHunyuanVideo15Denoise', pipeline_class=MODEL, workflow_id=WORKFLOW,
                        block_path='denoise', num_frames='5', num_inference_steps='2', guidance_scale='2.5', seed='123'),
    }
    nodes['second_model'] = deepcopy(nodes['model'])
    for key in ('text', 'vae', 'image', 'denoise'):
        connect(nodes[key], 'pipeline_components', 'model', 'pipeline_components')
    connect(nodes['vae'], 'image', 'source', 'image')
    for parent, child in (('text', 'vae'), ('vae', 'image'), ('image', 'denoise')):
        connect(nodes[child], 'state_in', parent, 'state_out')
    graph = {'sid': None, 'nodes': nodes, 'paths': [list(nodes)]}
    server.current_task = {'task_id': 'cpu-hunyuan-stage-probe', 'sid': None, 'status': 'running',
                           'args': [graph], 'attempt_index': 0, 'interrupt_requested': False}
    server._active_graph_node_ids = set(nodes)
    for name, definition in nodes.items():
        server.current_task['current_node'] = name
        server.execute_node(name, definition, None, quiet=True)
    owner = server.node_cache['model'].loader
    assert observed['text'][0] == (owner.transformer, owner.transformer.dtype, 'A sunny forest')
    assert observed['pixels'] == (1, 3, 1, 720, 1280)
    assert len(observed['denoise']) == 4 and all(
        shape == (1, 65, 2, 45, 80) and meanflow is None for shape, meanflow in observed['denoise']
    )
    state = server.node_cache['denoise'].output['state_out']._state
    assert state.get('latents').shape == (1, 32, 2, 45, 80)
    records = server.node_cache
    vae_fields = records['vae']._execution_input_record['fields']
    assert vae_fields['width']['value'] == 1280
    assert vae_fields['height']['value'] == 720
    denoise_fields = records['denoise']._execution_input_record['fields']
    assert {'width', 'height'}.isdisjoint(denoise_fields)
    for field, value in {'num_frames': 5, 'num_inference_steps': 2, 'guidance_scale': 2.5, 'seed': 123}.items():
        assert denoise_fields[field]['value'] == value

    connect(nodes['denoise'], 'pipeline_components', 'second_model', 'pipeline_components')
    server.current_task['current_node'] = 'denoise'
    with pytest.raises(RuntimeError, match='different Models Loader execution'):
        server.execute_node('denoise', nodes['denoise'], None, quiet=True)
    assert len(observed['denoise']) == 4
