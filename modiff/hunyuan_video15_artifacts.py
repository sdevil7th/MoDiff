"""Exact full 720p artifacts over the existing official Hunyuan stages."""

HUNYUAN_720P_T2V_REPOSITORY = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v'
HUNYUAN_720P_T2V_REVISION = 'f4dbc4a1efa4ac8ea56680cdf79d9f455105e814'
HUNYUAN_720P_I2V_REPOSITORY = 'hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v'
HUNYUAN_720P_I2V_REVISION = 'dcf088ef4e420dd54bc25930ee810bb6613dfcb5'


def stage_authoring_defaults(action, workflow, defaults, capability):
    """Keep I2V geometry in its VAE stage and sampling in its exact recipe."""
    result = dict(defaults)
    if (action == 'WorkflowHunyuanVideo15Decode' and capability.get('defaultRepo') in
            {HUNYUAN_720P_T2V_REPOSITORY, HUNYUAN_720P_I2V_REPOSITORY}):
        result['vae_tiling'] = True
    if action == 'WorkflowHunyuanVideo15Denoise' and workflow == 'image2video':
        result.pop('width', None)
        result.pop('height', None)
        sampling = capability['modeDefaults']['image_to_video']
        result.update(num_inference_steps=sampling['steps'], guidance_scale=sampling['guidanceScale'])
    return result

HUNYUAN_720P_FILES = {
    HUNYUAN_720P_T2V_REPOSITORY: [
        'guider/guider_config.json',
        'model_index.json',
        'scheduler/scheduler_config.json',
        'text_encoder/config.json',
        'text_encoder/model-00001-of-00003.safetensors',
        'text_encoder/model-00002-of-00003.safetensors',
        'text_encoder/model-00003-of-00003.safetensors',
        'text_encoder/model.safetensors.index.json',
        'text_encoder_2/config.json',
        'text_encoder_2/model.safetensors',
        'tokenizer/added_tokens.json',
        'tokenizer/chat_template.jinja',
        'tokenizer/merges.txt',
        'tokenizer/special_tokens_map.json',
        'tokenizer/tokenizer.json',
        'tokenizer/tokenizer_config.json',
        'tokenizer/vocab.json',
        'tokenizer_2/added_tokens.json',
        'tokenizer_2/special_tokens_map.json',
        'tokenizer_2/tokenizer_config.json',
        'transformer/config.json',
        'transformer/diffusion_pytorch_model-00001-of-00007.safetensors',
        'transformer/diffusion_pytorch_model-00002-of-00007.safetensors',
        'transformer/diffusion_pytorch_model-00003-of-00007.safetensors',
        'transformer/diffusion_pytorch_model-00004-of-00007.safetensors',
        'transformer/diffusion_pytorch_model-00005-of-00007.safetensors',
        'transformer/diffusion_pytorch_model-00006-of-00007.safetensors',
        'transformer/diffusion_pytorch_model-00007-of-00007.safetensors',
        'transformer/diffusion_pytorch_model.safetensors.index.json',
        'vae/config.json',
        'vae/diffusion_pytorch_model-00001-of-00002.safetensors',
        'vae/diffusion_pytorch_model-00002-of-00002.safetensors',
        'vae/diffusion_pytorch_model.safetensors.index.json',
    ],
    HUNYUAN_720P_I2V_REPOSITORY: [
        'feature_extractor/preprocessor_config.json',
        'guider/guider_config.json',
        'image_encoder/config.json',
        'image_encoder/model.safetensors',
        'model_index.json',
        'scheduler/scheduler_config.json',
        'text_encoder/config.json',
        'text_encoder/model-00001-of-00003.safetensors',
        'text_encoder/model-00002-of-00003.safetensors',
        'text_encoder/model-00003-of-00003.safetensors',
        'text_encoder/model.safetensors.index.json',
        'text_encoder_2/config.json',
        'text_encoder_2/model.safetensors',
        'tokenizer/added_tokens.json',
        'tokenizer/chat_template.jinja',
        'tokenizer/merges.txt',
        'tokenizer/special_tokens_map.json',
        'tokenizer/tokenizer.json',
        'tokenizer/tokenizer_config.json',
        'tokenizer/vocab.json',
        'tokenizer_2/added_tokens.json',
        'tokenizer_2/special_tokens_map.json',
        'tokenizer_2/tokenizer_config.json',
        'transformer/config.json',
        'transformer/diffusion_pytorch_model-00001-of-00004.safetensors',
        'transformer/diffusion_pytorch_model-00002-of-00004.safetensors',
        'transformer/diffusion_pytorch_model-00003-of-00004.safetensors',
        'transformer/diffusion_pytorch_model-00004-of-00004.safetensors',
        'transformer/diffusion_pytorch_model.safetensors.index.json',
        'vae/config.json',
        'vae/diffusion_pytorch_model.safetensors',
    ],
}
