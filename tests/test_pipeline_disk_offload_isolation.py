"""Disk hook filenames must never identify weights across owners or loads."""
import os
from unittest.mock import patch

import pytest
import torch


@pytest.mark.skipif(os.environ.get('MODIFF_TEST_FLUX_VAE_ACCELERATOR') != '1',
                    reason='Explicit accelerator regression; no downloads')
@pytest.mark.parametrize('different_shapes', [False, True])
def test_pipeline_disk_weights_survive_components_reloads_and_repeats(tmp_path, different_shapes):
    from diffusers import DiffusionPipeline
    from modiff.diffusers_offload import apply_pipeline_offload

    class Pipeline(DiffusionPipeline):
        def __init__(self, transformer, controlnet, image_encoder):
            super().__init__()
            self.register_modules(transformer=transformer, controlnet=controlnet,
                                  image_encoder=image_encoder)

    # Same component group names; both equal-shaped silent corruption and
    # different-shaped corruption must be caught. Reload uses the same node.
    for generation in range(2):
        torch.manual_seed(80 + generation)
        components = [torch.nn.Linear(4, 4 + i if different_shapes else 4).eval()
                      for i in range(3)]
        inputs = torch.randn(2, 4, device='cuda:0')
        with torch.no_grad():
            expected = [module.to('cuda:0')(inputs).clone() for module in components]
        for module in components:
            module.to('cpu')
        pipeline = Pipeline(*components)
        with patch('modiff.diffusers_offload._disk_path', return_value=tmp_path):
            result = apply_pipeline_offload(pipeline, mode='group_disk', device='cuda:0',
                                            node_id='same-workflow-node')
        assert result.applied
        with torch.no_grad():
            for _ in range(2):
                for module, reference in zip(components, expected):
                    torch.testing.assert_close(module(inputs), reference, rtol=0, atol=0)
        files_before = set(tmp_path.rglob('*.safetensors'))
        # Re-applying to the SAME owner keeps its valid existing hook storage.
        with patch('modiff.diffusers_offload._disk_path', return_value=tmp_path):
            apply_pipeline_offload(pipeline, mode='group_disk', device='cuda:0',
                                   node_id='same-workflow-node')
        assert set(tmp_path.rglob('*.safetensors')) == files_before
