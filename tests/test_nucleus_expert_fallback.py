"""Exercise the real Diffusers expert modules without downloading model weights."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from diffusers.models.transformers.transformer_nucleusmoe_image import SwiGLUExperts
from torch import nn
from torch.nn import functional

from modules.DiffusersImage.main import configure_nucleus_expert_matmul


class _TransformerFixture(nn.Module):
    def __init__(self, grouped):
        super().__init__()
        self.experts = nn.ModuleList([
            SwiGLUExperts(hidden_size=8, moe_intermediate_dim=16, num_experts=4, use_grouped_mm=grouped),
            SwiGLUExperts(hidden_size=8, moe_intermediate_dim=16, num_experts=4, use_grouped_mm=grouped),
        ])
        self.config = SimpleNamespace(use_grouped_mm=grouped)

    def register_to_config(self, **options):
        self.config.use_grouped_mm = options['use_grouped_mm']


class NucleusExpertFallbackTests(unittest.TestCase):
    def test_missing_fused_operator_enables_diffusers_fallback_on_each_expert(self):
        transformer = _TransformerFixture(True)
        with patch.object(functional, "grouped_mm", None, create=True):
            configure_nucleus_expert_matmul(SimpleNamespace(transformer=transformer))
        self.assertFalse(transformer.config.use_grouped_mm)
        self.assertTrue(all(not expert.use_grouped_mm for expert in transformer.experts))

    def test_fused_operator_and_explicit_unfused_configuration_are_preserved(self):
        for original in (True, False):
            with self.subTest(original=original):
                transformer = _TransformerFixture(original)
                with patch.object(functional, "grouped_mm", lambda: None, create=True):
                    configure_nucleus_expert_matmul(SimpleNamespace(transformer=transformer))
                self.assertIs(transformer.config.use_grouped_mm, original)
                self.assertTrue(all(expert.use_grouped_mm is original for expert in transformer.experts))
