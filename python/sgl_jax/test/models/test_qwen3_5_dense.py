"""CPU coverage for dense Qwen3.5 using the small official 0.8B fixture."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import jax
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

import sgl_jax.srt.hf_transformers_utils  # noqa: F401
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

DENSE_FIXTURE = os.environ.get("QWEN3_5_DENSE_FIXTURE", "Qwen/Qwen3.5-0.8B")
DENSE_REVISION = os.environ.get("QWEN3_5_DENSE_REVISION") or None

_mesh = create_device_mesh(
    ici_parallelism=[1, 1], dcn_parallelism=[1, 1], devices=[jax.devices()[0]]
)
jax.sharding.set_mesh(_mesh)


class TestDenseQwen3_5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.cfg = AutoConfig.from_pretrained(
                DENSE_FIXTURE, revision=DENSE_REVISION, trust_remote_code=False
            )
        except OSError as e:
            if os.path.isdir(DENSE_FIXTURE):
                raise
            raise unittest.SkipTest(
                f"Qwen3.5 fixture {DENSE_FIXTURE!r} unavailable: {e}"
            ) from e

    def test_dense_config_and_registry(self):
        from sgl_jax.srt.models.registry import ModelRegistry

        tc = self.cfg.text_config
        self.assertEqual(type(self.cfg).__name__, "Qwen3_5Config")
        self.assertEqual(type(tc).__name__, "_Qwen3_5DenseTextConfig")
        self.assertEqual(self.cfg.model_type, "qwen3_5")
        self.assertEqual(tc.hidden_size, 1024)
        self.assertEqual(tc.intermediate_size, 3584)
        self.assertEqual(tc.num_hidden_layers, 24)
        self.assertEqual(tc.full_attention_layer_ids, list(range(3, 24, 4)))
        self.assertEqual(tc.num_experts, 0)

        model_cls, architecture = ModelRegistry.resolve_model_cls(self.cfg.architectures)
        self.assertEqual(architecture, "Qwen3_5ForConditionalGeneration")
        self.assertEqual(model_cls.__name__, architecture)

    def test_decoder_uses_dense_mlp(self):
        from sgl_jax.srt.configs.qwen3_5 import Qwen3_5Config
        from sgl_jax.srt.models.qwen3 import Qwen3MLP
        from sgl_jax.srt.models.qwen3_5 import Qwen3_5DecoderLayer

        tiny = Qwen3_5Config(
            text_config={
                "vocab_size": 64,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 16,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 2,
                "rope_parameters": {
                    "rope_type": "default",
                    "mrope_section": [1, 1, 0],
                    "mrope_interleaved": True,
                    "rope_theta": 1.0e6,
                    "partial_rotary_factor": 0.25,
                },
            }
        )
        for layer_id in (0, 3):
            layer = Qwen3_5DecoderLayer(tiny, _mesh, layer_id)
            self.assertFalse(layer.is_moe)
            self.assertIsInstance(layer.mlp, Qwen3MLP)

    def test_weight_mapping_covers_official_index(self):
        from sgl_jax.srt.models.qwen3_5 import _create_qwen3_5_weight_mappings

        if os.path.isdir(DENSE_FIXTURE):
            index_path = Path(DENSE_FIXTURE) / "model.safetensors.index.json"
        else:
            index_path = Path(
                hf_hub_download(
                    repo_id=DENSE_FIXTURE,
                    filename="model.safetensors.index.json",
                    revision=DENSE_REVISION,
                )
            )
        with open(index_path) as f:
            checkpoint_keys = set(json.load(f)["weight_map"])

        mappings, _, _ = _create_qwen3_5_weight_mappings(self.cfg)
        text_keys = {k for k in checkpoint_keys if k.startswith("model.language_model.")}
        self.assertEqual(len(text_keys), 320)
        expected = text_keys | ({"lm_head.weight"} & checkpoint_keys)
        self.assertEqual(set(mappings), expected)


if __name__ == "__main__":
    unittest.main()
