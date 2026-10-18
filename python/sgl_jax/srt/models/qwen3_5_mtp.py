"""Qwen3.5 multi-token prediction (MTP) draft model.

Every Qwen3.5 and Qwen3.8 checkpoint ships one MTP layer under ``mtp.*``: a
full-attention decoder layer that predicts the token after next from the
next token's embedding and the target's final hidden state. As in the
upstream torch reference (``sglang/srt/models/qwen3_5_mtp.py``)::

    h = fc(concat(pre_fc_norm_embedding(embed(token)), pre_fc_norm_hidden(hidden)))
    h = norm(decoder_layer(h))

All norms are GemmaRMSNorm. The draft shares the target's embedding and LM
head, which the speculative worker installs with ``set_embed_and_head``.
"""

from __future__ import annotations

import copy
import logging

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import MemoryPools
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.model_loader.weights import WeightLoader, WeightSpec
from sgl_jax.srt.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    _create_qwen3_5_weight_mappings,
)

logger = logging.getLogger(__name__)


class Qwen3_5MTPModel(nnx.Module):
    """The MTP layers. ModelConfig reduces the draft's config to them."""

    def __init__(self, config: PretrainedConfig, mesh: jax.sharding.Mesh, dtype: jnp.dtype):
        text_cfg = config.text_config
        hidden = text_cfg.hidden_size
        self.embed_tokens = Embed(
            num_embeddings=text_cfg.vocab_size,
            features=hidden,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )
        self.pre_fc_norm_embedding = GemmaRMSNorm(hidden, epsilon=text_cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(hidden, epsilon=text_cfg.rms_norm_eps)
        self.fc = LinearBase(
            input_size=2 * hidden,
            output_size=hidden,
            mesh=mesh,
            use_bias=False,
            params_dtype=dtype,
            kernel_axes=(None, None),
            scope_name="fc",
        )
        self.layers = nnx.data(
            [
                Qwen3_5DecoderLayer(config, mesh, layer_id=i, dtype=dtype)
                for i in range(text_cfg.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(hidden, epsilon=text_cfg.rms_norm_eps)

    def __call__(self, forward_batch: ForwardBatch, memory_pools: MemoryPools):
        embeds = self.embed_tokens(forward_batch.input_ids)
        hidden = forward_batch.spec_info.hidden_states
        sharding = jax.typeof(embeds).sharding
        if isinstance(sharding, jax.sharding.NamedSharding):
            hidden = jax.sharding.reshard(hidden, sharding)
        hidden_states, _ = self.fc(
            jnp.concatenate(
                (self.pre_fc_norm_embedding(embeds), self.pre_fc_norm_hidden(hidden)), axis=-1
            )
        )
        residual = None
        layers_kv_fused = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, _ = layer(
                forward_batch.positions, hidden_states, forward_batch, memory_pools, residual
            )
            layers_kv_fused.append(kv_fused)
        return self.norm(hidden_states + residual), layers_kv_fused


class Qwen3_5ForCausalLMMTP(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.mesh = mesh
        self.dtype = dtype
        text_cfg = config.text_config
        if text_cfg.is_moe:
            raise NotImplementedError("Qwen3.5 MTP drafts support dense checkpoints only")
        self.model = Qwen3_5MTPModel(config, mesh, dtype)
        self.lm_head = ParallelLMHead(
            text_cfg.vocab_size,
            text_cfg.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            mesh=mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )
        self.logits_processor = LogitsProcessor(
            text_cfg.vocab_size,
            mesh=mesh,
            enable_dp_lm_head=getattr(config, "enable_dp_lm_head", False),
        )
        self.hot_token_ids = None

    def get_shared_weight_paths(self):
        return ("model.embed_tokens.embedding", "lm_head.embedding")

    def get_embed_and_head(self):
        return self.model.embed_tokens.embedding.value, self.lm_head.embedding.value

    def set_embed_and_head(self, embed: jax.Array, head: jax.Array) -> None:
        self.model.embed_tokens.embedding.value = embed
        self.lm_head.embedding.value = head

    def __call__(
        self,
        forward_batch: ForwardBatch,
        memory_pools: MemoryPools,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused = self.model(forward_batch, memory_pools)
        output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        return output, {"token_to_kv_pool": layers_kv_fused}, True, None

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(self, model_config, self.mesh, dtype=self.dtype)
        loader.load(self._weight_mappings())
        logger.info("Qwen3.5 MTP weights loaded")

    def _weight_mappings(self) -> dict[str, WeightSpec]:
        # The target's table for one full-attention layer, renamed to mtp.*.
        layer_mappings, _, _ = _create_qwen3_5_weight_mappings(self.config, self.lm_head)
        mappings = {}
        for source, spec in layer_mappings.items():
            if not source.startswith("model.language_model.layers."):
                continue
            spec = copy.copy(spec)
            spec.target_path = spec.target_path.replace("language_model.model.", "model.", 1)
            mappings[source.replace("model.language_model.", "mtp.", 1)] = spec
        for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
            mappings[f"mtp.{name}.weight"] = WeightSpec(
                target_path=f"model.{name}.weight", sharding=(None,), transpose=False
            )
        mappings["mtp.fc.weight"] = WeightSpec(
            target_path="model.fc.weight", sharding=(None, None), transpose=True
        )
        return mappings


EntryClass = [Qwen3_5ForCausalLMMTP]
