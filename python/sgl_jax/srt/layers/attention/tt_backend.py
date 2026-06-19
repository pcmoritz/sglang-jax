from __future__ import annotations

import math
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.attention.native_backend import forward_attention
from sgl_jax.srt.layers.attention.tt_sdpa import (
    paged_fill_cache,
    paged_scaled_dot_product_attention_decode,
    paged_update_cache,
    scaled_dot_product_attention,
    scaled_dot_product_attention_decode,
)
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope


@register_pytree_node_class
@dataclass
class TTSDPAMetadata:
    page_table: jax.Array = None
    swa_page_table: jax.Array = None
    cur_pos: jax.Array = None
    fill_page_table: jax.Array = None
    fill_batch_idx: jax.Array = None
    fill_tokens_per_seq: int | None = None

    def tree_flatten(self):
        return (
            self.page_table,
            self.swa_page_table,
            self.cur_pos,
            self.fill_page_table,
            self.fill_batch_idx,
        ), {"fill_tokens_per_seq": self.fill_tokens_per_seq}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, fill_tokens_per_seq=aux_data.get("fill_tokens_per_seq"))


def _decode_page_table(
    cache_loc: np.ndarray,
    seq_lens: np.ndarray,
    page_size: int,
    block_size: int,
    dp_size: int,
    per_dp_bs: int,
    remap: np.ndarray | list[np.ndarray] | None = None,
) -> np.ndarray:
    aligned_lens = ((seq_lens + page_size - 1) // page_size) * page_size
    page_counts = (seq_lens + block_size - 1) // block_size
    max_pages = max(int(page_counts.max(initial=0)), 1)
    page_table = np.zeros((len(seq_lens), max_pages), dtype=np.int32)
    per_dp_loc_len = len(cache_loc) // dp_size

    for dp_rank in range(dp_size):
        row_base = dp_rank * per_dp_bs
        token_base = dp_rank * per_dp_loc_len
        rank_aligned = aligned_lens[row_base : row_base + per_dp_bs]
        offsets = np.zeros(per_dp_bs, dtype=np.int64)
        if per_dp_bs > 1:
            offsets[1:] = np.cumsum(rank_aligned[:-1], dtype=np.int64)
        mapping = remap[dp_rank] if isinstance(remap, list) else remap

        for local_row in range(per_dp_bs):
            row = row_base + local_row
            n_pages = int(page_counts[row])
            if n_pages == 0:
                continue
            start = token_base + int(offsets[local_row])
            locs = cache_loc[start : start + int(rank_aligned[local_row]) : block_size]
            if mapping is not None:
                locs = mapping[locs]
            page_table[row, :n_pages] = locs[:n_pages] // block_size

    return page_table


def _tt_block_size(page_size: int) -> int:
    return max(32, ((page_size + 31) // 32) * 32)


def _pad_page_table(table: jax.Array, min_users: int = 8, min_blocks: int = 16) -> jax.Array:
    pad_users = max(0, min_users - table.shape[0])
    pad_blocks = max(0, min_blocks - table.shape[1])
    if pad_users == 0 and pad_blocks == 0:
        return table
    return jnp.pad(table, ((0, pad_users), (0, pad_blocks)))


def _pad_page_table_host(table: np.ndarray, min_users: int = 8, min_blocks: int = 16) -> np.ndarray:
    pad_users = max(0, min_users - table.shape[0])
    pad_blocks = max(0, min_blocks - table.shape[1])
    if pad_users == 0 and pad_blocks == 0:
        return table
    return np.pad(table, ((0, pad_users), (0, pad_blocks)))


def _pad_decode_positions_host(positions: np.ndarray, min_users: int = 8) -> np.ndarray:
    pad_users = max(0, min_users - positions.shape[0])
    if pad_users == 0:
        return positions
    return np.pad(positions, (0, pad_users), constant_values=-1)


@dataclass
class TTSDPAAttention(AttentionBackend):
    """TTNN paged SDPA decode backend with native fallback for non-decode."""

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        head_dim,
        page_size: int = 1,
        kv_partition_axis: str = "tensor",
        attention_data_partition_axis: str = "data",
        mesh: jax.sharding.Mesh = None,
    ):
        self.num_heads = num_attn_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_attn_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.block_size = _tt_block_size(page_size)
        self.kv_partition_axis = kv_partition_axis
        self.attention_data_partition_axis = attention_data_partition_axis
        self.mesh = mesh
        self.updates_kv_cache_in_place = True
        self.kv_sharding = NamedSharding(self.mesh, P(None, self.kv_partition_axis, None))
        self.forward_metadata = nnx.data(TTSDPAMetadata())

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        metadata = TTSDPAMetadata()
        if batch.forward_mode == ForwardMode.EXTEND:
            self._set_prefill_fill_metadata(batch, metadata)
            return metadata

        if batch.forward_mode != ForwardMode.DECODE:
            return metadata

        if batch.dp_size <= 0:
            raise ValueError(f"Invalid dp_size: {batch.dp_size}")
        if batch.per_dp_bs_size <= 0:
            raise ValueError(f"Invalid per_dp_bs_size: {batch.per_dp_bs_size}")
        if len(batch.cache_loc) % batch.dp_size != 0:
            raise ValueError(
                "Inconsistent cache_loc layout for DP sharding: "
                f"len(cache_loc)={len(batch.cache_loc)} is not divisible by dp_size={batch.dp_size}"
            )

        seq_lens = self._active_decode_seq_lens(batch)
        cache_loc = np.asarray(batch.cache_loc, dtype=np.int32)
        per_dp_bs = max(max(batch.real_bs_per_dp), 1)
        decode_users = max(8, len(batch.input_ids))
        page_table = _decode_page_table(
            cache_loc,
            seq_lens,
            self.page_size,
            self.block_size,
            batch.dp_size,
            per_dp_bs,
        )
        cur_pos = np.where(seq_lens > 0, seq_lens - 1, -1).astype(np.int32)
        page_table = _pad_page_table_host(page_table, min_users=decode_users)
        cur_pos = _pad_decode_positions_host(cur_pos, min_users=decode_users)
        if os.environ.get("SGLANG_TT_DEBUG_ATTENTION") == "1":
            print(
                "TTSDPA metadata decode",
                "seq_lens",
                seq_lens.tolist(),
                "input_ids",
                np.asarray(batch.input_ids, dtype=np.int32)[: min(len(batch.input_ids), 64)].tolist(),
                "positions",
                np.asarray(batch.positions, dtype=np.int32)[: min(len(batch.positions), 64)].tolist(),
                "cache_loc",
                cache_loc[: min(cache_loc.size, 64)].tolist(),
                "page_table",
                page_table.tolist(),
                "cur_pos",
                cur_pos.tolist(),
            )

        metadata.page_table = device_array(
            page_table,
            sharding=NamedSharding(self.mesh, P(self.attention_data_partition_axis, None)),
        )
        metadata.cur_pos = device_array(
            cur_pos,
            sharding=NamedSharding(self.mesh, P(self.attention_data_partition_axis)),
        )

        swa_mapping = getattr(self, "swa_index_mapping", None)
        if swa_mapping is not None:
            metadata.swa_page_table = device_array(
                _pad_page_table_host(
                    _decode_page_table(
                        cache_loc,
                        seq_lens,
                        self.page_size,
                        self.block_size,
                        batch.dp_size,
                        per_dp_bs,
                        remap=swa_mapping,
                    ),
                    min_users=decode_users,
                ),
                sharding=NamedSharding(self.mesh, P(self.attention_data_partition_axis, None)),
            )

        return metadata

    def _active_decode_seq_lens(self, batch: ModelWorkerBatch) -> np.ndarray:
        per_dp_bs = max(max(batch.real_bs_per_dp), 1)
        seq_lens = np.zeros(batch.dp_size * per_dp_bs, dtype=np.int32)
        src_stride = batch.per_dp_bs_size
        src_seq_lens = np.asarray(batch.seq_lens, dtype=np.int32)
        for dp_rank, real_bs in enumerate(batch.real_bs_per_dp):
            if real_bs == 0:
                continue
            src = dp_rank * src_stride
            dst = dp_rank * per_dp_bs
            seq_lens[dst : dst + real_bs] = src_seq_lens[src : src + real_bs]
        return seq_lens

    def _set_prefill_fill_metadata(self, batch: ModelWorkerBatch, metadata: TTSDPAMetadata):
        extend_lens = getattr(batch, "extend_seq_lens", None)
        prefix_lens = getattr(batch, "extend_prefix_lens", None)
        real_bs = int(getattr(batch, "real_bs", len(extend_lens) if extend_lens is not None else 0))
        if extend_lens is not None:
            extend_lens = np.asarray(extend_lens[:real_bs], dtype=np.int32)
        if prefix_lens is not None:
            prefix_lens = np.asarray(prefix_lens[:real_bs], dtype=np.int32)

        if (
            batch.out_cache_loc is None
            or extend_lens is None
            or prefix_lens is None
            or len(extend_lens) == 0
            or np.any(prefix_lens != 0)
            or np.any(extend_lens != extend_lens[0])
        ):
            return

        tokens_per_seq = int(extend_lens[0])
        if tokens_per_seq <= 0:
            return

        num_reqs = len(extend_lens)
        required_tokens = num_reqs * tokens_per_seq
        out_cache_loc = np.asarray(batch.out_cache_loc, dtype=np.int32)
        if out_cache_loc.size < required_tokens:
            return

        loc_2d = out_cache_loc[:required_tokens].reshape(num_reqs, tokens_per_seq)
        if np.any(loc_2d < 0):
            return

        expected_offsets = np.arange(tokens_per_seq, dtype=np.int32)
        if np.any(loc_2d != loc_2d[:, :1] + expected_offsets):
            return

        page_starts = loc_2d[:, :: self.block_size]
        if np.any(page_starts % self.block_size != 0):
            return

        page_table = (page_starts // self.block_size).astype(np.int32, copy=False)
        if os.environ.get("SGLANG_TT_DEBUG_ATTENTION") == "1":
            print(
                "TTSDPA metadata prefill",
                "extend_lens",
                extend_lens.tolist(),
                "prefix_lens",
                prefix_lens.tolist(),
                "out_cache_loc",
                out_cache_loc[: min(out_cache_loc.size, 64)].tolist(),
                "page_table",
                page_table.tolist(),
                "tokens_per_seq",
                tokens_per_seq,
            )
        metadata.fill_page_table = device_array(
            np.pad(
                page_table,
                ((0, max(0, 8 - page_table.shape[0])), (0, max(0, 16 - page_table.shape[1]))),
            ),
            sharding=NamedSharding(self.mesh, P(self.attention_data_partition_axis, None)),
        )
        metadata.fill_batch_idx = device_array(
            np.arange(num_reqs, dtype=np.int32),
            sharding=NamedSharding(self.mesh, P(self.attention_data_partition_axis)),
        )
        metadata.fill_tokens_per_seq = tokens_per_seq

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "kv_partition_axis": self.kv_partition_axis,
            "attention_data_partition_axis": self.attention_data_partition_axis,
            "mesh": self.mesh,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            aux_data["num_heads"],
            aux_data["num_kv_heads"],
            aux_data["head_dim"],
            aux_data["page_size"],
            kv_partition_axis=aux_data.get("kv_partition_axis", "tensor"),
            attention_data_partition_axis=aux_data.get("attention_data_partition_axis", "data"),
            mesh=aux_data.get("mesh"),
        )
        obj.forward_metadata = children[0]
        return obj

    @named_scope
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        **kwargs,
    ):
        use_native_attention = (
            forward_batch.forward_mode != ForwardMode.DECODE
            or kwargs.get("attention_sink") is not None
            or getattr(layer, "xai_temperature_len", -1) > 0
            or self.forward_metadata.page_table is None
            or self.forward_metadata.cur_pos is None
        )
        if os.environ.get("SGLANG_TT_DEBUG_ATTENTION") == "1":
            print(
                "TTSDPA trace",
                "layer",
                layer.layer_id,
                "mode",
                forward_batch.forward_mode,
                "use_native",
                use_native_attention,
                "q",
                q.shape,
                "k",
                k.shape,
                "page_table",
                None if self.forward_metadata.page_table is None else self.forward_metadata.page_table.shape,
                "cur_pos",
                None if self.forward_metadata.cur_pos is None else self.forward_metadata.cur_pos.shape,
                "fill_page_table",
                None
                if self.forward_metadata.fill_page_table is None
                else self.forward_metadata.fill_page_table.shape,
            )
        if use_native_attention:
            if (
                self.forward_metadata.fill_page_table is not None
                and self.forward_metadata.fill_batch_idx is not None
                and self.forward_metadata.fill_tokens_per_seq is not None
                and os.environ.get("SGLANG_TT_DISABLE_PAGED_FILL_CACHE") != "1"
            ):
                if os.environ.get("SGLANG_TT_PREFILL_WITH_UPDATE") == "1":
                    kv_cache = self._paged_update_prefill_cache(
                        token_to_kv_pool, layer.layer_id, k, v
                    )
                else:
                    kv_cache = self._paged_fill_prefill_cache(token_to_kv_pool, layer.layer_id, k, v)
                return self._prefill_attention(q, k, v, kv_cache, layer, **kwargs)
            else:
                kv_cache = token_to_kv_pool.set_kv_buffer_legacy(
                    layer.layer_id, forward_batch.out_cache_loc, k, v
                )
            k_buffer, v_buffer = self._flatten_paged_kv_cache(*kv_cache)
            return self._native_attention(
                q, k_buffer, v_buffer, kv_cache, layer, forward_batch, **kwargs
            )

        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer.layer_id)
        q_heads = q.reshape(q.shape[0], -1, getattr(layer, "head_dim", self.head_dim))
        q_ttnn = q_heads.reshape(1, q_heads.shape[0], q_heads.shape[1], q_heads.shape[2])

        page_table = self.forward_metadata.page_table
        is_swa_layer = layer.sliding_window_size is not None and layer.sliding_window_size > 0
        if is_swa_layer and self.forward_metadata.swa_page_table is not None:
            page_table = self.forward_metadata.swa_page_table

        scale = (
            float(layer.scaling)
            if layer.scaling is not None
            else 1.0 / math.sqrt(getattr(layer, "head_dim", self.head_dim))
        )
        q_ttnn, page_table, cur_pos = self._pad_decode_batch(
            q_ttnn, page_table, self.forward_metadata.cur_pos
        )
        k_update = self._pad_decode_update(
            self._decode_update_value(k, k_cache.shape[-1]), q_ttnn.shape[1]
        )
        v_update = self._pad_decode_update(
            self._decode_update_value(v, v_cache.shape[-1]), q_ttnn.shape[1]
        )

        if os.environ.get("SGLANG_TT_SKIP_DECODE_UPDATE") == "1":
            cur_pos = jnp.maximum(cur_pos - 1, -1)
        else:
            k_cache = paged_update_cache(k_cache, k_update, cur_pos, page_table)
            v_cache = paged_update_cache(v_cache, v_update, cur_pos, page_table)
            k_cache, v_cache = self._cache_barrier((k_cache, v_cache))
        if os.environ.get("SGLANG_TT_DISABLE_PAGED_DECODE") == "1":
            k_buffer, v_buffer = self._flatten_paged_kv_cache(k_cache, v_cache)
            return self._native_attention(
                q,
                k_buffer,
                v_buffer,
                (k_cache, v_cache),
                layer,
                forward_batch,
                **kwargs,
            )
        if os.environ.get("SGLANG_TT_DENSE_TTNN_DECODE") == "1":
            k_dense, v_dense = self._gather_paged_kv_cache(k_cache, v_cache, page_table)
            attn_output = scaled_dot_product_attention_decode(
                q_ttnn,
                k_dense,
                v_dense,
                jnp.maximum(cur_pos, 0),
                scale=scale,
                is_causal=True,
            )
        elif os.environ.get("SGLANG_TT_DENSE_DECODE") == "1":
            attn_output = self._dense_paged_decode(
                q_ttnn,
                k_cache,
                v_cache,
                page_table,
                cur_pos,
                scale,
                layer.sliding_window_size if is_swa_layer else None,
            )
        else:
            attn_output = paged_scaled_dot_product_attention_decode(
                q_ttnn,
                k_cache,
                v_cache,
                page_table,
                cur_pos,
                scale=scale,
                sliding_window_size=layer.sliding_window_size if is_swa_layer else None,
            )
        attn_output = attn_output[:, : q.shape[0], :, :]
        output_sharding = NamedSharding(
            self.mesh, P(self.attention_data_partition_axis, self.kv_partition_axis)
        )
        return attn_output.reshape(q.shape[0], -1, out_sharding=output_sharding), (
            k_cache,
            v_cache,
        )

    def _native_attention(
        self,
        q: jax.Array,
        k_buffer: jax.Array,
        v_buffer: jax.Array,
        kv_cache,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        **kwargs,
    ):
        scale = 1.0 / jnp.sqrt(layer.head_dim) if layer.scaling is None else layer.scaling
        is_causal = not (
            forward_batch.forward_mode == ForwardMode.DECODE
            or layer.attn_type == AttentionType.ENCODER_ONLY
        )
        attention_sink = kwargs.get("attention_sink")
        if attention_sink is not None and hasattr(attention_sink, "value"):
            attention_sink = attention_sink.value

        attn_output = forward_attention(
            q,
            k_buffer,
            v_buffer,
            forward_batch.seq_lens,
            forward_batch.cache_loc,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            layer.q_head_num,
            layer.kv_head_num,
            scale,
            is_causal,
            forward_batch.forward_mode,
            self.kv_sharding,
            mesh=self.mesh,
            xai_temperature_len=getattr(layer, "xai_temperature_len", None),
            attention_sink=attention_sink,
            sliding_window_size=layer.sliding_window_size,
        )
        return attn_output, kv_cache

    def _prefill_attention(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        kv_cache,
        layer: RadixAttention,
        **kwargs,
    ):
        if kwargs.get("attention_sink") is not None:
            return self._manual_prefill_attention(q, k, v, kv_cache, layer, **kwargs)

        tokens_per_seq = self.forward_metadata.fill_tokens_per_seq
        batch = self.forward_metadata.fill_batch_idx.shape[0]
        active_tokens = batch * tokens_per_seq
        total_tokens = q.shape[0]

        q = q[:active_tokens].reshape(batch, tokens_per_seq, q.shape[1], q.shape[2])
        k = k[:active_tokens].reshape(batch, tokens_per_seq, k.shape[1], k.shape[2])
        v = v[:active_tokens].reshape(batch, tokens_per_seq, v.shape[1], v.shape[2])

        padded_tokens = ((tokens_per_seq + 31) // 32) * 32
        if padded_tokens != tokens_per_seq:
            pad_tokens = padded_tokens - tokens_per_seq
            q = jnp.pad(q, ((0, 0), (0, pad_tokens), (0, 0), (0, 0)))
            k = jnp.pad(k, ((0, 0), (0, pad_tokens), (0, 0), (0, 0)))
            v = jnp.pad(v, ((0, 0), (0, pad_tokens), (0, 0), (0, 0)))

        scale = 1.0 / jnp.sqrt(layer.head_dim) if layer.scaling is None else layer.scaling
        output = scaled_dot_product_attention(
            jnp.transpose(q, (0, 2, 1, 3)),
            jnp.transpose(k, (0, 2, 1, 3)),
            jnp.transpose(v, (0, 2, 1, 3)),
            scale=scale,
            is_causal=True,
            sliding_window_size=layer.sliding_window_size,
        )
        output = jnp.transpose(output, (0, 2, 1, 3))[:, :tokens_per_seq]
        output = output.reshape(active_tokens, -1)
        if active_tokens < total_tokens:
            output = jnp.pad(output, ((0, total_tokens - active_tokens), (0, 0)))
        return output, kv_cache

    def _manual_prefill_attention(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        kv_cache,
        layer: RadixAttention,
        **kwargs,
    ):
        tokens_per_seq = self.forward_metadata.fill_tokens_per_seq
        batch = self.forward_metadata.fill_batch_idx.shape[0]
        active_tokens = batch * tokens_per_seq
        total_tokens = q.shape[0]
        q = q[:active_tokens].reshape(batch, tokens_per_seq, q.shape[1], q.shape[2])
        k = k[:active_tokens].reshape(batch, tokens_per_seq, k.shape[1], k.shape[2])
        v = v[:active_tokens].reshape(batch, tokens_per_seq, v.shape[1], v.shape[2])

        if layer.kv_head_num != layer.q_head_num:
            repeats = layer.q_head_num // layer.kv_head_num
            head_sharding = NamedSharding(self.mesh, P(None, None, "tensor", None))
            k = jnp.repeat(k, repeats, axis=2, out_sharding=head_sharding)
            v = jnp.repeat(v, repeats, axis=2, out_sharding=head_sharding)

        scale = 1.0 / jnp.sqrt(layer.head_dim) if layer.scaling is None else layer.scaling
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k) * scale

        positions = jnp.arange(tokens_per_seq)
        causal_mask = positions[:, None] >= positions[None, :]
        if layer.sliding_window_size is not None:
            causal_mask = causal_mask & (
                positions[:, None] - positions[None, :] < layer.sliding_window_size
            )
        mask_value = jnp.asarray(jnp.finfo(logits.dtype).min, logits.dtype)
        logits = jnp.where(causal_mask[None, None, :, :], logits, mask_value)

        attention_sink = kwargs.get("attention_sink")
        if attention_sink is not None and hasattr(attention_sink, "value"):
            attention_sink = attention_sink.value

        max_logit = jnp.max(logits, axis=-1, keepdims=True)
        exp_logits = jnp.exp(logits - max_logit)
        denom = jnp.sum(exp_logits, axis=-1, keepdims=True)
        if attention_sink is not None:
            denom = denom + jnp.exp(attention_sink[None, :, None, None] - max_logit)
        weights = exp_logits / denom
        output_sharding = NamedSharding(self.mesh, P("data", "tensor"))
        output = jnp.einsum("bhqk,bkhd->bqhd", weights, v).reshape(
            active_tokens, -1, out_sharding=output_sharding
        )
        if active_tokens < total_tokens:
            output = jnp.pad(output, ((0, total_tokens - active_tokens), (0, 0)))
        return output, kv_cache

    def _cache_barrier(self, kv_cache: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        return tuple(jax.lax.optimization_barrier(cache) for cache in kv_cache)

    def _decode_update_value(self, value: jax.Array, head_dim: int) -> jax.Array:
        if value.shape[-1] < head_dim:
            value = jnp.pad(value, ((0, 0), (0, 0), (0, head_dim - value.shape[-1])))
        elif value.shape[-1] > head_dim:
            value = value[..., :head_dim]
        return value.reshape(1, value.shape[0], value.shape[1], value.shape[2])

    def _pad_decode_batch(
        self,
        q: jax.Array,
        page_table: jax.Array,
        cur_pos: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        target_users = max(q.shape[1], 8)
        if page_table.shape[0] > target_users:
            page_table = page_table[:target_users]
        if cur_pos.shape[0] > target_users:
            cur_pos = cur_pos[:target_users]
        if q.shape[1] < target_users:
            q = jnp.pad(q, ((0, 0), (0, target_users - q.shape[1]), (0, 0), (0, 0)))
        page_table = _pad_page_table(page_table, min_users=target_users)
        if cur_pos.shape[0] < target_users:
            cur_pos = jnp.pad(cur_pos, (0, target_users - cur_pos.shape[0]), constant_values=-1)
        return q, page_table, cur_pos

    def _pad_decode_update(self, value: jax.Array, users: int) -> jax.Array:
        if value.shape[1] >= users:
            return value
        return jnp.pad(value, ((0, 0), (0, users - value.shape[1]), (0, 0), (0, 0)))

    def _dense_paged_decode(
        self,
        q: jax.Array,
        k_cache: jax.Array,
        v_cache: jax.Array,
        page_table: jax.Array,
        cur_pos: jax.Array,
        scale: float,
        sliding_window_size: int | None,
    ) -> jax.Array:
        users = page_table.shape[0]
        blocks_per_user = page_table.shape[1]
        block_size = k_cache.shape[2]
        kv_heads = k_cache.shape[1]
        head_dim = k_cache.shape[3]
        k, v = self._gather_paged_kv_cache(k_cache, v_cache, page_table)
        tokens = k.shape[2]

        q = q[0]
        if kv_heads != q.shape[1]:
            repeats = q.shape[1] // kv_heads
            repeated_sharding = NamedSharding(
                self.mesh, P(self.attention_data_partition_axis, self.kv_partition_axis, None, None)
            )
            k = jnp.repeat(k, repeats, axis=1, out_sharding=repeated_sharding)
            v = jnp.repeat(v, repeats, axis=1, out_sharding=repeated_sharding)

        logits = jnp.einsum("uhd,uhtd->uht", q, k) * scale
        positions = jnp.arange(tokens, dtype=cur_pos.dtype)
        valid = positions[None, None, :] <= cur_pos[:, None, None]
        if sliding_window_size is not None:
            valid = valid & (positions[None, None, :] > cur_pos[:, None, None] - sliding_window_size)
        mask_value = jnp.asarray(jnp.finfo(logits.dtype).min, logits.dtype)
        logits = jnp.where(valid, logits, mask_value)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        weights = jnp.exp(logits)
        weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
        return jnp.einsum("uht,uhtd->uhd", weights, v)[None, ...]

    def _gather_paged_kv_cache(
        self,
        k_cache: jax.Array,
        v_cache: jax.Array,
        page_table: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        users = page_table.shape[0]
        blocks_per_user = page_table.shape[1]
        block_size = k_cache.shape[2]
        kv_heads = k_cache.shape[1]
        head_dim = k_cache.shape[3]
        tokens = blocks_per_user * block_size

        flat_pages = page_table.reshape(-1)
        gather_sharding = NamedSharding(
            self.mesh, P(self.attention_data_partition_axis, self.kv_partition_axis, None, None)
        )
        k = k_cache.at[flat_pages].get(out_sharding=gather_sharding)
        v = v_cache.at[flat_pages].get(out_sharding=gather_sharding)
        k = k.reshape(users, blocks_per_user, kv_heads, block_size, head_dim)
        v = v.reshape(users, blocks_per_user, kv_heads, block_size, head_dim)
        k = jnp.transpose(k, (0, 2, 1, 3, 4)).reshape(users, kv_heads, tokens, head_dim)
        v = jnp.transpose(v, (0, 2, 1, 3, 4)).reshape(users, kv_heads, tokens, head_dim)
        return k, v

    def _paged_fill_prefill_cache(
        self,
        token_to_kv_pool: KVCache,
        layer_id: int,
        k: jax.Array,
        v: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer_id)
        k_fill = self._prefill_fill_value(k, k_cache.shape[-1])
        v_fill = self._prefill_fill_value(v, v_cache.shape[-1])
        page_table = self.forward_metadata.fill_page_table
        batch_idx = self.forward_metadata.fill_batch_idx
        return self._cache_barrier((
            paged_fill_cache(k_cache, k_fill, page_table, batch_idx),
            paged_fill_cache(v_cache, v_fill, page_table, batch_idx),
        ))

    def _paged_update_prefill_cache(
        self,
        token_to_kv_pool: KVCache,
        layer_id: int,
        k: jax.Array,
        v: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        k_cache, v_cache = token_to_kv_pool.get_kv_buffer(layer_id)
        tokens_per_seq = self.forward_metadata.fill_tokens_per_seq
        batch = self.forward_metadata.fill_batch_idx.shape[0]
        users = self.forward_metadata.fill_page_table.shape[0]
        k_values = self._prefill_update_values(k, k_cache.shape[-1])
        v_values = self._prefill_update_values(v, v_cache.shape[-1])
        page_table = self.forward_metadata.fill_page_table

        for pos in range(tokens_per_seq):
            cur_pos = jnp.full((batch,), pos, dtype=jnp.int32)
            if batch < users:
                cur_pos = jnp.pad(cur_pos, (0, users - batch), constant_values=-1)
            k_update = self._pad_decode_update(k_values[pos : pos + 1], users)
            v_update = self._pad_decode_update(v_values[pos : pos + 1], users)
            k_cache = paged_update_cache(k_cache, k_update, cur_pos, page_table)
            v_cache = paged_update_cache(v_cache, v_update, cur_pos, page_table)

        return self._cache_barrier((k_cache, v_cache))

    def _prefill_update_values(self, value: jax.Array, head_dim: int) -> jax.Array:
        tokens_per_seq = self.forward_metadata.fill_tokens_per_seq
        batch = self.forward_metadata.fill_batch_idx.shape[0]
        value = value[: batch * tokens_per_seq]
        value = self._pad_prefill_head_dim(value, head_dim)
        value = value.reshape(batch, tokens_per_seq, value.shape[1], value.shape[2])
        return jnp.transpose(value, (1, 0, 2, 3))

    def _prefill_fill_value(self, value: jax.Array, head_dim: int) -> jax.Array:
        tokens_per_seq = self.forward_metadata.fill_tokens_per_seq
        batch = self.forward_metadata.fill_batch_idx.shape[0]
        value = value[: batch * tokens_per_seq]
        value = self._pad_prefill_head_dim(value, head_dim)
        value = value.reshape(batch, tokens_per_seq, value.shape[1], value.shape[2])
        return jnp.transpose(value, (0, 2, 1, 3))

    def _pad_prefill_head_dim(self, value: jax.Array, head_dim: int) -> jax.Array:
        if value.shape[-1] < head_dim:
            value = jnp.pad(value, ((0, 0), (0, 0), (0, head_dim - value.shape[-1])))
        elif value.shape[-1] > head_dim:
            value = value[..., :head_dim]
        return value

    def _flatten_paged_kv_cache(
        self,
        k_cache: jax.Array,
        v_cache: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        num_pages, num_heads, page_size, head_dim = k_cache.shape
        flat_shape = (num_pages * page_size, num_heads, head_dim)
        k_3d = jax.lax.reshape(
            jnp.transpose(k_cache, (0, 2, 1, 3)),
            flat_shape,
            out_sharding=P(None, self.kv_partition_axis, None),
        )
        v_3d = jax.lax.reshape(
            jnp.transpose(v_cache, (0, 2, 1, 3)),
            flat_shape,
            out_sharding=P(None, self.kv_partition_axis, None),
        )
        return k_3d, v_3d

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(max_context_len, page_size)
        res = 1024 * 1024 // 2 // num_page_per_req // 4
        assert (
            res > 0
        ), f"max running requests: {res} must larger than 0, please increase page size or decrease max context length"
        return res
