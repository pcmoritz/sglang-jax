from __future__ import annotations

import jax
from jax._src import core as jax_core
from jax.extend import core
from jax.interpreters import mlir
from jax.interpreters.mlir import ir, register_lowering
from jaxlib.mlir.dialects import stablehlo


_tt_paged_sdpa_decode_fused_kv_p = core.Primitive(
    "tt_paged_scaled_dot_product_attention_decode_fused_kv"
)
_tt_sdpa_p = core.Primitive("tt_scaled_dot_product_attention")
_tt_sdpa_decode_p = core.Primitive("tt_scaled_dot_product_attention_decode")
_tt_paged_sdpa_decode_p = core.Primitive("tt_paged_scaled_dot_product_attention_decode")
_tt_paged_update_cache_p = core.Primitive("tt_paged_update_cache")
_tt_paged_fill_cache_p = core.Primitive("tt_paged_fill_cache")


def scaled_dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    scale: float | None,
    is_causal: bool = True,
    sliding_window_size: int | None = None,
) -> jax.Array:
    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5
    return _tt_sdpa_p.bind(
        query,
        key,
        value,
        scale=scale,
        is_causal=is_causal,
        sliding_window_size=sliding_window_size,
    )


def scaled_dot_product_attention_decode(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    cur_pos_tensor: jax.Array,
    *,
    scale: float | None,
    is_causal: bool = True,
) -> jax.Array:
    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5
    return _tt_sdpa_decode_p.bind(
        query,
        key,
        value,
        cur_pos_tensor,
        scale=scale,
        is_causal=is_causal,
    )


def paged_scaled_dot_product_attention_decode(
    query: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    page_table: jax.Array,
    cur_pos_tensor: jax.Array,
    *,
    scale: float | None,
    sliding_window_size: int | None = None,
) -> jax.Array:
    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5
    return _tt_paged_sdpa_decode_p.bind(
        query,
        key_cache,
        value_cache,
        page_table,
        cur_pos_tensor,
        scale=scale,
        sliding_window_size=sliding_window_size,
    )


def paged_scaled_dot_product_attention_decode_fused_kv(
    query: jax.Array,
    fused_kv_cache: jax.Array,
    page_table: jax.Array,
    cur_pos_tensor: jax.Array,
    *,
    scale: float | None,
    sliding_window_size: int | None = None,
) -> jax.Array:
    if scale is None:
        scale = 1.0 / query.shape[-1] ** 0.5
    return _tt_paged_sdpa_decode_fused_kv_p.bind(
        query,
        fused_kv_cache,
        page_table,
        cur_pos_tensor,
        scale=scale,
        sliding_window_size=sliding_window_size,
    )


def _paged_sdpa_fused_kv_abstract_eval(
    query,
    fused_kv_cache,
    page_table,
    cur_pos_tensor,
    *,
    scale,
    sliding_window_size,
):
    del fused_kv_cache, page_table, cur_pos_tensor, scale, sliding_window_size
    return jax_core.ShapedArray(query.shape, query.dtype)


def _paged_sdpa_abstract_eval(
    query,
    key_cache,
    value_cache,
    page_table,
    cur_pos_tensor,
    *,
    scale,
    sliding_window_size,
):
    del key_cache, value_cache, page_table, cur_pos_tensor, scale, sliding_window_size
    return jax_core.ShapedArray(query.shape, query.dtype)


def _sdpa_abstract_eval(
    query,
    key,
    value,
    *,
    scale,
    is_causal,
    sliding_window_size,
):
    del key, value, scale, is_causal, sliding_window_size
    return jax_core.ShapedArray(query.shape, query.dtype)


def _sdpa_decode_abstract_eval(
    query,
    key,
    value,
    cur_pos_tensor,
    *,
    scale,
    is_causal,
):
    del key, value, cur_pos_tensor, scale, is_causal
    return jax_core.ShapedArray(query.shape, query.dtype)


def paged_update_cache(
    cache: jax.Array,
    value: jax.Array,
    update_indices: jax.Array,
    page_table: jax.Array,
    *,
    share_cache: bool = False,
) -> jax.Array:
    return _tt_paged_update_cache_p.bind(
        cache,
        value,
        update_indices,
        page_table,
        share_cache=share_cache,
    )


def _paged_update_cache_abstract_eval(
    cache,
    value,
    update_indices,
    page_table,
    *,
    share_cache,
):
    del value, update_indices, page_table, share_cache
    return jax_core.ShapedArray(cache.shape, cache.dtype)


def paged_fill_cache(
    cache: jax.Array,
    value: jax.Array,
    page_table: jax.Array,
    batch_idx: jax.Array,
) -> jax.Array:
    return _tt_paged_fill_cache_p.bind(cache, value, page_table, batch_idx)


def _paged_fill_cache_abstract_eval(cache, value, page_table, batch_idx):
    del value, page_table, batch_idx
    return jax_core.ShapedArray(cache.shape, cache.dtype)


def _custom_call_attrs(scale, sliding_window_size):
    attrs = {"scale": ir.StringAttr.get(str(float(scale)))}
    if sliding_window_size is not None:
        attrs["sliding_window_size"] = ir.StringAttr.get(str(int(sliding_window_size)))
    return attrs


def _paged_sdpa_decode_attrs(scale, sliding_window_size):
    attrs = _custom_call_attrs(scale, sliding_window_size)
    attrs.update(
        {
            "has_attention_mask": ir.StringAttr.get("False"),
            "has_cur_pos_tensor": ir.StringAttr.get("True"),
            "has_attention_sink": ir.StringAttr.get("False"),
            "is_causal": ir.StringAttr.get("True"),
        }
    )
    return attrs


def _sdpa_attrs(scale, is_causal, sliding_window_size):
    attrs = _custom_call_attrs(scale, sliding_window_size)
    attrs.update(
        {
            "is_causal": ir.StringAttr.get("True" if is_causal else "False"),
            "has_attention_mask": ir.StringAttr.get("False"),
            "has_attention_sink": ir.StringAttr.get("False"),
        }
    )
    return attrs


def _emit_custom_call(ctx, target, operands, attrs):
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
    if hasattr(mlir, "custom_call"):
        extra_attributes = {}
        if attrs:
            extra_attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(attrs)
        op = mlir.custom_call(
            target,
            result_types=[result_type],
            operands=operands,
            extra_attributes=extra_attributes,
        )
        return op.results

    result = stablehlo.custom_call(
        [result_type],
        operands,
        ir.StringAttr.get(target),
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(""),
        api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
    )
    if attrs:
        result.owner.attributes["mhlo.frontend_attributes"] = ir.DictAttr.get(attrs)
    return [result]


def _register_tt_lowering(primitive, lowering):
    try:
        register_lowering(primitive, lowering, platform="tt")
    except NotImplementedError:
        register_lowering(primitive, lowering)


def _sdpa_lowering(
    ctx,
    query,
    key,
    value,
    *,
    scale,
    is_causal,
    sliding_window_size,
):
    return _emit_custom_call(
        ctx,
        "tt.scaled_dot_product_attention",
        [query, key, value],
        _sdpa_attrs(scale, is_causal, sliding_window_size),
    )


def _sdpa_decode_lowering(
    ctx,
    query,
    key,
    value,
    cur_pos_tensor,
    *,
    scale,
    is_causal,
):
    return _emit_custom_call(
        ctx,
        "tt.scaled_dot_product_attention_decode",
        [query, key, value, cur_pos_tensor],
        _sdpa_attrs(scale, is_causal, None),
    )


def _paged_sdpa_fused_kv_lowering(
    ctx,
    query,
    fused_kv_cache,
    page_table,
    cur_pos_tensor,
    *,
    scale,
    sliding_window_size,
):
    return _emit_custom_call(
        ctx,
        "tt.paged_scaled_dot_product_attention_decode_fused_kv",
        [query, fused_kv_cache, page_table, cur_pos_tensor],
        _custom_call_attrs(scale, sliding_window_size),
    )


def _paged_sdpa_lowering(
    ctx,
    query,
    key_cache,
    value_cache,
    page_table,
    cur_pos_tensor,
    *,
    scale,
    sliding_window_size,
):
    return _emit_custom_call(
        ctx,
        "tt.paged_scaled_dot_product_attention_decode",
        [query, key_cache, value_cache, page_table, cur_pos_tensor],
        _paged_sdpa_decode_attrs(scale, sliding_window_size),
    )


def _paged_update_cache_lowering(
    ctx,
    cache,
    value,
    update_indices,
    page_table,
    *,
    share_cache,
):
    return _emit_custom_call(
        ctx,
        "tt.paged_update_cache",
        [cache, value, update_indices, page_table],
        {"share_cache": ir.StringAttr.get("true" if share_cache else "false")},
    )


def _paged_fill_cache_lowering(ctx, cache, value, page_table, batch_idx):
    return _emit_custom_call(
        ctx,
        "tt.paged_fill_cache",
        [cache, value, page_table, batch_idx],
        {},
    )


_tt_sdpa_p.def_abstract_eval(_sdpa_abstract_eval)
_register_tt_lowering(_tt_sdpa_p, _sdpa_lowering)
_tt_sdpa_decode_p.def_abstract_eval(_sdpa_decode_abstract_eval)
_register_tt_lowering(_tt_sdpa_decode_p, _sdpa_decode_lowering)
_tt_paged_sdpa_decode_p.def_abstract_eval(_paged_sdpa_abstract_eval)
_register_tt_lowering(_tt_paged_sdpa_decode_p, _paged_sdpa_lowering)
_tt_paged_sdpa_decode_fused_kv_p.def_abstract_eval(_paged_sdpa_fused_kv_abstract_eval)
_register_tt_lowering(_tt_paged_sdpa_decode_fused_kv_p, _paged_sdpa_fused_kv_lowering)
_tt_paged_update_cache_p.def_abstract_eval(_paged_update_cache_abstract_eval)
_register_tt_lowering(_tt_paged_update_cache_p, _paged_update_cache_lowering)
_tt_paged_fill_cache_p.def_abstract_eval(_paged_fill_cache_abstract_eval)
_register_tt_lowering(_tt_paged_fill_cache_p, _paged_fill_cache_lowering)
