from __future__ import annotations

import jax.numpy as jnp
from jax._src import core as jax_core
from jax.extend import core
from jax.interpreters import mlir
from jax.interpreters.mlir import ir, register_lowering
from jaxlib.mlir.dialects import stablehlo


_tt_mark_parameter_p = core.Primitive("tt_mark_parameter")
_tt_weight_dtype_override_p = core.Primitive("tt_weight_dtype_override")

_VALID_WEIGHT_DTYPES = {"bfp_bf4", "bfp_bf8", "bf16"}


def mark_parameter(tensor):
    return _tt_mark_parameter_p.bind(tensor)


def annotate_weight_dtype(tensor, dtype: str):
    if dtype not in _VALID_WEIGHT_DTYPES:
        raise ValueError(
            f"TT weight dtype must be one of {sorted(_VALID_WEIGHT_DTYPES)}, got {dtype}"
        )

    original_shape = tensor.shape
    if tensor.ndim < 3:
        tensor = jnp.reshape(tensor, (1,) * (3 - tensor.ndim) + original_shape)

    tensor = mark_parameter(tensor)
    tensor = _tt_weight_dtype_override_p.bind(tensor, dtype=dtype)

    if len(original_shape) < 3:
        tensor = jnp.reshape(tensor, original_shape)
    return tensor


def _mark_parameter_abstract_eval(tensor):
    return jax_core.ShapedArray(tensor.shape, tensor.dtype)


def _weight_dtype_override_abstract_eval(tensor, *, dtype):
    del dtype
    return jax_core.ShapedArray(tensor.shape, tensor.dtype)


def _create_tt_mark_function(
    module_op: ir.Operation, tensor_type: ir.RankedTensorType
) -> str:
    shape = "x".join(str(dim) for dim in tensor_type.shape)
    element_type = str(tensor_type.element_type)
    func_name = f"tt.mark_{shape}_{element_type}"

    for op in module_op.regions[0].blocks[0].operations:
        if (
            "sym_name" in op.attributes
            and str(op.attributes["sym_name"]).strip('"') == func_name
        ):
            return func_name

    func_type = ir.FunctionType.get([tensor_type], [tensor_type])
    with ir.InsertionPoint.at_block_begin(module_op.regions[0].blocks[0]):
        func_op = ir.Operation.create(
            "func.func",
            attributes={
                "sym_name": ir.StringAttr.get(func_name),
                "function_type": ir.TypeAttr.get(func_type),
                "sym_visibility": ir.StringAttr.get("private"),
            },
            regions=1,
        )
        entry_block = func_op.regions[0].blocks.append(tensor_type)
        with ir.InsertionPoint(entry_block):
            ir.Operation.create("func.return", operands=[entry_block.arguments[0]])

    return func_name


def _module_from_current_insertion_point():
    op = ir.InsertionPoint.current.block.owner
    while op is not None and getattr(op, "name", None) != "builtin.module":
        op = getattr(op, "parent", None)
    return op


def _mark_parameter_lowering(ctx, tensor):
    del ctx
    tensor_type = ir.RankedTensorType(tensor.type)
    module_op = _module_from_current_insertion_point()
    func_name = (
        _create_tt_mark_function(module_op, tensor_type)
        if module_op is not None
        else "tt.mark"
    )
    op = ir.Operation.create(
        "func.call",
        results=[tensor_type],
        operands=[tensor],
        attributes={
            "callee": ir.FlatSymbolRefAttr.get(func_name),
            "ttcore.argument_type": ir.StringAttr.get("parameter"),
        },
    )
    return [op.result]


def _weight_dtype_override_lowering(ctx, tensor, *, dtype):
    result_type = mlir.aval_to_ir_type(ctx.avals_out[0])
    attrs = {
        "mhlo.frontend_attributes": ir.DictAttr.get(
            {"ttcore.weight_dtype": ir.StringAttr.get(dtype)}
        )
    }

    if hasattr(mlir, "custom_call"):
        op = mlir.custom_call(
            "tt.weight_dtype_override",
            result_types=[result_type],
            operands=[tensor],
            extra_attributes=attrs,
        )
        return op.results

    result = stablehlo.custom_call(
        [result_type],
        [tensor],
        ir.StringAttr.get("tt.weight_dtype_override"),
        has_side_effect=ir.BoolAttr.get(False),
        backend_config=ir.StringAttr.get(""),
        api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
    )
    result.owner.attributes["mhlo.frontend_attributes"] = attrs[
        "mhlo.frontend_attributes"
    ]
    return [result]


def _register_tt_lowering(primitive, lowering):
    try:
        register_lowering(primitive, lowering, platform="tt")
    except NotImplementedError:
        register_lowering(primitive, lowering)


_tt_mark_parameter_p.def_abstract_eval(_mark_parameter_abstract_eval)
_register_tt_lowering(_tt_mark_parameter_p, _mark_parameter_lowering)
_tt_weight_dtype_override_p.def_abstract_eval(_weight_dtype_override_abstract_eval)
_register_tt_lowering(_tt_weight_dtype_override_p, _weight_dtype_override_lowering)
