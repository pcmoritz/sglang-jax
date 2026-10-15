"""FusedEPMoE on TT, over tt.fused_moe_ep.

FusedEPMoE stores whole experts split over ("data", "tensor"). Every device
runs tt.fused_moe_ep (TTNN's fused MoE kernel through libtt, BFP4 experts) on
all tokens: it computes the (token, expert) pairs of its own experts and
returns each token's routing-weighted sum of them. The layer is the sum over
devices.
"""

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def fused_ep_moe(layer, tokens, topk_weights, topk_ids, out_sharding=None):
    if layer.activation != "silu" or layer.w1_scale is not None or layer.w1_shared is not None:
        raise NotImplementedError("FusedEPMoE on TT runs unquantized SiLU experts, no shared ones")
    axes = ("data", "tensor")

    def experts(x, weights, ids, w1, w3, w2):
        out = jax.ShapeDtypeStruct(x.shape, x.dtype, manual_axis_type=jax.typeof(x).manual_axis_type)
        partial = jax.ffi.ffi_call("tt.fused_moe_ep", out)(x, weights, ids, w1, w3, w2)
        return jax.lax.psum(partial, axes)

    replicated = NamedSharding(layer.mesh, P())
    tokens, topk_weights, topk_ids = (
        jax.sharding.reshard(a, replicated) for a in (tokens, topk_weights, topk_ids)
    )
    split = P(axes, None, None)
    output = jax.shard_map(
        experts, mesh=layer.mesh, in_specs=(P(),) * 3 + (split,) * 3, out_specs=P(), check_vma=False
    )(tokens, topk_weights, topk_ids, layer.w1.value, layer.w3.value, layer.w2.value)
    return jax.sharding.reshard(output, out_sharding or replicated)
