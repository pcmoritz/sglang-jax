"""FusedEPMoE's experts on TT match a float64 reference on the serving mesh.

TTNN's fused MoE kernel stores the experts in BFP4, so the outputs track the
reference to a correlation of about 0.98 and a relative L2 error of about 0.2.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers import fused_moe

HIDDEN, INTERMEDIATE, EXPERTS, K = 256, 256, 16, 4
# Calibrated for BFP4 experts; an output scaled by 2 has a relative error of 1.
MAX_REL_ERROR = 0.25


def run_experts(tokens, *, padded=0, sharded=False):
    """FusedEPMoE's output on TT and its float64 reference.

    The last ``padded`` tokens get expert ids of -1, as the Qwen models mark
    padding. With ``sharded``, the inputs and the requested output are split
    over the tokens, as with sequence parallelism; otherwise they're replicated.
    """
    if "tt" not in os.environ.get("JAX_PLATFORMS", "").split(","):
        pytest.skip("requires JAX_PLATFORMS=tt,cpu and a Tenstorrent device")
    devices = np.array(jax.devices("tt"))
    mesh = jax.sharding.Mesh(
        devices.reshape(1, -1), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    rng = np.random.default_rng(tokens)
    bf16 = lambda shape, scale: (rng.standard_normal(shape) * scale).astype(jnp.bfloat16)
    x = bf16((tokens, HIDDEN), 0.5)
    wg = bf16((EXPERTS, HIDDEN, INTERMEDIATE), 1 / 16)
    wu = bf16((EXPERTS, HIDDEN, INTERMEDIATE), 1 / 16)
    wd = bf16((EXPERTS, INTERMEDIATE, HIDDEN), 1 / 16)
    ids = np.stack([rng.choice(EXPERTS, K, replace=False) for _ in range(tokens)]).astype(np.int32)
    ids[tokens - padded :] = -1
    weights = rng.random((tokens, K)).astype(np.float32)
    weights /= weights.sum(-1, keepdims=True)

    with jax.set_mesh(mesh):
        layer = fused_moe.FusedEPMoE(
            hidden_size=HIDDEN,
            num_experts=EXPERTS,
            num_experts_per_tok=K,
            ep_size=devices.size,
            intermediate_dim=INTERMEDIATE,
            mesh=mesh,
            weight_dtype=jnp.bfloat16,
            dtype=jnp.bfloat16,
        )
        split = NamedSharding(mesh, P(("data", "tensor"), None, None))
        layer.w1.value, layer.w3.value, layer.w2.value = (
            jax.device_put(w, split) for w in (wg, wu, wd)
        )
        sharding = NamedSharding(mesh, P(("data", "tensor"), None) if sharded else P())
        put = lambda value: jax.device_put(value, sharding)
        run = jax.jit(lambda m, *args: m(*args, out_sharding=sharding))
        out = run(layer, put(x), put(weights), put(ids))
    assert out.sharding.is_equivalent_to(sharding, out.ndim)

    reference = np.zeros(x.shape)
    x64, wg, wu, wd = (np.asarray(a).astype(np.float64) for a in (x, wg, wu, wd))
    for t in range(tokens):
        for weight, e in zip(weights[t], ids[t]):
            if e < 0:
                continue
            g, u = x64[t] @ wg[e], x64[t] @ wu[e]
            reference[t] += weight * ((g / (1 + np.exp(-g)) * u) @ wd[e])
    return np.asarray(out).astype(np.float64), reference


def check(out, reference):
    assert np.corrcoef(out.ravel(), reference.ravel())[0, 1] > 0.97
    error = np.linalg.norm(out - reference) / np.linalg.norm(reference)
    assert error < MAX_REL_ERROR, f"relative L2 error {error:.3f}"


# Decode sizes, a prefill, and one token past libtt's 512-token chunk.
@pytest.mark.parametrize("tokens", [1, 3, 45, 70, 513])
def test_experts(tokens):
    check(*run_experts(tokens))


def test_padding_ids():
    out, reference = run_experts(45, padded=13)
    check(out[:32], reference[:32])
    assert not np.any(out[32:]), "padded tokens got expert outputs"


def test_sharded_tokens():
    check(*run_experts(64, sharded=True))
