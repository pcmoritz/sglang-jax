"""TT adapter semantics, with CPU references replacing only the device calls."""

import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.hardware_backend.tt.attention import ops
from sgl_jax.srt.hardware_backend.tt.attention.gdn_backend import TTGDNAttnBackend
from sgl_jax.srt.kernels.gdn.gated_delta import (
    _gated_delta_step,
    _scatter_idx0_safe,
    decode_gated_delta_rule_ref,
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_ref,
)


@pytest.fixture(autouse=True)
def isolated_mesh():
    # Some model tests install an explicit mesh at module import time.
    with jax.set_mesh(jax.sharding.Mesh(np.empty((), dtype=object), ())):
        yield


def test_weight_precision_policy(monkeypatch):
    annotations = []

    def annotate(value, dtype):
        annotations.append((value.shape, dtype))
        return value

    monkeypatch.setattr(ops, "annotate_weight_dtype", annotate)
    leaves = (
        jnp.ones((128, 64), jnp.bfloat16),  # Projection weights.
        jnp.ones((128,), jnp.float32),  # Decay, bias, and normalization parameters.
        jnp.ones((), jnp.float32),
        jnp.ones((32,), jnp.int32),
    )
    backend = TTGDNAttnBackend.__new__(TTGDNAttnBackend)
    prepared = backend.prepare_model_state(leaves)
    assert all(a is b for a, b in zip(prepared, leaves))
    assert annotations == [((128, 64), "bfp_bf8"), ((128,), "bf16")]
    # A global weight override could also affect unannotated recurrent matmuls.
    assert "experimental_weight_dtype" not in backend.compiler_options


def reference_chunk(q, k, v, gate, beta, state):
    def step(state, inputs):
        return _gated_delta_step(state, *inputs)

    state, out = jax.lax.scan(step, state[0], (q[0], k[0], v[0], gate[0], beta[0]))
    return state[None], out[None]


def reference_decode(state, q, k, v, b, a, A_log, dt_bias, indices, initial):
    active = jnp.where(initial[:, None, None, None], state[indices], 0)
    gate = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
        a.astype(jnp.float32) + dt_bias.astype(jnp.float32)
    )
    active, out = _gated_delta_step(active, q, k, v, gate, jax.nn.sigmoid(b.astype(jnp.float32)))
    return _scatter_idx0_safe(state, indices, active), out


def reference_conv(state, value, weight, indices, initial):
    out, state = jax_causal_conv1d_update(
        value, state, indices, weight, activation="silu", has_initial_state=initial
    )
    return state, out


def make_backend(device, length, initial, num_k_heads=2, num_v_heads=4):
    backend = TTGDNAttnBackend(
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        mesh=jax.sharding.Mesh(np.array([device]), ("tensor",)),
        dtype=jnp.bfloat16,
    )
    backend.forward_metadata = SimpleNamespace(
        cu_q_lens=jnp.array([0, length], dtype=jnp.int32),
        recurrent_indices=jnp.array([1], dtype=jnp.int32),
        has_initial_state=jnp.array([initial]),
        recurrent_track_indices=None,
    )
    return backend


def inputs(count, num_k_heads=2, num_v_heads=4, seed=35):
    rng = np.random.default_rng(seed)
    channels = (2 * num_k_heads + num_v_heads) * 128

    def rand(shape, scale=0.1, dtype=jnp.bfloat16):
        return jnp.asarray(rng.normal(0, scale, shape), dtype=dtype)

    x = rand((count, channels))
    # The scheduler reserves an immutable zero slot for dummy/fresh requests.
    conv = rand((3, channels, 3)).at[0].set(0)
    state = rand((3, num_v_heads, 128, 128), dtype=jnp.float32).at[0].set(0)
    return (
        x,
        conv,
        state,
        rand((count, num_v_heads)),
        rand((count, num_v_heads)),
        rand((channels, 4)),
        rand((num_v_heads,), dtype=jnp.float32),
        rand((num_v_heads,), dtype=jnp.float32),
    )


def reference(backend, args, decode=False):
    x, conv, state, b, a, weight, A_log, bias = args
    meta = backend.forward_metadata
    if decode:
        y, new_conv = jax_causal_conv1d_update(
            x,
            conv,
            meta.recurrent_indices,
            weight,
            activation="silu",
            has_initial_state=meta.has_initial_state,
        )
        new_state, out = decode_gated_delta_rule_ref(
            y,
            b,
            a,
            state,
            A_log,
            bias,
            meta.recurrent_indices,
            n_kq=backend.num_k_heads,
            n_v=backend.num_v_heads,
            d_k=128,
            d_v=128,
            has_initial_state=meta.has_initial_state,
        )
    else:
        y, new_conv = jax_causal_conv1d_prefill(
            x.T,
            weight,
            cu_seqlens=meta.cu_q_lens,
            conv_state=conv,
            state_indices=meta.recurrent_indices,
            has_initial_state=meta.has_initial_state,
            activation="silu",
        )
        new_state, out = ragged_gated_delta_rule_ref(
            y.T,
            b,
            a,
            state,
            A_log,
            bias,
            cu_seqlens=meta.cu_q_lens,
            state_indices=meta.recurrent_indices,
            has_initial_state=meta.has_initial_state,
            n_kq=backend.num_k_heads,
            n_v=backend.num_v_heads,
            d_k=128,
            d_v=128,
        )
    return out, new_conv, new_state


@pytest.fixture
def reference_ops(monkeypatch):
    monkeypatch.setattr(ops, "gated_delta_rule", reference_chunk)
    monkeypatch.setattr(ops, "gated_delta_decode", reference_decode)
    monkeypatch.setattr(ops, "causal_conv1d_update", reference_conv)
    monkeypatch.setattr(ops, "state_pool_update", _scatter_idx0_safe)


@pytest.mark.parametrize("length", [1, 5, 32, 47, 64])
@pytest.mark.parametrize("initial", [False, True])
def test_prefill(reference_ops, length, initial):
    with jax.default_device(jax.devices("cpu")[0]):
        backend = make_backend(jax.devices("cpu")[0], length, initial)
        args = inputs((length + 31) // 32 * 32)
        expected = reference(backend, args)
        actual = backend.forward_extend(*args, seq_lens=None)
        # The reference leaves padded outputs unspecified; only live tokens
        # and the complete saved state are part of the serving contract.
        expected = (expected[0][:length], *expected[1:])
        np.testing.assert_array_equal(np.asarray(actual[0][length:]), 0)
        actual = (actual[0][:length], *actual[1:])
        for result, wanted in zip(actual, expected):
            np.testing.assert_allclose(
                np.asarray(result, dtype=np.float32),
                np.asarray(wanted, dtype=np.float32),
                rtol=0.01,
                atol=1e-5,
            )


@pytest.mark.parametrize("initial", [False, True])
def test_decode(reference_ops, initial):
    with jax.default_device(jax.devices("cpu")[0]):
        backend = make_backend(jax.devices("cpu")[0], 1, initial)
        args = inputs(1)
        expected = reference(backend, args, decode=True)
        actual = backend.forward_decode(*args)
        for result, wanted in zip(actual, expected):
            np.testing.assert_allclose(
                np.asarray(result, dtype=np.float32),
                np.asarray(wanted, dtype=np.float32),
                rtol=0.01,
                atol=1e-5,
            )


@pytest.mark.parametrize("decode", [False, True])
def test_explicit_serving_mesh(decode):
    cpu = jax.devices("cpu")[0]
    mesh = jax.sharding.Mesh(
        np.array([[cpu]]), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    P = jax.sharding.PartitionSpec
    specs = (P("data", "tensor"),) * 5 + (P("tensor"),) * 3
    with jax.default_device(cpu):
        operands = inputs(1 if decode else 32)
    operands = tuple(
        jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))
        for x, spec in zip(operands, specs)
    )

    meta_sharding = jax.sharding.NamedSharding(mesh, P("data"))
    metadata = tuple(
        jax.device_put(x, meta_sharding)
        for x in (np.array([1], np.int32), np.array([False]), np.array([0, 5], np.int32))
    )

    def forward(indices, initial, lengths, *args):
        backend = make_backend(cpu, 1 if decode else 5, False)
        backend.mesh = mesh
        backend.forward_metadata = SimpleNamespace(
            cu_q_lens=lengths,
            recurrent_indices=indices,
            has_initial_state=initial,
            recurrent_track_indices=None,
        )
        if decode:
            return backend.forward_decode(*args)
        return backend.forward_extend(*args, seq_lens=None)

    # Check JAX's explicit-sharding rules without executing the TT FFI on CPU.
    with jax.set_mesh(mesh):
        result = jax.eval_shape(forward, *metadata, *operands)
    assert result[1].shape == operands[1].shape
    assert result[2].shape == operands[2].shape


@pytest.mark.skipif(
    "tt" not in os.environ.get("JAX_PLATFORMS", "").split(","),
    reason="requires JAX_PLATFORMS=tt,cpu and a Tenstorrent device",
)
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("heads", [(2, 4), (16, 32), (20, 40)])
def test_device_state_handoff(trace, heads):
    """Real kernels: warmup, replay, chunk continuation, slot reuse and padding."""
    cpu, tt = jax.devices("cpu")[0], jax.devices("tt")[0]

    def compile_forward(decode):
        def forward(indices, initial, lengths, *args):
            backend = make_backend(tt, 1, False, *heads)
            backend.forward_metadata = SimpleNamespace(
                cu_q_lens=lengths,
                recurrent_indices=indices,
                has_initial_state=initial,
                recurrent_track_indices=None,
            )
            if decode:
                return backend.forward_decode(*args)
            return backend.forward_extend(*args, seq_lens=None)

        return jax.jit(
            forward,
            donate_argnums=(4, 5),
            compiler_options={"optimization_level": "1", "enable_trace": str(trace).lower()},
        )

    compiled = {decode: compile_forward(decode) for decode in (False, True)}

    def to_device(tree):
        return jax.tree.map(lambda x: jax.device_put(np.asarray(x), tt), tree)

    with jax.default_device(cpu):
        base = inputs(1, *heads)
    host_states, device_states = base[1:3], to_device(base[1:3])
    previous_device_states = tuple(np.asarray(x, dtype=np.float32) for x in host_states)
    weights = to_device(base[5:])
    # (decode, live length, slot, has initial state)
    cases = [(False, 5, 1, False)] + [(True, 1, 1, True)] * 4
    cases += [(False, 17, 1, True)] + [(True, 1, 1, True)] * 4
    cases += [(False, 31, 2, False), (True, 1, 2, True), (True, 1, 0, False)]
    cases += [(False, 47, 2, True), (True, 1, 2, True), (False, 64, 1, False)]
    for step, (decode, length, slot, initial) in enumerate(cases):
        with jax.default_device(cpu):
            count = 1 if decode else (length + 31) // 32 * 32
            sample = inputs(count, *heads, seed=35 + step)
            host = (sample[0], *host_states, *sample[3:5], *base[5:])
            backend = make_backend(cpu, length, initial, *heads)
            backend.forward_metadata.recurrent_indices = jnp.array([slot], jnp.int32)
            expected = reference(backend, host, decode=decode)
        dynamic = to_device((sample[0], *sample[3:5]))
        metadata = to_device(
            (np.array([slot], np.int32), np.array([initial]), np.array([0, length], np.int32))
        )
        actual = compiled[decode](*metadata, dynamic[0], *device_states, *dynamic[1:], *weights)
        for i, (result, wanted) in enumerate(zip(actual, expected)):
            result, wanted = (np.asarray(x, dtype=np.float32) for x in (result, wanted))
            if i == 0:
                if slot == 0:  # Dummy-request output is unspecified.
                    continue
                result, wanted = result[:length], wanted[:length]
            np.testing.assert_allclose(
                result,
                wanted,
                rtol=0.06,
                atol=3e-5 if i == 0 else 1e-3,
                err_msg=f"trace={trace}, step={step}, decode={decode}, output={i}",
            )
            if i > 0:
                untouched = [s for s in range(3) if s != slot or slot == 0]
                np.testing.assert_array_equal(
                    result[untouched], previous_device_states[i - 1][untouched]
                )
        previous_device_states = tuple(np.asarray(x, dtype=np.float32) for x in actual[1:])
        host_states, device_states = expected[1:], actual[1:]


@pytest.mark.skipif(
    "tt" not in os.environ.get("JAX_PLATFORMS", "").split(","),
    reason="requires JAX_PLATFORMS=tt,cpu and a Tenstorrent device",
)
@pytest.mark.parametrize("clear", [False, True])
def test_device_pool_layers_are_independent(clear):
    from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

    tt = jax.devices("tt")[0]
    mesh = jax.sharding.Mesh(np.array([[tt]]), ("data", "tensor"))
    pool = RecurrentStatePool([0, 1], 1, 4, 128, 4, mesh, num_k_heads=2)
    if clear:
        pool.clear()
    update = jax.jit(
        ops.state_pool_update, donate_argnums=(0,), compiler_options={"enable_trace": "false"}
    )
    indices = jax.device_put(np.array([1], np.int32), tt)
    for buffers in (pool.recurrent_buffers, [x[0] for x in pool.conv_buffers]):
        values = jax.device_put(np.ones((1, *buffers[0].shape[1:]), np.dtype(buffers[0].dtype)), tt)
        changed = update(buffers[0], indices, values)
        np.testing.assert_array_equal(np.asarray(changed)[1], 1)
        np.testing.assert_array_equal(np.asarray(buffers[1]), 0)
