"""TT block verification and noncausal drafting against a dense reference."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.hardware_backend.tt.attention.tt_backend import (
    TTAttention,
    TTTokenToKVPool,
)
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode


def mesh():
    return Mesh(
        np.asarray(jax.devices()).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 2,
    )


def batch():
    return SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=np.array([30, 65, 0], np.int32),
        logits_indices_selector=np.array([0, 1], np.int32),
        spec_info_padded=SimpleNamespace(draft_token_num=4, custom_mask=None),
    )


def test_verify_metadata():
    backend = TTAttention(32, mesh())
    pages = np.pad(np.array([5, 2, 9, 7, 11], np.int32), (0, 11))
    metadata = backend.get_eagle_forward_metadata(batch(), page_indices=pages)
    expected = np.zeros((12, 16), np.int32)
    expected[:4, :2] = [5, 2]
    expected[4:8, :3] = [9, 7, 11]
    np.testing.assert_array_equal(metadata.page_table, expected)
    np.testing.assert_array_equal(metadata.positions, [33] * 4 + [68] * 4 + [-1] * 4)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("trace", [False, True])
def test_block_attention(causal, trace):
    if jax.default_backend() != "tt":
        pytest.skip("requires the TT plugin")
    device = jax.devices()[0]
    tt_mesh = mesh()
    backend = TTAttention(32, tt_mesh)
    pages = np.pad(np.array([5, 2, 9, 7, 11], np.int32), (0, 11))
    backend.forward_metadata = backend.get_eagle_forward_metadata(batch(), page_indices=pages)
    pool = TTTokenToKVPool(16 * 32, 32, jnp.bfloat16, 8, 128, 1, tt_mesh)
    layer = RadixAttention(
        32,
        128,
        128**-0.5,
        8,
        0,
        attn_type=AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY,
    )
    positions = np.concatenate([np.arange(30, 34), np.arange(65, 69), [-1] * 4]).astype(np.int32)
    locations = np.array(
        [
            5 * 32 + 30,
            5 * 32 + 31,
            2 * 32,
            2 * 32 + 1,
            11 * 32 + 1,
            11 * 32 + 2,
            11 * 32 + 3,
            11 * 32 + 4,
        ]
        + [-1] * 4,
        np.int32,
    )
    rng = np.random.default_rng(2)
    host = [
        np.asarray(rng.normal(size=shape), dtype=jnp.bfloat16)
        for shape in [(12, 32, 128), (12, 8, 128), (12, 8, 128), (17, 8, 32, 128), (17, 8, 32, 128)]
    ]
    q, k, v, key_cache, value_cache = host
    pool.kv_buffer[0] = tuple(jax.device_put(x, device) for x in (key_cache, value_cache))

    def forward(q, k, v, pool, backend, positions, locations):
        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY, positions=positions, out_cache_loc=locations
        )
        return backend(q, k, v, layer, forward_batch, pool)

    run = jax.jit(
        forward,
        donate_argnums=(3,),
        compiler_options={
            "optimization_level": "O1",
            "enable_trace": str(trace).lower(),
        },
    )
    for _ in range(2):
        k = np.asarray(rng.normal(size=k.shape), dtype=jnp.bfloat16)
        v = np.asarray(rng.normal(size=v.shape), dtype=jnp.bfloat16)
        output, caches = run(
            *[jax.device_put(x, device) for x in (q, k, v)],
            pool,
            backend,
            jax.device_put(positions, device),
            jax.device_put(locations, device),
        )
        for row, location in enumerate(locations):
            if location >= 0:
                key_cache[location // 32, :, location % 32] = k[row]
                value_cache[location // 32, :, location % 32] = v[row]
        for actual, expected in zip(caches, (key_cache, value_cache)):
            np.testing.assert_array_equal(np.asarray(actual), expected)
        expected = []
        for row in range(8):
            seq = row // 4
            table = [5, 2] if seq == 0 else [9, 7, 11]
            end = int(positions[row]) + 1 if causal else [34, 69][seq]
            keys = (
                key_cache[table].transpose(0, 2, 1, 3).reshape(-1, 8, 128)[:end].astype(np.float32)
            )
            values = (
                value_cache[table]
                .transpose(0, 2, 1, 3)
                .reshape(-1, 8, 128)[:end]
                .astype(np.float32)
            )
            keys, values = (np.repeat(x, 4, axis=1) for x in (keys, values))
            scores = np.einsum("hd,thd->ht", q[row].astype(np.float32), keys) * 128**-0.5
            scores -= scores.max(axis=-1, keepdims=True)
            weights = np.exp(scores)
            weights /= weights.sum(axis=-1, keepdims=True)
            expected.append(np.einsum("ht,thd->hd", weights, values).reshape(-1))
        np.testing.assert_allclose(
            np.asarray(output)[:8].astype(np.float32), expected, atol=0.03, rtol=0.03
        )
        pool.kv_buffer[0] = caches


@pytest.mark.parametrize("overlap,anchor", [(False, False), (True, False), (False, True)])
def test_supported_speculative_options(overlap, anchor, monkeypatch):
    from sgl_jax.srt.server_args import ServerArgs

    monkeypatch.setenv("JAX_PLATFORMS", "tt")
    args = ServerArgs(
        model_path="unused",
        device="tt",
        attention_backend="tt",
        speculative_algorithm="DFLASH",
        speculative_draft_model_path="unused",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=16,
        grammar_backend="none",
        disable_overlap_schedule=not overlap,
        speculative_sample_from_anchor=anchor,
    )
    if overlap or anchor:
        with pytest.raises(ValueError, match="TT DFLASH"):
            args.check_server_args()
    else:
        args.check_server_args()
