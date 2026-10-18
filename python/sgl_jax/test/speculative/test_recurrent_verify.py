"""Speculative verify of hybrid recurrent models: the chain verify and the
verify-slot layout that keeps each request's state after its accepted tokens."""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.speculative.kernel import verify_tree_greedy
from sgl_jax.srt.mem_cache.recurrent_state_pool import (
    accepted_state_slots,
    verify_state_slots,
)
from sgl_jax.srt.speculative.eagle_util import (
    build_chain_verify_inputs,
    verify_chain_greedy,
)


@pytest.mark.parametrize("seed", range(20))
def test_chain_verify_matches_tree_walk(seed):
    rng = np.random.default_rng(seed)
    bs, n, vocab = int(rng.integers(1, 5)), int(rng.integers(2, 7)), 7
    logits = rng.normal(size=(bs * n, vocab)).astype(np.float32)
    target = logits.argmax(-1).reshape(bs, n)
    draft = rng.integers(0, vocab, size=(bs, n))
    for row in range(bs):
        # Let a random prefix of each chain match the target.
        accepted = int(rng.integers(0, n))
        draft[row, 1 : accepted + 1] = target[row, :accepted]
    packed = build_chain_verify_inputs(draft[:, 0], draft[:, 1:], np.zeros(bs), n, bs)
    expected = verify_tree_greedy(
        jnp.zeros(bs * n, jnp.int32),
        jnp.full((bs, n), -1, jnp.int32),
        jnp.zeros(bs, jnp.int32),
        jnp.asarray(draft),
        *(packed[i].reshape(bs, n) for i in (2, 3, 4)),
        jnp.asarray(target.reshape(-1)),
    )
    actual = verify_chain_greedy(jnp.asarray(draft.reshape(-1)), jnp.asarray(logits), n)
    for got, want in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))


def test_verify_state_slots_layout():
    slots_per_rank, tokens = 4, 4
    slots = np.array([3, 1, 0, 2])
    states = verify_state_slots(slots, tokens, slots_per_rank)
    # The last token's state goes to the request's own slot, padding stays 0.
    np.testing.assert_array_equal(states[:, -1], slots)
    assert not states[2].any()
    # The others use distinct slots past the pool's own, inside its buffers.
    verify = states[slots > 0, :-1].reshape(-1)
    assert len(set(verify.tolist())) == len(verify)
    assert verify.min() > slots_per_rank
    assert verify.max() < 1 + slots_per_rank * tokens


def test_accepted_state_slots():
    slots_per_rank, tokens = 4, 4
    slots = np.array([3, 1, 0, 2])
    states = verify_state_slots(slots, tokens, slots_per_rank)
    accepted = np.array([1, 4, 1, 3])
    source = accepted_state_slots(slots, accepted, tokens, slots_per_rank)
    # All tokens accepted: the state is already in place; padding copies nothing.
    assert source[1] == 0 and source[2] == 0
    assert source[0] == states[0, 0] and source[3] == states[3, 2]
