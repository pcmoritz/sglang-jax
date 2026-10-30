"""Speculative verify of hybrid recurrent models on Tenstorrent: the verify-slot
layout that keeps each request's state after its accepted tokens."""

import numpy as np

from sgl_jax.srt.hardware_backend.tt.attention.gdn_backend import (
    accepted_state_slots,
    verify_state_slots,
)


def test_verify_state_slots_layout():
    slots_per_rank, tokens = 4, 4
    slots = np.array([3, 1, 0, 2])
    states = np.asarray(verify_state_slots(slots, tokens, slots_per_rank))
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


def test_accepted_state_slots_match_the_layout():
    # Every accept count, including out-of-range ones, against the verify layout.
    slots_per_rank = 5
    slots = np.array([0, 1, 2, 3, 4, 5])
    for tokens in range(1, 9):
        states = np.asarray(verify_state_slots(slots, tokens, slots_per_rank))
        for accepted in range(-1, tokens + 2):
            count = min(max(accepted, 1), tokens)
            expected = np.where((slots > 0) & (count < tokens), states[:, count - 1], 0)
            source = accepted_state_slots(slots, np.full_like(slots, accepted), tokens, slots_per_rank)
            np.testing.assert_array_equal(np.asarray(source), expected)
