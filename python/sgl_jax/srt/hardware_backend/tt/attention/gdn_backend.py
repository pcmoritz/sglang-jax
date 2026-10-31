"""TT recurrent attention: explicit kernels with scheduler-owned state slots."""

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.hardware_backend.tt.attention import ops
from sgl_jax.srt.hardware_backend.tt.attention.tt_backend import (
    MAX_DECODE_USERS,
    TTAttention,
)
from sgl_jax.srt.kernels.gdn.gated_delta import _l2norm
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    LinearRecurrentAttnBackendMetadata,
)
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.jax_utils import device_array


@jax.tree_util.register_pytree_node_class
@dataclass
class TTGDNMetadata(LinearRecurrentAttnBackendMetadata):
    max_prefill_len: int = 0
    # Speculative verify: draft tokens per request, 0 otherwise.
    verify_tokens: int = 0

    def tree_flatten(self):
        children, _ = super().tree_flatten()
        return children, (self.max_prefill_len, self.verify_tokens)

    @classmethod
    def tree_unflatten(cls, aux, children):
        meta = super().tree_unflatten({}, children)
        meta.max_prefill_len, meta.verify_tokens = aux
        return meta


def _per_device(forward):
    """Runs forward on each device's heads, like GDNAttnBackend, passing it
    the forward metadata after self."""

    heads = P("data", "tensor", None)
    state = P("data", "tensor", None, None)
    in_specs = (
        P("data"),  # forward metadata
        P("data", "tensor"),  # mixed_qkv
        heads,  # conv_state
        state,  # recurrent_state
        P("data", "tensor"),  # b
        P("data", "tensor"),  # a
        P("tensor", None),  # conv1d weight
        P("tensor"),  # A_log
        P("tensor"),  # dt_bias
    )

    @functools.wraps(forward)
    def run(self, *args, **kwargs):
        return jax.shard_map(
            lambda *args: forward(self, *args, **kwargs),
            mesh=self.mesh,
            in_specs=in_specs,
            out_specs=(heads, heads, state),
        )(self.forward_metadata, *args)

    return run


def verify_state_slots(slots, tokens: int, slots_per_rank: int):
    """[B, tokens] slots for the state after each of a verify's tokens.

    The state after the last token goes to the request's own slot, the others
    to the request's verify slots. Rows with slot 0 (padding) stay 0.
    """
    steps = jnp.arange(tokens, dtype=jnp.int32)
    first = slots_per_rank + 1 + (slots - 1) * (tokens - 1)
    states = jnp.where(steps == tokens - 1, slots[:, None], first[:, None] + steps)
    return jnp.where(slots[:, None] > 0, states, 0).astype(jnp.int32)


def accepted_state_slots(slots, accepted, tokens: int, slots_per_rank: int):
    """The verify slot holding each request's state after its `accepted`
    tokens (of `tokens`), or 0 where the request's own slot already holds it
    or the row is padding."""
    states = verify_state_slots(slots, tokens, slots_per_rank)
    count = jnp.clip(accepted, 1, tokens)
    picked = jnp.arange(tokens) == (count - 1)[:, None]
    source = jnp.where(picked, states, 0).sum(axis=1)
    return jnp.where(count < tokens, source, 0).astype(jnp.int32)


@functools.cache
def _keep_states(mesh, tokens: int, slots_per_rank: int):
    """A program that, after a verify of `tokens` tokens per request, copies
    each request's state after its accepted tokens from its verify slot to
    its own slot, in place, in every layer's recurrent and convolution pools."""

    def copy(pool, source, target):
        for start in range(0, source.shape[0], MAX_DECODE_USERS):
            rows = slice(start, start + MAX_DECODE_USERS)
            pool = ops.state_pool_copy(pool, source[rows], target[rows])
        return pool

    def keep(recurrent, conv, slots, accepted):
        source = accepted_state_slots(slots, accepted, tokens, slots_per_rank)
        target = jnp.where(source > 0, slots, 0)
        rows = P("data")
        copy_states = jax.shard_map(
            copy,
            mesh=mesh,
            in_specs=(P("data", "tensor", None, None), rows, rows),
            out_specs=P("data", "tensor", None, None),
        )
        copy_windows = jax.shard_map(
            copy,
            mesh=mesh,
            in_specs=(P("data", "tensor", None), rows, rows),
            out_specs=P("data", "tensor", None),
        )
        return (
            [copy_states(pool, source, target) for pool in recurrent],
            [[copy_windows(pool, source, target) for pool in pools] for pools in conv],
        )

    return jax.jit(keep, donate_argnums=(0, 1), compiler_options=TTGDNAttnBackend.compiler_options)


class TTGDNAttnBackend(GDNAttnBackend):
    # Quantize only annotated matrix weights; recurrent arithmetic stays FP32.
    compiler_options = {
        key: value
        for key, value in TTAttention.compiler_options.items()
        if key != "experimental_weight_dtype"
    }
    # Materialize weight transposes once instead of strided DRAM reads on decode.
    compiler_options["experimental_enable_permute_matmul_fusion"] = "false"

    def __init__(self, **kwargs):
        kwargs["prefill_impl"] = "chunked_jax"
        super().__init__(**kwargs)
        if self.mesh.shape.get("data", 1) != 1:
            raise NotImplementedError("TT GDN currently supports tensor parallelism only")
        if (self.head_k_dim, self.head_v_dim, self.conv_kernel_size) != (128, 128, 4):
            raise NotImplementedError("TT GDN requires 128-wide heads and a four-tap convolution")

    def get_forward_metadata(self, batch):
        if batch.forward_mode == ForwardMode.TARGET_VERIFY:
            return self._verify_metadata(batch)
        meta = super().get_forward_metadata(batch)
        if batch.forward_mode.is_extend():
            # Prefill packs each live sequence; exclude the scheduler's dummy rows.
            size = batch.real_bs
            # A static slice runs on the mesh; an indexed get would build its
            # index as a single-device array, a separate program on one device.
            meta.cu_q_lens = jax.lax.slice_in_dim(meta.cu_q_lens, 0, size + 1)
            # The native kernel needs dense sequences. Bucket the longest one
            # instead of padding every sequence to the entire batch length.
            length = int(batch.extend_seq_lens[:size].max())
            meta = TTGDNMetadata(
                **vars(meta), max_prefill_len=1 << (max(length, 32) - 1).bit_length()
            )
        return meta

    def keep_accepted(self, pool, accepted, tokens: int) -> None:
        """After the current verify, keep each request's state after its
        `accepted` tokens (one count per verify row, on device). The copy is in
        place: the next forward's trace was captured with the pool's current
        buffers."""
        keep = _keep_states(self.mesh, tokens, pool.slots_per_rank)
        pool.replace_buffer(
            keep(
                pool.recurrent_buffers,
                pool.conv_buffers,
                self.forward_metadata.recurrent_indices,
                accepted,
            )
        )

    def _verify_metadata(self, batch):
        """Each request verifies a linear chain of draft tokens."""
        size = len(batch.seq_lens)
        # Padding rows and precompile's dummy batches, which come without
        # recurrent slots, use slot 0, which the kernels never write. A verified
        # request always has a state.
        slots = np.zeros(size, dtype=np.int32)
        if batch.recurrent_indices is not None:
            indices = np.asarray(batch.recurrent_indices)[:size]
            slots[: len(indices)] = indices
        meta = TTGDNMetadata(verify_tokens=batch.spec_info_padded.draft_token_num)
        meta.recurrent_indices, meta.has_initial_state = device_array(
            (slots, np.ones(size, dtype=bool)),
            sharding=NamedSharding(self.mesh, P("data")),
        )
        return meta

    def _metadata(self, meta):
        if meta.recurrent_track_indices is not None:
            raise NotImplementedError("TT GDN recurrent snapshots are not yet supported")
        return meta, meta.recurrent_indices, meta.has_initial_state

    def _qkv(self, mixed):
        shape = mixed.shape[:-1]
        # One device's heads. The kernels let each q/k head serve its group of
        # value heads, so q and k are not repeated.
        key_dim = self.key_dim // self.mesh.shape["tensor"]
        qk = mixed[..., : 2 * key_dim].reshape(*shape, -1, self.head_k_dim)
        v = mixed[..., 2 * key_dim :].reshape(*shape, -1, self.head_v_dim)
        q, k = jnp.split(_l2norm(qk.astype(jnp.float32)), 2, axis=-2)
        return q * self.head_k_dim**-0.5, k, v.astype(jnp.float32)

    @_per_device
    def forward_decode(
        self,
        meta,
        mixed_qkv,
        conv_state_in,
        recurrent_state_in,
        b,
        a,
        conv1d_weight,
        A_log,
        dt_bias,
    ):
        _, indices, initial = self._metadata(meta)
        new_conv, conv_out = ops.causal_conv1d_update(
            conv_state_in, mixed_qkv, conv1d_weight, indices, initial
        )
        # The kernel reads q, k and v as heads of the convolution output, and
        # normalizes and scales q and k like _qkv.
        new_rec, out = ops.gated_delta_decode(
            recurrent_state_in,
            conv_out,
            b,
            a,
            A_log,
            dt_bias,
            indices,
            initial,
        )
        return out.astype(mixed_qkv.dtype), new_conv, new_rec

    def forward_extend(self, *args, seq_lens):
        if self.forward_metadata.verify_tokens:
            return self._forward_verify(*args)
        return self._forward_prefill(*args, seq_lens=seq_lens)

    @_per_device
    def _forward_verify(
        self,
        meta,
        mixed_qkv,
        conv_state_in,
        recurrent_state_in,
        b,
        a,
        conv1d_weight,
        A_log,
        dt_bias,
    ):
        """Run each request's draft tokens and keep the state after each one.

        The state after token t goes to slot ``states[:, t]``: the request's
        own slot for the last token, its verify slots (``verify_state_slots``)
        for the others. Once the accepted count is known, the speculative
        worker copies its state to the request's slot.
        """
        tokens = meta.verify_tokens
        indices, initial = meta.recurrent_indices, meta.has_initial_state
        batch = indices.shape[0]
        rows = recurrent_state_in.shape[0]
        if (rows - 1) % tokens:
            raise ValueError("The recurrent state pool has no verify slots for this draft length")
        states = verify_state_slots(indices, tokens, (rows - 1) // tokens)

        # The kernels carry each state through its tokens and write the state
        # after each one to its slot, a tile row of tokens per call.
        conv, state, outputs = conv_state_in, recurrent_state_in, []
        group = max(1, MAX_DECODE_USERS // tokens)
        for start in range(0, batch, group):
            requests = slice(start, start + group)
            steps = slice(start * tokens, (start + group) * tokens)
            conv, convolved = ops.causal_conv1d_update(
                conv,
                mixed_qkv[steps],
                conv1d_weight,
                indices[requests],
                initial[requests],
                state_slots=states[requests],
            )
            state, out = ops.gated_delta_decode(
                state,
                convolved,
                b[steps],
                a[steps],
                A_log,
                dt_bias,
                indices[requests],
                initial[requests],
                state_slots=states[requests],
            )
            outputs.append(out)
        out = jnp.concatenate(outputs)
        return out.astype(mixed_qkv.dtype), conv, state

    @_per_device
    def _forward_prefill(
        self,
        meta,
        mixed_qkv,
        conv_state_in,
        recurrent_state_in,
        b,
        a,
        conv1d_weight,
        A_log,
        dt_bias,
        seq_lens,
    ):
        del seq_lens
        meta, indices, initial = self._metadata(meta)
        count = mixed_qkv.shape[0]
        batch = meta.cu_q_lens.shape[0] - 1
        indices, initial = indices[:batch], initial[:batch]
        cu_q_lens = meta.cu_q_lens
        starts = cu_q_lens[:-1]
        lengths = jnp.diff(cu_q_lens)
        width = getattr(meta, "max_prefill_len", 0) or count
        width = count if batch == 1 else min(width, count)
        valid = jnp.arange(width) < lengths[..., None]

        def gather(pool):
            # Slot 0 is the pool's immutable zero state. Select it for a fresh
            # request instead of materializing a masked copy of the state.
            slots = jnp.where(initial, indices, 0)
            return pool.at[slots].get(mode="clip")

        # Pack ragged requests into independent sequences for the native kernel.
        def pack(value):
            if batch == 1:
                return value[None]
            positions = starts[:, None] + jnp.arange(width)
            return value.at[positions].get(mode="clip")

        packed = pack(mixed_qkv)
        saved = gather(conv_state_in).swapaxes(-1, -2)
        history = jnp.concatenate((saved, packed), axis=-2)
        convolved = None
        for tap in range(self.conv_kernel_size):
            window = history[..., tap : tap + width, :].astype(jnp.float32)
            weight = conv1d_weight[:, tap].astype(jnp.float32)
            term = window * weight
            convolved = term if convolved is None else convolved + term
        convolved = convolved.astype(mixed_qkv.dtype)
        convolved = jax.nn.silu(convolved)
        tail_indices = lengths[..., None] + jnp.arange(self.conv_kernel_size - 1)
        tail_indices += jnp.arange(batch)[:, None] * history.shape[-2]
        tail = history.reshape((-1, history.shape[-1])).at[tail_indices].get(mode="clip")
        new_conv = ops.state_pool_update(conv_state_in, indices, tail.swapaxes(-1, -2))

        q, k, v = self._qkv(convolved)
        beta = jax.nn.sigmoid(pack(b).astype(jnp.float32))
        gate = -jnp.exp(A_log.astype(jnp.float32)) * jax.nn.softplus(
            pack(a).astype(jnp.float32) + dt_bias.astype(jnp.float32)
        )
        # Zero beta and log-decay preserve state; zero queries mask padded output.
        q = jnp.where(valid[..., None, None], q, 0)
        beta, gate = (jnp.where(valid[..., None], x, 0) for x in (beta, gate))
        state, out = ops.gated_delta_rule(q, k, v, gate, beta, gather(recurrent_state_in))
        new_rec = ops.state_pool_update(recurrent_state_in, indices, state)
        out = out.reshape(-1, *out.shape[-2:])
        if batch > 1:
            token = jnp.arange(count)
            sequence = jnp.searchsorted(cu_q_lens[1:], token, side="right")
            sequence = jnp.minimum(sequence, batch - 1)
            row = sequence * width + token - starts[sequence]
            out = out.at[row].get(mode="clip")
            out = jnp.where((token < cu_q_lens[-1])[:, None, None], out, 0)
        return out.astype(mixed_qkv.dtype), new_conv, new_rec
