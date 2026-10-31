"""Device memory for the KV cache on Tenstorrent devices."""

import gc
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Tokens of the throwaway KV cache the weights are placed next to.
_WARMUP_TOKENS = 1024


def place_weights(model_runner, max_num_reqs, total_device_memory, dp_size):
    """Run one decode step so the device holds the model's weights.

    libtt keeps parameters in host memory and makes their device copies (BFP8
    matmul weights, tiled embedding tables) when a program first uses them.
    ModelRunner sizes the KV cache from the device memory free before that, so
    the copies would not fit next to it. One decode step over a small KV cache
    makes them first, and later programs reuse them.
    """
    from sgl_jax.srt.layers.logits_processor import LogitsMetadata
    from sgl_jax.srt.model_executor.compilation_manager import CompilationManager
    from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

    runner = model_runner
    runner.init_memory_pool(max_num_reqs, _WARMUP_TOKENS, total_device_memory, dp_size=dp_size)
    shapes = CompilationManager(
        server_args=runner.server_args,
        max_padded_batch_size=dp_size,
        max_padded_num_tokens=dp_size,
        dp_size=dp_size,
        tp_size=runner.tp_size,
        page_size=runner.page_size,
        max_req_len=runner.max_total_num_tokens // dp_size - 1,
        vocab_size=runner.model_config.vocab_size,
        max_total_num_tokens=runner.max_total_num_tokens,
        has_recurrent_state=runner.linear_recurrent_config is not None,
        attn_backend=runner.attn_backend,
    )
    bs, tokens, cache_loc, pages = next(shapes.iter_model_shapes(ForwardMode.DECODE))
    batch = shapes._make_dummy_batch(
        bs,
        tokens,
        ForwardMode.DECODE,
        cache_loc,
        dp_size=dp_size,
        per_dp_bs_size=bs // dp_size,
        dummy_seq_len=shapes.decode_dummy_seq_len(pages),
    )
    forward_batch = ForwardBatch.init_new(batch, runner)
    runner.attn_backend.forward_metadata = runner.attn_backend.get_forward_metadata(batch)
    output, _, _ = runner.forward(
        forward_batch, LogitsMetadata.from_model_worker_batch(batch, runner.mesh)
    )
    # Reading the logits waits for the step; block_until_ready only waits for dispatch.
    np.asarray(output.next_token_logits)

    runner.req_to_token_pool = None
    runner.token_to_kv_pool = None
    runner.token_to_kv_pool_allocator = None
    runner.memory_pools = None
    del forward_batch, output
    gc.collect()
    logger.info("Placed the weights on the device before sizing the KV cache")
