"""Checkpoint reads must preserve values and avoid full reads for partial shards."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from safetensors.numpy import save_file

from sgl_jax.srt.utils.weight_utils import SequentialSafetensorManager, WeightLoader


@pytest.mark.parametrize("dtype,st_dtype", [(np.float32, "F32"), (jnp.bfloat16, "BF16")])
@pytest.mark.parametrize("spec", [P(), P("tensor", None), P(None, "tensor")])
def test_checkpoint_read_preserves_sharded_weights(tmp_path, dtype, st_dtype, spec):
    devices = jax.local_devices()
    if len(devices) < 4:
        pytest.skip("Requires four devices; CPU CI sets JAX_NUM_CPU_DEVICES=4")
    mesh = Mesh(np.asarray(devices[:4]), ("tensor",))
    sharding = NamedSharding(mesh, spec)
    expected = (np.arange(128, dtype=np.float32).reshape(8, 16) - 64).astype(dtype)
    filename = str(tmp_path / "model.safetensors")
    save_file({"weight": expected}, filename)
    loader = object.__new__(WeightLoader)
    loader.mesh = mesh

    with SequentialSafetensorManager() as file_manager:
        handle = mock.Mock(wraps=file_manager.get_handle(filename))
        with mock.patch.object(file_manager, "get_handle", return_value=handle):
            (actual,) = loader._create_lazy_tensors(
                "weight",
                [{"file": filename, "shape": expected.shape, "dtype": st_dtype}],
                file_manager,
                target_sharding=sharding,
            )
            np.testing.assert_array_equal(
                np.asarray(actual).view(np.uint8), expected.view(np.uint8)
            )

    assert actual.dtype == expected.dtype
    assert actual.sharding == sharding
    if spec == P():
        assert handle.get_tensor.called
        handle.get_slice.assert_not_called()
    else:
        assert handle.get_slice.called
        handle.get_tensor.assert_not_called()
