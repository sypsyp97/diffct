"""Select the CPU CUDA simulator before any diffct or Numba import."""

import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "1"

import pytest


@pytest.fixture(scope="session")
def simulated_kernels():
    """Require a working simulator in CI; permit missing local dependencies."""
    required = any(
        os.environ.get(name, "").lower() in {"1", "true", "yes"}
        for name in ("FORMAL_REQUIRE_CUDASIM", "CI")
    )
    try:
        import numpy as np
        from numba import cuda
        from numba.core import config
    except (ImportError, OSError) as exc:
        message = f"CUDA simulator dependencies unavailable: {exc}"
        if required:
            pytest.fail(message, pytrace=False)
        pytest.skip(message)

    if not config.ENABLE_CUDASIM:
        pytest.fail("Numba was imported before NUMBA_ENABLE_CUDASIM=1", pytrace=False)

    @cuda.jit
    def probe(array):
        array[0] += 1

    try:
        value = np.zeros(1, dtype=np.float32)
        probe[1, 1](value)
    except (ImportError, OSError, RuntimeError) as exc:
        message = f"CUDA simulator cannot launch a CPU kernel: {exc}"
        if required:
            pytest.fail(message, pytrace=False)
        pytest.skip(message)
    assert value[0] == 1

    # Package import failures are errors, not simulator-availability skips.
    from diffct import kernels

    return kernels
