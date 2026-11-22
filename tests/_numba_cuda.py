"""Import Numba's CUDA target for tests that need real kernel dispatchers."""

import torch


def import_numba_cuda():
    """Return None only if libcudart is missing on a CPU host."""
    try:
        from numba import cuda
    except RuntimeError as error:
        if torch.cuda.is_available() or "libcudart" not in str(error).partition("\n")[0]:
            raise
        from cuda.pathfinder import DynamicLibNotFoundError

        if type(error) is not DynamicLibNotFoundError:
            raise
        return None
    return cuda
