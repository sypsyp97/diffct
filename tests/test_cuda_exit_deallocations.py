"""CPU regression checks for CUDA deallocation queues during shutdown."""

import logging

import pytest


@pytest.mark.parametrize("capacity", [None, 80 * 1024**3], ids=["unset", "80-GiB"])
@pytest.mark.parametrize("with_size", [False, True], ids=["unsized", "sized"])
def test_exit_hook_keeps_cuda_deallocations_pending(monkeypatch, capacity, with_size):
    try:
        from numba.cuda.cudadrv import driver
        from numba.cuda.cudadrv.driver import _PendingDeallocs
    except ImportError as error:
        pytest.skip(f"CUDA deallocation queue cannot be imported: {error}")

    from numba.core import config
    from diffct.utils import _keep_cuda_modules_at_exit

    # Queue logging normally starts when a CUDA driver is initialized.
    monkeypatch.setattr(driver, "_logger", logging.getLogger(__name__), raising=False)
    # Register restoration before the hook changes shared configuration.
    for configuration in (config, driver.config):
        monkeypatch.setattr(configuration, "CUDA_DEALLOCS_COUNT", 10)
        monkeypatch.setattr(configuration, "CUDA_DEALLOCS_RATIO", 0.2)

    queue = _PendingDeallocs() if capacity is None else _PendingDeallocs(capacity)
    destroyed = []

    def destructor(handle):
        destroyed.append(handle)

    _keep_cuda_modules_at_exit()

    # Sizes are bookkeeping values; this test does not allocate CUDA memory.
    size = (80 * 1024**3) // 32
    for handle in range(64):
        if with_size:
            queue.add_item(destructor, handle, size)
        else:
            queue.add_item(destructor, handle)
        assert destroyed == []
        assert len(queue) == handle + 1
