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
    # Capture all shared state before the hook can replace or add attributes.
    class_attributes = dict(vars(_PendingDeallocs))
    configurations = {id(configuration): configuration for configuration in (config, driver.config)}
    configuration_attributes = {
        identifier: {
            name: value
            for name, value in vars(configuration).items()
            if name.startswith("CUDA_DEALLOCS_")
        }
        for identifier, configuration in configurations.items()
    }

    queue = _PendingDeallocs() if capacity is None else _PendingDeallocs(capacity)
    destroyed = []

    def destructor(handle):
        destroyed.append(handle)

    try:
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
    finally:
        for name in set(vars(_PendingDeallocs)) - class_attributes.keys():
            delattr(_PendingDeallocs, name)
        for name, value in class_attributes.items():
            if name not in vars(_PendingDeallocs) or vars(_PendingDeallocs)[name] is not value:
                setattr(_PendingDeallocs, name, value)
        for identifier, configuration in configurations.items():
            saved = configuration_attributes[identifier]
            for name in tuple(vars(configuration)):
                if name.startswith("CUDA_DEALLOCS_") and name not in saved:
                    delattr(configuration, name)
            for name, value in saved.items():
                setattr(configuration, name, value)

    assert dict(vars(_PendingDeallocs)) == class_attributes
    for identifier, configuration in configurations.items():
        assert {
            name: value
            for name, value in vars(configuration).items()
            if name.startswith("CUDA_DEALLOCS_")
        } == configuration_attributes[identifier]

    # A fresh queue must regain the stock count-triggered flush after restoration.
    stock_queue = _PendingDeallocs()
    stock_destroyed = []
    count_limit = driver.config.CUDA_DEALLOCS_COUNT
    for handle in range(count_limit):
        stock_queue.add_item(stock_destroyed.append, handle)
        assert stock_destroyed == []
    stock_queue.add_item(stock_destroyed.append, count_limit)
    assert stock_destroyed == list(range(count_limit + 1))
    assert len(stock_queue) == 0
