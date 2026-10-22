"""Spawn test workers without running interpreter finalizers after success."""

import os
import sys

import torch.multiprocessing as mp


def _run_worker(rank, function, args):
    # The worker must finish its assertions, output writes and explicit teardown.
    # Keep failures in PyTorch's exception-reporting path; never exit in finally.
    try:
        function(rank, *args)
    except KeyboardInterrupt as error:
        # PyTorch otherwise treats a worker's KeyboardInterrupt as success.
        raise RuntimeError("spawned test worker interrupted") from error
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid late Gloo tensor cleanup acquiring the GIL during finalization.
    os._exit(0)


def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method="spawn"):
    """Preserve torch.multiprocessing.spawn joins and worker error reporting."""
    return mp.spawn(_run_worker, args=(fn, args), nprocs=nprocs, join=join,
                    daemon=daemon, start_method=start_method)
