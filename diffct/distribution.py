"""Shared deterministic ownership for view and first-axis space partitions."""

import hashlib
import math
import numbers
from copy import deepcopy
from weakref import WeakKeyDictionary

import torch
import torch.distributed as dist


# Reuse groups while the same parent runtime is live. Never destroy a group
# here: older autograd graphs can still own and use it after later calls.
_HYBRID_GROUPS = WeakKeyDictionary()


def _hybrid_groups(parent, ranks, grid, view_rank, space_rank):
    view_size, space_size = grid
    if view_size == 1 or space_size == 1:
        return parent, parent
    cached_by_grid = _HYBRID_GROUPS.setdefault(parent, {})
    cached = cached_by_grid.get(grid)
    hit = False
    if cached is not None:
        try:
            hit = (dist.get_process_group_ranks(cached[0])
                   == ranks[view_rank * space_size:(view_rank + 1) * space_size]
                   and dist.get_process_group_ranks(cached[1]) == ranks[space_rank::space_size])
        except (KeyError, ValueError, RuntimeError):
            pass
    hits = [None] * len(ranks)
    dist.all_gather_object(hits, hit, group=parent)
    if all(hits):
        return cached
    backend = dist.get_backend(parent)
    space_group = view_group = None
    for view in range(view_size):
        members = ranks[view * space_size:(view + 1) * space_size]
        group = dist.new_group(members, backend=backend, use_local_synchronization=True)
        if view == view_rank:
            space_group = group
    for space in range(space_size):
        members = ranks[space::space_size]
        group = dist.new_group(members, backend=backend, use_local_synchronization=True)
        if space == space_rank:
            view_group = group
    cached_by_grid[grid] = space_group, view_group
    return space_group, view_group


def balanced_slice(size, rank, world_size):
    base, extra = divmod(size, world_size)
    start = rank * base + min(rank, extra)
    return slice(start, start + base + int(rank < extra))


def control_status(values, *, process_group=None, world_size=1):
    """Combine fixed-size control values without serializing Python objects."""
    if world_size == 1:
        return values
    with torch._C._DisableFuncTorch():
        device = (torch.device("cuda", torch.cuda.current_device())
                  if dist.get_backend(process_group) == "nccl" else torch.device("cpu"))
        status = torch.tensor(values, dtype=torch.int64, device=device)
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=process_group)
        return status.tolist()


def _canonical_metadata(value):
    """Keep container types while making dictionary order deterministic."""
    if isinstance(value, dict):
        return ("dict", tuple(sorted(
            ((_canonical_metadata(key), _canonical_metadata(item)) for key, item in value.items()),
            key=repr,
        )))
    if isinstance(value, (tuple, list)):
        return ("tuple" if isinstance(value, tuple) else "list",
                tuple(_canonical_metadata(item) for item in value))
    return value


def preflight(error=None, metadata=None, *, process_group=None, world_size=1,
              cache=None, cache_key=None, revision=None,
              mismatch="distributed ranks must agree on request metadata"):
    """Reject invalid/disagreeing requests collectively before numerical I/O.

    Cache only local digest computation. Every call compares current metadata
    across ranks, including calls through different warmed Partition objects.
    """
    result = (None if error is None else (type(error).__name__, str(error)), metadata)
    results = [result]
    if world_size > 1:
        # Control messages do not participate in data transforms or autograd.
        with torch._C._DisableFuncTorch():
            try:
                changed = (cache is None or cache_key not in cache
                           or cache[cache_key][:2] != (revision, metadata))
                digest = (int.from_bytes(hashlib.blake2b(
                    repr(_canonical_metadata(metadata)).encode("utf-8"), digest_size=8,
                ).digest(), "big", signed=True) if changed else cache[cache_key][2])
                exchange = error is not None
            except Exception:
                # Metadata that cannot be digested here (e.g. self-referential) must not leave
                # this rank outside the collective; the object exchange below reports it.
                changed, digest, exchange = False, 0, True
            # MAX(digest) and MAX(~digest) encode both signed int64 bounds.
            # Complementing also handles -2**63 without negation overflow.
            failed, upper, inverse_lower = control_status(
                (int(exchange), digest, ~digest),
                process_group=process_group, world_size=world_size,
            )
            if failed or upper != ~inverse_lower:
                results = [None] * world_size
                dist.all_gather_object(results, result, group=process_group)
    failure = next((entry[0] for entry in results if entry[0] is not None), None)
    if failure is not None:
        classes = {kind.__name__: kind for kind in (TypeError, ValueError, RuntimeError,
                                                   PermissionError, FileNotFoundError)}
        raise classes.get(failure[0], RuntimeError)(failure[1])
    if any(entry[1] != metadata for entry in results):
        raise ValueError(mismatch)
    if cache is not None and world_size > 1 and changed:
        cache[cache_key] = (revision, deepcopy(metadata), digest)


class Partition:
    """Map parent-group ranks as ``view_rank * space_world_size + space_rank``.

    ``partition_grid`` is ``(view_ranks, space_ranks)``. Metadata agreement
    precedes subgroup creation; every parent rank creates groups in one order.
    """

    def __init__(self, volume_shape, n_views, *, distributed=False, process_group=None,
                 partition="views", partition_grid=None):
        if not isinstance(distributed, bool):
            raise TypeError("distributed must be a bool")
        if process_group is not None and not distributed:
            raise ValueError("process_group requires distributed=True")
        self.process_group = process_group
        self._preflight_cache = {}
        self.rank, self.world_size = 0, 1
        if distributed:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("distributed=True requires an initialized process group")
            self.rank, self.world_size = dist.get_rank(process_group), dist.get_world_size(process_group)
            if self.rank < 0 or self.world_size <= 0:
                raise ValueError("process_group must include the calling rank")
        metadata = (tuple(volume_shape), n_views, partition, partition_grid)
        if self.world_size > 1:
            configurations = [None] * self.world_size
            dist.all_gather_object(configurations, metadata, group=process_group)
            if any(configuration != metadata for configuration in configurations):
                raise ValueError("distributed ranks must agree on shape, views, partition and partition_grid")
        if partition not in ("views", "space", "hybrid"):
            raise ValueError("partition must be 'views', 'space', or 'hybrid'")
        if partition != "hybrid" and partition_grid is not None:
            raise ValueError("partition_grid requires partition='hybrid'")
        if partition == "hybrid":
            if partition_grid is None:
                space = math.isqrt(self.world_size)
                while self.world_size % space:
                    space -= 1
                partition_grid = (self.world_size // space, space)
            if (not isinstance(partition_grid, (tuple, list)) or len(partition_grid) != 2
                    or any(isinstance(value, bool) or not isinstance(value, numbers.Integral)
                           or value <= 0 for value in partition_grid)
                    or math.prod(partition_grid) != self.world_size):
                raise ValueError("partition_grid must contain positive (view_ranks, space_ranks) with product world_size")
        else:
            partition_grid = (1, self.world_size) if partition == "space" else (self.world_size, 1)
        self.partition_grid = tuple(partition_grid)
        self.view_world_size, self.space_world_size = self.partition_grid
        self.view_rank, self.space_rank = divmod(self.rank, self.space_world_size)
        self.space_group = process_group
        self.view_group = process_group
        self.ranks = (dist.get_process_group_ranks(process_group or dist.group.WORLD)
                      if distributed else [0])
        if partition == "hybrid" and self.world_size > 1:
            parent = dist.group.WORLD if process_group is None else process_group
            self.space_group, self.view_group = _hybrid_groups(
                parent, self.ranks, self.partition_grid, self.view_rank, self.space_rank,
            )
        first = balanced_slice(volume_shape[0], self.space_rank, self.space_world_size)
        self.volume_slice = (first, *(slice(0, size) for size in volume_shape[1:]))
        self.local_volume_shape = (first.stop - first.start, *volume_shape[1:])
        self.view_slice = balanced_slice(n_views, self.view_rank, self.view_world_size)

    def global_rank(self, rank):
        return self.ranks[rank]

    def preflight(self, error=None, metadata=None, *, cache_key=None, revision=None,
                  mismatch="distributed ranks must agree on request metadata"):
        """Reject invalid/disagreeing requests on every parent rank before I/O."""
        return preflight(error, metadata, process_group=self.process_group, world_size=self.world_size,
                         cache=self._preflight_cache if cache_key is not None else None,
                         cache_key=cache_key, revision=revision, mismatch=mismatch)
