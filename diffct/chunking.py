"""Spatial slices and immutable execution limits for streamed projectors."""

from dataclasses import dataclass
from itertools import product
import math

import torch


@dataclass(frozen=True)
class _ChunkPlan:
    chunk_shape: tuple
    view_chunk_size: int
    devices: tuple
    detector_chunk_shape: tuple = None
    gpu_budget_bytes: int = 0
    allocation_limit: int = (1 << 63) - 1
    schedule: str = "spatial"
    window_size: int = 1
    slots: int = 1
    estimated_gpu_bytes: int = 0
    fallback_reason: str = ""
    foreign_output: bool = False
    reservation_budget_bytes: int = (1 << 63) - 1
    estimated_reserved_bytes: int = 0


def _pixel_tiles(shape, chunk_shape):
    for starts in product(*(range(0, size, part) for size, part in zip(shape, chunk_shape))):
        clipped = tuple(min(part, size - start) for size, start, part in zip(shape, starts, chunk_shape))
        yield tuple(slice(start, start + size) for start, size in zip(starts, clipped)), clipped


def _memory_budget(device):
    """Usable allocator bytes, retaining 25% headroom for allocation overhead.

    Reusable reserved blocks count as available, but never bypass the process
    fraction ceiling. Older PyTorch without its public fraction getter can
    only observe driver/allocator memory; explicit limits remain available.
    """
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reusable = max(0, torch.cuda.memory_reserved(device) - allocated)
    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    fraction = getter(device) if getter is not None else 1.0
    allocator_room = max(0, int(total * fraction) - allocated)
    backend = getattr(torch.cuda, "get_allocator_backend", lambda: "native")()
    foreign_splits = 0
    if backend == "native":
        caller = torch.cuda.current_stream(device).cuda_stream
        for segment in _native_segments(device):
            blocks = segment["blocks"]
            if segment["stream"] != caller and any(block["state"] != "inactive" for block in blocks):
                foreign_splits += sum(block["size"] for block in blocks if block["state"] == "inactive")
    available = max(0, min(free + reusable, allocator_room) - foreign_splits)
    budget = available * 3 // 4
    allocation_limit = (1 << 63) - 1  # Unbounded limit in scalar int64 agreement.
    # Native requests >1 MiB and <10 MiB reserve a 20 MiB segment. A smaller
    # current available room must use the <=1 MiB small pool, whose segments are 2 MiB.
    # Source: pytorch/c10/cuda/CUDACachingAllocator.cpp (Native allocator notes).
    if backend == "native" and available < 20 * 1024 ** 2:
        allocation_limit = 1024 ** 2
    return budget, allocation_limit


def _memory_lower_bounds(device):
    """Lower bounds of `_memory_budget` and `_reservation_budget` without an allocator snapshot.

    Returns (budget, allocation_limit, reservation budget), each at most the value the two exact
    functions return at the same allocator state. The snapshot is what makes those functions
    slow; it only serves two corrections, and both are bounded by allocator counters: free splits
    pinned by other streams are inactive reserved bytes, at most reserved - allocated, and pinned
    segments hold at most every reserved byte.
    """
    free, total = torch.cuda.mem_get_info(device)
    # One nested statistics query; memory_allocated/memory_reserved each flatten every statistic.
    stats = torch.cuda.memory_stats_as_nested_dict(device)
    if stats:
        allocated = stats["allocated_bytes"]["all"]["current"]
        reserved = stats["reserved_bytes"]["all"]["current"]
    else:                                   # allocator not initialised: nothing allocated or reserved
        allocated = reserved = 0
    reusable = max(0, reserved - allocated)
    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    fraction = getter(device) if getter is not None else 1.0
    allocator_room = max(0, int(total * fraction) - allocated)
    backend = getattr(torch.cuda, "get_allocator_backend", lambda: "native")()
    foreign_splits = reusable if backend == "native" else 0
    available = max(0, min(free + reusable, allocator_room) - foreign_splits)
    allocation_limit = (1 << 63) - 1
    if backend == "native" and available < 20 * 1024 ** 2:
        allocation_limit = 1024 ** 2
    if torch.cuda.get_allocator_backend() != "native":
        reservation = (1 << 63) - 1
    else:
        reservation = max(0, min(free, int(total * fraction) - reserved))
    return available * 3 // 4, allocation_limit, reservation


def _native_segments(device):
    index = torch.device(device).index if not isinstance(device, int) else device
    index = torch.cuda.current_device() if index is None else index
    return [segment for segment in torch.cuda.memory_snapshot() if segment["device"] == index]


def _reservation_budget(device):
    """Room for fresh segments, conservatively excluding every pinned segment.

    Free splits in a live segment may serve the caller, but cannot guarantee
    placement of the complete new inventory. Wholly inactive segments can be
    released by the native allocator, including those owned by other streams.
    """
    if torch.cuda.get_allocator_backend() != "native":
        return (1 << 63) - 1
    free, total = torch.cuda.mem_get_info(device)
    segments = _native_segments(device)
    pinned = sum(segment["total_size"] for segment in segments
                 if any(block["state"] != "inactive" for block in segment["blocks"]))
    reclaimable = sum(segment["total_size"] for segment in segments
                      if all(block["state"] == "inactive" for block in segment["blocks"]))
    getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    fraction = getter(device) if getter is not None else 1.0
    return max(0, min(free + reclaimable, int(total * fraction) - pinned))


def _working_set_bytes(beam, shape, views, detector_shape, surface):
    """Conservative native staging/layout, cotangent, geometry and VJP scratch."""
    rank = len(shape)
    voxels = math.prod(shape)
    rays = views * math.prod(detector_shape)
    geometry = (rank * (views + rays) if surface
                else rank * (4 if beam == "cone" else 3) * views)
    # A resident native tile, staging and completed cone layout can coexist.
    # Two ray buffers cover reusable output/cotangent and transfer scratch; two
    # geometry buffers cover staged values and their first-order VJP/reduction.
    tile_buffers = 3 if beam == "cone" else 2
    return 4 * (tile_buffers * voxels + 2 * rays + 2 * geometry)


def _largest_buffer_bytes(beam, shape, views, detector_shape, surface):
    rank = len(shape)
    rays = views * math.prod(detector_shape)
    geometry = (rank * (views + rays) if surface
                else rank * (4 if beam == "cone" else 3) * views)
    return 4 * max(math.prod(shape), rays, geometry)


def _allocation_bytes(elements):
    size = 4 * elements
    if torch.cuda.get_allocator_backend() == "native":
        return (size + 511) // 512 * 512
    return size


def _reservation_bytes(buffer_sizes):
    """Fresh, monotone fixed-buffer inventory bound for standard native pools.

    Arbitrary free holes and allocator overrides are outside this model. Large
    and middle requests share one pool, so mixed inventories use separate fresh
    segments per request instead of assuming their placement is independent.
    """
    sizes = [size for size in buffer_sizes if size > 0]
    if torch.cuda.get_allocator_backend() != "native":
        return sum(sizes)
    mib = 1024 ** 2
    sizes = [(size + 511) // 512 * 512 for size in sizes]
    large = [size for size in sizes if size >= 10 * mib]
    reservation = sum((size + 2 * mib - 1) // (2 * mib) * (2 * mib) for size in large)
    middle = [size for size in sizes if mib < size < 10 * mib]
    if large:
        reservation += len(middle) * 20 * mib
        middle = []
    for segment_size, requests in (
        (2 * mib, [size for size in sizes if size <= mib]),
        (20 * mib, middle),
    ):
        if not requests:
            continue
        total = sum(requests)
        # At a new segment's birth every older segment holds more than
        # segment_size - max(requests); the new one holds at least min(requests).
        # This bounds online best-fit placement in any allocation order.
        count = (1 if total <= segment_size else
                 (total - min(requests)) // (segment_size - max(requests)) + 1)
        reservation += count * segment_size
    return reservation


def _pipeline_buffers(beam, shape, views, pixels, points, *, slots=1, retained=1,
                      geometry_grad=False, peers=False, ray_accumulators=0):
    """Live typed allocation inventory, with native rounding per buffer."""
    rank, rays = len(shape), views * math.prod(pixels)
    native = _allocation_bytes(math.prod(shape))
    # Two producer epochs permit the preceding native/layout/output storage
    # to stay in flight while the next retained tile window queues work.
    tiles = slots * (retained + int(beam == "cone")) + int(peers)
    sizes = ((views * rank, rays * rank) if points
             else (views * rank,) * (4 if beam == "cone" else 3))
    geometry = tuple(_allocation_bytes(size) for size in sizes)
    return ((native,) * tiles + (_allocation_bytes(rays),) * (slots + ray_accumulators)
            + geometry * (slots * (1 + int(geometry_grad))))


def _pipeline_bytes(beam, shape, views, pixels, points, *, slots=1, retained=1,
                    geometry_grad=False, peers=False, ray_accumulators=0):
    """Live typed bytes; segment reservation is admitted separately."""
    rank, rays = len(shape), views * math.prod(pixels)
    native = _allocation_bytes(math.prod(shape))
    tile = native * (slots * (retained + int(beam == "cone")) + int(peers))
    sizes = ((views * rank, rays * rank) if points
             else (views * rank,) * (4 if beam == "cone" else 3))
    geometry = sum(_allocation_bytes(size) for size in sizes)
    return (tile + slots * (_allocation_bytes(rays) + geometry)
            + ray_accumulators * _allocation_bytes(rays) + (slots * geometry if geometry_grad else 0))


def _host_geometry_bytes(beam, views, pixels, points):
    rank = 3 if beam == "cone" else 2
    if not points:
        return 8 * 2 * rank * views * (4 if beam == "cone" else 3)
    # Grids/offsets, world expressions and first-order cotangent/graph work.
    # Arbitrary sampler internals and explicit parameter snapshots are separate.
    return 8 * rank * (math.prod(pixels) * (1 + 6 * views) + 4 * views)


def _tiles(shape, chunk_shape, spacing):
    """Yield tensor slices, clipped shapes and physical centres in xyz order."""
    starts = (range(0, size, chunk) for size, chunk in zip(shape, chunk_shape))
    for start in product(*starts):
        tile_shape = tuple(min(chunk, size - first)
                           for size, chunk, first in zip(shape, chunk_shape, start))
        slices = tuple(slice(first, first + size) for first, size in zip(start, tile_shape))
        centre = tuple((first + size / 2 - full / 2) * spacing
                       for first, size, full in zip(start, tile_shape, shape))
        yield slices, tile_shape, centre[::-1]
