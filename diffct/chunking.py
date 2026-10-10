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
    available = max(0, min(free + reusable, allocator_room))
    budget = available * 3 // 4
    allocation_limit = (1 << 63) - 1  # Unbounded limit in scalar int64 agreement.
    backend = getattr(torch.cuda, "get_allocator_backend", lambda: "native")()
    # Native requests >1 MiB and <10 MiB reserve a 20 MiB segment. A smaller
    # current available room must use the <=1 MiB small pool, whose segments are 2 MiB.
    # Source: pytorch/c10/cuda/CUDACachingAllocator.cpp (Native allocator notes).
    if backend == "native" and available < 20 * 1024 ** 2:
        allocation_limit = 1024 ** 2
    return budget, allocation_limit


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


def _pipeline_bytes(beam, shape, views, pixels, points, *, slots=1, retained=1,
                    geometry_grad=False, peers=False, ray_accumulators=0):
    """Live typed allocations, including native allocator rounding per buffer."""
    rank, rays = len(shape), views * math.prod(pixels)
    native = _allocation_bytes(math.prod(shape))
    # Two producer epochs permit the preceding native/layout/output storage
    # to stay in flight while the next retained tile window queues work.
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
