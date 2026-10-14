"""Physical forward-difference TV with differentiable slab boundary faces."""

import math
import numbers

import torch
import torch.distributed as dist

from ._solver_state import finite, scalar_sum
from .backends import _zero_dependency


def _peers(operator, nonempty):
    spaces = operator.space_world_size
    current = operator.space_rank
    occupied = [rank for rank in range(spaces) if rank < min(spaces, operator.global_volume_shape[0])]
    previous = max((rank for rank in occupied if rank < current), default=None) if nonempty else None
    following = min((rank for rank in occupied if rank > current), default=None) if nonempty else None
    def global_rank(rank):
        if rank is None:
            return None
        parent = operator.view_rank * spaces + rank
        return operator._partition.global_rank(parent)
    return global_rank(previous), global_rank(following)


class _FollowingFace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, volume, operator):
        previous, following = _peers(operator, volume.shape[0] > 0)
        ctx.operator, ctx.previous, ctx.following = operator, previous, following
        ctx.shape, ctx.dtype, ctx.device = volume.shape, volume.dtype, volume.device
        nccl = dist.get_backend(operator.space_group) == "nccl"
        ctx.communication_device = (torch.device("cuda", torch.cuda.current_device())
                                    if nccl else torch.device("cpu"))
        if nccl:
            # Initialize the subgroup communicator collectively, including
            # empty slabs, before the first subset-only face exchange.
            dist.barrier(group=operator.space_group)
        face = torch.zeros(volume.shape[1:], dtype=volume.dtype, device=ctx.communication_device)
        operations = []
        if previous is not None:
            first = volume[0].detach().to(device=ctx.communication_device).contiguous()
            operations.append(dist.P2POp(dist.isend, first, previous, operator.space_group))
        if following is not None:
            operations.append(dist.P2POp(dist.irecv, face, following, operator.space_group))
        if operations:
            for work in dist.batch_isend_irecv(operations):
                work.wait()
        return face.to(device=volume.device)

    @staticmethod
    def backward(ctx, grad_face):
        received = torch.zeros(ctx.shape[1:], dtype=ctx.dtype, device=ctx.communication_device)
        operations = []
        if ctx.following is not None:
            outgoing = grad_face.to(device=ctx.communication_device).contiguous()
            operations.append(dist.P2POp(dist.isend, outgoing, ctx.following, ctx.operator.space_group))
        if ctx.previous is not None:
            operations.append(dist.P2POp(dist.irecv, received, ctx.previous, ctx.operator.space_group))
        if operations:
            for work in dist.batch_isend_irecv(operations):
                work.wait()
        gradient = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        if ctx.shape[0] > 0:
            gradient[0] = received.to(device=ctx.device)
        return gradient, None


def total_variation(volume, *, operator=None, voxel_spacing=1., eps=1e-6, reduction="mean"):
    """Isotropic TV over every voxel, with physical forward differences.

    Differences vanish at the global outer boundary. Spatially partitioned
    volumes exchange a single neighboring face in forward and backward; the
    reported scalar counts replicated ownership once and gradients stay local.
    """
    if not isinstance(volume, torch.Tensor) or not torch.is_floating_point(volume) or volume.ndim < 1:
        raise TypeError("volume must be a real floating-point tensor")
    eps = finite(eps, "eps", positive=True)
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be mean or sum")
    spacing = (voxel_spacing,) * volume.ndim if isinstance(voxel_spacing, numbers.Real) else tuple(voxel_spacing)
    if len(spacing) != volume.ndim:
        raise ValueError("voxel_spacing must match the volume axes")
    spacing = tuple(finite(pitch, "voxel_spacing", positive=True) for pitch in spacing)
    if operator is not None and tuple(volume.shape) != tuple(operator.volume_shape):
        raise ValueError("volume shape must match its operator ownership")
    distributed_space = operator is not None and getattr(operator, "world_size", 1) > 1 and operator.partition in ("space", "hybrid")
    halo = _FollowingFace.apply(volume, operator) if distributed_space else None
    squared = torch.zeros_like(volume)
    for axis, pitch in enumerate(spacing):
        current, following = [slice(None)] * volume.ndim, [slice(None)] * volume.ndim
        current[axis], following[axis] = slice(0, -1), slice(1, None)
        difference = torch.zeros_like(volume)
        difference[tuple(current)] = (volume[tuple(following)] - volume[tuple(current)]) / pitch
        if axis == 0 and halo is not None and volume.shape[0] and operator.volume_slice[0].stop < operator.global_volume_shape[0]:
            difference[-1] = (halo - volume[-1]) / pitch
        squared = squared + difference.square()
    local = torch.sqrt(squared + eps).sum()
    if halo is not None:
        # The final slab also needs backward to receive the previous slab's
        # contribution, even though its own global outer difference is zero.
        local = local + _zero_dependency(halo)
    total = local
    elements = volume.numel()
    if operator is not None:
        global_value = scalar_sum(operator, local.detach().item(), "volume")
        total = local + (local.new_tensor(global_value) - local.detach())
        elements = math.prod(operator.global_volume_shape)
    return total if reduction == "sum" else total / max(elements, 1)
