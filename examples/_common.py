"""Shared helpers for the reconstruction examples: phantom, scan, FDK, CGLS."""

import math
import os

import numpy as np
import torch
import torch.distributed as dist

from diffct import (
    angular_integration_weights,
    circular_trajectory_3d,
    cone_cosine_weights,
    cone_weighted_backproject,
    ramp_filter_1d,
    saddle_trajectory_3d,
    sinusoidal_trajectory_3d,
    spiral_trajectory_3d,
)

TRAJECTORIES = ("circular", "helical", "saddle", "sinusoidal")


def shepp_logan_3d(n):
    """3D Shepp-Logan phantom on an n^3 grid, sampled at voxel centres, values in [0, 1]."""
    ellipsoids = np.array([
        # x, y, z, a, b, c, phi, value
        [0, 0, 0, 0.69, 0.92, 0.81, 0, 1.0],
        [0, -0.0184, 0, 0.6624, 0.874, 0.78, 0, -0.8],
        [0.22, 0, 0, 0.11, 0.31, 0.22, -math.pi / 10, -0.2],
        [-0.22, 0, 0, 0.16, 0.41, 0.28, math.pi / 10, -0.2],
        [0, 0.35, -0.15, 0.21, 0.25, 0.41, 0, 0.1],
        [0, 0.10, 0.25, 0.046, 0.046, 0.05, 0, 0.1],
        [0, -0.10, 0.25, 0.046, 0.046, 0.05, 0, 0.1],
        [-0.08, -0.605, 0, 0.046, 0.023, 0.05, 0, 0.1],
        [0, -0.605, 0, 0.023, 0.023, 0.02, 0, 0.1],
        [0.06, -0.605, 0, 0.023, 0.046, 0.02, 0, 0.1],
    ], dtype=np.float32)
    axis = (np.arange(n, dtype=np.float32) + 0.5 - n / 2) / (n / 2)
    zz, yy, xx = np.meshgrid(axis, axis, axis, indexing="ij")
    phantom = np.zeros((n, n, n), dtype=np.float32)
    for x0, y0, z0, a, b, c, phi, value in ellipsoids:
        xc, yc, zc = xx - x0, yy - y0, zz - z0
        xp = math.cos(phi) * xc - math.sin(phi) * yc
        yp = math.sin(phi) * xc + math.cos(phi) * yc
        phantom += value * ((xp / a) ** 2 + (yp / b) ** 2 + (zc / c) ** 2 <= 1.0)
    return np.clip(phantom, 0.0, 1.0)


class Scan:
    """Cone-beam scan for an n^3 volume with unit voxels.

    The source is 2.5 n from the isocentre and the detector 1.5 n beyond it
    (magnification 1.6). A detector pitch of 0.8 samples the isocentre every
    0.5 voxel, fine enough that FDK resolution is set by the volume grid.
    """

    def __init__(self, n, views=360):
        self.n = n
        self.views = views
        self.sid = 2.5 * n
        self.sdd = 4.0 * n
        self.pitch = 0.8
        self.detector = (3 * n, 2 * n)

    def trajectory(self, name):
        n, views, sid, sdd = self.n, self.views, self.sid, self.sdd
        if name == "circular":
            return circular_trajectory_3d(views, sid, sdd, device="cpu")
        if name == "helical":
            return spiral_trajectory_3d(views, sid, sdd, z_range=0.3 * n, n_turns=1.0, device="cpu")
        if name == "saddle":
            return saddle_trajectory_3d(views, sid, sdd, z_amplitude=0.15 * n,
                                        radial_amplitude=0.1 * sid, device="cpu")
        if name == "sinusoidal":
            return sinusoidal_trajectory_3d(views, sid, sdd, amplitude=0.15 * n, frequency=2.0, device="cpu")
        raise ValueError(f"unknown trajectory {name!r}; choose from {TRAJECTORIES}")


def fdk(scan, sinogram, trajectory, window="shepp-logan"):
    """FDK reconstruction of a full ``(views, U, V)`` sinogram on its device.

    FDK is designed for the circular orbit and is approximate away from its
    central plane; for other trajectories the approximation is coarser. ``window`` trades noise against sharpness: "ram-lak" keeps
    all frequencies, "hann" is the smoothest.
    """
    device = sinogram.device
    u, v = scan.detector
    weights = cone_cosine_weights(u, v, scan.pitch, scan.pitch, scan.sdd, device=device).unsqueeze(0)
    filtered = ramp_filter_1d(sinogram * weights, dim=1, sample_spacing=scan.pitch,
                              pad_factor=2, window=window).contiguous()
    angles = torch.linspace(0.0, 2 * math.pi, scan.views + 1, device=device)[:-1]
    filtered = filtered * angular_integration_weights(angles, redundant_full_scan=True).view(-1, 1, 1)
    n = scan.n
    return cone_weighted_backproject(filtered, *[g.to(device) for g in trajectory], n, n, n,
                                     scan.pitch, scan.pitch, voxel_spacing=1.0).clamp_min(0)


def cgls(operator, measurements, iterations):
    """Conjugate gradient least squares for ``min ||A x - y||``.

    In distributed mode ``measurements`` is the rank's view shard and the
    volume is replicated: sinogram inner products are summed over ranks,
    volume inner products are not.
    """
    def shard_dot(a, b):
        value = torch.sum(a.double() * b.double())
        if operator.world_size > 1:
            dist.all_reduce(value)
        return value

    x = torch.zeros(operator.volume_shape, device=measurements.device)
    r = measurements.clone()
    s = operator.backproject(r)
    p = s.clone()
    gamma = torch.sum(s.double() * s.double())
    for _ in range(iterations):
        q = operator.project(p)
        qq = shard_dot(q, q)
        # gamma is replicated and qq is reduced, so every rank stops together.
        if gamma == 0 or qq == 0:
            break
        alpha = (gamma / qq).float()
        x += alpha * p
        r -= alpha * q
        s = operator.backproject(r)
        gamma_new = torch.sum(s.double() * s.double())
        p = s + (gamma_new / gamma).float() * p
        gamma = gamma_new
    return x


def sirt(operator, measurements, iterations):
    """SIRT with nonnegativity: x += C A^T R (y - A x), R = 1/(A 1), C = 1/(A^T 1).

    R lives on the rank's view shard and C on the replicated volume, so the
    update needs no extra communication in distributed mode.
    """
    ones = torch.ones(operator.volume_shape, device=measurements.device)
    row = operator.project(ones)
    row_inverse = torch.where(row > 0, 1.0 / row, torch.zeros_like(row))
    column = operator.backproject(torch.ones_like(measurements))
    column_inverse = torch.where(column > 0, 1.0 / column, torch.zeros_like(column))
    x = torch.zeros_like(ones)
    for _ in range(iterations):
        residual = (measurements - operator.project(x)) * row_inverse
        x = (x + column_inverse * operator.backproject(residual)).clamp_min(0)
    return x


def total_variation(x, eps=1e-6):
    """Isotropic total variation of a 3D volume (mean over voxels)."""
    dz = x[1:, :-1, :-1] - x[:-1, :-1, :-1]
    dy = x[:-1, 1:, :-1] - x[:-1, :-1, :-1]
    dx = x[:-1, :-1, 1:] - x[:-1, :-1, :-1]
    return torch.sqrt(dx * dx + dy * dy + dz * dz + eps).mean()


def tv_reconstruction(operator, measurements, iterations, weight=1.0, lr=0.1):
    """Nonnegative least squares with a TV penalty, solved by Adam through autograd.

    The data term is the mean squared residual per ray, so ``weight`` does not
    depend on the scan size. Every rank adds the full TV term: the projector already sums the data
    gradient over ranks, while the TV gradient is computed identically on
    each rank and must not be divided by the number of ranks.
    """
    rays = torch.tensor(float(measurements.numel()), device=measurements.device)
    if operator.world_size > 1:
        dist.all_reduce(rays)
    x = torch.zeros(operator.volume_shape, device=measurements.device, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=lr)
    for _ in range(iterations):
        optimizer.zero_grad()
        data = 0.5 * (operator.project(x) - measurements).square().sum() / rays
        loss = data + weight * total_variation(x)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(min=0)
    return x.detach()


def psnr(reconstruction, truth):
    """Peak signal-to-noise ratio in dB for images with peak value 1."""
    mse = torch.mean((reconstruction - truth) ** 2).item()
    return 10 * math.log10(1.0 / mse)


def init_process():
    """Use the GPU of this process; start NCCL when launched by torchrun."""
    if not torch.cuda.is_available():
        raise RuntimeError("These examples need a CUDA GPU")
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl", device_id=device)
    return distributed, device


def synchronize(distributed):
    torch.cuda.synchronize()
    if distributed:
        dist.barrier()


def save_slices(path, panels, title):
    """Save images (2D) or central slices (3D) side by side; skipped without matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping", path)
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    for ax, (label, volume) in zip(np.atleast_1d(axes), panels):
        image = volume[volume.shape[0] // 2] if volume.ndim == 3 else volume
        ax.imshow(image.detach().cpu().numpy(), cmap="gray", vmin=0.0, vmax=0.4)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("saved", path)
