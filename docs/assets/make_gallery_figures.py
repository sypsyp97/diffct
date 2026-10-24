"""Render the documentation gallery figures into docs/source/_static/gallery/.

Projection and reconstruction figures run diffct with ``backend="torch"`` on the
CPU, so no GPU is needed. Each figure reproduces the pipeline of one example
page at the geometry stated in its title. The walnut figures and the rotating
walnut reuse the archived arrays in docs/video/ without recomputation.
Run from the repository root:

    uv run --no-project --with torch --with numba --with matplotlib --with-editable . \
        python docs/assets/make_gallery_figures.py
"""
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

from diffct import (  # noqa: E402
    Projector,
    circular_trajectory_2d_fan,
    circular_trajectory_2d_parallel,
    circular_trajectory_3d,
    fbp,
    fdk,
    saddle_trajectory_3d,
    sinusoidal_trajectory_3d,
    spiral_trajectory_3d,
)
from _common import Scan, psnr, shepp_logan_3d  # noqa: E402

OUT = ROOT / "docs" / "source" / "_static" / "gallery"
VIDEO = ROOT / "docs" / "video"
BACKGROUND, INK, MUTED, ACCENT = "#111418", "#ECEAE6", "#9AA3AD", "#F2A541"
WINDOW = "shepp-logan"
# Display window for Shepp-Logan images: the 0.2 soft tissue and its 0.1 features.
PHANTOM = (0.05, 0.45)


def panels(filename, images, title):
    """Save a row of grayscale panels; ``images`` holds (array, name, detail, limits)."""
    fig, axes = plt.subplots(1, len(images), figsize=(2.6 * len(images), 3.0), squeeze=False,
                             facecolor=BACKGROUND)
    for ax, (image, name, detail, limits) in zip(axes[0], images):
        ax.imshow(image, cmap="gray", vmin=limits[0], vmax=limits[1], origin="upper",
                  aspect="auto" if image.shape[0] != image.shape[1] else "equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(name, color=INK, fontsize=11, pad=16 if detail else 6)
        if detail:
            ax.text(.5, 1.02, detail, transform=ax.transAxes, ha="center", va="bottom",
                    color=MUTED, fontsize=9)
    save(fig, filename, title)


def save(fig, filename, title):
    fig.suptitle(title, color=INK, fontsize=11, x=.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, .95), w_pad=.6)
    fig.savefig(OUT / filename, dpi=150, facecolor=BACKGROUND)
    plt.close(fig)
    print("saved", filename)


def gray(x):
    return x.detach().cpu().numpy()


def analytical_2d(image, n):
    views = 360
    # Parallel beam over 180 degrees; detector pitch 0.5 voxel (examples/analytical_reconstruction.py).
    trajectory = circular_trajectory_2d_parallel(views, end_angle=math.pi, device="cpu")
    sinogram = Projector(trajectory, (n, n), 3 * n, beam="parallel", detector_spacing=0.5,
                         backend="torch").project(image)
    angles = torch.linspace(0.0, math.pi, views + 1)[:-1]
    parallel = fbp(sinogram, trajectory, (n, n), angles, detector_spacing=0.5, window=WINDOW,
                   backend="torch").clamp_min(0)
    panels("fbp_parallel.png", [
        (gray(image), "Phantom", f"{n}²", PHANTOM),
        (gray(sinogram).T, "Sinogram", f"{views} views, 180°", (0, gray(sinogram).max())),
        (gray(parallel), "FBP", f"{psnr(parallel, image):.1f} dB", PHANTOM),
    ], "Parallel-beam FBP, Shepp-Logan window")

    # Fan beam over a full turn, magnification 1.6.
    sid, sdd, pitch = 2.5 * n, 4.0 * n, 0.8
    trajectory = circular_trajectory_2d_fan(views, sid, sdd, device="cpu")
    sinogram = Projector(trajectory, (n, n), 3 * n, beam="fan", detector_spacing=pitch,
                         backend="torch").project(image)
    angles = torch.linspace(0.0, 2 * math.pi, views + 1)[:-1]
    fan = fbp(sinogram, trajectory, (n, n), angles, beam="fan", detector_spacing=pitch,
              window=WINDOW, isocenter=(0, 0), backend="torch").clamp_min(0)
    panels("fbp_fan.png", [
        (gray(image), "Phantom", f"{n}²", PHANTOM),
        (gray(sinogram).T, "Sinogram", f"{views} views, 360°", (0, gray(sinogram).max())),
        (gray(fan), "FBP", f"{psnr(fan, image):.1f} dB", PHANTOM),
    ], "Fan-beam FBP, magnification 1.6")


def cone_fdk(n):
    phantom = torch.from_numpy(shepp_logan_3d(n))
    scan = Scan(n, 360)
    trajectory = scan.trajectory("circular")
    sinogram = Projector(trajectory, (n, n, n), scan.detector, detector_spacing=scan.pitch,
                         backend="torch").project(phantom)
    angles = torch.linspace(0.0, 2 * math.pi, scan.views + 1)[:-1]
    volume = fdk(sinogram, trajectory, (n, n, n), angles, detector_spacing=scan.pitch,
                 window=WINDOW, isocenter=(0, 0, 0), backend="torch").clamp_min(0)
    v = gray(sinogram[0]).T
    panels("fdk_cone.png", [
        (v, "Projection", "view 0", (0, v.max())),
        (gray(volume[n // 2]), "FDK axial", f"{psnr(volume, phantom):.1f} dB (volume)", PHANTOM),
        (gray(volume[:, n // 2]), "FDK coronal", "", PHANTOM),
        (gray(phantom[:, n // 2]), "Phantom coronal", f"{n}³", PHANTOM),
    ], "Cone-beam FDK, circular orbit, 360 views")


def iterative_2d(beam, n=128):
    # The loop shown on the iterative example pages, with the PyTorch backend on CPU.
    views = 360
    if beam == "parallel":
        trajectory = circular_trajectory_2d_parallel(views, device="cpu")
        operator = Projector(trajectory, (n, n), 384, beam="parallel", detector_spacing=0.5,
                             backend="torch")
    else:
        trajectory = circular_trajectory_2d_fan(views, 2.5 * n, 4.0 * n, device="cpu")
        operator = Projector(trajectory, (n, n), 3 * n, beam="fan", detector_spacing=0.8,
                             backend="torch")
    truth = torch.zeros((n, n))
    truth[40:88, 40:88] = 1.0
    measured = operator.project(truth).detach()
    image = torch.zeros((n, n), requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=1e-2)
    losses, snapshots = [], {}
    for step in range(200):
        optimizer.zero_grad()
        loss = 0.5 * (operator.project(image) - measured).square().sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            image.clamp_(min=0)
        losses.append(loss.item())
        if step + 1 in (50, 200):
            snapshots[step + 1] = image.detach().clone()

    fig = plt.figure(figsize=(10.4, 3.0), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 4)
    for column, (array, name, detail) in enumerate([
            (truth, "Truth", f"{n}²"),
            (snapshots[50], "Step 50", ""),
            (snapshots[200], "Step 200", f"{psnr(snapshots[200], truth):.1f} dB")]):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(gray(array), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(name, color=INK, fontsize=11, pad=16 if detail else 6)
        if detail:
            ax.text(.5, 1.02, detail, transform=ax.transAxes, ha="center", va="bottom",
                    color=MUTED, fontsize=9)
    ax = fig.add_subplot(grid[0, 3], facecolor=BACKGROUND)
    ax.semilogy(np.arange(1, 201), losses, color=ACCENT, linewidth=1.6)
    ax.set_title("Data loss", color=INK, fontsize=11, pad=6)
    ax.set_xlabel("Adam step", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#3A4048")
    save(fig, f"iterative_{beam}.png",
         f"{beam.capitalize()}-beam least squares with autograd, Adam, 200 steps")


def walnut():
    measured = np.load(VIDEO / "walnut_measured.npz")
    panels("walnut.png", [
        (measured[f"{key}_axial"], name, detail, (0, 1))
        for key, name, detail in [("fdk", "FDK", "Hann window"), ("sirt", "SIRT", "200 it"),
                                  ("cgls", "CGLS", "20 it"), ("tv", "TV", "300 it")]
    ], "Measured walnut, 240 views, 256³ (axial slice)")
    simulated = np.load(VIDEO / "recon_slices.npz")
    panels("iterative_cone.png", [
        (simulated[f"{key}_coronal"], name, detail, (0, 1))
        for key, name, detail in [("phantom", "Ground truth", ""), ("fdk", "FDK", ""),
                                  ("cgls", "CGLS", "30 it"), ("sirt", "SIRT", "200 it"),
                                  ("tv", "TV", "200 it")]
    ], "Simulated helical scan, 720 views, 256³ (coronal slice)")


def curved_detector():
    # Geometry of examples/curved_detector.py with a 64³ phantom.
    n = 64
    trajectory = circular_trajectory_3d(90, sid=160.0, sdd=256.0, device="cpu")
    radius = torch.tensor(256.0, dtype=torch.float64)

    def cylinder(u, v):
        angle = u / radius
        return torch.stack((radius * angle.sin(), v, radius * (angle.cos() - 1)), dim=-1)

    operator = Projector(trajectory, (n, n, n), (192, 128), detector_spacing=(0.8, 1.0),
                         detector_surface=cylinder, backend="torch")
    sinogram = operator.project(torch.from_numpy(shepp_logan_3d(n)))
    top = gray(sinogram).max()
    panels("curved_detector.png", [
        (gray(sinogram[view]).T, f"View {view}", f"{4 * view}°", (0, top)) for view in (0, 22, 45)
    ], "Native cylindrical detector, radius = SDD")


def trajectories():
    sid, sdd, n = 2.5, 4.0, 1.0
    shapes = {
        "Circular": circular_trajectory_3d(240, sid, sdd, device="cpu"),
        "Helical": spiral_trajectory_3d(240, sid, sdd, z_range=1.6, n_turns=2.0, device="cpu"),
        "Saddle": saddle_trajectory_3d(240, sid, sdd, z_amplitude=.6, radial_amplitude=.25,
                                       device="cpu"),
        "Sinusoidal": sinusoidal_trajectory_3d(240, sid, sdd, amplitude=.6, frequency=3.0,
                                               device="cpu"),
    }
    fig = plt.figure(figsize=(10.4, 3.0), facecolor=BACKGROUND)
    for column, (name, trajectory) in enumerate(shapes.items()):
        ax = fig.add_subplot(1, 4, column + 1, projection="3d", facecolor=BACKGROUND)
        source = trajectory[0].numpy()
        ax.plot(source[:, 0], source[:, 1], source[:, 2], color=ACCENT, linewidth=1.6)
        u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
        ax.plot_wireframe(.5 * n * np.cos(u) * np.sin(v), .5 * n * np.sin(u) * np.sin(v),
                          .5 * n * np.cos(v), color=MUTED, linewidth=.3)
        ax.set_box_aspect((1, 1, .8))
        ax.set_xlim(-sid, sid)
        ax.set_ylim(-sid, sid)
        ax.set_zlim(-1.6, 1.6)
        ax.view_init(elev=22, azim=-55)
        ax.set_axis_off()
        ax.set_title(name, color=INK, fontsize=11, pad=0)
    save(fig, "trajectories.png", "Source paths of the built-in 3D trajectories")


def analytical_learning(n=64, steps=60):
    # A learnable correction to the ramp filter, trained through fbp() on noisy data.
    torch.manual_seed(7)
    views = 180
    truth = torch.from_numpy(shepp_logan_3d(n))[n // 2]
    trajectory = circular_trajectory_2d_parallel(views, end_angle=math.pi, device="cpu")
    clean = Projector(trajectory, (n, n), 3 * n, beam="parallel", detector_spacing=0.5,
                      backend="torch").project(truth)
    measurements = clean + torch.randn_like(clean) * (0.02 * clean.abs().max())
    angles = torch.arange(views) * (math.pi / views)
    bins = measurements.shape[1]
    fft_size = 2 * bins
    lag = torch.arange(fft_size)
    lag = torch.minimum(lag, fft_size - lag)
    impulse = torch.zeros(fft_size)
    impulse[0] = math.pi / 2
    odd = lag % 2 == 1
    impulse[odd] = -2 / (math.pi * lag[odd].float().square())
    ramp = torch.fft.rfft(impulse).real / 0.5
    log_gain = torch.nn.Parameter(torch.zeros_like(ramp))

    def reconstruct():
        return fbp(measurements, trajectory, (n, n), angles, detector_spacing=0.5,
                   filter_response=ramp * log_gain.exp(), backend="torch")

    with torch.no_grad():
        before = reconstruct()
    optimizer = torch.optim.Adam([log_gain], lr=0.02)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (reconstruct() - truth).square().mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = reconstruct()

    fig = plt.figure(figsize=(10.4, 3.0), facecolor=BACKGROUND)
    grid = fig.add_gridspec(1, 4)
    for column, (array, name, detail) in enumerate([
            (truth, "Truth", f"{n}², 2% noise"),
            (before, "Ram-Lak", f"{psnr(before, truth):.1f} dB"),
            (after, f"Learned, {steps} steps", f"{psnr(after, truth):.1f} dB")]):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(gray(array), cmap="gray", vmin=PHANTOM[0], vmax=PHANTOM[1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(name, color=INK, fontsize=11, pad=16)
        ax.text(.5, 1.02, detail, transform=ax.transAxes, ha="center", va="bottom",
                color=MUTED, fontsize=9)
    ax = fig.add_subplot(grid[0, 3], facecolor=BACKGROUND)
    frequency = np.linspace(0, .5, ramp.numel())
    ax.plot(frequency, gray(ramp), color=MUTED, linewidth=1.4, label="Ram-Lak")
    ax.plot(frequency, gray(ramp * log_gain.exp()), color=ACCENT, linewidth=1.6, label="Learned")
    ax.set_title("Filter response", color=INK, fontsize=11, pad=6)
    ax.set_xlabel("cycles / sample", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, labelcolor=INK, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color("#3A4048")
    save(fig, "analytical_learning.png", "Learning the FBP filter through autograd")


def rotating_walnut(frames=60, size=360):
    """Animated WebP of the archived walnut surface render, with a transparent background."""
    surface = np.load(VIDEO / "walnut_surface.npz")["frames"]
    step = len(surface) // frames
    images = [Image.fromarray(frame).resize((size, size), Image.LANCZOS) for frame in surface[::step]]
    images[0].save(OUT / "walnut_rotating.webp", save_all=True, append_images=images[1:],
                   duration=1000 * 6 // len(images), loop=0, lossless=False, quality=80, method=6)
    print("saved walnut_rotating.webp")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    n = 256
    analytical_2d(torch.from_numpy(shepp_logan_3d(n))[n // 2], n)
    cone_fdk(96)
    iterative_2d("parallel")
    iterative_2d("fan")
    walnut()
    curved_detector()
    trajectories()
    analytical_learning()
    rotating_walnut()


if __name__ == "__main__":
    main()
