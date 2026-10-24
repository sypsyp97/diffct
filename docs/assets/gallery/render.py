"""Draw the documentation figures from the arrays of compute.py.

    python docs/assets/gallery/render.py --data DIR

DIR holds the ``.npz`` files of compute.py. The walnut figures read the archived
arrays in docs/video/. Figures go to docs/source/_static/gallery/: one full figure
per page and a ``thumb_*.png`` card image per example.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "source" / "_static" / "gallery"
VIDEO = ROOT / "docs" / "video"
BACKGROUND, INK, MUTED, ACCENT, GRID = "#0f1216", "#eceae6", "#9aa3ad", "#f2a541", "#3a4048"
DPI = 220

# Display windows. Patient data are in water units, x = 1 + HU / 1000.
SOFT_TISSUE = (0.84, 1.24)   # HU 40 / 400
LUNG = (0.0, 1.15)           # HU -600 / 1500
SHEPP = (0.05, 0.45)         # the 0.2 soft tissue and its 0.1 features
UNIT = (0.0, 1.0)


def style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def image(ax, array, limits, title=None, detail=None, aspect="equal"):
    ax.imshow(np.asarray(array, dtype=np.float32), cmap="gray", vmin=limits[0], vmax=limits[1],
              interpolation="lanczos", aspect=aspect)
    style(ax)
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=15 if detail else 5)
    if detail:
        ax.text(.5, 1.015, detail, transform=ax.transAxes, ha="center", va="bottom", color=MUTED, fontsize=8)


def curve_axes(ax, title, xlabel):
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, color=INK, fontsize=10, pad=5)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, linewidth=.4)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def figure(rows, columns, width=2.6, height=2.75):
    fig, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False,
                             facecolor=BACKGROUND)
    return fig, axes


def save(fig, name, row_labels=None, axes=None):
    if row_labels:
        for ax, label in zip(axes[:, 0], row_labels):
            ax.set_ylabel(label, color=MUTED, fontsize=9)
    fig.tight_layout(pad=.6, w_pad=.5, h_pad=.9)
    fig.savefig(OUT / name, dpi=DPI, facecolor=BACKGROUND)
    plt.close(fig)
    print("saved", name)


def thumb(name, array, limits, crop=None, aspect=1.6):
    """A card image: one panel, cropped to a 1.6:1 box around the object."""
    array = np.asarray(array, dtype=np.float32)
    if crop is not None:
        array = array[crop]
    h, w = array.shape
    if w / h < aspect:
        cut = int((h - w / aspect) / 2)
        array = array[cut:h - cut]
    else:
        cut = int((w - h * aspect) / 2)
        array = array[:, cut:w - cut]
    fig = plt.figure(figsize=(4.8, 3.0), facecolor=BACKGROUND)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(array, cmap="gray", vmin=limits[0], vmax=limits[1], interpolation="lanczos", aspect="auto")
    style(ax)
    fig.savefig(OUT / f"thumb_{name}.png", dpi=160, facecolor=BACKGROUND)
    plt.close(fig)


def body_box(array, threshold, pad=12):
    mask = np.asarray(array, dtype=np.float32) > threshold
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return (slice(max(rows[0] - pad, 0), rows[-1] + pad), slice(max(cols[0] - pad, 0), cols[-1] + pad))


def analytical(data):
    for beam, cases in (("parallel", ("shepp", "abdomen")), ("fan", ("shepp", "chest"))):
        fig, axes = figure(len(cases), 3)
        for row, case in enumerate(cases):
            d = np.load(data / f"fbp_{beam}_{case}.npz")
            limits = SHEPP if case == "shepp" else (LUNG if case == "chest" else SOFT_TISSUE)
            sino = d["sinogram"].astype(np.float32)
            image(axes[row, 0], d["truth"], limits, "Ground truth" if row == 0 else None)
            image(axes[row, 1], sino.T, (0, sino.max()), "Sinogram" if row == 0 else None,
                  f"{int(d['views'])} views" if row == 0 else None, aspect="auto")
            image(axes[row, 2], d["image"], limits, "FBP" if row == 0 else None)
            axes[row, 2].text(.97, .03, f"{float(d['psnr']):.1f} dB", transform=axes[row, 2].transAxes,
                              ha="right", va="bottom", color=INK, fontsize=8)
        labels = ["Shepp-Logan"] + [str(np.load(data / f"fbp_{beam}_{c}.npz")["label"]) for c in cases[1:]]
        save(fig, f"fbp_{beam}.png", labels, axes)
        last = np.load(data / f"fbp_{beam}_{cases[1]}.npz")
        thumb(f"fbp_{beam}", last["image"], LUNG if cases[1] == "chest" else SOFT_TISSUE,
              body_box(last["truth"], 0.5))


def gradient_2d(data):
    fig, axes = figure(2, 5)
    for row, beam in enumerate(("parallel", "fan")):
        d = np.load(data / f"iterative_{beam}.npz")
        image(axes[row, 0], d["truth"], SOFT_TISSUE, "Ground truth" if row == 0 else None)
        for column, (step, value, quality) in enumerate(zip(d["steps"], d["images"], d["psnr"]), start=1):
            image(axes[row, column], value, SOFT_TISSUE, f"Adam step {int(step)}" if row == 0 else None)
            axes[row, column].text(.97, .03, f"{quality:.1f} dB", transform=axes[row, column].transAxes,
                                   ha="right", va="bottom", color=INK, fontsize=8)
        ax = axes[row, 4]
        curve_axes(ax, "Data loss" if row == 0 else "", "Adam step")
        ax.semilogy(np.arange(1, len(d["losses"]) + 1), d["losses"], color=ACCENT, linewidth=1.4)
        ax.set_box_aspect(1)
    save(fig, "gradient_2d.png", ["Parallel beam", "Fan beam"], axes)


def gradient_walnut(data):
    d = np.load(data / "gradient_walnut.npz")
    steps = [int(s) for s in d["steps"]]
    fig, axes = figure(2, len(steps) + 2)
    for row, section in enumerate(("axial", "coronal")):
        image(axes[row, 0], d[f"fdk_{section}"], UNIT, "FDK reference" if row == 0 else None)
        for column, step in enumerate(steps, start=1):
            image(axes[row, column], d[f"{section}_{step}"], UNIT, f"Adam step {step}" if row == 0 else None)
    for row in range(2):
        axes[row, -1].remove()
    ax = fig.add_subplot(2, len(steps) + 2, len(steps) + 2)
    curve_axes(ax, "MSE to the measurement", "Adam step")
    ax.semilogy(np.arange(1, len(d["losses"]) + 1), d["losses"], color=ACCENT, linewidth=1.4)
    ax.set_box_aspect(1)
    save(fig, "gradient_walnut.png", ["Axial", "Coronal"], axes)
    thumb("gradient", d[f"axial_{steps[-1]}"], UNIT, body_box(d[f"axial_{steps[-1]}"], 0.15))


def helical(data, case, limits, iteration, feet_first):
    """Axial slices in radiological view (patient left on the image right, anterior up),
    coronal and sagittal sections head up. ``feet_first``: scanner z grows toward the head."""
    d = np.load(data / f"helical_{case}.npz")
    axial = d[f"axial_{iteration}"][:, ::-1, ::-1]
    coronal = d[f"coronal_{iteration}"][:, ::-1]
    sagittal = d[f"sagittal_{iteration}"][:, ::-1]
    if not feet_first:
        coronal, sagittal = coronal[::-1], sagittal[::-1]
    box = body_box(axial[len(axial) // 2], 0.5)
    rows = np.where((coronal > 0.5).mean(1) > 0.05)[0]
    zrange = slice(rows[0] + 30, rows[-1] - 30)  # drop the partially covered ends of the scan
    fig = plt.figure(figsize=(13.0, 7.2), facecolor=BACKGROUND)
    grid = fig.add_gridspec(2, 3)
    for column, (slab, z) in enumerate(zip(axial, d["axial_z"])):
        image(fig.add_subplot(grid[0, column]), slab[box], limits, f"Axial, z = {z:.0f} mm")
    image(fig.add_subplot(grid[1, 0]), coronal[zrange][:, box[1]], limits, "Coronal")
    image(fig.add_subplot(grid[1, 1]), sagittal[zrange][:, box[0]], limits, "Sagittal")
    ax = fig.add_subplot(grid[1, 2])
    curve_axes(ax, "CGLS residual", "iteration")
    ax.semilogy(np.arange(len(d["residual"])), d["residual"], color=ACCENT, linewidth=1.4)
    ax.set_box_aspect(.8)
    save(fig, f"helical_{case}.png")
    fig, axes = figure(1, len(d["iterations"]), width=3.6, height=3.0)
    for ax, it in zip(axes[0], d["iterations"]):
        image(ax, d[f"axial_{int(it)}"][1][::-1, ::-1][box], limits, f"CGLS, {int(it)} iterations")
    save(fig, f"helical_{case}_iterations.png")
    return axial[1][box], coronal[zrange][:, box[1]]


def hero(axial, coronal):
    fig = plt.figure(figsize=(6.4, 4.0), facecolor="black")
    ax = fig.add_axes((0, 0, .58, 1))
    ax.imshow(np.asarray(axial, np.float32), cmap="gray", vmin=SOFT_TISSUE[0], vmax=SOFT_TISSUE[1],
              interpolation="lanczos")
    style(ax)
    ax = fig.add_axes((.6, 0, .4, 1))
    ax.imshow(np.asarray(coronal, np.float32), cmap="gray", vmin=SOFT_TISSUE[0], vmax=SOFT_TISSUE[1],
              interpolation="lanczos", aspect="auto")
    style(ax)
    fig.savefig(OUT / "hero.png", dpi=250, facecolor="black")
    plt.close(fig)
    print("saved hero.png")


def fdk_cone(data):
    d = np.load(data / "fdk_cone.npz")
    fig, axes = figure(2, 3)
    projection = d["projection"].astype(np.float32)
    image(axes[0, 0], projection, (0, projection.max()), "Projection", "view 0", aspect="auto")
    image(axes[0, 1], d["axial"], SHEPP, "FDK, central slice")
    image(axes[0, 2], d["off_axial"], SHEPP, "FDK, off-centre slice")
    image(axes[1, 0], d["phantom_coronal"], SHEPP, "Phantom, coronal")
    image(axes[1, 1], d["coronal"], SHEPP, "FDK, coronal")
    image(axes[1, 2], d["phantom_off_axial"], SHEPP, "Phantom, off-centre slice")
    save(fig, "fdk_cone.png")
    thumb("fdk_cone", d["coronal"], SHEPP)


def curved(data):
    d = np.load(data / "curved_detector.npz")
    fig, axes = figure(1, 3, width=3.6, height=2.4)
    top = max(float(p.max()) for p in d["projections"])
    for ax, projection, angle in zip(axes[0], d["projections"], d["angles"]):
        image(ax, projection[::-1], (0, top), f"{angle:.0f}°", aspect="auto")  # head up
    save(fig, "curved_detector.png")
    thumb("curved_detector", d["projections"][0][::-1], (0, top))


def learning(data):
    d = np.load(data / "learning.npz")
    fig, axes = figure(1, 4)
    image(axes[0, 0], d["truth"], SOFT_TISSUE, "Ground truth", f"{int(d['views'])} views, 2% noise")
    image(axes[0, 1], d["before"], SOFT_TISSUE, "Ram-Lak filter", f"{d['psnr'][0]:.1f} dB")
    image(axes[0, 2], d["after"], SOFT_TISSUE, f"Learned, {int(d['steps'])} steps", f"{d['psnr'][1]:.1f} dB")
    ax = axes[0, 3]
    curve_axes(ax, "Filter response", "cycles / sample")
    frequency = np.linspace(0, .5, len(d["ramp"]))
    ax.plot(frequency, d["ramp"], color=MUTED, linewidth=1.2, label="Ram-Lak")
    ax.plot(frequency, d["learned"], color=ACCENT, linewidth=1.4, label="Learned")
    ax.legend(frameon=False, labelcolor=INK, fontsize=7)
    ax.set_box_aspect(1)
    save(fig, "learning.png")
    thumb("learning", d["after"], SOFT_TISSUE, body_box(d["truth"], .5))


def walnut():
    measured = np.load(VIDEO / "walnut_measured.npz")
    names = [("fdk", "FDK"), ("sirt", "SIRT, 200 it"), ("cgls", "CGLS, 20 it"), ("tv", "TV, 300 it")]
    box = body_box(measured["fdk_axial"], .15, 6)
    fig, axes = figure(2, 4)
    for column, (key, name) in enumerate(names):
        image(axes[0, column], measured[f"{key}_axial"][box], UNIT, name)
        image(axes[1, column], measured[f"{key}_coronal"][box], UNIT)
    save(fig, "walnut.png", ["Axial", "Coronal"], axes)
    thumb("walnut", measured["tv_axial"], UNIT, box)
    simulated = np.load(VIDEO / "recon_slices.npz")
    names = [("phantom", "Ground truth"), ("fdk", "FDK"), ("cgls", "CGLS, 30 it"), ("sirt", "SIRT, 200 it"),
             ("tv", "TV, 200 it")]
    fig, axes = figure(2, 5)
    for column, (key, name) in enumerate(names):
        image(axes[0, column], simulated[f"{key}_axial"][box], UNIT, name)
        image(axes[1, column], simulated[f"{key}_coronal"][box], UNIT)
    save(fig, "iterative_cone.png", ["Axial", "Coronal"], axes)
    thumb("iterative_cone", simulated["tv_coronal"], UNIT, box)


def trajectories():
    sys.path.insert(0, str(ROOT))
    from diffct import (circular_trajectory_3d, saddle_trajectory_3d, sinusoidal_trajectory_3d,
                        spiral_trajectory_3d)
    sid, sdd = 2.5, 4.0
    shapes = {
        "Circular": circular_trajectory_3d(360, sid, sdd, device="cpu"),
        "Helical": spiral_trajectory_3d(720, sid, sdd, z_range=2.0, n_turns=3.0, device="cpu"),
        "Saddle": saddle_trajectory_3d(360, sid, sdd, z_amplitude=.6, radial_amplitude=0.0, device="cpu"),
        "Sinusoidal": sinusoidal_trajectory_3d(360, sid, sdd, amplitude=.6, frequency=3.0, device="cpu"),
    }
    fig = plt.figure(figsize=(12.0, 3.4), facecolor=BACKGROUND)
    for column, (name, trajectory) in enumerate(shapes.items()):
        ax = fig.add_subplot(1, 4, column + 1, projection="3d", facecolor=BACKGROUND)
        source = trajectory[0].numpy()
        ax.plot(*source.T, color=ACCENT, linewidth=1.6)
        ax.scatter(*source[0], color=ACCENT, s=18)
        u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:16j]
        ax.plot_surface(.8 * np.cos(u) * np.sin(v), .8 * np.sin(u) * np.sin(v), .8 * np.cos(v),
                        color="#6c7a89", alpha=.35, linewidth=0)
        ax.set_box_aspect((1, 1, .7))
        ax.set_xlim(-sid, sid)
        ax.set_ylim(-sid, sid)
        ax.set_zlim(-1.4, 1.4)
        ax.view_init(elev=24, azim=-58)
        ax.set_axis_off()
        ax.set_title(name, color=INK, fontsize=11, pad=-4)
    fig.subplots_adjust(left=0, right=1, top=.92, bottom=0, wspace=0)
    fig.savefig(OUT / "trajectories.png", dpi=DPI, facecolor=BACKGROUND)
    plt.close(fig)
    print("saved trajectories.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30, help="CGLS iteration shown for the helical scans")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    analytical(args.data)
    gradient_2d(args.data)
    gradient_walnut(args.data)
    abdomen = helical(args.data, "L213", SOFT_TISSUE, args.iteration, feet_first=False)
    chest = helical(args.data, "C002", LUNG, args.iteration, feet_first=True)
    hero(*abdomen)
    thumb("helical", abdomen[0], SOFT_TISSUE)
    thumb("helical_chest", chest[0], LUNG)
    fdk_cone(args.data)
    curved(args.data)
    learning(args.data)
    walnut()
    trajectories()


if __name__ == "__main__":
    main()
