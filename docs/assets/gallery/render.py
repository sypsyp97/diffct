"""Draw the documentation figures from the arrays of compute.py.

    python docs/assets/gallery/render.py --data DIR

DIR holds the ``.npz`` files of compute.py. The walnut figures read the archived
arrays in docs/video/. Figures go to docs/source/_static/gallery/: one figure per
page, drawn at the width the page shows it.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs" / "source" / "_static" / "gallery"
VIDEO = ROOT / "docs" / "video"
BACKGROUND, INK, MUTED, ACCENT, GRID = "#0f1216", "#eceae6", "#a9b1ba", "#f2a541", "#3a4048"
WIDTH, DPI = 7.2, 200  # inches at the content width of a page; about 1440 px
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "font.family": "DejaVu Sans"})

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


def image(ax, array, limits, title=None, aspect="equal", note=None):
    ax.imshow(np.asarray(array, dtype=np.float32), cmap="gray", vmin=limits[0], vmax=limits[1],
              interpolation="lanczos", aspect=aspect)
    style(ax)
    if title:
        ax.set_title(title, color=INK, pad=4)
    if note:
        ax.text(.97, .03, note, transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=8,
                bbox=dict(facecolor="black", alpha=.55, edgecolor="none", pad=1.5))


def curve_axes(ax, title, xlabel):
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, color=INK, pad=4)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.tick_params(which="both", colors=MUTED, labelsize=7.5)
    ax.grid(color=GRID, linewidth=.4)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def figure(rows, columns, height=None):
    side = WIDTH / columns
    fig, axes = plt.subplots(rows, columns, figsize=(WIDTH, height or side * rows * 1.08),
                             squeeze=False, facecolor=BACKGROUND)
    return fig, axes


def save(fig, name, row_labels=None, axes=None):
    if row_labels is not None:
        for ax, label in zip(axes[:, 0], row_labels):
            ax.set_ylabel(label, color=MUTED)
    fig.tight_layout(pad=.4, w_pad=.4, h_pad=.6)
    fig.savefig(OUT / name, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", name)


def body_box(array, threshold, pad=10):
    """Rows and columns where more than 2% of the pixels exceed ``threshold``; thin lines
    such as the patient table do not count."""
    mask = np.asarray(array, dtype=np.float32) > threshold
    rows = np.where(mask.mean(1) > .02)[0]
    cols = np.where(mask.mean(0) > .02)[0]
    return (slice(max(rows[0] - pad, 0), rows[-1] + pad), slice(max(cols[0] - pad, 0), cols[-1] + pad))


def psnr(image, truth):
    """PSNR in dB with the peak of the ground truth as the signal level."""
    image, truth = np.asarray(image, np.float64), np.asarray(truth, np.float64)
    return 10 * np.log10(truth.max() ** 2 / np.mean((image - truth) ** 2))


def patient_limits(case):
    return LUNG if case == "chest" else SOFT_TISSUE


def analytical(data):
    for beam, cases in (("parallel", ("shepp", "abdomen")), ("fan", ("shepp", "chest"))):
        fig, axes = figure(2, 3)
        for row, case in enumerate(cases):
            d = np.load(data / f"fbp_{beam}_{case}.npz")
            limits = SHEPP if case == "shepp" else patient_limits(case)
            box = body_box(d["truth"], limits[0] + .05) if case != "shepp" else (slice(None), slice(None))
            sino = d["sinogram"].astype(np.float32)
            image(axes[row, 0], d["truth"][box], limits, "Ground truth" if row == 0 else None)
            image(axes[row, 1], sino.T, (0, sino.max()), f"Sinogram, {int(d['views'])} views" if row == 0 else None,
                  aspect="auto")
            image(axes[row, 2], d["image"][box], limits, "FBP" if row == 0 else None,
                  note=f"{psnr(d['image'], d['truth']):.1f} dB")
        save(fig, f"fbp_{beam}.png", ["Shepp-Logan", "Patient slice"], axes)


def gradient_2d(data, lr):
    fig, axes = figure(2, 4)
    for row, beam in enumerate(("parallel", "fan")):
        d = np.load(data / f"iterative_{beam}_lr{lr:g}.npz")
        box = body_box(d["truth"], .9)
        image(axes[row, 0], d["truth"][box], SOFT_TISSUE, "Ground truth" if row == 0 else None)
        picks = [0, len(d["steps"]) - 1] if len(d["steps"]) > 2 else range(len(d["steps"]))
        for column, k in enumerate(picks, start=1):
            image(axes[row, column], d["images"][k][box], SOFT_TISSUE,
                  f"Adam step {int(d['steps'][k])}" if row == 0 else None, note=f"{psnr(d['images'][k], d['truth']):.1f} dB")
        ax = axes[row, 3]
        curve_axes(ax, "Data loss" if row == 0 else "", "Adam step")
        ax.semilogy(np.arange(1, len(d["losses"]) + 1), d["losses"], color=ACCENT, linewidth=1.3)
        ax.set_box_aspect(1)
    save(fig, "gradient_2d.png", ["Parallel beam", "Fan beam"], axes)


def gradient_walnut(data):
    d = np.load(data / "gradient_walnut.npz")
    steps = [int(s) for s in d["steps"] if s >= 50]
    box = body_box(d["fdk_axial"], .15)
    boxc = body_box(d["fdk_coronal"], .15)
    fig, axes = figure(2, len(steps) + 1)
    for row, (section, crop) in enumerate((("axial", box), ("coronal", boxc))):
        image(axes[row, 0], d[f"fdk_{section}"][crop], UNIT, "FDK, for reference" if row == 0 else None)
        for column, step in enumerate(steps, start=1):
            image(axes[row, column], d[f"{section}_{step}"][crop], UNIT, f"Adam step {step}" if row == 0 else None)
    save(fig, "gradient_walnut.png", ["Axial", "Coronal"], axes)


def helical(data, case, limits, iteration, feet_first):
    """Axial slices in radiological view (patient left on the image right, anterior up),
    coronal and sagittal sections head up. ``feet_first``: scanner z grows toward the feet."""
    d = np.load(data / f"helical_{case}.npz")
    axial = d[f"axial_{iteration}"][:, ::-1, ::-1]
    coronal = d[f"coronal_{iteration}"][:, ::-1]
    sagittal = d[f"sagittal_{iteration}"][:, ::-1]
    if not feet_first:
        coronal, sagittal = coronal[::-1], sagittal[::-1]
    box = body_box(axial[1], .7)
    rows = np.where((coronal > .7).mean(1) > .05)[0]
    zrange = slice(rows[0] + 30, rows[-1] - 30)  # drop the partially covered ends of the scan
    fig = plt.figure(figsize=(WIDTH, 5.4), facecolor=BACKGROUND)
    grid = fig.add_gridspec(2, 3)
    for column, (slab, z) in enumerate(zip(axial, d["axial_z"])):
        image(fig.add_subplot(grid[0, column]), slab[box], limits, f"Axial, z = {z:.0f} mm")
    image(fig.add_subplot(grid[1, 0]), coronal[zrange][:, box[1]], limits, "Coronal")
    image(fig.add_subplot(grid[1, 1]), sagittal[zrange][:, box[0]], limits, "Sagittal")
    ax = fig.add_subplot(grid[1, 2])
    curve_axes(ax, "CGLS residual", "iteration")
    ax.semilogy(np.arange(len(d["residual"])), d["residual"], color=ACCENT, linewidth=1.3)
    ax.axvline(iteration, color=MUTED, linewidth=.8, linestyle="--")
    ax.set_box_aspect(.9)
    save(fig, f"helical_{case}.png")


def fdk_cone(data):
    d = np.load(data / "fdk_cone.npz")
    fig, axes = figure(2, 3)
    for row, prefix in enumerate(("phantom_", "")):
        image(axes[row, 0], d[f"{prefix}axial"], SHEPP, "Central slice" if row == 0 else None)
        image(axes[row, 1], d[f"{prefix}off_axial"], SHEPP, "Off-centre slice" if row == 0 else None)
        image(axes[row, 2], d[f"{prefix}coronal"], SHEPP, "Coronal" if row == 0 else None,
              note=None if prefix else f"{float(d['psnr']):.1f} dB")
    save(fig, "fdk_cone.png", ["Phantom", "FDK"], axes)


def curved(data):
    d = np.load(data / "curved_detector.npz")
    projections = np.asarray(d["projections"], np.float32)
    top = float(np.percentile(projections, 99.7))
    fig, axes = figure(1, 3, height=2.3)
    for ax, projection, angle in zip(axes[0], projections, d["angles"]):
        image(ax, projection[::-1], (0, top), f"{angle:.0f}°", aspect="auto")  # head up
    save(fig, "curved_detector.png")


def learning(data):
    d = np.load(data / "learning.npz")
    window = (0.7, 1.35)
    box = body_box(d["truth"], .5)
    fig, axes = figure(1, 4, height=2.45)
    image(axes[0, 0], d["truth"][box], window, "Ground truth")
    image(axes[0, 1], d["before"][box], window, "Ram-Lak", note=f"{psnr(d['before'], d['truth']):.1f} dB")
    image(axes[0, 2], d["after"][box], window, "Learned filter", note=f"{psnr(d['after'], d['truth']):.1f} dB")
    ax = axes[0, 3]
    curve_axes(ax, "Filter response", "cycles / sample")
    frequency = np.linspace(0, .5, len(d["ramp"]))
    ax.plot(frequency, d["ramp"], color=MUTED, linewidth=1.1, label="Ram-Lak")
    ax.plot(frequency, d["learned"], color=ACCENT, linewidth=1.3, label="Learned")
    ax.legend(frameon=False, labelcolor=INK, fontsize=7.5)
    ax.set_box_aspect(1)
    save(fig, "learning.png")


def walnut():
    measured = np.load(VIDEO / "walnut_measured.npz")
    box = body_box(measured["fdk_axial"], .15)
    boxc = body_box(measured["fdk_coronal"], .15)
    names = [("fdk", "FDK"), ("sirt", "SIRT, 200 it"), ("cgls", "CGLS, 20 it"), ("tv", "TV, 300 it")]
    fig, axes = figure(2, 4)
    for column, (key, name) in enumerate(names):
        image(axes[0, column], measured[f"{key}_axial"][box], UNIT, name)
        image(axes[1, column], measured[f"{key}_coronal"][boxc], UNIT)
    save(fig, "walnut.png", ["Axial", "Coronal"], axes)
    simulated = np.load(VIDEO / "recon_slices.npz")
    names = [("phantom", "Ground truth"), ("fdk", "FDK"), ("sirt", "SIRT, 200 it"), ("tv", "TV, 200 it")]
    fig, axes = figure(2, 4)
    for column, (key, name) in enumerate(names):
        image(axes[0, column], simulated[f"{key}_axial"][box], UNIT, name)
        image(axes[1, column], simulated[f"{key}_coronal"][boxc], UNIT)
    save(fig, "iterative_cone.png", ["Axial", "Coronal"], axes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30, help="CGLS iteration shown for the helical scans")
    parser.add_argument("--lr", type=float, required=True, help="Adam learning rate of the 2D gradient figure")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    analytical(args.data)
    gradient_2d(args.data, args.lr)
    gradient_walnut(args.data)
    helical(args.data, "L213", SOFT_TISSUE, args.iteration, feet_first=False)
    fdk_cone(args.data)
    curved(args.data)
    learning(args.data)
    walnut()


if __name__ == "__main__":
    main()
