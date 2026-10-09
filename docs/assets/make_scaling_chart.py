"""Render scaling_light.png and scaling_dark.png from the measured timings in scaling.json.

    python docs/assets/make_scaling_chart.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OPERATIONS = [("forward_ms", "forward"), ("adjoint_ms", "adjoint"), ("cgls_ms_per_iter", "CGLS iteration")]
THEMES = {
    "light": dict(surface="#ffffff", ink="#1f2328", muted="#59636e", grid="#e6e8eb",
                  bars=["#9cc3f0", "#3b82d6", "#163f7a"]),
    "dark": dict(surface="#0d1117", ink="#f0f6fc", muted="#9198a1", grid="#262c36",
                 bars=["#1f4f8f", "#3f8ae6", "#a8cdf7"]),
}


def main():
    record = json.loads((HERE / "scaling.json").read_text())
    configs = record["configurations"]
    sizes = list(record["timings"])
    for theme, colors in THEMES.items():
        fig, axes = plt.subplots(1, len(sizes), figsize=(12, 4.6), facecolor=colors["surface"])
        for ax, size in zip(np.atleast_1d(axes), sizes):
            timings = record["timings"][size]
            ax.set_facecolor(colors["surface"])
            x = np.arange(len(OPERATIONS))
            width = 0.25
            for k, config in enumerate(configs):
                values = [timings[config["key"]][op] for op, _ in OPERATIONS]
                base = [timings[configs[0]["key"]][op] for op, _ in OPERATIONS]
                xs = x + (k - 1) * (width + 0.025)
                ax.bar(xs, values, width, color=colors["bars"][k], label=config["label"], zorder=2)
                for xi, value, reference in zip(xs, values, base):
                    text = f"{value:.1f}" if k == 0 else f"{reference / value:.1f}×"
                    ax.text(xi, value, text, ha="center", va="bottom", fontsize=9.5,
                            color=colors["ink"] if k else colors["muted"])
            ax.set_xticks(x, [label for _, label in OPERATIONS], color=colors["ink"], fontsize=11.5)
            ax.set_title(f"{size}³ volume", color=colors["ink"], fontsize=13, loc="left", pad=10)
            ax.set_ylabel("time (ms)", color=colors["muted"], fontsize=10.5)
            ax.tick_params(colors=colors["muted"], labelsize=10, length=0)
            ax.grid(axis="y", color=colors["grid"], linewidth=0.9, zorder=0)
            ax.margins(y=0.12)
            for spine in ax.spines.values():
                spine.set_visible(False)
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, frameon=False, fontsize=11, loc="upper center", ncol=len(configs),
                            bbox_to_anchor=(0.5, 1.0))
        for text in legend.get_texts():
            text.set_color(colors["ink"])
        fig.text(0.01, 0.015, record["caption"], color=colors["muted"], fontsize=9.5)
        fig.tight_layout(rect=(0, 0.05, 1, 0.92), w_pad=3.0)
        fig.savefig(HERE / f"scaling_{theme}.png", dpi=220, facecolor=colors["surface"])
        plt.close(fig)
        print("saved", HERE / f"scaling_{theme}.png")


if __name__ == "__main__":
    main()
