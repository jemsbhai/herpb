"""Generate figures from EXP-003 addition depth characterization.

Usage:
    python scripts/generate_add_depth_figures.py

Produces:
  - fig9_add_depth_boundary.pdf   Max depth vs operand bits (3 key sizes)
  - fig10_add_depth_heatmap.pdf   Correctness heatmap (operand × depth)
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.figsize": (7, 4.5),
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_summary():
    with open("experiments/exp_003_add_depth/summary.json") as f:
        return json.load(f)


def fig_depth_boundary(summary):
    """Max successful depth vs operand bits for each key size."""
    fig, ax = plt.subplots()

    key_configs = [
        (1024, "#67a9cf", "s", "1024-bit key"),
        (2048, "#2166ac", "o", "2048-bit key"),
        (4096, "#053061", "^", "4096-bit key"),
    ]

    op_bits_all = sorted(set(
        v["operand_bits"] for v in summary.values()
    ))

    for key_bits, color, marker, label in key_configs:
        ob_list = []
        actual_list = []
        predicted_list = []

        for ob in op_bits_all:
            skey = f"{key_bits}bit_op{ob}"
            if skey in summary:
                s = summary[skey]
                ob_list.append(ob)
                actual_list.append(max(1, s["actual_max_100pct_depth"]))
                predicted_list.append(s["predicted_max_depth"])

        ax.plot(ob_list, actual_list, f"{marker}-", color=color,
                label=f"{label} (actual)", zorder=3)

    # Plot theoretical line for 2048-bit key
    ob_range = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512])
    theory_line = 1024 / ob_range  # plaintext_bits / operand_bits
    ax.plot(ob_range, theory_line, "--", color="#999999", alpha=0.7,
            label="Predicted (key/2 ÷ op_bits)", zorder=2)

    ax.set_xlabel("Operand size (bits)")
    ax.set_ylabel("Max addition depth (100% correct)")
    ax.set_title("MRPB Addition Depth Limit vs Operand Size")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(op_bits_all)
    ax.set_xticklabels([str(b) for b in op_bits_all], rotation=45, ha="right")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", ls="--")

    for ext in [".pdf", ".png"]:
        fig.savefig(f"figures/fig9_add_depth_boundary{ext}")
    print("  Saved: fig9_add_depth_boundary.pdf + .png")
    plt.close(fig)


def fig_heatmap(summary):
    """Correctness heatmap for 2048-bit key: operand_bits × depth."""
    fig, ax = plt.subplots(figsize=(9, 5))

    key_bits = 2048
    op_bits_all = sorted(set(
        v["operand_bits"] for v in summary.values()
        if v["key_bits"] == key_bits
    ))
    all_depths = sorted(set(
        int(d) for skey, v in summary.items()
        if v["key_bits"] == key_bits
        for d in v["by_depth"].keys()
    ))

    matrix = np.full((len(op_bits_all), len(all_depths)), np.nan)

    for i, ob in enumerate(op_bits_all):
        skey = f"{key_bits}bit_op{ob}"
        if skey in summary:
            for j, d in enumerate(all_depths):
                ds = str(d)
                if ds in summary[skey]["by_depth"]:
                    matrix[i, j] = summary[skey]["by_depth"][ds]["correctness"]

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(all_depths)))
    ax.set_xticklabels([str(d) for d in all_depths], rotation=45, ha="right")
    ax.set_yticks(range(len(op_bits_all)))
    ax.set_yticklabels([f"{b}-bit" for b in op_bits_all])

    ax.set_xlabel("Addition chain depth")
    ax.set_ylabel("Operand size")
    ax.set_title(f"MRPB Addition Correctness ({key_bits}-bit key)")

    # Annotate cells
    for i in range(len(op_bits_all)):
        for j in range(len(all_depths)):
            val = matrix[i, j]
            if not np.isnan(val):
                text = f"{val:.0%}" if val in (0, 1) else f"{val:.0%}"
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, label="Correctness rate", shrink=0.8)

    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(f"figures/fig10_add_depth_heatmap{ext}")
    print("  Saved: fig10_add_depth_heatmap.pdf + .png")
    plt.close(fig)


def main():
    print("Loading EXP-003 summary...")
    summary = load_summary()
    print(f"  {len(summary)} entries loaded")

    print("\nGenerating figures...")
    fig_depth_boundary(summary)
    fig_heatmap(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
