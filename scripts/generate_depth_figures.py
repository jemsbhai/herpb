"""Generate publication-ready figures from EXP-002 depth benchmark results.

Usage:
    python scripts/generate_depth_figures.py

Produces figures in figures/ directory:
  - fig6_eval_time_depth.pdf     Eval time vs depth (all circuits)
  - fig7_ct_size_depth.pdf       RPB ciphertext growth vs depth
  - fig8_depth_overhead.pdf      MRPB/RPB eval overhead by circuit type
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    "lines.markersize": 6,
    "grid.alpha": 0.3,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def load_summary():
    p = Path("experiments/exp_002_depth/summary.json")
    with open(p) as f:
        return json.load(f)


def extract(summary, circuit, scheme, metric):
    """Extract metric across all depths for a given circuit+scheme."""
    depths = sorted(set(
        v["depth"] for v in summary.values() if v["circuit"] == circuit
    ))
    vals = []
    for d in depths:
        key = f"{circuit}_depth{d}"
        vals.append(summary[key][scheme][metric])
    return depths, vals


def fig_eval_time(summary):
    """Eval time vs depth for all circuits, both schemes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

    circuit_titles = {
        "multiply_chain": "Multiplication Chain",
        "add_chain": "Addition Chain",
        "mixed_alt": "Mixed (Alt Add/Mul)",
    }

    for ax, circuit in zip(axes, ["multiply_chain", "add_chain", "mixed_alt"]):
        depths, rpb_vals = extract(summary, circuit, "rpb", "eval_time_ms_mean")
        _, mrpb_vals = extract(summary, circuit, "mrpb", "eval_time_ms_mean")

        ax.plot(depths, rpb_vals, "o-", color="#2166ac", label="RPB", zorder=3)
        ax.plot(depths, mrpb_vals, "^-", color="#b2182b", label="MRPB", zorder=3)

        ax.set_xlabel("Circuit depth")
        ax.set_title(circuit_titles[circuit])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(depths)
        ax.set_xticklabels([str(d) for d in depths])
        ax.legend(framealpha=0.9)
        ax.grid(True, which="both", ls="--")

    axes[0].set_ylabel("Evaluation time (ms)")
    fig.suptitle("Homomorphic Evaluation Time vs Circuit Depth", fontsize=14, y=1.02)
    fig.tight_layout()

    for ext in [".pdf", ".png"]:
        fig.savefig(f"figures/fig6_eval_time_depth{ext}")
    print("  Saved: fig6_eval_time_depth.pdf + .png")
    plt.close(fig)


def fig_ct_size(summary):
    """RPB ciphertext size growth vs depth."""
    fig, ax = plt.subplots()

    colors = {"multiply_chain": "#2166ac", "add_chain": "#67a9cf",
              "mixed_alt": "#92c5de"}
    labels = {"multiply_chain": "Multiply", "add_chain": "Add",
              "mixed_alt": "Mixed"}
    markers = {"multiply_chain": "o", "add_chain": "s", "mixed_alt": "D"}

    for circuit in ["multiply_chain", "add_chain", "mixed_alt"]:
        depths, vals = extract(summary, circuit, "rpb", "ct_size_bytes_mean")
        ax.plot(depths, vals, f"{markers[circuit]}-",
                color=colors[circuit], label=labels[circuit], zorder=3)

    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Ciphertext size (bytes)")
    ax.set_title("RPB Ciphertext Growth with Depth")
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.legend(framealpha=0.9)
    ax.grid(True, ls="--")

    # Add reference line for MRPB fixed size (512 bytes for 2048-bit key)
    ax.axhline(y=512, color="#b2182b", ls="--", alpha=0.7, label="MRPB fixed (512B)")
    ax.legend(framealpha=0.9)

    for ext in [".pdf", ".png"]:
        fig.savefig(f"figures/fig7_ct_size_depth{ext}")
    print("  Saved: fig7_ct_size_depth.pdf + .png")
    plt.close(fig)


def fig_overhead(summary):
    """MRPB/RPB eval time overhead ratio by circuit type."""
    fig, ax = plt.subplots()

    colors = {"multiply_chain": "#8c510a", "add_chain": "#d8b365",
              "mixed_alt": "#f6e8c3"}
    labels = {"multiply_chain": "Multiply", "add_chain": "Add",
              "mixed_alt": "Mixed"}
    markers = {"multiply_chain": "o", "add_chain": "s", "mixed_alt": "D"}

    for circuit in ["multiply_chain", "add_chain", "mixed_alt"]:
        depths, rpb_vals = extract(summary, circuit, "rpb", "eval_time_ms_mean")
        _, mrpb_vals = extract(summary, circuit, "mrpb", "eval_time_ms_mean")

        # Skip depth 1 where both are near-zero (noisy ratio)
        ratios = []
        valid_depths = []
        for d, r, m in zip(depths, rpb_vals, mrpb_vals):
            if r > 0.0001:
                ratios.append(m / r)
                valid_depths.append(d)

        ax.plot(valid_depths, ratios, f"{markers[circuit]}-",
                color=colors[circuit], label=labels[circuit], zorder=3)

    ax.axhline(y=1, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Overhead ratio (MRPB / RPB)")
    ax.set_title("Modular Arithmetic Overhead vs Circuit Depth")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax.legend(framealpha=0.9)
    ax.grid(True, ls="--")

    for ext in [".pdf", ".png"]:
        fig.savefig(f"figures/fig8_depth_overhead{ext}")
    print("  Saved: fig8_depth_overhead.pdf + .png")
    plt.close(fig)


def main():
    print("Loading EXP-002 summary...")
    summary = load_summary()
    print(f"  {len(summary)} entries loaded")

    print("\nGenerating depth figures...")
    fig_eval_time(summary)
    fig_ct_size(summary)
    fig_overhead(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
