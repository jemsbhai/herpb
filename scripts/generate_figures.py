"""Generate publication-ready figures from EXP-001 benchmark results.

Usage:
    python scripts/generate_figures.py

Produces figures in figures/ directory:
  - fig1_enc_time.pdf      Encryption time comparison (4 variants)
  - fig2_dec_time.pdf      Decryption time comparison
  - fig3_ct_size.pdf       Ciphertext size comparison
  - fig4_keygen_scaling.pdf MRPB keygen time vs key_bits
  - fig5_enc_overhead.pdf  MRPB/RPB encryption overhead ratio
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Style ---
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

COLORS = {
    "rpb_shared":    "#2166ac",
    "rpb_distinct":  "#67a9cf",
    "mrpb_shared":   "#b2182b",
    "mrpb_distinct": "#ef8a62",
}

LABELS = {
    "rpb_shared":    "RPB shared-k",
    "rpb_distinct":  "RPB distinct-k",
    "mrpb_shared":   "MRPB shared-k",
    "mrpb_distinct": "MRPB distinct-k",
}

MARKERS = {
    "rpb_shared":    "o",
    "rpb_distinct":  "s",
    "mrpb_shared":   "^",
    "mrpb_distinct": "D",
}


def load_summary():
    p = Path("experiments/exp_001_benchmark/summary.json")
    if not p.exists():
        print(f"ERROR: {p} not found. Run EXP-001 first.")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def get_bit_labels(summary):
    """Return sorted bit-length labels and numeric values."""
    raw = list(summary.keys())
    parsed = [(k, int(k.replace("bits", ""))) for k in raw]
    parsed.sort(key=lambda x: x[1])
    return [p[0] for p in parsed], [p[1] for p in parsed]


def extract_metric(summary, labels, scheme, metric):
    """Extract a metric array for one scheme across all bit-lengths."""
    return [summary[lbl][scheme][metric] for lbl in labels]


def fig_line_plot(summary, labels, bit_vals, metric, ylabel, title, filename,
                  log_y=True):
    """Generic line plot for a metric across all 4 scheme variants."""
    fig, ax = plt.subplots()

    for scheme in ["rpb_shared", "rpb_distinct", "mrpb_shared", "mrpb_distinct"]:
        vals = extract_metric(summary, labels, scheme, metric)
        ax.plot(bit_vals, vals,
                marker=MARKERS[scheme], color=COLORS[scheme],
                label=LABELS[scheme], zorder=3)

    ax.set_xlabel("Input size (bits)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks(bit_vals)
    ax.set_xticklabels([str(b) for b in bit_vals], rotation=45, ha="right")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", ls="--")

    out = Path("figures") / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved: {out} + .png")
    plt.close(fig)


def fig_ct_size(summary, labels, bit_vals):
    """Ciphertext size: grouped bar chart."""
    fig, ax = plt.subplots()

    x = np.arange(len(bit_vals))
    width = 0.2

    for i, scheme in enumerate(["rpb_shared", "rpb_distinct",
                                 "mrpb_shared", "mrpb_distinct"]):
        vals = extract_metric(summary, labels, scheme, "ct_size_bytes_mean")
        ax.bar(x + i * width, vals, width,
               color=COLORS[scheme], label=LABELS[scheme], zorder=3)

    ax.set_xlabel("Input size (bits)")
    ax.set_ylabel("Ciphertext size (bytes)")
    ax.set_title("Ciphertext Size Comparison")
    ax.set_yscale("log")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([str(b) for b in bit_vals], rotation=45, ha="right")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", ls="--")

    out = Path("figures/fig3_ct_size.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved: {out} + .png")
    plt.close(fig)


def fig_keygen(summary, labels):
    """MRPB keygen time vs key_bits."""
    key_bits = [summary[lbl]["mrpb_key_bits_used"] for lbl in labels]
    keygen_s = [summary[lbl]["mrpb_keygen_time_s"] for lbl in labels]

    fig, ax = plt.subplots()
    ax.plot(key_bits, keygen_s, "o-", color="#b2182b", linewidth=2, markersize=7,
            zorder=3)

    for kb, ks in zip(key_bits, keygen_s):
        if ks > 60:
            lbl = f"{ks/3600:.1f}h" if ks > 3600 else f"{ks/60:.0f}m"
            ax.annotate(lbl, (kb, ks), textcoords="offset points",
                        xytext=(8, 5), fontsize=9, color="#b2182b")

    ax.set_xlabel("RSA modulus size (bits)")
    ax.set_ylabel("Keygen time (seconds)")
    ax.set_title("MRPB Key Generation Scaling")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls="--")

    out = Path("figures/fig4_keygen_scaling.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved: {out} + .png")
    plt.close(fig)


def fig_overhead_ratio(summary, labels, bit_vals):
    """MRPB/RPB overhead ratio for encryption and decryption."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Encryption overhead
    for variant, color, marker in [("shared", "#8c510a", "o"),
                                    ("distinct", "#d8b365", "s")]:
        rpb_vals = extract_metric(summary, labels, f"rpb_{variant}",
                                  "enc_time_ms_mean")
        mrpb_vals = extract_metric(summary, labels, f"mrpb_{variant}",
                                   "enc_time_ms_mean")
        ratio = [m / r if r > 0 else 0 for m, r in zip(mrpb_vals, rpb_vals)]
        ax1.plot(bit_vals, ratio, f"{marker}-", color=color,
                 label=f"{variant}-k", zorder=3)

    ax1.axhline(y=1, color="gray", ls=":", alpha=0.7)
    ax1.set_xlabel("Input size (bits)")
    ax1.set_ylabel("Overhead ratio (MRPB / RPB)")
    ax1.set_title("Encryption Overhead")
    ax1.set_xscale("log")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, ls="--")

    # Decryption overhead
    for variant, color, marker in [("shared", "#8c510a", "o"),
                                    ("distinct", "#d8b365", "s")]:
        rpb_vals = extract_metric(summary, labels, f"rpb_{variant}",
                                  "dec_time_ms_mean")
        mrpb_vals = extract_metric(summary, labels, f"mrpb_{variant}",
                                   "dec_time_ms_mean")
        ratio = [m / r if r > 0 else 0 for m, r in zip(mrpb_vals, rpb_vals)]
        ax2.plot(bit_vals, ratio, f"{marker}-", color=color,
                 label=f"{variant}-k", zorder=3)

    ax2.axhline(y=1, color="gray", ls=":", alpha=0.7)
    ax2.set_xlabel("Input size (bits)")
    ax2.set_ylabel("Overhead ratio (MRPB / RPB)")
    ax2.set_title("Decryption Overhead")
    ax2.set_xscale("log")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, ls="--")

    fig.suptitle("Cost of IND-CPA Security (MRPB vs RPB)", fontsize=14, y=1.02)
    fig.tight_layout()

    out = Path("figures/fig5_enc_overhead.pdf")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    print(f"  Saved: {out} + .png")
    plt.close(fig)


def main():
    print("Loading EXP-001 summary...")
    summary = load_summary()
    labels, bit_vals = get_bit_labels(summary)
    print(f"  Bit-lengths: {bit_vals}")

    print("\nGenerating figures...")

    fig_line_plot(summary, labels, bit_vals,
                  "enc_time_ms_mean", "Encryption time (ms)",
                  "Encryption Time: RPB vs MRPB",
                  "fig1_enc_time.pdf")

    fig_line_plot(summary, labels, bit_vals,
                  "dec_time_ms_mean", "Decryption time (ms)",
                  "Decryption Time: RPB vs MRPB",
                  "fig2_dec_time.pdf")

    fig_ct_size(summary, labels, bit_vals)

    fig_keygen(summary, labels)

    fig_overhead_ratio(summary, labels, bit_vals)

    print("\nDone. All figures in figures/")


if __name__ == "__main__":
    main()
