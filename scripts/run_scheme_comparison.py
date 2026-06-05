#!/usr/bin/env python3
"""
EXP-005: Scheme Comparison — RPB vs MRPB vs CKKS vs Paillier
=============================================================

Local comparison (5-10 min on 4090/64GB).  Produces the paper's
comparison table and figures.

Usage:
  python scripts/run_scheme_comparison.py
"""

import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# ── Project imports ──
from gmpy2 import mpz
import gmpy2

from herpb.schemes import rpb, mrpb
from herpb.schemes.rpb import Rational
from herpb.utils.encoding import encode_vector, dot_product_error_bound
from herpb.rag.similarity import (
    rpb_encrypt_vector, rpb_dot_product,
    mrpb_encrypt_vector, mrpb_dot_product,
)

# ── Third-party HE ──
import tenseal as ts
from phe import paillier

# ── Paths ──
PROJECT = Path(__file__).parent.parent
OUTPUT = PROJECT / "experiments" / "exp_005_scheme_comparison"
FIGURES = PROJECT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Config ──
SEED = 2026
DIMS = [8, 32, 128]
NUM_DOCS = 50
NUM_QUERIES = 20
PRECISION_BITS = 16
TOP_KS = [1, 5, 10]
MRPB_KEY_BITS = 2048
PAILLIER_KEY_BITS = 2048

# CKKS configs to test
CKKS_CONFIGS = {
    "ckks_4096": {"poly": 4096, "bits": [40, 20, 40], "scale": 20},
    "ckks_8192": {"poly": 8192, "bits": [60, 40, 40, 60], "scale": 40},
}


# =====================================================================
# Helpers
# =====================================================================

def random_unit_vectors(n, dim, rng):
    vecs = rng.standard_normal((n, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def true_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def recall_at_k(true_rank, pred_rank, k):
    return len(set(true_rank[:k]) & set(pred_rank[:k])) / k


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


# =====================================================================
# Dot-product implementations for each scheme
# =====================================================================

def dot_rpb(q_f, d_f, k, c):
    """Both-encrypted RPB dot product."""
    q_rats = encode_vector(q_f, k)
    d_rats = encode_vector(d_f, k)

    t0 = time.perf_counter()
    q_cts = rpb_encrypt_vector(q_rats, c)
    d_cts = rpb_encrypt_vector(d_rats, c)
    dot_ct = rpb_dot_product(q_cts, d_cts)
    num, den = rpb.decrypt(dot_ct)
    elapsed = time.perf_counter() - t0

    result = float(num) / float(den)
    ct_bytes = dot_ct.byte_size()
    return result, elapsed, ct_bytes


def dot_mrpb(q_f, d_f, k, sk):
    """Both-encrypted MRPB dot product."""
    q_rats = encode_vector(q_f, k)
    d_rats = encode_vector(d_f, k)

    t0 = time.perf_counter()
    q_cts = mrpb_encrypt_vector(q_rats, sk)
    d_cts = mrpb_encrypt_vector(d_rats, sk)
    dot_ct = mrpb_dot_product(q_cts, d_cts)
    num, den = mrpb.decrypt(dot_ct, sk)
    elapsed = time.perf_counter() - t0

    result = float(num) / float(den)
    ct_bytes = dot_ct.byte_size()
    return result, elapsed, ct_bytes


def dot_ckks(q_f, d_f, context):
    """Both-encrypted CKKS dot product."""
    t0 = time.perf_counter()
    enc_q = ts.ckks_vector(context, q_f)
    enc_d = ts.ckks_vector(context, d_f)
    enc_result = enc_q.dot(enc_d)
    result = enc_result.decrypt()[0]
    elapsed = time.perf_counter() - t0

    ct_bytes = len(enc_result.serialize())
    return result, elapsed, ct_bytes


def dot_paillier(q_f, d_f, pub_key, priv_key):
    """Query-private Paillier dot product.

    Only query is encrypted; document is plaintext on server.
    This is the ONLY model Paillier supports (additive HE).
    """
    t0 = time.perf_counter()

    # Client encrypts query
    enc_q = [pub_key.encrypt(float(qi)) for qi in q_f]

    # Server: scalar-multiply and sum (homomorphic)
    enc_dot = enc_q[0] * float(d_f[0])
    for i in range(1, len(q_f)):
        enc_dot = enc_dot + enc_q[i] * float(d_f[i])

    # Client decrypts
    result = priv_key.decrypt(enc_dot)
    elapsed = time.perf_counter() - t0

    # Ciphertext size: single encrypted result
    # phe doesn't expose serialization easily; estimate from key size
    ct_bytes = PAILLIER_KEY_BITS // 4  # ~512 bytes for 2048-bit key

    return result, elapsed, ct_bytes


# =====================================================================
# Main comparison
# =====================================================================

def run_comparison():
    print("EXP-005: Scheme Comparison")
    print("=" * 60)

    rng = np.random.default_rng(SEED)
    py_rng = random.Random(SEED)

    # ── Key generation ──
    print("\nKey generation...")

    t0 = time.perf_counter()
    rpb_c = mpz(gmpy2.mpz_urandomb(gmpy2.random_state(42), 256))
    rpb_keygen_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, mrpb_sk = mrpb.keygen(key_bits=MRPB_KEY_BITS, precision=0, seed=SEED)
    mrpb_keygen_s = time.perf_counter() - t0

    ckks_contexts = {}
    ckks_keygen_s = {}
    for name, cfg in CKKS_CONFIGS.items():
        t0 = time.perf_counter()
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=cfg["poly"],
            coeff_mod_bit_sizes=cfg["bits"],
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** cfg["scale"]
        ckks_contexts[name] = ctx
        ckks_keygen_s[name] = time.perf_counter() - t0

    t0 = time.perf_counter()
    paillier_pub, paillier_priv = paillier.generate_paillier_keypair(n_length=PAILLIER_KEY_BITS)
    paillier_keygen_s = time.perf_counter() - t0

    print(f"  RPB blinding:    {rpb_keygen_s*1000:.1f} ms")
    print(f"  MRPB (2048-bit): {mrpb_keygen_s*1000:.1f} ms")
    for name, t in ckks_keygen_s.items():
        print(f"  {name}:       {t*1000:.1f} ms")
    print(f"  Paillier:        {paillier_keygen_s*1000:.1f} ms")

    # ── Per-dimension comparison ──
    all_rows = []
    ranking_rows = []

    for dim in DIMS:
        print(f"\n{'─' * 60}")
        print(f"Dimension = {dim}")
        print(f"{'─' * 60}")

        docs_f = random_unit_vectors(NUM_DOCS, dim, rng)
        queries_f = random_unit_vectors(NUM_QUERIES, dim, rng)

        for qi in range(NUM_QUERIES):
            q_f = queries_f[qi].tolist()

            # Ground truth
            true_scores = [true_dot(q_f, d.tolist()) for d in docs_f]
            true_ranking = sorted(range(NUM_DOCS), key=lambda i: -true_scores[i])

            scheme_results = {}  # scheme_name -> list of scores

            for di in range(NUM_DOCS):
                d_f = docs_f[di].tolist()
                true_val = true_scores[di]

                # ── RPB ──
                rpb_val, rpb_t, rpb_bytes = dot_rpb(q_f, d_f, PRECISION_BITS, rpb_c)
                all_rows.append({
                    "scheme": "rpb", "security": "operand-hiding",
                    "model": "both_encrypted",
                    "dimension": dim, "query": qi, "doc": di,
                    "true_dot": true_val, "result": rpb_val,
                    "abs_error": abs(true_val - rpb_val),
                    "time_s": rpb_t, "ciphertext_bytes": rpb_bytes,
                })
                scheme_results.setdefault("rpb", []).append(rpb_val)

                # ── MRPB ──
                mrpb_val, mrpb_t, mrpb_bytes = dot_mrpb(q_f, d_f, PRECISION_BITS, mrpb_sk)
                all_rows.append({
                    "scheme": "mrpb", "security": "IND-CPA",
                    "model": "both_encrypted",
                    "dimension": dim, "query": qi, "doc": di,
                    "true_dot": true_val, "result": mrpb_val,
                    "abs_error": abs(true_val - mrpb_val),
                    "time_s": mrpb_t, "ciphertext_bytes": mrpb_bytes,
                })
                scheme_results.setdefault("mrpb", []).append(mrpb_val)

                # ── CKKS variants ──
                for ckks_name, ctx in ckks_contexts.items():
                    if dim > CKKS_CONFIGS[ckks_name]["poly"] // 2:
                        continue
                    try:
                        ckks_val, ckks_t, ckks_bytes = dot_ckks(q_f, d_f, ctx)
                        all_rows.append({
                            "scheme": ckks_name, "security": "IND-CPA",
                            "model": "both_encrypted",
                            "dimension": dim, "query": qi, "doc": di,
                            "true_dot": true_val, "result": ckks_val,
                            "abs_error": abs(true_val - ckks_val),
                            "time_s": ckks_t, "ciphertext_bytes": ckks_bytes,
                        })
                        scheme_results.setdefault(ckks_name, []).append(ckks_val)
                    except Exception as e:
                        if qi == 0 and di == 0:
                            print(f"  [WARN] {ckks_name} dim={dim}: {e}")

                # ── Paillier ──
                pail_val, pail_t, pail_bytes = dot_paillier(q_f, d_f, paillier_pub, paillier_priv)
                all_rows.append({
                    "scheme": "paillier", "security": "semantic",
                    "model": "query_private",
                    "dimension": dim, "query": qi, "doc": di,
                    "true_dot": true_val, "result": pail_val,
                    "abs_error": abs(true_val - pail_val),
                    "time_s": pail_t, "ciphertext_bytes": pail_bytes,
                })
                scheme_results.setdefault("paillier", []).append(pail_val)

            # ── Ranking comparison ──
            for scheme_name, scores in scheme_results.items():
                pred_ranking = sorted(range(len(scores)), key=lambda i: -scores[i])
                for k in TOP_KS:
                    if k <= NUM_DOCS:
                        r = recall_at_k(true_ranking, pred_ranking, k)
                        ranking_rows.append({
                            "scheme": scheme_name, "dimension": dim,
                            "query": qi, "top_k": k, "recall_at_k": r,
                        })

            if (qi + 1) % 5 == 0:
                print(f"  Queries completed: {qi + 1}/{NUM_QUERIES}")

    # ── Save raw results ──
    save_csv(OUTPUT / "dot_product_results.csv", all_rows)
    save_csv(OUTPUT / "ranking_results.csv", ranking_rows)

    # ── Aggregate summary ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    schemes = sorted(set(r["scheme"] for r in all_rows))
    summary_rows = []

    for scheme in schemes:
        s_rows = [r for r in all_rows if r["scheme"] == scheme]
        r_rows = [r for r in ranking_rows if r["scheme"] == scheme]

        errors = [r["abs_error"] for r in s_rows]
        times = [r["time_s"] for r in s_rows]
        ct_sizes = [r["ciphertext_bytes"] for r in s_rows]

        recall_1 = [r["recall_at_k"] for r in r_rows if r["top_k"] == 1]
        recall_5 = [r["recall_at_k"] for r in r_rows if r["top_k"] == 5]
        recall_10 = [r["recall_at_k"] for r in r_rows if r["top_k"] == 10]

        row = {
            "scheme": scheme,
            "security": s_rows[0]["security"],
            "model": s_rows[0]["model"],
            "max_error": max(errors),
            "mean_error": np.mean(errors),
            "median_error": np.median(errors),
            "mean_time_ms": np.mean(times) * 1000,
            "median_time_ms": np.median(times) * 1000,
            "mean_ct_bytes": np.mean(ct_sizes),
            "recall_at_1": np.mean(recall_1) if recall_1 else None,
            "recall_at_5": np.mean(recall_5) if recall_5 else None,
            "recall_at_10": np.mean(recall_10) if recall_10 else None,
            "total_trials": len(s_rows),
        }
        summary_rows.append(row)

        r1 = f"{row['recall_at_1']:.3f}" if row["recall_at_1"] is not None else "N/A"
        r5 = f"{row['recall_at_5']:.3f}" if row["recall_at_5"] is not None else "N/A"
        r10 = f"{row['recall_at_10']:.3f}" if row["recall_at_10"] is not None else "N/A"
        print(f"\n  {scheme} ({row['security']}, {row['model']}):")
        print(f"    Max error:    {row['max_error']:.2e}")
        print(f"    Mean error:   {row['mean_error']:.2e}")
        print(f"    Mean time:    {row['mean_time_ms']:.2f} ms")
        print(f"    CT size:      {row['mean_ct_bytes']:.0f} bytes")
        print(f"    Recall@1/5/10: {r1} / {r5} / {r10}")

    save_csv(OUTPUT / "summary.csv", summary_rows)

    with open(OUTPUT / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)

    # ── Figures ──
    generate_comparison_figures(all_rows, ranking_rows, summary_rows)

    print(f"\nResults saved to {OUTPUT}")


# =====================================================================
# Figure generation
# =====================================================================

def generate_comparison_figures(all_rows, ranking_rows, summary_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 300

    # ── Color scheme ──
    COLORS = {
        "rpb": "#2ecc71",
        "mrpb": "#3498db",
        "ckks_4096": "#e74c3c",
        "ckks_8192": "#c0392b",
        "paillier": "#9b59b6",
    }

    def get_color(s):
        return COLORS.get(s, "#95a5a6")

    schemes = sorted(set(r["scheme"] for r in summary_rows))

    # ── Fig 18: Error comparison by scheme ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, dim in enumerate(DIMS):
        ax = axes[idx]
        dim_schemes = []
        max_errs = []
        mean_errs = []
        colors = []

        for scheme in schemes:
            s_rows = [r for r in all_rows
                      if r["scheme"] == scheme and r["dimension"] == dim]
            if not s_rows:
                continue
            dim_schemes.append(scheme)
            max_errs.append(max(r["abs_error"] for r in s_rows))
            mean_errs.append(np.mean([r["abs_error"] for r in s_rows]))
            colors.append(get_color(scheme))

        x = np.arange(len(dim_schemes))
        ax.bar(x - 0.15, max_errs, 0.3, label="Max", color=colors, alpha=0.8)
        ax.bar(x + 0.15, mean_errs, 0.3, label="Mean", color=colors, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_schemes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Dot-product error")
        ax.set_title(f"dim={dim}")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Dot-Product Error by Scheme (k=16)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig18_error_comparison.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / "fig18_error_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig18_error_comparison")

    # ── Fig 19: Recall@K comparison ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, dim in enumerate(DIMS):
        ax = axes[idx]
        for scheme in schemes:
            s_rows = [r for r in ranking_rows
                      if r["scheme"] == scheme and r["dimension"] == dim]
            if not s_rows:
                continue
            top_ks = sorted(set(r["top_k"] for r in s_rows))
            recalls = [np.mean([r["recall_at_k"] for r in s_rows if r["top_k"] == tk])
                       for tk in top_ks]
            ax.plot(top_ks, recalls, "o-", label=scheme, color=get_color(scheme),
                    linewidth=2, markersize=6)

        ax.set_xlabel("Top-K")
        ax.set_title(f"dim={dim}")
        ax.set_ylim(0.75, 1.02)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Recall@K")
            ax.legend(fontsize=8)

    fig.suptitle("Ranking Preservation: Recall@K by Scheme", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig19_recall_comparison.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / "fig19_recall_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig19_recall_comparison")

    # ── Fig 20: Timing comparison ──
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(DIMS))
    width = 0.15
    offsets = np.linspace(-width * (len(schemes) - 1) / 2,
                           width * (len(schemes) - 1) / 2, len(schemes))

    for i, scheme in enumerate(schemes):
        times = []
        for dim in DIMS:
            s_rows = [r for r in all_rows
                      if r["scheme"] == scheme and r["dimension"] == dim]
            if s_rows:
                times.append(np.median([r["time_s"] for r in s_rows]) * 1000)
            else:
                times.append(0)
        ax.bar(x + offsets[i], times, width, label=scheme,
               color=get_color(scheme), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"dim={d}" for d in DIMS])
    ax.set_ylabel("Median time per dot product (ms)")
    ax.set_title("Performance: Dot-Product Latency by Scheme (k=16)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig20_timing_comparison.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / "fig20_timing_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig20_timing_comparison")

    # ── Fig 21: Paper-ready comparison table as figure ──
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")

    col_labels = ["Scheme", "Security", "Model", "Max Error",
                  "Recall@1", "Recall@5", "Recall@10",
                  "Median ms", "CT bytes"]
    table_data = []
    for s in summary_rows:
        table_data.append([
            s["scheme"],
            s["security"],
            s["model"],
            f"{s['max_error']:.2e}",
            f"{s['recall_at_1']:.3f}" if s["recall_at_1"] is not None else "—",
            f"{s['recall_at_5']:.3f}" if s["recall_at_5"] is not None else "—",
            f"{s['recall_at_10']:.3f}" if s["recall_at_10"] is not None else "—",
            f"{s['median_time_ms']:.1f}",
            f"{s['mean_ct_bytes']:.0f}",
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by scheme
    for i, s in enumerate(summary_rows):
        color = get_color(s["scheme"])
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color + "22")

    fig.suptitle("Scheme Comparison Summary (all dimensions pooled)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig21_comparison_table.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig21_comparison_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig21_comparison_table")


if __name__ == "__main__":
    t_start = time.perf_counter()
    run_comparison()
    total = time.perf_counter() - t_start
    print(f"\nTotal time: {total:.1f}s ({total/60:.1f}m)")
