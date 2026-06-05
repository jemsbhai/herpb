#!/usr/bin/env python3
"""
EXP-008: Scheme Comparison with Secure Two-Server Protocol
==========================================================

Apples-to-apples comparison at dim=384:
  - Plaintext (baseline)
  - Two-server query-private (ours, no Beaver triples)
  - Two-server mutual-privacy (ours, Beaver triples)
  - CKKS (TenSEAL, both-encrypted)
  - Paillier (query-private)

Usage:
  python scripts/run_secure_comparison.py
"""

import csv
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import gmpy2
from gmpy2 import mpz

from herpb.utils.encoding import encode_vector, dot_product_error_bound
from herpb.rag.two_server import (
    generate_modulus,
    build_query_private_corpus,
    build_mutual_privacy_corpus,
    retrieve_query_private,
    retrieve_mutual_privacy,
    retrieve_plaintext,
)

import tenseal as ts
from phe import paillier

# ── Paths ──
PROJECT = Path(__file__).parent.parent
OUTPUT = PROJECT / "experiments" / "exp_008_secure_comparison"
FIGURES = PROJECT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Config ──
SEED = 2026
DIM = 384
PRECISION = 16
NUM_QUERIES = 10
DOC_COUNTS = [50, 200]
PAILLIER_MAX_DOCS = 50
MUTUAL_PRIVACY_MAX_DOCS = 50  # Beaver triple gen is slow at scale
TOP_KS = [1, 5, 10]


def random_unit_vectors(n, dim, rng):
    vecs = rng.standard_normal((n, dim))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


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
    print(f"  -> {path.name} ({len(rows)} rows)")


# =====================================================================
# Benchmark runners
# =====================================================================

def bench_plaintext(queries, docs, n_docs):
    """Baseline: plaintext dot product retrieval."""
    results = []
    for qi in range(len(queries)):
        q = queries[qi].tolist()
        t0 = time.perf_counter()
        scores = [true_dot(q, d.tolist()) for d in docs[:n_docs]]
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        results.append({"ranked": ranked, "scores": scores, "time_ms": elapsed * 1000})
    return results


def bench_query_private(queries, docs, n_docs, n_mod):
    """Our scheme: two-server query-private."""
    corpus = build_query_private_corpus(
        [d.tolist() for d in docs[:n_docs]], precision_bits=PRECISION, n=n_mod
    )
    results = []
    for qi in range(len(queries)):
        q = queries[qi].tolist()
        t0 = time.perf_counter()
        res = retrieve_query_private(q, corpus, k=n_docs, seed=qi)
        elapsed = time.perf_counter() - t0
        # Reconstruct full ranking and scores
        full_scores = [0.0] * n_docs
        # retrieve returns top-k, we need full ranking
        # Re-run to get all scores
        results.append({
            "ranked": res.indices,
            "scores": res.scores,
            "time_ms": elapsed * 1000,
        })
    return results


def bench_mutual_privacy(queries, docs, n_docs, n_mod):
    """Our scheme: two-server mutual privacy with Beaver triples."""
    print(f"    Generating Beaver triples ({n_docs} docs x {DIM} dims)...", end="", flush=True)
    t_setup = time.perf_counter()
    corpus = build_mutual_privacy_corpus(
        [d.tolist() for d in docs[:n_docs]], precision_bits=PRECISION, n=n_mod, seed=SEED
    )
    setup_time = time.perf_counter() - t_setup
    print(f" {setup_time:.1f}s")

    results = []
    for qi in range(len(queries)):
        q = queries[qi].tolist()
        t0 = time.perf_counter()
        res = retrieve_mutual_privacy(q, corpus, k=n_docs, seed=qi + 1000)
        elapsed = time.perf_counter() - t0
        results.append({
            "ranked": res.indices,
            "scores": res.scores,
            "time_ms": elapsed * 1000,
        })
    return results, setup_time


def bench_ckks(queries, docs, n_docs):
    """CKKS via TenSEAL."""
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40

    results = []
    for qi in range(len(queries)):
        q = queries[qi].tolist()
        t0 = time.perf_counter()
        enc_q = ts.ckks_vector(ctx, q)
        scores = []
        for di in range(n_docs):
            enc_d = ts.ckks_vector(ctx, docs[di].tolist())
            score = enc_q.dot(enc_d).decrypt()[0]
            scores.append(score)
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        results.append({"ranked": ranked, "scores": scores, "time_ms": elapsed * 1000})
    return results


def bench_paillier(queries, docs, n_docs):
    """Paillier: query-private, additive HE."""
    pub, priv = paillier.generate_paillier_keypair(n_length=2048)
    results = []
    for qi in range(len(queries)):
        q = queries[qi].tolist()
        t0 = time.perf_counter()
        enc_q = [pub.encrypt(float(qi_val)) for qi_val in q]
        scores = []
        for di in range(n_docs):
            d = docs[di].tolist()
            enc_dot = enc_q[0] * d[0]
            for i in range(1, len(q)):
                enc_dot = enc_dot + enc_q[i] * d[i]
            scores.append(priv.decrypt(enc_dot))
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        results.append({"ranked": ranked, "scores": scores, "time_ms": elapsed * 1000})
    return results


# =====================================================================
# Main
# =====================================================================

def run():
    print("EXP-008: Secure Scheme Comparison")
    print(f"  dim={DIM}, precision={PRECISION}, queries={NUM_QUERIES}")
    print("=" * 70)

    rng = np.random.default_rng(SEED)
    max_docs = max(DOC_COUNTS)
    all_docs = random_unit_vectors(max_docs, DIM, rng)
    queries = random_unit_vectors(NUM_QUERIES, DIM, rng)
    n_mod = generate_modulus(2048, seed=SEED)

    # Precompute ground truth rankings
    ground_truth = {}
    for n_docs in DOC_COUNTS:
        for qi in range(NUM_QUERIES):
            q = queries[qi].tolist()
            scores = [true_dot(q, d.tolist()) for d in all_docs[:n_docs]]
            ground_truth[(n_docs, qi)] = sorted(range(n_docs), key=lambda i: -scores[i])

    all_rows = []

    for n_docs in DOC_COUNTS:
        print(f"\n{'=' * 70}")
        print(f"N = {n_docs} documents")
        print(f"{'=' * 70}")

        schemes = [
            ("plaintext", "none", "none"),
            ("two_server_qp", "info-theoretic", "query_private"),
            ("ckks_8192", "IND-CPA (RLWE)", "both_encrypted"),
        ]
        if n_docs <= PAILLIER_MAX_DOCS:
            schemes.append(("paillier", "semantic (DCRA)", "query_private"))
        if n_docs <= MUTUAL_PRIVACY_MAX_DOCS:
            schemes.insert(2, ("two_server_mp", "info-theoretic", "mutual_privacy"))

        for scheme_name, security, model in schemes:
            print(f"\n  [{scheme_name}] ({security}, {model})")

            # Run benchmark
            setup_time = 0
            if scheme_name == "plaintext":
                results = bench_plaintext(queries, all_docs, n_docs)
            elif scheme_name == "two_server_qp":
                results = bench_query_private(queries, all_docs, n_docs, n_mod)
            elif scheme_name == "two_server_mp":
                results, setup_time = bench_mutual_privacy(queries, all_docs, n_docs, n_mod)
            elif scheme_name == "ckks_8192":
                results = bench_ckks(queries, all_docs, n_docs)
            elif scheme_name == "paillier":
                results = bench_paillier(queries, all_docs, n_docs)

            # Compute metrics
            for qi, res in enumerate(results):
                true_ranking = ground_truth[(n_docs, qi)]
                pred_ranking = res["ranked"]

                for top_k in TOP_KS:
                    if top_k <= n_docs:
                        r = recall_at_k(true_ranking, pred_ranking, top_k)
                        all_rows.append({
                            "scheme": scheme_name,
                            "security": security,
                            "model": model,
                            "n_docs": n_docs,
                            "dimension": DIM,
                            "query_idx": qi,
                            "top_k": top_k,
                            "recall_at_k": r,
                            "query_ms": res["time_ms"],
                            "setup_time_s": setup_time,
                        })

            # Per-scheme summary
            times = [r["time_ms"] for r in results]
            r1_vals = [row["recall_at_k"] for row in all_rows
                       if row["scheme"] == scheme_name and row["n_docs"] == n_docs
                       and row["top_k"] == 1]
            mean_t = np.mean(times)
            mean_r1 = np.mean(r1_vals) if r1_vals else 0
            print(f"    Mean query: {mean_t:.0f}ms | R@1: {mean_r1:.3f} | "
                  f"Throughput: {1000/mean_t:.2f} q/s")

    save_csv(OUTPUT / "comparison_results.csv", all_rows)

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("PAPER TABLE")
    print(f"{'=' * 70}")
    print(f"{'Scheme':<18} {'Security':<18} {'Model':<16} {'N':>5} "
          f"{'ms':>8} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'q/s':>8}")
    print("-" * 100)

    for n_docs in DOC_COUNTS:
        for scheme in ["plaintext", "two_server_qp", "two_server_mp",
                        "ckks_8192", "paillier"]:
            s_rows = [r for r in all_rows if r["scheme"] == scheme
                      and r["n_docs"] == n_docs]
            if not s_rows:
                continue

            times = [r["query_ms"] for r in s_rows if r["top_k"] == 1]
            r1 = np.mean([r["recall_at_k"] for r in s_rows if r["top_k"] == 1])
            r5 = np.mean([r["recall_at_k"] for r in s_rows if r["top_k"] == 5])
            r10 = np.mean([r["recall_at_k"] for r in s_rows if r["top_k"] == 10])
            mean_t = np.mean(times)
            qps = 1000 / mean_t if mean_t > 0 else float("inf")
            security = s_rows[0]["security"]
            model = s_rows[0]["model"]

            print(f"{scheme:<18} {security:<18} {model:<16} {n_docs:>5} "
                  f"{mean_t:>8.0f} {r1:>6.3f} {r5:>6.3f} {r10:>6.3f} {qps:>8.2f}")
        print()

    # ── Figures ──
    generate_figures(all_rows)
    print(f"\nResults saved to {OUTPUT}")


def generate_figures(all_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {
        "plaintext": "#95a5a6",
        "two_server_qp": "#2ecc71",
        "two_server_mp": "#27ae60",
        "ckks_8192": "#e74c3c",
        "paillier": "#9b59b6",
    }
    LABELS = {
        "plaintext": "Plaintext",
        "two_server_qp": "Ours (query-private)",
        "two_server_mp": "Ours (mutual-privacy)",
        "ckks_8192": "CKKS-8192",
        "paillier": "Paillier",
    }
    dpi = 300

    # ── Fig 29: Latency comparison ──
    fig, axes = plt.subplots(1, len(DOC_COUNTS), figsize=(7 * len(DOC_COUNTS), 5),
                              sharey=False)
    if len(DOC_COUNTS) == 1:
        axes = [axes]

    for idx, n_docs in enumerate(DOC_COUNTS):
        ax = axes[idx]
        schemes = []
        times = []
        colors = []
        for s in ["plaintext", "two_server_qp", "two_server_mp", "ckks_8192", "paillier"]:
            s_rows = [r for r in all_rows if r["scheme"] == s
                      and r["n_docs"] == n_docs and r["top_k"] == 1]
            if not s_rows:
                continue
            schemes.append(LABELS.get(s, s))
            times.append(np.mean([r["query_ms"] for r in s_rows]))
            colors.append(COLORS.get(s, "gray"))

        bars = ax.bar(range(len(schemes)), times, color=colors, alpha=0.85)
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels(schemes, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Query latency (ms)")
        ax.set_title(f"N = {n_docs} documents")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                    f"{t:.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"End-to-End Query Latency (dim={DIM}, k={PRECISION})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig29_secure_latency.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / "fig29_secure_latency.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig29_secure_latency")

    # ── Fig 30: Recall comparison ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    n_docs = max(DOC_COUNTS)
    for s in ["two_server_qp", "two_server_mp", "ckks_8192", "paillier"]:
        s_rows = [r for r in all_rows if r["scheme"] == s and r["n_docs"] == n_docs]
        if not s_rows:
            # Try smaller n_docs
            s_rows = [r for r in all_rows if r["scheme"] == s]
        if not s_rows:
            continue
        top_ks = sorted(set(r["top_k"] for r in s_rows))
        recalls = [np.mean([r["recall_at_k"] for r in s_rows if r["top_k"] == tk])
                    for tk in top_ks]
        ax.plot(top_ks, recalls, "o-", label=LABELS.get(s, s),
                color=COLORS.get(s, "gray"), linewidth=2, markersize=6)

    ax.set_xlabel("Top-K")
    ax.set_ylabel("Recall@K")
    ax.set_title(f"Ranking Accuracy (dim={DIM})")
    ax.set_ylim(0.9, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig30_secure_recall.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / "fig30_secure_recall.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig30_secure_recall")

    # ── Fig 31: Security vs Performance tradeoff ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    n_docs = min(DOC_COUNTS)  # Use smallest to include all schemes
    security_levels = {
        "plaintext": 0,
        "two_server_qp": 2,
        "two_server_mp": 3,
        "ckks_8192": 3,
        "paillier": 3,
    }
    security_labels = {0: "None", 2: "Query privacy\n(info-theoretic)",
                       3: "Full privacy\n(info-theoretic / computational)"}

    for s in ["plaintext", "two_server_qp", "two_server_mp", "ckks_8192", "paillier"]:
        s_rows = [r for r in all_rows if r["scheme"] == s
                  and r["n_docs"] == n_docs and r["top_k"] == 1]
        if not s_rows:
            continue
        mean_t = np.mean([r["query_ms"] for r in s_rows])
        sec = security_levels.get(s, 1)
        ax.scatter(mean_t, sec, s=200, color=COLORS.get(s, "gray"),
                   zorder=5, edgecolors="black", linewidth=0.5)
        ax.annotate(LABELS.get(s, s), (mean_t, sec),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Query latency (ms)")
    ax.set_xscale("log")
    ax.set_ylabel("Security level")
    ax.set_yticks([0, 2, 3])
    ax.set_yticklabels(["None", "Query privacy", "Full privacy"])
    ax.set_title(f"Security vs Performance (N={n_docs}, dim={DIM})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig31_security_tradeoff.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / "fig31_security_tradeoff.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig31_security_tradeoff")


if __name__ == "__main__":
    t_start = time.perf_counter()
    run()
    total = time.perf_counter() - t_start
    print(f"\nTotal time: {total:.0f}s ({total/60:.1f}m)")
