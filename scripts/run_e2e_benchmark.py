#!/usr/bin/env python3
"""
EXP-006: Apples-to-Apples End-to-End Retrieval Benchmark
=========================================================

Measures what a system deployer cares about:
  "Given a float query and a corpus, how long to get ranked results?"

ALL schemes measured on the SAME:
  - Hardware (CPU; GPU results supplementary)
  - Task (top-k retrieval by dot product)
  - Dimensions (384 = real sentence-transformer)
  - Security level (2048-bit RSA / 128-bit CKKS)
  - Input vectors (identical random unit vectors)

Per-query latency is broken into:
  OFFLINE (amortized): keygen + encode + encrypt all documents
  ONLINE  (per query): encode query + encrypt query + N dot products
                       + N decrypts + sort/rank

Storage cost: total ciphertext bytes for the corpus.

Usage:
  python scripts/run_e2e_benchmark.py
"""

import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
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
OUTPUT = PROJECT / "experiments" / "exp_006_e2e_benchmark"
FIGURES = PROJECT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Parameters ──
SEED = 2026
DIM = 384            # real sentence-transformer dimension
PRECISION = 16       # encoding bits
KEY_BITS = 2048      # MRPB / Paillier
NUM_QUERIES = 10
DOC_COUNTS = [50, 200]
PAILLIER_MAX_DOCS = 50  # Paillier is too slow for more
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
    print(f"  Saved {len(rows)} rows -> {path}")


# =====================================================================
# Scheme wrappers — each returns (ranked_indices, online_time_s, details)
# =====================================================================

class PlaintextScheme:
    name = "plaintext"
    security = "none"

    def setup(self, docs_f, dim):
        self.docs_f = docs_f
        self.offline_time = 0.0
        self.storage_bytes = len(docs_f) * dim * 4  # float32

    def query(self, q_f, k):
        t0 = time.perf_counter()
        scores = [true_dot(q_f, d.tolist()) for d in self.docs_f]
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        return ranked[:k], scores, elapsed, {}


class RPBScheme:
    name = "rpb"
    security = "operand-hiding"

    def setup(self, docs_f, dim):
        self.dim = dim
        rs = gmpy2.random_state(42)
        self.c = gmpy2.mpz_urandomb(rs, 256) | mpz(1)

        t0 = time.perf_counter()
        self.docs_enc = []
        self.docs_rats = []
        total_bytes = 0
        for d in docs_f:
            rats = encode_vector(d.tolist(), PRECISION)
            cts = rpb_encrypt_vector(rats, self.c)
            self.docs_rats.append(rats)
            self.docs_enc.append(cts)
            total_bytes += sum(ct.byte_size() for ct in cts)
        self.offline_time = time.perf_counter() - t0
        self.storage_bytes = total_bytes

    def query(self, q_f, k):
        t0 = time.perf_counter()

        # Encode + encrypt query
        t_enc_start = time.perf_counter()
        q_rats = encode_vector(q_f, PRECISION)
        q_cts = rpb_encrypt_vector(q_rats, self.c)
        t_enc = time.perf_counter() - t_enc_start

        # Dot products + decrypt
        t_dot_start = time.perf_counter()
        scores = []
        for d_cts in self.docs_enc:
            dot_ct = rpb_dot_product(q_cts, d_cts)
            num, den = rpb.decrypt(dot_ct)
            scores.append(float(num) / float(den))
        t_dot = time.perf_counter() - t_dot_start

        # Rank
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0

        details = {"encrypt_query_ms": t_enc * 1000, "dot_products_ms": t_dot * 1000}
        return ranked[:k], scores, elapsed, details


class MRPBScheme:
    name = "mrpb"
    security = "IND-CPA"

    def setup(self, docs_f, dim):
        self.dim = dim
        t0 = time.perf_counter()
        _, self.sk = mrpb.keygen(key_bits=KEY_BITS, precision=0, seed=SEED)
        t_keygen = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.docs_enc = []
        total_bytes = 0
        for d in docs_f:
            rats = encode_vector(d.tolist(), PRECISION)
            cts = mrpb_encrypt_vector(rats, self.sk)
            self.docs_enc.append(cts)
            total_bytes += sum(ct.byte_size() for ct in cts)
        t_enc_docs = time.perf_counter() - t0

        self.offline_time = t_keygen + t_enc_docs
        self.storage_bytes = total_bytes

    def query(self, q_f, k):
        t0 = time.perf_counter()

        t_enc_start = time.perf_counter()
        q_rats = encode_vector(q_f, PRECISION)
        q_cts = mrpb_encrypt_vector(q_rats, self.sk)
        t_enc = time.perf_counter() - t_enc_start

        t_dot_start = time.perf_counter()
        scores = []
        for d_cts in self.docs_enc:
            dot_ct = mrpb_dot_product(q_cts, d_cts)
            num, den = mrpb.decrypt(dot_ct, self.sk)
            scores.append(float(num) / float(den))
        t_dot = time.perf_counter() - t_dot_start

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0

        details = {"encrypt_query_ms": t_enc * 1000, "dot_products_ms": t_dot * 1000}
        return ranked[:k], scores, elapsed, details


class CKKSScheme:
    def __init__(self, poly_deg=8192, coeff_bits=None, scale_bits=40):
        if coeff_bits is None:
            coeff_bits = [60, 40, 40, 60]
        self.poly_deg = poly_deg
        self.coeff_bits = coeff_bits
        self.scale_bits = scale_bits
        self.name = f"ckks_{poly_deg}"
        self.security = "IND-CPA"

    def setup(self, docs_f, dim):
        t0 = time.perf_counter()
        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=self.poly_deg,
            coeff_mod_bit_sizes=self.coeff_bits,
        )
        self.ctx.generate_galois_keys()
        self.ctx.global_scale = 2 ** self.scale_bits
        t_keygen = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.docs_enc = []
        total_bytes = 0
        for d in docs_f:
            enc_d = ts.ckks_vector(self.ctx, d.tolist())
            self.docs_enc.append(enc_d)
            total_bytes += len(enc_d.serialize())
        t_enc_docs = time.perf_counter() - t0

        self.offline_time = t_keygen + t_enc_docs
        self.storage_bytes = total_bytes

    def query(self, q_f, k):
        t0 = time.perf_counter()

        t_enc_start = time.perf_counter()
        enc_q = ts.ckks_vector(self.ctx, q_f)
        t_enc = time.perf_counter() - t_enc_start

        t_dot_start = time.perf_counter()
        scores = []
        for enc_d in self.docs_enc:
            result = enc_q.dot(enc_d)
            scores.append(result.decrypt()[0])
        t_dot = time.perf_counter() - t_dot_start

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0

        details = {"encrypt_query_ms": t_enc * 1000, "dot_products_ms": t_dot * 1000}
        return ranked[:k], scores, elapsed, details


class PaillierScheme:
    name = "paillier"
    security = "semantic"

    def setup(self, docs_f, dim):
        t0 = time.perf_counter()
        self.pub, self.priv = paillier.generate_paillier_keypair(n_length=KEY_BITS)
        t_keygen = time.perf_counter() - t0

        # Paillier is query-private: docs stay plaintext on server
        self.docs_f = docs_f
        self.offline_time = t_keygen
        self.storage_bytes = len(docs_f) * dim * 4  # plaintext float32

    def query(self, q_f, k):
        t0 = time.perf_counter()

        # Encrypt query
        t_enc_start = time.perf_counter()
        enc_q = [self.pub.encrypt(float(qi)) for qi in q_f]
        t_enc = time.perf_counter() - t_enc_start

        # Dot products (query-private: server multiplies encrypted q by plaintext d)
        t_dot_start = time.perf_counter()
        scores = []
        for d in self.docs_f:
            d_list = d.tolist()
            enc_dot = enc_q[0] * d_list[0]
            for i in range(1, len(q_f)):
                enc_dot = enc_dot + enc_q[i] * d_list[i]
            scores.append(self.priv.decrypt(enc_dot))
        t_dot = time.perf_counter() - t_dot_start

        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0

        details = {"encrypt_query_ms": t_enc * 1000, "dot_products_ms": t_dot * 1000}
        return ranked[:k], scores, elapsed, details


# =====================================================================
# Main benchmark
# =====================================================================

def run_benchmark():
    print("EXP-006: End-to-End Retrieval Benchmark")
    print(f"  dim={DIM}, precision={PRECISION}, key_bits={KEY_BITS}")
    print(f"  doc_counts={DOC_COUNTS}, queries={NUM_QUERIES}")
    print("=" * 70)

    rng = np.random.default_rng(SEED)

    # Generate maximum corpus
    max_docs = max(DOC_COUNTS)
    all_docs = random_unit_vectors(max_docs, DIM, rng)
    all_queries = random_unit_vectors(NUM_QUERIES, DIM, rng)

    # Compute ground truth rankings for all (query, corpus) pairs
    print("\n  Computing ground truth...")
    ground_truth = {}  # (n_docs, qi) -> ranked indices
    for n_docs in DOC_COUNTS:
        docs = all_docs[:n_docs]
        for qi in range(NUM_QUERIES):
            q = all_queries[qi].tolist()
            scores = [true_dot(q, d.tolist()) for d in docs]
            ground_truth[(n_docs, qi)] = sorted(range(n_docs), key=lambda i: -scores[i])

    # Schemes to test
    schemes = [
        PlaintextScheme(),
        RPBScheme(),
        MRPBScheme(),
        CKKSScheme(poly_deg=8192, coeff_bits=[60, 40, 40, 60], scale_bits=40),
        PaillierScheme(),
    ]

    query_rows = []
    setup_rows = []

    for n_docs in DOC_COUNTS:
        docs = all_docs[:n_docs]
        print(f"\n{'=' * 70}")
        print(f"CORPUS SIZE: {n_docs} documents x {DIM} dimensions")
        print(f"{'=' * 70}")

        for scheme in schemes:
            # Skip Paillier for large corpora
            if scheme.name == "paillier" and n_docs > PAILLIER_MAX_DOCS:
                print(f"\n  [{scheme.name}] SKIPPED (too slow for {n_docs} docs)")
                continue

            print(f"\n  [{scheme.name}] ({scheme.security})")

            # ── Offline setup ──
            print(f"    Setting up...", end="", flush=True)
            scheme.setup(docs, DIM)
            print(f" {scheme.offline_time*1000:.0f}ms, "
                  f"storage={scheme.storage_bytes/1024:.0f}KB")

            setup_rows.append({
                "scheme": scheme.name,
                "security": scheme.security,
                "n_docs": n_docs,
                "dimension": DIM,
                "offline_time_ms": scheme.offline_time * 1000,
                "storage_bytes": scheme.storage_bytes,
                "storage_kb": scheme.storage_bytes / 1024,
                "bytes_per_doc": scheme.storage_bytes / n_docs,
            })

            # ── Online queries ──
            query_times = []
            for qi in range(NUM_QUERIES):
                q_f = all_queries[qi].tolist()
                true_ranking = ground_truth[(n_docs, qi)]

                ranked, scores, elapsed, details = scheme.query(q_f, max(TOP_KS))

                for top_k in TOP_KS:
                    r = recall_at_k(true_ranking, ranked, top_k)
                    query_rows.append({
                        "scheme": scheme.name,
                        "security": scheme.security,
                        "n_docs": n_docs,
                        "dimension": DIM,
                        "query_idx": qi,
                        "top_k": top_k,
                        "recall_at_k": r,
                        "query_time_ms": elapsed * 1000,
                        "encrypt_query_ms": details.get("encrypt_query_ms", 0),
                        "dot_products_ms": details.get("dot_products_ms", 0),
                    })
                query_times.append(elapsed)

                if (qi + 1) % 5 == 0 or qi == 0:
                    print(f"    Query {qi+1}/{NUM_QUERIES}: {elapsed*1000:.0f}ms")

            mean_t = np.mean(query_times) * 1000
            # Get recall@1 for this scheme/n_docs
            r1_rows = [r for r in query_rows if r["scheme"] == scheme.name
                       and r["n_docs"] == n_docs and r["top_k"] == 1]
            mean_r1 = np.mean([r["recall_at_k"] for r in r1_rows]) if r1_rows else 0

            print(f"    Mean query time: {mean_t:.0f}ms | "
                  f"Recall@1: {mean_r1:.3f} | "
                  f"Throughput: {1000/mean_t:.2f} q/s")

    # ── Save results ──
    save_csv(OUTPUT / "query_results.csv", query_rows)
    save_csv(OUTPUT / "setup_results.csv", setup_rows)

    # ── Print summary table ──
    print(f"\n{'=' * 70}")
    print("SUMMARY TABLE (for paper)")
    print(f"{'=' * 70}")
    print(f"{'Scheme':<12} {'Security':<14} {'N':>5} {'Query ms':>10} "
          f"{'R@1':>6} {'R@5':>6} {'R@10':>6} {'Storage':>10} {'q/s':>8}")
    print("-" * 90)

    for n_docs in DOC_COUNTS:
        for scheme in schemes:
            s_rows = [r for r in query_rows if r["scheme"] == scheme.name
                      and r["n_docs"] == n_docs]
            if not s_rows:
                continue

            mean_time = np.mean([r["query_time_ms"] for r in s_rows
                                 if r["top_k"] == TOP_KS[0]])
            r1 = np.mean([r["recall_at_k"] for r in s_rows if r["top_k"] == 1])
            r5 = np.mean([r["recall_at_k"] for r in s_rows if r["top_k"] == 5])
            r10 = np.mean([r["recall_at_k"] for r in s_rows if r["top_k"] == 10])

            su = [r for r in setup_rows if r["scheme"] == scheme.name
                  and r["n_docs"] == n_docs]
            storage = f"{su[0]['storage_kb']:.0f}KB" if su else "?"

            qps = 1000 / mean_time if mean_time > 0 else float("inf")

            print(f"{scheme.name:<12} {scheme.security:<14} {n_docs:>5} "
                  f"{mean_time:>10.1f} {r1:>6.3f} {r5:>6.3f} {r10:>6.3f} "
                  f"{storage:>10} {qps:>8.2f}")
        print()

    # ── Generate figures ──
    generate_figures(query_rows, setup_rows)
    print(f"\nAll results saved to {OUTPUT}")


# =====================================================================
# Figure generation
# =====================================================================

def generate_figures(query_rows, setup_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {
        "plaintext": "#95a5a6",
        "rpb": "#2ecc71",
        "mrpb": "#3498db",
        "ckks_8192": "#e74c3c",
        "paillier": "#9b59b6",
    }
    dpi = 300

    # ── Fig 22: Per-query latency comparison ──
    fig, axes = plt.subplots(1, len(DOC_COUNTS), figsize=(7 * len(DOC_COUNTS), 5),
                              sharey=False)
    if len(DOC_COUNTS) == 1:
        axes = [axes]

    for idx, n_docs in enumerate(DOC_COUNTS):
        ax = axes[idx]
        schemes = sorted(set(r["scheme"] for r in query_rows if r["n_docs"] == n_docs))
        times = []
        colors = []
        for s in schemes:
            s_rows = [r for r in query_rows if r["scheme"] == s
                      and r["n_docs"] == n_docs and r["top_k"] == 1]
            if s_rows:
                times.append(np.mean([r["query_time_ms"] for r in s_rows]))
                colors.append(COLORS.get(s, "#95a5a6"))

        ax.bar(range(len(schemes)), times, color=colors, alpha=0.85)
        ax.set_xticks(range(len(schemes)))
        ax.set_xticklabels(schemes, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Query latency (ms)")
        ax.set_title(f"N = {n_docs} documents")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"End-to-End Query Latency (dim={DIM}, k={PRECISION})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig22_e2e_latency.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / "fig22_e2e_latency.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig22_e2e_latency")

    # ── Fig 23: Recall@K comparison ──
    fig, axes = plt.subplots(1, len(DOC_COUNTS), figsize=(7 * len(DOC_COUNTS), 5),
                              sharey=True)
    if len(DOC_COUNTS) == 1:
        axes = [axes]

    for idx, n_docs in enumerate(DOC_COUNTS):
        ax = axes[idx]
        schemes = sorted(set(r["scheme"] for r in query_rows if r["n_docs"] == n_docs))
        for s in schemes:
            if s == "plaintext":
                continue
            s_rows = [r for r in query_rows if r["scheme"] == s and r["n_docs"] == n_docs]
            if not s_rows:
                continue
            top_ks = sorted(set(r["top_k"] for r in s_rows))
            recalls = [np.mean([r["recall_at_k"] for r in s_rows if r["top_k"] == tk])
                       for tk in top_ks]
            ax.plot(top_ks, recalls, "o-", label=s, color=COLORS.get(s, "#95a5a6"),
                    linewidth=2, markersize=6)

        ax.set_xlabel("Top-K")
        ax.set_title(f"N = {n_docs}")
        ax.set_ylim(0.8, 1.02)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel("Recall@K")
            ax.legend(fontsize=8)

    fig.suptitle(f"Ranking Accuracy (dim={DIM})", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig23_e2e_recall.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / "fig23_e2e_recall.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig23_e2e_recall")

    # ── Fig 24: Storage cost ──
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    n_docs = max(DOC_COUNTS)
    schemes = sorted(set(r["scheme"] for r in setup_rows if r["n_docs"] == n_docs))
    storage = []
    colors = []
    for s in schemes:
        s_rows = [r for r in setup_rows if r["scheme"] == s and r["n_docs"] == n_docs]
        if s_rows:
            storage.append(s_rows[0]["storage_kb"])
            colors.append(COLORS.get(s, "#95a5a6"))

    ax.bar(range(len(schemes)), storage, color=colors, alpha=0.85)
    ax.set_xticks(range(len(schemes)))
    ax.set_xticklabels(schemes, rotation=30, ha="right")
    ax.set_ylabel("Corpus storage (KB)")
    ax.set_title(f"Storage Cost ({n_docs} docs x {DIM} dims)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig24_storage.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(FIGURES / "fig24_storage.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig24_storage")


if __name__ == "__main__":
    t_start = time.perf_counter()
    run_benchmark()
    total = time.perf_counter() - t_start
    print(f"\nTotal benchmark time: {total:.1f}s ({total/60:.1f}m)")
