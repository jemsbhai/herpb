#!/usr/bin/env python3
"""
EXP-009: Scheme Comparison with 64-bit Native Secret Sharing
=============================================================

Apples-to-apples comparison at dim=384:
  - Plaintext (baseline)
  - Two-server query-private (64-bit native)
  - Two-server mutual-privacy (64-bit matrix-vector Beaver)
  - CKKS (TenSEAL)
  - Paillier

Usage:
  python scripts/run_64bit_comparison.py
"""

import math
import random
import time
from pathlib import Path

import numpy as np

from herpb.utils.encoding import encode_vector
from herpb.schemes.secret_sharing_64 import (
    share_vector, reconstruct_scalar,
    query_private_batch,
    share_matrix, setup_matrix_beaver,
    generate_query_triple, mutual_privacy_batch,
)

import tenseal as ts
from phe import paillier

PROJECT = Path(__file__).parent.parent
OUTPUT = PROJECT / "experiments" / "exp_009_64bit_comparison"
FIGURES = PROJECT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 2026
DIM = 384
PRECISION = 16
NUM_QUERIES = 10
DOC_COUNTS = [50, 200, 1000, 5000]
PAILLIER_MAX_DOCS = 50
TOP_KS = [1, 5, 10]


def random_unit_vectors(n, dim, rng):
    vecs = rng.standard_normal((n, dim))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def encode_matrix(vecs, k):
    """Encode float vectors as integer numerator matrix."""
    rows = []
    for v in vecs:
        ints = [n for n, _ in encode_vector(v.tolist(), k)]
        rows.append(ints)
    return np.array(rows, dtype=np.int64)


# =====================================================================
# Benchmark runners
# =====================================================================

def bench_plaintext(queries, docs_f, n_docs):
    results = []
    for qi in range(len(queries)):
        t0 = time.perf_counter()
        scores = (docs_f[:n_docs] @ queries[qi]).tolist()
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        results.append({"ranked": ranked, "time_ms": elapsed * 1000})
    return results


def bench_query_private_64(queries_int, doc_matrix, n_docs):
    D = doc_matrix[:n_docs]
    results = []
    for qi in range(len(queries_int)):
        q_ints = queries_int[qi]
        t0 = time.perf_counter()
        q1, q2 = share_vector(q_ints)
        dots1, dots2 = query_private_batch(q1, q2, D)
        scores = [
            reconstruct_scalar(np.uint64(dots1[j]), np.uint64(dots2[j]))
            for j in range(n_docs)
        ]
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        results.append({"ranked": ranked, "time_ms": elapsed * 1000})
    return results


def bench_mutual_privacy_64(queries_int, doc_matrix, n_docs):
    D = doc_matrix[:n_docs]
    rng = np.random.default_rng(SEED)

    print(f"    Setup (share docs + Beaver)...", end="", flush=True)
    t_setup = time.perf_counter()
    D1, D2 = share_matrix(D, rng)
    setup = setup_matrix_beaver(D1, D2, rng)
    setup_s = time.perf_counter() - t_setup
    print(f" {setup_s:.3f}s")

    results = []
    for qi in range(len(queries_int)):
        q_ints = queries_int[qi]
        t0 = time.perf_counter()
        triple = generate_query_triple(setup, np.random.default_rng(qi))
        q1, q2 = share_vector(q_ints, np.random.default_rng(qi + 1000))
        Z1, Z2 = mutual_privacy_batch(q1, q2, setup, triple)
        scores = [
            reconstruct_scalar(np.uint64(Z1[j]), np.uint64(Z2[j]))
            for j in range(n_docs)
        ]
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        results.append({"ranked": ranked, "time_ms": elapsed * 1000})
    return results, setup_s


def bench_ckks(queries_f, docs_f, n_docs):
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40

    results = []
    for qi in range(len(queries_f)):
        q = queries_f[qi].tolist()
        t0 = time.perf_counter()
        enc_q = ts.ckks_vector(ctx, q)
        scores = []
        for di in range(n_docs):
            enc_d = ts.ckks_vector(ctx, docs_f[di].tolist())
            scores.append(enc_q.dot(enc_d).decrypt()[0])
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        results.append({"ranked": ranked, "time_ms": elapsed * 1000})
    return results


def bench_paillier(queries_f, docs_f, n_docs):
    pub, priv = paillier.generate_paillier_keypair(n_length=2048)
    results = []
    for qi in range(len(queries_f)):
        q = queries_f[qi].tolist()
        t0 = time.perf_counter()
        enc_q = [pub.encrypt(float(v)) for v in q]
        scores = []
        for di in range(n_docs):
            d = docs_f[di].tolist()
            enc_dot = enc_q[0] * d[0]
            for i in range(1, len(q)):
                enc_dot = enc_dot + enc_q[i] * d[i]
            scores.append(priv.decrypt(enc_dot))
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        results.append({"ranked": ranked, "time_ms": elapsed * 1000})
    return results


# =====================================================================
# Main
# =====================================================================

def recall_at_k(true_r, pred_r, k):
    return len(set(true_r[:k]) & set(pred_r[:k])) / k


def run():
    print("EXP-009: 64-bit Native Secret Sharing Comparison")
    print(f"  dim={DIM}, k={PRECISION}, queries={NUM_QUERIES}")
    print("=" * 80)

    rng = np.random.default_rng(SEED)
    max_docs = max(DOC_COUNTS)
    all_docs_f = random_unit_vectors(max_docs, DIM, rng)
    queries_f = random_unit_vectors(NUM_QUERIES, DIM, rng)

    # Pre-encode all docs and queries as integer matrices
    print("Encoding vectors...", end="", flush=True)
    t0 = time.perf_counter()
    all_docs_int = encode_matrix(all_docs_f, PRECISION)
    queries_int = [
        [n for n, _ in encode_vector(queries_f[qi].tolist(), PRECISION)]
        for qi in range(NUM_QUERIES)
    ]
    print(f" {time.perf_counter() - t0:.2f}s")

    # Ground truth
    ground_truth = {}
    for n_docs in DOC_COUNTS:
        for qi in range(NUM_QUERIES):
            scores = (all_docs_f[:n_docs] @ queries_f[qi]).tolist()
            ground_truth[(n_docs, qi)] = sorted(range(n_docs), key=lambda i: -scores[i])

    rows = []

    for n_docs in DOC_COUNTS:
        print(f"\n{'=' * 80}")
        print(f"N = {n_docs}")
        print(f"{'=' * 80}")

        schemes = [
            ("plaintext", "none"),
            ("ours_qp_64", "info-theoretic"),
            ("ours_mp_64", "info-theoretic"),
            ("ckks_8192", "computational"),
        ]
        if n_docs <= PAILLIER_MAX_DOCS:
            schemes.append(("paillier", "computational"))

        for scheme, security in schemes:
            print(f"\n  [{scheme}]")
            setup_s = 0

            if scheme == "plaintext":
                res = bench_plaintext(queries_f, all_docs_f, n_docs)
            elif scheme == "ours_qp_64":
                res = bench_query_private_64(queries_int, all_docs_int, n_docs)
            elif scheme == "ours_mp_64":
                res, setup_s = bench_mutual_privacy_64(queries_int, all_docs_int, n_docs)
            elif scheme == "ckks_8192":
                res = bench_ckks(queries_f, all_docs_f, n_docs)
            elif scheme == "paillier":
                res = bench_paillier(queries_f, all_docs_f, n_docs)

            # Metrics
            times = [r["time_ms"] for r in res]
            recalls = {tk: [] for tk in TOP_KS}
            for qi, r in enumerate(res):
                for tk in TOP_KS:
                    if tk <= n_docs:
                        recalls[tk].append(
                            recall_at_k(ground_truth[(n_docs, qi)], r["ranked"], tk)
                        )

            mean_t = np.mean(times)
            r1 = np.mean(recalls[1]) if recalls[1] else 0
            r5 = np.mean(recalls[5]) if recalls[5] else 0
            r10 = np.mean(recalls[10]) if recalls[10] else 0
            qps = 1000 / mean_t if mean_t > 0 else float("inf")

            print(f"    {mean_t:.2f} ms | R@1={r1:.3f} R@5={r5:.3f} R@10={r10:.3f} | "
                  f"{qps:.1f} q/s")

            rows.append({
                "scheme": scheme, "security": security,
                "n_docs": n_docs, "mean_ms": round(mean_t, 2),
                "r1": round(r1, 3), "r5": round(r5, 3), "r10": round(r10, 3),
                "qps": round(qps, 1), "setup_s": round(setup_s, 3),
            })

    # ── Paper table ──
    print(f"\n{'=' * 80}")
    print("PAPER TABLE")
    print(f"{'=' * 80}")
    header = (f"{'Scheme':<16} {'Security':<16} {'N':>6} {'ms':>10} "
              f"{'R@1':>6} {'R@5':>6} {'R@10':>6} {'q/s':>10}")
    print(header)
    print("-" * len(header))

    for r in rows:
        print(f"{r['scheme']:<16} {r['security']:<16} {r['n_docs']:>6} "
              f"{r['mean_ms']:>10.2f} {r['r1']:>6.3f} {r['r5']:>6.3f} "
              f"{r['r10']:>6.3f} {r['qps']:>10.1f}")

    # Save CSV
    import csv
    csv_path = OUTPUT / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved to {csv_path}")


if __name__ == "__main__":
    t_start = time.perf_counter()
    run()
    total = time.perf_counter() - t_start
    print(f"\nTotal: {total:.0f}s ({total/60:.1f}m)")
