#!/usr/bin/env python3
"""
EXP-010: Comprehensive Server Benchmark (A100/B200)
=====================================================

Eight phases covering everything needed for the paper:

  Phase 1: Correctness verification (all schemes, quick)
  Phase 2: Query-private scaling (N = 100 to 1M, 64-bit)
  Phase 3: Mutual-privacy scaling (N = 100 to 100K, matrix-vector Beaver)
  Phase 4: CKKS baseline scaling (N = 50 to 5000)
  Phase 5: Paillier baseline (N = 50 only)
  Phase 6: Real embeddings (MS MARCO + all-MiniLM-L6-v2)
  Phase 7: Precision sweep (k = 8 to 24, ranking accuracy)
  Phase 8: 2048-bit Montgomery kernel (GPU extension benchmarks)

Usage:
  python scripts/run_server_benchmark.py                    # all phases
  python scripts/run_server_benchmark.py --phases 1,2,3     # specific phases
  python scripts/run_server_benchmark.py --resume           # skip completed phases

Estimated time: 2-4 hours on A100. Phase 4 (CKKS at large N) is slowest.

Output:
  experiments/exp_010_server/           (CSV results per phase)
  figures/                              (publication-ready PDF + PNG)
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Project imports ──
from herpb.utils.encoding import encode_vector, dot_product_error_bound
from herpb.schemes.secret_sharing_64 import (
    share_vector, reconstruct_scalar, share_matrix,
    query_private_batch, setup_matrix_beaver,
    generate_query_triple, mutual_privacy_batch,
)

# ── Paths ──
PROJECT = Path(__file__).parent.parent
OUTPUT = PROJECT / "experiments" / "exp_010_server"
FIGURES = PROJECT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Config ──
SEED = 2026
DIM = 384
PRECISION = 16
NUM_QUERIES = 20

# Phase 2: query-private scaling
QP_DOC_COUNTS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000]

# Phase 3: mutual-privacy scaling
MP_DOC_COUNTS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

# Phase 4: CKKS scaling
CKKS_DOC_COUNTS = [50, 200, 1_000, 5_000]

# Phase 7: precision sweep
PRECISION_VALUES = [8, 10, 12, 14, 16, 20, 24]


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def random_unit_vectors(n, dim, rng):
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def encode_matrix(vecs, k):
    rows = []
    for v in vecs:
        ints = [n for n, _ in encode_vector(v.tolist(), k)]
        rows.append(ints)
    return np.array(rows, dtype=np.int64)


def recall_at_k(true_ranking, pred_ranking, k):
    return len(set(true_ranking[:k]) & set(pred_ranking[:k])) / k


def true_dot_scores(queries, docs):
    """Return (N_queries, N_docs) score matrix."""
    return queries @ docs.T


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    log(f"Saved {path.name} ({len(rows)} rows)")


def phase_done(phase_num):
    return (OUTPUT / f"phase{phase_num}_done.flag").exists()


def mark_done(phase_num):
    (OUTPUT / f"phase{phase_num}_done.flag").write_text(
        datetime.now().isoformat()
    )


# =====================================================================
# Phase 1: Correctness Verification
# =====================================================================

def run_phase1():
    log("PHASE 1: Correctness Verification")
    log("=" * 60)

    rng = np.random.default_rng(SEED)
    n_docs = 50
    docs = random_unit_vectors(n_docs, DIM, rng)
    queries = random_unit_vectors(5, DIM, rng)
    doc_ints = encode_matrix(docs, PRECISION)

    rows = []
    for qi in range(5):
        q = queries[qi]
        q_ints = [n for n, _ in encode_vector(q.tolist(), PRECISION)]

        # Ground truth
        true_scores = (docs @ q).tolist()
        true_ranking = sorted(range(n_docs), key=lambda i: -true_scores[i])

        # Query-private
        q1, q2 = share_vector(q_ints)
        dots1, dots2 = query_private_batch(q1, q2, doc_ints)
        qp_scores = [
            reconstruct_scalar(np.uint64(dots1[j]), np.uint64(dots2[j])) / (2**(2*PRECISION))
            for j in range(n_docs)
        ]
        qp_ranking = sorted(range(n_docs), key=lambda i: -qp_scores[i])
        qp_match = qp_ranking == true_ranking

        # Mutual-privacy
        rng_mp = np.random.default_rng(qi)
        D1, D2 = share_matrix(doc_ints, rng_mp)
        setup = setup_matrix_beaver(D1, D2, rng_mp)
        triple = generate_query_triple(setup, rng_mp)
        q1m, q2m = share_vector(q_ints, rng_mp)
        Z1, Z2 = mutual_privacy_batch(q1m, q2m, setup, triple)
        mp_scores = [
            reconstruct_scalar(np.uint64(Z1[j]), np.uint64(Z2[j])) / (2**(2*PRECISION))
            for j in range(n_docs)
        ]
        mp_ranking = sorted(range(n_docs), key=lambda i: -mp_scores[i])
        mp_match = mp_ranking == true_ranking

        # Max score error
        max_err = max(abs(qp_scores[j] - true_scores[j]) for j in range(n_docs))
        bound = dot_product_error_bound(PRECISION, DIM)

        rows.append({
            "query": qi, "qp_ranking_match": qp_match,
            "mp_ranking_match": mp_match,
            "max_score_error": f"{max_err:.2e}",
            "error_bound": f"{bound:.2e}",
            "within_bound": max_err <= bound + 1e-10,
        })
        status = "PASS" if (qp_match and mp_match) else "FAIL"
        log(f"  Query {qi}: {status} | err={max_err:.2e} <= {bound:.2e}")

    save_csv(OUTPUT / "phase1_correctness.csv", rows)
    all_pass = all(r["qp_ranking_match"] and r["mp_ranking_match"] for r in rows)
    log(f"Phase 1: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    return all_pass


# =====================================================================
# Phase 2: Query-Private Scaling
# =====================================================================

def run_phase2():
    log("PHASE 2: Query-Private Scaling")
    log("=" * 60)

    rng = np.random.default_rng(SEED)
    max_n = max(QP_DOC_COUNTS)
    log(f"Generating {max_n} random unit vectors at dim={DIM}...")
    all_docs = random_unit_vectors(max_n, DIM, rng)
    queries = random_unit_vectors(NUM_QUERIES, DIM, rng)

    log("Encoding document matrix...")
    t0 = time.perf_counter()
    all_docs_int = encode_matrix(all_docs, PRECISION)
    log(f"  Encoded in {time.perf_counter()-t0:.1f}s")

    queries_int = []
    for qi in range(NUM_QUERIES):
        queries_int.append([n for n, _ in encode_vector(queries[qi].tolist(), PRECISION)])

    rows = []
    for n_docs in QP_DOC_COUNTS:
        log(f"\n  N = {n_docs:,}")
        D = all_docs_int[:n_docs]
        scores_matrix = true_dot_scores(queries, all_docs[:n_docs])

        times = []
        recalls = {1: [], 5: [], 10: []}

        for qi in range(NUM_QUERIES):
            true_ranking = np.argsort(-scores_matrix[qi]).tolist()

            t0 = time.perf_counter()
            q1, q2 = share_vector(queries_int[qi])
            dots1, dots2 = query_private_batch(q1, q2, D)
            enc_scores = [
                reconstruct_scalar(np.uint64(dots1[j]), np.uint64(dots2[j]))
                for j in range(n_docs)
            ]
            enc_ranking = sorted(range(n_docs), key=lambda i: -enc_scores[i])
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)

            for k_val in [1, 5, 10]:
                if k_val <= n_docs:
                    recalls[k_val].append(recall_at_k(true_ranking, enc_ranking, k_val))

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        r1 = np.mean(recalls[1]) if recalls[1] else 0
        r5 = np.mean(recalls[5]) if recalls[5] else 0
        r10 = np.mean(recalls[10]) if recalls[10] else 0
        qps = 1000 / mean_ms if mean_ms > 0 else float("inf")

        log(f"    {mean_ms:.2f} ms (std {std_ms:.2f}) | R@1={r1:.3f} | {qps:.1f} q/s")

        rows.append({
            "scheme": "query_private_64",
            "n_docs": n_docs, "dim": DIM, "precision": PRECISION,
            "mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3),
            "median_ms": round(np.median(times), 3),
            "p95_ms": round(np.percentile(times, 95), 3),
            "r1": round(r1, 4), "r5": round(r5, 4), "r10": round(r10, 4),
            "qps": round(qps, 2), "num_queries": NUM_QUERIES,
        })

    save_csv(OUTPUT / "phase2_qp_scaling.csv", rows)


# =====================================================================
# Phase 3: Mutual-Privacy Scaling
# =====================================================================

def run_phase3():
    log("PHASE 3: Mutual-Privacy Scaling (Matrix-Vector Beaver)")
    log("=" * 60)

    rng = np.random.default_rng(SEED)
    max_n = max(MP_DOC_COUNTS)
    all_docs = random_unit_vectors(max_n, DIM, rng)
    queries = random_unit_vectors(NUM_QUERIES, DIM, rng)
    all_docs_int = encode_matrix(all_docs, PRECISION)

    queries_int = []
    for qi in range(NUM_QUERIES):
        queries_int.append([n for n, _ in encode_vector(queries[qi].tolist(), PRECISION)])

    rows = []
    for n_docs in MP_DOC_COUNTS:
        log(f"\n  N = {n_docs:,}")
        D = all_docs_int[:n_docs]
        scores_matrix = true_dot_scores(queries, all_docs[:n_docs])

        # Setup (one-time)
        rng_setup = np.random.default_rng(SEED + n_docs)
        t_setup = time.perf_counter()
        D1, D2 = share_matrix(D, rng_setup)
        setup = setup_matrix_beaver(D1, D2, rng_setup)
        setup_s = time.perf_counter() - t_setup
        log(f"    Setup: {setup_s:.2f}s")

        times = []
        triple_gen_times = []
        recalls = {1: [], 5: [], 10: []}

        for qi in range(NUM_QUERIES):
            true_ranking = np.argsort(-scores_matrix[qi]).tolist()

            t0 = time.perf_counter()
            rng_q = np.random.default_rng(qi + 10000)
            triple = generate_query_triple(setup, rng_q)
            t_triple = time.perf_counter() - t0
            triple_gen_times.append(t_triple * 1000)

            q1, q2 = share_vector(queries_int[qi], rng_q)
            Z1, Z2 = mutual_privacy_batch(q1, q2, setup, triple)

            enc_scores = [
                reconstruct_scalar(np.uint64(Z1[j]), np.uint64(Z2[j]))
                for j in range(n_docs)
            ]
            enc_ranking = sorted(range(n_docs), key=lambda i: -enc_scores[i])
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)

            for k_val in [1, 5, 10]:
                if k_val <= n_docs:
                    recalls[k_val].append(recall_at_k(true_ranking, enc_ranking, k_val))

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        r1 = np.mean(recalls[1]) if recalls[1] else 0
        qps = 1000 / mean_ms if mean_ms > 0 else float("inf")

        log(f"    Query: {mean_ms:.2f} ms (std {std_ms:.2f}) | R@1={r1:.3f} | {qps:.1f} q/s")
        log(f"    Triple gen: {np.mean(triple_gen_times):.2f} ms avg")

        # Communication: f vector = d * 8 bytes
        online_comm_bytes = DIM * 8

        rows.append({
            "scheme": "mutual_privacy_64",
            "n_docs": n_docs, "dim": DIM, "precision": PRECISION,
            "setup_s": round(setup_s, 3),
            "mean_ms": round(mean_ms, 3), "std_ms": round(std_ms, 3),
            "triple_gen_ms": round(np.mean(triple_gen_times), 3),
            "r1": round(r1, 4),
            "qps": round(qps, 2),
            "online_comm_bytes": online_comm_bytes,
            "per_doc_storage_bytes": DIM * 8 * 2,  # two uint64 shares
            "num_queries": NUM_QUERIES,
        })

    save_csv(OUTPUT / "phase3_mp_scaling.csv", rows)


# =====================================================================
# Phase 4: CKKS Baseline
# =====================================================================

def run_phase4():
    log("PHASE 4: CKKS Baseline Scaling")
    log("=" * 60)

    try:
        import tenseal as ts
    except ImportError:
        log("  TenSEAL not installed, skipping")
        return

    rng = np.random.default_rng(SEED)
    max_n = max(CKKS_DOC_COUNTS)
    all_docs = random_unit_vectors(max_n, DIM, rng)
    queries = random_unit_vectors(min(NUM_QUERIES, 5), DIM, rng)  # fewer queries, CKKS is slow

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40

    rows = []
    for n_docs in CKKS_DOC_COUNTS:
        log(f"\n  N = {n_docs} (CKKS, {len(queries)} queries)")
        scores_matrix = true_dot_scores(queries, all_docs[:n_docs])

        times = []
        recalls_1 = []

        for qi in range(len(queries)):
            true_ranking = np.argsort(-scores_matrix[qi]).tolist()

            t0 = time.perf_counter()
            enc_q = ts.ckks_vector(ctx, queries[qi].tolist())
            scores = []
            for di in range(n_docs):
                enc_d = ts.ckks_vector(ctx, all_docs[di].tolist())
                scores.append(enc_q.dot(enc_d).decrypt()[0])
            enc_ranking = sorted(range(n_docs), key=lambda i: -scores[i])
            elapsed = time.perf_counter() - t0
            times.append(elapsed * 1000)
            recalls_1.append(recall_at_k(true_ranking, enc_ranking, 1))

        mean_ms = np.mean(times)
        r1 = np.mean(recalls_1)
        log(f"    {mean_ms:.0f} ms | R@1={r1:.3f}")

        rows.append({
            "scheme": "ckks_8192",
            "n_docs": n_docs, "dim": DIM,
            "mean_ms": round(mean_ms, 1),
            "r1": round(r1, 4),
            "qps": round(1000 / mean_ms, 4),
            "num_queries": len(queries),
        })

    save_csv(OUTPUT / "phase4_ckks.csv", rows)


# =====================================================================
# Phase 5: Paillier Baseline
# =====================================================================

def run_phase5():
    log("PHASE 5: Paillier Baseline")
    log("=" * 60)

    try:
        from phe import paillier
    except ImportError:
        log("  phe not installed, skipping")
        return

    rng = np.random.default_rng(SEED)
    n_docs = 50
    docs = random_unit_vectors(n_docs, DIM, rng)
    queries = random_unit_vectors(3, DIM, rng)

    pub, priv = paillier.generate_paillier_keypair(n_length=2048)

    times = []
    recalls_1 = []

    for qi in range(len(queries)):
        true_scores = (docs @ queries[qi]).tolist()
        true_ranking = sorted(range(n_docs), key=lambda i: -true_scores[i])

        t0 = time.perf_counter()
        enc_q = [pub.encrypt(float(v)) for v in queries[qi].tolist()]
        scores = []
        for di in range(n_docs):
            d = docs[di].tolist()
            enc_dot = enc_q[0] * d[0]
            for i in range(1, DIM):
                enc_dot = enc_dot + enc_q[i] * d[i]
            scores.append(priv.decrypt(enc_dot))
        enc_ranking = sorted(range(n_docs), key=lambda i: -scores[i])
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)
        recalls_1.append(recall_at_k(true_ranking, enc_ranking, 1))

    mean_ms = np.mean(times)
    log(f"  N={n_docs}: {mean_ms:.0f} ms | R@1={np.mean(recalls_1):.3f}")

    save_csv(OUTPUT / "phase5_paillier.csv", [{
        "scheme": "paillier", "n_docs": n_docs, "dim": DIM,
        "mean_ms": round(mean_ms, 1),
        "r1": round(np.mean(recalls_1), 4),
        "num_queries": len(queries),
    }])


# =====================================================================
# Phase 6: Real Embeddings (MS MARCO + Sentence-BERT)
# =====================================================================

def run_phase6():
    log("PHASE 6: Real Embeddings")
    log("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
    except ImportError:
        log("  sentence-transformers or datasets not installed, skipping")
        return

    model_name = "all-MiniLM-L6-v2"
    log(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    log("  Loading MS MARCO passages...")
    ds = load_dataset("ms_marco", "v1.1", split="train", trust_remote_code=True)
    passages = []
    for row in ds:
        for p in row["passages"]["passage_text"]:
            passages.append(p)
            if len(passages) >= 10_000:
                break
        if len(passages) >= 10_000:
            break
    log(f"  Collected {len(passages)} passages")

    log("  Encoding passages...")
    doc_embeddings = model.encode(passages[:5_000], batch_size=256,
                                   show_progress_bar=True, normalize_embeddings=True)
    doc_embeddings = doc_embeddings.astype(np.float32)
    dim = doc_embeddings.shape[1]
    log(f"  Encoded {doc_embeddings.shape[0]} passages, dim={dim}")

    # Generate queries from different passages
    query_texts = [passages[5000 + i] for i in range(50)]
    query_embeddings = model.encode(query_texts, normalize_embeddings=True).astype(np.float32)

    rows = []
    for n_docs in [100, 500, 1_000, 5_000]:
        log(f"\n  Real embeddings, N = {n_docs}")
        docs = doc_embeddings[:n_docs]
        doc_ints = encode_matrix(docs, PRECISION)
        n_queries = min(20, len(query_embeddings))

        scores_matrix = true_dot_scores(query_embeddings[:n_queries], docs)

        times_qp = []
        times_mp = []
        recalls_qp = []
        recalls_mp = []

        for qi in range(n_queries):
            true_ranking = np.argsort(-scores_matrix[qi]).tolist()
            q_ints = [n for n, _ in encode_vector(query_embeddings[qi].tolist(), PRECISION)]

            # Query-private
            t0 = time.perf_counter()
            q1, q2 = share_vector(q_ints)
            dots1, dots2 = query_private_batch(q1, q2, doc_ints)
            enc_scores = [
                reconstruct_scalar(np.uint64(dots1[j]), np.uint64(dots2[j]))
                for j in range(n_docs)
            ]
            enc_ranking = sorted(range(n_docs), key=lambda i: -enc_scores[i])
            times_qp.append((time.perf_counter() - t0) * 1000)
            recalls_qp.append(recall_at_k(true_ranking, enc_ranking, 1))

        # Mutual-privacy (one setup for all queries)
        rng_mp = np.random.default_rng(SEED)
        D1, D2 = share_matrix(doc_ints, rng_mp)
        setup = setup_matrix_beaver(D1, D2, rng_mp)

        for qi in range(n_queries):
            true_ranking = np.argsort(-scores_matrix[qi]).tolist()
            q_ints = [n for n, _ in encode_vector(query_embeddings[qi].tolist(), PRECISION)]

            t0 = time.perf_counter()
            rng_q = np.random.default_rng(qi + 50000)
            triple = generate_query_triple(setup, rng_q)
            q1, q2 = share_vector(q_ints, rng_q)
            Z1, Z2 = mutual_privacy_batch(q1, q2, setup, triple)
            enc_scores = [
                reconstruct_scalar(np.uint64(Z1[j]), np.uint64(Z2[j]))
                for j in range(n_docs)
            ]
            enc_ranking = sorted(range(n_docs), key=lambda i: -enc_scores[i])
            times_mp.append((time.perf_counter() - t0) * 1000)
            recalls_mp.append(recall_at_k(true_ranking, enc_ranking, 1))

        log(f"    QP: {np.mean(times_qp):.2f} ms, R@1={np.mean(recalls_qp):.3f}")
        log(f"    MP: {np.mean(times_mp):.2f} ms, R@1={np.mean(recalls_mp):.3f}")

        rows.append({
            "n_docs": n_docs, "dim": dim,
            "qp_mean_ms": round(np.mean(times_qp), 3),
            "qp_r1": round(np.mean(recalls_qp), 4),
            "mp_mean_ms": round(np.mean(times_mp), 3),
            "mp_r1": round(np.mean(recalls_mp), 4),
            "num_queries": n_queries,
            "model": model_name,
            "dataset": "ms_marco_v1.1",
        })

    save_csv(OUTPUT / "phase6_real_embeddings.csv", rows)


# =====================================================================
# Phase 7: Precision Sweep
# =====================================================================

def run_phase7():
    log("PHASE 7: Precision Sweep")
    log("=" * 60)

    rng = np.random.default_rng(SEED)
    n_docs = 500
    docs = random_unit_vectors(n_docs, DIM, rng)
    queries = random_unit_vectors(NUM_QUERIES, DIM, rng)

    rows = []
    for k in PRECISION_VALUES:
        log(f"\n  k = {k}")
        doc_ints = encode_matrix(docs, k)
        scores_matrix = true_dot_scores(queries, docs)

        recalls_1 = []
        recalls_10 = []
        max_errors = []

        for qi in range(NUM_QUERIES):
            true_ranking = np.argsort(-scores_matrix[qi]).tolist()
            q_ints = [n for n, _ in encode_vector(queries[qi].tolist(), k)]

            q1, q2 = share_vector(q_ints)
            dots1, dots2 = query_private_batch(q1, q2, doc_ints)

            enc_scores = [
                reconstruct_scalar(np.uint64(dots1[j]), np.uint64(dots2[j])) / (2**(2*k))
                for j in range(n_docs)
            ]
            enc_ranking = sorted(range(n_docs), key=lambda i: -enc_scores[i])

            true_scores = scores_matrix[qi].tolist()
            max_err = max(abs(enc_scores[j] - true_scores[j]) for j in range(n_docs))
            max_errors.append(max_err)

            recalls_1.append(recall_at_k(true_ranking, enc_ranking, 1))
            recalls_10.append(recall_at_k(true_ranking, enc_ranking, 10))

        bound = dot_product_error_bound(k, DIM)
        r1 = np.mean(recalls_1)
        r10 = np.mean(recalls_10)
        log(f"    R@1={r1:.3f} R@10={r10:.3f} | max_err={np.max(max_errors):.2e} "
            f"<= bound={bound:.2e}")

        rows.append({
            "precision_k": k, "n_docs": n_docs, "dim": DIM,
            "r1": round(r1, 4), "r10": round(r10, 4),
            "max_observed_error": f"{np.max(max_errors):.4e}",
            "theoretical_bound": f"{bound:.4e}",
            "within_bound": all(e <= bound + 1e-10 for e in max_errors),
            "num_queries": NUM_QUERIES,
        })

    save_csv(OUTPUT / "phase7_precision.csv", rows)


# =====================================================================
# Phase 8: 2048-bit Montgomery Kernel (GPU Extension)
# =====================================================================

def run_phase8():
    log("PHASE 8: 2048-bit Montgomery GPU Kernel")
    log("=" * 60)

    try:
        import cupy as cp
        from herpb.gpu.montgomery import (
            MontgomeryContext, to_montgomery_batch, from_montgomery_batch,
            montgomery_multiply_batch,
        )
    except ImportError:
        log("  CuPy or GPU module not available, skipping")
        return
    except Exception as e:
        log(f"  GPU init failed: {e}, skipping")
        return

    import gmpy2
    from gmpy2 import mpz

    # Generate test modulus
    rs = gmpy2.random_state(42)
    p = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, 1024) | (mpz(1) << 1023))
    q = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, 1024) | (mpz(1) << 1023))
    n = p * q

    ctx = MontgomeryContext(n)

    batch_sizes = [1_000, 10_000, 100_000, 1_000_000]
    rows = []

    for batch in batch_sizes:
        log(f"\n  Batch = {batch:,}")

        # Generate random operands
        a_vals = [gmpy2.mpz_urandomb(rs, 2048) % n for _ in range(batch)]
        b_vals = [gmpy2.mpz_urandomb(rs, 2048) % n for _ in range(batch)]

        # CPU baseline (GMP)
        t0 = time.perf_counter()
        cpu_results = [(a * b) % n for a, b in zip(a_vals, b_vals)]
        cpu_ms = (time.perf_counter() - t0) * 1000
        log(f"    CPU/GMP: {cpu_ms:.1f} ms")

        # GPU Montgomery
        try:
            a_mont = to_montgomery_batch(a_vals, ctx)
            b_mont = to_montgomery_batch(b_vals, ctx)

            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            c_mont = montgomery_multiply_batch(a_mont, b_mont, ctx)
            cp.cuda.Stream.null.synchronize()
            gpu_kernel_ms = (time.perf_counter() - t0) * 1000

            c_host = from_montgomery_batch(c_mont, ctx)
            log(f"    GPU kernel: {gpu_kernel_ms:.2f} ms")
            log(f"    Speedup: {cpu_ms / gpu_kernel_ms:.1f}x")

            # Verify correctness (sample)
            for i in [0, batch // 2, batch - 1]:
                assert c_host[i] == cpu_results[i], f"Mismatch at {i}"

            rows.append({
                "batch_size": batch,
                "cpu_gmp_ms": round(cpu_ms, 2),
                "gpu_kernel_ms": round(gpu_kernel_ms, 2),
                "speedup": round(cpu_ms / gpu_kernel_ms, 1),
            })
        except Exception as e:
            log(f"    GPU failed: {e}")

    save_csv(OUTPUT / "phase8_montgomery.csv", rows)


# =====================================================================
# Figure Generation
# =====================================================================

def generate_figures():
    log("Generating publication figures...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 300

    # ── Load all results ──
    def load_csv(name):
        path = OUTPUT / name
        if not path.exists():
            return []
        with open(path) as f:
            return list(csv.DictReader(f))

    p2 = load_csv("phase2_qp_scaling.csv")
    p3 = load_csv("phase3_mp_scaling.csv")
    p4 = load_csv("phase4_ckks.csv")
    p5 = load_csv("phase5_paillier.csv")
    p6 = load_csv("phase6_real_embeddings.csv")
    p7 = load_csv("phase7_precision.csv")

    # ── Fig A: Scaling comparison (log-log) ──
    if p2 and p4:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        ns_qp = [int(r["n_docs"]) for r in p2]
        ms_qp = [float(r["mean_ms"]) for r in p2]
        ax.plot(ns_qp, ms_qp, "o-", label="Ours (query-private)", color="#2ecc71", linewidth=2)

        if p3:
            ns_mp = [int(r["n_docs"]) for r in p3]
            ms_mp = [float(r["mean_ms"]) for r in p3]
            ax.plot(ns_mp, ms_mp, "s-", label="Ours (mutual-privacy)", color="#27ae60", linewidth=2)

        ns_ckks = [int(r["n_docs"]) for r in p4]
        ms_ckks = [float(r["mean_ms"]) for r in p4]
        ax.plot(ns_ckks, ms_ckks, "^-", label="CKKS-8192", color="#e74c3c", linewidth=2)

        if p5:
            for r in p5:
                ax.scatter(int(r["n_docs"]), float(r["mean_ms"]),
                          marker="D", s=100, color="#9b59b6", zorder=5, label="Paillier")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Corpus size N")
        ax.set_ylabel("Query latency (ms)")
        ax.set_title(f"Scaling: Query Latency vs Corpus Size (dim={DIM})")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_scaling.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(FIGURES / "fig_scaling.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log("  Saved fig_scaling")

    # ── Fig B: Precision vs Recall ──
    if p7:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        ks = [int(r["precision_k"]) for r in p7]
        r1s = [float(r["r1"]) for r in p7]
        r10s = [float(r["r10"]) for r in p7]
        ax.plot(ks, r1s, "o-", label="Recall@1", color="#2ecc71", linewidth=2)
        ax.plot(ks, r10s, "s-", label="Recall@10", color="#3498db", linewidth=2)
        ax.set_xlabel("Precision k (bits)")
        ax.set_ylabel("Recall")
        ax.set_title(f"Ranking Accuracy vs Encoding Precision (N=500, dim={DIM})")
        ax.set_ylim(0.8, 1.02)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_precision.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(FIGURES / "fig_precision.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log("  Saved fig_precision")

    # ── Fig C: Real embeddings ──
    if p6:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        ns = [int(r["n_docs"]) for r in p6]
        qp_ms = [float(r["qp_mean_ms"]) for r in p6]
        mp_ms = [float(r["mp_mean_ms"]) for r in p6]
        ax.plot(ns, qp_ms, "o-", label="Query-private", color="#2ecc71", linewidth=2)
        ax.plot(ns, mp_ms, "s-", label="Mutual-privacy", color="#27ae60", linewidth=2)
        ax.set_xlabel("Corpus size N")
        ax.set_ylabel("Query latency (ms)")
        ax.set_title("Real Embeddings: MS MARCO + all-MiniLM-L6-v2")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_real_embeddings.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(FIGURES / "fig_real_embeddings.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log("  Saved fig_real_embeddings")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="EXP-010 Server Benchmark")
    parser.add_argument("--phases", type=str, default="1,2,3,4,5,6,7,8",
                        help="Comma-separated phase numbers to run")
    parser.add_argument("--resume", action="store_true",
                        help="Skip phases that have completed")
    args = parser.parse_args()

    phases = [int(x) for x in args.phases.split(",")]

    log("EXP-010: Comprehensive Server Benchmark")
    log(f"  Phases: {phases}")
    log(f"  dim={DIM}, k={PRECISION}, queries={NUM_QUERIES}")
    log(f"  Output: {OUTPUT}")
    log("")

    # System info
    import platform
    log(f"  Python: {platform.python_version()}")
    log(f"  NumPy:  {np.__version__}")
    log(f"  CPU cores: {os.cpu_count()}")
    try:
        import subprocess
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        ).decode().strip()
        log(f"  GPU: {gpu}")
    except Exception:
        log("  GPU: not detected")
    log("")

    t_total = time.perf_counter()

    phase_funcs = {
        1: ("Correctness", run_phase1),
        2: ("QP Scaling", run_phase2),
        3: ("MP Scaling", run_phase3),
        4: ("CKKS Baseline", run_phase4),
        5: ("Paillier Baseline", run_phase5),
        6: ("Real Embeddings", run_phase6),
        7: ("Precision Sweep", run_phase7),
        8: ("Montgomery GPU", run_phase8),
    }

    for p in phases:
        if p not in phase_funcs:
            log(f"Unknown phase {p}, skipping")
            continue

        name, func = phase_funcs[p]
        if args.resume and phase_done(p):
            log(f"Phase {p} ({name}): already complete, skipping")
            continue

        log(f"\n{'#' * 70}")
        log(f"# Phase {p}: {name}")
        log(f"{'#' * 70}")

        t0 = time.perf_counter()
        try:
            func()
            mark_done(p)
            log(f"Phase {p} complete in {time.perf_counter()-t0:.1f}s")
        except Exception as e:
            log(f"Phase {p} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Generate figures from whatever data we have
    log(f"\n{'#' * 70}")
    log("# Figure Generation")
    log(f"{'#' * 70}")
    try:
        generate_figures()
    except Exception as e:
        log(f"Figure generation failed: {e}")

    total_s = time.perf_counter() - t_total
    log(f"\nTotal time: {total_s:.0f}s ({total_s/60:.1f}m / {total_s/3600:.1f}h)")
    log("Done.")


if __name__ == "__main__":
    main()
