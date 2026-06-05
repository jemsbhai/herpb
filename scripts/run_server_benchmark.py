#!/usr/bin/env python3
"""
EXP-010: Comprehensive A100/B200 Benchmark for KDD/ICDM Submission
====================================================================

14 phases covering ablations, real RAG workflows, and scaling:

  --- CORE VALIDATION ---
  Phase  1: Correctness verification (all schemes)
  Phase  2: Query-private scaling (N = 100 to 1M)
  Phase  3: Mutual-privacy scaling (N = 100 to 100K, matrix-vector Beaver)

  --- BASELINES ---
  Phase  4: CKKS baseline (N = 50 to 5K)
  Phase  5: Paillier baseline (N = 50)

  --- REAL RAG WORKFLOWS ---
  Phase  6: Real embeddings -- MS MARCO (5K passages, all-MiniLM-L6-v2)
  Phase  7: Real embeddings -- Natural Questions (10K passages)
  Phase  8: Multi-model comparison (MiniLM d=384, mpnet d=768, BGE d=1024)
  Phase  9: BEIR benchmark suite (6 datasets, end-to-end retrieval metrics)

  --- ABLATIONS ---
  Phase 10: Precision sweep (k = 4 to 32)
  Phase 11: Dimension sweep (d = 64, 128, 256, 384, 512, 768, 1024)
  Phase 12: Latency breakdown (encode, share, dot, reconstruct, rank)
  Phase 13: Concurrent query throughput (batch 1, 10, 50, 100 queries)

  --- GPU EXTENSION ---
  Phase 14: 2048-bit Montgomery kernel (deep circuits extension)

Usage:
  python scripts/run_server_benchmark.py                    # all phases
  python scripts/run_server_benchmark.py --phases 1,2,3     # specific phases
  python scripts/run_server_benchmark.py --resume           # skip completed

  # Recommended ordering for 8-hour budget:
  # First (must-have, ~2h):   --phases 1,2,3,4,5,10,11,12
  # Second (real data, ~3h):  --phases 6,7,8,9 --resume
  # Third (bonus, ~1h):       --phases 13,14 --resume

Estimated total: 6-8 hours on A100.
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

from herpb.utils.encoding import encode_vector, dot_product_error_bound
from herpb.schemes.secret_sharing_64 import (
    share_vector, reconstruct_scalar, share_matrix,
    query_private_batch, setup_matrix_beaver,
    generate_query_triple, mutual_privacy_batch,
)

PROJECT = Path(__file__).parent.parent
OUTPUT = PROJECT / "experiments" / "exp_010_server"
FIGURES = PROJECT / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 2026
DEFAULT_DIM = 384
DEFAULT_K = 16
NUM_QUERIES = 20


# =====================================================================
# Utilities
# =====================================================================

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def random_unit_vectors(n, dim, rng):
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def encode_matrix(vecs, k):
    rows = []
    for v in vecs:
        rows.append([n for n, _ in encode_vector(v.tolist(), k)])
    return np.array(rows, dtype=np.int64)


def recall_at_k(true_ranking, pred_ranking, k):
    return len(set(true_ranking[:k]) & set(pred_ranking[:k])) / k


def ndcg_at_k(true_ranking, pred_ranking, k):
    """Normalized Discounted Cumulative Gain at k."""
    def dcg(ranking, k):
        return sum(1.0 / math.log2(i + 2) for i, idx in enumerate(ranking[:k])
                   if idx in set(true_ranking[:k]))
    ideal = dcg(true_ranking, k)
    if ideal == 0:
        return 1.0
    return dcg(pred_ranking, k) / ideal


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    log(f"Saved {path.name} ({len(rows)} rows)")


def phase_done(p):
    return (OUTPUT / f"phase{p}_done.flag").exists()


def mark_done(p):
    (OUTPUT / f"phase{p}_done.flag").write_text(datetime.now().isoformat())


def run_qp_benchmark(queries_int, doc_int_matrix, n_docs, num_queries=None):
    """Run query-private benchmark, return per-query times and rankings."""
    D = doc_int_matrix[:n_docs]
    nq = num_queries or len(queries_int)
    times, rankings = [], []
    for qi in range(nq):
        t0 = time.perf_counter()
        q1, q2 = share_vector(queries_int[qi])
        dots1, dots2 = query_private_batch(q1, q2, D)
        scores = [reconstruct_scalar(np.uint64(dots1[j]), np.uint64(dots2[j]))
                  for j in range(n_docs)]
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        times.append((time.perf_counter() - t0) * 1000)
        rankings.append(ranked)
    return times, rankings


def run_mp_benchmark(queries_int, doc_int_matrix, n_docs, num_queries=None):
    """Run mutual-privacy benchmark with matrix-vector Beaver."""
    D = doc_int_matrix[:n_docs]
    nq = num_queries or len(queries_int)
    rng = np.random.default_rng(SEED)
    D1, D2 = share_matrix(D, rng)
    setup = setup_matrix_beaver(D1, D2, rng)

    times, rankings = [], []
    for qi in range(nq):
        t0 = time.perf_counter()
        triple = generate_query_triple(setup, np.random.default_rng(qi + 10000))
        q1, q2 = share_vector(queries_int[qi], np.random.default_rng(qi + 20000))
        Z1, Z2 = mutual_privacy_batch(q1, q2, setup, triple)
        scores = [reconstruct_scalar(np.uint64(Z1[j]), np.uint64(Z2[j]))
                  for j in range(n_docs)]
        ranked = sorted(range(n_docs), key=lambda i: -scores[i])
        times.append((time.perf_counter() - t0) * 1000)
        rankings.append(ranked)
    return times, rankings


def compute_metrics(true_rankings, pred_rankings, times):
    """Compute R@1, R@5, R@10, nDCG@10, mean/p50/p95 latency."""
    r1 = np.mean([recall_at_k(t, p, 1) for t, p in zip(true_rankings, pred_rankings)])
    r5 = np.mean([recall_at_k(t, p, 5) for t, p in zip(true_rankings, pred_rankings)])
    r10 = np.mean([recall_at_k(t, p, 10) for t, p in zip(true_rankings, pred_rankings)])
    n10 = np.mean([ndcg_at_k(t, p, 10) for t, p in zip(true_rankings, pred_rankings)])
    return {
        "r1": round(r1, 4), "r5": round(r5, 4), "r10": round(r10, 4),
        "ndcg10": round(n10, 4),
        "mean_ms": round(np.mean(times), 3),
        "p50_ms": round(np.median(times), 3),
        "p95_ms": round(np.percentile(times, 95), 3),
        "std_ms": round(np.std(times), 3),
        "qps": round(1000 / np.mean(times), 2) if np.mean(times) > 0 else 0,
    }


# =====================================================================
# Phase 1: Correctness
# =====================================================================

def run_phase1():
    log("PHASE 1: Correctness Verification")
    rng = np.random.default_rng(SEED)
    docs = random_unit_vectors(50, DEFAULT_DIM, rng)
    queries = random_unit_vectors(10, DEFAULT_DIM, rng)
    doc_ints = encode_matrix(docs, DEFAULT_K)
    queries_int = [[n for n, _ in encode_vector(q.tolist(), DEFAULT_K)] for q in queries]

    true_scores = queries @ docs.T
    true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(10)]

    _, qp_rankings = run_qp_benchmark(queries_int, doc_ints, 50, 10)
    _, mp_rankings = run_mp_benchmark(queries_int, doc_ints, 50, 10)

    all_pass = True
    for qi in range(10):
        qp_ok = qp_rankings[qi] == true_rankings[qi]
        mp_ok = mp_rankings[qi] == true_rankings[qi]
        if not (qp_ok and mp_ok):
            all_pass = False
            log(f"  FAIL query {qi}: QP={qp_ok} MP={mp_ok}")

    log(f"  Result: {'ALL 10 PASS' if all_pass else 'FAILURES'}")
    save_csv(OUTPUT / "phase1_correctness.csv", [{"all_pass": all_pass, "n_queries": 10}])


# =====================================================================
# Phase 2: Query-Private Scaling
# =====================================================================

def run_phase2():
    log("PHASE 2: Query-Private Scaling")
    QP_COUNTS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000]
    rng = np.random.default_rng(SEED)
    max_n = max(QP_COUNTS)

    log(f"  Generating {max_n:,} vectors (dim={DEFAULT_DIM})...")
    all_docs = random_unit_vectors(max_n, DEFAULT_DIM, rng)
    queries = random_unit_vectors(NUM_QUERIES, DEFAULT_DIM, rng)
    all_docs_int = encode_matrix(all_docs, DEFAULT_K)
    queries_int = [[n for n, _ in encode_vector(q.tolist(), DEFAULT_K)] for q in queries]

    rows = []
    for n_docs in QP_COUNTS:
        log(f"\n  N = {n_docs:,}")
        true_scores = queries @ all_docs[:n_docs].T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(NUM_QUERIES)]

        times, pred_rankings = run_qp_benchmark(queries_int, all_docs_int, n_docs)
        m = compute_metrics(true_rankings, pred_rankings, times)
        m.update({"scheme": "query_private", "n_docs": n_docs, "dim": DEFAULT_DIM, "k": DEFAULT_K})
        rows.append(m)
        log(f"    {m['mean_ms']:.2f} ms | R@1={m['r1']:.3f} | nDCG@10={m['ndcg10']:.3f} | {m['qps']} q/s")

    save_csv(OUTPUT / "phase2_qp_scaling.csv", rows)


# =====================================================================
# Phase 3: Mutual-Privacy Scaling
# =====================================================================

def run_phase3():
    log("PHASE 3: Mutual-Privacy Scaling")
    MP_COUNTS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]
    rng = np.random.default_rng(SEED)
    max_n = max(MP_COUNTS)

    all_docs = random_unit_vectors(max_n, DEFAULT_DIM, rng)
    queries = random_unit_vectors(NUM_QUERIES, DEFAULT_DIM, rng)
    all_docs_int = encode_matrix(all_docs, DEFAULT_K)
    queries_int = [[n for n, _ in encode_vector(q.tolist(), DEFAULT_K)] for q in queries]

    rows = []
    for n_docs in MP_COUNTS:
        log(f"\n  N = {n_docs:,}")
        true_scores = queries @ all_docs[:n_docs].T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(NUM_QUERIES)]

        times, pred_rankings = run_mp_benchmark(queries_int, all_docs_int, n_docs)
        m = compute_metrics(true_rankings, pred_rankings, times)
        m.update({"scheme": "mutual_privacy", "n_docs": n_docs, "dim": DEFAULT_DIM, "k": DEFAULT_K})
        rows.append(m)
        log(f"    {m['mean_ms']:.2f} ms | R@1={m['r1']:.3f} | {m['qps']} q/s")

    save_csv(OUTPUT / "phase3_mp_scaling.csv", rows)


# =====================================================================
# Phase 4: CKKS Baseline
# =====================================================================

def run_phase4():
    log("PHASE 4: CKKS Baseline")
    try:
        import tenseal as ts
    except ImportError:
        log("  TenSEAL not installed, skipping"); return

    CKKS_COUNTS = [50, 200, 1_000, 5_000]
    rng = np.random.default_rng(SEED)
    all_docs = random_unit_vectors(max(CKKS_COUNTS), DEFAULT_DIM, rng)
    queries = random_unit_vectors(5, DEFAULT_DIM, rng)  # fewer, CKKS is slow

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 40, 60])
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40

    rows = []
    for n_docs in CKKS_COUNTS:
        log(f"\n  CKKS N = {n_docs}")
        true_scores = queries @ all_docs[:n_docs].T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(len(queries))]

        times, pred_rankings = [], []
        for qi in range(len(queries)):
            t0 = time.perf_counter()
            enc_q = ts.ckks_vector(ctx, queries[qi].tolist())
            scores = []
            for di in range(n_docs):
                enc_d = ts.ckks_vector(ctx, all_docs[di].tolist())
                scores.append(enc_q.dot(enc_d).decrypt()[0])
            ranked = sorted(range(n_docs), key=lambda i: -scores[i])
            times.append((time.perf_counter() - t0) * 1000)
            pred_rankings.append(ranked)

        m = compute_metrics(true_rankings, pred_rankings, times)
        m.update({"scheme": "ckks_8192", "n_docs": n_docs, "dim": DEFAULT_DIM})
        rows.append(m)
        log(f"    {m['mean_ms']:.0f} ms | R@1={m['r1']:.3f}")

    save_csv(OUTPUT / "phase4_ckks.csv", rows)


# =====================================================================
# Phase 5: Paillier Baseline
# =====================================================================

def run_phase5():
    log("PHASE 5: Paillier Baseline")
    try:
        from phe import paillier
    except ImportError:
        log("  phe not installed, skipping"); return

    rng = np.random.default_rng(SEED)
    docs = random_unit_vectors(50, DEFAULT_DIM, rng)
    queries = random_unit_vectors(3, DEFAULT_DIM, rng)
    pub, priv = paillier.generate_paillier_keypair(n_length=2048)

    true_scores = queries @ docs.T
    true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(3)]

    times, pred_rankings = [], []
    for qi in range(3):
        t0 = time.perf_counter()
        enc_q = [pub.encrypt(float(v)) for v in queries[qi].tolist()]
        scores = []
        for di in range(50):
            d = docs[di].tolist()
            enc_dot = enc_q[0] * d[0]
            for i in range(1, DEFAULT_DIM):
                enc_dot = enc_dot + enc_q[i] * d[i]
            scores.append(priv.decrypt(enc_dot))
        ranked = sorted(range(50), key=lambda i: -scores[i])
        times.append((time.perf_counter() - t0) * 1000)
        pred_rankings.append(ranked)

    m = compute_metrics(true_rankings, pred_rankings, times)
    m.update({"scheme": "paillier", "n_docs": 50, "dim": DEFAULT_DIM})
    save_csv(OUTPUT / "phase5_paillier.csv", [m])
    log(f"  {m['mean_ms']:.0f} ms | R@1={m['r1']:.3f}")


# =====================================================================
# Phase 6: Real Embeddings -- MS MARCO
# =====================================================================

def load_sentence_model(model_name):
    from sentence_transformers import SentenceTransformer
    log(f"  Loading model: {model_name}")
    return SentenceTransformer(model_name)


def run_phase6():
    log("PHASE 6: Real Embeddings -- MS MARCO")
    try:
        from datasets import load_dataset
    except ImportError:
        log("  datasets not installed, skipping"); return

    model = load_sentence_model("all-MiniLM-L6-v2")
    log("  Loading MS MARCO passages...")
    ds = load_dataset("ms_marco", "v1.1", split="train", trust_remote_code=True)

    passages, queries_text = [], []
    for row in ds:
        for p in row["passages"]["passage_text"]:
            passages.append(p)
            if len(passages) >= 6000:
                break
        if row["query"] and len(queries_text) < 100:
            queries_text.append(row["query"])
        if len(passages) >= 6000 and len(queries_text) >= 100:
            break

    log(f"  Collected {len(passages)} passages, {len(queries_text)} queries")
    doc_emb = model.encode(passages[:5000], batch_size=256, show_progress_bar=True,
                            normalize_embeddings=True).astype(np.float32)
    query_emb = model.encode(queries_text[:50], batch_size=64,
                              normalize_embeddings=True).astype(np.float32)
    dim = doc_emb.shape[1]
    log(f"  Encoded: {doc_emb.shape[0]} docs, {query_emb.shape[0]} queries, dim={dim}")

    rows = []
    for n_docs in [500, 1_000, 5_000]:
        log(f"\n  MS MARCO N = {n_docs}")
        docs = doc_emb[:n_docs]
        doc_ints = encode_matrix(docs, DEFAULT_K)
        nq = min(50, query_emb.shape[0])

        true_scores = query_emb[:nq] @ docs.T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(nq)]
        queries_int = [[n for n, _ in encode_vector(query_emb[qi].tolist(), DEFAULT_K)]
                       for qi in range(nq)]

        # Query-private
        times_qp, ranks_qp = run_qp_benchmark(queries_int, doc_ints, n_docs, nq)
        m_qp = compute_metrics(true_rankings, ranks_qp, times_qp)

        # Mutual-privacy
        times_mp, ranks_mp = run_mp_benchmark(queries_int, doc_ints, n_docs, nq)
        m_mp = compute_metrics(true_rankings, ranks_mp, times_mp)

        log(f"    QP: {m_qp['mean_ms']:.2f} ms, R@1={m_qp['r1']:.3f}, nDCG@10={m_qp['ndcg10']:.3f}")
        log(f"    MP: {m_mp['mean_ms']:.2f} ms, R@1={m_mp['r1']:.3f}, nDCG@10={m_mp['ndcg10']:.3f}")

        for scheme, m in [("qp", m_qp), ("mp", m_mp)]:
            m.update({"dataset": "ms_marco", "model": "all-MiniLM-L6-v2",
                      "n_docs": n_docs, "dim": dim, "scheme": scheme})
            rows.append(m)

    save_csv(OUTPUT / "phase6_msmarco.csv", rows)


# =====================================================================
# Phase 7: Real Embeddings -- Natural Questions
# =====================================================================

def run_phase7():
    log("PHASE 7: Real Embeddings -- Natural Questions")
    try:
        from datasets import load_dataset
    except ImportError:
        log("  datasets not installed, skipping"); return

    model = load_sentence_model("all-MiniLM-L6-v2")
    log("  Loading Natural Questions...")
    ds = load_dataset("natural_questions", "default", split="train",
                      trust_remote_code=True, streaming=True)

    passages, queries_text = [], []
    for row in ds:
        if row.get("document", {}).get("tokens", {}).get("token"):
            text = " ".join(row["document"]["tokens"]["token"][:200])
            passages.append(text)
        if row.get("question", {}).get("text"):
            queries_text.append(row["question"]["text"])
        if len(passages) >= 10_000 and len(queries_text) >= 100:
            break

    if len(passages) < 100:
        log("  Not enough passages loaded, skipping")
        return

    n_passages = min(len(passages), 10_000)
    n_queries = min(len(queries_text), 50)
    log(f"  Collected {n_passages} passages, {n_queries} queries")

    doc_emb = model.encode(passages[:n_passages], batch_size=256,
                            show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
    query_emb = model.encode(queries_text[:n_queries], batch_size=64,
                              normalize_embeddings=True).astype(np.float32)
    dim = doc_emb.shape[1]

    rows = []
    for n_docs in [1_000, 5_000, min(n_passages, 10_000)]:
        if n_docs > n_passages:
            continue
        log(f"\n  NQ N = {n_docs}")
        docs = doc_emb[:n_docs]
        doc_ints = encode_matrix(docs, DEFAULT_K)
        queries_int = [[n for n, _ in encode_vector(query_emb[qi].tolist(), DEFAULT_K)]
                       for qi in range(n_queries)]

        true_scores = query_emb[:n_queries] @ docs.T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(n_queries)]

        times, ranks = run_qp_benchmark(queries_int, doc_ints, n_docs, n_queries)
        m = compute_metrics(true_rankings, ranks, times)
        m.update({"dataset": "natural_questions", "model": "all-MiniLM-L6-v2",
                  "n_docs": n_docs, "dim": dim, "scheme": "qp"})
        rows.append(m)
        log(f"    QP: {m['mean_ms']:.2f} ms, R@1={m['r1']:.3f}, nDCG@10={m['ndcg10']:.3f}")

    save_csv(OUTPUT / "phase7_nq.csv", rows)


# =====================================================================
# Phase 8: Multi-Model Comparison
# =====================================================================

def run_phase8():
    log("PHASE 8: Multi-Model Dimension Comparison")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log("  sentence-transformers not installed, skipping"); return

    models = [
        ("all-MiniLM-L6-v2", 384),
        ("all-mpnet-base-v2", 768),
        ("BAAI/bge-large-en-v1.5", 1024),
    ]

    # Shared corpus text
    rng_text = np.random.default_rng(SEED)
    sentences = [
        f"This is test sentence number {i} about topic {rng_text.choice(['science','art','history','tech','medicine'])}"
        for i in range(2000)
    ]
    query_sentences = [f"Tell me about {t}" for t in ["science","art","history","tech","medicine",
                       "biology","painting","ancient","software","health"]] * 5

    rows = []
    for model_name, expected_dim in models:
        log(f"\n  Model: {model_name} (expected dim={expected_dim})")
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            log(f"    Failed to load: {e}, skipping")
            continue

        doc_emb = model.encode(sentences[:1000], batch_size=128,
                                normalize_embeddings=True).astype(np.float32)
        query_emb = model.encode(query_sentences[:20], batch_size=64,
                                  normalize_embeddings=True).astype(np.float32)
        dim = doc_emb.shape[1]
        log(f"    Actual dim={dim}")

        n_docs = 1000
        doc_ints = encode_matrix(doc_emb, DEFAULT_K)
        queries_int = [[n for n, _ in encode_vector(query_emb[qi].tolist(), DEFAULT_K)]
                       for qi in range(20)]

        true_scores = query_emb @ doc_emb.T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(20)]

        times_qp, ranks_qp = run_qp_benchmark(queries_int, doc_ints, n_docs, 20)
        m_qp = compute_metrics(true_rankings, ranks_qp, times_qp)

        times_mp, ranks_mp = run_mp_benchmark(queries_int, doc_ints, n_docs, 20)
        m_mp = compute_metrics(true_rankings, ranks_mp, times_mp)

        log(f"    QP: {m_qp['mean_ms']:.2f} ms, R@1={m_qp['r1']:.3f}")
        log(f"    MP: {m_mp['mean_ms']:.2f} ms, R@1={m_mp['r1']:.3f}")

        for scheme, m in [("qp", m_qp), ("mp", m_mp)]:
            m.update({"model": model_name, "dim": dim, "n_docs": n_docs, "scheme": scheme})
            rows.append(m)

    save_csv(OUTPUT / "phase8_multimodel.csv", rows)


# =====================================================================
# Phase 9: BEIR Benchmark Suite
# =====================================================================

def run_phase9():
    log("PHASE 9: BEIR Benchmark Suite")
    try:
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log("  Dependencies missing, skipping"); return

    model = load_sentence_model("all-MiniLM-L6-v2")

    # BEIR datasets available on HuggingFace
    beir_datasets = [
        ("mteb/scifact", "corpus", "queries", 5183, 300),
        ("mteb/fiqa", "corpus", "queries", 57638, 648),
        ("mteb/nfcorpus", "corpus", "queries", 3633, 323),
        ("mteb/arguana", "corpus", "queries", 8674, 1406),
        ("mteb/scidocs", "corpus", "queries", 25657, 1000),
        ("mteb/trec-covid", "corpus", "queries", 171332, 50),
    ]

    rows = []
    for ds_name, corpus_split, query_split, expected_corpus, expected_queries in beir_datasets:
        short_name = ds_name.split("/")[-1]
        log(f"\n  BEIR: {short_name}")
        try:
            corpus_ds = load_dataset(ds_name, "corpus", split="corpus", trust_remote_code=True)
            queries_ds = load_dataset(ds_name, "queries", split="queries", trust_remote_code=True)
        except Exception as e:
            log(f"    Failed to load: {e}, skipping")
            continue

        # Extract text
        corpus_texts = []
        for row in corpus_ds:
            text = row.get("text", "") or ""
            title = row.get("title", "") or ""
            corpus_texts.append(f"{title} {text}".strip()[:512])
            if len(corpus_texts) >= 5000:
                break

        query_texts = []
        for row in queries_ds:
            text = row.get("text", "") or ""
            query_texts.append(text)
            if len(query_texts) >= 50:
                break

        if len(corpus_texts) < 100 or len(query_texts) < 10:
            log(f"    Too few items ({len(corpus_texts)} docs, {len(query_texts)} queries), skipping")
            continue

        n_docs = len(corpus_texts)
        n_queries = len(query_texts)
        log(f"    Encoding {n_docs} docs, {n_queries} queries...")

        doc_emb = model.encode(corpus_texts, batch_size=256, show_progress_bar=True,
                                normalize_embeddings=True).astype(np.float32)
        query_emb = model.encode(query_texts, batch_size=64,
                                  normalize_embeddings=True).astype(np.float32)
        dim = doc_emb.shape[1]

        doc_ints = encode_matrix(doc_emb, DEFAULT_K)
        queries_int = [[n for n, _ in encode_vector(query_emb[qi].tolist(), DEFAULT_K)]
                       for qi in range(n_queries)]

        true_scores = query_emb @ doc_emb.T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(n_queries)]

        times, ranks = run_qp_benchmark(queries_int, doc_ints, n_docs, n_queries)
        m = compute_metrics(true_rankings, ranks, times)
        m.update({"dataset": short_name, "n_docs": n_docs, "n_queries": n_queries,
                  "dim": dim, "scheme": "qp"})
        rows.append(m)
        log(f"    R@1={m['r1']:.3f} nDCG@10={m['ndcg10']:.3f} | {m['mean_ms']:.2f} ms")

    save_csv(OUTPUT / "phase9_beir.csv", rows)


# =====================================================================
# Phase 10: Precision Sweep
# =====================================================================

def run_phase10():
    log("PHASE 10: Precision Sweep")
    K_VALUES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    rng = np.random.default_rng(SEED)
    docs = random_unit_vectors(1000, DEFAULT_DIM, rng)
    queries = random_unit_vectors(NUM_QUERIES, DEFAULT_DIM, rng)

    true_scores = queries @ docs.T
    true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(NUM_QUERIES)]

    rows = []
    for k in K_VALUES:
        # Check if values fit in int64
        max_val = DEFAULT_DIM * (2 ** (2 * k))
        if max_val > 2**62:
            log(f"  k={k}: max_val={max_val:.0e} > 2^62, OVERFLOW RISK, skipping")
            continue

        doc_ints = encode_matrix(docs, k)
        queries_int = [[n for n, _ in encode_vector(q.tolist(), k)] for q in queries]

        times, ranks = run_qp_benchmark(queries_int, doc_ints, 1000)
        m = compute_metrics(true_rankings, ranks, times)
        bound = dot_product_error_bound(k, DEFAULT_DIM)
        m.update({"precision_k": k, "dim": DEFAULT_DIM, "n_docs": 1000,
                  "error_bound": f"{bound:.4e}"})
        rows.append(m)
        log(f"  k={k}: R@1={m['r1']:.3f} R@10={m['r10']:.3f} nDCG@10={m['ndcg10']:.3f} bound={bound:.2e}")

    save_csv(OUTPUT / "phase10_precision.csv", rows)


# =====================================================================
# Phase 11: Dimension Sweep
# =====================================================================

def run_phase11():
    log("PHASE 11: Dimension Sweep")
    DIMS = [64, 128, 256, 384, 512, 768, 1024]
    rng = np.random.default_rng(SEED)
    n_docs = 5000

    rows = []
    for dim in DIMS:
        log(f"\n  dim = {dim}")
        docs = random_unit_vectors(n_docs, dim, rng)
        queries = random_unit_vectors(NUM_QUERIES, dim, rng)
        doc_ints = encode_matrix(docs, DEFAULT_K)
        queries_int = [[n for n, _ in encode_vector(q.tolist(), DEFAULT_K)] for q in queries]

        true_scores = queries @ docs.T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(NUM_QUERIES)]

        times_qp, ranks_qp = run_qp_benchmark(queries_int, doc_ints, n_docs)
        m_qp = compute_metrics(true_rankings, ranks_qp, times_qp)

        times_mp, ranks_mp = run_mp_benchmark(queries_int, doc_ints, n_docs)
        m_mp = compute_metrics(true_rankings, ranks_mp, times_mp)

        log(f"    QP: {m_qp['mean_ms']:.2f} ms | MP: {m_mp['mean_ms']:.2f} ms | R@1={m_qp['r1']:.3f}")

        for scheme, m in [("qp", m_qp), ("mp", m_mp)]:
            m.update({"dim": dim, "n_docs": n_docs, "k": DEFAULT_K, "scheme": scheme,
                      "per_doc_bytes_mp": dim * 8 * 2,
                      "online_comm_bytes_mp": dim * 8})
            rows.append(m)

    save_csv(OUTPUT / "phase11_dimension.csv", rows)


# =====================================================================
# Phase 12: Latency Breakdown
# =====================================================================

def run_phase12():
    log("PHASE 12: Latency Breakdown")
    rng = np.random.default_rng(SEED)
    n_docs = 5000
    docs = random_unit_vectors(n_docs, DEFAULT_DIM, rng)
    queries = random_unit_vectors(NUM_QUERIES, DEFAULT_DIM, rng)
    doc_ints = encode_matrix(docs, DEFAULT_K)

    rows = []
    for qi in range(NUM_QUERIES):
        q = queries[qi]

        # 1. Encode
        t0 = time.perf_counter()
        q_ints = [n for n, _ in encode_vector(q.tolist(), DEFAULT_K)]
        t_encode = time.perf_counter() - t0

        # 2. Share
        t0 = time.perf_counter()
        q1, q2 = share_vector(q_ints)
        t_share = time.perf_counter() - t0

        # 3. Dot product (both servers)
        t0 = time.perf_counter()
        dots1, dots2 = query_private_batch(q1, q2, doc_ints)
        t_dot = time.perf_counter() - t0

        # 4. Reconstruct
        t0 = time.perf_counter()
        scores = [reconstruct_scalar(np.uint64(dots1[j]), np.uint64(dots2[j]))
                  for j in range(n_docs)]
        t_reconstruct = time.perf_counter() - t0

        # 5. Rank
        t0 = time.perf_counter()
        _ = sorted(range(n_docs), key=lambda i: -scores[i])
        t_rank = time.perf_counter() - t0

        total = t_encode + t_share + t_dot + t_reconstruct + t_rank
        rows.append({
            "query": qi, "n_docs": n_docs, "dim": DEFAULT_DIM,
            "encode_ms": round(t_encode * 1000, 4),
            "share_ms": round(t_share * 1000, 4),
            "dot_product_ms": round(t_dot * 1000, 4),
            "reconstruct_ms": round(t_reconstruct * 1000, 4),
            "rank_ms": round(t_rank * 1000, 4),
            "total_ms": round(total * 1000, 4),
            "dot_pct": round(100 * t_dot / total, 1),
        })

    save_csv(OUTPUT / "phase12_breakdown.csv", rows)
    avg_dot_pct = np.mean([r["dot_pct"] for r in rows])
    log(f"  Dot product accounts for {avg_dot_pct:.0f}% of total latency")


# =====================================================================
# Phase 13: Concurrent Query Throughput
# =====================================================================

def run_phase13():
    log("PHASE 13: Concurrent Query Throughput (Batch)")
    rng = np.random.default_rng(SEED)
    n_docs = 10_000
    docs = random_unit_vectors(n_docs, DEFAULT_DIM, rng)
    doc_ints = encode_matrix(docs, DEFAULT_K)

    BATCH_SIZES = [1, 5, 10, 20, 50, 100]
    rows = []

    for batch in BATCH_SIZES:
        queries = random_unit_vectors(batch, DEFAULT_DIM, rng)
        queries_int = [[n for n, _ in encode_vector(q.tolist(), DEFAULT_K)] for q in queries]

        # Measure total time for all queries in the batch
        t0 = time.perf_counter()
        for qi in range(batch):
            q1, q2 = share_vector(queries_int[qi])
            query_private_batch(q1, q2, doc_ints)
        total_ms = (time.perf_counter() - t0) * 1000
        per_query_ms = total_ms / batch
        throughput = batch / (total_ms / 1000)

        log(f"  Batch={batch}: {total_ms:.1f} ms total, {per_query_ms:.2f} ms/query, {throughput:.0f} q/s")
        rows.append({
            "batch_size": batch, "n_docs": n_docs, "dim": DEFAULT_DIM,
            "total_ms": round(total_ms, 2),
            "per_query_ms": round(per_query_ms, 2),
            "throughput_qps": round(throughput, 1),
        })

    save_csv(OUTPUT / "phase13_throughput.csv", rows)


# =====================================================================
# Phase 14: 2048-bit Montgomery Kernel
# =====================================================================

def run_phase14():
    log("PHASE 14: 2048-bit Montgomery GPU Kernel")
    try:
        import cupy as cp
        from herpb.gpu.montgomery import (
            MontgomeryContext, to_montgomery_batch,
            from_montgomery_batch, montgomery_multiply_batch,
        )
    except Exception as e:
        log(f"  GPU module not available ({e}), skipping"); return

    import gmpy2
    from gmpy2 import mpz
    rs = gmpy2.random_state(42)
    p = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, 1024) | (mpz(1) << 1023))
    q = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, 1024) | (mpz(1) << 1023))
    n = p * q
    ctx = MontgomeryContext(n)

    rows = []
    for batch in [1_000, 10_000, 100_000, 1_000_000]:
        log(f"  Batch = {batch:,}")
        a_vals = [gmpy2.mpz_urandomb(rs, 2048) % n for _ in range(batch)]
        b_vals = [gmpy2.mpz_urandomb(rs, 2048) % n for _ in range(batch)]

        t0 = time.perf_counter()
        cpu_results = [(a * b) % n for a, b in zip(a_vals, b_vals)]
        cpu_ms = (time.perf_counter() - t0) * 1000

        try:
            a_m = to_montgomery_batch(a_vals, ctx)
            b_m = to_montgomery_batch(b_vals, ctx)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            c_m = montgomery_multiply_batch(a_m, b_m, ctx)
            cp.cuda.Stream.null.synchronize()
            gpu_ms = (time.perf_counter() - t0) * 1000

            log(f"    CPU: {cpu_ms:.1f} ms | GPU: {gpu_ms:.2f} ms | {cpu_ms/gpu_ms:.1f}x")
            rows.append({"batch": batch, "cpu_ms": round(cpu_ms, 2),
                         "gpu_ms": round(gpu_ms, 2), "speedup": round(cpu_ms/gpu_ms, 1)})
        except Exception as e:
            log(f"    GPU failed: {e}")

    save_csv(OUTPUT / "phase14_montgomery.csv", rows)


# =====================================================================
# Figure Generation
# =====================================================================

def generate_all_figures():
    log("Generating publication figures...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dpi = 300

    def load(name):
        path = OUTPUT / name
        if not path.exists(): return []
        with open(path) as f: return list(csv.DictReader(f))

    p2 = load("phase2_qp_scaling.csv")
    p3 = load("phase3_mp_scaling.csv")
    p4 = load("phase4_ckks.csv")
    p10 = load("phase10_precision.csv")
    p11 = load("phase11_dimension.csv")
    p12 = load("phase12_breakdown.csv")
    p13 = load("phase13_throughput.csv")

    # Fig 1: Log-log scaling
    if p2 and p4:
        fig, ax = plt.subplots(figsize=(8, 5))
        ns = [int(r["n_docs"]) for r in p2]; ms = [float(r["mean_ms"]) for r in p2]
        ax.plot(ns, ms, "o-", label="Ours (query-private)", color="#2ecc71", lw=2, ms=6)
        if p3:
            ns3 = [int(r["n_docs"]) for r in p3]; ms3 = [float(r["mean_ms"]) for r in p3]
            ax.plot(ns3, ms3, "s-", label="Ours (mutual-privacy)", color="#27ae60", lw=2, ms=6)
        ns4 = [int(r["n_docs"]) for r in p4]; ms4 = [float(r["mean_ms"]) for r in p4]
        ax.plot(ns4, ms4, "^-", label="CKKS-8192", color="#e74c3c", lw=2, ms=6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Corpus size N"); ax.set_ylabel("Query latency (ms)")
        ax.set_title(f"Query Latency Scaling (dim={DEFAULT_DIM})")
        ax.legend(); ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_scaling.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(FIGURES / "fig_scaling.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig); log("  fig_scaling")

    # Fig 2: Precision vs recall
    if p10:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ks = [int(r["precision_k"]) for r in p10]
        r1s = [float(r["r1"]) for r in p10]
        n10s = [float(r["ndcg10"]) for r in p10]
        ax.plot(ks, r1s, "o-", label="Recall@1", color="#2ecc71", lw=2)
        ax.plot(ks, n10s, "s-", label="nDCG@10", color="#3498db", lw=2)
        ax.set_xlabel("Precision k (bits)"); ax.set_ylabel("Score")
        ax.set_title("Ranking Accuracy vs Encoding Precision")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_precision.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(FIGURES / "fig_precision.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig); log("  fig_precision")

    # Fig 3: Dimension sweep
    if p11:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        qp = [r for r in p11 if r["scheme"] == "qp"]
        mp = [r for r in p11 if r["scheme"] == "mp"]
        if qp:
            ax.plot([int(r["dim"]) for r in qp], [float(r["mean_ms"]) for r in qp],
                    "o-", label="Query-private", color="#2ecc71", lw=2)
        if mp:
            ax.plot([int(r["dim"]) for r in mp], [float(r["mean_ms"]) for r in mp],
                    "s-", label="Mutual-privacy", color="#27ae60", lw=2)
        ax.set_xlabel("Embedding dimension d"); ax.set_ylabel("Query latency (ms)")
        ax.set_title(f"Latency vs Dimension (N=5000)")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_dimension.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(FIGURES / "fig_dimension.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig); log("  fig_dimension")

    # Fig 4: Latency breakdown
    if p12:
        fig, ax = plt.subplots(figsize=(7, 4))
        components = ["encode_ms", "share_ms", "dot_product_ms", "reconstruct_ms", "rank_ms"]
        labels = ["Encode", "Share", "Dot Product", "Reconstruct", "Rank"]
        colors = ["#f39c12", "#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]
        means = [np.mean([float(r[c]) for r in p12]) for c in components]
        ax.barh(labels, means, color=colors, alpha=0.85)
        ax.set_xlabel("Time (ms)"); ax.set_title(f"Latency Breakdown (N=5000, dim={DEFAULT_DIM})")
        for i, v in enumerate(means):
            ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_breakdown.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(FIGURES / "fig_breakdown.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig); log("  fig_breakdown")

    # Fig 5: Throughput scaling
    if p13:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bs = [int(r["batch_size"]) for r in p13]
        qps = [float(r["throughput_qps"]) for r in p13]
        ax.plot(bs, qps, "o-", color="#2ecc71", lw=2, ms=6)
        ax.set_xlabel("Concurrent queries (batch size)"); ax.set_ylabel("Throughput (q/s)")
        ax.set_title(f"Throughput vs Batch Size (N=10K, dim={DEFAULT_DIM})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "fig_throughput.pdf", dpi=dpi, bbox_inches="tight")
        fig.savefig(FIGURES / "fig_throughput.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig); log("  fig_throughput")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="EXP-010 Server Benchmark")
    parser.add_argument("--phases", type=str, default=",".join(str(i) for i in range(1, 15)),
                        help="Comma-separated phase numbers")
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed phases")
    args = parser.parse_args()
    phases = [int(x) for x in args.phases.split(",")]

    log("EXP-010: Comprehensive Server Benchmark")
    log(f"  Phases: {phases}")
    log(f"  Output: {OUTPUT}")

    import platform
    log(f"  Python: {platform.python_version()}, NumPy: {np.__version__}, CPUs: {os.cpu_count()}")
    try:
        import subprocess
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        ).decode().strip()
        log(f"  GPU: {gpu}")
    except Exception:
        log("  GPU: not detected")

    phase_map = {
        1: ("Correctness", run_phase1),
        2: ("QP Scaling", run_phase2),
        3: ("MP Scaling", run_phase3),
        4: ("CKKS Baseline", run_phase4),
        5: ("Paillier Baseline", run_phase5),
        6: ("MS MARCO Real", run_phase6),
        7: ("Natural Questions Real", run_phase7),
        8: ("Multi-Model", run_phase8),
        9: ("BEIR Suite", run_phase9),
        10: ("Precision Sweep", run_phase10),
        11: ("Dimension Sweep", run_phase11),
        12: ("Latency Breakdown", run_phase12),
        13: ("Throughput Batch", run_phase13),
        14: ("Montgomery GPU", run_phase14),
    }

    t_total = time.perf_counter()
    for p in phases:
        if p not in phase_map:
            continue
        name, func = phase_map[p]
        if args.resume and phase_done(p):
            log(f"Phase {p} ({name}): done, skipping")
            continue
        log(f"\n{'#' * 70}\n# Phase {p}: {name}\n{'#' * 70}")
        t0 = time.perf_counter()
        try:
            func()
            mark_done(p)
            log(f"Phase {p} done in {time.perf_counter()-t0:.1f}s")
        except Exception as e:
            log(f"Phase {p} FAILED: {e}")
            import traceback; traceback.print_exc()

    try:
        generate_all_figures()
    except Exception as e:
        log(f"Figure gen failed: {e}")

    total = time.perf_counter() - t_total
    log(f"\nTotal: {total:.0f}s ({total/60:.1f}m / {total/3600:.1f}h)")


if __name__ == "__main__":
    main()
