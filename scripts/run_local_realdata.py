#!/usr/bin/env python3
"""
EXP-010 Local: Real-data phases that failed on A100 due to env issues.
Runs on local 4090 machine where sentence-transformers works.

Covers:
  Phase 6: MS MARCO real embeddings
  Phase 7: SQuAD real embeddings (NQ fallback)
  Phase 8: Multi-model (MiniLM d=384, mpnet d=768, BGE d=1024)
  Phase 9: BEIR benchmark suite (6 datasets)

Usage:
  python scripts/run_local_realdata.py
"""

import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from herpb.utils.encoding import encode_vector
from herpb.schemes.secret_sharing_64 import (
    share_vector, reconstruct_scalar, share_matrix,
    query_private_batch, setup_matrix_beaver,
    generate_query_triple, mutual_privacy_batch,
)

PROJECT = Path(__file__).parent.parent
OUTPUT = PROJECT / "experiments" / "exp_010_server"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
DEFAULT_K = 16
NUM_QUERIES = 20


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def encode_matrix(vecs, k):
    rows = []
    for v in vecs:
        rows.append([n for n, _ in encode_vector(v.tolist(), k)])
    return np.array(rows, dtype=np.int64)


def recall_at_k(true_r, pred_r, k):
    return len(set(true_r[:k]) & set(pred_r[:k])) / k


def ndcg_at_k(true_r, pred_r, k):
    def dcg(ranking, k):
        return sum(1.0 / math.log2(i + 2) for i, idx in enumerate(ranking[:k])
                   if idx in set(true_r[:k]))
    ideal = dcg(true_r, k)
    return dcg(pred_r, k) / ideal if ideal > 0 else 1.0


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    log(f"Saved {path.name} ({len(rows)} rows)")


def run_qp(queries_int, doc_int_matrix, n_docs, nq=None):
    D = doc_int_matrix[:n_docs]
    nq = nq or len(queries_int)
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


def run_mp(queries_int, doc_int_matrix, n_docs, nq=None):
    D = doc_int_matrix[:n_docs]
    nq = nq or len(queries_int)
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


def metrics(true_rankings, pred_rankings, times):
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
        "qps": round(1000 / np.mean(times), 2) if np.mean(times) > 0 else 0,
    }


def benchmark_on_embeddings(doc_emb, query_emb, doc_counts, dataset_name, model_name):
    """Run QP and MP benchmarks on pre-computed embeddings."""
    dim = doc_emb.shape[1]
    rows = []

    for n_docs in doc_counts:
        if n_docs > doc_emb.shape[0]:
            continue
        log(f"\n  {dataset_name} N={n_docs}")
        docs = doc_emb[:n_docs]
        doc_ints = encode_matrix(docs, DEFAULT_K)
        nq = min(NUM_QUERIES, query_emb.shape[0])

        queries_int = [[n for n, _ in encode_vector(query_emb[qi].tolist(), DEFAULT_K)]
                       for qi in range(nq)]
        true_scores = query_emb[:nq] @ docs.T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(nq)]

        # Query-private
        times_qp, ranks_qp = run_qp(queries_int, doc_ints, n_docs, nq)
        m_qp = metrics(true_rankings, ranks_qp, times_qp)
        log(f"    QP: {m_qp['mean_ms']:.2f} ms, R@1={m_qp['r1']:.3f}, nDCG@10={m_qp['ndcg10']:.3f}")
        m_qp.update({"dataset": dataset_name, "model": model_name,
                     "n_docs": n_docs, "dim": dim, "scheme": "qp", "n_queries": nq})
        rows.append(m_qp)

        # Mutual-privacy
        times_mp, ranks_mp = run_mp(queries_int, doc_ints, n_docs, nq)
        m_mp = metrics(true_rankings, ranks_mp, times_mp)
        log(f"    MP: {m_mp['mean_ms']:.2f} ms, R@1={m_mp['r1']:.3f}, nDCG@10={m_mp['ndcg10']:.3f}")
        m_mp.update({"dataset": dataset_name, "model": model_name,
                     "n_docs": n_docs, "dim": dim, "scheme": "mp", "n_queries": nq})
        rows.append(m_mp)

    return rows


# =====================================================================
# Phase 6: MS MARCO
# =====================================================================

def run_phase6():
    log("PHASE 6: MS MARCO Real Embeddings")
    log("=" * 60)
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    model = SentenceTransformer("all-MiniLM-L6-v2")

    passages, query_texts = [], []

    # Try multiple MS MARCO sources
    for ds_name in ["microsoft/ms_marco", "ms_marco"]:
        try:
            log(f"  Trying {ds_name}...")
            ds = load_dataset(ds_name, "v1.1", split="train")
            for row in ds:
                for p in row["passages"]["passage_text"]:
                    passages.append(p)
                    if len(passages) >= 6000:
                        break
                if row["query"] and len(query_texts) < 100:
                    query_texts.append(row["query"])
                if len(passages) >= 6000 and len(query_texts) >= 100:
                    break
            if len(passages) >= 100:
                log(f"  Loaded from {ds_name}")
                break
        except Exception as e:
            log(f"    Failed: {e}")

    # Fallback: SQuAD
    if len(passages) < 100:
        log("  Falling back to SQuAD...")
        ds = load_dataset("squad", split="train")
        for row in ds:
            passages.append(row["context"])
            query_texts.append(row["question"])
            if len(passages) >= 6000:
                break

    passages = list(dict.fromkeys(passages))[:5000]
    query_texts = list(dict.fromkeys(query_texts))[:50]
    log(f"  {len(passages)} passages, {len(query_texts)} queries")

    log("  Encoding passages...")
    doc_emb = model.encode(passages, batch_size=128, show_progress_bar=True,
                           normalize_embeddings=True).astype(np.float32)
    log("  Encoding queries...")
    query_emb = model.encode(query_texts, batch_size=64,
                             normalize_embeddings=True).astype(np.float32)

    rows = benchmark_on_embeddings(doc_emb, query_emb, [500, 1000, 5000],
                                   "ms_marco", "all-MiniLM-L6-v2")
    save_csv(OUTPUT / "phase6_msmarco.csv", rows)


# =====================================================================
# Phase 7: SQuAD (NQ fallback)
# =====================================================================

def run_phase7():
    log("PHASE 7: SQuAD Real Embeddings")
    log("=" * 60)
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    model = SentenceTransformer("all-MiniLM-L6-v2")

    log("  Loading SQuAD...")
    ds = load_dataset("squad", split="train")
    passages, query_texts = [], []
    for row in ds:
        passages.append(row["context"])
        query_texts.append(row["question"])
        if len(passages) >= 10000:
            break

    passages = list(dict.fromkeys(passages))
    query_texts = list(dict.fromkeys(query_texts))[:50]
    log(f"  {len(passages)} unique passages, {len(query_texts)} queries")

    log("  Encoding...")
    doc_emb = model.encode(passages[:5000], batch_size=128, show_progress_bar=True,
                           normalize_embeddings=True).astype(np.float32)
    query_emb = model.encode(query_texts, batch_size=64,
                             normalize_embeddings=True).astype(np.float32)

    rows = benchmark_on_embeddings(doc_emb, query_emb, [1000, 5000],
                                   "squad", "all-MiniLM-L6-v2")
    save_csv(OUTPUT / "phase7_squad.csv", rows)


# =====================================================================
# Phase 8: Multi-Model
# =====================================================================

def run_phase8():
    log("PHASE 8: Multi-Model Comparison")
    log("=" * 60)
    from sentence_transformers import SentenceTransformer

    models = [
        ("all-MiniLM-L6-v2", 384),
        ("all-mpnet-base-v2", 768),
        ("BAAI/bge-large-en-v1.5", 1024),
    ]

    # Shared corpus
    np_rng = np.random.default_rng(SEED)
    sentences = [f"This is test document number {i} covering topics in "
                 f"{np_rng.choice(['science','medicine','law','engineering','finance'])}"
                 for i in range(2000)]
    query_sentences = [
        "What are recent advances in machine learning?",
        "How does the immune system fight infections?",
        "What are the principles of contract law?",
        "How do bridges resist earthquake forces?",
        "What drives stock market volatility?",
        "How do neural networks learn representations?",
        "What causes antibiotic resistance?",
        "What is the doctrine of precedent?",
        "How is concrete strength measured?",
        "What is quantitative easing?",
    ] * 2

    all_rows = []
    for model_name, expected_dim in models:
        log(f"\n  Model: {model_name}")
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            log(f"    Failed to load: {e}")
            continue

        doc_emb = model.encode(sentences[:1000], batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)
        query_emb = model.encode(query_sentences[:20], batch_size=64,
                                 normalize_embeddings=True).astype(np.float32)
        dim = doc_emb.shape[1]
        log(f"    dim={dim}")

        rows = benchmark_on_embeddings(doc_emb, query_emb, [1000],
                                       f"synthetic_{model_name.split('/')[-1]}", model_name)
        all_rows.extend(rows)

    save_csv(OUTPUT / "phase8_multimodel.csv", all_rows)


# =====================================================================
# Phase 9: BEIR Benchmark Suite
# =====================================================================

def run_phase9():
    log("PHASE 9: BEIR Benchmark Suite")
    log("=" * 60)
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset

    model = SentenceTransformer("all-MiniLM-L6-v2")

    beir_datasets = [
        "mteb/scifact",
        "mteb/nfcorpus",
        "mteb/arguana",
        "mteb/fiqa",
        "mteb/scidocs",
        "mteb/trec-covid",
    ]

    all_rows = []
    for ds_name in beir_datasets:
        short = ds_name.split("/")[-1]
        log(f"\n  BEIR: {short}")
        try:
            corpus_ds = load_dataset(ds_name, "corpus", split="corpus")
            queries_ds = load_dataset(ds_name, "queries", split="queries")
        except Exception as e:
            log(f"    Failed: {e}")
            continue

        corpus_texts = []
        for row in corpus_ds:
            title = row.get("title", "") or ""
            text = row.get("text", "") or ""
            corpus_texts.append(f"{title} {text}".strip()[:512])
            if len(corpus_texts) >= 5000:
                break

        query_texts = []
        for row in queries_ds:
            text = row.get("text", "") or ""
            if text:
                query_texts.append(text)
            if len(query_texts) >= 50:
                break

        if len(corpus_texts) < 50 or len(query_texts) < 5:
            log(f"    Too few items, skipping")
            continue

        n_docs = len(corpus_texts)
        nq = len(query_texts)
        log(f"    Encoding {n_docs} docs, {nq} queries...")

        doc_emb = model.encode(corpus_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)
        query_emb = model.encode(query_texts, batch_size=64,
                                 normalize_embeddings=True).astype(np.float32)
        dim = doc_emb.shape[1]

        doc_ints = encode_matrix(doc_emb, DEFAULT_K)
        queries_int = [[n for n, _ in encode_vector(query_emb[qi].tolist(), DEFAULT_K)]
                       for qi in range(nq)]

        true_scores = query_emb @ doc_emb.T
        true_rankings = [np.argsort(-true_scores[qi]).tolist() for qi in range(nq)]

        times, ranks = run_qp(queries_int, doc_ints, n_docs, nq)
        m = metrics(true_rankings, ranks, times)
        m.update({"dataset": short, "n_docs": n_docs, "n_queries": nq,
                  "dim": dim, "scheme": "qp", "model": "all-MiniLM-L6-v2"})
        all_rows.append(m)
        log(f"    R@1={m['r1']:.3f} nDCG@10={m['ndcg10']:.3f} | {m['mean_ms']:.2f} ms")

    save_csv(OUTPUT / "phase9_beir.csv", all_rows)


# =====================================================================

if __name__ == "__main__":
    log("EXP-010 Local: Real-Data Phases")
    log(f"Output: {OUTPUT}")
    t_total = time.perf_counter()

    for name, func in [("Phase 6: MS MARCO", run_phase6),
                        ("Phase 7: SQuAD", run_phase7),
                        ("Phase 8: Multi-Model", run_phase8),
                        ("Phase 9: BEIR Suite", run_phase9)]:
        log(f"\n{'#' * 60}\n# {name}\n{'#' * 60}")
        try:
            func()
        except Exception as e:
            log(f"{name} FAILED: {e}")
            import traceback; traceback.print_exc()

    total = time.perf_counter() - t_total
    log(f"\nTotal: {total:.0f}s ({total/60:.1f}m)")
