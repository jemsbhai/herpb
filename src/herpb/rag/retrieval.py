"""
Encrypted top-k retrieval for RAG pipelines.

Protocol (client-side ranking, standard HE retrieval model):

  OFFLINE (one-time):
    1. Data owner encodes document embeddings as rationals
    2. Data owner encrypts each document vector → EncryptedCorpus
    3. EncryptedCorpus is sent to the server

  ONLINE (per query):
    4. Client encodes and encrypts query vector
    5. Client sends encrypted query to server
    6. Server computes encrypted dot product for every document
    7. Server returns all encrypted scores to client
    8. Client decrypts scores, ranks, selects top-k

  PARALLELISM (Level 1 — CPU multiprocessing):
    - Each document's dot product is independent → embarrassingly parallel
    - Pass num_workers > 1 to retrieve functions for automatic parallelism
    - Query ciphertexts shared via process initializer (pickled once)
    - Each worker receives one document's ciphertexts per task
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

from gmpy2 import mpz
import gmpy2

from ..schemes import rpb, mrpb
from ..schemes.rpb import RPBCiphertext
from ..schemes.mrpb import MRPBCiphertext, MRPBSecretKey
from ..utils.encoding import encode_vector
from .similarity import (
    rpb_encrypt_vector,
    rpb_dot_product,
    mrpb_encrypt_vector,
    mrpb_dot_product,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Result of a top-k retrieval query."""
    indices: list[int]
    scores: list[float]
    query_time_s: float
    num_documents: int
    num_workers: int = 1


@dataclass
class RPBCorpus:
    """Pre-encrypted document corpus for RPB retrieval."""
    doc_ciphertexts: list[list[RPBCiphertext]]
    precision_bits: int
    dimension: int

    @property
    def num_documents(self) -> int:
        return len(self.doc_ciphertexts)


@dataclass
class MRPBCorpus:
    """Pre-encrypted document corpus for MRPB retrieval."""
    doc_ciphertexts: list[list[MRPBCiphertext]]
    precision_bits: int
    dimension: int
    n: mpz

    @property
    def num_documents(self) -> int:
        return len(self.doc_ciphertexts)


# ---------------------------------------------------------------------------
# Multiprocessing workers (module-level for pickling)
# ---------------------------------------------------------------------------

_worker_state = {}


def _rpb_worker_init(q_cts):
    """Initialize RPB worker with shared query ciphertexts."""
    _worker_state["q_cts"] = q_cts


def _rpb_dot_worker(d_cts):
    """Compute one RPB dot product. Returns float score."""
    q_cts = _worker_state["q_cts"]
    dot_ct = rpb_dot_product(q_cts, d_cts)
    num, den = rpb.decrypt(dot_ct)
    return float(num) / float(den)


def _mrpb_worker_init(q_cts, sk):
    """Initialize MRPB worker with shared query ciphertexts and secret key."""
    _worker_state["q_cts"] = q_cts
    _worker_state["sk"] = sk


def _mrpb_dot_worker(d_cts):
    """Compute one MRPB dot product. Returns float score."""
    q_cts = _worker_state["q_cts"]
    sk = _worker_state["sk"]
    dot_ct = mrpb_dot_product(q_cts, d_cts)
    num, den = mrpb.decrypt(dot_ct, sk)
    return float(num) / float(den)


# ---------------------------------------------------------------------------
# Corpus construction (offline phase)
# ---------------------------------------------------------------------------

def build_rpb_corpus(
    doc_embeddings: list[list[float]],
    precision_bits: int = 16,
    c: mpz | None = None,
) -> RPBCorpus:
    """Encode and encrypt a document corpus for RPB retrieval.

    This is the OFFLINE phase — done once, result stored on server.
    """
    if not doc_embeddings:
        raise ValueError("Cannot build corpus from empty document list")

    dim = len(doc_embeddings[0])

    if c is None:
        rs = gmpy2.random_state()
        c = gmpy2.mpz_urandomb(rs, 256)
        while c == 0:
            c = gmpy2.mpz_urandomb(rs, 256)

    doc_cts = []
    for doc in doc_embeddings:
        if len(doc) != dim:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {len(doc)}")
        rats = encode_vector(doc, precision_bits)
        cts = rpb_encrypt_vector(rats, c)
        doc_cts.append(cts)

    return RPBCorpus(
        doc_ciphertexts=doc_cts,
        precision_bits=precision_bits,
        dimension=dim,
    )


def build_mrpb_corpus(
    doc_embeddings: list[list[float]],
    sk: MRPBSecretKey,
    precision_bits: int = 16,
) -> MRPBCorpus:
    """Encode and encrypt a document corpus for MRPB retrieval."""
    if not doc_embeddings:
        raise ValueError("Cannot build corpus from empty document list")

    dim = len(doc_embeddings[0])
    doc_cts = []

    for doc in doc_embeddings:
        if len(doc) != dim:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {len(doc)}")
        rats = encode_vector(doc, precision_bits)
        cts = mrpb_encrypt_vector(rats, sk)
        doc_cts.append(cts)

    return MRPBCorpus(
        doc_ciphertexts=doc_cts,
        precision_bits=precision_bits,
        dimension=dim,
        n=sk.n,
    )


# ---------------------------------------------------------------------------
# RPB retrieval (serial + parallel)
# ---------------------------------------------------------------------------

def rpb_retrieve(
    query: list[float],
    corpus: RPBCorpus,
    k: int,
    c: mpz | None = None,
    num_workers: int = 1,
) -> RetrievalResult:
    """Retrieve top-k documents from an RPB-encrypted corpus.

    Args:
        query: Query embedding (float vector).
        corpus: Pre-encrypted document corpus.
        k: Number of results to return.
        c: Blinding factor for query encryption (random if None).
        num_workers: Number of parallel workers (1 = serial).
            Set to os.cpu_count() for max parallelism.

    Returns:
        RetrievalResult with top-k indices and scores.
    """
    if len(query) != corpus.dimension:
        raise ValueError(
            f"Query dim {len(query)} != corpus dim {corpus.dimension}"
        )
    k = min(k, corpus.num_documents)

    if c is None:
        rs = gmpy2.random_state()
        c = gmpy2.mpz_urandomb(rs, 256)
        while c == 0:
            c = gmpy2.mpz_urandomb(rs, 256)

    t0 = time.perf_counter()

    # Encode and encrypt query (once)
    q_rats = encode_vector(query, corpus.precision_bits)
    q_cts = rpb_encrypt_vector(q_rats, c)

    # Compute dot product with every document
    if num_workers > 1:
        actual_workers = min(num_workers, corpus.num_documents)
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            initializer=_rpb_worker_init,
            initargs=(q_cts,),
        ) as pool:
            scores = list(pool.map(_rpb_dot_worker, corpus.doc_ciphertexts))
    else:
        scores = []
        for doc_cts in corpus.doc_ciphertexts:
            dot_ct = rpb_dot_product(q_cts, doc_cts)
            num, den = rpb.decrypt(dot_ct)
            scores.append(float(num) / float(den))

    elapsed = time.perf_counter() - t0

    # Rank and select top-k
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_k_indices = ranked[:k]
    top_k_scores = [scores[i] for i in top_k_indices]

    return RetrievalResult(
        indices=top_k_indices,
        scores=top_k_scores,
        query_time_s=elapsed,
        num_documents=corpus.num_documents,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# MRPB retrieval (serial + parallel)
# ---------------------------------------------------------------------------

def mrpb_retrieve(
    query: list[float],
    corpus: MRPBCorpus,
    sk: MRPBSecretKey,
    k: int,
    num_workers: int = 1,
) -> RetrievalResult:
    """Retrieve top-k documents from an MRPB-encrypted corpus.

    Args:
        query: Query embedding (float vector).
        corpus: Pre-encrypted MRPB corpus.
        sk: Secret key (held by client only).
        k: Number of results to return.
        num_workers: Number of parallel workers (1 = serial).

    Returns:
        RetrievalResult with top-k indices and scores.
    """
    if len(query) != corpus.dimension:
        raise ValueError(
            f"Query dim {len(query)} != corpus dim {corpus.dimension}"
        )
    k = min(k, corpus.num_documents)

    t0 = time.perf_counter()

    q_rats = encode_vector(query, corpus.precision_bits)
    q_cts = mrpb_encrypt_vector(q_rats, sk)

    if num_workers > 1:
        actual_workers = min(num_workers, corpus.num_documents)
        with ProcessPoolExecutor(
            max_workers=actual_workers,
            initializer=_mrpb_worker_init,
            initargs=(q_cts, sk),
        ) as pool:
            scores = list(pool.map(_mrpb_dot_worker, corpus.doc_ciphertexts))
    else:
        scores = []
        for doc_cts in corpus.doc_ciphertexts:
            dot_ct = mrpb_dot_product(q_cts, doc_cts)
            num, den = mrpb.decrypt(dot_ct, sk)
            scores.append(float(num) / float(den))

    elapsed = time.perf_counter() - t0

    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_k_indices = ranked[:k]
    top_k_scores = [scores[i] for i in top_k_indices]

    return RetrievalResult(
        indices=top_k_indices,
        scores=top_k_scores,
        query_time_s=elapsed,
        num_documents=corpus.num_documents,
        num_workers=num_workers,
    )


# ---------------------------------------------------------------------------
# Plaintext retrieval (baseline)
# ---------------------------------------------------------------------------

def plaintext_retrieve(
    query: list[float],
    doc_embeddings: list[list[float]],
    k: int,
) -> RetrievalResult:
    """Plaintext top-k retrieval (dot-product ranking). Baseline."""
    t0 = time.perf_counter()

    scores = [sum(q * d for q, d in zip(query, doc)) for doc in doc_embeddings]

    elapsed = time.perf_counter() - t0

    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_k_indices = ranked[:k]
    top_k_scores = [scores[i] for i in top_k_indices]

    return RetrievalResult(
        indices=top_k_indices,
        scores=top_k_scores,
        query_time_s=elapsed,
        num_documents=len(doc_embeddings),
    )
