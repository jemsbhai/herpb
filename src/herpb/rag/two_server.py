"""
Two-server encrypted retrieval for RAG pipelines.

Provides two deployment models:

  QUERY-PRIVATE (no Beaver triples, zero inter-server communication):
    Both servers hold plaintext document embeddings.
    Client's query is additively shared across the two servers.
    Each server computes its share of every dot product locally.
    Client reconstructs scores, ranks, returns top-k.
    Information-theoretic query privacy: each server's view is
    independent of the query regardless of computational power.

  MUTUAL-PRIVACY (Beaver triples, one round of inter-server communication):
    Both query AND documents are additively shared.
    Pre-generated Beaver triples enable secure multiplication.
    Neither server learns query or document contents.
    One communication round per query (2d values per document).

Both models guarantee exact arithmetic: the only error is the initial
fixed-point quantization, which does not grow with computation depth.
"""

import time
from dataclasses import dataclass

import gmpy2
from gmpy2 import mpz

from ..utils.encoding import encode_vector
from ..schemes.secret_sharing import (
    share_vector,
    reconstruct,
    query_private_dot,
    mutual_privacy_dot,
    generate_beaver_triples,
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
    mode: str  # "query_private" or "mutual_privacy"


@dataclass
class QueryPrivateCorpus:
    """Corpus for query-private retrieval.

    Documents are stored as plaintext integer numerators on both servers.
    Only the query is secret-shared.
    """
    doc_numerators: list[list[int]]  # [n_docs][dim] integer numerators
    precision_bits: int
    dimension: int
    n: mpz  # modulus

    @property
    def num_documents(self) -> int:
        return len(self.doc_numerators)


@dataclass
class MutualPrivacyCorpus:
    """Corpus for mutual-privacy retrieval.

    Documents are additively shared across two servers.
    Beaver triples are pre-generated for each document.
    """
    doc_shares1: list[list[mpz]]  # Server 1's shares [n_docs][dim]
    doc_shares2: list[list[mpz]]  # Server 2's shares [n_docs][dim]
    beaver_triples_per_doc: list   # [n_docs] lists of BeaverTriples
    precision_bits: int
    dimension: int
    n: mpz

    @property
    def num_documents(self) -> int:
        return len(self.doc_shares1)


# ---------------------------------------------------------------------------
# Modulus generation
# ---------------------------------------------------------------------------

def generate_modulus(bits: int = 2048, seed: int = 42) -> mpz:
    """Generate an RSA-size modulus n = p*q for use as the sharing modulus.

    The factorization is not needed (no trapdoor). We use a composite
    modulus for consistency with standard parameter sizes. Any large
    integer would work; the security is information-theoretic.
    """
    rs = gmpy2.random_state(seed)
    half = bits // 2
    p = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half) | (mpz(1) << (half - 1)))
    q = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half) | (mpz(1) << (half - 1)))
    return p * q


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

def build_query_private_corpus(
    doc_embeddings: list[list[float]],
    precision_bits: int = 16,
    n: mpz = None,
) -> QueryPrivateCorpus:
    """Encode documents as integer numerators for query-private retrieval.

    No encryption needed: documents stay plaintext on both servers.
    """
    if not doc_embeddings:
        raise ValueError("Cannot build corpus from empty document list")

    dim = len(doc_embeddings[0])
    if n is None:
        n = generate_modulus()

    doc_nums = []
    for doc in doc_embeddings:
        if len(doc) != dim:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {len(doc)}")
        rats = encode_vector(doc, precision_bits)
        nums = [num for num, _ in rats]
        doc_nums.append(nums)

    return QueryPrivateCorpus(
        doc_numerators=doc_nums,
        precision_bits=precision_bits,
        dimension=dim,
        n=n,
    )


def build_mutual_privacy_corpus(
    doc_embeddings: list[list[float]],
    precision_bits: int = 16,
    n: mpz = None,
    seed: int = None,
) -> MutualPrivacyCorpus:
    """Encode and secret-share documents for mutual-privacy retrieval.

    Each document vector is encoded as integer numerators, then
    additively shared. Beaver triples are pre-generated per document.
    """
    if not doc_embeddings:
        raise ValueError("Cannot build corpus from empty document list")

    dim = len(doc_embeddings[0])
    if n is None:
        n = generate_modulus()

    doc_s1, doc_s2, all_triples = [], [], []
    base_seed = seed if seed is not None else 0

    for j, doc in enumerate(doc_embeddings):
        if len(doc) != dim:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {len(doc)}")
        rats = encode_vector(doc, precision_bits)
        nums = [num for num, _ in rats]
        s1, s2 = share_vector(nums, n, seed=base_seed + j * 2)
        triples = generate_beaver_triples(dim, n, seed=base_seed + j * 2 + 1)
        doc_s1.append(s1)
        doc_s2.append(s2)
        all_triples.append(triples)

    return MutualPrivacyCorpus(
        doc_shares1=doc_s1,
        doc_shares2=doc_s2,
        beaver_triples_per_doc=all_triples,
        precision_bits=precision_bits,
        dimension=dim,
        n=n,
    )


# ---------------------------------------------------------------------------
# Retrieval: query-private
# ---------------------------------------------------------------------------

def retrieve_query_private(
    query: list[float],
    corpus: QueryPrivateCorpus,
    k: int,
    seed: int = None,
) -> RetrievalResult:
    """Retrieve top-k documents with query privacy.

    No inter-server communication. Each server independently computes
    its share of every dot product.

    Args:
        query: Query embedding (float vector).
        corpus: Pre-encoded document corpus (plaintext on both servers).
        k: Number of results to return.
        seed: Random seed for share generation.

    Returns:
        RetrievalResult with top-k indices and decoded float scores.
    """
    if len(query) != corpus.dimension:
        raise ValueError(
            f"Query dim {len(query)} != corpus dim {corpus.dimension}"
        )
    k = min(k, corpus.num_documents)

    t0 = time.perf_counter()

    # Encode query as integer numerators
    q_rats = encode_vector(query, corpus.precision_bits)
    q_ints = [num for num, _ in q_rats]

    # Split query into additive shares
    q1, q2 = share_vector(q_ints, corpus.n, seed=seed)

    # Each server computes its dot product share for every document
    scores = []
    denom = (1 << corpus.precision_bits) ** 2  # 2^{2k}

    for doc_nums in corpus.doc_numerators:
        dot1, dot2 = query_private_dot(q1, q2, doc_nums, corpus.n)
        dot_int = int(reconstruct(dot1, dot2, corpus.n))
        scores.append(dot_int / denom)

    elapsed = time.perf_counter() - t0

    # Rank and return top-k
    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_k_idx = ranked[:k]
    top_k_scores = [scores[i] for i in top_k_idx]

    return RetrievalResult(
        indices=top_k_idx,
        scores=top_k_scores,
        query_time_s=elapsed,
        num_documents=corpus.num_documents,
        mode="query_private",
    )


# ---------------------------------------------------------------------------
# Retrieval: mutual privacy
# ---------------------------------------------------------------------------

def retrieve_mutual_privacy(
    query: list[float],
    corpus: MutualPrivacyCorpus,
    k: int,
    seed: int = None,
) -> RetrievalResult:
    """Retrieve top-k documents with mutual privacy.

    Both query and documents are secret-shared. Uses Beaver triples
    for secure multiplication. One round of inter-server communication
    per query (2*dim values exchanged per document).

    Args:
        query: Query embedding (float vector).
        corpus: Secret-shared document corpus with pre-generated triples.
        k: Number of results to return.
        seed: Random seed for query share generation.

    Returns:
        RetrievalResult with top-k indices and decoded float scores.
    """
    if len(query) != corpus.dimension:
        raise ValueError(
            f"Query dim {len(query)} != corpus dim {corpus.dimension}"
        )
    k = min(k, corpus.num_documents)

    t0 = time.perf_counter()

    # Encode and share the query
    q_rats = encode_vector(query, corpus.precision_bits)
    q_ints = [num for num, _ in q_rats]
    q1, q2 = share_vector(q_ints, corpus.n, seed=seed)

    # Compute dot product with each document using Beaver triples
    scores = []
    denom = (1 << corpus.precision_bits) ** 2

    for j in range(corpus.num_documents):
        dot1, dot2 = mutual_privacy_dot(
            q1, q2,
            corpus.doc_shares1[j], corpus.doc_shares2[j],
            corpus.beaver_triples_per_doc[j],
            corpus.n,
        )
        dot_int = int(reconstruct(dot1, dot2, corpus.n))
        scores.append(dot_int / denom)

    elapsed = time.perf_counter() - t0

    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_k_idx = ranked[:k]
    top_k_scores = [scores[i] for i in top_k_idx]

    return RetrievalResult(
        indices=top_k_idx,
        scores=top_k_scores,
        query_time_s=elapsed,
        num_documents=corpus.num_documents,
        mode="mutual_privacy",
    )


# ---------------------------------------------------------------------------
# Plaintext baseline (for verification)
# ---------------------------------------------------------------------------

def retrieve_plaintext(
    query: list[float],
    doc_embeddings: list[list[float]],
    k: int,
) -> RetrievalResult:
    """Plaintext top-k retrieval. Baseline for correctness verification."""
    t0 = time.perf_counter()
    scores = [sum(q * d for q, d in zip(query, doc)) for doc in doc_embeddings]
    elapsed = time.perf_counter() - t0

    ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
    top_k_idx = ranked[:k]
    top_k_scores = [scores[i] for i in top_k_idx]

    return RetrievalResult(
        indices=top_k_idx,
        scores=top_k_scores,
        query_time_s=elapsed,
        num_documents=len(doc_embeddings),
        mode="plaintext",
    )
