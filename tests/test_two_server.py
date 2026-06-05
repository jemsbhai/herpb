"""
Tests for two-server encrypted retrieval.

Verifies end-to-end: float query in, ranked indices out, matching
plaintext retrieval exactly.
"""

import math
import random
import pytest
from gmpy2 import mpz

from herpb.utils.encoding import dot_product_error_bound
from herpb.rag.two_server import (
    generate_modulus,
    build_query_private_corpus,
    build_mutual_privacy_corpus,
    retrieve_query_private,
    retrieve_mutual_privacy,
    retrieve_plaintext,
)


# ── Helpers ───────────────────────────────────────────────────────────

def random_unit_vector(dim, rng):
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def make_corpus(n_docs, dim, rng):
    return [random_unit_vector(dim, rng) for _ in range(n_docs)]


@pytest.fixture(scope="module")
def n_mod():
    return generate_modulus(2048, seed=42)


# ── Query-private retrieval ──────────────────────────────────────────

class TestQueryPrivateRetrieval:

    def test_top1(self, n_mod):
        rng = random.Random(10)
        docs = make_corpus(20, 8, rng)
        query = random_unit_vector(8, rng)

        plain = retrieve_plaintext(query, docs, k=1)
        corpus = build_query_private_corpus(docs, precision_bits=16, n=n_mod)
        secure = retrieve_query_private(query, corpus, k=1, seed=10)

        assert secure.indices == plain.indices
        assert secure.mode == "query_private"

    def test_top5(self, n_mod):
        rng = random.Random(20)
        docs = make_corpus(50, 16, rng)
        query = random_unit_vector(16, rng)

        plain = retrieve_plaintext(query, docs, k=5)
        corpus = build_query_private_corpus(docs, precision_bits=16, n=n_mod)
        secure = retrieve_query_private(query, corpus, k=5, seed=20)

        assert secure.indices == plain.indices

    def test_top10_dim32(self, n_mod):
        rng = random.Random(30)
        docs = make_corpus(100, 32, rng)
        query = random_unit_vector(32, rng)

        plain = retrieve_plaintext(query, docs, k=10)
        corpus = build_query_private_corpus(docs, precision_bits=16, n=n_mod)
        secure = retrieve_query_private(query, corpus, k=10, seed=30)

        assert secure.indices == plain.indices

    @pytest.mark.parametrize("k_prec", [10, 12, 16, 20])
    def test_across_precisions(self, n_mod, k_prec):
        rng = random.Random(40)
        docs = make_corpus(30, 16, rng)
        query = random_unit_vector(16, rng)

        plain = retrieve_plaintext(query, docs, k=5)
        corpus = build_query_private_corpus(docs, precision_bits=k_prec, n=n_mod)
        secure = retrieve_query_private(query, corpus, k=5, seed=40)

        assert secure.indices == plain.indices

    def test_scores_within_error_bound(self, n_mod):
        dim, k = 16, 16
        rng = random.Random(50)
        docs = make_corpus(20, dim, rng)
        query = random_unit_vector(dim, rng)

        plain = retrieve_plaintext(query, docs, k=20)
        corpus = build_query_private_corpus(docs, precision_bits=k, n=n_mod)
        secure = retrieve_query_private(query, corpus, k=20, seed=50)

        bound = dot_product_error_bound(k, dim)
        for ps, ss in zip(plain.scores, secure.scores):
            assert abs(ps - ss) <= bound + 1e-10

    def test_full_ranking(self, n_mod):
        rng = random.Random(60)
        docs = make_corpus(10, 8, rng)
        query = random_unit_vector(8, rng)

        plain = retrieve_plaintext(query, docs, k=10)
        corpus = build_query_private_corpus(docs, precision_bits=16, n=n_mod)
        secure = retrieve_query_private(query, corpus, k=10, seed=60)

        assert secure.indices == plain.indices
        assert len(secure.indices) == 10

    def test_single_document(self, n_mod):
        rng = random.Random(70)
        docs = make_corpus(1, 8, rng)
        query = random_unit_vector(8, rng)

        corpus = build_query_private_corpus(docs, precision_bits=16, n=n_mod)
        result = retrieve_query_private(query, corpus, k=1, seed=70)

        assert result.indices == [0]


# ── Mutual-privacy retrieval ─────────────────────────────────────────

class TestMutualPrivacyRetrieval:

    def test_top1(self, n_mod):
        rng = random.Random(110)
        docs = make_corpus(10, 8, rng)
        query = random_unit_vector(8, rng)

        plain = retrieve_plaintext(query, docs, k=1)
        corpus = build_mutual_privacy_corpus(docs, precision_bits=16, n=n_mod, seed=110)
        secure = retrieve_mutual_privacy(query, corpus, k=1, seed=111)

        assert secure.indices == plain.indices
        assert secure.mode == "mutual_privacy"

    def test_top5_dim16(self, n_mod):
        rng = random.Random(120)
        docs = make_corpus(20, 16, rng)
        query = random_unit_vector(16, rng)

        plain = retrieve_plaintext(query, docs, k=5)
        corpus = build_mutual_privacy_corpus(docs, precision_bits=16, n=n_mod, seed=120)
        secure = retrieve_mutual_privacy(query, corpus, k=5, seed=121)

        assert secure.indices == plain.indices

    def test_ranking_matches_query_private(self, n_mod):
        """Both protocols should give identical rankings."""
        rng = random.Random(130)
        docs = make_corpus(15, 8, rng)
        query = random_unit_vector(8, rng)

        qp_corpus = build_query_private_corpus(docs, precision_bits=16, n=n_mod)
        mp_corpus = build_mutual_privacy_corpus(docs, precision_bits=16, n=n_mod, seed=130)

        qp_result = retrieve_query_private(query, qp_corpus, k=10, seed=131)
        mp_result = retrieve_mutual_privacy(query, mp_corpus, k=10, seed=132)

        assert qp_result.indices == mp_result.indices

    def test_scores_within_error_bound(self, n_mod):
        dim, k = 8, 16
        rng = random.Random(140)
        docs = make_corpus(10, dim, rng)
        query = random_unit_vector(dim, rng)

        plain = retrieve_plaintext(query, docs, k=10)
        corpus = build_mutual_privacy_corpus(docs, precision_bits=k, n=n_mod, seed=140)
        secure = retrieve_mutual_privacy(query, corpus, k=10, seed=141)

        bound = dot_product_error_bound(k, dim)
        for ps, ss in zip(plain.scores, secure.scores):
            assert abs(ps - ss) <= bound + 1e-10


# ── Validation ────────────────────────────────────────────────────────

class TestValidation:

    def test_empty_corpus_rejected_qp(self, n_mod):
        with pytest.raises(ValueError, match="empty"):
            build_query_private_corpus([], precision_bits=16, n=n_mod)

    def test_empty_corpus_rejected_mp(self, n_mod):
        with pytest.raises(ValueError, match="empty"):
            build_mutual_privacy_corpus([], precision_bits=16, n=n_mod)

    def test_dim_mismatch_qp(self, n_mod):
        with pytest.raises(ValueError, match="mismatch"):
            build_query_private_corpus([[1.0, 2.0], [1.0]], precision_bits=16, n=n_mod)

    def test_query_dim_mismatch(self, n_mod):
        corpus = build_query_private_corpus([[1.0, 0.0], [0.0, 1.0]],
                                            precision_bits=16, n=n_mod)
        with pytest.raises(ValueError, match="dim"):
            retrieve_query_private([1.0, 0.0, 0.0], corpus, k=1)

    def test_k_capped(self, n_mod):
        rng = random.Random(200)
        docs = make_corpus(5, 4, rng)
        query = random_unit_vector(4, rng)
        corpus = build_query_private_corpus(docs, precision_bits=16, n=n_mod)
        result = retrieve_query_private(query, corpus, k=100, seed=200)
        assert len(result.indices) == 5
