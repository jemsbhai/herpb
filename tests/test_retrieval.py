"""
Tests for encrypted top-k retrieval.

Verifies:
  1. RPB retrieval returns same top-k as plaintext
  2. MRPB retrieval returns same top-k as plaintext
  3. Scores match within quantization bound
  4. Works at multiple precisions and dimensions
  5. Edge cases: k=1, k=num_docs, single document
  6. Corpus construction validation
"""

import math
import random
import pytest
from gmpy2 import mpz

from herpb.schemes import mrpb
from herpb.utils.encoding import dot_product_error_bound
from herpb.rag.retrieval import (
    build_rpb_corpus,
    build_mrpb_corpus,
    rpb_retrieve,
    mrpb_retrieve,
    plaintext_retrieve,
)


# ── Helpers ───────────────────────────────────────────────────────────

def random_unit_vector(dim: int, rng: random.Random) -> list[float]:
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def make_corpus(n_docs: int, dim: int, rng: random.Random) -> list[list[float]]:
    return [random_unit_vector(dim, rng) for _ in range(n_docs)]


@pytest.fixture(scope="module")
def mrpb_sk():
    _, sk = mrpb.keygen(key_bits=2048, precision=0, seed=42)
    return sk


# ── RPB: top-k correctness ───────────────────────────────────────────

class TestRPBRetrieval:

    def test_top1_matches_plaintext(self):
        rng = random.Random(10)
        docs = make_corpus(20, 8, rng)
        query = random_unit_vector(8, rng)

        plain = plaintext_retrieve(query, docs, k=1)
        corpus = build_rpb_corpus(docs, precision_bits=16)
        encrypted = rpb_retrieve(query, corpus, k=1)

        assert encrypted.indices == plain.indices

    def test_top5_matches_plaintext(self):
        rng = random.Random(20)
        docs = make_corpus(50, 16, rng)
        query = random_unit_vector(16, rng)

        plain = plaintext_retrieve(query, docs, k=5)
        corpus = build_rpb_corpus(docs, precision_bits=16)
        encrypted = rpb_retrieve(query, corpus, k=5)

        assert encrypted.indices == plain.indices

    def test_top10_matches_plaintext(self):
        rng = random.Random(30)
        docs = make_corpus(100, 32, rng)
        query = random_unit_vector(32, rng)

        plain = plaintext_retrieve(query, docs, k=10)
        corpus = build_rpb_corpus(docs, precision_bits=16)
        encrypted = rpb_retrieve(query, corpus, k=10)

        assert encrypted.indices == plain.indices

    @pytest.mark.parametrize("k_prec", [10, 12, 16, 20])
    def test_across_precisions(self, k_prec):
        rng = random.Random(40)
        docs = make_corpus(30, 16, rng)
        query = random_unit_vector(16, rng)

        plain = plaintext_retrieve(query, docs, k=5)
        corpus = build_rpb_corpus(docs, precision_bits=k_prec)
        encrypted = rpb_retrieve(query, corpus, k=5)

        assert encrypted.indices == plain.indices

    def test_k_equals_num_docs(self):
        """Retrieving all docs should return full ranking."""
        rng = random.Random(50)
        docs = make_corpus(10, 8, rng)
        query = random_unit_vector(8, rng)

        plain = plaintext_retrieve(query, docs, k=10)
        corpus = build_rpb_corpus(docs, precision_bits=16)
        encrypted = rpb_retrieve(query, corpus, k=10)

        assert encrypted.indices == plain.indices
        assert len(encrypted.indices) == 10

    def test_single_document(self):
        rng = random.Random(60)
        docs = make_corpus(1, 8, rng)
        query = random_unit_vector(8, rng)

        corpus = build_rpb_corpus(docs, precision_bits=16)
        result = rpb_retrieve(query, corpus, k=1)

        assert result.indices == [0]
        assert len(result.scores) == 1

    def test_scores_within_bound(self):
        rng = random.Random(70)
        dim = 16
        k_prec = 16
        docs = make_corpus(20, dim, rng)
        query = random_unit_vector(dim, rng)

        plain = plaintext_retrieve(query, docs, k=20)
        corpus = build_rpb_corpus(docs, precision_bits=k_prec)
        encrypted = rpb_retrieve(query, corpus, k=20)

        bound = dot_product_error_bound(k_prec, dim)
        for p_score, e_score in zip(plain.scores, encrypted.scores):
            assert abs(p_score - e_score) <= bound + 1e-10

    def test_result_metadata(self):
        rng = random.Random(80)
        docs = make_corpus(15, 8, rng)
        query = random_unit_vector(8, rng)

        corpus = build_rpb_corpus(docs, precision_bits=16)
        result = rpb_retrieve(query, corpus, k=3)

        assert result.num_documents == 15
        assert len(result.indices) == 3
        assert len(result.scores) == 3
        assert result.query_time_s > 0


# ── MRPB: top-k correctness ──────────────────────────────────────────

class TestMRPBRetrieval:

    def test_top1_matches_plaintext(self, mrpb_sk):
        rng = random.Random(110)
        docs = make_corpus(20, 8, rng)
        query = random_unit_vector(8, rng)

        plain = plaintext_retrieve(query, docs, k=1)
        corpus = build_mrpb_corpus(docs, mrpb_sk, precision_bits=16)
        encrypted = mrpb_retrieve(query, corpus, mrpb_sk, k=1)

        assert encrypted.indices == plain.indices

    def test_top5_matches_plaintext(self, mrpb_sk):
        rng = random.Random(120)
        docs = make_corpus(30, 16, rng)
        query = random_unit_vector(16, rng)

        plain = plaintext_retrieve(query, docs, k=5)
        corpus = build_mrpb_corpus(docs, mrpb_sk, precision_bits=16)
        encrypted = mrpb_retrieve(query, corpus, mrpb_sk, k=5)

        assert encrypted.indices == plain.indices

    def test_top10_dim32(self, mrpb_sk):
        rng = random.Random(130)
        docs = make_corpus(50, 32, rng)
        query = random_unit_vector(32, rng)

        plain = plaintext_retrieve(query, docs, k=10)
        corpus = build_mrpb_corpus(docs, mrpb_sk, precision_bits=16)
        encrypted = mrpb_retrieve(query, corpus, mrpb_sk, k=10)

        assert encrypted.indices == plain.indices

    def test_scores_within_bound(self, mrpb_sk):
        rng = random.Random(140)
        dim = 8
        k_prec = 16
        docs = make_corpus(10, dim, rng)
        query = random_unit_vector(dim, rng)

        plain = plaintext_retrieve(query, docs, k=10)
        corpus = build_mrpb_corpus(docs, mrpb_sk, precision_bits=k_prec)
        encrypted = mrpb_retrieve(query, corpus, mrpb_sk, k=10)

        bound = dot_product_error_bound(k_prec, dim)
        for p_score, e_score in zip(plain.scores, encrypted.scores):
            assert abs(p_score - e_score) <= bound + 1e-10


# ── Validation ────────────────────────────────────────────────────────

class TestValidation:

    def test_empty_corpus_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            build_rpb_corpus([], precision_bits=16)

    def test_dimension_mismatch_rejected(self):
        with pytest.raises(ValueError, match="mismatch"):
            build_rpb_corpus([[1.0, 2.0], [1.0]], precision_bits=16)

    def test_query_dim_mismatch(self):
        corpus = build_rpb_corpus([[1.0, 0.0], [0.0, 1.0]], precision_bits=16)
        with pytest.raises(ValueError, match="dim"):
            rpb_retrieve([1.0, 0.0, 0.0], corpus, k=1)

    def test_k_capped_at_num_docs(self):
        rng = random.Random(200)
        docs = make_corpus(5, 4, rng)
        query = random_unit_vector(4, rng)
        corpus = build_rpb_corpus(docs, precision_bits=16)
        result = rpb_retrieve(query, corpus, k=100)
        assert len(result.indices) == 5

    def test_empty_corpus_rejected_mrpb(self, mrpb_sk):
        with pytest.raises(ValueError, match="empty"):
            build_mrpb_corpus([], mrpb_sk, precision_bits=16)


# ── Parallel retrieval ────────────────────────────────────────────────

class TestParallelRetrieval:
    """Verify parallel retrieval gives identical results to serial."""

    def test_rpb_parallel_matches_serial(self):
        rng = random.Random(300)
        docs = make_corpus(30, 8, rng)
        query = random_unit_vector(8, rng)
        corpus = build_rpb_corpus(docs, precision_bits=16)

        serial = rpb_retrieve(query, corpus, k=5, num_workers=1)
        parallel = rpb_retrieve(query, corpus, k=5, num_workers=4)

        assert parallel.indices == serial.indices
        for s, p in zip(serial.scores, parallel.scores):
            assert abs(s - p) < 1e-12

    def test_rpb_parallel_top10(self):
        rng = random.Random(310)
        docs = make_corpus(50, 16, rng)
        query = random_unit_vector(16, rng)

        plain = plaintext_retrieve(query, docs, k=10)
        corpus = build_rpb_corpus(docs, precision_bits=16)
        parallel = rpb_retrieve(query, corpus, k=10, num_workers=4)

        assert parallel.indices == plain.indices
        assert parallel.num_workers == 4

    def test_mrpb_parallel_matches_serial(self, mrpb_sk):
        rng = random.Random(320)
        docs = make_corpus(20, 8, rng)
        query = random_unit_vector(8, rng)
        corpus = build_mrpb_corpus(docs, mrpb_sk, precision_bits=16)

        serial = mrpb_retrieve(query, corpus, mrpb_sk, k=5, num_workers=1)
        parallel = mrpb_retrieve(query, corpus, mrpb_sk, k=5, num_workers=4)

        assert parallel.indices == serial.indices
        for s, p in zip(serial.scores, parallel.scores):
            assert abs(s - p) < 1e-12

    def test_parallel_with_2_workers(self):
        rng = random.Random(330)
        docs = make_corpus(10, 8, rng)
        query = random_unit_vector(8, rng)

        plain = plaintext_retrieve(query, docs, k=5)
        corpus = build_rpb_corpus(docs, precision_bits=16)
        result = rpb_retrieve(query, corpus, k=5, num_workers=2)

        assert result.indices == plain.indices
