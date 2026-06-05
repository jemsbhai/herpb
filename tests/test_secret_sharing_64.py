"""
Tests for 64-bit two-server additive secret sharing with
matrix-vector Beaver triples.

Covers: share/reconstruct, homomorphic ops, query-private dot,
mutual-privacy dot, ranking preservation, matrix Beaver correctness.
"""

import math
import random
import pytest
import numpy as np

from herpb.schemes.secret_sharing_64 import (
    to_uint64, to_signed,
    share_scalar, reconstruct_scalar,
    share_vector, reconstruct_vector,
    share_matrix,
    query_private_dot, query_private_batch,
    setup_matrix_beaver, generate_query_triple,
    mutual_privacy_batch,
)
from herpb.utils.encoding import encode_vector


# ── Helpers ───────────────────────────────────────────────────────────

def random_unit_vector(dim, rng):
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def true_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


# ── Two's complement ─────────────────────────────────────────────────

class TestTwosComplement:

    def test_positive(self):
        assert to_signed(to_uint64(42)) == 42

    def test_negative(self):
        assert to_signed(to_uint64(-42)) == -42

    def test_zero(self):
        assert to_signed(to_uint64(0)) == 0

    def test_large_positive(self):
        assert to_signed(to_uint64(2**40)) == 2**40

    def test_large_negative(self):
        assert to_signed(to_uint64(-(2**40))) == -(2**40)


# ── Share / Reconstruct ──────────────────────────────────────────────

class TestShareReconstruct:

    def test_scalar_positive(self):
        for v in [0, 1, 42, 65536, 2**40]:
            s1, s2 = share_scalar(v)
            assert reconstruct_scalar(s1, s2) == v

    def test_scalar_negative(self):
        for v in [-1, -42, -65536, -(2**40)]:
            s1, s2 = share_scalar(v)
            assert reconstruct_scalar(s1, s2) == v

    def test_vector(self):
        vals = [100, -200, 0, 300, -50]
        s1, s2 = share_vector(vals)
        result = reconstruct_vector(s1, s2)
        np.testing.assert_array_equal(result, vals)

    def test_shares_look_random(self):
        s1, s2 = share_scalar(42)
        assert s1 > 2**32, "Share should be a large random uint64"


# ── Homomorphic ops (free) ───────────────────────────────────────────

class TestHomomorphicOps:

    def test_addition(self):
        a, b = 42, 58
        a1, a2 = share_scalar(a)
        b1, b2 = share_scalar(b)
        assert reconstruct_scalar(a1 + b1, a2 + b2) == a + b

    def test_addition_negative(self):
        a, b = 100, -250
        a1, a2 = share_scalar(a)
        b1, b2 = share_scalar(b)
        assert reconstruct_scalar(a1 + b1, a2 + b2) == a + b

    def test_scalar_mul(self):
        m, k = 100, np.uint64(7)
        s1, s2 = share_scalar(m)
        assert reconstruct_scalar(s1 * k, s2 * k) == m * int(k)


# ── Query-private dot product ────────────────────────────────────────

class TestQueryPrivateDot:

    def test_basic(self):
        q = [3, -5, 7]
        d = [2, 4, -1]
        q1, q2 = share_vector(q)
        d_arr = np.array(d, dtype=np.int64)
        dot1, dot2 = query_private_dot(q1, q2, d_arr)
        assert reconstruct_scalar(dot1, dot2) == sum(a * b for a, b in zip(q, d))

    def test_encoded_vectors(self):
        k = 16
        q_f = [0.5, -0.3, 0.7, 0.1]
        d_f = [0.2, 0.8, -0.1, 0.4]
        q_ints = [n for n, _ in encode_vector(q_f, k)]
        d_ints = [n for n, _ in encode_vector(d_f, k)]

        q1, q2 = share_vector(q_ints)
        d_arr = np.array(d_ints, dtype=np.int64)
        dot1, dot2 = query_private_dot(q1, q2, d_arr)
        result = reconstruct_scalar(dot1, dot2)

        expected = sum(qn * dn for qn, dn in zip(q_ints, d_ints))
        assert result == expected

    @pytest.mark.parametrize("dim", [8, 64, 384])
    def test_random_unit_vectors(self, dim):
        k = 16
        rng = random.Random(dim)
        q_f = random_unit_vector(dim, rng)
        d_f = random_unit_vector(dim, rng)
        q_ints = [n for n, _ in encode_vector(q_f, k)]
        d_ints = [n for n, _ in encode_vector(d_f, k)]

        q1, q2 = share_vector(q_ints)
        d_arr = np.array(d_ints, dtype=np.int64)
        dot1, dot2 = query_private_dot(q1, q2, d_arr)
        result = reconstruct_scalar(dot1, dot2)

        expected = sum(qn * dn for qn, dn in zip(q_ints, d_ints))
        assert result == expected


# ── Query-private batch ──────────────────────────────────────────────

class TestQueryPrivateBatch:

    def test_batch_matches_individual(self):
        k = 16
        dim = 32
        n_docs = 10
        rng = random.Random(42)
        q_f = random_unit_vector(dim, rng)
        docs_f = [random_unit_vector(dim, rng) for _ in range(n_docs)]

        q_ints = [n for n, _ in encode_vector(q_f, k)]
        doc_ints = [[n for n, _ in encode_vector(d, k)] for d in docs_f]
        doc_matrix = np.array(doc_ints, dtype=np.int64)

        q1, q2 = share_vector(q_ints)

        # Batch
        dots1, dots2 = query_private_batch(q1, q2, doc_matrix)

        # Individual
        for j in range(n_docs):
            d1, d2 = query_private_dot(q1, q2, doc_matrix[j])
            batch_result = reconstruct_scalar(
                np.uint64(dots1[j]), np.uint64(dots2[j])
            )
            indiv_result = reconstruct_scalar(d1, d2)
            assert batch_result == indiv_result

    def test_dim384_n200(self):
        """Full-scale test at real RAG dimensions."""
        k = 16
        dim = 384
        n_docs = 200
        np_rng = np.random.default_rng(99)
        vecs = np_rng.standard_normal((n_docs + 1, dim))
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        q_f = vecs[0].tolist()
        docs_f = [vecs[i].tolist() for i in range(1, n_docs + 1)]

        q_ints = [n for n, _ in encode_vector(q_f, k)]
        doc_ints = [[n for n, _ in encode_vector(d, k)] for d in docs_f]
        doc_matrix = np.array(doc_ints, dtype=np.int64)

        q1, q2 = share_vector(q_ints)
        dots1, dots2 = query_private_batch(q1, q2, doc_matrix)

        # Verify all dot products
        for j in range(n_docs):
            result = reconstruct_scalar(np.uint64(dots1[j]), np.uint64(dots2[j]))
            expected = sum(qi * di for qi, di in zip(q_ints, doc_ints[j]))
            assert result == expected


# ── Matrix-vector Beaver triples (mutual privacy) ────────────────────

class TestMatrixBeaver:

    def test_basic_3x2(self):
        """Tiny test: 3 docs, dim 2."""
        q = [3, -5]
        D = [[2, 4], [-1, 3], [7, -2]]
        expected = [2*3 + 4*(-5), (-1)*3 + 3*(-5), 7*3 + (-2)*(-5)]

        rng = np.random.default_rng(42)
        D_arr = np.array(D, dtype=np.int64)
        D1, D2 = share_matrix(D_arr, rng)

        setup = setup_matrix_beaver(D1, D2, rng)
        triple = generate_query_triple(setup, rng)

        q1, q2 = share_vector(q, rng)
        Z1, Z2 = mutual_privacy_batch(q1, q2, setup, triple)

        for j in range(3):
            result = reconstruct_scalar(np.uint64(Z1[j]), np.uint64(Z2[j]))
            assert result == expected[j], f"Doc {j}: got {result}, expected {expected[j]}"

    def test_encoded_dim8(self):
        k = 16
        dim = 8
        n_docs = 5
        rng_py = random.Random(55)
        rng_np = np.random.default_rng(55)

        q_f = random_unit_vector(dim, rng_py)
        docs_f = [random_unit_vector(dim, rng_py) for _ in range(n_docs)]

        q_ints = [n for n, _ in encode_vector(q_f, k)]
        doc_ints = [[n for n, _ in encode_vector(d, k)] for d in docs_f]
        D_arr = np.array(doc_ints, dtype=np.int64)

        D1, D2 = share_matrix(D_arr, rng_np)
        setup = setup_matrix_beaver(D1, D2, rng_np)
        triple = generate_query_triple(setup, rng_np)

        q1, q2 = share_vector(q_ints, rng_np)
        Z1, Z2 = mutual_privacy_batch(q1, q2, setup, triple)

        for j in range(n_docs):
            result = reconstruct_scalar(np.uint64(Z1[j]), np.uint64(Z2[j]))
            expected = sum(qi * di for qi, di in zip(q_ints, doc_ints[j]))
            assert result == expected

    def test_matches_query_private(self):
        """Mutual privacy gives same results as query-private."""
        k = 16
        dim = 32
        n_docs = 15
        rng_py = random.Random(66)
        rng_np = np.random.default_rng(66)

        q_f = random_unit_vector(dim, rng_py)
        docs_f = [random_unit_vector(dim, rng_py) for _ in range(n_docs)]

        q_ints = [n for n, _ in encode_vector(q_f, k)]
        doc_ints = [[n for n, _ in encode_vector(d, k)] for d in docs_f]
        D_arr = np.array(doc_ints, dtype=np.int64)

        # Query-private
        q1_qp, q2_qp = share_vector(q_ints, np.random.default_rng(100))
        dots1_qp, dots2_qp = query_private_batch(q1_qp, q2_qp, D_arr)

        # Mutual-privacy
        D1, D2 = share_matrix(D_arr, rng_np)
        setup = setup_matrix_beaver(D1, D2, rng_np)
        triple = generate_query_triple(setup, rng_np)
        q1_mp, q2_mp = share_vector(q_ints, np.random.default_rng(101))
        Z1, Z2 = mutual_privacy_batch(q1_mp, q2_mp, setup, triple)

        for j in range(n_docs):
            qp_result = reconstruct_scalar(np.uint64(dots1_qp[j]), np.uint64(dots2_qp[j]))
            mp_result = reconstruct_scalar(np.uint64(Z1[j]), np.uint64(Z2[j]))
            assert qp_result == mp_result, f"Mismatch at doc {j}"


# ── Ranking preservation ─────────────────────────────────────────────

class TestRankingPreservation:

    def test_query_private_ranking_dim384(self):
        k = 16
        dim = 384
        n_docs = 50
        np_rng = np.random.default_rng(77)

        vecs = np_rng.standard_normal((n_docs + 1, dim))
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        q_f = vecs[0].tolist()
        docs_f = [vecs[i].tolist() for i in range(1, n_docs + 1)]

        # True ranking
        true_scores = [true_dot(q_f, d) for d in docs_f]
        true_ranking = sorted(range(n_docs), key=lambda i: -true_scores[i])

        # Secret-shared ranking
        q_ints = [n for n, _ in encode_vector(q_f, k)]
        doc_ints = [[n for n, _ in encode_vector(d, k)] for d in docs_f]
        D_arr = np.array(doc_ints, dtype=np.int64)

        q1, q2 = share_vector(q_ints)
        dots1, dots2 = query_private_batch(q1, q2, D_arr)

        enc_scores = [
            reconstruct_scalar(np.uint64(dots1[j]), np.uint64(dots2[j]))
            for j in range(n_docs)
        ]
        enc_ranking = sorted(range(n_docs), key=lambda i: -enc_scores[i])

        assert enc_ranking == true_ranking

    def test_mutual_privacy_ranking(self):
        k = 16
        dim = 32
        n_docs = 20
        rng_py = random.Random(88)
        rng_np = np.random.default_rng(88)

        q_f = random_unit_vector(dim, rng_py)
        docs_f = [random_unit_vector(dim, rng_py) for _ in range(n_docs)]

        true_scores = [true_dot(q_f, d) for d in docs_f]
        true_ranking = sorted(range(n_docs), key=lambda i: -true_scores[i])

        q_ints = [n for n, _ in encode_vector(q_f, k)]
        doc_ints = [[n for n, _ in encode_vector(d, k)] for d in docs_f]
        D_arr = np.array(doc_ints, dtype=np.int64)

        D1, D2 = share_matrix(D_arr, rng_np)
        setup = setup_matrix_beaver(D1, D2, rng_np)
        triple = generate_query_triple(setup, rng_np)
        q1, q2 = share_vector(q_ints, rng_np)
        Z1, Z2 = mutual_privacy_batch(q1, q2, setup, triple)

        enc_scores = [
            reconstruct_scalar(np.uint64(Z1[j]), np.uint64(Z2[j]))
            for j in range(n_docs)
        ]
        enc_ranking = sorted(range(n_docs), key=lambda i: -enc_scores[i])

        assert enc_ranking == true_ranking
