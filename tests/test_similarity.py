"""
Tests for encrypted similarity computation.

Verifies:
  1. RPB encrypted dot product matches plaintext exactly (zero encryption error)
  2. MRPB encrypted dot product matches plaintext exactly
  3. Rankings are preserved after encryption (critical for RAG)
  4. Various precision levels
  5. Edge cases: orthogonal, identical, near-zero vectors
  6. Pipeline end-to-end correctness
"""

import math
import random
import pytest
from gmpy2 import mpz
import gmpy2

from herpb.schemes import rpb, mrpb
from herpb.utils.encoding import (
    encode_vector,
    quantization_error_bound,
    dot_product_error_bound,
)
from herpb.rag.similarity import (
    rpb_encrypt_scalar,
    rpb_encrypt_vector,
    rpb_dot_product,
    mrpb_encrypt_scalar,
    mrpb_encrypt_vector,
    mrpb_dot_product,
    plaintext_dot_product,
    plaintext_dot_product_exact,
    rpb_similarity_pipeline,
    mrpb_similarity_pipeline,
)


# ── Helpers ───────────────────────────────────────────────────────────

def random_unit_vector(dim: int, rng: random.Random) -> list[float]:
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def true_dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def make_rpb_blinding(bits: int = 256) -> mpz:
    rs = gmpy2.random_state(12345)
    c = gmpy2.mpz_urandomb(rs, bits)
    while c == 0:
        c = gmpy2.mpz_urandomb(rs, bits)
    return c


@pytest.fixture(scope="module")
def mrpb_keys():
    """Generate MRPB key pair once for all tests (2048-bit, ~1s)."""
    pk, sk = mrpb.keygen(key_bits=2048, precision=0, seed=42)
    return pk, sk


# ── RPB: encrypted dot product correctness ────────────────────────────

class TestRPBDotProduct:

    def test_basic_2d(self):
        """Simple 2D dot product: [1, 0] · [0, 1] = 0."""
        k = 16
        q = encode_vector([1.0, 0.0], k)
        d = encode_vector([0.0, 1.0], k)
        c = make_rpb_blinding()

        q_cts = rpb_encrypt_vector(q, c)
        d_cts = rpb_encrypt_vector(d, c)

        result_ct = rpb_dot_product(q_cts, d_cts)
        num, den = rpb.decrypt(result_ct)

        assert num == 0, f"Expected 0, got {num}/{den}"

    def test_parallel_vectors(self):
        """[1, 0] · [1, 0] = 1."""
        k = 16
        q = encode_vector([1.0, 0.0], k)
        d = encode_vector([1.0, 0.0], k)
        c = make_rpb_blinding()

        q_cts = rpb_encrypt_vector(q, c)
        d_cts = rpb_encrypt_vector(d, c)

        result_ct = rpb_dot_product(q_cts, d_cts)
        num, den = rpb.decrypt(result_ct)
        result = float(num) / float(den)

        assert abs(result - 1.0) < 1e-10

    @pytest.mark.parametrize("k", [8, 12, 16, 20])
    def test_random_4d(self, k):
        """Random 4D vectors: encryption error must be zero."""
        rng = random.Random(42)
        q_f = random_unit_vector(4, rng)
        d_f = random_unit_vector(4, rng)

        q_rats = encode_vector(q_f, k)
        d_rats = encode_vector(d_f, k)
        c = make_rpb_blinding()

        # Plaintext encoded dot product
        expected = plaintext_dot_product(q_rats, d_rats)

        # Encrypted dot product
        q_cts = rpb_encrypt_vector(q_rats, c)
        d_cts = rpb_encrypt_vector(d_rats, c)
        result_ct = rpb_dot_product(q_cts, d_cts)
        num, den = rpb.decrypt(result_ct)
        encrypted_result = float(num) / float(den)

        # Encryption error must be zero (exact arithmetic)
        assert abs(encrypted_result - expected) < 1e-12, (
            f"k={k}: encryption error {abs(encrypted_result - expected)}"
        )

        # Quantization error must be within analytical bound
        true_val = true_dot(q_f, d_f)
        bound = dot_product_error_bound(k, 4)
        assert abs(true_val - encrypted_result) <= bound + 1e-10

    def test_negative_components(self):
        """Dot product with negative values."""
        k = 16
        q_f = [0.5, -0.5, 0.3, -0.7]
        d_f = [-0.2, 0.8, -0.1, 0.4]

        q_rats = encode_vector(q_f, k)
        d_rats = encode_vector(d_f, k)
        c = make_rpb_blinding()

        expected = plaintext_dot_product(q_rats, d_rats)

        q_cts = rpb_encrypt_vector(q_rats, c)
        d_cts = rpb_encrypt_vector(d_rats, c)
        result_ct = rpb_dot_product(q_cts, d_cts)
        num, den = rpb.decrypt(result_ct)
        encrypted_result = float(num) / float(den)

        assert abs(encrypted_result - expected) < 1e-12

    def test_different_blinding_factors(self):
        """Using different c for q and d should still give correct results."""
        k = 16
        q_f = [0.5, 0.3]
        d_f = [0.4, 0.6]
        q_rats = encode_vector(q_f, k)
        d_rats = encode_vector(d_f, k)

        c_q = make_rpb_blinding(bits=128)
        c_d = mpz(c_q + 99999)  # different blinding

        q_cts = rpb_encrypt_vector(q_rats, c_q)
        d_cts = rpb_encrypt_vector(d_rats, c_d)

        result_ct = rpb_dot_product(q_cts, d_cts)
        num, den = rpb.decrypt(result_ct)
        encrypted_result = float(num) / float(den)

        expected = plaintext_dot_product(q_rats, d_rats)
        assert abs(encrypted_result - expected) < 1e-12


# ── MRPB: encrypted dot product correctness ───────────────────────────

class TestMRPBDotProduct:

    def test_basic_2d(self, mrpb_keys):
        _, sk = mrpb_keys
        k = 16
        q = encode_vector([1.0, 0.0], k)
        d = encode_vector([0.0, 1.0], k)

        q_cts = mrpb_encrypt_vector(q, sk)
        d_cts = mrpb_encrypt_vector(d, sk)

        result_ct = mrpb_dot_product(q_cts, d_cts)
        num, den = mrpb.decrypt(result_ct, sk)

        assert num == 0, f"Expected 0, got {num}/{den}"

    def test_parallel_vectors(self, mrpb_keys):
        _, sk = mrpb_keys
        k = 16
        q = encode_vector([1.0, 0.0], k)
        d = encode_vector([1.0, 0.0], k)

        q_cts = mrpb_encrypt_vector(q, sk)
        d_cts = mrpb_encrypt_vector(d, sk)

        result_ct = mrpb_dot_product(q_cts, d_cts)
        num, den = mrpb.decrypt(result_ct, sk)
        result = float(num) / float(den)

        assert abs(result - 1.0) < 1e-10

    @pytest.mark.parametrize("k", [8, 12, 16])
    def test_random_4d(self, k, mrpb_keys):
        """Encrypted MRPB dot product must equal plaintext encoded version."""
        _, sk = mrpb_keys
        rng = random.Random(99)
        q_f = random_unit_vector(4, rng)
        d_f = random_unit_vector(4, rng)

        q_rats = encode_vector(q_f, k)
        d_rats = encode_vector(d_f, k)

        expected = plaintext_dot_product(q_rats, d_rats)

        q_cts = mrpb_encrypt_vector(q_rats, sk)
        d_cts = mrpb_encrypt_vector(d_rats, sk)
        result_ct = mrpb_dot_product(q_cts, d_cts)
        num, den = mrpb.decrypt(result_ct, sk)
        encrypted_result = float(num) / float(den)

        # Encryption error must be zero
        assert abs(encrypted_result - expected) < 1e-12, (
            f"k={k}: encryption error {abs(encrypted_result - expected)}"
        )

    def test_8d_with_negatives(self, mrpb_keys):
        """8D vector with mixed signs."""
        _, sk = mrpb_keys
        k = 16
        q_f = [0.1, -0.2, 0.3, -0.4, 0.5, -0.1, 0.2, -0.3]
        d_f = [-0.3, 0.4, -0.1, 0.2, -0.5, 0.3, -0.2, 0.1]

        q_rats = encode_vector(q_f, k)
        d_rats = encode_vector(d_f, k)

        expected = plaintext_dot_product(q_rats, d_rats)

        q_cts = mrpb_encrypt_vector(q_rats, sk)
        d_cts = mrpb_encrypt_vector(d_rats, sk)
        result_ct = mrpb_dot_product(q_cts, d_cts)
        num, den = mrpb.decrypt(result_ct, sk)
        encrypted_result = float(num) / float(den)

        assert abs(encrypted_result - expected) < 1e-12


# ── Ranking preservation (critical for RAG) ───────────────────────────

class TestRankingPreservation:
    """Verify that encrypted dot products preserve document ranking.

    This is the most important property for RAG: if doc_A is more similar
    to the query than doc_B in plaintext, this must hold after encryption.
    """

    def test_rpb_ranking_5_docs(self):
        """Encrypt query and 5 docs, verify ranking matches plaintext."""
        k = 16
        rng = random.Random(77)
        c = make_rpb_blinding()

        query = random_unit_vector(8, rng)
        docs = [random_unit_vector(8, rng) for _ in range(5)]

        # Plaintext ranking
        true_scores = [true_dot(query, d) for d in docs]
        true_ranking = sorted(range(5), key=lambda i: -true_scores[i])

        # Encrypted ranking
        q_rats = encode_vector(query, k)
        q_cts = rpb_encrypt_vector(q_rats, c)

        enc_scores = []
        for doc in docs:
            d_rats = encode_vector(doc, k)
            d_cts = rpb_encrypt_vector(d_rats, c)
            result_ct = rpb_dot_product(q_cts, d_cts)
            num, den = rpb.decrypt(result_ct)
            enc_scores.append(float(num) / float(den))

        enc_ranking = sorted(range(5), key=lambda i: -enc_scores[i])

        assert true_ranking == enc_ranking, (
            f"Ranking mismatch!\n"
            f"  True:      {true_ranking} (scores: {[f'{s:.6f}' for s in true_scores]})\n"
            f"  Encrypted: {enc_ranking} (scores: {[f'{s:.6f}' for s in enc_scores]})"
        )

    def test_mrpb_ranking_5_docs(self, mrpb_keys):
        """Same ranking test with MRPB (IND-CPA secure)."""
        _, sk = mrpb_keys
        k = 16
        rng = random.Random(77)

        query = random_unit_vector(8, rng)
        docs = [random_unit_vector(8, rng) for _ in range(5)]

        true_scores = [true_dot(query, d) for d in docs]
        true_ranking = sorted(range(5), key=lambda i: -true_scores[i])

        q_rats = encode_vector(query, k)
        q_cts = mrpb_encrypt_vector(q_rats, sk)

        enc_scores = []
        for doc in docs:
            d_rats = encode_vector(doc, k)
            d_cts = mrpb_encrypt_vector(d_rats, sk)
            result_ct = mrpb_dot_product(q_cts, d_cts)
            num, den = mrpb.decrypt(result_ct, sk)
            enc_scores.append(float(num) / float(den))

        enc_ranking = sorted(range(5), key=lambda i: -enc_scores[i])

        assert true_ranking == enc_ranking

    @pytest.mark.parametrize("k", [8, 12, 16, 20])
    def test_rpb_ranking_across_precisions(self, k):
        """Rankings must be stable across precision levels."""
        rng = random.Random(55)
        c = make_rpb_blinding()

        query = random_unit_vector(8, rng)
        docs = [random_unit_vector(8, rng) for _ in range(5)]

        true_scores = [true_dot(query, d) for d in docs]
        true_ranking = sorted(range(5), key=lambda i: -true_scores[i])

        q_rats = encode_vector(query, k)
        q_cts = rpb_encrypt_vector(q_rats, c)

        enc_scores = []
        for doc in docs:
            d_rats = encode_vector(doc, k)
            d_cts = rpb_encrypt_vector(d_rats, c)
            result_ct = rpb_dot_product(q_cts, d_cts)
            num, den = rpb.decrypt(result_ct)
            enc_scores.append(float(num) / float(den))

        enc_ranking = sorted(range(5), key=lambda i: -enc_scores[i])

        assert true_ranking == enc_ranking, (
            f"k={k}: ranking mismatch"
        )


# ── Plaintext helpers ─────────────────────────────────────────────────

class TestPlaintextHelpers:

    def test_exact_dot_product(self):
        q = [(3, 10), (7, 10)]
        d = [(4, 10), (2, 10)]
        num, den = plaintext_dot_product_exact(q, d)
        # (3*4 + 7*2) / 10^2 = 26/100
        assert num == 26
        assert den == 100

    def test_float_dot_product(self):
        q = encode_vector([0.5, 0.5], 16)
        d = encode_vector([1.0, 0.0], 16)
        result = plaintext_dot_product(q, d)
        assert abs(result - 0.5) < 1e-10

    def test_empty_exact(self):
        assert plaintext_dot_product_exact([], []) == (0, 1)


# ── Pipeline end-to-end ───────────────────────────────────────────────

class TestPipeline:

    def test_rpb_pipeline(self):
        q = [0.5, -0.3, 0.7, 0.1]
        d = [0.2, 0.8, -0.1, 0.4]

        result = rpb_similarity_pipeline(q, d, precision_bits=16)

        assert result["encryption_error"] < 1e-12, (
            "RPB encryption must be exact"
        )
        assert result["quantization_error"] < dot_product_error_bound(16, 4)

    def test_mrpb_pipeline(self, mrpb_keys):
        _, sk = mrpb_keys
        q = [0.5, -0.3, 0.7, 0.1]
        d = [0.2, 0.8, -0.1, 0.4]

        result = mrpb_similarity_pipeline(q, d, precision_bits=16, sk=sk)

        assert result["encryption_error"] < 1e-12, (
            "MRPB encryption must be exact"
        )
        assert result["quantization_error"] < dot_product_error_bound(16, 4)


# ── Input validation ──────────────────────────────────────────────────

class TestValidation:

    def test_length_mismatch_rpb(self):
        c = make_rpb_blinding()
        q = rpb_encrypt_vector(encode_vector([1.0, 0.0], 16), c)
        d = rpb_encrypt_vector(encode_vector([1.0], 16), c)
        with pytest.raises(ValueError, match="length mismatch"):
            rpb_dot_product(q, d)

    def test_empty_rpb(self):
        with pytest.raises(ValueError, match="empty"):
            rpb_dot_product([], [])

    def test_length_mismatch_mrpb(self, mrpb_keys):
        _, sk = mrpb_keys
        q = mrpb_encrypt_vector(encode_vector([1.0, 0.0], 16), sk)
        d = mrpb_encrypt_vector(encode_vector([1.0], 16), sk)
        with pytest.raises(ValueError, match="length mismatch"):
            mrpb_dot_product(q, d)

    def test_empty_mrpb(self):
        with pytest.raises(ValueError, match="empty"):
            mrpb_dot_product([], [])
