"""
Tests for two-server additive secret sharing and Beaver triple multiplication.

Verifies:
  1. Share/reconstruct round-trip
  2. Additive homomorphism
  3. Scalar multiplication
  4. Beaver triple correctness
  5. Query-private dot product matches plaintext
  6. Mutual-privacy dot product matches plaintext
  7. Ranking preservation through secret sharing
"""

import math
import random
import pytest
import gmpy2
from gmpy2 import mpz

from herpb.schemes.secret_sharing import (
    share,
    reconstruct,
    share_vector,
    generate_beaver_triple,
    generate_beaver_triples,
    beaver_multiply,
    shared_add,
    shared_scalar_mul,
    query_private_dot,
    mutual_privacy_dot,
)
from herpb.utils.encoding import encode_vector


# ── Helpers ───────────────────────────────────────────────────────────

def make_modulus(bits=2048, seed=42):
    rs = gmpy2.random_state(seed)
    half = bits // 2
    p = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half) | (mpz(1) << (half - 1)))
    q = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half) | (mpz(1) << (half - 1)))
    return p * q


def random_unit_vector(dim, rng):
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def true_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


@pytest.fixture(scope="module")
def n_mod():
    return make_modulus(2048, seed=42)


# ── Share/Reconstruct ────────────────────────────────────────────────

class TestShareReconstruct:

    def test_zero(self, n_mod):
        s1, s2 = share(0, n_mod)
        assert reconstruct(s1, s2, n_mod) == 0

    def test_positive(self, n_mod):
        for v in [1, 42, 65536, 2**20]:
            s1, s2 = share(v, n_mod)
            assert reconstruct(s1, s2, n_mod) == v

    def test_negative(self, n_mod):
        for v in [-1, -42, -65536]:
            s1, s2 = share(v, n_mod)
            assert reconstruct(s1, s2, n_mod) == v

    def test_shares_are_random(self, n_mod):
        """Each share individually should look random (large)."""
        s1, s2 = share(42, n_mod)
        assert s1 > 2**100, "Share should be a large random value"

    def test_vector_roundtrip(self, n_mod):
        values = [100, -200, 0, 300, -50]
        s1s, s2s = share_vector(values, n_mod, seed=99)
        for v, s1, s2 in zip(values, s1s, s2s):
            assert reconstruct(s1, s2, n_mod) == v


# ── Homomorphic addition (free) ──────────────────────────────────────

class TestHomomorphicAdd:

    def test_add_two_values(self, n_mod):
        a, b = 42, 58
        a1, a2 = share(a, n_mod)
        b1, b2 = share(b, n_mod)
        c1, c2 = shared_add(a1, b1, a2, b2, n_mod)
        assert reconstruct(c1, c2, n_mod) == a + b

    def test_add_negative(self, n_mod):
        a, b = 100, -250
        a1, a2 = share(a, n_mod)
        b1, b2 = share(b, n_mod)
        c1, c2 = shared_add(a1, b1, a2, b2, n_mod)
        assert reconstruct(c1, c2, n_mod) == a + b

    def test_chained_add(self, n_mod):
        values = [10, 20, 30, 40, 50]
        shares = [share(v, n_mod) for v in values]
        total1, total2 = mpz(0), mpz(0)
        for s1, s2 in shares:
            total1, total2 = shared_add(total1, s1, total2, s2, n_mod)
        assert reconstruct(total1, total2, n_mod) == sum(values)


# ── Scalar multiplication (free) ─────────────────────────────────────

class TestScalarMul:

    def test_scalar_mul(self, n_mod):
        m = 100
        k = 7
        s1, s2 = share(m, n_mod)
        p1, p2 = shared_scalar_mul(s1, k, s2, n_mod)
        assert reconstruct(p1, p2, n_mod) == m * k

    def test_scalar_mul_negative(self, n_mod):
        m, k = -30, 15
        s1, s2 = share(m, n_mod)
        p1, p2 = shared_scalar_mul(s1, k, s2, n_mod)
        assert reconstruct(p1, p2, n_mod) == m * k


# ── Beaver triple multiplication ─────────────────────────────────────

class TestBeaverMultiply:

    def test_single_multiply(self, n_mod):
        x, y = 42, 58
        x1, x2 = share(x, n_mod)
        y1, y2 = share(y, n_mod)
        triple = generate_beaver_triple(n_mod)
        z1, z2 = beaver_multiply(x1, y1, x2, y2, triple, n_mod)
        assert reconstruct(z1, z2, n_mod) == x * y

    def test_multiply_negatives(self, n_mod):
        x, y = -30, 17
        x1, x2 = share(x, n_mod)
        y1, y2 = share(y, n_mod)
        triple = generate_beaver_triple(n_mod)
        z1, z2 = beaver_multiply(x1, y1, x2, y2, triple, n_mod)
        assert reconstruct(z1, z2, n_mod) == x * y

    def test_multiply_zero(self, n_mod):
        x, y = 0, 12345
        x1, x2 = share(x, n_mod)
        y1, y2 = share(y, n_mod)
        triple = generate_beaver_triple(n_mod)
        z1, z2 = beaver_multiply(x1, y1, x2, y2, triple, n_mod)
        assert reconstruct(z1, z2, n_mod) == 0

    def test_multiply_large(self, n_mod):
        x, y = 2**16, 2**16
        x1, x2 = share(x, n_mod)
        y1, y2 = share(y, n_mod)
        triple = generate_beaver_triple(n_mod)
        z1, z2 = beaver_multiply(x1, y1, x2, y2, triple, n_mod)
        assert reconstruct(z1, z2, n_mod) == x * y

    def test_batch_100(self, n_mod):
        rng = random.Random(42)
        triples = generate_beaver_triples(100, n_mod, seed=42)
        for i in range(100):
            x = rng.randint(-2**16, 2**16)
            y = rng.randint(-2**16, 2**16)
            x1, x2 = share(x, n_mod)
            y1, y2 = share(y, n_mod)
            z1, z2 = beaver_multiply(x1, y1, x2, y2, triples[i], n_mod)
            assert reconstruct(z1, z2, n_mod) == x * y, f"Failed at i={i}: {x}*{y}"


# ── Query-private dot product ─────────────────────────────────────────

class TestQueryPrivateDot:

    def test_basic_dot(self, n_mod):
        q = [3, -5, 7]
        d = [2, 4, -1]
        q1, q2 = share_vector(q, n_mod, seed=10)
        dot1, dot2 = query_private_dot(q1, q2, d, n_mod)
        result = reconstruct(dot1, dot2, n_mod)
        assert result == sum(a * b for a, b in zip(q, d))

    def test_encoded_vectors(self, n_mod):
        """Test with actual float-to-rational encoded vectors."""
        k = 16
        q_f = [0.5, -0.3, 0.7, 0.1]
        d_f = [0.2, 0.8, -0.1, 0.4]

        q_rats = encode_vector(q_f, k)
        d_rats = encode_vector(d_f, k)

        q_ints = [n for n, _ in q_rats]
        d_ints = [n for n, _ in d_rats]

        q1, q2 = share_vector(q_ints, n_mod, seed=20)
        dot1, dot2 = query_private_dot(q1, q2, d_ints, n_mod)
        result = int(reconstruct(dot1, dot2, n_mod))

        # The exact integer dot product
        expected = sum(qn * dn for qn, dn in zip(q_ints, d_ints))
        assert result == expected

        # Decode to float and check against true dot
        dot_float = result / (2**k * 2**k)
        true_val = true_dot(q_f, d_f)
        assert abs(dot_float - true_val) < 0.01

    @pytest.mark.parametrize("dim", [8, 32, 128])
    def test_random_vectors(self, n_mod, dim):
        k = 16
        rng = random.Random(dim)
        q_f = random_unit_vector(dim, rng)
        d_f = random_unit_vector(dim, rng)

        q_rats = encode_vector(q_f, k)
        d_rats = encode_vector(d_f, k)
        q_ints = [n for n, _ in q_rats]
        d_ints = [n for n, _ in d_rats]

        q1, q2 = share_vector(q_ints, n_mod, seed=dim)
        dot1, dot2 = query_private_dot(q1, q2, d_ints, n_mod)
        result = int(reconstruct(dot1, dot2, n_mod))

        expected = sum(qn * dn for qn, dn in zip(q_ints, d_ints))
        assert result == expected


# ── Mutual-privacy dot product ────────────────────────────────────────

class TestMutualPrivacyDot:

    def test_basic_dot(self, n_mod):
        q = [3, -5, 7]
        d = [2, 4, -1]
        q1, q2 = share_vector(q, n_mod, seed=30)
        d1, d2 = share_vector(d, n_mod, seed=31)
        triples = generate_beaver_triples(3, n_mod, seed=32)

        dot1, dot2 = mutual_privacy_dot(q1, q2, d1, d2, triples, n_mod)
        result = reconstruct(dot1, dot2, n_mod)
        assert result == sum(a * b for a, b in zip(q, d))

    def test_encoded_vectors(self, n_mod):
        k = 16
        q_f = [0.5, -0.3, 0.7, 0.1]
        d_f = [0.2, 0.8, -0.1, 0.4]

        q_rats = encode_vector(q_f, k)
        d_rats = encode_vector(d_f, k)
        q_ints = [n for n, _ in q_rats]
        d_ints = [n for n, _ in d_rats]

        q1, q2 = share_vector(q_ints, n_mod, seed=40)
        d1, d2 = share_vector(d_ints, n_mod, seed=41)
        triples = generate_beaver_triples(len(q_ints), n_mod, seed=42)

        dot1, dot2 = mutual_privacy_dot(q1, q2, d1, d2, triples, n_mod)
        result = int(reconstruct(dot1, dot2, n_mod))

        expected = sum(qn * dn for qn, dn in zip(q_ints, d_ints))
        assert result == expected

    @pytest.mark.parametrize("dim", [8, 32])
    def test_random_vectors(self, n_mod, dim):
        k = 16
        rng = random.Random(dim + 100)
        q_f = random_unit_vector(dim, rng)
        d_f = random_unit_vector(dim, rng)

        q_rats = encode_vector(q_f, k)
        d_rats = encode_vector(d_f, k)
        q_ints = [n for n, _ in q_rats]
        d_ints = [n for n, _ in d_rats]

        q1, q2 = share_vector(q_ints, n_mod, seed=dim)
        d1, d2 = share_vector(d_ints, n_mod, seed=dim + 1)
        triples = generate_beaver_triples(dim, n_mod, seed=dim + 2)

        dot1, dot2 = mutual_privacy_dot(q1, q2, d1, d2, triples, n_mod)
        result = int(reconstruct(dot1, dot2, n_mod))

        expected = sum(qn * dn for qn, dn in zip(q_ints, d_ints))
        assert result == expected


# ── Ranking preservation ──────────────────────────────────────────────

class TestRankingPreservation:

    def test_query_private_ranking(self, n_mod):
        """Encrypted ranking must match plaintext ranking."""
        k = 16
        rng = random.Random(77)
        query = random_unit_vector(32, rng)
        docs = [random_unit_vector(32, rng) for _ in range(20)]

        # True ranking
        true_scores = [true_dot(query, d) for d in docs]
        true_ranking = sorted(range(20), key=lambda i: -true_scores[i])

        # Encoded + shared ranking
        q_rats = encode_vector(query, k)
        q_ints = [n for n, _ in q_rats]
        q1, q2 = share_vector(q_ints, n_mod, seed=77)

        enc_scores = []
        for doc in docs:
            d_rats = encode_vector(doc, k)
            d_ints = [n for n, _ in d_rats]
            dot1, dot2 = query_private_dot(q1, q2, d_ints, n_mod)
            score = int(reconstruct(dot1, dot2, n_mod))
            enc_scores.append(score)

        enc_ranking = sorted(range(20), key=lambda i: -enc_scores[i])
        assert enc_ranking == true_ranking

    def test_mutual_privacy_ranking(self, n_mod):
        """Mutual-privacy ranking must match plaintext ranking."""
        k = 16
        dim = 16
        rng = random.Random(88)
        query = random_unit_vector(dim, rng)
        docs = [random_unit_vector(dim, rng) for _ in range(10)]

        true_scores = [true_dot(query, d) for d in docs]
        true_ranking = sorted(range(10), key=lambda i: -true_scores[i])

        q_rats = encode_vector(query, k)
        q_ints = [n for n, _ in q_rats]
        q1, q2 = share_vector(q_ints, n_mod, seed=88)

        enc_scores = []
        for doc in docs:
            d_rats = encode_vector(doc, k)
            d_ints = [n for n, _ in d_rats]
            d1, d2 = share_vector(d_ints, n_mod, seed=hash(tuple(d_ints)) % (2**31))
            triples = generate_beaver_triples(dim, n_mod, seed=hash(tuple(d_ints)) % (2**31) + 1)
            dot1, dot2 = mutual_privacy_dot(q1, q2, d1, d2, triples, n_mod)
            score = int(reconstruct(dot1, dot2, n_mod))
            enc_scores.append(score)

        enc_ranking = sorted(range(10), key=lambda i: -enc_scores[i])
        assert enc_ranking == true_ranking
