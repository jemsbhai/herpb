"""
Tests for float-to-rational encoding.

Verifies:
  1. Round-trip accuracy at multiple precisions
  2. Error is always within the analytical bound
  3. Edge cases (0, ±1, small values, negative)
  4. Invalid input rejection (NaN, inf, bad precision)
  5. Vector encode/decode
  6. Dot-product error bound correctness (empirical vs analytical)
  7. MRPB key-size estimates are sane
"""

import math
import random
import pytest

from herpb.utils.encoding import (
    float_to_rational,
    rational_to_float,
    encode_vector,
    decode_vector,
    quantization_error_bound,
    encoding_error,
    dot_product_error_bound,
    cosine_similarity_error_bound,
    max_numerator_bits,
    dot_product_operand_bits,
    mrpb_min_key_bits,
    recommend_precision,
)


# ── Helpers ───────────────────────────────────────────────────────────

def random_unit_vector(dim: int, rng: random.Random) -> list[float]:
    """Generate a random unit-norm vector."""
    raw = [rng.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── Round-trip correctness ────────────────────────────────────────────

class TestRoundTrip:

    @pytest.mark.parametrize("k", [8, 12, 16, 20, 24])
    def test_roundtrip_zero(self, k):
        num, den = float_to_rational(0.0, k)
        assert num == 0
        assert den == (1 << k)
        assert rational_to_float(num, den) == 0.0

    @pytest.mark.parametrize("k", [8, 12, 16, 20, 24])
    def test_roundtrip_one(self, k):
        num, den = float_to_rational(1.0, k)
        assert num == den  # round(1.0 * 2^k) == 2^k
        assert rational_to_float(num, den) == 1.0

    @pytest.mark.parametrize("k", [8, 12, 16, 20, 24])
    def test_roundtrip_negative_one(self, k):
        num, den = float_to_rational(-1.0, k)
        assert num == -den
        assert rational_to_float(num, den) == -1.0

    @pytest.mark.parametrize("k", [8, 12, 16, 20, 24])
    def test_roundtrip_half(self, k):
        num, den = float_to_rational(0.5, k)
        assert rational_to_float(num, den) == 0.5

    @pytest.mark.parametrize("k", [8, 16, 24])
    def test_roundtrip_small_positive(self, k):
        x = 1e-5
        num, den = float_to_rational(x, k)
        reconstructed = rational_to_float(num, den)
        assert abs(reconstructed - x) <= quantization_error_bound(k)

    @pytest.mark.parametrize("k", [8, 16, 24])
    def test_roundtrip_small_negative(self, k):
        x = -1e-5
        num, den = float_to_rational(x, k)
        reconstructed = rational_to_float(num, den)
        assert abs(reconstructed - x) <= quantization_error_bound(k)

    def test_denominator_is_power_of_two(self):
        for k in range(1, 25):
            _, den = float_to_rational(0.3, k)
            assert den == (1 << k)
            assert (den & (den - 1)) == 0  # power-of-two check


# ── Error bound verification ─────────────────────────────────────────

class TestErrorBounds:

    @pytest.mark.parametrize("k", [8, 12, 16, 20, 24])
    def test_error_within_bound(self, k):
        """Every encoding error must be ≤ 2^{-(k+1)}."""
        bound = quantization_error_bound(k)
        rng = random.Random(42)
        for _ in range(10_000):
            x = rng.uniform(-10.0, 10.0)
            err = encoding_error(x, k)
            assert err <= bound + 1e-15, (
                f"k={k}, x={x}: error {err} exceeds bound {bound}"
            )

    @pytest.mark.parametrize("k", [8, 12, 16, 20])
    def test_bound_is_tight(self, k):
        """The bound should be achievable (at midpoints)."""
        # A midpoint: x = (n + 0.5) / 2^k for some integer n
        scale = 1 << k
        x = 0.5 / scale  # midpoint between 0/2^k and 1/2^k
        err = encoding_error(x, k)
        bound = quantization_error_bound(k)
        # Error at exact midpoint should equal the bound
        assert abs(err - bound) < 1e-15

    def test_bound_decreases_with_precision(self):
        prev = float("inf")
        for k in range(1, 25):
            b = quantization_error_bound(k)
            assert b < prev
            prev = b


# ── Input validation ──────────────────────────────────────────────────

class TestValidation:

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            float_to_rational(float("nan"), 16)

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            float_to_rational(float("inf"), 16)

    def test_neg_inf_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            float_to_rational(float("-inf"), 16)

    def test_zero_precision_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            float_to_rational(1.0, 0)

    def test_negative_precision_rejected(self):
        with pytest.raises(ValueError, match="positive integer"):
            float_to_rational(1.0, -5)

    def test_zero_denominator_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            rational_to_float(1, 0)

    def test_negative_denominator_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            rational_to_float(1, -1)


# ── Vector operations ─────────────────────────────────────────────────

class TestVectorOps:

    def test_encode_decode_roundtrip(self):
        vec = [0.1, -0.5, 0.0, 0.99, -0.001]
        k = 20
        encoded = encode_vector(vec, k)
        decoded = decode_vector(encoded)
        bound = quantization_error_bound(k)
        for orig, rec in zip(vec, decoded):
            assert abs(orig - rec) <= bound + 1e-15

    def test_shared_denominator(self):
        """All components must share the same denominator."""
        vec = [0.1, 0.2, 0.3, 0.4, 0.5]
        k = 16
        encoded = encode_vector(vec, k)
        dens = [d for _, d in encoded]
        assert all(d == dens[0] for d in dens)
        assert dens[0] == (1 << k)

    def test_empty_vector(self):
        assert encode_vector([], 16) == []
        assert decode_vector([]) == []


# ── Dot-product error bound (empirical) ───────────────────────────────

class TestDotProductError:

    @pytest.mark.parametrize("k", [8, 12, 16, 20])
    @pytest.mark.parametrize("dim", [10, 100, 384])
    def test_empirical_within_analytical_bound(self, k, dim):
        """Empirical dot-product error must never exceed the bound."""
        bound = dot_product_error_bound(k, dim, max_component=1.0)
        rng = random.Random(123)

        for _ in range(100):
            q = random_unit_vector(dim, rng)
            d = random_unit_vector(dim, rng)

            # True dot product
            true_dot = dot(q, d)

            # Encoded dot product (in plaintext, simulating HE)
            q_enc = encode_vector(q, k)
            d_enc = encode_vector(d, k)
            enc_dot = sum(nq * nd for (nq, _), (nd, _) in zip(q_enc, d_enc))
            # Denominator is (2^k)^2
            enc_dot_float = enc_dot / ((1 << k) ** 2)

            err = abs(true_dot - enc_dot_float)
            assert err <= bound + 1e-12, (
                f"k={k}, dim={dim}: error {err:.2e} exceeds bound {bound:.2e}"
            )

    def test_error_shrinks_with_precision(self):
        """Higher k should give smaller worst-case error."""
        dim = 384
        prev = float("inf")
        for k in [8, 12, 16, 20, 24]:
            b = dot_product_error_bound(k, dim)
            assert b < prev
            prev = b


# ── Cosine similarity error bound ────────────────────────────────────

class TestCosineSimilarityError:

    def test_equals_dot_for_unit_vectors(self):
        """For unit vectors, cosine error = dot-product error."""
        k, dim = 16, 384
        assert cosine_similarity_error_bound(k, dim) == dot_product_error_bound(k, dim)


# ── MRPB key-size analysis ───────────────────────────────────────────

class TestMRPBKeySizing:

    def test_key_bits_increase_with_dim(self):
        prev = 0
        for dim in [10, 50, 100, 384, 768]:
            bits = mrpb_min_key_bits(16, dim, safety_factor=1.0)
            assert bits > prev
            prev = bits

    def test_key_bits_increase_with_precision(self):
        prev = 0
        for k in [8, 12, 16, 20, 24]:
            bits = mrpb_min_key_bits(k, 384, safety_factor=1.0)
            assert bits > prev
            prev = bits

    def test_safety_factor_applied(self):
        base = mrpb_min_key_bits(16, 384, safety_factor=1.0)
        doubled = mrpb_min_key_bits(16, 384, safety_factor=2.0)
        assert doubled == 2 * base

    def test_dot_product_operand_bits(self):
        """Product of two k-bit numerators → 2k bits."""
        k = 16
        num_bits = max_numerator_bits(k, 1.0)
        prod_bits = dot_product_operand_bits(k, 1.0)
        assert prod_bits == 2 * num_bits


# ── Precision recommendation ─────────────────────────────────────────

class TestRecommendPrecision:

    def test_returns_results(self):
        rec = recommend_precision(384, target_cosine_error=0.001)
        assert len(rec["recommendations"]) == 25  # k = 8..32
        assert rec["selected_k"] is not None
        assert rec["selected_k"] >= 8

    def test_selected_meets_target(self):
        target = 0.001
        rec = recommend_precision(384, target_cosine_error=target)
        k = rec["selected_k"]
        # Verify the selected k actually meets the target
        err = cosine_similarity_error_bound(k, 384)
        assert err <= target

    def test_selected_is_smallest(self):
        """The selected k should be the smallest that meets the target."""
        target = 0.001
        rec = recommend_precision(384, target_cosine_error=target)
        k = rec["selected_k"]
        # k-1 should NOT meet the target
        if k > 8:
            err_prev = cosine_similarity_error_bound(k - 1, 384)
            assert err_prev > target
