"""Tests for RPB (Ratio-Preserving Blinding) scheme.

Validates:
1. Correctness: decrypt(encrypt(a, b)) == a/b
2. Shared-k and distinct-k produce same ratio
3. Homomorphic multiplication
4. Homomorphic addition
5. Homomorphic subtraction
6. Large blinding factors
7. Edge cases
"""

import pytest
from gmpy2 import mpz

from src.herpb.schemes.rpb import (
    Rational,
    shared_k_encrypt,
    distinct_k_encrypt,
    decrypt,
    eval_multiply,
    eval_add,
    eval_subtract,
    max_t_shared,
    max_T_distinct,
    split_T,
)
from src.herpb.utils.math_utils import reduce_fraction


class TestRPBCorrectness:
    """Core correctness tests — does decrypt recover the right ratio?"""

    def test_shared_k_simple(self, simple_rationals):
        """The canonical test: a=2/3, b=4/5 -> ratio = 5/6."""
        s = simple_rationals
        a = Rational(s["a_num"], s["a_den"])
        b = Rational(s["b_num"], s["b_den"])

        ct = shared_k_encrypt(a, b, t=3, c=mpz(7))
        num, den = decrypt(ct)

        assert num == s["expected_num"]
        assert den == s["expected_den"]

    def test_distinct_k_simple(self, simple_rationals):
        """Same test with distinct-k variant."""
        s = simple_rationals
        a = Rational(s["a_num"], s["a_den"])
        b = Rational(s["b_num"], s["b_den"])

        ct = distinct_k_encrypt(a, b, ta=3, tb=3, c=mpz(13))
        num, den = decrypt(ct)

        assert num == s["expected_num"]
        assert den == s["expected_den"]

    def test_both_methods_agree(self, simple_rationals):
        """Shared-k and distinct-k produce the same plaintext ratio."""
        s = simple_rationals
        a = Rational(s["a_num"], s["a_den"])
        b = Rational(s["b_num"], s["b_den"])

        ct_shared = shared_k_encrypt(a, b, t=5, c=mpz(999))
        ct_distinct = distinct_k_encrypt(a, b, ta=5, tb=5, c=mpz(1234))

        assert decrypt(ct_shared) == decrypt(ct_distinct)

    @pytest.mark.parametrize("c_val", [1, -1, 42, -50, 10**100, -(10**500)])
    def test_various_blinding_factors(self, simple_rationals, c_val):
        """Correctness holds for various c values including very large ones."""
        s = simple_rationals
        a = Rational(s["a_num"], s["a_den"])
        b = Rational(s["b_num"], s["b_den"])

        ct = shared_k_encrypt(a, b, t=3, c=mpz(c_val))
        num, den = decrypt(ct)

        assert num == s["expected_num"]
        assert den == s["expected_den"]

    @pytest.mark.parametrize("t", [0, 1, 2, 3, 5, 10])
    def test_various_precision(self, simple_rationals, t):
        """Correctness holds for various precision parameters."""
        s = simple_rationals
        a = Rational(s["a_num"], s["a_den"])
        b = Rational(s["b_num"], s["b_den"])

        ct = shared_k_encrypt(a, b, t=t, c=mpz(7))
        num, den = decrypt(ct)

        assert num == s["expected_num"]
        assert den == s["expected_den"]

    def test_zero_c_raises(self, simple_rationals):
        """c=0 should raise ValueError."""
        s = simple_rationals
        a = Rational(s["a_num"], s["a_den"])
        b = Rational(s["b_num"], s["b_den"])

        with pytest.raises(ValueError, match="nonzero"):
            shared_k_encrypt(a, b, t=3, c=mpz(0))

    def test_zero_denominator_raises(self):
        """Rational with den=0 should raise."""
        with pytest.raises(ValueError):
            Rational(mpz(1), mpz(0))


class TestRPBHomomorphic:
    """Tests for homomorphic operations on RPB ciphertexts."""

    def test_multiply(self, simple_rationals, second_pair):
        """(2/3)/(4/5) * (7/11)/(3/13) = 5/6 * 91/33 = 455/198 = 65/198*7... let me compute."""
        a1 = Rational(mpz(2), mpz(3))
        b1 = Rational(mpz(4), mpz(5))
        a2 = Rational(mpz(7), mpz(11))
        b2 = Rational(mpz(3), mpz(13))

        ct1 = shared_k_encrypt(a1, b1, t=3, c=mpz(7))
        ct2 = shared_k_encrypt(a2, b2, t=3, c=mpz(11))

        ct_prod = eval_multiply(ct1, ct2)
        num, den = decrypt(ct_prod)

        # Expected: (5/6) * (91/33) = 455/198 = 65/198... let's reduce
        expected_num, expected_den = reduce_fraction(mpz(5 * 91), mpz(6 * 33))
        assert num == expected_num
        assert den == expected_den

    def test_add(self, simple_rationals, second_pair):
        """(5/6) + (91/33) via homomorphic addition."""
        a1 = Rational(mpz(2), mpz(3))
        b1 = Rational(mpz(4), mpz(5))
        a2 = Rational(mpz(7), mpz(11))
        b2 = Rational(mpz(3), mpz(13))

        ct1 = shared_k_encrypt(a1, b1, t=3, c=mpz(7))
        ct2 = shared_k_encrypt(a2, b2, t=3, c=mpz(11))

        ct_sum = eval_add(ct1, ct2)
        num, den = decrypt(ct_sum)

        # Expected: 5/6 + 91/33 = (5*33 + 91*6)/(6*33) = (165+546)/198 = 711/198
        expected_num, expected_den = reduce_fraction(mpz(711), mpz(198))
        assert num == expected_num
        assert den == expected_den

    def test_subtract(self):
        """Homomorphic subtraction."""
        a1 = Rational(mpz(7), mpz(11))
        b1 = Rational(mpz(3), mpz(13))
        a2 = Rational(mpz(2), mpz(3))
        b2 = Rational(mpz(4), mpz(5))

        ct1 = shared_k_encrypt(a1, b1, t=3, c=mpz(5))
        ct2 = shared_k_encrypt(a2, b2, t=3, c=mpz(9))

        ct_diff = eval_subtract(ct1, ct2)
        num, den = decrypt(ct_diff)

        # Expected: 91/33 - 5/6 = (91*6 - 5*33)/(33*6) = (546-165)/198 = 381/198
        expected_num, expected_den = reduce_fraction(mpz(381), mpz(198))
        assert num == expected_num
        assert den == expected_den

    def test_chained_multiply(self):
        """Three chained multiplications — tests unlimited depth claim."""
        pairs = [
            (Rational(mpz(1), mpz(2)), Rational(mpz(1), mpz(3))),  # ratio = 3/2
            (Rational(mpz(2), mpz(5)), Rational(mpz(1), mpz(7))),  # ratio = 14/5
            (Rational(mpz(3), mpz(4)), Rational(mpz(2), mpz(9))),  # ratio = 27/8
        ]

        cts = [
            shared_k_encrypt(a, b, t=2, c=mpz(i + 3))
            for i, (a, b) in enumerate(pairs)
        ]

        result = cts[0]
        for ct in cts[1:]:
            result = eval_multiply(result, ct)

        num, den = decrypt(result)

        # Expected: (3/2) * (14/5) * (27/8) = 3*14*27 / 2*5*8 = 1134/80 = 567/40
        expected_num, expected_den = reduce_fraction(
            mpz(3 * 14 * 27), mpz(2 * 5 * 8)
        )
        assert num == expected_num
        assert den == expected_den


class TestRPBBounds:
    """Tests for the precision bound computation."""

    def test_max_t_shared_basic(self, simple_rationals):
        """max_t_shared returns a usable value for small inputs."""
        s = simple_rationals
        a = Rational(s["a_num"], s["a_den"])
        b = Rational(s["b_num"], s["b_den"])

        bound = mpz(10) ** 20000
        t = max_t_shared(a, b, c_abs=mpz(50), bound=bound)

        assert t > 0, "Should allow some precision for small inputs"

    def test_max_T_distinct_basic(self, simple_rationals):
        s = simple_rationals
        a = Rational(s["a_num"], s["a_den"])
        b = Rational(s["b_num"], s["b_den"])

        bound = mpz(10) ** 20000
        T = max_T_distinct(a, b, c_abs=mpz(50), bound=bound)

        assert T > 0

    def test_split_T(self):
        assert split_T(10, 6, 6) == (6, 4)
        assert split_T(3, 6, 6) == (3, 0)
        assert split_T(8, 3, 3) == (3, 3)  # 3+3 = 6 <= 8, but capped
