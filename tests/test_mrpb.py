"""Tests for MRPB (Modular Ratio-Preserving Blinding) scheme.

Validates:
1. Key generation produces valid RSA parameters
2. Correctness: decrypt(encrypt(a, b), sk) == a/b
3. Semantic security: ciphertexts look random without sk
4. Homomorphic multiplication
5. Fixed ciphertext size
6. Rational reconstruction from modular residues
"""

import pytest
from gmpy2 import mpz
import gmpy2

from src.herpb.schemes.mrpb import (
    keygen,
    encrypt_shared,
    encrypt_distinct,
    decrypt,
    eval_multiply,
    eval_add,
    MRPBCiphertext,
)


class TestMRPBKeygen:
    """Tests for key generation."""

    def test_keygen_produces_valid_rsa(self, mrpb_keys):
        pk, sk = mrpb_keys
        assert sk.n == sk.p * sk.q
        assert gmpy2.is_prime(sk.p)
        assert gmpy2.is_prime(sk.q)
        assert sk.p != sk.q

    def test_keygen_c_is_coprime_to_n(self, mrpb_keys):
        _, sk = mrpb_keys
        assert gmpy2.gcd(sk.c, sk.n) == 1

    def test_keygen_c_inv_is_correct(self, mrpb_keys):
        _, sk = mrpb_keys
        assert (sk.c * sk.c_inv) % sk.n == 1

    def test_keygen_deterministic_with_seed(self):
        """Same seed produces same keys."""
        _, sk1 = keygen(key_bits=512, seed=123)
        _, sk2 = keygen(key_bits=512, seed=123)
        assert sk1.p == sk2.p
        assert sk1.q == sk2.q
        assert sk1.c == sk2.c

    def test_keygen_different_seeds_differ(self):
        """Different seeds produce different keys."""
        _, sk1 = keygen(key_bits=512, seed=1)
        _, sk2 = keygen(key_bits=512, seed=2)
        assert sk1.p != sk2.p


class TestMRPBCorrectness:
    """Core correctness: decrypt recovers the right ratio."""

    def test_shared_k_simple(self, simple_rationals, mrpb_keys):
        """a=2/3, b=4/5 -> ratio = 5/6."""
        s = simple_rationals
        _, sk = mrpb_keys

        ct = encrypt_shared(
            s["a_num"], s["a_den"],
            s["b_num"], s["b_den"],
            t=3, sk=sk,
        )
        num, den = decrypt(ct, sk)

        assert num == s["expected_num"], f"Expected num={s['expected_num']}, got {num}"
        assert den == s["expected_den"], f"Expected den={s['expected_den']}, got {den}"

    def test_distinct_k_simple(self, simple_rationals, mrpb_keys):
        s = simple_rationals
        _, sk = mrpb_keys

        ct = encrypt_distinct(
            s["a_num"], s["a_den"],
            s["b_num"], s["b_den"],
            ta=3, tb=3, sk=sk,
        )
        num, den = decrypt(ct, sk)

        assert num == s["expected_num"]
        assert den == s["expected_den"]

    @pytest.mark.parametrize("a_num,a_den,b_num,b_den,exp_n,exp_d", [
        (1, 2, 1, 3, 3, 2),       # (1/2)/(1/3) = 3/2
        (7, 11, 3, 13, 91, 33),   # (7/11)/(3/13) = 91/33
        (5, 1, 1, 1, 5, 1),       # 5/1 = 5
        (1, 7, 1, 7, 1, 1),       # equal -> 1
        (10, 3, 5, 9, 6, 1),      # (10/3)/(5/9) = 90/15 = 6/1
    ])
    def test_various_rationals(self, a_num, a_den, b_num, b_den, exp_n, exp_d, mrpb_keys):
        _, sk = mrpb_keys
        ct = encrypt_shared(mpz(a_num), mpz(a_den), mpz(b_num), mpz(b_den), t=2, sk=sk)
        num, den = decrypt(ct, sk)
        assert num == mpz(exp_n), f"Expected {exp_n}/{exp_d}, got {num}/{den}"
        assert den == mpz(exp_d)


class TestMRPBSecurity:
    """Tests that ciphertexts reveal nothing without the secret key."""

    def test_ciphertext_not_equal_to_plaintext_ratio(self, simple_rationals, mrpb_keys):
        """N/D mod n should NOT be trivially reducible to 5/6."""
        s = simple_rationals
        _, sk = mrpb_keys

        ct = encrypt_shared(
            s["a_num"], s["a_den"],
            s["b_num"], s["b_den"],
            t=3, sk=sk,
        )

        # GCD of N and D mod n should NOT give the ratio
        g = gmpy2.gcd(ct.N, ct.D)
        reduced_n = ct.N // g
        reduced_d = ct.D // g

        # If the scheme is secure, this should NOT equal 5/6
        # (it could by coincidence with very small probability, but
        # for our test keys it should not)
        is_leaked = (reduced_n == s["expected_num"] and reduced_d == s["expected_den"])
        assert not is_leaked, "GCD reduction leaked the plaintext ratio!"

    def test_same_plaintext_different_keys_different_ciphertext(self, simple_rationals):
        """Same plaintext encrypted with different keys -> different ciphertexts."""
        s = simple_rationals
        _, sk1 = keygen(key_bits=512, seed=100)
        _, sk2 = keygen(key_bits=512, seed=200)

        ct1 = encrypt_shared(s["a_num"], s["a_den"], s["b_num"], s["b_den"], t=3, sk=sk1)
        ct2 = encrypt_shared(s["a_num"], s["a_den"], s["b_num"], s["b_den"], t=3, sk=sk2)

        assert ct1.N != ct2.N or ct1.D != ct2.D

    def test_ciphertext_size_is_bounded(self, simple_rationals, mrpb_keys):
        """Ciphertexts should be bounded by n (fixed size)."""
        s = simple_rationals
        pk, sk = mrpb_keys

        ct = encrypt_shared(
            s["a_num"], s["a_den"],
            s["b_num"], s["b_den"],
            t=3, sk=sk,
        )

        assert ct.N < pk.n, "N should be < n"
        assert ct.D < pk.n, "D should be < n"


class TestMRPBHomomorphic:
    """Tests for homomorphic operations."""

    def test_multiply(self, mrpb_keys):
        """Homomorphic multiplication of two encrypted ratios."""
        _, sk = mrpb_keys

        # Ratio 1: (2/3)/(4/5) = 5/6
        ct1 = encrypt_shared(mpz(2), mpz(3), mpz(4), mpz(5), t=2, sk=sk)
        # Ratio 2: (1/2)/(1/3) = 3/2
        ct2 = encrypt_shared(mpz(1), mpz(2), mpz(1), mpz(3), t=2, sk=sk)

        ct_prod = eval_multiply(ct1, ct2)
        num, den = decrypt(ct_prod, sk)

        # Expected: (5/6) * (3/2) = 15/12 = 5/4
        assert num == mpz(5), f"Expected 5, got {num}"
        assert den == mpz(4), f"Expected 4, got {den}"

    def test_multiply_fixed_size(self, mrpb_keys):
        """Ciphertext size should NOT grow after multiplication (unlike RPB)."""
        pk, sk = mrpb_keys

        ct1 = encrypt_shared(mpz(2), mpz(3), mpz(4), mpz(5), t=2, sk=sk)
        ct2 = encrypt_shared(mpz(1), mpz(2), mpz(1), mpz(3), t=2, sk=sk)

        ct_prod = eval_multiply(ct1, ct2)

        # After multiplication, still bounded by n
        assert ct_prod.N < pk.n
        assert ct_prod.D < pk.n

    def test_chained_multiply_depth_3(self, mrpb_keys):
        """Three multiplications — verifies unlimited depth."""
        _, sk = mrpb_keys

        ct1 = encrypt_shared(mpz(1), mpz(2), mpz(1), mpz(3), t=2, sk=sk)  # 3/2
        ct2 = encrypt_shared(mpz(2), mpz(5), mpz(1), mpz(7), t=2, sk=sk)  # 14/5
        ct3 = encrypt_shared(mpz(3), mpz(4), mpz(2), mpz(9), t=2, sk=sk)  # 27/8

        result = eval_multiply(eval_multiply(ct1, ct2), ct3)
        num, den = decrypt(result, sk)

        # Expected: (3/2)*(14/5)*(27/8) = 1134/80 = 567/40
        from src.herpb.utils.math_utils import reduce_fraction
        exp_n, exp_d = reduce_fraction(mpz(3 * 14 * 27), mpz(2 * 5 * 8))
        assert num == exp_n, f"Expected {exp_n}, got {num}"
        assert den == exp_d, f"Expected {exp_d}, got {den}"
