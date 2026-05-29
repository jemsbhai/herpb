"""Shared test fixtures for RPB and MRPB schemes."""

import pytest
from gmpy2 import mpz


@pytest.fixture
def simple_rationals():
    """The canonical test case from the C++ reference: a=2/3, b=4/5."""
    return {
        "a_num": mpz(2), "a_den": mpz(3),
        "b_num": mpz(4), "b_den": mpz(5),
        "expected_num": mpz(5),  # (2/3)/(4/5) = 10/12 = 5/6
        "expected_den": mpz(6),
    }


@pytest.fixture
def second_pair():
    """A second pair for homomorphic operation tests: e=7/11, f=3/13."""
    return {
        "a_num": mpz(7), "a_den": mpz(11),
        "b_num": mpz(3), "b_den": mpz(13),
        "expected_num": mpz(91),  # (7/11)/(3/13) = 91/33
        "expected_den": mpz(33),
    }


@pytest.fixture
def mrpb_keys():
    """Pre-generated MRPB key pair for deterministic tests."""
    from src.herpb.schemes.mrpb import keygen
    pk, sk = keygen(key_bits=1024, seed=42)
    return pk, sk
