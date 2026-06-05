"""
Tests for GPU-accelerated batch modular scalar multiply.

Verifies CUDA kernel correctness against gmpy2 reference, then
benchmarks GPU vs CPU serial vs CPU parallel.
"""

import time
import pytest
import numpy as np
import gmpy2
from gmpy2 import mpz

from herpb.gpu.batch_ops import (
    mpz_to_limbs,
    limbs_to_mpz,
    batch_mod_scalar_mul,
    gpu_batch_plaintext_mul,
    LIMBS,
)


# ── Helpers ───────────────────────────────────────────────────────────

def random_mpz_below(n: mpz, seed: int) -> mpz:
    """Generate random mpz in [1, n)."""
    rs = gmpy2.random_state(seed)
    return gmpy2.mpz_urandomb(rs, int(gmpy2.num_digits(n, 2))) % n


def make_rsa_modulus(bits: int = 2048, seed: int = 42) -> mpz:
    rs = gmpy2.random_state(seed)
    half = bits // 2
    p = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half) | (mpz(1) << (half - 1)))
    q = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half) | (mpz(1) << (half - 1)))
    return p * q


@pytest.fixture(scope="module")
def rsa_n():
    return make_rsa_modulus(2048, seed=42)


# ── Conversion round-trip ─────────────────────────────────────────────

class TestConversion:

    def test_zero(self):
        limbs = mpz_to_limbs(mpz(0))
        assert limbs_to_mpz(limbs) == 0

    def test_one(self):
        limbs = mpz_to_limbs(mpz(1))
        assert limbs_to_mpz(limbs) == 1

    def test_small(self):
        for v in [42, 255, 65536, 2**32 - 1, 2**32, 2**64 - 1]:
            limbs = mpz_to_limbs(mpz(v))
            assert limbs_to_mpz(limbs) == v

    def test_large_roundtrip(self, rsa_n):
        for seed in range(10):
            val = random_mpz_below(rsa_n, seed)
            limbs = mpz_to_limbs(val)
            recovered = limbs_to_mpz(limbs)
            assert recovered == val, f"seed={seed}: {val} != {recovered}"

    def test_limb_count(self):
        limbs = mpz_to_limbs(mpz(1), n_limbs=64)
        assert len(limbs) == 64
        assert limbs.dtype == np.uint32


# ── CUDA kernel correctness ──────────────────────────────────────────

class TestBatchModScalarMul:

    def test_single_multiply(self, rsa_n):
        A = random_mpz_below(rsa_n, seed=100)
        s = 12345
        expected = (A * s) % rsa_n

        result = batch_mod_scalar_mul([A], [s], rsa_n)
        assert result[0] == expected

    def test_scalar_zero(self, rsa_n):
        A = random_mpz_below(rsa_n, seed=101)
        result = batch_mod_scalar_mul([A], [0], rsa_n)
        assert result[0] == 0

    def test_scalar_one(self, rsa_n):
        A = random_mpz_below(rsa_n, seed=102)
        result = batch_mod_scalar_mul([A], [1], rsa_n)
        assert result[0] == A

    def test_negative_scalar(self, rsa_n):
        A = random_mpz_below(rsa_n, seed=103)
        s = -42
        expected = (A * s) % rsa_n

        result = batch_mod_scalar_mul([A], [s], rsa_n)
        assert result[0] == expected

    def test_batch_100(self, rsa_n):
        """100 random multiplies, verify all match gmpy2."""
        As = [random_mpz_below(rsa_n, seed=200 + i) for i in range(100)]
        scalars = [(i * 137 - 50) % 100000 - 50000 for i in range(100)]

        results = batch_mod_scalar_mul(As, scalars, rsa_n)

        for i in range(100):
            expected = (As[i] * scalars[i]) % rsa_n
            assert results[i] == expected, f"Mismatch at index {i}"

    def test_batch_1000(self, rsa_n):
        """1000 multiplies — exercises multiple CUDA blocks."""
        As = [random_mpz_below(rsa_n, seed=300 + i) for i in range(1000)]
        scalars = [((i * 31) % 131072) - 65536 for i in range(1000)]

        results = batch_mod_scalar_mul(As, scalars, rsa_n)

        for i in range(1000):
            expected = (As[i] * scalars[i]) % rsa_n
            assert results[i] == expected, f"Mismatch at index {i}"

    def test_max_scalar_17bit(self, rsa_n):
        """Scalar at 2^17 - 1 (max for k=16 encoding)."""
        A = random_mpz_below(rsa_n, seed=400)
        s = (1 << 17) - 1

        expected = (A * s) % rsa_n
        result = batch_mod_scalar_mul([A], [s], rsa_n)
        assert result[0] == expected

    def test_large_scalar_24bit(self, rsa_n):
        """Scalar at 2^24 (for k=24 encoding)."""
        A = random_mpz_below(rsa_n, seed=401)
        s = (1 << 24)

        expected = (A * s) % rsa_n
        result = batch_mod_scalar_mul([A], [s], rsa_n)
        assert result[0] == expected


# ── High-level plaintext multiply ─────────────────────────────────────

class TestGPUPlaintextMul:

    def test_basic(self, rsa_n):
        ct_N = [random_mpz_below(rsa_n, seed=500 + i) for i in range(4)]
        ct_D = [random_mpz_below(rsa_n, seed=600 + i) for i in range(4)]
        plain_nums = [100, -200, 300, -400]
        plain_den = 65536  # 2^16

        new_N, new_D = gpu_batch_plaintext_mul(
            ct_N, ct_D, plain_nums, plain_den, rsa_n
        )

        for i in range(4):
            expected_N = (ct_N[i] * plain_nums[i]) % rsa_n
            expected_D = (ct_D[i] * plain_den) % rsa_n
            assert new_N[i] == expected_N, f"N mismatch at {i}"
            assert new_D[i] == expected_D, f"D mismatch at {i}"


# ── Benchmark ─────────────────────────────────────────────────────────

class TestBenchmark:

    @pytest.mark.parametrize("batch_size", [1000, 10000, 100000])
    def test_gpu_vs_cpu_benchmark(self, rsa_n, batch_size):
        """Compare GPU batch vs CPU serial. Reports speedup."""
        As = [random_mpz_below(rsa_n, seed=700 + i) for i in range(batch_size)]
        scalars = [((i * 31) % 131072) - 65536 for i in range(batch_size)]

        # CPU serial (gmpy2)
        t0 = time.perf_counter()
        cpu_results = [(As[i] * scalars[i]) % rsa_n for i in range(batch_size)]
        cpu_time = time.perf_counter() - t0

        # GPU batch (includes data transfer)
        t0 = time.perf_counter()
        gpu_results = batch_mod_scalar_mul(As, scalars, rsa_n)
        gpu_time = time.perf_counter() - t0

        # Verify correctness
        mismatches = sum(1 for i in range(batch_size) if gpu_results[i] != cpu_results[i])
        assert mismatches == 0, f"{mismatches}/{batch_size} mismatches"

        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
        print(f"\n  batch={batch_size}: CPU={cpu_time*1000:.1f}ms, "
              f"GPU={gpu_time*1000:.1f}ms, speedup={speedup:.2f}x")
