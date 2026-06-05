"""
Tests for GPU Montgomery multiplication.

Verifies CIOS kernel correctness against gmpy2, then benchmarks
GPU vs CPU for batch 2048-bit modular multiply.
"""

import time
import pytest
import numpy as np
import gmpy2
from gmpy2 import mpz

from herpb.gpu.montgomery import (
    MontgomeryContext,
    batch_mod_mul_gpu,
    batch_mod_mul_gpu_timed,
)


# ── Helpers ───────────────────────────────────────────────────────────

def random_mpz_below(n: mpz, seed: int) -> mpz:
    rs = gmpy2.random_state(seed)
    return gmpy2.mpz_urandomb(rs, int(gmpy2.num_digits(n, 2))) % n


def make_rsa_modulus(bits=2048, seed=42):
    rs = gmpy2.random_state(seed)
    half = bits // 2
    p = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half) | (mpz(1) << (half - 1)))
    q = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half) | (mpz(1) << (half - 1)))
    return p * q


@pytest.fixture(scope="module")
def rsa_n():
    return make_rsa_modulus(2048, seed=42)


@pytest.fixture(scope="module")
def mont_ctx(rsa_n):
    return MontgomeryContext(rsa_n)


# ── Context tests ─────────────────────────────────────────────────────

class TestMontgomeryContext:

    def test_n0_prime_property(self, rsa_n, mont_ctx):
        """n * n0' should be -1 mod 2^32."""
        W = mpz(1) << 32
        product = (rsa_n * mont_ctx.n0_prime) % W
        assert product == W - 1

    def test_R2_mod_n(self, rsa_n, mont_ctx):
        R = mpz(1) << 2048
        assert mont_ctx.R2_mod_n == (R * R) % rsa_n

    def test_to_from_mont_roundtrip(self, rsa_n, mont_ctx):
        for seed in range(10):
            x = random_mpz_below(rsa_n, seed)
            x_mont = mont_ctx.to_mont(x)
            x_back = mont_ctx.from_mont(x_mont)
            assert x_back == x, f"seed={seed}"


# ── Kernel correctness ───────────────────────────────────────────────

class TestMontMulCorrectness:

    def test_single_multiply(self, rsa_n, mont_ctx):
        a = random_mpz_below(rsa_n, seed=100)
        b = random_mpz_below(rsa_n, seed=101)
        expected = (a * b) % rsa_n

        result = batch_mod_mul_gpu([a], [b], rsa_n, mont_ctx)
        assert result[0] == expected

    def test_multiply_by_one(self, rsa_n, mont_ctx):
        a = random_mpz_below(rsa_n, seed=110)
        result = batch_mod_mul_gpu([a], [mpz(1)], rsa_n, mont_ctx)
        assert result[0] == a

    def test_multiply_by_zero(self, rsa_n, mont_ctx):
        a = random_mpz_below(rsa_n, seed=120)
        result = batch_mod_mul_gpu([a], [mpz(0)], rsa_n, mont_ctx)
        assert result[0] == 0

    def test_square(self, rsa_n, mont_ctx):
        a = random_mpz_below(rsa_n, seed=130)
        expected = (a * a) % rsa_n
        result = batch_mod_mul_gpu([a], [a], rsa_n, mont_ctx)
        assert result[0] == expected

    def test_batch_100(self, rsa_n, mont_ctx):
        As = [random_mpz_below(rsa_n, seed=200 + i) for i in range(100)]
        Bs = [random_mpz_below(rsa_n, seed=300 + i) for i in range(100)]

        results = batch_mod_mul_gpu(As, Bs, rsa_n, mont_ctx)

        for i in range(100):
            expected = (As[i] * Bs[i]) % rsa_n
            assert results[i] == expected, f"Mismatch at index {i}"

    def test_batch_1000(self, rsa_n, mont_ctx):
        As = [random_mpz_below(rsa_n, seed=400 + i) for i in range(1000)]
        Bs = [random_mpz_below(rsa_n, seed=500 + i) for i in range(1000)]

        results = batch_mod_mul_gpu(As, Bs, rsa_n, mont_ctx)

        mismatches = 0
        for i in range(1000):
            if results[i] != (As[i] * Bs[i]) % rsa_n:
                mismatches += 1
        assert mismatches == 0, f"{mismatches}/1000 mismatches"

    def test_small_values(self, rsa_n, mont_ctx):
        """Small values that fit in a single limb."""
        pairs = [(mpz(2), mpz(3)), (mpz(7), mpz(11)), (mpz(1000), mpz(9999))]
        for a, b in pairs:
            expected = (a * b) % rsa_n
            result = batch_mod_mul_gpu([a], [b], rsa_n, mont_ctx)
            assert result[0] == expected, f"{a}*{b}: expected {expected}, got {result[0]}"

    def test_near_modulus(self, rsa_n, mont_ctx):
        """Values close to n (tests reduction correctness)."""
        a = rsa_n - 1
        b = rsa_n - 1
        expected = (a * b) % rsa_n
        result = batch_mod_mul_gpu([a], [b], rsa_n, mont_ctx)
        assert result[0] == expected


# ── Benchmark ─────────────────────────────────────────────────────────

class TestMontBenchmark:

    @pytest.mark.parametrize("batch_size", [1000, 10000, 100000])
    def test_gpu_vs_cpu(self, rsa_n, mont_ctx, batch_size):
        """Full 2048x2048 modular multiply: GPU vs CPU."""
        As = [random_mpz_below(rsa_n, seed=700 + i) for i in range(batch_size)]
        Bs = [random_mpz_below(rsa_n, seed=800 + i) for i in range(batch_size)]

        # CPU serial (gmpy2)
        t0 = time.perf_counter()
        cpu_results = [(As[i] * Bs[i]) % rsa_n for i in range(batch_size)]
        cpu_time = time.perf_counter() - t0

        # GPU (Montgomery) with breakdown
        gpu_results, timing = batch_mod_mul_gpu_timed(As, Bs, rsa_n, mont_ctx)

        # Verify
        mismatches = sum(1 for i in range(batch_size) if gpu_results[i] != cpu_results[i])
        assert mismatches == 0, f"{mismatches}/{batch_size} mismatches"

        speedup = cpu_time * 1000 / timing['total_ms'] if timing['total_ms'] > 0 else 0
        kernel_speedup = cpu_time * 1000 / timing['kernel_ms'] if timing['kernel_ms'] > 0 else 0

        print(f"\n  batch={batch_size}:")
        print(f"    CPU (gmpy2):       {cpu_time*1000:.1f}ms")
        print(f"    GPU total:         {timing['total_ms']:.1f}ms  (speedup: {speedup:.2f}x)")
        print(f"    GPU kernel only:   {timing['kernel_ms']:.1f}ms  (speedup: {kernel_speedup:.1f}x)")
        print(f"      setup:     {timing['setup_ms']:.1f}ms")
        print(f"      to_mont:   {timing['to_mont_ms']:.1f}ms")
        print(f"      multiply:  {timing['kernel_ms']:.1f}ms")
        print(f"      from_mont: {timing['from_mont_ms']:.1f}ms")
        print(f"      readback:  {timing['readback_ms']:.1f}ms")
