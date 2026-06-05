"""
GPU Montgomery multiplication for 2048-bit modular arithmetic (CuPy CUDA C).

Full ciphertext x ciphertext modular multiply on GPU using the CIOS
(Coarsely Integrated Operand Scanning) variant of Montgomery multiplication.

Each CUDA thread computes one independent a*b mod n — no inter-thread
communication needed. For a 384-dim dot product over 10K documents,
that is 3.84M independent Montgomery multiplies in a single kernel launch.

CIOS Montgomery multiply:
  1. Interleaves multiplication and reduction in the inner loop
  2. Never materializes the full 4096-bit product
  3. Working storage: 66 uint32 words per thread (registers/local memory)
  4. O(S^2) = O(64^2) = 4096 multiply-accumulates per call

Requires: cupy with CUDA support.
"""

import time
import numpy as np
import cupy as cp
import gmpy2
from gmpy2 import mpz

from .batch_ops import (
    mpz_to_limbs,
    limbs_to_mpz,
    batch_mpz_to_limbs,
    LIMBS,
    LIMB_BYTES,
)


# =====================================================================
# CUDA C kernel: CIOS Montgomery Multiplication
# =====================================================================

_MONT_MUL_SRC = r"""
#define S 64

extern "C" __global__
void mont_mul_batch(
    const unsigned int* __restrict__ a,
    const unsigned int* __restrict__ b,
    const unsigned int* __restrict__ n,
    unsigned int* __restrict__ out,
    const unsigned int n0_prime,
    const int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const int off = idx * S;
    unsigned int t[S + 2];
    for (int k = 0; k < S + 2; k++) t[k] = 0u;

    for (int i = 0; i < S; i++) {
        unsigned long long bi = (unsigned long long)b[off + i];

        /* --- Multiply: t += a * b[i] --- */
        unsigned long long C = 0ULL;
        for (int j = 0; j < S; j++) {
            C += (unsigned long long)t[j]
               + (unsigned long long)a[off + j] * bi;
            t[j] = (unsigned int)(C & 0xFFFFFFFFULL);
            C >>= 32;
        }
        C += (unsigned long long)t[S];
        t[S]     = (unsigned int)(C & 0xFFFFFFFFULL);
        t[S + 1] = (unsigned int)(C >> 32);

        /* --- Reduce: m = t[0]*n' mod 2^32, then t = (t + m*n) >> 32 --- */
        unsigned long long m = (unsigned long long)(t[0] * n0_prime);

        C = (unsigned long long)t[0] + m * (unsigned long long)n[0];
        C >>= 32;

        for (int j = 1; j < S; j++) {
            C += (unsigned long long)t[j]
               + m * (unsigned long long)n[j];
            t[j - 1] = (unsigned int)(C & 0xFFFFFFFFULL);
            C >>= 32;
        }
        C += (unsigned long long)t[S];
        t[S - 1] = (unsigned int)(C & 0xFFFFFFFFULL);
        C >>= 32;

        t[S]     = (unsigned int)((unsigned long long)t[S + 1] + C);
        t[S + 1] = 0u;
    }

    /* --- Final subtraction: if t >= n then t -= n --- */
    int ge = 0;
    if (t[S] > 0u) {
        ge = 1;
    } else {
        int decided = 0;
        for (int j = S - 1; j >= 0; j--) {
            if (t[j] > n[j]) { ge = 1; decided = 1; break; }
            if (t[j] < n[j]) { ge = 0; decided = 1; break; }
        }
        if (!decided) ge = 1;
    }

    if (ge) {
        unsigned long long borrow = 0ULL;
        for (int j = 0; j < S; j++) {
            unsigned long long diff =
                (unsigned long long)t[j]
              - (unsigned long long)n[j] - borrow;
            t[j] = (unsigned int)(diff & 0xFFFFFFFFULL);
            borrow = (diff >> 63) & 1ULL;
        }
    }

    for (int j = 0; j < S; j++) {
        out[off + j] = t[j];
    }
}

/* --- Modular addition: out = (a + b) mod n --- */
extern "C" __global__
void mod_add_batch(
    const unsigned int* __restrict__ a,
    const unsigned int* __restrict__ b,
    const unsigned int* __restrict__ n,
    unsigned int* __restrict__ out,
    const int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const int off = idx * S;
    unsigned int t[S + 1];

    /* t = a + b */
    unsigned long long carry = 0ULL;
    for (int j = 0; j < S; j++) {
        carry += (unsigned long long)a[off + j]
               + (unsigned long long)b[off + j];
        t[j] = (unsigned int)(carry & 0xFFFFFFFFULL);
        carry >>= 32;
    }
    t[S] = (unsigned int)carry;

    /* if t >= n, subtract n */
    int ge = 0;
    if (t[S] > 0u) {
        ge = 1;
    } else {
        int decided = 0;
        for (int j = S - 1; j >= 0; j--) {
            if (t[j] > n[j]) { ge = 1; decided = 1; break; }
            if (t[j] < n[j]) { ge = 0; decided = 1; break; }
        }
        if (!decided) ge = 1;
    }

    if (ge) {
        unsigned long long borrow = 0ULL;
        for (int j = 0; j < S; j++) {
            unsigned long long diff =
                (unsigned long long)t[j]
              - (unsigned long long)n[j] - borrow;
            t[j] = (unsigned int)(diff & 0xFFFFFFFFULL);
            borrow = (diff >> 63) & 1ULL;
        }
    }

    for (int j = 0; j < S; j++) {
        out[off + j] = t[j];
    }
}
"""

_mont_mul_kernel = cp.RawKernel(_MONT_MUL_SRC, "mont_mul_batch")
_mod_add_kernel = cp.RawKernel(_MONT_MUL_SRC, "mod_add_batch")


# =====================================================================
# Montgomery parameter precomputation (CPU, one-time)
# =====================================================================

class MontgomeryContext:
    """Precomputed Montgomery constants for a given modulus n.

    R = 2^2048.  All constants are stored as both mpz (for CPU fallback)
    and CuPy device arrays (for GPU operations).
    """

    def __init__(self, n_mod: mpz):
        self.n = n_mod
        self.n_bits = int(gmpy2.num_digits(n_mod, 2))

        R = mpz(1) << (LIMBS * 32)       # R = 2^2048

        # n' = -n^{-1} mod 2^32
        W = mpz(1) << 32
        n_inv = gmpy2.invert(n_mod, W)    # n * n_inv = 1 mod W
        self.n0_prime = int((-n_inv) % W)  # n * n0' = -1 mod W

        # R mod n  and  R^2 mod n
        self.R_mod_n = R % n_mod
        self.R2_mod_n = (R * R) % n_mod

        # GPU-side constants (immutable after init)
        self.d_n = cp.asarray(mpz_to_limbs(n_mod))
        self.d_R2 = cp.asarray(mpz_to_limbs(self.R2_mod_n))
        self.d_one = cp.asarray(mpz_to_limbs(self.R_mod_n))  # 1 in Montgomery form = R mod n

    def to_mont(self, x: mpz) -> mpz:
        """Convert x to Montgomery form: x * R mod n (CPU)."""
        return (x * self.R_mod_n) % self.n  # or use REDC(x * R^2)

    def from_mont(self, x_mont: mpz) -> mpz:
        """Convert from Montgomery form: x_mont * R^{-1} mod n (CPU)."""
        R_inv = gmpy2.invert(mpz(1) << (LIMBS * 32), self.n)
        return (x_mont * R_inv) % self.n


# =====================================================================
# GPU batch operations
# =====================================================================

def _launch_mont_mul(d_a, d_b, d_n, n0_prime, batch, tpb=256):
    """Launch Montgomery multiply kernel. Returns device array."""
    d_out = cp.zeros(batch * LIMBS, dtype=cp.uint32)
    blocks = (batch + tpb - 1) // tpb
    _mont_mul_kernel(
        (blocks,), (tpb,),
        (d_a, d_b, d_n, d_out, np.uint32(n0_prime), np.int32(batch)),
    )
    return d_out


def _launch_mod_add(d_a, d_b, d_n, batch, tpb=256):
    """Launch modular addition kernel. Returns device array."""
    d_out = cp.zeros(batch * LIMBS, dtype=cp.uint32)
    blocks = (batch + tpb - 1) // tpb
    _mod_add_kernel(
        (blocks,), (tpb,),
        (d_a, d_b, d_n, d_out, np.int32(batch)),
    )
    return d_out


def gpu_to_montgomery(values_limbs: cp.ndarray, ctx: MontgomeryContext) -> cp.ndarray:
    """Convert batch of values to Montgomery form on GPU.

    values_limbs: (batch * LIMBS,) uint32 on device — standard form
    Returns: (batch * LIMBS,) uint32 on device — Montgomery form
    """
    batch = values_limbs.size // LIMBS
    # MontMul(x, R^2) = x * R^2 * R^{-1} mod n = x * R mod n
    d_R2 = cp.tile(ctx.d_R2, batch)  # replicate R^2 for each element
    return _launch_mont_mul(values_limbs, d_R2, ctx.d_n, ctx.n0_prime, batch)


def gpu_from_montgomery(mont_limbs: cp.ndarray, ctx: MontgomeryContext) -> cp.ndarray:
    """Convert batch from Montgomery form on GPU.

    Returns standard form: MontMul(x_mont, 1) = x_mont * R^{-1} mod n = x
    """
    batch = mont_limbs.size // LIMBS
    # Build [1, 0, 0, ..., 1, 0, 0, ...] on CPU, transfer once
    one_cpu = np.zeros(batch * LIMBS, dtype=np.uint32)
    one_cpu[::LIMBS] = 1  # every 64th element = 1
    d_one_std = cp.asarray(one_cpu)
    return _launch_mont_mul(mont_limbs, d_one_std, ctx.d_n, ctx.n0_prime, batch)


def batch_mod_mul_gpu(
    a_values: list[mpz],
    b_values: list[mpz],
    n_mod: mpz,
    ctx: MontgomeryContext = None,
) -> list[mpz]:
    """Compute a[i] * b[i] mod n for all i, fully on GPU via Montgomery.

    End-to-end: convert to Montgomery form, multiply, convert back.
    """
    batch = len(a_values)
    if batch == 0:
        return []

    if ctx is None:
        ctx = MontgomeryContext(n_mod)

    # Convert to limbs and send to GPU
    a_limbs = batch_mpz_to_limbs(a_values)
    b_limbs = batch_mpz_to_limbs(b_values)

    d_a = cp.asarray(a_limbs.ravel())
    d_b = cp.asarray(b_limbs.ravel())

    # Convert to Montgomery form
    d_a_mont = gpu_to_montgomery(d_a, ctx)
    d_b_mont = gpu_to_montgomery(d_b, ctx)

    # Montgomery multiply
    d_c_mont = _launch_mont_mul(d_a_mont, d_b_mont, ctx.d_n, ctx.n0_prime, batch)

    # Convert back from Montgomery form
    d_c = gpu_from_montgomery(d_c_mont, ctx)

    # Transfer to CPU
    c_flat = cp.asnumpy(d_c)
    c_limbs = c_flat.reshape(batch, LIMBS)

    return [limbs_to_mpz(c_limbs[i]) for i in range(batch)]


def batch_mod_mul_gpu_timed(
    a_values: list[mpz],
    b_values: list[mpz],
    n_mod: mpz,
    ctx: MontgomeryContext = None,
) -> tuple[list[mpz], dict]:
    """Same as batch_mod_mul_gpu but with per-phase timing."""
    batch = len(a_values)
    if batch == 0:
        return [], {}

    if ctx is None:
        ctx = MontgomeryContext(n_mod)

    t0 = time.perf_counter()
    a_limbs = batch_mpz_to_limbs(a_values)
    b_limbs = batch_mpz_to_limbs(b_values)
    d_a = cp.asarray(a_limbs.ravel())
    d_b = cp.asarray(b_limbs.ravel())
    cp.cuda.Stream.null.synchronize()
    t_setup = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_a_mont = gpu_to_montgomery(d_a, ctx)
    d_b_mont = gpu_to_montgomery(d_b, ctx)
    cp.cuda.Stream.null.synchronize()
    t_to_mont = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_c_mont = _launch_mont_mul(d_a_mont, d_b_mont, ctx.d_n, ctx.n0_prime, batch)
    cp.cuda.Stream.null.synchronize()
    t_kernel = time.perf_counter() - t0

    t0 = time.perf_counter()
    d_c = gpu_from_montgomery(d_c_mont, ctx)
    cp.cuda.Stream.null.synchronize()
    t_from_mont = time.perf_counter() - t0

    t0 = time.perf_counter()
    c_flat = cp.asnumpy(d_c)
    c_limbs = c_flat.reshape(batch, LIMBS)
    results = [limbs_to_mpz(c_limbs[i]) for i in range(batch)]
    t_readback = time.perf_counter() - t0

    timing = {
        "setup_ms": t_setup * 1000,
        "to_mont_ms": t_to_mont * 1000,
        "kernel_ms": t_kernel * 1000,
        "from_mont_ms": t_from_mont * 1000,
        "readback_ms": t_readback * 1000,
        "total_ms": (t_setup + t_to_mont + t_kernel + t_from_mont + t_readback) * 1000,
    }
    return results, timing
