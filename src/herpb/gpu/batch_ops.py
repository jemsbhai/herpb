"""
GPU-accelerated batch modular scalar multiply via CuPy RawKernel (CUDA C).

Core operation: A[i] * s[i] mod n  for millions of (A, s) pairs.
Each A is a 2048-bit number (64 x uint32 limbs), s is a small scalar.

Requires: cupy with CUDA support.
"""

import numpy as np
import cupy as cp
import gmpy2
from gmpy2 import mpz

LIMBS = 64  # 2048 bits / 32 bits per limb


# =====================================================================
# CUDA C kernel source (ASCII only for Windows compatibility)
# =====================================================================

_KERNEL_SRC = r"""
extern "C" __global__
void mod_scalar_mul(
    const unsigned int* __restrict__ A,
    const unsigned int* __restrict__ scalars,
    const unsigned int* __restrict__ n,
    unsigned int* __restrict__ out,
    const int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    unsigned int s = scalars[idx];

    /* Zero scalar -> zero output */
    if (s == 0u) {
        for (int j = 0; j < 64; j++) out[idx * 64 + j] = 0u;
        return;
    }

    /* -- Step 1: Multiply A * s -> product (66 limbs) -- */
    unsigned int product[66];
    unsigned long long carry = 0ULL;
    unsigned long long s64 = (unsigned long long)s;
    for (int j = 0; j < 64; j++) {
        unsigned long long val = (unsigned long long)A[idx * 64 + j] * s64 + carry;
        product[j] = (unsigned int)(val & 0xFFFFFFFFULL);
        carry = val >> 32;
    }
    product[64] = (unsigned int)(carry & 0xFFFFFFFFULL);
    product[65] = 0u;

    /* -- Step 2: Estimate quotient from top limbs -- */
    unsigned long long top_n = (unsigned long long)n[63];
    unsigned long long q_hat = 0ULL;
    if (top_n > 0ULL) {
        unsigned long long top_p =
            ((unsigned long long)product[64] << 32) |
             (unsigned long long)product[63];
        q_hat = top_p / (top_n + 1ULL);
    }

    /* -- Step 3: Subtract q_hat * n from product -- */
    unsigned long long borrow = 0ULL;
    carry = 0ULL;
    for (int j = 0; j < 64; j++) {
        unsigned long long qn_j = (unsigned long long)n[j] * q_hat + carry;
        unsigned int qn_lo = (unsigned int)(qn_j & 0xFFFFFFFFULL);
        carry = qn_j >> 32;

        unsigned long long p_val = (unsigned long long)product[j];
        unsigned long long sub  = (unsigned long long)qn_lo + borrow;
        if (p_val >= sub) {
            product[j] = (unsigned int)(p_val - sub);
            borrow = 0ULL;
        } else {
            product[j] = (unsigned int)(p_val + 0x100000000ULL - sub);
            borrow = 1ULL;
        }
    }

    /* Handle limb 64 */
    unsigned long long qn_hi = carry;
    unsigned long long p_hi  = (unsigned long long)product[64];
    unsigned long long sub_hi = qn_hi + borrow;
    int undershot = (p_hi < sub_hi) ? 1 : 0;
    product[64] = undershot ? 0u : (unsigned int)(p_hi - sub_hi);
    product[65] = 0u;

    /* If undershot, add n back once */
    if (undershot) {
        carry = 0ULL;
        for (int j = 0; j < 64; j++) {
            unsigned long long val =
                (unsigned long long)product[j] +
                (unsigned long long)n[j] + carry;
            product[j] = (unsigned int)(val & 0xFFFFFFFFULL);
            carry = val >> 32;
        }
        product[64] = (unsigned int)((unsigned long long)product[64] + carry);
    }

    /* -- Step 4: While product >= n, subtract n (<= 3 corrections) -- */
    for (int corr = 0; corr < 4; corr++) {
        int ge = 0;
        if (product[65] > 0u || product[64] > 0u) {
            ge = 1;
        } else {
            int decided = 0;
            for (int j = 63; j >= 0; j--) {
                if (product[j] > n[j]) { ge = 1; decided = 1; break; }
                if (product[j] < n[j]) { ge = 0; decided = 1; break; }
            }
            if (!decided) ge = 1;   /* equal -> subtract to get 0 */
        }
        if (!ge) break;

        borrow = 0ULL;
        for (int j = 0; j < 64; j++) {
            unsigned long long a_val = (unsigned long long)product[j];
            unsigned long long b_val = (unsigned long long)n[j] + borrow;
            if (a_val >= b_val) {
                product[j] = (unsigned int)(a_val - b_val);
                borrow = 0ULL;
            } else {
                product[j] = (unsigned int)(a_val + 0x100000000ULL - b_val);
                borrow = 1ULL;
            }
        }
        product[64] = 0u;
        product[65] = 0u;
    }

    /* -- Store result -- */
    for (int j = 0; j < 64; j++) {
        out[idx * 64 + j] = product[j];
    }
}
"""

# Compile once on first import
_kernel = cp.RawKernel(_KERNEL_SRC, "mod_scalar_mul")


# =====================================================================
# Conversion: gmpy2.mpz <-> uint32 limb arrays
# =====================================================================

def mpz_to_limbs(x: mpz, n_limbs: int = LIMBS) -> np.ndarray:
    """Convert non-negative gmpy2.mpz to little-endian uint32 limb array."""
    limbs = np.zeros(n_limbs, dtype=np.uint32)
    val = int(x)
    for i in range(n_limbs):
        limbs[i] = val & 0xFFFFFFFF
        val >>= 32
    return limbs


def limbs_to_mpz(limbs: np.ndarray) -> mpz:
    """Convert little-endian uint32 limb array to gmpy2.mpz."""
    result = 0
    for i in range(len(limbs) - 1, -1, -1):
        result = (result << 32) | int(limbs[i])
    return gmpy2.mpz(result)


def batch_mpz_to_limbs(values: list[mpz], n_limbs: int = LIMBS) -> np.ndarray:
    """Convert list of mpz to 2D limb array (batch, n_limbs)."""
    arr = np.zeros((len(values), n_limbs), dtype=np.uint32)
    for i, v in enumerate(values):
        arr[i] = mpz_to_limbs(v, n_limbs)
    return arr


# =====================================================================
# Python wrapper
# =====================================================================

def batch_mod_scalar_mul(
    A_values: list[mpz],
    scalars: list[int],
    n_mod: mpz,
    threads_per_block: int = 256,
) -> list[mpz]:
    """Compute A[i] * scalars[i] mod n for all i, on GPU.

    Handles negative scalars: result = n - (A * |s| mod n).
    """
    batch = len(A_values)
    if batch == 0:
        return []

    # Separate signs
    abs_scalars = np.array([abs(s) for s in scalars], dtype=np.uint32)
    signs = [s < 0 for s in scalars]

    # Convert to limb arrays and send to GPU
    A_limbs = batch_mpz_to_limbs(A_values)          # (batch, 64) uint32
    n_limbs = mpz_to_limbs(n_mod)                    # (64,) uint32

    d_A = cp.asarray(A_limbs.ravel())                # flatten for kernel
    d_scalars = cp.asarray(abs_scalars)
    d_n = cp.asarray(n_limbs)
    d_out = cp.zeros(batch * LIMBS, dtype=cp.uint32)

    # Launch kernel
    blocks = (batch + threads_per_block - 1) // threads_per_block
    _kernel(
        (blocks,), (threads_per_block,),
        (d_A, d_scalars, d_n, d_out, np.int32(batch)),
    )

    # Transfer back and reshape
    out_flat = cp.asnumpy(d_out)
    out_limbs = out_flat.reshape(batch, LIMBS)

    # Convert to mpz + handle signs
    results = []
    for i in range(batch):
        val = limbs_to_mpz(out_limbs[i])
        if signs[i] and val != 0:
            val = n_mod - val
        results.append(val)

    return results


# =====================================================================
# High-level: batch plaintext-ciphertext multiply for MRPB
# =====================================================================

def gpu_batch_plaintext_mul(
    ct_N_values: list[mpz],
    ct_D_values: list[mpz],
    plain_nums: list[int],
    plain_den: int,
    n_mod: mpz,
) -> tuple[list[mpz], list[mpz]]:
    """Batch multiply MRPB ciphertexts by plaintext scalars on GPU.

    For query-private dot product: Enc(q_i * d_i) = (N_i * num mod n, D_i * den mod n).
    """
    dim = len(ct_N_values)
    new_N = batch_mod_scalar_mul(ct_N_values, plain_nums, n_mod)
    den_scalars = [plain_den] * dim
    new_D = batch_mod_scalar_mul(ct_D_values, den_scalars, n_mod)
    return new_N, new_D
