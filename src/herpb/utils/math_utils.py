"""Big-integer math utilities for RPB/MRPB schemes.

Uses gmpy2 for fast arbitrary-precision arithmetic. All functions operate on
gmpy2.mpz integers unless otherwise noted.
"""

import gmpy2
from gmpy2 import mpz


def iabs(x: mpz) -> mpz:
    """Absolute value for mpz."""
    return -x if x < 0 else x


def bgcd(a: mpz, b: mpz) -> mpz:
    """GCD of two mpz integers. Returns 1 if both are zero."""
    g = gmpy2.gcd(iabs(a), iabs(b))
    return g if g != 0 else mpz(1)


def blcm(a: mpz, b: mpz) -> mpz:
    """LCM of two mpz integers."""
    return gmpy2.lcm(iabs(a), iabs(b))


def pow10(k: int) -> mpz:
    """Compute 10^k as mpz. Cached internally by gmpy2."""
    return gmpy2.mpz(10) ** k


def digits10(x: mpz) -> int:
    """Count decimal digits of |x|."""
    if x == 0:
        return 1
    return len(gmpy2.digits(iabs(x), 10))


def reduce_fraction(n: mpz, d: mpz) -> tuple[mpz, mpz]:
    """Reduce n/d to lowest terms with d > 0."""
    if d == 0:
        raise ZeroDivisionError("Denominator is zero")
    if d < 0:
        n, d = -n, -d
    g = bgcd(n, d)
    return n // g, d // g


def cmp_abs_fraction(n1: mpz, d1: mpz, n2: mpz, d2: mpz) -> int:
    """Compare |n1/d1| vs |n2/d2|. Returns -1, 0, or 1."""
    lhs = iabs(n1) * d2
    rhs = iabs(n2) * d1
    if lhs < rhs:
        return -1
    if lhs > rhs:
        return 1
    return 0


def byte_size(x: mpz) -> int:
    """Approximate byte size of an mpz value."""
    if x == 0:
        return 1
    # gmpy2 stores in limbs; num_digits in base 256 ~ byte count
    return (gmpy2.num_digits(iabs(x), 2) + 7) // 8


def random_mpz(bits: int) -> mpz:
    """Generate a random mpz integer with exactly `bits` bits.

    Uses gmpy2's random state seeded from system entropy.
    NOT cryptographically secure — use for benchmarking only.
    """
    rs = gmpy2.random_state()
    x = gmpy2.mpz_urandomb(rs, bits)
    # Ensure the top bit is set so we get exactly `bits` bits
    x = x | (mpz(1) << (bits - 1))
    return x


def random_mpz_range(lo: int, hi: int, exclude_zero: bool = True) -> mpz:
    """Random mpz in [lo, hi]. Optionally excludes zero."""
    import random
    val = 0
    while val == 0 and exclude_zero:
        val = random.randint(lo, hi)
    return mpz(val)
