"""RPB: Ratio-Preserving Blinding scheme.

Ported from C++ reference implementation. Provides operand-hiding encryption
for rational numbers. The ratio a/b is recoverable via GCD reduction (by
anyone), but individual operands a, b are hidden behind blinding factor c.

Security model: operand-hiding (NOT IND-CPA — the ratio leaks).
Homomorphic depth: unlimited (no noise, no bootstrapping).
Ciphertext growth: unbounded (grows with each operation).
"""

from dataclasses import dataclass
from gmpy2 import mpz

from ..utils.math_utils import (
    blcm, pow10, digits10, byte_size, reduce_fraction,
    iabs, cmp_abs_fraction,
)


@dataclass
class Rational:
    """A rational number num/den."""
    num: mpz
    den: mpz

    def __post_init__(self):
        self.num = mpz(self.num)
        self.den = mpz(self.den)
        if self.den == 0:
            raise ValueError("Denominator cannot be zero")


@dataclass
class RPBCiphertext:
    """Ciphertext pair (N, D) from RPB encryption."""
    N: mpz
    D: mpz

    def digits(self) -> tuple[int, int]:
        return digits10(self.N), digits10(self.D)

    def byte_size(self) -> int:
        return byte_size(self.N) + byte_size(self.D)


# ---------------------------------------------------------------------------
# Encryption: Shared-k variant
# ---------------------------------------------------------------------------

def shared_k_encrypt(
    a: Rational,
    b: Rational,
    t: int,
    c: mpz,
) -> RPBCiphertext:
    """Encrypt ratio a/b using the shared-k method.

    Parameters
    ----------
    a, b : Rational
        The two rationals whose ratio a/b is being encrypted.
    t : int
        Precision parameter (scaling exponent).
    c : mpz
        Blinding factor (must be nonzero).

    Returns
    -------
    RPBCiphertext
        (N, D) where GCD-reduce(N, D) = a/b.
    """
    if c == 0:
        raise ValueError("Blinding factor c must be nonzero")

    L = blcm(a.den, b.den)
    k = L * pow10(t)

    a1 = (a.num * k) // a.den  # exact because L is LCM
    b1 = (b.num * k) // b.den

    N = a1 * k * c
    D = b1 * k * c

    return RPBCiphertext(N=N, D=D)


# ---------------------------------------------------------------------------
# Encryption: Distinct-k variant
# ---------------------------------------------------------------------------

def distinct_k_encrypt(
    a: Rational,
    b: Rational,
    ta: int,
    tb: int,
    c: mpz,
) -> RPBCiphertext:
    """Encrypt ratio a/b using the distinct-k method.

    Parameters
    ----------
    a, b : Rational
        The two rationals whose ratio a/b is being encrypted.
    ta, tb : int
        Per-operand precision parameters.
    c : mpz
        Blinding factor (must be nonzero).

    Returns
    -------
    RPBCiphertext
        (N, D) where GCD-reduce(N, D) = a/b.
    """
    if c == 0:
        raise ValueError("Blinding factor c must be nonzero")

    ka = a.den * pow10(ta)
    kb = b.den * pow10(tb)

    a1 = (a.num * ka) // a.den  # = a.num * 10^ta
    b1 = (b.num * kb) // b.den  # = b.num * 10^tb

    N = a1 * kb * c  # cross-multiplied
    D = b1 * ka * c

    return RPBCiphertext(N=N, D=D)


# ---------------------------------------------------------------------------
# Decryption (public — no key needed)
# ---------------------------------------------------------------------------

def decrypt(ct: RPBCiphertext) -> tuple[mpz, mpz]:
    """Decrypt by GCD reduction. Returns (num, den) in lowest terms.

    NOTE: This is a PUBLIC operation — anyone with the ciphertext can do it.
    This is the fundamental limitation of RPB vs MRPB.
    """
    return reduce_fraction(ct.N, ct.D)


# ---------------------------------------------------------------------------
# Homomorphic operations
# ---------------------------------------------------------------------------

def eval_multiply(ct1: RPBCiphertext, ct2: RPBCiphertext) -> RPBCiphertext:
    """Homomorphic multiplication: result decrypts to (a/b) * (e/f)."""
    return RPBCiphertext(N=ct1.N * ct2.N, D=ct1.D * ct2.D)


def eval_add(ct1: RPBCiphertext, ct2: RPBCiphertext) -> RPBCiphertext:
    """Homomorphic addition: result decrypts to (a/b) + (e/f)."""
    return RPBCiphertext(
        N=ct1.N * ct2.D + ct2.N * ct1.D,
        D=ct1.D * ct2.D,
    )


def eval_subtract(ct1: RPBCiphertext, ct2: RPBCiphertext) -> RPBCiphertext:
    """Homomorphic subtraction: result decrypts to (a/b) - (e/f)."""
    return RPBCiphertext(
        N=ct1.N * ct2.D - ct2.N * ct1.D,
        D=ct1.D * ct2.D,
    )


# ---------------------------------------------------------------------------
# Bounds computation (from C++ reference)
# ---------------------------------------------------------------------------

def max_t_shared(
    a: Rational,
    b: Rational,
    c_abs: mpz,
    bound: mpz,
) -> int:
    """Find max precision t such that shared-k ciphertext stays under bound B.

    The ciphertext magnitude is approximately:
        |a/a_den| * L^2 * 10^(2t) * |c|
    We solve for max t where this <= B.
    """
    L = blcm(a.den, b.den)

    if cmp_abs_fraction(a.num, a.den, b.num, b.den) >= 0:
        max_num, max_den = iabs(a.num), a.den
    else:
        max_num, max_den = iabs(b.num), b.den

    C = L * L * c_abs * max_num
    R = bound * max_den

    if C == 0:
        return 0

    dR = digits10(R)
    dC = digits10(C)
    hi_est = max(0, (dR - dC) // 2 + 6)

    lo, hi, ans = 0, hi_est, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if C * pow10(2 * mid) <= R:
            ans = mid
            lo = mid + 1
        else:
            if mid == 0:
                break
            hi = mid - 1
    return ans


def max_T_distinct(
    a: Rational,
    b: Rational,
    c_abs: mpz,
    bound: mpz,
) -> int:
    """Find max total precision T = ta + tb for distinct-k under bound B."""
    if cmp_abs_fraction(a.num, a.den, b.num, b.den) >= 0:
        max_num, max_den = iabs(a.num), a.den
    else:
        max_num, max_den = iabs(b.num), b.den

    C = c_abs * a.den * b.den * max_num
    R = bound * max_den

    if C == 0:
        return 0

    dR = digits10(R)
    dC = digits10(C)
    hi_est = max(0, (dR - dC) + 6)

    lo, hi, ans = 0, hi_est, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if C * pow10(mid) <= R:
            ans = mid
            lo = mid + 1
        else:
            if mid == 0:
                break
            hi = mid - 1
    return ans


def split_T(T: int, Pa: int, Pb: int) -> tuple[int, int]:
    """Split total precision T into (ta, tb) respecting caps Pa, Pb."""
    ta = min(T, Pa)
    rem = max(0, T - Pa)
    tb = min(rem, Pb)
    return ta, tb
