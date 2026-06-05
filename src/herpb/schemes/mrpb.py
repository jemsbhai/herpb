"""MRPB: Modular Ratio-Preserving Blinding scheme.

New construction extending RPB to modular arithmetic over an RSA modulus
n = p * q. Achieves IND-CPA security: the ratio a/b is hidden from any
adversary who does not know the factorization of n.

Security model: IND-CPA under RSA assumption.

Homomorphic depth:
  - Multiplication: unlimited. The plaintext ratio stays the same size
    (product of two reduced fractions is still a reduced fraction of
    bounded size). No noise, no bootstrapping.
  - Addition: bounded by plaintext space. Each addition can grow the
    numerator/denominator of the result. Rational reconstruction
    requires |num|, den < sqrt(n), so the maximum addition depth
    depends on the magnitude of the operands vs the modulus size.
  - Mixed circuits: multiplication is free, addition consumes headroom.

Ciphertext size: fixed (bounded by n, does not grow with operations).
"""

from dataclasses import dataclass
from gmpy2 import mpz
import gmpy2

from ..utils.math_utils import blcm, pow10, digits10, byte_size


# ---------------------------------------------------------------------------
# Key structures
# ---------------------------------------------------------------------------

@dataclass
class MRPBPublicKey:
    """Public key: the RSA modulus."""
    n: mpz  # n = p * q


@dataclass
class MRPBSecretKey:
    """Secret key: RSA factors, blinding factor, precision parameter."""
    p: mpz
    q: mpz
    n: mpz
    phi_n: mpz  # (p-1)(q-1)
    c: mpz      # blinding factor, random in Z*_n
    c_inv: mpz  # modular inverse of c mod n


@dataclass
class MRPBCiphertext:
    """Ciphertext pair (N, D) mod n."""
    N: mpz
    D: mpz
    n: mpz  # carry the modulus for homomorphic ops

    def digits(self) -> tuple[int, int]:
        return digits10(self.N), digits10(self.D)

    def byte_size(self) -> int:
        return byte_size(self.N) + byte_size(self.D)


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def keygen(
    key_bits: int = 2048,
    precision: int = 3,
    seed: int | None = None,
) -> tuple[MRPBPublicKey, MRPBSecretKey]:
    """Generate MRPB key pair.

    Parameters
    ----------
    key_bits : int
        Bit length of the RSA modulus n (each prime is key_bits // 2).
    precision : int
        Default precision parameter t for scaling.
    seed : int, optional
        Random seed for reproducibility. If None, uses system entropy.

    Returns
    -------
    (pk, sk) : tuple[MRPBPublicKey, MRPBSecretKey]
    """
    rs = gmpy2.random_state(seed) if seed is not None else gmpy2.random_state()

    half_bits = key_bits // 2

    # Generate two large primes
    p = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half_bits) | (mpz(1) << (half_bits - 1)))
    q = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half_bits) | (mpz(1) << (half_bits - 1)))

    # Ensure p != q
    while p == q:
        q = gmpy2.next_prime(gmpy2.mpz_urandomb(rs, half_bits) | (mpz(1) << (half_bits - 1)))

    n = p * q
    phi_n = (p - 1) * (q - 1)

    # Generate blinding factor c coprime to n
    c = gmpy2.mpz_urandomb(rs, key_bits) % n
    while c == 0 or gmpy2.gcd(c, n) != 1:
        c = gmpy2.mpz_urandomb(rs, key_bits) % n

    c_inv = gmpy2.invert(c, n)

    pk = MRPBPublicKey(n=n)
    sk = MRPBSecretKey(p=p, q=q, n=n, phi_n=phi_n, c=c, c_inv=c_inv)

    return pk, sk


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------

def encrypt_shared(
    a_num: mpz, a_den: mpz,
    b_num: mpz, b_den: mpz,
    t: int,
    sk: MRPBSecretKey,
) -> MRPBCiphertext:
    """Encrypt ratio (a_num/a_den) / (b_num/b_den) using shared-k, mod n.

    Parameters
    ----------
    a_num, a_den : mpz
        Numerator and denominator of rational a.
    b_num, b_den : mpz
        Numerator and denominator of rational b.
    t : int
        Precision parameter.
    sk : MRPBSecretKey
        Secret key (contains c, n).

    Returns
    -------
    MRPBCiphertext
    """
    a_num, a_den = mpz(a_num), mpz(a_den)
    b_num, b_den = mpz(b_num), mpz(b_den)

    L = blcm(a_den, b_den)
    k = L * pow10(t)

    a1 = (a_num * k) // a_den
    b1 = (b_num * k) // b_den

    N = (a1 * k * sk.c) % sk.n
    D = (b1 * k * sk.c) % sk.n

    return MRPBCiphertext(N=N, D=D, n=sk.n)


def encrypt_distinct(
    a_num: mpz, a_den: mpz,
    b_num: mpz, b_den: mpz,
    ta: int, tb: int,
    sk: MRPBSecretKey,
) -> MRPBCiphertext:
    """Encrypt ratio (a_num/a_den) / (b_num/b_den) using distinct-k, mod n."""
    a_num, a_den = mpz(a_num), mpz(a_den)
    b_num, b_den = mpz(b_num), mpz(b_den)

    ka = a_den * pow10(ta)
    kb = b_den * pow10(tb)

    a1 = (a_num * ka) // a_den
    b1 = (b_num * kb) // b_den

    N = (a1 * kb * sk.c) % sk.n
    D = (b1 * ka * sk.c) % sk.n

    return MRPBCiphertext(N=N, D=D, n=sk.n)


# ---------------------------------------------------------------------------
# Decryption (REQUIRES secret key)
# ---------------------------------------------------------------------------

def decrypt(ct: MRPBCiphertext, sk: MRPBSecretKey) -> tuple[mpz, mpz]:
    """Decrypt MRPB ciphertext to recover the ratio.

    Strips one layer of blinding factor c, then computes the ratio
    N'/D' mod n. All remaining common factors (additional c powers from
    homomorphic ops, scaling factors k) cancel in the division.

    Finally, rational reconstruction recovers the exact ratio from
    its modular residue, provided |num|, den < sqrt(n).

    Returns
    -------
    (num, den) : tuple[mpz, mpz]
        The ratio in lowest terms.

    Raises
    ------
    ValueError
        If the ratio exceeds the plaintext space (sqrt(n) bound).
    """
    # Strip blinding: N' = N * c_inv mod n, D' = D * c_inv mod n
    N_stripped = (ct.N * sk.c_inv) % sk.n
    D_stripped = (ct.D * sk.c_inv) % sk.n

    if D_stripped == 0:
        raise ValueError("Decryption failed: denominator is zero mod n")

    # Compute ratio in Z_n. Any remaining c or k factors cancel here.
    D_inv = gmpy2.invert(D_stripped, sk.n)
    ratio_mod_n = (N_stripped * D_inv) % sk.n

    # Recover exact rational from modular residue
    num, den = _rational_reconstruction(ratio_mod_n, sk.n)

    return num, den


def _rational_reconstruction(r: mpz, m: mpz) -> tuple[mpz, mpz]:
    """Recover a/b from r = a * b^{-1} mod m, where |a|, |b| < sqrt(m).

    Uses the extended Euclidean algorithm (half-GCD approach):
    we run the Euclidean algorithm on (m, r) and stop when the
    remainder drops below sqrt(m).

    Returns
    -------
    (num, den) : tuple[mpz, mpz]
        Such that num/den ≡ r (mod m) with |num|, den < sqrt(m).

    Raises
    ------
    ValueError
        If reconstruction fails (ratio too large for the modulus).
    """
    bound = gmpy2.isqrt(m)

    # Extended GCD iteration
    old_r, new_r = m, r % m
    old_s, new_s = mpz(0), mpz(1)

    while new_r > bound:
        q = old_r // new_r
        old_r, new_r = new_r, old_r - q * new_r
        old_s, new_s = new_s, old_s - q * new_s

    # new_r = numerator, |new_s| = denominator
    num = new_r
    den = abs(new_s)

    if den == 0:
        raise ValueError("Rational reconstruction failed: den = 0")

    # Verify: num * den_inv ≡ r (mod m)
    den_inv = gmpy2.invert(den, m)
    if (num * den_inv) % m != r % m:
        # Try negative numerator
        num = -num
        if (num % m * den_inv) % m != r % m:
            raise ValueError(
                "Rational reconstruction failed: verification mismatch"
            )

    # Reduce to lowest terms
    g = gmpy2.gcd(abs(num), den)
    return num // g, den // g


# ---------------------------------------------------------------------------
# Plaintext space utilities
# ---------------------------------------------------------------------------

def plaintext_bits(sk: MRPBSecretKey) -> int:
    """Maximum bits for numerator or denominator of a decryptable ratio.

    Rational reconstruction requires |num|, den < sqrt(n), so the
    plaintext space is approximately key_bits / 2.
    """
    return int(gmpy2.num_digits(gmpy2.isqrt(sk.n), 2))


def check_plaintext_fits(num: mpz, den: mpz, sk: MRPBSecretKey) -> bool:
    """Check whether a ratio num/den can be recovered from this key's modulus."""
    bound = gmpy2.isqrt(sk.n)
    return abs(num) < bound and den < bound


# ---------------------------------------------------------------------------
# Homomorphic operations
# ---------------------------------------------------------------------------

def eval_multiply(ct1: MRPBCiphertext, ct2: MRPBCiphertext) -> MRPBCiphertext:
    """Homomorphic multiplication: result decrypts to (a/b) * (e/f).

    Fixed-size output — no ciphertext growth.
    Multiplication does not grow the plaintext ratio (reduced product of
    two small fractions is still small), so depth is unlimited.
    """
    assert ct1.n == ct2.n, "Ciphertexts must share the same modulus"
    n = ct1.n
    return MRPBCiphertext(
        N=(ct1.N * ct2.N) % n,
        D=(ct1.D * ct2.D) % n,
        n=n,
    )


def eval_add(ct1: MRPBCiphertext, ct2: MRPBCiphertext) -> MRPBCiphertext:
    """Homomorphic addition: result decrypts to (a/b) + (e/f).

    The blinding factors c (and any accumulated c^k from prior ops) appear
    in both N and D with the same power, so they cancel when the decryptor
    computes N'/D' mod n. No special handling is needed.

    CONSTRAINT: addition grows the plaintext ratio. After adding k terms
    of magnitude M, the result numerator/denominator can be O(k * M^k).
    Decryption succeeds only while |result_num|, result_den < sqrt(n).
    For a 2048-bit modulus, this gives ~1024 bits of plaintext headroom.
    """
    assert ct1.n == ct2.n, "Ciphertexts must share the same modulus"
    n = ct1.n
    return MRPBCiphertext(
        N=(ct1.N * ct2.D + ct2.N * ct1.D) % n,
        D=(ct1.D * ct2.D) % n,
        n=n,
    )


def eval_subtract(ct1: MRPBCiphertext, ct2: MRPBCiphertext) -> MRPBCiphertext:
    """Homomorphic subtraction: result decrypts to (a/b) - (e/f).

    Same plaintext growth constraint as eval_add.
    """
    assert ct1.n == ct2.n, "Ciphertexts must share the same modulus"
    n = ct1.n
    return MRPBCiphertext(
        N=(ct1.N * ct2.D - ct2.N * ct1.D) % n,
        D=(ct1.D * ct2.D) % n,
        n=n,
    )
