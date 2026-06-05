"""
Encrypted similarity computation using RPB/MRPB.

Demonstrates that cosine similarity (and dot product for ranking) can be
computed EXACTLY in the encrypted domain.  The ONLY error source is the
initial float-to-rational quantization, which is bounded and one-time.

Protocol (query privacy, documents plaintext on server):
  1. Client encodes query q as rationals via fixed-point quantization
  2. Client encrypts each q_i as MRPB ciphertext (scalar over 1)
  3. Server has plaintext document embeddings, also encoded as rationals
  4. Server encrypts each d_i and computes homomorphic dot product
  5. Server returns encrypted dot product to client
  6. Client decrypts to get the similarity score

For ranking, dot product suffices (monotonic with cosine sim for fixed norms).

KEY ADVANTAGE over CKKS:
  - After encoding, ALL arithmetic is exact — no noise accumulation
  - Dot-product error is bounded by d·2^{-k}, independent of circuit depth
  - No noise budget, no bootstrapping, no Operation-Error Estimation
  - A standard 2048-bit key provides >2^1000 headroom for typical RAG dims
"""

from gmpy2 import mpz
import gmpy2

from ..schemes import rpb, mrpb
from ..schemes.rpb import Rational, RPBCiphertext
from ..schemes.mrpb import MRPBCiphertext, MRPBSecretKey


# ---------------------------------------------------------------------------
# RPB: scalar encryption (encrypt a single value as ratio-over-1)
# ---------------------------------------------------------------------------

def rpb_encrypt_scalar(
    num: int,
    den: int,
    c: mpz,
) -> RPBCiphertext:
    """Encrypt a single rational num/den using RPB.

    Internally encrypts the ratio (num/den) / (1/1) with t=0.
    The blinding factor c hides the operands; decryption (public)
    recovers the ratio num/den.

    Args:
        num: Numerator of the rational (from float_to_rational).
        den: Denominator (2^k from the encoding).
        c: Blinding factor (nonzero).

    Returns:
        RPBCiphertext encrypting num/den.
    """
    a = Rational(num=mpz(num), den=mpz(den))
    b = Rational(num=mpz(1), den=mpz(1))
    return rpb.shared_k_encrypt(a, b, t=0, c=mpz(c))


def rpb_encrypt_vector(
    rationals: list[tuple[int, int]],
    c: mpz,
) -> list[RPBCiphertext]:
    """Encrypt a vector of rationals using RPB.

    Each component is encrypted independently with the same blinding
    factor.  In a production system, each would use a fresh random c.

    Args:
        rationals: List of (numerator, denominator) from encode_vector.
        c: Blinding factor.

    Returns:
        List of RPBCiphertext, one per dimension.
    """
    return [rpb_encrypt_scalar(n, d, c) for n, d in rationals]


# ---------------------------------------------------------------------------
# MRPB: scalar encryption (IND-CPA secure)
# ---------------------------------------------------------------------------

def mrpb_encrypt_scalar(
    num: int,
    den: int,
    sk: MRPBSecretKey,
) -> MRPBCiphertext:
    """Encrypt a single rational num/den using MRPB.

    Internally encrypts the ratio (num/den) / (1/1) with t=0.
    Decryption requires the secret key.

    Args:
        num: Numerator of the rational.
        den: Denominator (2^k from the encoding).
        sk: MRPB secret key.

    Returns:
        MRPBCiphertext encrypting num/den.
    """
    return mrpb.encrypt_shared(
        a_num=mpz(num), a_den=mpz(den),
        b_num=mpz(1), b_den=mpz(1),
        t=0,
        sk=sk,
    )


def mrpb_encrypt_vector(
    rationals: list[tuple[int, int]],
    sk: MRPBSecretKey,
) -> list[MRPBCiphertext]:
    """Encrypt a vector of rationals using MRPB (IND-CPA secure).

    Args:
        rationals: List of (numerator, denominator) from encode_vector.
        sk: MRPB secret key.

    Returns:
        List of MRPBCiphertext, one per dimension.
    """
    return [mrpb_encrypt_scalar(n, d, sk) for n, d in rationals]


# ---------------------------------------------------------------------------
# Homomorphic dot product
# ---------------------------------------------------------------------------

def rpb_dot_product(
    q_cts: list[RPBCiphertext],
    d_cts: list[RPBCiphertext],
) -> RPBCiphertext:
    """Compute dot product of two encrypted vectors using RPB.

    Performs d multiplications and (d-1) additions homomorphically.
    The result decrypts to dot(q, d) exactly (within encoding precision).

    Args:
        q_cts: Encrypted query vector.
        d_cts: Encrypted document vector.

    Returns:
        RPBCiphertext whose decryption is dot(q, d).

    Raises:
        ValueError: If vectors have different lengths or are empty.
    """
    if len(q_cts) != len(d_cts):
        raise ValueError(
            f"Vector length mismatch: {len(q_cts)} vs {len(d_cts)}"
        )
    if len(q_cts) == 0:
        raise ValueError("Cannot compute dot product of empty vectors")

    # Multiply component-wise
    products = [rpb.eval_multiply(qc, dc) for qc, dc in zip(q_cts, d_cts)]

    # Sum all products
    result = products[0]
    for p in products[1:]:
        result = rpb.eval_add(result, p)

    return result


def mrpb_dot_product(
    q_cts: list[MRPBCiphertext],
    d_cts: list[MRPBCiphertext],
) -> MRPBCiphertext:
    """Compute dot product of two encrypted vectors using MRPB.

    Same as rpb_dot_product but operates mod n.  The result is
    decryptable as long as the plaintext ratio stays within sqrt(n).

    For typical RAG parameters (dim=384, k=16, 2048-bit key):
      max |numerator| ≈ 2^41  <<  sqrt(n) ≈ 2^1024
    So the plaintext space is never a bottleneck.

    Args:
        q_cts: Encrypted query vector (MRPB).
        d_cts: Encrypted document vector (MRPB).

    Returns:
        MRPBCiphertext whose decryption is dot(q, d).
    """
    if len(q_cts) != len(d_cts):
        raise ValueError(
            f"Vector length mismatch: {len(q_cts)} vs {len(d_cts)}"
        )
    if len(q_cts) == 0:
        raise ValueError("Cannot compute dot product of empty vectors")

    products = [mrpb.eval_multiply(qc, dc) for qc, dc in zip(q_cts, d_cts)]

    result = products[0]
    for p in products[1:]:
        result = mrpb.eval_add(result, p)

    return result


# ---------------------------------------------------------------------------
# Plaintext verification
# ---------------------------------------------------------------------------

def plaintext_dot_product(
    q_rationals: list[tuple[int, int]],
    d_rationals: list[tuple[int, int]],
) -> float:
    """Compute dot product from encoded rationals (for verification).

    Returns the exact float value of
        sum_i (q_num_i / q_den_i) * (d_num_i / d_den_i)
    """
    total = 0.0
    for (qn, qd), (dn, dd) in zip(q_rationals, d_rationals):
        total += (qn / qd) * (dn / dd)
    return total


def plaintext_dot_product_exact(
    q_rationals: list[tuple[int, int]],
    d_rationals: list[tuple[int, int]],
) -> tuple[int, int]:
    """Exact integer dot product from encoded rationals.

    Returns (numerator_sum, denominator) where all rationals share
    the same denominator (2^k from encoding).

    Since all components share denominator D = 2^k:
        dot = sum(q_num_i * d_num_i) / D^2
    """
    if not q_rationals:
        return (0, 1)

    D = q_rationals[0][1]  # shared denominator 2^k
    num_sum = sum(qn * dn for (qn, _), (dn, _) in zip(q_rationals, d_rationals))
    return (num_sum, D * D)


# ---------------------------------------------------------------------------
# Full pipeline: encode → encrypt → dot → decrypt → verify
# ---------------------------------------------------------------------------

def rpb_similarity_pipeline(
    q_floats: list[float],
    d_floats: list[float],
    precision_bits: int = 16,
    c: mpz | None = None,
) -> dict:
    """End-to-end RPB dot product: encode, encrypt, compute, decrypt, verify.

    Args:
        q_floats: Query embedding (float vector).
        d_floats: Document embedding (float vector).
        precision_bits: Encoding precision k.
        c: Blinding factor (random if None).

    Returns:
        Dict with 'true_dot', 'encoded_dot', 'encrypted_dot',
        'quantization_error', and 'encryption_error'.
    """
    from ..utils.encoding import encode_vector

    if c is None:
        rs = gmpy2.random_state()
        c = gmpy2.mpz_urandomb(rs, 256)
        while c == 0:
            c = gmpy2.mpz_urandomb(rs, 256)

    # True dot product (float)
    true_dot = sum(a * b for a, b in zip(q_floats, d_floats))

    # Encode
    q_rats = encode_vector(q_floats, precision_bits)
    d_rats = encode_vector(d_floats, precision_bits)

    # Plaintext encoded dot product (quantization error only)
    encoded_dot = plaintext_dot_product(q_rats, d_rats)

    # Encrypt both vectors
    q_cts = rpb_encrypt_vector(q_rats, c)
    d_cts = rpb_encrypt_vector(d_rats, c)

    # Homomorphic dot product
    dot_ct = rpb_dot_product(q_cts, d_cts)

    # Decrypt
    num, den = rpb.decrypt(dot_ct)
    encrypted_dot = float(num) / float(den)

    return {
        "true_dot": true_dot,
        "encoded_dot": encoded_dot,
        "encrypted_dot": encrypted_dot,
        "quantization_error": abs(true_dot - encoded_dot),
        "encryption_error": abs(encoded_dot - encrypted_dot),
    }


def mrpb_similarity_pipeline(
    q_floats: list[float],
    d_floats: list[float],
    precision_bits: int = 16,
    sk: MRPBSecretKey | None = None,
    key_bits: int = 2048,
) -> dict:
    """End-to-end MRPB dot product: encode, encrypt, compute, decrypt, verify.

    Args:
        q_floats: Query embedding (float vector).
        d_floats: Document embedding (float vector).
        precision_bits: Encoding precision k.
        sk: MRPB secret key (generated if None).
        key_bits: Key size if generating (default 2048).

    Returns:
        Dict with 'true_dot', 'encoded_dot', 'encrypted_dot',
        'quantization_error', and 'encryption_error'.
        'encryption_error' should be 0.0 (exact arithmetic).
    """
    from ..utils.encoding import encode_vector

    if sk is None:
        _, sk = mrpb.keygen(key_bits=key_bits, precision=0, seed=42)

    # True dot product (float)
    true_dot = sum(a * b for a, b in zip(q_floats, d_floats))

    # Encode
    q_rats = encode_vector(q_floats, precision_bits)
    d_rats = encode_vector(d_floats, precision_bits)

    # Plaintext encoded dot product
    encoded_dot = plaintext_dot_product(q_rats, d_rats)

    # Encrypt both vectors
    q_cts = mrpb_encrypt_vector(q_rats, sk)
    d_cts = mrpb_encrypt_vector(d_rats, sk)

    # Homomorphic dot product
    dot_ct = mrpb_dot_product(q_cts, d_cts)

    # Decrypt
    num, den = mrpb.decrypt(dot_ct, sk)
    encrypted_dot = float(num) / float(den)

    return {
        "true_dot": true_dot,
        "encoded_dot": encoded_dot,
        "encrypted_dot": encrypted_dot,
        "quantization_error": abs(true_dot - encoded_dot),
        "encryption_error": abs(encoded_dot - encrypted_dot),
    }
