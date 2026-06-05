"""
Two-server additive secret sharing over Z_{2^64} with matrix-vector
Beaver triple multiplication.

Operates on native 64-bit unsigned integers (numpy uint64). Security
is information-theoretic: each share is a uniform 64-bit value that
perfectly masks the plaintext regardless of computational power.

The 64-bit ring Z_{2^64} suffices because:
  - Encoded embedding components are bounded by 2^k (k <= 20)
  - Component products are bounded by 2^{2k}
  - Dot products at dimension d are bounded by d * 2^{2k}
  - At d=384, k=16: max value = 384 * 2^32 ~ 2^41 << 2^64
  - Two's complement handles signed values natively via overflow

Performance:
  - Sharing: numpy random uint64 generation (~ns per element)
  - Addition of shares: native uint64 add (~0.3 ns)
  - Scalar-by-public multiply: native uint64 mul (~0.3 ns)
  - Dot product: numpy vectorized uint64 dot (~sub-microsecond)
"""

import numpy as np
import warnings
from dataclasses import dataclass
from numpy.typing import NDArray

# Suppress numpy overflow warnings: uint64 wraparound is our
# modular arithmetic, not a bug.
warnings.filterwarnings('ignore', 'overflow encountered', RuntimeWarning,
                        module=__name__)
np.seterr(over='ignore')

# Type alias for clarity
U64 = np.uint64
Vec64 = NDArray[np.uint64]
Mat64 = NDArray[np.uint64]

# The modulus is implicit: all uint64 arithmetic wraps mod 2^64.
# We never need to write "% n" because hardware does it for us.


# =====================================================================
# Additive secret sharing over Z_{2^64}
# =====================================================================

def to_uint64(value: int) -> U64:
    """Convert a signed integer to uint64 via two's complement."""
    return np.uint64(value % (2**64))


def to_signed(value: U64) -> int:
    """Convert uint64 back to signed integer (two's complement)."""
    v = int(value)
    if v >= 2**63:
        v -= 2**64
    return v


def share_scalar(m: int, rng: np.random.Generator = None) -> tuple[U64, U64]:
    """Split integer m into two additive shares in Z_{2^64}."""
    if rng is None:
        rng = np.random.default_rng()
    s1 = U64(rng.integers(0, 2**64, dtype=np.uint64))
    s2 = to_uint64(m) - s1  # wraps mod 2^64
    return s1, s2


def reconstruct_scalar(s1: U64, s2: U64) -> int:
    """Reconstruct signed integer from two uint64 shares."""
    return to_signed(s1 + s2)


def share_vector(values: list[int] | NDArray, rng: np.random.Generator = None) -> tuple[Vec64, Vec64]:
    """Split a vector of signed integers into additive shares."""
    if rng is None:
        rng = np.random.default_rng()
    vals = np.array(values, dtype=np.int64).view(np.uint64)
    s1 = rng.integers(0, 2**64, size=len(values), dtype=np.uint64)
    s2 = vals - s1  # wraps mod 2^64
    return s1, s2


def reconstruct_vector(s1: Vec64, s2: Vec64) -> NDArray[np.int64]:
    """Reconstruct signed integer vector from shares."""
    return (s1 + s2).view(np.int64)


def share_matrix(M: NDArray[np.int64], rng: np.random.Generator = None) -> tuple[Mat64, Mat64]:
    """Split a matrix of signed integers into additive shares."""
    if rng is None:
        rng = np.random.default_rng()
    vals = M.astype(np.int64).view(np.uint64)
    s1 = rng.integers(0, 2**64, size=M.shape, dtype=np.uint64)
    s2 = vals - s1
    return s1, s2


# =====================================================================
# Query-private dot product (no Beaver triples, no communication)
# =====================================================================

def query_private_dot(
    query_shares1: Vec64,
    query_shares2: Vec64,
    doc_plaintext: NDArray[np.int64],
) -> tuple[U64, U64]:
    """Compute shares of dot(query, document) with query privacy.

    Documents are plaintext. Each server multiplies its query share
    by the plaintext document components and sums. Zero communication.
    """
    doc_u64 = doc_plaintext.astype(np.int64).view(np.uint64)
    dot1 = np.sum(query_shares1 * doc_u64, dtype=np.uint64)
    dot2 = np.sum(query_shares2 * doc_u64, dtype=np.uint64)
    return U64(dot1), U64(dot2)


def query_private_batch(
    query_shares1: Vec64,
    query_shares2: Vec64,
    doc_matrix: NDArray[np.int64],
) -> tuple[Vec64, Vec64]:
    """Compute shares of dot(query, d_j) for all documents at once.

    doc_matrix is N x d. Returns N-element share vectors.
    Uses numpy matrix-vector multiply for maximum throughput.
    """
    docs_u64 = doc_matrix.astype(np.int64).view(np.uint64)
    # Matrix-vector multiply: (N x d) @ (d,) -> (N,)
    dots1 = docs_u64 @ query_shares1
    dots2 = docs_u64 @ query_shares2
    return dots1.astype(np.uint64), dots2.astype(np.uint64)


# =====================================================================
# Matrix-Vector Beaver Triples for mutual-privacy dot product
# =====================================================================

@dataclass
class MatrixBeaverSetup:
    """Pre-computed Beaver material for a static document matrix.

    Generated once when the document corpus is uploaded.

    A: random N x d mask matrix (secret, shared as A1 + A2)
    E: D - A (public, reconstructed during setup)
    """
    E: Mat64              # Public: D - A (N x d)
    A1: Mat64             # Server 1's share of A
    A2: Mat64             # Server 2's share of A
    n_docs: int
    dim: int


@dataclass
class QueryBeaverTriple:
    """Per-query Beaver material.

    b: random d-vector (shared as b1 + b2)
    c: A @ b (N-vector, shared as c1 + c2)
    """
    b1: Vec64             # Server 1's share of b
    b2: Vec64             # Server 2's share of b
    c1: Vec64             # Server 1's share of c = A @ b
    c2: Vec64             # Server 2's share of c = A @ b


def setup_matrix_beaver(
    doc_shares1: Mat64,
    doc_shares2: Mat64,
    rng: np.random.Generator = None,
) -> MatrixBeaverSetup:
    """One-time offline setup for matrix-vector Beaver triples.

    Given shared document matrix D = D1 + D2, generates random mask A,
    shares it, and computes public E = D - A.

    Communication: servers exchange (D1 - A1) and (D2 - A2) to reconstruct E.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_docs, dim = doc_shares1.shape
    A1 = rng.integers(0, 2**64, size=(n_docs, dim), dtype=np.uint64)
    A2 = rng.integers(0, 2**64, size=(n_docs, dim), dtype=np.uint64)

    # E = D - A = (D1 + D2) - (A1 + A2) = (D1 - A1) + (D2 - A2)
    E1 = doc_shares1 - A1
    E2 = doc_shares2 - A2
    E = E1 + E2  # Servers exchange E1, E2 to reconstruct E (public)

    return MatrixBeaverSetup(E=E, A1=A1, A2=A2, n_docs=n_docs, dim=dim)


def generate_query_triple(
    setup: MatrixBeaverSetup,
    rng: np.random.Generator = None,
) -> QueryBeaverTriple:
    """Per-query offline: generate Beaver triple for one query.

    Generates random b, computes c = A @ b, shares both.
    This can be precomputed before the query arrives.
    """
    if rng is None:
        rng = np.random.default_rng()

    dim = setup.dim
    b = rng.integers(0, 2**64, size=dim, dtype=np.uint64)
    A = setup.A1 + setup.A2  # Dealer knows full A
    c = A @ b  # N-vector, c = A @ b mod 2^64

    # Share b and c
    b1 = rng.integers(0, 2**64, size=dim, dtype=np.uint64)
    b2 = b - b1

    c1 = rng.integers(0, 2**64, size=setup.n_docs, dtype=np.uint64)
    c2 = c.astype(np.uint64) - c1

    return QueryBeaverTriple(b1=b1, b2=b2, c1=c1, c2=c2)


def mutual_privacy_batch(
    query_shares1: Vec64,
    query_shares2: Vec64,
    setup: MatrixBeaverSetup,
    triple: QueryBeaverTriple,
) -> tuple[Vec64, Vec64]:
    """Compute shares of D @ q with mutual privacy.

    Online communication: servers exchange f = q - b (d values = dim * 8 bytes).
    At d=384: 3 KB exchanged per query, regardless of corpus size.

    Returns (Z1, Z2) where Z1 + Z2 = D @ q.
    """
    # Step 1: Each server computes its part of f = q - b
    f1 = query_shares1 - triple.b1
    f2 = query_shares2 - triple.b2

    # Step 2: Servers exchange f1, f2 to open f (d values)
    f = f1 + f2  # Public: f = q - b

    # Step 3: Each server computes its output share
    # Z1 + Z2 = E@f + E@b + A@f + c = (E+A)@(f+b) = D@q
    E = setup.E
    Z1 = (E @ f
          + E @ triple.b1
          + setup.A1 @ f
          + triple.c1).astype(np.uint64)
    Z2 = (E @ triple.b2
          + setup.A2 @ f
          + triple.c2).astype(np.uint64)

    return Z1, Z2
