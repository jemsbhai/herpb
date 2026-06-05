"""
Two-server additive secret sharing over Z_n with Beaver triple multiplication.

Provides information-theoretic security: no computational assumptions.
Each server's view is statistically independent of the secret input,
regardless of the server's computational power.

Query-private model:
  Client splits query into additive shares, one per server.
  Each server computes a share of the dot product using plaintext documents.
  Client reconstructs the dot product from both shares.
  Neither server alone learns any information about the query.

Mutual-privacy model (Beaver triples):
  Both query AND documents are secret-shared across two servers.
  Multiplication of shared values uses pre-generated Beaver triples.
  One round of inter-server communication per batch of multiplications.
  Neither server learns query or document contents.
"""

import gmpy2
from gmpy2 import mpz
from dataclasses import dataclass


# =====================================================================
# Additive secret sharing over Z_n
# =====================================================================

def share(m: int, n: mpz, rs: gmpy2.random_state = None) -> tuple[mpz, mpz]:
    """Split integer m into two additive shares mod n.

    Returns (s1, s2) where s1 + s2 = m mod n and s1 is uniform in Z_n.
    """
    if rs is None:
        rs = gmpy2.random_state()
    bits = int(gmpy2.num_digits(n, 2))
    s1 = gmpy2.mpz_urandomb(rs, bits) % n
    s2 = (mpz(m) - s1) % n
    return s1, s2


def reconstruct(s1: mpz, s2: mpz, n: mpz) -> mpz:
    """Reconstruct value from two additive shares.

    Returns m = s1 + s2 mod n, adjusted to signed range for small values.
    """
    m = (s1 + s2) % n
    # If m > n/2, interpret as negative (for signed plaintexts)
    if m > n // 2:
        m = m - n
    return m


def share_vector(values: list[int], n: mpz, seed: int = None) -> tuple[list[mpz], list[mpz]]:
    """Split a vector of integers into per-component additive shares."""
    rs = gmpy2.random_state(seed) if seed is not None else gmpy2.random_state()
    shares1, shares2 = [], []
    for v in values:
        s1, s2 = share(v, n, rs)
        shares1.append(s1)
        shares2.append(s2)
    return shares1, shares2


# =====================================================================
# Beaver triples for secure multiplication
# =====================================================================

@dataclass
class BeaverTriple:
    """One Beaver triple split across two servers.

    Satisfies: (a1+a2) * (b1+b2) = c1+c2  (mod n)
    """
    a1: mpz
    b1: mpz
    c1: mpz
    a2: mpz
    b2: mpz
    c2: mpz


def generate_beaver_triple(n: mpz, rs: gmpy2.random_state = None) -> BeaverTriple:
    """Generate one Beaver triple, pre-shared for two servers.

    A trusted dealer samples random a, b, computes c = a*b mod n,
    then splits each into additive shares.
    """
    if rs is None:
        rs = gmpy2.random_state()
    bits = int(gmpy2.num_digits(n, 2))

    a = gmpy2.mpz_urandomb(rs, bits) % n
    b = gmpy2.mpz_urandomb(rs, bits) % n
    c = (a * b) % n

    a1 = gmpy2.mpz_urandomb(rs, bits) % n
    a2 = (a - a1) % n
    b1 = gmpy2.mpz_urandomb(rs, bits) % n
    b2 = (b - b1) % n
    c1 = gmpy2.mpz_urandomb(rs, bits) % n
    c2 = (c - c1) % n

    return BeaverTriple(a1=a1, b1=b1, c1=c1, a2=a2, b2=b2, c2=c2)


def generate_beaver_triples(count: int, n: mpz, seed: int = None) -> list[BeaverTriple]:
    """Generate multiple Beaver triples (offline precomputation)."""
    rs = gmpy2.random_state(seed) if seed is not None else gmpy2.random_state()
    return [generate_beaver_triple(n, rs) for _ in range(count)]


def beaver_multiply(
    x1: mpz, y1: mpz,
    x2: mpz, y2: mpz,
    triple: BeaverTriple,
    n: mpz,
) -> tuple[mpz, mpz]:
    """Multiply two shared values using a Beaver triple.

    Input: [x] = (x1, x2), [y] = (y1, y2) where x = x1+x2, y = y1+y2
    Output: [z] = (z1, z2) where z = z1+z2 = x*y mod n

    Communication: servers exchange (e1, f1) and (e2, f2) to open
    e = x - a and f = y - b.  One round, 2 values per server.
    """
    # Each server masks its input share with its triple share
    e1 = (x1 - triple.a1) % n
    e2 = (x2 - triple.a2) % n
    f1 = (y1 - triple.b1) % n
    f2 = (y2 - triple.b2) % n

    # Inter-server communication: open e and f
    e = (e1 + e2) % n
    f = (f1 + f2) % n

    # Each server computes its output share
    # z = xy = (a+e)(b+f) = ab + af + be + ef = c + f*a + e*b + e*f
    z1 = (triple.c1 + f * triple.a1 + e * triple.b1 + e * f) % n
    z2 = (triple.c2 + f * triple.a2 + e * triple.b2) % n
    # Note: only server 1 adds e*f to avoid double-counting

    return z1, z2


# =====================================================================
# Shared-value addition (free in additive sharing)
# =====================================================================

def shared_add(x1: mpz, y1: mpz, x2: mpz, y2: mpz, n: mpz) -> tuple[mpz, mpz]:
    """Add two shared values. No communication needed.

    [x+y] = (x1+y1, x2+y2)
    """
    return (x1 + y1) % n, (x2 + y2) % n


def shared_scalar_mul(x1: mpz, k: int, x2: mpz, n: mpz) -> tuple[mpz, mpz]:
    """Multiply a shared value by a public scalar. No communication needed.

    [k*x] = (k*x1, k*x2)
    """
    return (mpz(k) * x1) % n, (mpz(k) * x2) % n


# =====================================================================
# Query-private dot product (no Beaver triples needed)
# =====================================================================

def query_private_dot(
    query_shares1: list[mpz],
    query_shares2: list[mpz],
    doc_plaintext: list[int],
    n: mpz,
) -> tuple[mpz, mpz]:
    """Compute shares of dot(query, document) in the query-private model.

    Documents are plaintext on both servers. Query is additively shared.
    Each server multiplies its query share by the plaintext document
    component (public-scalar multiply, free) and sums.

    No inter-server communication. No Beaver triples.

    Returns (dot_share1, dot_share2) where dot_share1 + dot_share2 = dot(q, d) mod n
    """
    dim = len(query_shares1)
    dot1 = mpz(0)
    dot2 = mpz(0)
    for i in range(dim):
        dot1 = (dot1 + query_shares1[i] * mpz(doc_plaintext[i])) % n
        dot2 = (dot2 + query_shares2[i] * mpz(doc_plaintext[i])) % n
    return dot1, dot2


# =====================================================================
# Mutual-privacy dot product (uses Beaver triples)
# =====================================================================

def mutual_privacy_dot(
    query_shares1: list[mpz],
    query_shares2: list[mpz],
    doc_shares1: list[mpz],
    doc_shares2: list[mpz],
    triples: list[BeaverTriple],
    n: mpz,
) -> tuple[mpz, mpz]:
    """Compute shares of dot(query, document) with mutual privacy.

    Both query and document are additively shared. Each dimension
    requires one Beaver triple for the multiplication.

    Communication: one round of 2*dim values between servers.

    Returns (dot_share1, dot_share2) where dot_share1 + dot_share2 = dot(q, d) mod n
    """
    dim = len(query_shares1)
    if len(triples) < dim:
        raise ValueError(f"Need {dim} Beaver triples, got {len(triples)}")

    # Multiply each dimension using Beaver triples
    dot1 = mpz(0)
    dot2 = mpz(0)
    for i in range(dim):
        prod1, prod2 = beaver_multiply(
            query_shares1[i], doc_shares1[i],
            query_shares2[i], doc_shares2[i],
            triples[i], n,
        )
        dot1 = (dot1 + prod1) % n
        dot2 = (dot2 + prod2) % n

    return dot1, dot2
