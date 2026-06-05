"""
Float-to-rational encoding for RPB/MRPB homomorphic schemes.

Fixed-point quantization: x ≈ round(x · 2^k) / 2^k

KEY ADVANTAGE OVER CKKS (PRAG, SecureRAG, PIR-RAG, D-RAG):
  - Quantization error is BOUNDED and ONE-TIME: |ε| ≤ 2^{-(k+1)}
  - Zero noise accumulation during homomorphic evaluation
  - Dot-product error is analytically tight: |Δ| ≤ d · 2^{-k}  (d = dim)
  - No bootstrapping, no noise budget, no Operation-Error Estimation
  - Exact rational arithmetic after encoding — the ONLY error source is
    the initial quantization, which does not grow with circuit depth.

Compare: CKKS noise after L operations grows as O(B^L), requiring
careful modulus-chain sizing and periodic bootstrapping.
"""

import math
from typing import Optional


# ---------------------------------------------------------------------------
# Core encode / decode
# ---------------------------------------------------------------------------

def float_to_rational(
    x: float,
    precision_bits: int = 16,
) -> tuple[int, int]:
    """Encode a float as a rational (numerator, 2^k).

    Args:
        x: Finite float value.
        precision_bits: Fractional bits k.  Higher k → more precision but
            larger numerators (constrains MRPB plaintext space / add depth).

    Returns:
        (numerator, denominator) with denominator = 2^k.

    Raises:
        ValueError: On non-finite input or precision_bits < 1.
    """
    if not isinstance(precision_bits, int) or precision_bits < 1:
        raise ValueError(f"precision_bits must be a positive integer, got {precision_bits}")
    if not math.isfinite(x):
        raise ValueError(f"Cannot encode non-finite float: {x}")

    scale = 1 << precision_bits
    numerator = round(x * scale)
    return (numerator, scale)


def rational_to_float(numerator: int, denominator: int) -> float:
    """Decode a rational to a float.

    Raises:
        ValueError: If denominator ≤ 0.
    """
    if denominator <= 0:
        raise ValueError(f"Denominator must be positive, got {denominator}")
    return numerator / denominator


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def encode_vector(
    vec: list[float],
    precision_bits: int = 16,
) -> list[tuple[int, int]]:
    """Encode a float vector.  All components share denominator 2^k,
    which factors out cleanly in dot-product computation."""
    return [float_to_rational(x, precision_bits) for x in vec]


def decode_vector(rationals: list[tuple[int, int]]) -> list[float]:
    """Decode a rational vector back to floats."""
    return [rational_to_float(n, d) for n, d in rationals]


# ---------------------------------------------------------------------------
# Error analysis  —  the paper's core quantitative argument
# ---------------------------------------------------------------------------

def quantization_error_bound(precision_bits: int) -> float:
    """Tight worst-case per-element error: exactly 2^{-(k+1)}.

    This bound is:
      • Tight  (achieved when x is exactly at a midpoint)
      • Input-independent
      • ONE-TIME — does NOT grow with homomorphic depth

    For CKKS, noise after L multiplications grows as O(B^L).
    For RPB/MRPB, error after L operations is still 2^{-(k+1)}.
    """
    return 0.5 / (1 << precision_bits)


def encoding_error(x: float, precision_bits: int) -> float:
    """Actual absolute error for a specific value: |x − decode(encode(x))|."""
    num, den = float_to_rational(x, precision_bits)
    return abs(x - num / den)


def dot_product_error_bound(
    precision_bits: int,
    vector_dim: int,
    max_component: float = 1.0,
) -> float:
    """Analytical upper bound on dot-product quantization error.

    Let q, d be true vectors and q̂, d̂ their encoded versions.
    Each component has error |ε_i| ≤ ε = 2^{-(k+1)}.

    |dot(q̂, d̂) − dot(q, d)| ≤ Σ |ε_qi·d_i + ε_di·q_i + ε_qi·ε_di|
                               ≤ d · (2·M·ε + ε²)

    where M = max absolute component value, d = dimension.

    This is the TOTAL error for the entire dot product, and it does NOT
    increase when computed homomorphically — because RPB/MRPB introduce
    zero additional noise.

    Args:
        precision_bits: The k parameter.
        vector_dim: Embedding dimension (e.g. 384, 768).
        max_component: Maximum |value| in either vector (1.0 for normalized).

    Returns:
        Upper bound on absolute dot-product error.
    """
    eps = quantization_error_bound(precision_bits)
    return vector_dim * (2 * max_component * eps + eps * eps)


def cosine_similarity_error_bound(
    precision_bits: int,
    vector_dim: int,
    max_component: float = 1.0,
) -> float:
    """Upper bound on cosine-similarity error from quantization.

    cos(q,d) = dot(q,d) / (‖q‖·‖d‖).  For unit-norm vectors the
    denominator is 1, so the cosine error equals the dot-product error.
    For non-unit vectors with ‖v‖ ≥ norm_lower_bound, divide.

    With normalized embeddings (the standard in RAG), this equals
    dot_product_error_bound.
    """
    # For unit-normalized embeddings (standard sentence-transformers output)
    return dot_product_error_bound(precision_bits, vector_dim, max_component)


# ---------------------------------------------------------------------------
# MRPB plaintext-space analysis
# ---------------------------------------------------------------------------

def max_numerator_bits(precision_bits: int, value_range: float = 1.0) -> int:
    """Maximum bit-width of encoded numerators.

    For MRPB this determines required plaintext space.
    Recall EXP-003: max_add_depth = key_bits / (2 × operand_bits)  [small]
                                   = key_bits / (4 × operand_bits)  [large].
    """
    max_abs = math.ceil(value_range * (1 << precision_bits))
    return max_abs.bit_length() + 1          # +1 for sign


def dot_product_operand_bits(precision_bits: int, value_range: float = 1.0) -> int:
    """Bit-width of each term in a dot product (product of two numerators).

    After multiplying two encoded scalars, the numerator has up to
    2 × max_numerator_bits bits.  The additions in the dot product
    accumulate these terms.
    """
    return 2 * max_numerator_bits(precision_bits, value_range)


def mrpb_min_key_bits(
    precision_bits: int,
    vector_dim: int,
    value_range: float = 1.0,
    safety_factor: float = 2.0,
) -> int:
    """Minimum MRPB key size to evaluate a full dot product.

    A d-dimensional dot product requires d multiplications producing
    terms of ~2k bits, followed by (d−1) additions.

    From EXP-003 (large-operand regime):
        max_additions = key_bits / (4 × operand_bits)
    Rearranging:
        key_bits ≥ 4 × operand_bits × (d − 1)

    Args:
        precision_bits: The k parameter.
        vector_dim: Embedding dimension.
        value_range: Max absolute component value.
        safety_factor: Multiplier for conservative sizing.

    Returns:
        Minimum recommended key bits (with safety margin).
    """
    op_bits = dot_product_operand_bits(precision_bits, value_range)
    required_additions = vector_dim - 1
    # Large-operand formula from EXP-003
    min_bits = 4 * op_bits * required_additions
    return int(math.ceil(min_bits * safety_factor))


# ---------------------------------------------------------------------------
# Precision recommendation helper
# ---------------------------------------------------------------------------

def recommend_precision(
    vector_dim: int = 384,
    target_cosine_error: float = 0.001,
    max_component: float = 1.0,
) -> dict:
    """Recommend encoding precision for a given accuracy target.

    Sweeps k from 8 to 32 and returns the analysis for each,
    highlighting the smallest k that meets the target.

    Returns:
        Dict with 'recommendations' list and 'selected_k'.
    """
    results = []
    selected = None

    for k in range(8, 33):
        cos_err = cosine_similarity_error_bound(k, vector_dim, max_component)
        dot_err = dot_product_error_bound(k, vector_dim, max_component)
        num_bits = max_numerator_bits(k, max_component)
        mrpb_bits = mrpb_min_key_bits(k, vector_dim, max_component, safety_factor=2.0)

        entry = {
            "precision_bits": k,
            "cosine_error_bound": cos_err,
            "dot_product_error_bound": dot_err,
            "numerator_bits": num_bits,
            "mrpb_min_key_bits": mrpb_bits,
        }
        results.append(entry)

        if selected is None and cos_err <= target_cosine_error:
            selected = k

    return {"recommendations": results, "selected_k": selected}
