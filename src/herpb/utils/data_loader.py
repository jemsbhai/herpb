"""Data loading utilities for experiment input files.

Reads the collaborator's format: each line contains 4 space-separated
big integers: a_num a_den b_num b_den.
"""

import warnings
from pathlib import Path
from gmpy2 import mpz

from .math_utils import digits10


def load_rational_pairs(filepath: str | Path) -> list[tuple[mpz, mpz, mpz, mpz]]:
    """Load rational number pairs from a text file.

    Each line: a_num a_den b_num b_den
    Lines with != 4 fields are skipped with a warning.

    Returns list of (a_num, a_den, b_num, b_den) tuples.
    """
    pairs = []
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                warnings.warn(
                    f"{filepath.name}:{lineno}: expected 4 fields, got {len(parts)} — skipping"
                )
                continue
            a_num, a_den, b_num, b_den = (mpz(x) for x in parts)
            pairs.append((a_num, a_den, b_num, b_den))
    return pairs


def describe_dataset(pairs: list[tuple[mpz, mpz, mpz, mpz]]) -> dict:
    """Compute summary statistics for a dataset of rational pairs."""
    if not pairs:
        return {"count": 0}

    a_digits = [digits10(p[0]) for p in pairs]
    b_digits = [digits10(p[2]) for p in pairs]

    return {
        "count": len(pairs),
        "a_num_digits": {"min": min(a_digits), "max": max(a_digits)},
        "b_num_digits": {"min": min(b_digits), "max": max(b_digits)},
    }
