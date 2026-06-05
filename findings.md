# Findings

## Curated Summary

### Finding 1: Both RPB and MRPB achieve 100% correctness (EXP-001)
Across 280,000 trials (7 bit-lengths × 4 scheme variants × 10,000 each),
both schemes produce exact rational reconstruction with zero errors.

### Finding 2: No noise accumulation — unlimited homomorphic depth (EXP-000, EXP-002)
Unlike BGV/BFV/CKKS, both RPB and MRPB have zero noise growth under
homomorphic operations. No bootstrapping is needed. Verified empirically
to depth 64 across multiply chains, addition chains, and mixed circuits
with 100% correctness (4,200 trials in EXP-002).

### Finding 3: MRPB decryption is the primary performance bottleneck (EXP-001)
Rational reconstruction via half-GCD over large RSA moduli is 30-40×
slower than RPB's simple GCD reduction. At 8000 bits: RPB decrypts in
~1ms, MRPB in ~78ms. Encryption overhead is more modest (2-4×).

### Finding 4: MRPB keygen scales superlinearly with input size (EXP-001)
The RSA modulus must be ~2× the ratio's bit-length for rational
reconstruction to succeed. Keygen grows superlinearly: 2048-bit keys
take 0.02s, 93440-bit keys take 4.8 hours.

### Finding 5: Distinct-k consistently outperforms shared-k (EXP-001)
Across both schemes and all bit-lengths, distinct-k is 2-3× faster for
encryption and produces ~2× smaller ciphertexts (in RPB).

### Finding 6: MRPB eval time scales linearly with depth, RPB is ~10× faster (EXP-002)
At depth 64: RPB eval takes ~0.014ms (multiply) / ~0.017ms (add);
MRPB takes ~0.156ms (multiply) / ~0.186ms (add). The modular arithmetic
overhead is consistent across depths.

### Finding 7: RPB ciphertext grows linearly, MRPB stays fixed (EXP-002)
RPB ciphertexts grow ~4.5 bytes per depth level (285 bytes at depth 64).
MRPB ciphertexts are fixed at 512 bytes regardless of depth — a key
advantage for deep circuits or bandwidth-constrained settings.

### Finding 8: MRPB addition survives depth 64 with small operands (EXP-002)
The sqrt(n) plaintext space bound was not reached at any tested depth
with single-digit rational operands. The bound is a function of operand
magnitude, not circuit depth alone. This is a deterministic, predictable
constraint — unlike the stochastic noise budgets in lattice-based HE.

---

## Raw Findings Log

**2026-05-29 — EXP-002 completed**
All 4,200 trials pass. No MRPB reconstruction failures. Both schemes
handle depth-64 circuits of all three types without degradation.

**2026-05-29 — EXP-001 completed (7/8 bit-lengths)**
10000-bit case deferred to A100 GPU due to keygen time (~134K-bit RSA).
All other results are final.
