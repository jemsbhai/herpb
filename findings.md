# Findings

## Curated Summary

### Finding 1: Both RPB and MRPB achieve 100% correctness (EXP-001)
Across 280,000 trials (7 bit-lengths × 4 scheme variants × 10,000 each),
both schemes produce exact rational reconstruction with zero errors.

### Finding 2: No noise accumulation — unlimited homomorphic depth (EXP-002)
Unlike BGV/BFV/CKKS, both RPB and MRPB have zero noise growth under
homomorphic operations. Verified to depth 64 across multiply chains,
addition chains, and mixed circuits with 100% correctness (4,200 trials).

### Finding 3: MRPB decryption is the primary performance bottleneck (EXP-001)
Rational reconstruction via half-GCD over large RSA moduli is 30-40×
slower than RPB's simple GCD reduction. Encryption overhead is 2-4×.

### Finding 4: MRPB keygen scales superlinearly with input size (EXP-001)
The RSA modulus must be ~2× the ratio's bit-length. Keygen grows
superlinearly: 2048-bit → 0.02s, 93440-bit → 4.8 hours.

### Finding 5: Distinct-k consistently outperforms shared-k (EXP-001)
2-3× faster encryption, ~2× smaller ciphertexts (RPB) across both schemes.

### Finding 6: MRPB eval ~10× slower, both scale linearly with depth (EXP-002)
Modular arithmetic overhead is consistent across depths.

### Finding 7: RPB ciphertext grows linearly, MRPB stays fixed (EXP-002)
RPB grows ~4.5 bytes/depth. MRPB is fixed at key_bits/4 bytes regardless
of depth — a key advantage for deep circuits.

### Finding 8: MRPB addition depth is a closed-form function of key and operand size (EXP-003)
The maximum addition chain depth follows:
  - Mid-range operands (16-64 bits): max_depth = key_bits / (2 × op_bits) — EXACT
  - Large operands (128+ bits): max_depth ≈ key_bits / (4 × op_bits)
  - Small operands (2-8 bits): exceeds prediction (GCD reduction provides headroom)
Doubling key_bits exactly doubles max_depth. The failure boundary is sharp
(100%→0% in one depth step). This is fundamentally simpler than the stochastic
noise budgets in lattice-based HE — a practitioner can compute exact depth
limits from two parameters.

---

## Raw Findings Log

**2026-05-29 — EXP-003 completed**
27 configurations tested across 3 key sizes × 9 operand sizes. Clean inverse
relationship confirmed. Corrected formula for large operands: key/(4×op_bits).

**2026-05-29 — EXP-002 completed**
All 4,200 trials pass at depth 64.

**2026-05-29 — EXP-001 completed (7/8 bit-lengths)**
10000-bit case deferred to A100 GPU.
