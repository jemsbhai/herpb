# Findings

## Curated Summary

### Finding 1: Both RPB and MRPB achieve 100% correctness (EXP-001)
Across 280,000 trials (7 bit-lengths × 4 scheme variants × 10,000 each),
both schemes produce exact rational reconstruction with zero errors.
Homomorphic operations (add, multiply, subtract) preserve correctness
at depth 3+ (tested in EXP-000 unit tests).

### Finding 2: No noise accumulation — unlimited homomorphic depth (EXP-000)
Unlike BGV/BFV/CKKS, both RPB and MRPB have zero noise growth under
homomorphic operations. No bootstrapping is needed. This is a fundamental
structural advantage of ratio-preserving blinding over lattice-based HE.

### Finding 3: MRPB decryption is the primary performance bottleneck (EXP-001)
Rational reconstruction via half-GCD over large RSA moduli is 30-40×
slower than RPB's simple GCD reduction. At 8000 bits: RPB decrypts in
~1ms, MRPB in ~78ms. Encryption overhead is more modest (2-4×).

### Finding 4: MRPB keygen scales superlinearly with input size (EXP-001)
The RSA modulus must be ~2× the ratio's bit-length for rational
reconstruction to succeed. This means keygen time grows superlinearly:
2048-bit keys take 0.02s, but 93440-bit keys take 4.8 hours. This is
a one-time amortized cost, but it limits MRPB's practical range without
HPC resources.

### Finding 5: Distinct-k consistently outperforms shared-k (EXP-001)
Across both schemes and all bit-lengths, distinct-k is 2-3× faster for
encryption and produces ~2× smaller ciphertexts (in RPB). This confirms
the collaborator's C++ findings and is due to linear vs quadratic scaling
in the precision parameter.

---

## Raw Findings Log

**2026-05-29 — EXP-001 completed (7/8 bit-lengths)**
10000-bit case deferred to A100 GPU due to keygen time (~134K-bit RSA).
All other results are final. See LOGBOOK.md EXP-001 for full tables.
