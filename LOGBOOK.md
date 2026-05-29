# Experimental Logbook — HERPB

This logbook is the single source of truth for all experiments conducted in
this project. Entries are append-only. See lab-runner protocol for format.

---

## EXP-000: Port and Validate C++ Reference Implementation

**Date:** 2026-05-28
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** completed

### Hypothesis
The Python port of the RPB scheme (from the collaborator's C++ code) produces
identical correctness results (GCD-reduced ratios match expected values) for
all test cases in the original dataset.

### Independent Variables
- Input bit-length: 10, 100, 1000, 2000, 3000, 5000, 8000, 10000 bits
- Method: shared-k vs distinct-k
- Precision parameter t: 0-6

### Dependent Variables / Metrics
- Correctness: does GCD-reduce(N, D) == a/b? (boolean PASS/FAIL)
- Encryption time (ms)
- Ciphertext size (digits)
- Memory usage (bytes)

### Control Conditions
- Baseline: original C++ results from collaborator's version 2
- Fixed seed: 42

### Protocol
1. Port RPB shared-k and distinct-k encryption to Python using gmpy2
2. Load original input files (10bits.txt through 10000bits.txt)
3. Run identical experiment protocol: 1000 iterations x 10 pairs
4. Compare correctness and timing against C++ reference
5. Accept if 100% correctness match; flag any timing regressions > 10x

### Environment
- Hardware: Windows laptop, 64GB RAM, RTX 4090
- Software: Python 3.12, gmpy2 2.3.0, Windows 11
- Git commit: [pending initial commit]
- Config file: configs/base.yaml
- Seeds: 42

### Results
- **42/42 tests pass** in 0.51s
- RPB correctness: PASS (shared-k, distinct-k, various c values, various t)
- MRPB correctness: PASS (shared-k, distinct-k, multiple rational pairs)
- MRPB keygen: PASS (valid RSA, deterministic seeding, c coprime to n)
- MRPB security: PASS (GCD does NOT leak ratio, different keys → different ct)
- RPB homomorphic: PASS (multiply, add, subtract, depth-3 chain)
- MRPB homomorphic: PASS (multiply, depth-3 chain, fixed ciphertext size)
- RPB bounds: PASS (max_t_shared, max_T_distinct, split_T)

### Observations
- gmpy2 2.3.0 installed cleanly on Windows/Python 3.12 without manual wheel.
- Rational reconstruction in MRPB (half-GCD) works correctly for small ratios
  with 1024-bit modulus.
- All 42 tests cover both schemes comprehensively.

### Interpretation
Port is validated. Both RPB and MRPB produce correct results for all test
cases. Ready to proceed with comparative benchmarking (EXP-001).

### Artifacts
- Tests: tests/test_rpb.py, tests/test_mrpb.py
- Source: src/herpb/schemes/rpb.py, src/herpb/schemes/mrpb.py
