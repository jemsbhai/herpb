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

### Results
- **42/42 tests pass** in 0.51s — full details in earlier commit

### Artifacts
- Tests: tests/test_rpb.py, tests/test_mrpb.py
- Source: src/herpb/schemes/rpb.py, src/herpb/schemes/mrpb.py

---

## EXP-001: RPB vs MRPB Comparative Benchmark

**Date:** 2026-05-29
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** completed (7 of 8 bit-lengths; 10000 deferred to A100)

### Results

**Correctness: 100% across all 280,000 trials (7 bit-lengths × 4 variants × 10000 each).**

#### Encryption Time (ms, mean)

| Input   | RPB-shared | RPB-distinct | MRPB-shared | MRPB-distinct |
|---------|-----------|-------------|------------|--------------|
| 10      | 0.0023    | 0.0022      | 0.0015     | 0.0014       |
| 100     | 0.0042    | 0.0025      | 0.0059     | 0.0029       |
| 1000    | 0.0401    | 0.0119      | 0.0837     | 0.0368       |
| 2000    | 0.1073    | 0.0293      | 0.2517     | 0.1103       |
| 3000    | 0.1495    | 0.0396      | 0.4638     | 0.2101       |
| 5000    | 0.3867    | 0.1065      | 1.0655     | 0.4817       |
| 8000    | 0.7269    | 0.1959      | 1.7102     | 0.7740       |

#### MRPB Key Scaling

| Input   | key_bits | keygen time   |
|---------|----------|---------------|
| 10      | 2048     | 0.02s         |
| 100     | 2048     | 0.03s         |
| 1000    | 13568    | 13.4s         |
| 2000    | 26880    | 20.1s         |
| 3000    | 40192    | 6 min 45s     |
| 5000    | 66816    | 1 hr 21 min   |
| 8000    | 93440    | 4 hr 48 min   |

### Key Findings
1. MRPB encryption ~2-4× slower than RPB (modular reduction overhead)
2. MRPB decryption ~30-40× slower (rational reconstruction via half-GCD)
3. Distinct-k consistently ~2-3× faster than shared-k in both schemes
4. MRPB keygen scales superlinearly — practical barrier at large input sizes

### Artifacts
- Script: scripts/run_experiment.py
- Raw data: experiments/exp_001_benchmark/results_raw.csv
- Summary: experiments/exp_001_benchmark/summary.json

---

## EXP-002: Homomorphic Depth Benchmark

**Date:** 2026-05-29
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** completed

### Hypothesis
RPB multiplication and addition chains work at arbitrary depth (unbounded
integer growth). MRPB multiplication chains work at arbitrary depth (ratio
stays bounded). MRPB addition chains fail at a depth determined by the
plaintext space bound sqrt(n) — for a 2048-bit modulus, this is ~1024 bits
of headroom, allowing several hundred additions of small rationals before
reconstruction fails.

### Independent Variables
- Circuit type: multiply_chain, add_chain, mixed_alt
- Depth: 1, 2, 4, 8, 16, 32, 64
- Scheme: RPB (shared-k), MRPB (shared-k)

### Dependent Variables / Metrics
- Correctness rate per depth
- MRPB decrypt_ok rate
- Evaluation time (ms)
- RPB ciphertext size growth (bytes)

### Control Conditions
- 8 small rationals (1/2, 1/3, 2/3, 3/4, 3/5, 2/5, 5/7, 4/9)
- Seed 42, RPB c=7 t=2, MRPB key_bits=2048 t=2
- 100 trials per depth per circuit type

### Results

**100% correctness across ALL 4,200 trials — every depth, every circuit, both schemes.**

MRPB addition chains did NOT fail even at depth 64. The small operands
keep the accumulated result well within the sqrt(n) ≈ 1024-bit bound.

#### Evaluation Time (ms, mean — homomorphic ops only)

| Depth | mul RPB  | mul MRPB | add RPB  | add MRPB | mix RPB  | mix MRPB |
|-------|----------|----------|----------|----------|----------|----------|
| 1     | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    | 0.000    |
| 2     | 0.000    | 0.003    | 0.001    | 0.003    | 0.001    | 0.003    |
| 4     | 0.001    | 0.008    | 0.001    | 0.010    | 0.001    | 0.009    |
| 8     | 0.002    | 0.018    | 0.002    | 0.022    | 0.002    | 0.021    |
| 16    | 0.004    | 0.038    | 0.005    | 0.045    | 0.005    | 0.042    |
| 32    | 0.007    | 0.078    | 0.008    | 0.092    | 0.009    | 0.085    |
| 64    | 0.014    | 0.156    | 0.017    | 0.186    | 0.018    | 0.171    |

#### RPB Ciphertext Size Growth (bytes, mean)

| Depth | multiply | add   | mixed |
|-------|----------|-------|-------|
| 1     | 6        | 6     | 6     |
| 2     | 10       | 10    | 10    |
| 4     | 19       | 20    | 19    |
| 8     | 37       | 38    | 38    |
| 16    | 72       | 74    | 74    |
| 32    | 143      | 147   | 147   |
| 64    | 285      | 293   | 292   |

### Observations

1. **100% correctness at depth 64** — stronger than hypothesized. Both schemes
   handle deep circuits without any failures.

2. **MRPB addition did NOT fail** at any tested depth. With small operands
   (single-digit rationals), the accumulated ratio stays within sqrt(n) even
   after 64 additions. Larger operands would hit the bound sooner — this is
   a function of operand magnitude, not circuit depth alone.

3. **Eval time scales linearly with depth** in both schemes. MRPB is ~10×
   slower per operation (modular arithmetic vs plain bignum).

4. **RPB ciphertext grows linearly** — roughly 4.5 bytes per depth level.
   At depth 64, ciphertexts are ~285 bytes. MRPB stays fixed at 512 bytes
   (2× 256 bytes for a 2048-bit modulus) regardless of depth.

5. **MRPB decryption time is constant** across depths (~0.017ms). Rational
   reconstruction cost depends on modulus size, not circuit complexity.

6. **Addition is ~30% slower than multiplication** in eval time (both schemes).
   The cross-multiply-and-add in eval_add is more expensive than the simple
   multiply in eval_multiply.

### Interpretation

**Hypothesis partially confirmed, partially exceeded:**
- Multiplication unlimited: confirmed ✓
- Addition limited by sqrt(n): technically correct but NOT reached at depth 64
  with small operands. The practical bound depends on operand magnitude.
- Both schemes show no noise-like degradation — the unlimited depth property
  is a real structural advantage over BGV/BFV/CKKS.

**For the paper:** The depth benchmark is a strong selling point. Emphasize
that no existing HE scheme achieves noise-free unlimited multiplicative depth.
The addition constraint (sqrt(n) plaintext space) is analogous to noise budgets
in lattice HE but deterministic and predictable.

### Artifacts
- Script: scripts/run_depth_benchmark.py
- Config: experiments/exp_002_depth/config.yaml
- Raw data: experiments/exp_002_depth/results_raw.csv
- Summary: experiments/exp_002_depth/summary.json
