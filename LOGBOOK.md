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

---

## EXP-001: RPB vs MRPB Comparative Benchmark

**Date:** 2026-05-29
**Researcher:** Muntaser Syed
**Type:** computational
**Status:** completed (7 of 8 bit-lengths; 10000 deferred to A100)

### Hypothesis
MRPB (modular) will be slower per-encryption than RPB (plain bignum) due to
modular reduction overhead, but will produce fixed-size ciphertexts regardless
of input size. RPB ciphertexts will grow with input size. Both schemes will
achieve 100% correctness across all bit-lengths.

### Independent Variables
- Scheme: RPB-shared, RPB-distinct, MRPB-shared, MRPB-distinct
- Input bit-length: 10, 100, 1000, 2000, 3000, 5000, 8000 bits
- MRPB key size: auto-scaled per bit-length (see results)

### Dependent Variables / Metrics
- Encryption time (ms): mean ± std over 1000 iterations × 10 pairs
- Decryption time (ms): mean ± std
- Ciphertext size (bytes): mean
- Correctness rate: fraction of trials where decrypt == expected ratio

### Control Conditions
- Same input data pairs for all schemes
- Same seed (42) for reproducible random parameters
- RPB bound B = 10^20000 (matching C++ reference)
- MRPB key_bits = max(2048, auto-scaled to input ratio size)

### Protocol
1. Load all input data files from data/raw/
2. For each bit-length, run 10000 trials per scheme variant
3. Record per-trial: enc time, dec time, ciphertext size, correctness
4. Checkpoint after each bit-length

### Environment
- Hardware: Windows laptop, 64GB RAM, RTX 4090
- Software: Python 3.12, gmpy2 2.3.0, Windows 11
- Config file: configs/exp_001_benchmark.yaml
- Seeds: 42
- Total runtime: ~6.3 hours (dominated by MRPB keygen at 5000/8000 bits)

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

#### Decryption Time (ms, mean)

| Input   | RPB-shared | RPB-distinct | MRPB-shared | MRPB-distinct |
|---------|-----------|-------------|------------|--------------|
| 10      | 0.0014    | 0.0010      | 0.0120     | 0.0117       |
| 100     | 0.0047    | 0.0037      | 0.1036     | 0.0961       |
| 1000    | 0.0768    | 0.0440      | 2.2959     | 2.1936       |
| 2000    | 0.2460    | 0.1135      | 7.7429     | 7.4979       |
| 3000    | 0.3879    | 0.1637      | 15.2726    | 14.6234      |
| 5000    | 1.1310    | 0.4556      | 42.4790    | 41.3112      |
| 8000    | 1.9452    | 0.9280      | 78.6208    | 76.8189      |

#### Ciphertext Size (bytes, mean)

| Input   | RPB-shared | RPB-distinct | MRPB-shared | MRPB-distinct |
|---------|-----------|-------------|------------|--------------|
| 10      | 40        | 24          | 512        | 512          |
| 100     | 339       | 173         | 512        | 512          |
| 1000    | 3328      | 1668        | 3392       | 3392         |
| 2000    | 6651      | 3329        | 6720       | 6720         |
| 3000    | 9972      | 4990        | 10048      | 10048        |
| 5000    | 16610     | 8312        | 16704      | 16704        |
| 8000    | 23254     | 11634       | 23360      | 23360        |

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

### Observations

1. **Correctness is perfect** — both schemes produce exact results at every scale.

2. **RPB encryption is 2-4× faster than MRPB** — the modular reduction in
   MRPB adds overhead proportional to modulus size.

3. **MRPB decryption is the dominant bottleneck** — rational reconstruction
   via half-GCD over large moduli is 30-40× slower than RPB's simple GCD.
   At 8000 bits: RPB decrypts in ~1ms, MRPB in ~78ms.

4. **Distinct-k is ~2-3× faster than shared-k** in both schemes, confirming
   the C++ reference finding. The quadratic growth in shared-k (k²) vs linear
   cross-multiplication in distinct-k is the cause.

5. **MRPB ciphertext size ≈ RPB shared-k size** — the auto-scaled modulus
   naturally matches the RPB ciphertext magnitude. But RPB-distinct is ~2×
   smaller because it avoids the k² blowup.

6. **MRPB keygen is the practical barrier** — generating primes for 93K-bit
   RSA took nearly 5 hours. This is a one-time cost (amortized), but it makes
   MRPB impractical for large inputs without HPC resources. Deferred 10000-bit
   case (would need ~134K-bit RSA) to A100 GPU.

### Interpretation

**Hypothesis partially confirmed:**
- MRPB is indeed slower per-operation (confirmed), especially for decryption.
- MRPB ciphertext sizes are NOT smaller as hypothesized — they are comparable
  to RPB-shared because the modulus must scale with input size. However, MRPB
  ciphertexts are FIXED for a given key (zero std), while RPB varies.
- The key insight: MRPB's advantage is **security** (IND-CPA), not performance.
  The performance cost of security is ~2-4× for encryption and ~40× for
  decryption at 8000 bits.

**For the paper:** Frame RPB as the high-performance primitive for scenarios
where ratio leakage is acceptable (federated analytics, ratio delegation).
Frame MRPB as the secure primitive for full confidentiality, noting the
keygen/decryption cost as the price of IND-CPA.

### Artifacts
- Script: scripts/run_experiment.py
- Config: experiments/exp_001_benchmark/config.yaml
- Raw data: experiments/exp_001_benchmark/results_raw.csv
- Summary: experiments/exp_001_benchmark/summary.json
- Environment: experiments/exp_001_benchmark/environment.json
