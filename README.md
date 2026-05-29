# HERPB: Homomorphic Encryption via Ratio-Preserving Blinding

Research implementation of two novel cryptographic primitives for privacy-preserving
rational number arithmetic:

- **RPB (Ratio-Preserving Blinding):** Operand-hiding scheme with unlimited
  homomorphic depth, no noise accumulation, and no bootstrapping. Leaks the
  ratio a/b but hides individual operands a and b.

- **MRPB (Modular Ratio-Preserving Blinding):** Extension to modular arithmetic
  over an RSA modulus. Achieves IND-CPA security — the ratio itself is hidden.
  Fixed-size ciphertexts. Same unlimited homomorphic depth.

Both schemes support addition and multiplication of encrypted rationals without
noise growth or bootstrapping — a property not shared by BGV, BFV, or CKKS.

## Project Structure

```
herpb/
├── src/herpb/           # Core library
│   ├── schemes/         # RPB and MRPB implementations
│   ├── utils/           # Bignum, math, benchmarking utilities
│   └── analysis/        # Statistical analysis of experiment results
├── tests/               # Test suite (pytest)
├── configs/             # Experiment configurations (YAML)
├── data/                # Input data (rational number pairs)
├── experiments/         # Experiment outputs
├── scripts/             # Entry-point scripts
├── figures/             # Publication-ready figures
├── notebooks/           # Exploratory analysis
└── paper/               # LaTeX source
```

## Setup

```powershell
cd E:\data\code\claudecode\herpb
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running Tests

```powershell
pytest tests/ -v
```

## Running Experiments

```powershell
python scripts/run_experiment.py --config configs/base.yaml
```

## Authors

- Muntaser Syed (Florida Institute of Technology)
- [Collaborator name TBD]

## License

MIT
