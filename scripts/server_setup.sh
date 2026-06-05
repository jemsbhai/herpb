#!/bin/bash
# ================================================================
# Server setup for EXP-010: 64-bit Secret Sharing Benchmark
# Works on A100 / B200 / any CUDA-capable server
# Run: bash scripts/server_setup.sh
# ================================================================

set -e
echo "=== EXP-010 Server Setup ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# ── Clone and install package ──
if [ ! -d "herpb" ]; then
    echo "Cloning repository..."
    git clone https://github.com/jemsbhai/herpb.git
fi
cd herpb
git pull

# ── Upgrade pip/setuptools first (prevents UNKNOWN package bug) ──
pip install --upgrade pip setuptools wheel --break-system-packages 2>/dev/null \
    || pip install --upgrade pip setuptools wheel

pip install -e . --break-system-packages 2>/dev/null || pip install -e .

# Verify the package installed correctly
PKG_NAME=$(pip show herpb 2>/dev/null | grep "^Name:" | awk '{print $2}')
if [ "$PKG_NAME" != "herpb" ]; then
    echo "WARNING: Package installed as '$PKG_NAME' instead of 'herpb'"
    echo "Trying: pip install -e . --no-build-isolation"
    pip install -e . --no-build-isolation --break-system-packages 2>/dev/null \
        || pip install -e . --no-build-isolation
fi

# ── Core dependencies ──
pip install numpy scipy matplotlib pyyaml tqdm --break-system-packages 2>/dev/null \
    || pip install numpy scipy matplotlib pyyaml tqdm

# ── HE baselines (CKKS, Paillier) ──
pip install tenseal phe --break-system-packages 2>/dev/null \
    || pip install tenseal phe

# ── Real embeddings ──
pip install sentence-transformers datasets --break-system-packages 2>/dev/null \
    || pip install sentence-transformers datasets

# ── GPU (CuPy for Montgomery kernel extension) ──
echo ""
echo "Detecting CUDA version..."
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d. -f1)
echo "CUDA major version: $CUDA_VER"

if [ "$CUDA_VER" -ge 12 ]; then
    pip install cupy-cuda12x --break-system-packages 2>/dev/null \
        || pip install cupy-cuda12x
elif [ "$CUDA_VER" -ge 11 ]; then
    pip install cupy-cuda11x --break-system-packages 2>/dev/null \
        || pip install cupy-cuda11x
else
    echo "CUDA <11 detected, skipping CuPy"
fi

# ── System info ──
echo ""
echo "=== System Info ==="
python -c "
import platform, os
print(f'Python:   {platform.python_version()}')
print(f'OS:       {platform.platform()}')
print(f'CPU:      {os.cpu_count()} cores')
try:
    import subprocess
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader']).decode().strip()
    print(f'GPU:      {gpu}')
except: print('GPU:      not detected')
"

# ── Verify imports ──
echo ""
echo "=== Dependency Verification ==="
python -c "
import numpy; print(f'numpy:          {numpy.__version__}')
import scipy; print(f'scipy:          {scipy.__version__}')
try:
    import tenseal as ts; print(f'tenseal:        {ts.__version__}')
except: print('tenseal:        NOT INSTALLED')
try:
    from phe import paillier; print('paillier:       OK')
except: print('paillier:       NOT INSTALLED')
try:
    from sentence_transformers import SentenceTransformer; print('sent-xformers:  OK')
except: print('sent-xformers:  NOT INSTALLED')
try:
    import cupy as cp
    print(f'cupy:           {cp.__version__} (CUDA {cp.cuda.runtime.runtimeGetVersion()})')
except: print('cupy:           NOT INSTALLED (Montgomery phases will be skipped)')
"

# ── Quick functional test ──
echo ""
echo "=== Functional Test ==="
python -c "
import numpy as np
from herpb.utils.encoding import encode_vector
from herpb.schemes.secret_sharing_64 import (
    share_vector, reconstruct_scalar,
    query_private_batch, share_matrix,
    setup_matrix_beaver, generate_query_triple,
    mutual_privacy_batch,
)

# Test query-private
k = 16
q = [0.5, -0.3, 0.7, 0.1]
d = [0.2, 0.8, -0.1, 0.4]
q_ints = [n for n, _ in encode_vector(q, k)]
d_ints = [n for n, _ in encode_vector(d, k)]
D = np.array([d_ints], dtype=np.int64)
q1, q2 = share_vector(q_ints)
dots1, dots2 = query_private_batch(q1, q2, D)
result = reconstruct_scalar(np.uint64(dots1[0]), np.uint64(dots2[0]))
score = result / (2**k)**2
true_val = sum(a*b for a, b in zip(q, d))
print(f'Query-private dot: {score:.6f} (true: {true_val:.6f}) ', end='')
print('PASS' if abs(score - true_val) < 0.01 else 'FAIL')

# Test mutual-privacy
rng = np.random.default_rng(42)
D1, D2 = share_matrix(D, rng)
setup = setup_matrix_beaver(D1, D2, rng)
triple = generate_query_triple(setup, rng)
q1, q2 = share_vector(q_ints, rng)
Z1, Z2 = mutual_privacy_batch(q1, q2, setup, triple)
result2 = reconstruct_scalar(np.uint64(Z1[0]), np.uint64(Z2[0]))
score2 = result2 / (2**k)**2
print(f'Mutual-priv  dot: {score2:.6f} (true: {true_val:.6f}) ', end='')
print('PASS' if abs(score2 - true_val) < 0.01 else 'FAIL')

print()
print('=== Setup complete. Run:')
print('  python scripts/run_server_benchmark.py 2>&1 | tee exp010.log')
print('  # or specific phases:')
print('  python scripts/run_server_benchmark.py --phases 1,2,3')
print('  # estimated time: 2-4 hours total (Phase 6 real embeddings is slowest)')
"
