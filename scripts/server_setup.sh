#!/bin/bash
# Server setup for EXP-007: Paper Benchmark
# Works on A100 / B200 / any CUDA-capable server
# Run: bash scripts/server_setup.sh

set -e

echo "=== EXP-007 Server Setup ==="

# ── Clone and install package ──
if [ ! -d "herpb" ]; then
    git clone https://github.com/jemsbhai/herpb.git
fi
cd herpb

pip install -e . --break-system-packages 2>/dev/null || pip install -e .

# ── Core dependencies ──
pip install numpy scipy matplotlib pyyaml tqdm --break-system-packages 2>/dev/null \
    || pip install numpy scipy matplotlib pyyaml tqdm

# ── HE baselines ──
pip install tenseal phe --break-system-packages 2>/dev/null \
    || pip install tenseal phe

# ── Real embeddings ──
pip install sentence-transformers datasets --break-system-packages 2>/dev/null \
    || pip install sentence-transformers datasets

# ── GPU (try CuPy for available CUDA) ──
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

# ── Verify ──
echo ""
echo "=== Verification ==="
python -c "
import gmpy2; print(f'gmpy2:    {gmpy2.version()}')
import numpy; print(f'numpy:    {numpy.__version__}')
import scipy; print(f'scipy:    {scipy.__version__}')
try:
    import tenseal as ts; print(f'tenseal:  {ts.__version__}')
except: print('tenseal:  NOT INSTALLED')
try:
    from phe import paillier; print('paillier: OK')
except: print('paillier: NOT INSTALLED')
try:
    from sentence_transformers import SentenceTransformer; print('sent-xfm: OK')
except: print('sent-xfm: NOT INSTALLED')
try:
    import cupy as cp
    print(f'cupy:     {cp.__version__} (CUDA {cp.cuda.runtime.runtimeGetVersion()})')
except: print('cupy:     NOT INSTALLED (GPU phases will be skipped)')
print()
print('Quick scheme test...')
from herpb.schemes import rpb, mrpb
from herpb.utils.encoding import encode_vector
from herpb.rag.similarity import rpb_encrypt_vector, rpb_dot_product
import gmpy2
c = gmpy2.mpz(12345)
q = encode_vector([0.5, -0.3], 16)
d = encode_vector([0.4,  0.6], 16)
qc = rpb_encrypt_vector(q, c)
dc = rpb_encrypt_vector(d, c)
dot = rpb_dot_product(qc, dc)
n, dn = rpb.decrypt(dot)
print(f'RPB dot([0.5,-0.3],[0.4,0.6]) = {float(n)/float(dn):.6f} (true: 0.02)')
print()
print('=== Setup complete. Run:')
print('  python scripts/run_paper_benchmark.py 2>&1 | tee exp007.log')
print('  # or specific phases:')
print('  python scripts/run_paper_benchmark.py --phase 2,5,6')
print('  # resume after interruption:')
print('  python scripts/run_paper_benchmark.py --resume')
"
