#!/usr/bin/env python3
"""
EXP-004: Privacy-Preserving RAG Similarity Benchmark
=====================================================

Comprehensive benchmark comparing RPB/MRPB vs CKKS for encrypted
similarity computation in RAG pipelines.  Produces every number
and figure for the IEEE ICTAI 2026 paper.

Usage:
  python scripts/run_rag_benchmark.py                    # All phases
  python scripts/run_rag_benchmark.py --phase 1          # Specific phase
  python scripts/run_rag_benchmark.py --phase 1,2,3      # Multiple phases
  python scripts/run_rag_benchmark.py --resume            # Resume from checkpoint
  python scripts/run_rag_benchmark.py --figures-only      # Regenerate figures

A100 setup:
  git clone <repo_url> && cd herpb
  pip install -e . --break-system-packages
  pip install tenseal sentence-transformers datasets numpy tqdm pyyaml matplotlib --break-system-packages
  python scripts/run_rag_benchmark.py
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

# ── Project imports ───────────────────────────────────────────────────
from gmpy2 import mpz
import gmpy2

from herpb.schemes import rpb, mrpb
from herpb.schemes.rpb import Rational
from herpb.utils.encoding import (
    encode_vector,
    quantization_error_bound,
    dot_product_error_bound,
)
from herpb.rag.similarity import (
    rpb_encrypt_vector,
    rpb_dot_product,
    mrpb_encrypt_vector,
    mrpb_dot_product,
    plaintext_dot_product,
)

# ── Optional imports ──────────────────────────────────────────────────
try:
    import tenseal as ts
    HAS_TENSEAL = True
except ImportError:
    HAS_TENSEAL = False
    print("[WARN] tenseal not installed — Phase 4 (CKKS comparison) will be skipped")

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("[WARN] sentence-transformers not installed — real embedding tests skipped")

try:
    from tqdm import tqdm
except ImportError:
    # Minimal fallback
    def tqdm(it, **kw):
        total = kw.get("total", None)
        desc = kw.get("desc", "")
        for i, x in enumerate(it):
            if total and (i % max(1, total // 20) == 0):
                print(f"  {desc}: {i}/{total}", flush=True)
            yield x

# ── Constants ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "exp_004_rag_benchmark"
FIGURE_DIR = PROJECT_ROOT / "figures"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
CONFIG_FILE = PROJECT_ROOT / "configs" / "exp_004_rag_benchmark.yaml"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Section 1: Checkpoint Management
# =====================================================================

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed_phases": [], "started_at": datetime.now(timezone.utc).isoformat()}


def save_checkpoint(ckpt: dict):
    ckpt["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(ckpt, f, indent=2)


def mark_phase_complete(ckpt: dict, phase: str, duration_s: float):
    ckpt["completed_phases"].append({
        "phase": phase,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(duration_s, 2),
    })
    save_checkpoint(ckpt)


def is_phase_done(ckpt: dict, phase: str) -> bool:
    return any(p["phase"] == phase for p in ckpt.get("completed_phases", []))


# =====================================================================
# Section 2: Helpers
# =====================================================================

def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def random_unit_vectors(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n random unit vectors of given dimension."""
    vecs = rng.standard_normal((n, dim))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def true_dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def make_rpb_blinding(seed: int = 0) -> mpz:
    rs = gmpy2.random_state(seed)
    c = gmpy2.mpz_urandomb(rs, 256)
    while c == 0:
        c = gmpy2.mpz_urandomb(rs, 256)
    return c


def recall_at_k(true_ranking: list[int], pred_ranking: list[int], k: int) -> float:
    """Fraction of true top-k that appear in predicted top-k."""
    true_set = set(true_ranking[:k])
    pred_set = set(pred_ranking[:k])
    return len(true_set & pred_set) / k


def save_csv(filepath: Path, rows: list[dict]):
    if not rows:
        return
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {len(rows)} rows → {filepath}")


def save_json(filepath: Path, data):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved → {filepath}")


def timer():
    """Context-manager-style timer."""
    class Timer:
        def __init__(self):
            self.start = time.perf_counter()
        def elapsed(self):
            return time.perf_counter() - self.start
    return Timer()


# =====================================================================
# Section 3: Phase 1 — Correctness at Scale
# =====================================================================

def run_phase1(cfg: dict):
    """Verify encrypted dot product = plaintext (zero encryption error)."""
    print("\n" + "=" * 70)
    print("PHASE 1: Correctness at Scale")
    print("=" * 70)

    p1 = cfg["phase1_correctness"]
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    rows = []

    # ── RPB sweep ──
    c = make_rpb_blinding(42)
    total_rpb = len(p1["precision_bits"]) * len(p1["dimensions"]) * p1["trials_per_config"]
    pbar = tqdm(total=total_rpb, desc="Phase1 RPB")

    for k in p1["precision_bits"]:
        for dim in p1["dimensions"]:
            for trial in range(p1["trials_per_config"]):
                q_f = random_unit_vectors(1, dim, rng)[0].tolist()
                d_f = random_unit_vectors(1, dim, rng)[0].tolist()

                q_rats = encode_vector(q_f, k)
                d_rats = encode_vector(d_f, k)

                # Plaintext encoded
                expected = plaintext_dot_product(q_rats, d_rats)

                # Encrypted
                t0 = time.perf_counter()
                q_cts = rpb_encrypt_vector(q_rats, c)
                d_cts = rpb_encrypt_vector(d_rats, c)
                dot_ct = rpb_dot_product(q_cts, d_cts)
                num, den = rpb.decrypt(dot_ct)
                elapsed = time.perf_counter() - t0

                encrypted_result = float(num) / float(den)
                enc_error = abs(encrypted_result - expected)
                true_val = true_dot(q_f, d_f)
                quant_error = abs(true_val - encrypted_result)
                bound = dot_product_error_bound(k, dim)

                rows.append({
                    "scheme": "rpb",
                    "precision_bits": k,
                    "dimension": dim,
                    "trial": trial,
                    "true_dot": true_val,
                    "encoded_dot": expected,
                    "encrypted_dot": encrypted_result,
                    "encryption_error": enc_error,
                    "quantization_error": quant_error,
                    "analytical_bound": bound,
                    "within_bound": quant_error <= bound + 1e-12,
                    "time_s": elapsed,
                })
                pbar.update(1)
    pbar.close()

    # ── MRPB sweep (smaller subset) ──
    print("  Generating MRPB key (2048-bit)...")
    _, sk = mrpb.keygen(key_bits=p1["mrpb_key_bits"], precision=0, seed=cfg["experiment"]["seed"])

    total_mrpb = len(p1["mrpb_precision_bits"]) * len(p1["mrpb_dimensions"]) * p1["mrpb_trials_per_config"]
    pbar = tqdm(total=total_mrpb, desc="Phase1 MRPB")

    for k in p1["mrpb_precision_bits"]:
        for dim in p1["mrpb_dimensions"]:
            for trial in range(p1["mrpb_trials_per_config"]):
                q_f = random_unit_vectors(1, dim, rng)[0].tolist()
                d_f = random_unit_vectors(1, dim, rng)[0].tolist()

                q_rats = encode_vector(q_f, k)
                d_rats = encode_vector(d_f, k)
                expected = plaintext_dot_product(q_rats, d_rats)

                t0 = time.perf_counter()
                q_cts = mrpb_encrypt_vector(q_rats, sk)
                d_cts = mrpb_encrypt_vector(d_rats, sk)
                dot_ct = mrpb_dot_product(q_cts, d_cts)
                num, den = mrpb.decrypt(dot_ct, sk)
                elapsed = time.perf_counter() - t0

                encrypted_result = float(num) / float(den)
                enc_error = abs(encrypted_result - expected)
                true_val = true_dot(q_f, d_f)
                quant_error = abs(true_val - encrypted_result)
                bound = dot_product_error_bound(k, dim)

                rows.append({
                    "scheme": "mrpb",
                    "precision_bits": k,
                    "dimension": dim,
                    "trial": trial,
                    "true_dot": true_val,
                    "encoded_dot": expected,
                    "encrypted_dot": encrypted_result,
                    "encryption_error": enc_error,
                    "quantization_error": quant_error,
                    "analytical_bound": bound,
                    "within_bound": quant_error <= bound + 1e-12,
                    "time_s": elapsed,
                })
                pbar.update(1)
    pbar.close()

    save_csv(OUTPUT_DIR / "phase1_correctness.csv", rows)

    # Summary
    rpb_rows = [r for r in rows if r["scheme"] == "rpb"]
    mrpb_rows = [r for r in rows if r["scheme"] == "mrpb"]
    max_rpb_enc_err = max(r["encryption_error"] for r in rpb_rows)
    max_mrpb_enc_err = max(r["encryption_error"] for r in mrpb_rows)
    all_within = all(r["within_bound"] for r in rows)

    summary = {
        "total_trials": len(rows),
        "rpb_trials": len(rpb_rows),
        "mrpb_trials": len(mrpb_rows),
        "max_rpb_encryption_error": max_rpb_enc_err,
        "max_mrpb_encryption_error": max_mrpb_enc_err,
        "all_within_analytical_bound": all_within,
        "zero_encryption_error": max_rpb_enc_err < 1e-10 and max_mrpb_enc_err < 1e-10,
    }
    save_json(OUTPUT_DIR / "phase1_summary.json", summary)

    print(f"\n  Phase 1 complete: {len(rows)} trials")
    print(f"  Max RPB encryption error:  {max_rpb_enc_err:.2e}")
    print(f"  Max MRPB encryption error: {max_mrpb_enc_err:.2e}")
    print(f"  All within analytical bound: {all_within}")


# =====================================================================
# Section 4: Phase 2 — Error Analysis
# =====================================================================

def run_phase2(cfg: dict):
    """Validate analytical error bounds and demonstrate zero noise accumulation."""
    print("\n" + "=" * 70)
    print("PHASE 2: Error Analysis")
    print("=" * 70)

    p2 = cfg["phase2_error"]
    rng = np.random.default_rng(cfg["experiment"]["seed"] + 100)
    c = make_rpb_blinding(99)

    # ── Part A: Bound validation ──
    rows_bound = []
    total = len(p2["precision_bits"]) * len(p2["dimensions"]) * p2["trials_per_config"]
    pbar = tqdm(total=total, desc="Phase2A Bounds")

    for k in p2["precision_bits"]:
        for dim in p2["dimensions"]:
            bound = dot_product_error_bound(k, dim)
            errors = []
            for trial in range(p2["trials_per_config"]):
                q_f = random_unit_vectors(1, dim, rng)[0].tolist()
                d_f = random_unit_vectors(1, dim, rng)[0].tolist()

                q_rats = encode_vector(q_f, k)
                d_rats = encode_vector(d_f, k)
                encoded_dot = plaintext_dot_product(q_rats, d_rats)
                true_val = true_dot(q_f, d_f)
                err = abs(true_val - encoded_dot)
                errors.append(err)
                pbar.update(1)

            rows_bound.append({
                "precision_bits": k,
                "dimension": dim,
                "analytical_bound": bound,
                "max_observed_error": max(errors),
                "mean_error": np.mean(errors),
                "median_error": np.median(errors),
                "p99_error": np.percentile(errors, 99),
                "bound_ratio": max(errors) / bound if bound > 0 else 0,
                "all_within_bound": all(e <= bound + 1e-12 for e in errors),
                "trials": len(errors),
            })
    pbar.close()
    save_csv(OUTPUT_DIR / "phase2a_bound_validation.csv", rows_bound)

    # ── Part B: Zero noise accumulation (depth test) ──
    # Compute dot product, then chain additional multiplications.
    # Error should NOT grow with depth.
    rows_depth = []
    print("  Phase 2B: Depth-invariant error test...")

    for k in [12, 16, 20]:
        dim = 64
        for depth in p2["homomorphic_depths"]:
            for trial in range(p2["depth_trials"]):
                q_f = random_unit_vectors(1, dim, rng)[0].tolist()
                d_f = random_unit_vectors(1, dim, rng)[0].tolist()

                q_rats = encode_vector(q_f, k)
                d_rats = encode_vector(d_f, k)
                q_cts = rpb_encrypt_vector(q_rats, c)
                d_cts = rpb_encrypt_vector(d_rats, c)

                # Compute dot product
                dot_ct = rpb_dot_product(q_cts, d_cts)

                # Chain `depth` additional multiplications by Enc(1/1)
                one_ct = rpb.shared_k_encrypt(
                    Rational(mpz(1), mpz(1)),
                    Rational(mpz(1), mpz(1)),
                    t=0, c=c,
                )
                result_ct = dot_ct
                for _ in range(depth):
                    result_ct = rpb.eval_multiply(result_ct, one_ct)

                num, den = rpb.decrypt(result_ct)
                encrypted_val = float(num) / float(den)
                true_val = true_dot(q_f, d_f)
                encoded_val = plaintext_dot_product(q_rats, d_rats)

                rows_depth.append({
                    "precision_bits": k,
                    "dimension": dim,
                    "depth": depth,
                    "trial": trial,
                    "true_dot": true_val,
                    "encrypted_dot": encrypted_val,
                    "error_vs_true": abs(true_val - encrypted_val),
                    "error_vs_encoded": abs(encoded_val - encrypted_val),
                })

    save_csv(OUTPUT_DIR / "phase2b_depth_invariance.csv", rows_depth)

    # Verify error doesn't grow with depth
    for k in [12, 16, 20]:
        depth_errors = {}
        for r in rows_depth:
            if r["precision_bits"] == k:
                d = r["depth"]
                if d not in depth_errors:
                    depth_errors[d] = []
                depth_errors[d].append(r["error_vs_encoded"])
        print(f"  k={k}: max error by depth: " +
              ", ".join(f"d={d}:{max(errs):.2e}" for d, errs in sorted(depth_errors.items())))

    print("  Phase 2 complete")


# =====================================================================
# Section 5: Phase 3 — Ranking Preservation
# =====================================================================

def run_phase3(cfg: dict):
    """Verify rankings are preserved through encryption."""
    print("\n" + "=" * 70)
    print("PHASE 3: Ranking Preservation")
    print("=" * 70)

    p3 = cfg["phase3_ranking"]
    syn = p3["synthetic"]
    rng = np.random.default_rng(cfg["experiment"]["seed"] + 200)
    c = make_rpb_blinding(77)

    rows = []

    # ── RPB ranking tests ──
    for dim in syn["dimensions"]:
        for k in syn["precision_bits"]:
            print(f"  RPB ranking: dim={dim}, k={k}")
            # Generate corpus once per (dim, k)
            docs_f = random_unit_vectors(syn["num_documents"], dim, rng)
            queries_f = random_unit_vectors(syn["num_queries"], dim, rng)

            # Pre-encode and encrypt documents
            docs_rats = [encode_vector(d.tolist(), k) for d in docs_f]
            docs_cts = [rpb_encrypt_vector(dr, c) for dr in docs_rats]

            for qi in tqdm(range(syn["num_queries"]), desc=f"  RPB d={dim} k={k}", leave=False):
                q_f = queries_f[qi].tolist()
                q_rats = encode_vector(q_f, k)
                q_cts = rpb_encrypt_vector(q_rats, c)

                # True scores
                true_scores = [true_dot(q_f, d.tolist()) for d in docs_f]
                true_ranking = sorted(range(len(docs_f)), key=lambda i: -true_scores[i])

                # Encrypted scores
                enc_scores = []
                for d_cts in docs_cts:
                    dot_ct = rpb_dot_product(q_cts, d_cts)
                    num, den = rpb.decrypt(dot_ct)
                    enc_scores.append(float(num) / float(den))

                enc_ranking = sorted(range(len(docs_f)), key=lambda i: -enc_scores[i])

                for top_k in syn["top_k"]:
                    r_at_k = recall_at_k(true_ranking, enc_ranking, top_k)
                    rows.append({
                        "scheme": "rpb",
                        "dimension": dim,
                        "precision_bits": k,
                        "query_idx": qi,
                        "num_documents": syn["num_documents"],
                        "top_k": top_k,
                        "recall_at_k": r_at_k,
                    })

    save_csv(OUTPUT_DIR / "phase3_rpb_ranking.csv", rows)

    # ── MRPB validation subset ──
    mv = p3["mrpb_validation"]
    print("  Generating MRPB key for ranking validation...")
    _, sk = mrpb.keygen(key_bits=2048, precision=0, seed=cfg["experiment"]["seed"])
    mrpb_rows = []

    for dim in mv["dimensions"]:
        for k in mv["precision_bits"]:
            print(f"  MRPB ranking: dim={dim}, k={k}")
            docs_f = random_unit_vectors(mv["num_documents"], dim, rng)
            queries_f = random_unit_vectors(mv["num_queries"], dim, rng)

            docs_rats = [encode_vector(d.tolist(), k) for d in docs_f]
            docs_cts = [mrpb_encrypt_vector(dr, sk) for dr in docs_rats]

            for qi in tqdm(range(mv["num_queries"]), desc=f"  MRPB d={dim} k={k}", leave=False):
                q_f = queries_f[qi].tolist()
                q_rats = encode_vector(q_f, k)
                q_cts = mrpb_encrypt_vector(q_rats, sk)

                true_scores = [true_dot(q_f, d.tolist()) for d in docs_f]
                true_ranking = sorted(range(len(docs_f)), key=lambda i: -true_scores[i])

                enc_scores = []
                for d_cts in docs_cts:
                    dot_ct = mrpb_dot_product(q_cts, d_cts)
                    num, den = mrpb.decrypt(dot_ct, sk)
                    enc_scores.append(float(num) / float(den))

                enc_ranking = sorted(range(len(docs_f)), key=lambda i: -enc_scores[i])

                for top_k in mv["top_k"]:
                    r_at_k = recall_at_k(true_ranking, enc_ranking, top_k)
                    mrpb_rows.append({
                        "scheme": "mrpb",
                        "dimension": dim,
                        "precision_bits": k,
                        "query_idx": qi,
                        "num_documents": mv["num_documents"],
                        "top_k": top_k,
                        "recall_at_k": r_at_k,
                    })

    save_csv(OUTPUT_DIR / "phase3_mrpb_ranking.csv", mrpb_rows)

    # ── Summary ──
    all_rows = rows + mrpb_rows
    perfect = sum(1 for r in all_rows if r["recall_at_k"] == 1.0)
    total = len(all_rows)
    summary = {
        "total_ranking_tests": total,
        "perfect_recall": perfect,
        "perfect_fraction": perfect / total if total > 0 else 0,
        "any_ranking_error": perfect < total,
    }
    save_json(OUTPUT_DIR / "phase3_summary.json", summary)
    print(f"\n  Phase 3: {perfect}/{total} perfect recall ({100*perfect/total:.1f}%)")


# =====================================================================
# Section 6: Phase 4 — CKKS Comparison
# =====================================================================

def ckks_dot_product(v1: list[float], v2: list[float], context) -> tuple[float, float]:
    """Compute encrypted dot product using TenSEAL CKKS. Returns (result, time_s)."""
    t0 = time.perf_counter()
    enc_v1 = ts.ckks_vector(context, v1)
    enc_v2 = ts.ckks_vector(context, v2)
    enc_result = enc_v1.dot(enc_v2)
    result = enc_result.decrypt()[0]
    elapsed = time.perf_counter() - t0
    return result, elapsed


def run_phase4(cfg: dict):
    """Direct comparison with TenSEAL CKKS."""
    if not HAS_TENSEAL:
        print("\n[SKIP] Phase 4: tenseal not installed")
        return

    print("\n" + "=" * 70)
    print("PHASE 4: CKKS Comparison")
    print("=" * 70)

    p4 = cfg["phase4_ckks"]
    if not p4.get("enabled", True):
        print("  Phase 4 disabled in config")
        return

    rng = np.random.default_rng(cfg["experiment"]["seed"] + 300)
    c = make_rpb_blinding(55)

    # ── Part A: CKKS accuracy vs RPB/MRPB ──
    rows = []

    for poly_deg in p4["poly_modulus_degrees"]:
        for config_name, coeff_bits in p4["coeff_mod_configs"].items():
            for scale_bits in p4["global_scales"]:
                # Create CKKS context
                try:
                    context = ts.context(
                        ts.SCHEME_TYPE.CKKS,
                        poly_modulus_degree=poly_deg,
                        coeff_mod_bit_sizes=coeff_bits,
                    )
                    context.generate_galois_keys()
                    context.global_scale = 2 ** scale_bits
                except Exception as e:
                    print(f"  [SKIP] poly={poly_deg}, {config_name}, scale={scale_bits}: {e}")
                    continue

                for dim in p4["dimensions"]:
                    # Skip if poly degree too small for dimension
                    if dim > poly_deg // 2:
                        continue

                    print(f"  CKKS: poly={poly_deg}, {config_name}, scale={scale_bits}, dim={dim}")

                    docs_f = random_unit_vectors(p4["num_documents"], dim, rng)
                    queries_f = random_unit_vectors(p4["num_queries"], dim, rng)

                    for qi in tqdm(range(p4["num_queries"]),
                                   desc=f"  CKKS p={poly_deg} d={dim}", leave=False):
                        q_f = queries_f[qi].tolist()

                        # True scores
                        true_scores = [true_dot(q_f, d.tolist()) for d in docs_f]
                        true_ranking = sorted(range(len(docs_f)), key=lambda i: -true_scores[i])

                        # CKKS scores
                        ckks_scores = []
                        ckks_times = []
                        ckks_failed = False
                        for di, d in enumerate(docs_f):
                            try:
                                score, t_ckks = ckks_dot_product(q_f, d.tolist(), context)
                                ckks_scores.append(score)
                                ckks_times.append(t_ckks)
                            except Exception:
                                ckks_failed = True
                                break

                        if ckks_failed:
                            continue

                        ckks_ranking = sorted(range(len(docs_f)), key=lambda i: -ckks_scores[i])

                        # CKKS errors
                        ckks_errors = [abs(true_scores[i] - ckks_scores[i])
                                       for i in range(len(docs_f))]

                        # RPB scores for comparison (one precision)
                        for k in p4["precision_bits_for_comparison"]:
                            q_rats = encode_vector(q_f, k)
                            q_cts_rpb = rpb_encrypt_vector(q_rats, c)

                            rpb_scores = []
                            rpb_times = []
                            for d in docs_f:
                                d_rats = encode_vector(d.tolist(), k)
                                d_cts_rpb = rpb_encrypt_vector(d_rats, c)
                                t0 = time.perf_counter()
                                dot_ct = rpb_dot_product(q_cts_rpb, d_cts_rpb)
                                num, den = rpb.decrypt(dot_ct)
                                rpb_times.append(time.perf_counter() - t0)
                                rpb_scores.append(float(num) / float(den))

                            rpb_ranking = sorted(range(len(docs_f)), key=lambda i: -rpb_scores[i])
                            rpb_errors = [abs(true_scores[i] - rpb_scores[i])
                                          for i in range(len(docs_f))]

                            for top_k in p4["top_k"]:
                                ckks_recall = recall_at_k(true_ranking, ckks_ranking, top_k)
                                rpb_recall = recall_at_k(true_ranking, rpb_ranking, top_k)

                                rows.append({
                                    "query_idx": qi,
                                    "dimension": dim,
                                    "poly_modulus_degree": poly_deg,
                                    "coeff_config": config_name,
                                    "scale_bits": scale_bits,
                                    "rpb_precision_bits": k,
                                    "top_k": top_k,
                                    "ckks_recall_at_k": ckks_recall,
                                    "rpb_recall_at_k": rpb_recall,
                                    "ckks_max_error": max(ckks_errors),
                                    "ckks_mean_error": np.mean(ckks_errors),
                                    "rpb_max_error": max(rpb_errors),
                                    "rpb_mean_error": np.mean(rpb_errors),
                                    "ckks_mean_time_s": np.mean(ckks_times),
                                    "rpb_mean_time_s": np.mean(rpb_times),
                                    "num_documents": len(docs_f),
                                })

    save_csv(OUTPUT_DIR / "phase4_ckks_comparison.csv", rows)

    # Summary
    if rows:
        ckks_perfect = sum(1 for r in rows if r["ckks_recall_at_k"] == 1.0)
        rpb_perfect = sum(1 for r in rows if r["rpb_recall_at_k"] == 1.0)
        total = len(rows)
        summary = {
            "total_comparisons": total,
            "ckks_perfect_recall_fraction": ckks_perfect / total,
            "rpb_perfect_recall_fraction": rpb_perfect / total,
            "ckks_ranking_errors": total - ckks_perfect,
            "rpb_ranking_errors": total - rpb_perfect,
        }
        save_json(OUTPUT_DIR / "phase4_summary.json", summary)
        print(f"\n  CKKS perfect recall: {ckks_perfect}/{total} ({100*ckks_perfect/total:.1f}%)")
        print(f"  RPB  perfect recall: {rpb_perfect}/{total} ({100*rpb_perfect/total:.1f}%)")


# =====================================================================
# Section 7: Phase 5 — Scalability
# =====================================================================

def run_phase5(cfg: dict):
    """Measure throughput as corpus size grows."""
    print("\n" + "=" * 70)
    print("PHASE 5: Scalability")
    print("=" * 70)

    p5 = cfg["phase5_scalability"]
    k = p5["precision_bits"]
    dim = p5["dimension"]
    n_queries = p5["num_queries"]
    rng = np.random.default_rng(cfg["experiment"]["seed"] + 400)
    c = make_rpb_blinding(33)

    rows = []

    # Generate maximum-size corpus
    max_docs = max(p5["document_counts"] + p5.get("rpb_extra_counts", []))
    print(f"  Generating {max_docs} document embeddings (dim={dim})...")
    all_docs = random_unit_vectors(max_docs, dim, rng)
    queries = random_unit_vectors(n_queries, dim, rng)

    # Pre-encode all documents
    print("  Pre-encoding documents...")
    all_docs_rats = [encode_vector(d.tolist(), k) for d in tqdm(all_docs, desc="  Encoding")]

    # ── RPB scalability ──
    all_rpb_counts = sorted(set(p5["document_counts"] + p5.get("rpb_extra_counts", [])))
    print("  Pre-encrypting documents for RPB...")
    all_docs_rpb_cts = [rpb_encrypt_vector(dr, c) for dr in tqdm(all_docs_rats, desc="  RPB enc")]

    for n_docs in all_rpb_counts:
        docs_cts_subset = all_docs_rpb_cts[:n_docs]
        docs_f_subset = all_docs[:n_docs]

        query_times = []
        for qi in range(n_queries):
            q_f = queries[qi].tolist()
            q_rats = encode_vector(q_f, k)

            t0 = time.perf_counter()
            q_cts = rpb_encrypt_vector(q_rats, c)
            for d_cts in docs_cts_subset:
                dot_ct = rpb_dot_product(q_cts, d_cts)
                rpb.decrypt(dot_ct)
            elapsed = time.perf_counter() - t0
            query_times.append(elapsed)

        rows.append({
            "scheme": "rpb",
            "num_documents": n_docs,
            "dimension": dim,
            "precision_bits": k,
            "num_queries": n_queries,
            "mean_query_time_s": np.mean(query_times),
            "median_query_time_s": np.median(query_times),
            "p95_query_time_s": np.percentile(query_times, 95),
            "total_time_s": sum(query_times),
            "queries_per_second": n_queries / sum(query_times) if sum(query_times) > 0 else 0,
        })
        print(f"  RPB n={n_docs}: {np.mean(query_times):.2f}s/query")

    # ── MRPB scalability ──
    print("  Generating MRPB key...")
    _, sk = mrpb.keygen(key_bits=p5["mrpb_key_bits"], precision=0, seed=cfg["experiment"]["seed"])

    mrpb_counts = p5["document_counts"]
    print("  Pre-encrypting documents for MRPB...")
    max_mrpb_docs = max(mrpb_counts)
    mrpb_docs_cts = [mrpb_encrypt_vector(dr, sk)
                     for dr in tqdm(all_docs_rats[:max_mrpb_docs], desc="  MRPB enc")]

    for n_docs in mrpb_counts:
        docs_cts_subset = mrpb_docs_cts[:n_docs]

        query_times = []
        for qi in range(n_queries):
            q_f = queries[qi].tolist()
            q_rats = encode_vector(q_f, k)

            t0 = time.perf_counter()
            q_cts = mrpb_encrypt_vector(q_rats, sk)
            for d_cts in docs_cts_subset:
                dot_ct = mrpb_dot_product(q_cts, d_cts)
                mrpb.decrypt(dot_ct, sk)
            elapsed = time.perf_counter() - t0
            query_times.append(elapsed)

        rows.append({
            "scheme": "mrpb",
            "num_documents": n_docs,
            "dimension": dim,
            "precision_bits": k,
            "num_queries": n_queries,
            "mean_query_time_s": np.mean(query_times),
            "median_query_time_s": np.median(query_times),
            "p95_query_time_s": np.percentile(query_times, 95),
            "total_time_s": sum(query_times),
            "queries_per_second": n_queries / sum(query_times) if sum(query_times) > 0 else 0,
        })
        print(f"  MRPB n={n_docs}: {np.mean(query_times):.2f}s/query")

    # ── CKKS scalability ──
    if HAS_TENSEAL:
        try:
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60],
            )
            context.generate_galois_keys()
            context.global_scale = 2 ** 40

            ckks_counts = p5.get("ckks_counts", p5["document_counts"])
            for n_docs in ckks_counts:
                docs_subset = all_docs[:n_docs]

                query_times = []
                for qi in range(n_queries):
                    q_f = queries[qi].tolist()

                    t0 = time.perf_counter()
                    enc_q = ts.ckks_vector(context, q_f)
                    for d in docs_subset:
                        enc_d = ts.ckks_vector(context, d.tolist())
                        enc_q.dot(enc_d).decrypt()
                    elapsed = time.perf_counter() - t0
                    query_times.append(elapsed)

                rows.append({
                    "scheme": "ckks",
                    "num_documents": n_docs,
                    "dimension": dim,
                    "precision_bits": 40,  # scale bits
                    "num_queries": n_queries,
                    "mean_query_time_s": np.mean(query_times),
                    "median_query_time_s": np.median(query_times),
                    "p95_query_time_s": np.percentile(query_times, 95),
                    "total_time_s": sum(query_times),
                    "queries_per_second": n_queries / sum(query_times) if sum(query_times) > 0 else 0,
                })
                print(f"  CKKS n={n_docs}: {np.mean(query_times):.2f}s/query")
        except Exception as e:
            print(f"  [WARN] CKKS scalability failed: {e}")

    save_csv(OUTPUT_DIR / "phase5_scalability.csv", rows)
    print("  Phase 5 complete")


# =====================================================================
# Section 8: Phase 6 — Performance Profiling
# =====================================================================

def run_phase6(cfg: dict):
    """Per-operation timing breakdown."""
    print("\n" + "=" * 70)
    print("PHASE 6: Performance Profiling")
    print("=" * 70)

    p6 = cfg["phase6_performance"]
    rng = np.random.default_rng(cfg["experiment"]["seed"] + 500)
    c = make_rpb_blinding(11)
    rows = []

    # MRPB key
    print("  Generating MRPB key...")
    _, sk = mrpb.keygen(key_bits=p6["mrpb_key_bits"], precision=0, seed=cfg["experiment"]["seed"])

    for dim in p6["dimensions"]:
        for k in p6["precision_bits"]:
            print(f"  Profiling: dim={dim}, k={k}")

            for trial in range(p6["trials"]):
                q_f = random_unit_vectors(1, dim, rng)[0].tolist()
                d_f = random_unit_vectors(1, dim, rng)[0].tolist()

                # ── Encode ──
                t0 = time.perf_counter()
                q_rats = encode_vector(q_f, k)
                d_rats = encode_vector(d_f, k)
                t_encode = time.perf_counter() - t0

                # ── RPB encrypt ──
                t0 = time.perf_counter()
                q_cts_rpb = rpb_encrypt_vector(q_rats, c)
                t_rpb_enc_q = time.perf_counter() - t0

                t0 = time.perf_counter()
                d_cts_rpb = rpb_encrypt_vector(d_rats, c)
                t_rpb_enc_d = time.perf_counter() - t0

                # ── RPB dot product ──
                t0 = time.perf_counter()
                dot_ct_rpb = rpb_dot_product(q_cts_rpb, d_cts_rpb)
                t_rpb_dot = time.perf_counter() - t0

                # ── RPB decrypt ──
                t0 = time.perf_counter()
                rpb.decrypt(dot_ct_rpb)
                t_rpb_dec = time.perf_counter() - t0

                # ── MRPB encrypt ──
                t0 = time.perf_counter()
                q_cts_mrpb = mrpb_encrypt_vector(q_rats, sk)
                t_mrpb_enc_q = time.perf_counter() - t0

                t0 = time.perf_counter()
                d_cts_mrpb = mrpb_encrypt_vector(d_rats, sk)
                t_mrpb_enc_d = time.perf_counter() - t0

                # ── MRPB dot product ──
                t0 = time.perf_counter()
                dot_ct_mrpb = mrpb_dot_product(q_cts_mrpb, d_cts_mrpb)
                t_mrpb_dot = time.perf_counter() - t0

                # ── MRPB decrypt ──
                t0 = time.perf_counter()
                mrpb.decrypt(dot_ct_mrpb, sk)
                t_mrpb_dec = time.perf_counter() - t0

                # ── RPB ciphertext size ──
                rpb_ct_bytes = dot_ct_rpb.byte_size()

                # ── MRPB ciphertext size ──
                mrpb_ct_bytes = dot_ct_mrpb.byte_size()

                row = {
                    "dimension": dim,
                    "precision_bits": k,
                    "trial": trial,
                    "encode_s": t_encode,
                    "rpb_encrypt_query_s": t_rpb_enc_q,
                    "rpb_encrypt_doc_s": t_rpb_enc_d,
                    "rpb_dot_product_s": t_rpb_dot,
                    "rpb_decrypt_s": t_rpb_dec,
                    "rpb_total_s": t_rpb_enc_q + t_rpb_enc_d + t_rpb_dot + t_rpb_dec,
                    "rpb_ciphertext_bytes": rpb_ct_bytes,
                    "mrpb_encrypt_query_s": t_mrpb_enc_q,
                    "mrpb_encrypt_doc_s": t_mrpb_enc_d,
                    "mrpb_dot_product_s": t_mrpb_dot,
                    "mrpb_decrypt_s": t_mrpb_dec,
                    "mrpb_total_s": t_mrpb_enc_q + t_mrpb_enc_d + t_mrpb_dot + t_mrpb_dec,
                    "mrpb_ciphertext_bytes": mrpb_ct_bytes,
                }

                # ── CKKS timing ──
                if HAS_TENSEAL and p6.get("ckks_enabled", False):
                    try:
                        context = ts.context(
                            ts.SCHEME_TYPE.CKKS,
                            poly_modulus_degree=8192,
                            coeff_mod_bit_sizes=[60, 40, 40, 60],
                        )
                        context.generate_galois_keys()
                        context.global_scale = 2 ** 40

                        t0 = time.perf_counter()
                        enc_q = ts.ckks_vector(context, q_f)
                        t_ckks_enc_q = time.perf_counter() - t0

                        t0 = time.perf_counter()
                        enc_d = ts.ckks_vector(context, d_f)
                        t_ckks_enc_d = time.perf_counter() - t0

                        t0 = time.perf_counter()
                        enc_result = enc_q.dot(enc_d)
                        t_ckks_dot = time.perf_counter() - t0

                        t0 = time.perf_counter()
                        enc_result.decrypt()
                        t_ckks_dec = time.perf_counter() - t0

                        row["ckks_encrypt_query_s"] = t_ckks_enc_q
                        row["ckks_encrypt_doc_s"] = t_ckks_enc_d
                        row["ckks_dot_product_s"] = t_ckks_dot
                        row["ckks_decrypt_s"] = t_ckks_dec
                        row["ckks_total_s"] = t_ckks_enc_q + t_ckks_enc_d + t_ckks_dot + t_ckks_dec
                    except Exception:
                        pass

                rows.append(row)

    save_csv(OUTPUT_DIR / "phase6_performance.csv", rows)
    print("  Phase 6 complete")


# =====================================================================
# Section 9: Phase 7 — Real Embeddings (Optional)
# =====================================================================

def run_phase7(cfg: dict):
    """Retrieval benchmark with real sentence-transformer embeddings."""
    p3 = cfg["phase3_ranking"]
    re = p3.get("real_embeddings", {})
    if not re.get("enabled", False):
        print("\n[SKIP] Phase 7: real embeddings disabled in config")
        return
    if not HAS_ST:
        print("\n[SKIP] Phase 7: sentence-transformers not installed")
        return

    print("\n" + "=" * 70)
    print("PHASE 7: Real Embeddings Retrieval")
    print("=" * 70)

    model_name = re["model_name"]
    n_passages = re["num_passages"]
    n_queries = re["num_queries"]

    # ── Load or generate embeddings ──
    emb_cache = OUTPUT_DIR / "real_embeddings.npz"
    if emb_cache.exists():
        print(f"  Loading cached embeddings from {emb_cache}")
        data = np.load(emb_cache)
        doc_embs = data["doc_embeddings"]
        query_embs = data["query_embeddings"]
    else:
        print(f"  Loading model: {model_name}")
        model = SentenceTransformer(model_name)

        # Generate synthetic "documents" and "queries" using random sentences
        # In production, load MS MARCO or similar
        try:
            from datasets import load_dataset
            print(f"  Loading MS MARCO ({n_passages} passages)...")
            ds = load_dataset("ms_marco", "v1.1", split=f"train[:{n_passages}]")
            passages = [p for p in ds["passages"]]
            # Flatten passages (MS MARCO has nested structure)
            doc_texts = []
            for p_group in passages:
                for text in p_group["passage_text"]:
                    doc_texts.append(text)
                    if len(doc_texts) >= n_passages:
                        break
                if len(doc_texts) >= n_passages:
                    break
            query_texts = ds["query"][:n_queries]
        except Exception as e:
            print(f"  [WARN] Could not load MS MARCO: {e}")
            print("  Falling back to synthetic text embeddings")
            rng = np.random.default_rng(cfg["experiment"]["seed"])
            doc_texts = [f"Document about topic {i} with content {rng.integers(10000)}"
                        for i in range(n_passages)]
            query_texts = [f"Query about topic {i}" for i in range(n_queries)]

        print(f"  Encoding {len(doc_texts)} documents...")
        doc_embs = model.encode(doc_texts, show_progress_bar=True, batch_size=256,
                               normalize_embeddings=True)
        print(f"  Encoding {len(query_texts)} queries...")
        query_embs = model.encode(query_texts, show_progress_bar=True, batch_size=256,
                                  normalize_embeddings=True)

        np.savez(emb_cache, doc_embeddings=doc_embs, query_embeddings=query_embs)
        print(f"  Cached embeddings → {emb_cache}")

    dim = doc_embs.shape[1]
    print(f"  Embeddings: {doc_embs.shape[0]} docs × {dim}d, {query_embs.shape[0]} queries")

    # ── Retrieval benchmark ──
    c = make_rpb_blinding(42)
    rows = []

    for k in re["precision_bits"]:
        print(f"  Real embeddings retrieval: k={k}")

        # Pre-encrypt documents
        docs_rats = [encode_vector(d.tolist(), k) for d in doc_embs]
        docs_cts = [rpb_encrypt_vector(dr, c) for dr in tqdm(docs_rats, desc=f"  Enc docs k={k}")]

        for qi in tqdm(range(min(n_queries, len(query_embs))), desc=f"  Queries k={k}"):
            q_f = query_embs[qi].tolist()
            q_rats = encode_vector(q_f, k)
            q_cts = rpb_encrypt_vector(q_rats, c)

            true_scores = [true_dot(q_f, d.tolist()) for d in doc_embs]
            true_ranking = sorted(range(len(doc_embs)), key=lambda i: -true_scores[i])

            enc_scores = []
            for d_cts in docs_cts:
                dot_ct = rpb_dot_product(q_cts, d_cts)
                num, den = rpb.decrypt(dot_ct)
                enc_scores.append(float(num) / float(den))

            enc_ranking = sorted(range(len(doc_embs)), key=lambda i: -enc_scores[i])

            for top_k in re["top_k"]:
                if top_k <= len(doc_embs):
                    r_at_k = recall_at_k(true_ranking, enc_ranking, top_k)
                    rows.append({
                        "precision_bits": k,
                        "query_idx": qi,
                        "dimension": dim,
                        "num_documents": len(doc_embs),
                        "top_k": top_k,
                        "recall_at_k": r_at_k,
                        "dataset": "real_embeddings",
                    })

    save_csv(OUTPUT_DIR / "phase7_real_embeddings.csv", rows)
    if rows:
        perfect = sum(1 for r in rows if r["recall_at_k"] == 1.0)
        print(f"  Real embeddings: {perfect}/{len(rows)} perfect recall")


# =====================================================================
# Section 10: Figure Generation
# =====================================================================

def generate_figures(cfg: dict):
    """Generate all publication-ready figures from experiment data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("\n" + "=" * 70)
    print("FIGURE GENERATION")
    print("=" * 70)

    fig_dir = FIGURE_DIR
    dpi = cfg["figures"]["dpi"]
    fmts = cfg["figures"]["format"]

    def savefig(fig, name):
        for fmt in fmts:
            path = fig_dir / f"{name}.{fmt}"
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {name}")

    # ── Fig 11: Quantization error vs precision ──
    try:
        import pandas as pd
    except ImportError:
        print("  [WARN] pandas not installed, using manual CSV parsing")
        pd = None

    def load_csv_rows(filepath):
        if not filepath.exists():
            return []
        with open(filepath) as f:
            return list(csv.DictReader(f))

    # Fig 11: Error vs precision (analytical + empirical)
    rows_2a = load_csv_rows(OUTPUT_DIR / "phase2a_bound_validation.csv")
    if rows_2a:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        dims_to_plot = [4, 64, 384, 768]
        for dim in dims_to_plot:
            dim_rows = [r for r in rows_2a if int(r["dimension"]) == dim]
            if not dim_rows:
                continue
            ks = [int(r["precision_bits"]) for r in dim_rows]
            bounds = [float(r["analytical_bound"]) for r in dim_rows]
            observed = [float(r["max_observed_error"]) for r in dim_rows]
            ax.semilogy(ks, bounds, "o--", label=f"Bound (d={dim})", alpha=0.7)
            ax.semilogy(ks, observed, "s-", label=f"Observed (d={dim})", alpha=0.7)
        ax.set_xlabel("Precision bits (k)")
        ax.set_ylabel("Dot product error")
        ax.set_title("Quantization Error: Analytical Bound vs Observed")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        savefig(fig, "fig11_error_vs_precision")

    # Fig 12: Depth-invariant error
    rows_2b = load_csv_rows(OUTPUT_DIR / "phase2b_depth_invariance.csv")
    if rows_2b:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for k in [12, 16, 20]:
            k_rows = [r for r in rows_2b if int(r["precision_bits"]) == k]
            if not k_rows:
                continue
            depths = sorted(set(int(r["depth"]) for r in k_rows))
            max_errs = []
            for d in depths:
                d_rows = [r for r in k_rows if int(r["depth"]) == d]
                max_errs.append(max(float(r["error_vs_encoded"]) for r in d_rows))
            ax.plot(depths, max_errs, "o-", label=f"k={k}")
        ax.set_xlabel("Homomorphic depth (additional multiplications)")
        ax.set_ylabel("Max encryption error")
        ax.set_title("Zero Noise Accumulation: Error vs Homomorphic Depth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig(fig, "fig12_depth_invariant_error")

    # Fig 13: RPB/MRPB Recall@k (should be 1.0)
    rows_3 = load_csv_rows(OUTPUT_DIR / "phase3_rpb_ranking.csv")
    if rows_3:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        for idx, dim in enumerate([128, 384, 768]):
            ax = axes[idx]
            for k in [10, 12, 16, 20]:
                k_rows = [r for r in rows_3 if int(r["dimension"]) == dim
                          and int(r["precision_bits"]) == k]
                if not k_rows:
                    continue
                top_ks = sorted(set(int(r["top_k"]) for r in k_rows))
                recalls = []
                for tk in top_ks:
                    tk_rows = [r for r in k_rows if int(r["top_k"]) == tk]
                    recalls.append(np.mean([float(r["recall_at_k"]) for r in tk_rows]))
                ax.plot(top_ks, recalls, "o-", label=f"k={k}")
            ax.set_xlabel("Top-K")
            ax.set_title(f"Dimension {dim}")
            ax.set_ylim(0.9, 1.01)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Recall@K")
        fig.suptitle("RPB Ranking Preservation (1.0 = perfect)")
        savefig(fig, "fig13_rpb_recall")

    # Fig 14: CKKS vs RPB recall comparison (headline figure)
    rows_4 = load_csv_rows(OUTPUT_DIR / "phase4_ckks_comparison.csv")
    if rows_4:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left: recall comparison
        ax = axes[0]
        top_ks = sorted(set(int(r["top_k"]) for r in rows_4))
        ckks_recalls = []
        rpb_recalls = []
        for tk in top_ks:
            tk_rows = [r for r in rows_4 if int(r["top_k"]) == tk]
            ckks_recalls.append(np.mean([float(r["ckks_recall_at_k"]) for r in tk_rows]))
            rpb_recalls.append(np.mean([float(r["rpb_recall_at_k"]) for r in tk_rows]))
        ax.plot(top_ks, rpb_recalls, "o-", label="RPB (ours)", linewidth=2, color="green")
        ax.plot(top_ks, ckks_recalls, "s--", label="CKKS", linewidth=2, color="red")
        ax.set_xlabel("Top-K")
        ax.set_ylabel("Mean Recall@K")
        ax.set_title("Ranking Accuracy: RPB vs CKKS")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: error comparison
        ax = axes[1]
        ckks_errs = [float(r["ckks_max_error"]) for r in rows_4]
        rpb_errs = [float(r["rpb_max_error"]) for r in rows_4]
        ax.hist(ckks_errs, bins=50, alpha=0.6, label="CKKS noise", color="red")
        ax.hist(rpb_errs, bins=50, alpha=0.6, label="RPB quant. error", color="green")
        ax.set_xlabel("Max dot-product error")
        ax.set_ylabel("Count")
        ax.set_title("Error Distribution: CKKS Noise vs RPB Quantization")
        ax.legend()
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        savefig(fig, "fig14_ckks_vs_rpb_headline")

    # Fig 15: Performance breakdown by dimension
    rows_6 = load_csv_rows(OUTPUT_DIR / "phase6_performance.csv")
    if rows_6:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        k = 16  # fixed precision for clarity
        k_rows = [r for r in rows_6 if int(r["precision_bits"]) == k]

        if k_rows:
            dims = sorted(set(int(r["dimension"]) for r in k_rows))

            # RPB breakdown
            ax = axes[0]
            ops = ["rpb_encrypt_query_s", "rpb_encrypt_doc_s", "rpb_dot_product_s", "rpb_decrypt_s"]
            labels = ["Encrypt Q", "Encrypt D", "Dot Product", "Decrypt"]
            bottom = np.zeros(len(dims))
            for op, label in zip(ops, labels):
                vals = []
                for d in dims:
                    d_rows = [r for r in k_rows if int(r["dimension"]) == d]
                    vals.append(np.mean([float(r[op]) for r in d_rows]) * 1000)  # ms
                ax.bar(range(len(dims)), vals, bottom=bottom, label=label, alpha=0.8)
                bottom += np.array(vals)
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels(dims)
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Time (ms)")
            ax.set_title("RPB Per-Operation Timing (k=16)")
            ax.legend(fontsize=8)

            # MRPB breakdown
            ax = axes[1]
            ops = ["mrpb_encrypt_query_s", "mrpb_encrypt_doc_s", "mrpb_dot_product_s", "mrpb_decrypt_s"]
            bottom = np.zeros(len(dims))
            for op, label in zip(ops, labels):
                vals = []
                for d in dims:
                    d_rows = [r for r in k_rows if int(r["dimension"]) == d]
                    vals.append(np.mean([float(r[op]) for r in d_rows]) * 1000)
                ax.bar(range(len(dims)), vals, bottom=bottom, label=label, alpha=0.8)
                bottom += np.array(vals)
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels(dims)
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Time (ms)")
            ax.set_title("MRPB Per-Operation Timing (k=16)")
            ax.legend(fontsize=8)

        savefig(fig, "fig15_performance_breakdown")

    # Fig 16: Scalability
    rows_5 = load_csv_rows(OUTPUT_DIR / "phase5_scalability.csv")
    if rows_5:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for scheme in ["rpb", "mrpb", "ckks"]:
            s_rows = [r for r in rows_5 if r["scheme"] == scheme]
            if not s_rows:
                continue
            n_docs = [int(r["num_documents"]) for r in s_rows]
            times = [float(r["mean_query_time_s"]) for r in s_rows]
            style = {"rpb": ("o-", "green"), "mrpb": ("s-", "blue"), "ckks": ("^--", "red")}
            marker, color = style.get(scheme, ("o-", "gray"))
            ax.loglog(n_docs, times, marker, label=scheme.upper(), color=color, linewidth=2)
        ax.set_xlabel("Number of documents")
        ax.set_ylabel("Mean query time (s)")
        ax.set_title("Scalability: Query Time vs Corpus Size (dim=384, k=16)")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        savefig(fig, "fig16_scalability")

    # Fig 17: Ciphertext size comparison
    if rows_6:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        k = 16
        k_rows = [r for r in rows_6 if int(r["precision_bits"]) == k]
        if k_rows:
            dims = sorted(set(int(r["dimension"]) for r in k_rows))
            rpb_sizes = []
            mrpb_sizes = []
            for d in dims:
                d_rows = [r for r in k_rows if int(r["dimension"]) == d]
                rpb_sizes.append(np.mean([int(r["rpb_ciphertext_bytes"]) for r in d_rows]))
                mrpb_sizes.append(np.mean([int(r["mrpb_ciphertext_bytes"]) for r in d_rows]))
            ax.semilogy(dims, rpb_sizes, "o-", label="RPB (grows)", color="green", linewidth=2)
            ax.semilogy(dims, mrpb_sizes, "s-", label="MRPB (fixed)", color="blue", linewidth=2)
            ax.set_xlabel("Dimension")
            ax.set_ylabel("Dot-product ciphertext size (bytes)")
            ax.set_title("Ciphertext Size After Dot Product (k=16)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            savefig(fig, "fig17_ciphertext_size")

    print("\n  All figures generated")


# =====================================================================
# Section 11: Main
# =====================================================================

PHASES = {
    "1": ("Phase 1: Correctness", run_phase1),
    "2": ("Phase 2: Error Analysis", run_phase2),
    "3": ("Phase 3: Ranking Preservation", run_phase3),
    "4": ("Phase 4: CKKS Comparison", run_phase4),
    "5": ("Phase 5: Scalability", run_phase5),
    "6": ("Phase 6: Performance Profiling", run_phase6),
    "7": ("Phase 7: Real Embeddings", run_phase7),
}


def main():
    parser = argparse.ArgumentParser(description="EXP-004: RAG Benchmark")
    parser.add_argument("--phase", type=str, default=None,
                        help="Comma-separated phase numbers to run (e.g. '1,2,3')")
    parser.add_argument("--resume", action="store_true",
                        help="Skip phases already completed in checkpoint")
    parser.add_argument("--figures-only", action="store_true",
                        help="Only regenerate figures from existing data")
    args = parser.parse_args()

    cfg = load_config()
    ckpt = load_checkpoint()

    # Save frozen config
    import shutil
    frozen = OUTPUT_DIR / "config.yaml"
    if not frozen.exists():
        shutil.copy(CONFIG_FILE, frozen)

    # Environment snapshot
    env_info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "numpy_version": np.__version__,
        "gmpy2_version": gmpy2.version(),
        "has_tenseal": HAS_TENSEAL,
        "has_sentence_transformers": HAS_ST,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if HAS_TENSEAL:
        env_info["tenseal_version"] = ts.__version__
    save_json(OUTPUT_DIR / "environment.json", env_info)

    if args.figures_only:
        generate_figures(cfg)
        return

    # Determine which phases to run
    if args.phase:
        phase_ids = [p.strip() for p in args.phase.split(",")]
    else:
        phase_ids = list(PHASES.keys())

    print(f"\nEXP-004: Privacy-Preserving RAG Benchmark")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Phases: {', '.join(phase_ids)}")
    print(f"Resume: {args.resume}")
    print(f"TenSEAL: {HAS_TENSEAL}")
    print(f"Sentence-Transformers: {HAS_ST}")
    print()

    total_start = time.perf_counter()

    for pid in phase_ids:
        if pid not in PHASES:
            print(f"[WARN] Unknown phase: {pid}")
            continue

        name, func = PHASES[pid]

        if args.resume and is_phase_done(ckpt, pid):
            print(f"[SKIP] {name} (already completed)")
            continue

        print(f"\n{'─' * 70}")
        print(f"Starting {name}...")
        print(f"{'─' * 70}")

        t0 = time.perf_counter()
        try:
            func(cfg)
            duration = time.perf_counter() - t0
            mark_phase_complete(ckpt, pid, duration)
            print(f"\n  ✓ {name} completed in {duration:.1f}s ({duration/60:.1f}m)")
        except Exception as e:
            duration = time.perf_counter() - t0
            print(f"\n  ✗ {name} FAILED after {duration:.1f}s: {e}")
            traceback.print_exc()
            save_checkpoint(ckpt)
            # Continue to next phase instead of crashing
            continue

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'=' * 70}")
    print(f"All phases complete. Total time: {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")
    print(f"{'=' * 70}")

    # Generate figures at the end
    try:
        generate_figures(cfg)
    except Exception as e:
        print(f"[WARN] Figure generation failed: {e}")
        traceback.print_exc()

    # Final summary
    summary_path = OUTPUT_DIR / "final_summary.json"
    final = {
        "experiment_id": "EXP-004",
        "total_runtime_seconds": total_elapsed,
        "total_runtime_hours": total_elapsed / 3600,
        "phases_completed": [p["phase"] for p in ckpt.get("completed_phases", [])],
        "environment": env_info,
    }
    save_json(summary_path, final)
    print(f"\nFinal summary → {summary_path}")


if __name__ == "__main__":
    main()
