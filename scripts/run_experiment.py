"""EXP-001: RPB vs MRPB Comparative Benchmark.

Usage:
    python scripts/run_experiment.py --config configs/exp_001_benchmark.yaml

Measures encryption time, decryption time, ciphertext size, and correctness
for both RPB and MRPB across all input bit-lengths. Checkpoints after each
bit-length so crashes don't lose progress.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml
from gmpy2 import mpz
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from herpb.schemes import rpb, mrpb
from herpb.utils.data_loader import load_rational_pairs
from herpb.utils.math_utils import reduce_fraction, iabs, random_mpz_range, digits10


def run_rpb_trials(pairs, iterations, seed, bound, c_range, p_range):
    """Run RPB shared-k and distinct-k trials.

    Returns dict with keys: shared_results, distinct_results
    Each is a list of dicts with: time_ms, size_bytes, correct, t_used
    """
    rng = random.Random(seed)
    B = mpz(10) ** bound

    shared_results = []
    distinct_results = []

    for _iter in range(iterations):
        for a_num, a_den, b_num, b_den in pairs:
            a = rpb.Rational(a_num, a_den)
            b = rpb.Rational(b_num, b_den)

            # Expected ratio
            exp_num, exp_den = reduce_fraction(a_num * b_den, a_den * b_num)

            # Random parameters (matching C++ protocol)
            c_val = 0
            while c_val == 0:
                c_val = rng.randint(c_range[0], c_range[1])
            c = mpz(c_val)
            c_abs = iabs(c)

            P = rng.randint(p_range[0], p_range[1])

            # --- Shared-k ---
            t = rpb.max_t_shared(a, b, c_abs, B)
            t = min(t, P)

            start = time.perf_counter_ns()
            ct = rpb.shared_k_encrypt(a, b, t=t, c=c)
            enc_ns = time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            dec_num, dec_den = rpb.decrypt(ct)
            dec_ns = time.perf_counter_ns() - start

            correct = (dec_num == exp_num and dec_den == exp_den)

            shared_results.append({
                "enc_time_ms": enc_ns / 1e6,
                "dec_time_ms": dec_ns / 1e6,
                "ct_size_bytes": ct.byte_size(),
                "correct": correct,
                "t_used": t,
            })

            # --- Distinct-k ---
            Pa = rng.randint(p_range[0], p_range[1])
            Pb = rng.randint(p_range[0], p_range[1])
            T = rpb.max_T_distinct(a, b, c_abs, B)
            ta, tb = rpb.split_T(T, Pa, Pb)

            start = time.perf_counter_ns()
            ct = rpb.distinct_k_encrypt(a, b, ta=ta, tb=tb, c=c)
            enc_ns = time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            dec_num, dec_den = rpb.decrypt(ct)
            dec_ns = time.perf_counter_ns() - start

            correct = (dec_num == exp_num and dec_den == exp_den)

            distinct_results.append({
                "enc_time_ms": enc_ns / 1e6,
                "dec_time_ms": dec_ns / 1e6,
                "ct_size_bytes": ct.byte_size(),
                "correct": correct,
                "ta_used": ta,
                "tb_used": tb,
            })

    return {"shared": shared_results, "distinct": distinct_results}


def run_mrpb_trials(pairs, iterations, seed, sk, precision):
    """Run MRPB shared-k and distinct-k trials.

    Returns dict with keys: shared_results, distinct_results
    """
    shared_results = []
    distinct_results = []

    for _iter in range(iterations):
        for a_num, a_den, b_num, b_den in pairs:
            a_num, a_den = mpz(a_num), mpz(a_den)
            b_num, b_den = mpz(b_num), mpz(b_den)

            # Expected ratio
            exp_num, exp_den = reduce_fraction(a_num * b_den, a_den * b_num)

            t = precision

            # --- Shared-k ---
            start = time.perf_counter_ns()
            ct = mrpb.encrypt_shared(a_num, a_den, b_num, b_den, t=t, sk=sk)
            enc_ns = time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            try:
                dec_num, dec_den = mrpb.decrypt(ct, sk)
                dec_ns = time.perf_counter_ns() - start
                correct = (dec_num == exp_num and dec_den == exp_den)
            except (ValueError, ZeroDivisionError):
                dec_ns = time.perf_counter_ns() - start
                correct = False

            shared_results.append({
                "enc_time_ms": enc_ns / 1e6,
                "dec_time_ms": dec_ns / 1e6,
                "ct_size_bytes": ct.byte_size(),
                "correct": correct,
                "t_used": t,
            })

            # --- Distinct-k ---
            ta, tb = t, t

            start = time.perf_counter_ns()
            ct = mrpb.encrypt_distinct(
                a_num, a_den, b_num, b_den, ta=ta, tb=tb, sk=sk,
            )
            enc_ns = time.perf_counter_ns() - start

            start = time.perf_counter_ns()
            try:
                dec_num, dec_den = mrpb.decrypt(ct, sk)
                dec_ns = time.perf_counter_ns() - start
                correct = (dec_num == exp_num and dec_den == exp_den)
            except (ValueError, ZeroDivisionError):
                dec_ns = time.perf_counter_ns() - start
                correct = False

            distinct_results.append({
                "enc_time_ms": enc_ns / 1e6,
                "dec_time_ms": dec_ns / 1e6,
                "ct_size_bytes": ct.byte_size(),
                "correct": correct,
                "ta_used": ta,
                "tb_used": tb,
            })

    return {"shared": shared_results, "distinct": distinct_results}


def compute_stats(results):
    """Compute summary statistics for a list of trial dicts."""
    import numpy as np

    enc_times = np.array([r["enc_time_ms"] for r in results])
    dec_times = np.array([r["dec_time_ms"] for r in results])
    sizes = np.array([r["ct_size_bytes"] for r in results])
    correct = sum(1 for r in results if r["correct"])
    total = len(results)

    return {
        "count": total,
        "correctness": correct / total if total > 0 else 0.0,
        "enc_time_ms_mean": float(np.mean(enc_times)),
        "enc_time_ms_std": float(np.std(enc_times)),
        "enc_time_ms_median": float(np.median(enc_times)),
        "dec_time_ms_mean": float(np.mean(dec_times)),
        "dec_time_ms_std": float(np.std(dec_times)),
        "ct_size_bytes_mean": float(np.mean(sizes)),
        "ct_size_bytes_std": float(np.std(sizes)),
    }


def estimate_mrpb_key_bits(pairs):
    """Estimate minimum MRPB key_bits so rational reconstruction succeeds.

    The reduced ratio num/den must satisfy |num|, den < sqrt(n).
    So bits(n) >= 2 * max(bits(ratio_num), bits(ratio_den)) + margin.
    """
    max_ratio_bits = 0
    for a_num, a_den, b_num, b_den in pairs:
        num, den = reduce_fraction(mpz(a_num) * mpz(b_den), mpz(a_den) * mpz(b_num))
        nb = max(int(iabs(num)).bit_length(), int(den).bit_length())
        max_ratio_bits = max(max_ratio_bits, nb)

    # Need sqrt(n) > max_ratio, so bits(n) > 2 * max_ratio_bits
    # Add generous margin for scaling factor k
    required = 2 * max_ratio_bits + 256
    # Round up to nearest 256
    required = ((required + 255) // 256) * 256
    return max(2048, required)


def main():
    parser = argparse.ArgumentParser(description="HERPB Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    iterations = cfg["iterations"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save frozen config copy
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Save environment snapshot
    env_info = {
        "python": sys.version,
        "platform": sys.platform,
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        import gmpy2
        env_info["gmpy2_version"] = gmpy2.version()
    except Exception:
        pass
    with open(out_dir / "environment.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # Checkpoint file
    checkpoint_path = out_dir / "checkpoint.json"
    completed = set()
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            completed = set(json.load(f).get("completed", []))
        print(f"Resuming: {len(completed)} bit-lengths already done.")

    # CSV output
    csv_path = out_dir / "results_raw.csv"
    csv_exists = csv_path.exists() and len(completed) > 0
    csv_fields = [
        "input_bits", "scheme", "variant", "trial",
        "enc_time_ms", "dec_time_ms", "ct_size_bytes", "correct",
    ]
    csv_file = open(csv_path, "a" if csv_exists else "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not csv_exists:
        csv_writer.writeheader()

    # Summary accumulator
    summary_path = out_dir / "summary.json"
    summary = {}
    if summary_path.exists() and len(completed) > 0:
        with open(summary_path) as f:
            summary = json.load(f)

    data_dir = Path("data/raw")

    for inp in cfg["input_files"]:
        name = inp["name"]
        if name in completed:
            print(f"Skipping {name} (already done)")
            continue

        filepath = data_dir / inp["file"]
        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping")
            continue

        pairs = load_rational_pairs(filepath)
        print(f"\n{'='*60}")
        print(f"Running {name}: {len(pairs)} pairs × {iterations} iterations")
        print(f"{'='*60}")

        # --- RPB ---
        print(f"  RPB trials...")
        rpb_results = run_rpb_trials(
            pairs, iterations, seed,
            bound=cfg["rpb"]["bound_exponent"],
            c_range=cfg["rpb"]["c_range"],
            p_range=cfg["rpb"]["precision_range"],
        )

        rpb_shared_stats = compute_stats(rpb_results["shared"])
        rpb_distinct_stats = compute_stats(rpb_results["distinct"])

        print(f"    Shared-k:  enc={rpb_shared_stats['enc_time_ms_mean']:.4f}ms  "
              f"correct={rpb_shared_stats['correctness']:.1%}")
        print(f"    Distinct-k: enc={rpb_distinct_stats['enc_time_ms_mean']:.4f}ms  "
              f"correct={rpb_distinct_stats['correctness']:.1%}")

        # --- MRPB ---
        # Auto-scale key_bits based on input ratio size
        min_key_bits = estimate_mrpb_key_bits(pairs)
        requested_key_bits = cfg["mrpb"]["key_bits"]
        key_bits = max(requested_key_bits, min_key_bits)

        print(f"  MRPB trials (key_bits={key_bits}, "
              f"min_required={min_key_bits})...")
        print(f"    Generating RSA keys...", end=" ", flush=True)
        keygen_start = time.perf_counter()
        _, sk = mrpb.keygen(key_bits=key_bits, seed=seed)
        keygen_s = time.perf_counter() - keygen_start
        print(f"done in {keygen_s:.2f}s")

        mrpb_results = run_mrpb_trials(
            pairs, iterations, seed, sk,
            precision=cfg["mrpb"]["precision"],
        )

        mrpb_shared_stats = compute_stats(mrpb_results["shared"])
        mrpb_distinct_stats = compute_stats(mrpb_results["distinct"])

        print(f"    Shared-k:  enc={mrpb_shared_stats['enc_time_ms_mean']:.4f}ms  "
              f"correct={mrpb_shared_stats['correctness']:.1%}")
        print(f"    Distinct-k: enc={mrpb_distinct_stats['enc_time_ms_mean']:.4f}ms  "
              f"correct={mrpb_distinct_stats['correctness']:.1%}")

        # Write raw CSV rows
        for scheme, variant, results in [
            ("RPB", "shared", rpb_results["shared"]),
            ("RPB", "distinct", rpb_results["distinct"]),
            ("MRPB", "shared", mrpb_results["shared"]),
            ("MRPB", "distinct", mrpb_results["distinct"]),
        ]:
            for i, r in enumerate(results):
                csv_writer.writerow({
                    "input_bits": name,
                    "scheme": scheme,
                    "variant": variant,
                    "trial": i,
                    "enc_time_ms": f"{r['enc_time_ms']:.6f}",
                    "dec_time_ms": f"{r['dec_time_ms']:.6f}",
                    "ct_size_bytes": r["ct_size_bytes"],
                    "correct": int(r["correct"]),
                })
        csv_file.flush()

        # Save summary for this bit-length
        summary[name] = {
            "rpb_shared": rpb_shared_stats,
            "rpb_distinct": rpb_distinct_stats,
            "mrpb_shared": mrpb_shared_stats,
            "mrpb_distinct": mrpb_distinct_stats,
            "mrpb_key_bits_used": key_bits,
            "mrpb_keygen_time_s": keygen_s,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Checkpoint
        completed.add(name)
        with open(checkpoint_path, "w") as f:
            json.dump({"completed": list(completed)}, f)

        print(f"  ✓ {name} checkpointed.")

    csv_file.close()

    print(f"\n{'='*60}")
    print(f"ALL DONE. Results in {out_dir}/")
    print(f"  Raw data:  {csv_path}")
    print(f"  Summary:   {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
