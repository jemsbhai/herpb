"""EXP-002: Homomorphic Depth Benchmark.

Tests how RPB and MRPB behave at increasing circuit depths for
multiplication chains, addition chains, and mixed circuits.

Key questions:
1. Is multiplication depth truly unlimited in both schemes?
2. At what depth does MRPB addition fail (sqrt(n) plaintext bound)?
3. How does evaluation time scale with depth?
4. How does RPB ciphertext size grow with depth?

Usage:
    python scripts/run_depth_benchmark.py --config configs/exp_002_depth.yaml
"""

import argparse
import csv
import json
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import yaml
from gmpy2 import mpz

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from herpb.schemes import rpb, mrpb
from herpb.utils.math_utils import reduce_fraction


def make_operands(pool, depth, rng):
    """Pick `depth` operands from the pool (with replacement)."""
    return [rng.choice(pool) for _ in range(depth)]


def plaintext_eval(operands, circuit_type):
    """Compute the expected result in exact arithmetic.

    Each operand is (a_num, a_den, b_num, b_den) encoding ratio a/b.
    Returns Fraction result.
    """
    ratios = [Fraction(o[0], o[1]) / Fraction(o[2], o[3]) for o in operands]

    if circuit_type == "multiply_chain":
        result = Fraction(1)
        for r in ratios:
            result *= r
        return result

    elif circuit_type == "add_chain":
        result = Fraction(0)
        for r in ratios:
            result += r
        return result

    elif circuit_type == "mixed_alt":
        # r0 + r1, then * r2, then + r3, then * r4, ...
        result = ratios[0]
        for i, r in enumerate(ratios[1:], 1):
            if i % 2 == 1:
                result += r
            else:
                result *= r
        return result

    else:
        raise ValueError(f"Unknown circuit type: {circuit_type}")


def rpb_eval(operands, circuit_type, c, t):
    """Encrypt operands and evaluate circuit homomorphically in RPB."""
    cts = []
    enc_time_total = 0.0

    for a_num, a_den, b_num, b_den in operands:
        a = rpb.Rational(mpz(a_num), mpz(a_den))
        b = rpb.Rational(mpz(b_num), mpz(b_den))

        start = time.perf_counter_ns()
        ct = rpb.shared_k_encrypt(a, b, t=t, c=mpz(c))
        enc_time_total += time.perf_counter_ns() - start
        cts.append(ct)

    # Evaluate circuit
    start = time.perf_counter_ns()

    if circuit_type == "multiply_chain":
        result = cts[0]
        for ct in cts[1:]:
            result = rpb.eval_multiply(result, ct)

    elif circuit_type == "add_chain":
        result = cts[0]
        for ct in cts[1:]:
            result = rpb.eval_add(result, ct)

    elif circuit_type == "mixed_alt":
        result = cts[0]
        for i, ct in enumerate(cts[1:], 1):
            if i % 2 == 1:
                result = rpb.eval_add(result, ct)
            else:
                result = rpb.eval_multiply(result, ct)

    eval_ns = time.perf_counter_ns() - start

    # Decrypt
    start = time.perf_counter_ns()
    num, den = rpb.decrypt(result)
    dec_ns = time.perf_counter_ns() - start

    return {
        "num": num, "den": den,
        "enc_time_ms": enc_time_total / 1e6,
        "eval_time_ms": eval_ns / 1e6,
        "dec_time_ms": dec_ns / 1e6,
        "ct_size_bytes": result.byte_size(),
    }


def mrpb_eval(operands, circuit_type, sk, t):
    """Encrypt operands and evaluate circuit homomorphically in MRPB."""
    cts = []
    enc_time_total = 0.0

    for a_num, a_den, b_num, b_den in operands:
        start = time.perf_counter_ns()
        ct = mrpb.encrypt_shared(
            mpz(a_num), mpz(a_den), mpz(b_num), mpz(b_den),
            t=t, sk=sk,
        )
        enc_time_total += time.perf_counter_ns() - start
        cts.append(ct)

    # Evaluate circuit
    start = time.perf_counter_ns()

    if circuit_type == "multiply_chain":
        result = cts[0]
        for ct in cts[1:]:
            result = mrpb.eval_multiply(result, ct)

    elif circuit_type == "add_chain":
        result = cts[0]
        for ct in cts[1:]:
            result = mrpb.eval_add(result, ct)

    elif circuit_type == "mixed_alt":
        result = cts[0]
        for i, ct in enumerate(cts[1:], 1):
            if i % 2 == 1:
                result = mrpb.eval_add(result, ct)
            else:
                result = mrpb.eval_multiply(result, ct)

    eval_ns = time.perf_counter_ns() - start

    # Decrypt
    start = time.perf_counter_ns()
    try:
        num, den = mrpb.decrypt(result, sk)
        dec_ns = time.perf_counter_ns() - start
        correct = True
    except (ValueError, ZeroDivisionError) as e:
        dec_ns = time.perf_counter_ns() - start
        num, den = mpz(0), mpz(1)
        correct = False

    return {
        "num": num, "den": den,
        "enc_time_ms": enc_time_total / 1e6,
        "eval_time_ms": eval_ns / 1e6,
        "dec_time_ms": dec_ns / 1e6,
        "ct_size_bytes": result.byte_size(),
        "decrypt_ok": correct,
    }


def main():
    parser = argparse.ArgumentParser(description="HERPB Depth Benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    depths = cfg["depths"]
    trials = cfg["trials_per_depth"]
    circuits = cfg["circuits"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = [tuple(o) for o in cfg["operand_pool"]]

    # Save frozen config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # Generate MRPB keys
    print("Generating MRPB keys...", end=" ", flush=True)
    keygen_start = time.perf_counter()
    _, sk = mrpb.keygen(key_bits=cfg["mrpb"]["key_bits"], seed=seed)
    keygen_s = time.perf_counter() - keygen_start
    print(f"done in {keygen_s:.2f}s (key_bits={cfg['mrpb']['key_bits']})")
    print(f"  Plaintext space: ~{mrpb.plaintext_bits(sk)} bits")

    # CSV output
    csv_path = out_dir / "results_raw.csv"
    csv_fields = [
        "circuit", "depth", "trial", "scheme",
        "enc_time_ms", "eval_time_ms", "dec_time_ms",
        "ct_size_bytes", "correct",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    summary = {}

    for circuit in circuits:
        print(f"\n{'='*60}")
        print(f"Circuit: {circuit}")
        print(f"{'='*60}")

        for depth in depths:
            rng = random.Random(seed + depth)

            rpb_correct = 0
            mrpb_correct = 0
            mrpb_decrypt_ok = 0

            rpb_enc_times = []
            rpb_eval_times = []
            rpb_dec_times = []
            rpb_sizes = []

            mrpb_enc_times = []
            mrpb_eval_times = []
            mrpb_dec_times = []

            for trial in range(trials):
                operands = make_operands(pool, depth, rng)
                expected = plaintext_eval(operands, circuit)
                exp_num = mpz(expected.numerator)
                exp_den = mpz(expected.denominator)

                # --- RPB ---
                rpb_result = rpb_eval(
                    operands, circuit,
                    c=cfg["rpb"]["c_value"], t=cfg["rpb"]["t_value"],
                )
                rpb_ok = (rpb_result["num"] == exp_num and
                          rpb_result["den"] == exp_den)
                rpb_correct += int(rpb_ok)
                rpb_enc_times.append(rpb_result["enc_time_ms"])
                rpb_eval_times.append(rpb_result["eval_time_ms"])
                rpb_dec_times.append(rpb_result["dec_time_ms"])
                rpb_sizes.append(rpb_result["ct_size_bytes"])

                csv_writer.writerow({
                    "circuit": circuit, "depth": depth, "trial": trial,
                    "scheme": "RPB",
                    "enc_time_ms": f"{rpb_result['enc_time_ms']:.6f}",
                    "eval_time_ms": f"{rpb_result['eval_time_ms']:.6f}",
                    "dec_time_ms": f"{rpb_result['dec_time_ms']:.6f}",
                    "ct_size_bytes": rpb_result["ct_size_bytes"],
                    "correct": int(rpb_ok),
                })

                # --- MRPB ---
                mrpb_result = mrpb_eval(
                    operands, circuit, sk,
                    t=cfg["mrpb"]["precision"],
                )
                mrpb_ok = (mrpb_result["decrypt_ok"] and
                           mrpb_result["num"] == exp_num and
                           mrpb_result["den"] == exp_den)
                mrpb_correct += int(mrpb_ok)
                mrpb_decrypt_ok += int(mrpb_result["decrypt_ok"])
                mrpb_enc_times.append(mrpb_result["enc_time_ms"])
                mrpb_eval_times.append(mrpb_result["eval_time_ms"])
                mrpb_dec_times.append(mrpb_result["dec_time_ms"])

                csv_writer.writerow({
                    "circuit": circuit, "depth": depth, "trial": trial,
                    "scheme": "MRPB",
                    "enc_time_ms": f"{mrpb_result['enc_time_ms']:.6f}",
                    "eval_time_ms": f"{mrpb_result['eval_time_ms']:.6f}",
                    "dec_time_ms": f"{mrpb_result['dec_time_ms']:.6f}",
                    "ct_size_bytes": mrpb_result["ct_size_bytes"],
                    "correct": int(mrpb_ok),
                })

            csv_file.flush()

            import numpy as np
            key = f"{circuit}_depth{depth}"
            summary[key] = {
                "circuit": circuit,
                "depth": depth,
                "rpb": {
                    "correctness": rpb_correct / trials,
                    "enc_time_ms_mean": float(np.mean(rpb_enc_times)),
                    "eval_time_ms_mean": float(np.mean(rpb_eval_times)),
                    "dec_time_ms_mean": float(np.mean(rpb_dec_times)),
                    "ct_size_bytes_mean": float(np.mean(rpb_sizes)),
                    "ct_size_bytes_max": int(np.max(rpb_sizes)),
                },
                "mrpb": {
                    "correctness": mrpb_correct / trials,
                    "decrypt_ok_rate": mrpb_decrypt_ok / trials,
                    "enc_time_ms_mean": float(np.mean(mrpb_enc_times)),
                    "eval_time_ms_mean": float(np.mean(mrpb_eval_times)),
                    "dec_time_ms_mean": float(np.mean(mrpb_dec_times)),
                },
            }

            rpb_pct = rpb_correct / trials * 100
            mrpb_pct = mrpb_correct / trials * 100
            mrpb_dec_pct = mrpb_decrypt_ok / trials * 100

            status = "✓" if mrpb_pct == 100 else ("⚠" if mrpb_dec_pct > 0 else "✗")

            print(f"  depth={depth:3d}  "
                  f"RPB={rpb_pct:5.1f}%  "
                  f"MRPB={mrpb_pct:5.1f}% (dec_ok={mrpb_dec_pct:5.1f}%)  "
                  f"RPB_ct={int(np.mean(rpb_sizes)):,}B  "
                  f"eval_rpb={np.mean(rpb_eval_times):.3f}ms  "
                  f"eval_mrpb={np.mean(mrpb_eval_times):.3f}ms  "
                  f"{status}")

            # Save summary incrementally
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    csv_file.close()

    print(f"\n{'='*60}")
    print(f"ALL DONE. Results in {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
