"""EXP-003: MRPB Addition Failure Depth Characterization.

Finds the maximum addition chain depth before rational reconstruction
fails, as a function of operand bit-size and modulus size.

Predicted relationship: max_depth ≈ key_bits / (2 * operand_bits)

Usage:
    python scripts/run_add_depth.py --config configs/exp_003_add_depth.yaml
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
import gmpy2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from herpb.schemes import mrpb
from herpb.utils.math_utils import reduce_fraction


def random_rational(bits, rng):
    """Generate a random positive rational p/q with `bits`-bit p and q."""
    p = rng.getrandbits(bits) | (1 << (bits - 1))  # ensure exactly `bits` bits
    q = rng.getrandbits(bits) | (1 << (bits - 1))
    return p, q


def run_addition_chain(operands, sk, t):
    """Encrypt operands as rationals and chain-add in MRPB.

    Each operand is (p, q) representing ratio p/q.
    Returns (correct, decrypt_ok, eval_time_ms).
    """
    # Encrypt all operands
    cts = []
    for p, q in operands:
        ct = mrpb.encrypt_shared(mpz(p), mpz(1), mpz(q), mpz(1), t=t, sk=sk)
        cts.append(ct)

    # Chain addition
    start = time.perf_counter_ns()
    result = cts[0]
    for ct in cts[1:]:
        result = mrpb.eval_add(result, ct)
    eval_ns = time.perf_counter_ns() - start

    # Decrypt
    try:
        num, den = mrpb.decrypt(result, sk)
        decrypt_ok = True
    except (ValueError, ZeroDivisionError):
        decrypt_ok = False
        num, den = mpz(0), mpz(1)

    # Plaintext reference
    expected = Fraction(0)
    for p, q in operands:
        expected += Fraction(p, q)

    if decrypt_ok:
        correct = (num == mpz(expected.numerator) and
                   den == mpz(expected.denominator))
    else:
        correct = False

    return correct, decrypt_ok, eval_ns / 1e6


def main():
    parser = argparse.ArgumentParser(description="MRPB Addition Depth")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    # CSV output
    csv_path = out_dir / "results_raw.csv"
    csv_fields = [
        "key_bits", "operand_bits", "depth", "trial",
        "correct", "decrypt_ok", "eval_time_ms",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    summary = {}

    for key_cfg in cfg["mrpb_keys"]:
        key_bits = key_cfg["key_bits"]
        label = key_cfg["label"]

        print(f"\n{'='*60}")
        print(f"Key: {label} ({key_bits} bits)")
        print(f"{'='*60}")

        print(f"  Generating keys...", end=" ", flush=True)
        t0 = time.perf_counter()
        _, sk = mrpb.keygen(key_bits=key_bits, seed=seed)
        print(f"done in {time.perf_counter()-t0:.2f}s")

        pt_bits = mrpb.plaintext_bits(sk)
        print(f"  Plaintext space: ~{pt_bits} bits")

        for ob in cfg["operand_bits"]:
            # Predicted max depth
            predicted_max = max(1, pt_bits // ob) if ob > 0 else 999

            results_by_depth = {}

            for depth in cfg["depths"]:
                rng = random.Random(seed + key_bits + ob * 1000 + depth)

                n_correct = 0
                n_decrypt_ok = 0
                eval_times = []

                for trial in range(cfg["trials"]):
                    operands = [random_rational(ob, rng) for _ in range(depth)]

                    correct, decrypt_ok, eval_ms = run_addition_chain(
                        operands, sk, t=cfg["precision"],
                    )

                    n_correct += int(correct)
                    n_decrypt_ok += int(decrypt_ok)
                    eval_times.append(eval_ms)

                    csv_writer.writerow({
                        "key_bits": key_bits,
                        "operand_bits": ob,
                        "depth": depth,
                        "trial": trial,
                        "correct": int(correct),
                        "decrypt_ok": int(decrypt_ok),
                        "eval_time_ms": f"{eval_ms:.6f}",
                    })

                csv_file.flush()

                correctness = n_correct / cfg["trials"]
                decrypt_rate = n_decrypt_ok / cfg["trials"]

                results_by_depth[depth] = {
                    "correctness": correctness,
                    "decrypt_ok_rate": decrypt_rate,
                    "eval_time_ms_mean": sum(eval_times) / len(eval_times),
                }

                status = "✓" if correctness == 1.0 else ("⚠" if decrypt_rate > 0 else "✗")
                print(f"  {ob:4d}-bit  depth={depth:4d}  "
                      f"correct={correctness:5.1%}  "
                      f"dec_ok={decrypt_rate:5.1%}  "
                      f"{status}")

                # Early stop: if zero successes, skip deeper depths
                if decrypt_rate == 0 and depth >= 4:
                    print(f"  {ob:4d}-bit  → stopping (0% at depth {depth})")
                    # Fill remaining depths with zeros
                    for remaining_depth in cfg["depths"]:
                        if remaining_depth > depth:
                            results_by_depth[remaining_depth] = {
                                "correctness": 0.0,
                                "decrypt_ok_rate": 0.0,
                                "eval_time_ms_mean": 0.0,
                            }
                    break

            # Find max successful depth
            max_ok_depth = 0
            for d in sorted(results_by_depth.keys()):
                if results_by_depth[d]["correctness"] == 1.0:
                    max_ok_depth = d

            skey = f"{key_bits}bit_op{ob}"
            summary[skey] = {
                "key_bits": key_bits,
                "operand_bits": ob,
                "predicted_max_depth": predicted_max,
                "actual_max_100pct_depth": max_ok_depth,
                "by_depth": results_by_depth,
            }

            # Save incrementally
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    csv_file.close()

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY: Max 100%-correct addition depth")
    print(f"{'='*60}")
    print(f"{'Operand':>10} | ", end="")
    for key_cfg in cfg["mrpb_keys"]:
        print(f"  {key_cfg['label']:>12} (pred/actual)", end="")
    print()
    print("-" * 70)

    for ob in cfg["operand_bits"]:
        print(f"  {ob:>6}-bit | ", end="")
        for key_cfg in cfg["mrpb_keys"]:
            skey = f"{key_cfg['key_bits']}bit_op{ob}"
            if skey in summary:
                s = summary[skey]
                print(f"  {s['predicted_max_depth']:>5} / {s['actual_max_100pct_depth']:<5}", end="")
            else:
                print(f"  {'?':>5} / {'?':<5}", end="")
        print()

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
