#!/usr/bin/env python3
"""Compute SFT training allocation and curate the training set.

Formula (floor + weighted bonus):
1. Every type gets at least `floor` examples.
2. Remaining budget is distributed proportionally to (1 - accuracy/max_accuracy).
3. Targets are NOT capped by current supply — if supply is short, the script
   reports the deficit so you can run 3_generate_sft_responses.py to fill it.

Usage:
    python scripts/PGPS9K/3b_curate_sft_training_set.py --dry_run        # show allocation only
    python scripts/PGPS9K/3b_curate_sft_training_set.py                  # compute + curate
    python scripts/PGPS9K/3b_curate_sft_training_set.py --floor 25 --total 1500
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESPONSES = _PROJECT_ROOT / "data/PGPS9K/sft_responses_train.jsonl"
DEFAULT_ALLOCATION = _PROJECT_ROOT / "data/PGPS9K/sft_train_target_allocation.json"
DEFAULT_OUTPUT = _PROJECT_ROOT / "data/PGPS9K/sft_responses_train_1500.jsonl"

# Baseline validation accuracy per type (from eval_results_baseline_val_505_summary.json)
BASELINE_VAL_ACCURACY = {
    "Angle": 0.3889,
    "Angle Bisector of Triangle": 0.0,
    "Angle Relation in Triangle": 0.2857,
    "Arc Angle": 0.3571,
    "Circle Chord": 0.05,
    "Circumference and Area of Circle": 0.2941,
    "Geometric Mean": 0.0,
    "Inscribed Angle": 0.1,
    "Isosceles (Equilateral) Triangle": 0.2105,
    "Line Segment": 0.3333,
    "Median of Triangle": 0.1429,
    "Midsegment of Triangle ": 0.2667,
    "Parallel Lines": 0.3125,
    "Parallelogram": 0.25,
    "Perimeter and Area of Polygon": 0.3333,
    "Perimeter and Area of Quadrangle": 0.3704,
    "Perimeter and Area of Triangle": 0.4444,
    "Perpendicular Bisector of Triangle": 0.2,
    "Polygon Angle": 0.0,
    "Polygon Congruence": 0.25,
    "Polygon Similarity": 0.35,
    "Pythagorean Theorem": 0.25,
    "Rectangle": 0.375,
    "Rhombus and Square": 0.1364,
    "Secant Angle": 0.0588,
    "Secant Segment": 0.0667,
    "Similarity in Parallel Line": 0.1538,
    "Tangent": 0.0833,
    "Trapezoid and Kite": 0.2857,
    "Trigonometry": 0.2188,
}

BASELINE_VAL_COUNT = {
    "Angle": 18,
    "Angle Bisector of Triangle": 6,
    "Angle Relation in Triangle": 14,
    "Arc Angle": 14,
    "Circle Chord": 40,
    "Circumference and Area of Circle": 34,
    "Geometric Mean": 10,
    "Inscribed Angle": 20,
    "Isosceles (Equilateral) Triangle": 19,
    "Line Segment": 9,
    "Median of Triangle": 14,
    "Midsegment of Triangle ": 15,
    "Parallel Lines": 32,
    "Parallelogram": 24,
    "Perimeter and Area of Polygon": 6,
    "Perimeter and Area of Quadrangle": 27,
    "Perimeter and Area of Triangle": 9,
    "Perpendicular Bisector of Triangle": 5,
    "Polygon Angle": 11,
    "Polygon Congruence": 12,
    "Polygon Similarity": 20,
    "Pythagorean Theorem": 8,
    "Rectangle": 8,
    "Rhombus and Square": 22,
    "Secant Angle": 17,
    "Secant Segment": 15,
    "Similarity in Parallel Line": 13,
    "Tangent": 12,
    "Trapezoid and Kite": 28,
    "Trigonometry": 32,
}


def load_responses_by_type(path: Path) -> dict[str, list[dict]]:
    """Load all valid responses grouped by problem type."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            by_type[obj["type"]].append(obj)
    return dict(by_type)


def compute_allocation(
    floor: int,
    total: int,
) -> dict[str, dict]:
    """Compute ideal target allocation (floor + weighted bonus), ignoring supply."""
    max_accuracy = max(BASELINE_VAL_ACCURACY.values())
    types = sorted(BASELINE_VAL_ACCURACY.keys())

    # Step 1: assign floor to every type
    allocation = {}
    for t in types:
        allocation[t] = {
            "floor": floor,
            "bonus": 0,
            "target": 0,
            "val_accuracy": BASELINE_VAL_ACCURACY[t],
            "val_count": BASELINE_VAL_COUNT.get(t, 0),
        }

    floor_total = floor * len(types)
    remaining_budget = total - floor_total

    if remaining_budget < 0:
        print(f"WARNING: floor alone ({floor_total}) exceeds total ({total}). "
              f"Reducing floor allocations proportionally.")
        reduced = max(1, total // len(types))
        for t in types:
            allocation[t]["floor"] = reduced
        remaining_budget = total - reduced * len(types)

    # Step 2: compute bonus weights = (1 - accuracy/max_accuracy)
    bonus_weights = {}
    for t in types:
        bonus_weights[t] = 1.0 - BASELINE_VAL_ACCURACY[t] / max_accuracy

    weight_sum = sum(bonus_weights.values())
    if weight_sum == 0:
        # All types have identical accuracy — distribute uniformly
        weight_sum = len(types)
        for t in types:
            bonus_weights[t] = 1.0

    # Step 3: distribute remaining budget proportionally (no supply cap)
    # Use largest-remainder method for exact integer allocation
    raw_shares = {t: remaining_budget * bonus_weights[t] / weight_sum for t in types}
    int_shares = {t: int(v) for t, v in raw_shares.items()}
    remainders = {t: raw_shares[t] - int_shares[t] for t in types}

    allocated = sum(int_shares.values())
    leftover = remaining_budget - allocated

    # Give leftover 1 each to types with largest fractional remainders
    for t in sorted(remainders, key=lambda x: -remainders[x]):
        if leftover <= 0:
            break
        int_shares[t] += 1
        leftover -= 1

    for t in types:
        allocation[t]["bonus"] = int_shares[t]
        allocation[t]["target"] = allocation[t]["floor"] + allocation[t]["bonus"]

    return allocation


def print_allocation(allocation: dict[str, dict], available_per_type: dict[str, int]):
    """Pretty-print the allocation table with supply deficit info."""
    header = "{:<45} {:>6} {:>5} {:>5} {:>6} {:>7} {:>4}".format(
        "Type", "Target", "Floor", "Bonus", "Avail", "ValAcc", "ValN"
    )
    print(header)
    print("-" * 85)
    totals = {"target": 0, "floor": 0, "bonus": 0, "avail": 0}
    deficit_types = []
    for t, info in sorted(allocation.items(), key=lambda x: x[1]["val_accuracy"]):
        avail = available_per_type.get(t, 0)
        deficit = max(0, info["target"] - avail)
        deficit_marker = f"  (need {deficit} more)" if deficit > 0 else ""
        print("{:<45} {:>6} {:>5} {:>5} {:>6} {:>6.1%} {:>4}{}".format(
            t, info["target"], info["floor"], info["bonus"],
            avail, info["val_accuracy"], info["val_count"], deficit_marker
        ))
        totals["target"] += info["target"]
        totals["floor"] += info["floor"]
        totals["bonus"] += info["bonus"]
        totals["avail"] += avail
        if deficit > 0:
            deficit_types.append((t, info["target"], avail, deficit))
    print("-" * 85)
    print("{:<45} {:>6} {:>5} {:>5} {:>6}".format(
        "TOTAL", totals["target"], totals["floor"], totals["bonus"], totals["avail"]
    ))

    if deficit_types:
        total_deficit = sum(d for _, _, _, d in deficit_types)
        print(f"\n*** SUPPLY DEFICIT: {len(deficit_types)} types need {total_deficit} more responses ***")
        print("Run 3_generate_sft_responses.py to generate more, then re-run this script.")
        for t, target, avail, deficit in deficit_types:
            print(f"  {t}: need {target}, have {avail}, deficit {deficit}")

    return deficit_types


def curate_training_set(
    by_type: dict[str, list[dict]],
    allocation: dict[str, dict],
    seed: int,
) -> list[dict]:
    """Sample examples per type according to allocation targets.
    Uses min(target, available) per type."""
    rng = random.Random(seed)
    selected = []
    for t, info in sorted(allocation.items()):
        examples = by_type.get(t, [])
        n = min(info["target"], len(examples))
        if n >= len(examples):
            selected.extend(examples)
        else:
            selected.extend(rng.sample(examples, n))
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Compute SFT allocation and curate training set")
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES,
                        help="Path to full SFT responses JSONL")
    parser.add_argument("--total", type=int, default=1500,
                        help="Total training examples to select")
    parser.add_argument("--floor", type=int, default=20,
                        help="Minimum examples per type (hard floor)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only print allocation, don't write files")
    parser.add_argument("--allow_partial", action="store_true",
                        help="Curate even if some types have supply deficit (use what's available)")
    parser.add_argument("--allocation_out", type=Path, default=DEFAULT_ALLOCATION,
                        help="Output allocation JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output curated JSONL path")
    args = parser.parse_args()

    # Load responses
    print(f"Loading responses from {args.responses}...")
    by_type = load_responses_by_type(args.responses)
    available_per_type = {t: len(examples) for t, examples in by_type.items()}
    total_available = sum(available_per_type.values())
    print(f"Loaded {total_available} responses across {len(by_type)} types\n")

    # Compute allocation (ideal, no supply cap)
    allocation = compute_allocation(args.floor, args.total)
    deficit_types = print_allocation(allocation, available_per_type)

    if args.dry_run:
        print("\nDry run — no files written.")
        return

    if deficit_types and not args.allow_partial:
        print("\nAborting: supply deficit detected. Use --allow_partial to curate with available data,")
        print("or run 3_generate_sft_responses.py first to generate more responses.")
        sys.exit(1)

    # Save allocation JSON
    alloc_json = {
        "description": f"Target training examples per type ({args.total} total, floor={args.floor})",
        "formula": "floor + bonus where bonus ~ (1 - accuracy/max_accuracy)",
        "floor": args.floor,
        "total_target": args.total,
        "max_accuracy": max(BASELINE_VAL_ACCURACY.values()),
        "by_type": {},
    }
    for t in sorted(allocation.keys()):
        avail = available_per_type.get(t, 0)
        alloc_json["by_type"][t] = {
            **allocation[t],
            "available": avail,
            "deficit": max(0, allocation[t]["target"] - avail),
        }
    with open(args.allocation_out, "w") as f:
        json.dump(alloc_json, f, indent=2)
    print(f"\nAllocation saved to {args.allocation_out}")

    # Curate training set
    selected = curate_training_set(by_type, allocation, args.seed)
    with open(args.output, "w") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Curated {len(selected)} examples saved to {args.output}")

    # Verify distribution
    curated_counts = Counter(rec["type"] for rec in selected)
    print(f"\nVerification:")
    for t in sorted(allocation.keys()):
        target = allocation[t]["target"]
        actual = curated_counts.get(t, 0)
        avail = available_per_type.get(t, 0)
        if actual < target:
            print(f"  {t}: {actual}/{target} (limited by supply: {avail} available)")


if __name__ == "__main__":
    main()
