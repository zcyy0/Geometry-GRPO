"""Re-split PGPS9K data so all questions sharing an image stay in the same split.

Ensures no image-level leakage between train/val/test while maintaining
similar problem type ratios across splits.

Strategy:
  1. Merge all problems from train/val/test
  2. Group problems by image (diagram filename)
  3. For each image group, assign a primary type (most common type in group)
  4. Within each type, shuffle image groups and split ~83%/6%/11% (train/val/test)
     to approximate the original 7523/498/1000 ratio
  5. Write new train.json, val.json, test.json

Usage:
    python scripts/PGPS9K/0_split_by_image.py
    python scripts/PGPS9K/0_split_by_image.py --output_dir data/PGPS9K/PGPS9K_data/PGPS9K
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Re-split PGPS9K by image to prevent leakage"
    )
    parser.add_argument(
        "--data_dir",
        default="data/PGPS9K/PGPS9K_data/PGPS9K",
        help="Directory containing train.json, val.json, test.json",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: same as data_dir, overwrites originals)",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.055,
        help="Fraction for val split (default: 0.055 ≈ 498/9021)",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.111,
        help="Fraction for test split (default: 0.111 ≈ 1000/9021)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent.parent
    data_dir = base / args.data_dir
    output_dir = base / args.output_dir if args.output_dir else data_dir

    # 1. Load and merge all splits
    all_problems = {}
    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.json"
        if not path.exists():
            print(f"Skipping {path} (not found)")
            continue
        with open(path) as f:
            data = json.load(f)
        all_problems.update(data)
        print(f"Loaded {len(data)} problems from {split}")
    print(f"Total: {len(all_problems)} problems")

    # 2. Group by image
    image_groups = defaultdict(list)
    for pid, prob in all_problems.items():
        image_groups[prob["diagram"]].append((pid, prob))
    print(f"Unique images: {len(image_groups)}")

    # 3. Assign primary type to each image group
    type_to_groups = defaultdict(list)
    for diagram, problems in image_groups.items():
        # Most common type in this group
        type_counts = defaultdict(int)
        for _, prob in problems:
            type_counts[prob["type"]] += 1
        primary_type = max(type_counts, key=type_counts.get)
        type_to_groups[primary_type].append((diagram, problems))

    print(f"Problem types: {len(type_to_groups)}")

    # 4. Stratified split by type
    random.seed(args.seed)
    train_problems = {}
    val_problems = {}
    test_problems = {}

    for ptype, groups in sorted(type_to_groups.items()):
        random.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * args.test_ratio))
        n_val = max(1, round(n * args.val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            # Too few groups: put all in train
            n_train = n
            n_val = 0
            n_test = 0

        splits = {
            "test": groups[:n_test],
            "val": groups[n_test:n_test + n_val],
            "train": groups[n_test + n_val:],
        }

        for split_name, split_groups in splits.items():
            target = {"train": train_problems, "val": val_problems, "test": test_problems}[split_name]
            for _, problems in split_groups:
                for pid, prob in problems:
                    target[pid] = prob

    print(f"\nNew splits:")
    print(f"  train: {len(train_problems)} problems")
    print(f"  val:   {len(val_problems)} problems")
    print(f"  test:  {len(test_problems)} problems")

    # Verify no image overlap
    train_imgs = {p["diagram"] for p in train_problems.values()}
    val_imgs = {p["diagram"] for p in val_problems.values()}
    test_imgs = {p["diagram"] for p in test_problems.values()}
    assert not (train_imgs & val_imgs), "train/val image overlap!"
    assert not (train_imgs & test_imgs), "train/test image overlap!"
    assert not (val_imgs & test_imgs), "val/test image overlap!"
    print("  No image overlap between splits.")

    # Print type distribution comparison
    print(f"\n{'Type':<40} {'Train':>6} {'Val':>6} {'Test':>6}")
    print(f"{'-'*40} {'-'*6} {'-'*6} {'-'*6}")
    all_types = sorted(set(
        [p["type"] for p in train_problems.values()] +
        [p["type"] for p in val_problems.values()] +
        [p["type"] for p in test_problems.values()]
    ))
    for t in all_types:
        tr = sum(1 for p in train_problems.values() if p["type"] == t)
        va = sum(1 for p in val_problems.values() if p["type"] == t)
        te = sum(1 for p in test_problems.values() if p["type"] == t)
        total = tr + va + te
        print(f"{t:<40} {tr:>5} {va:>5} {te:>5}  "
              f"({tr/total:.0%}/{va/total:.0%}/{te/total:.0%})")

    # 5. Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in [("train", train_problems), ("val", val_problems), ("test", test_problems)]:
        out_path = output_dir / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {len(split_data)} problems to {out_path}")


if __name__ == "__main__":
    main()
