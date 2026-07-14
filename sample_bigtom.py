"""
sample_bigtom.py
----------------
Creates a balanced subset of the BigToM dataset by randomly sampling
a fixed number of examples per class (q_type).

Usage:
    python sample_bigtom.py \
        --input  bigtom_project_sub.json \
        --output bigtom_balanced_subset.json \
        --n      10 \
        --seed   42
"""

import argparse
import json
import random
from collections import defaultdict


def sample_balanced(data: list[dict], n: int, seed: int) -> list[dict]:
    """Return a list with exactly `n` items per q_type, sampled without replacement."""
    rng = random.Random(seed)

    # Group items by class
    by_class: dict[str, list[dict]] = defaultdict(list)
    for item in data:
        by_class[item["q_type"]].append(item)

    subset = []
    for q_type, items in sorted(by_class.items()):
        if len(items) < n:
            raise ValueError(
                f"Class '{q_type}' has only {len(items)} items, "
                f"but {n} were requested."
            )
        subset.extend(rng.sample(items, n))

    return subset


def main():
    parser = argparse.ArgumentParser(
        description="Sample a balanced subset from a BigToM-style JSON dataset."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input JSON file (list of items with a 'q_type' field)."
    )
    parser.add_argument(
        "--output", "-o",
        default="bigtom_balanced_subset.json",
        help="Path for the output JSON file (default: bigtom_balanced_subset.json)."
    )
    parser.add_argument(
        "--n", "-n",
        type=int,
        default=10,
        help="Number of samples to draw per class (default: 10)."
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    args = parser.parse_args()

    # Load
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from '{args.input}'.")

    # Sample
    subset = sample_balanced(data, n=args.n, seed=args.seed)

    # Summary
    from collections import Counter
    counts = Counter(item["q_type"] for item in subset)
    n_classes = len(counts)
    print(f"Sampled {len(subset)} items across {n_classes} classes ({args.n} per class).")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2, ensure_ascii=False)
    print(f"Saved balanced subset to '{args.output}'.")


if __name__ == "__main__":
    main()

