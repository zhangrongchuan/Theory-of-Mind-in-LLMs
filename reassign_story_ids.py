"""
Reassign story_ids in a JSONL file to new random values.
The new story_ids are generated as random 32-character hex strings (MD5 format),
similar to how they were originally created in utils.py.
"""

import json
import argparse
import secrets
from pathlib import Path
from typing import Dict, List, Any


def generate_random_story_id() -> str:
    """
    Generate a random 32-character hex string (MD5 format).
    Uses secrets for cryptographically secure randomness.
    """
    return secrets.token_hex(16)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, data: List[Dict[str, Any]]) -> None:
    """Save a list of dictionaries to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def reassign_story_ids(input_path: str, output_path: str) -> None:
    """
    Read a JSONL file, assign new random story_ids to all entries,
    and save to the output file.
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load the data
    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} entries from {input_path}")

    # Track old to new story_id mappings
    story_id_mapping = {}

    # Reassign story_ids
    for entry in data:
        old_story_id = entry.get("story_id")
        new_story_id = generate_random_story_id()
        entry["story_id"] = new_story_id

        if old_story_id:
            story_id_mapping[old_story_id] = new_story_id

    # Save the modified data
    save_jsonl(output_path, data)
    print(f"Saved {len(data)} entries to {output_path}")

    # Print mapping summary
    print(f"\nReassigned {len(story_id_mapping)} unique story_ids:")
    for old_id, new_id in story_id_mapping.items():
        print(f"  {old_id} -> {new_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Reassign story_ids in a JSONL file to new random values."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to the output JSONL file (default: overwrites input file)"
    )

    args = parser.parse_args()

    # If no output specified, overwrite the input file
    output_path = args.output if args.output else args.input_file

    reassign_story_ids(args.input_file, output_path)
    print("\nDone!")


if __name__ == "__main__":
    main()
