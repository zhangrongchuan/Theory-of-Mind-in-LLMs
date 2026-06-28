#!/usr/bin/env python3
"""
Script to extract sample IDs of incorrect entries from a JSONL file.

Usage:
    python find_incorrect_samples.py <input_jsonl_file> [output_file]

If no output file is specified, results are printed to stdout.
"""

import json
import sys
import argparse


def find_incorrect_samples(input_file, output_file=None):
    """
    Find all sample_ids where correct=0 in a JSONL file.

    Args:
        input_file: Path to the input JSONL file
        output_file: Optional path to write results (if None, prints to stdout)
    """
    incorrect_sample_ids = []
    total_entries = 0
    incorrect_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                    total_entries += 1

                    # Check if this entry is incorrect (correct: 0)
                    if entry.get('correct') == 0:
                        sample_id = entry.get('sample_id')
                        if sample_id is not None:
                            incorrect_sample_ids.append(sample_id)
                            incorrect_count += 1

                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse JSON on line {line_num}: {e}",
                          file=sys.stderr)
                    continue

    except FileNotFoundError:
        print(f"Error: File not found: {input_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Prepare output
    output_lines = [
        f"Total entries processed: {total_entries}",
        f"Incorrect entries found: {incorrect_count}",
        "",
        "Sample IDs of incorrect entries:",
    ]

    if incorrect_sample_ids:
        output_lines.extend(str(sid) for sid in incorrect_sample_ids)
    else:
        output_lines.append("(none)")

    output_text = "\n".join(output_lines)

    # Write output
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text + "\n")
            print(f"Results written to: {output_file}")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_text)

    return incorrect_sample_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract sample IDs of incorrect entries from a JSONL file."
    )
    parser.add_argument(
        "input_file",
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        help="Optional output file path (default: print to stdout)"
    )

    args = parser.parse_args()

    find_incorrect_samples(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
