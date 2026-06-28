#!/usr/bin/env python3
"""
Script to filter a JSON file based on sample IDs from a text file.

Reads a list of sample IDs from an input text file and creates a new JSON file
containing only the entries from the source JSON that match those sample IDs.

Usage:
    python filter_by_sample_ids.py <sample_ids_file> <source_json> [output_json]

Arguments:
    sample_ids_file: Path to text file containing sample IDs (one per line)
    source_json: Path to the source JSON file (array of objects)
    output_json: Output JSON file path (default: hitom_project_sub_incorrect.json)
"""

import json
import sys
import argparse


def parse_sample_ids(sample_ids_file):
    """
    Parse sample IDs from a text file.
    Handles the output format from find_incorrect_samples.py (skips header lines).

    Args:
        sample_ids_file: Path to the text file

    Returns:
        Set of sample IDs as integers
    """
    sample_ids = set()

    try:
        with open(sample_ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip empty lines and non-numeric lines (headers, etc.)
                if not line or not line.isdigit():
                    continue

                try:
                    sample_ids.add(int(line))
                except ValueError:
                    continue

    except FileNotFoundError:
        print(f"Error: Sample IDs file not found: {sample_ids_file}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading sample IDs file: {e}", file=sys.stderr)
        sys.exit(1)

    return sample_ids


def filter_json_by_sample_ids(source_json_file, sample_ids, output_json_file):
    """
    Filter a JSON file to only include entries with matching sample IDs.

    Args:
        source_json_file: Path to the source JSON file
        sample_ids: Set of sample IDs to include
        output_json_file: Path for the output JSON file
    """
    try:
        with open(source_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source JSON file not found: {source_json_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in source file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading source JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    # Check if data is a list
    if not isinstance(data, list):
        print("Error: Source JSON file must contain an array (list) of objects", file=sys.stderr)
        sys.exit(1)

    # Filter entries that have matching sample_ids
    filtered_data = []
    matched_ids = set()

    for entry in data:
        if not isinstance(entry, dict):
            continue

        entry_sample_id = entry.get('sample_id')
        if entry_sample_id is not None and entry_sample_id in sample_ids:
            filtered_data.append(entry)
            matched_ids.add(entry_sample_id)

    # Write output JSON file
    try:
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(1)

    # Summary
    unmatched_ids = sample_ids - matched_ids
    print(f"Sample IDs from input file: {len(sample_ids)}")
    print(f"Entries in source JSON: {len(data)}")
    print(f"Filtered entries written: {len(filtered_data)}")
    print(f"Output file: {output_json_file}")

    if unmatched_ids:
        print(f"\nWarning: {len(unmatched_ids)} sample ID(s) not found in source JSON:")
        for sid in sorted(unmatched_ids):
            print(f"  - {sid}")

    return len(filtered_data)


def main():
    parser = argparse.ArgumentParser(
        description="Filter a JSON file based on sample IDs from a text file."
    )
    parser.add_argument(
        "sample_ids_file",
        help="Path to text file containing sample IDs (one per line)"
    )
    parser.add_argument(
        "source_json",
        help="Path to the source JSON file (array of objects)"
    )
    parser.add_argument(
        "-o", "--output",
        dest="output_file",
        default="hitom_project_sub_incorrect.json",
        help="Output JSON file path (default: hitom_project_sub_incorrect.json)"
    )

    args = parser.parse_args()

    # Parse sample IDs from text file
    sample_ids = parse_sample_ids(args.sample_ids_file)

    if not sample_ids:
        print("Error: No valid sample IDs found in input file", file=sys.stderr)
        sys.exit(1)

    # Filter and create output JSON
    filter_json_by_sample_ids(args.source_json, sample_ids, args.output_file)


if __name__ == "__main__":
    main()
