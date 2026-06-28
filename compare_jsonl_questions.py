#!/usr/bin/env python3
"""
Compare two JSONL or JSON files and output sample_ids where 'question' values differ.

Usage:
    python compare_jsonl_questions.py <file1> <file2>
"""

import json
import sys
from pathlib import Path


def load_data(filepath):
    """Load data from JSONL or JSON file."""
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = {}

    # Check if it's a JSONL file (line-delimited JSON)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Try parsing as JSON array first
    try:
        json_array = json.loads(content)
        if isinstance(json_array, list):
            for item in json_array:
                if 'sample_id' in item:
                    data[item['sample_id']] = item
            return data
    except json.JSONDecodeError:
        pass

    # Try parsing as JSONL (one JSON object per line)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if 'sample_id' in item:
                    data[item['sample_id']] = item
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num} in {filepath}: {e}")

    return data


def compare_files(file1, file2):
    """Compare two files and return differences."""
    data1 = load_data(file1)
    data2 = load_data(file2)

    differences = []

    # Get all unique sample_ids from both files
    all_ids = set(data1.keys()) | set(data2.keys())

    for sample_id in sorted(all_ids):
        if sample_id not in data1:
            differences.append({
                'sample_id': sample_id,
                'issue': 'missing_in_file1',
                'file1_question': None,
                'file2_question': data2.get(sample_id, {}).get('question')
            })
        elif sample_id not in data2:
            differences.append({
                'sample_id': sample_id,
                'issue': 'missing_in_file2',
                'file1_question': data1.get(sample_id, {}).get('question'),
                'file2_question': None
            })
        else:
            # Both files have this sample_id, compare questions
            q1 = data1[sample_id].get('question')
            q2 = data2[sample_id].get('question')

            if q1 != q2:
                differences.append({
                    'sample_id': sample_id,
                    'issue': 'question_mismatch',
                    'file1_question': q1,
                    'file2_question': q2
                })

    return differences, len(data1), len(data2)


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_jsonl_questions.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    print(f"Comparing files:")
    print(f"  File 1: {file1}")
    print(f"  File 2: {file2}")
    print()

    try:
        differences, count1, count2 = compare_files(file1, file2)

        print(f"File 1 entries: {count1}")
        print(f"File 2 entries: {count2}")
        print(f"Differences found: {len(differences)}")
        print()

        if not differences:
            print("No differences found. All questions match for shared sample_ids.")
            return

        # Output sample_ids with differences
        print("=" * 80)
        print("SAMPLE_IDS WITH DIFFERENCES:")
        print("=" * 80)

        for diff in differences:
            sample_id = diff['sample_id']
            issue = diff['issue']

            if issue == 'missing_in_file1':
                print(f"  {sample_id}: Missing in file 1 (exists only in file 2)")
            elif issue == 'missing_in_file2':
                print(f"  {sample_id}: Missing in file 2 (exists only in file 1)")
            else:
                print(f"  {sample_id}: Question mismatch")
                if diff['file1_question']:
                    print(f"    File 1: {diff['file1_question'][:100]}...")
                else:
                    print(f"    File 1: (no question field)")
                if diff['file2_question']:
                    print(f"    File 2: {diff['file2_question'][:100]}...")
                else:
                    print(f"    File 2: (no question field)")

        # Also output just the IDs in a simple format
        print()
        print("=" * 80)
        print("SAMPLE_ID LIST (for easy copying):")
        print("=" * 80)
        print(", ".join(str(diff['sample_id']) for diff in differences))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
