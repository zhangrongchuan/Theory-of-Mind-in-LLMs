#!/usr/bin/env python3
"""
Switch entries between two JSONL files based on sample_id.

This script takes a source JSONL file and a target JSONL file, and replaces
entries in the target file with entries from the source file where the
sample_id matches. Only entries with sample_ids present in BOTH files are
switched. All other entries in the target file remain unchanged.

Usage:
    python switch_jsonl_entries.py <source_file> <target_file> [output_file]

Arguments:
    source_file: Path to the source JSONL file containing entries to switch in
    target_file: Path to the target JSONL file (entries to be updated)
    output_file: Optional path for the output file. If not provided, target_file is overwritten.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_jsonl(filepath):
    """Load a JSONL file and return a list of entries."""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num} in {filepath}: {e}")
    return entries


def save_jsonl(entries, filepath):
    """Save entries to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def switch_entries(source_file, target_file, output_file=None):
    """
    Switch entries from source_file into target_file based on sample_id.

    Only entries where sample_id exists in BOTH files are switched.
    All other entries in the target file remain unchanged.
    """
    source_path = Path(source_file)
    target_path = Path(target_file)

    if not source_path.exists():
        print(f"Error: Source file not found: {source_file}")
        sys.exit(1)

    if not target_path.exists():
        print(f"Error: Target file not found: {target_file}")
        sys.exit(1)

    print(f"Loading source file: {source_file}")
    source_entries = load_jsonl(source_file)

    print(f"Loading target file: {target_file}")
    target_entries = load_jsonl(target_file)

    # Create a lookup dictionary for source entries by sample_id
    source_by_id = {}
    for entry in source_entries:
        sample_id = entry.get('sample_id')
        if sample_id is not None:
            if sample_id in source_by_id:
                print(f"Warning: Duplicate sample_id {sample_id} in source file")
            source_by_id[sample_id] = entry

    # Create a set of sample_ids from source for quick lookup
    source_ids = set(source_by_id.keys())

    # Track statistics
    stats = {
        'source_total': len(source_entries),
        'target_total': len(target_entries),
        'switched': 0,
        'unchanged': 0,
        'source_only': 0,  # Entries in source but not in target
        'target_only': 0,  # Entries in target but not in source (unchanged)
    }

    # Calculate source-only entries
    target_ids = set(entry.get('sample_id') for entry in target_entries if entry.get('sample_id') is not None)
    source_only_ids = source_ids - target_ids
    stats['source_only'] = len(source_only_ids)

    if source_only_ids:
        print(f"\nNote: {len(source_only_ids)} entries from source are not in target (will be ignored):")
        for sid in sorted(source_only_ids)[:10]:  # Show first 10
            print(f"  - sample_id {sid}")
        if len(source_only_ids) > 10:
            print(f"  ... and {len(source_only_ids) - 10} more")

    # Process target entries: switch if sample_id exists in source
    new_target_entries = []
    switched_ids = []

    for entry in target_entries:
        sample_id = entry.get('sample_id')
        if sample_id is not None and sample_id in source_by_id:
            # Switch: use source entry instead of target entry
            new_target_entries.append(source_by_id[sample_id])
            switched_ids.append(sample_id)
            stats['switched'] += 1
        else:
            # Keep original entry
            new_target_entries.append(entry)
            stats['unchanged'] += 1

    stats['target_only'] = stats['unchanged']

    # Determine output path
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = target_path

    # Save the result
    print(f"\nSaving output to: {output_path}")
    save_jsonl(new_target_entries, output_path)

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Source file entries:       {stats['source_total']}")
    print(f"Target file entries:       {stats['target_total']}")
    print(f"Entries switched:          {stats['switched']}")
    print(f"Entries unchanged:         {stats['unchanged']}")
    print(f"Source-only (ignored):     {stats['source_only']}")
    print("=" * 50)

    if switched_ids:
        print(f"\nSwitched sample_ids: {sorted(switched_ids)}")

    return stats


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    source_file = sys.argv[1]
    target_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None

    switch_entries(source_file, target_file, output_file)


if __name__ == "__main__":
    main()
