#!/usr/bin/env python3
"""
Fix missing 'method' field in JSONL experiment result files.

Usage:
    python fix_missing_method.py <input_file> [--method METHOD_NAME] [--output OUTPUT_FILE]

Examples:
    # Fix in-place (creates backup first):
    python fix_missing_method.py experiment_results/hitom_cotp_results_soo_Qwen_Qwen3-1.7B.jsonl --method SoO

    # Write to new file:
    python fix_missing_method.py experiment_results/hitom_cotp_results_soo_Qwen_Qwen3-1.7B.jsonl --method SoO --output fixed_results.jsonl
"""

import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime


def fix_missing_method(input_file: str, method_name: str, output_file: str = None, backup: bool = True, reorder: bool = False):
    """
    Add missing 'method' field or fix its position in JSONL records.

    Args:
        input_file: Path to the input JSONL file
        method_name: The method name to add (e.g., "SoO")
        output_file: Path for output file (if None, overwrites input file)
        backup: Whether to create a backup of the original file
        reorder: Also fix position of existing method fields (default: False)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Determine output path
    if output_file:
        output_path = Path(output_file)
    else:
        output_path = input_path

    # Create backup if requested and we're overwriting the original
    if backup and output_path == input_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = input_path.parent / f"{input_path.stem}_backup_{timestamp}{input_path.suffix}"
        shutil.copy2(input_path, backup_path)
        print(f"Created backup: {backup_path}")

    # Process the file
    fixed_count = 0
    total_count = 0
    records = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                records.append(line)  # Keep empty lines as-is
                continue

            try:
                record = json.loads(line)
                total_count += 1

                # Check if method field is missing
                if 'method' not in record:
                    # Reconstruct record with method field at position 3 (after story_id, before question_order)
                    ordered_record = {}
                    for key in record:
                        if key == 'question_order' and 'method' not in ordered_record:
                            # Insert method right before question_order
                            ordered_record['method'] = method_name
                        ordered_record[key] = record[key]
                    # If question_order wasn't found, add method at the end
                    if 'method' not in ordered_record:
                        ordered_record['method'] = method_name

                    record = ordered_record
                    fixed_count += 1
                    if fixed_count <= 5:  # Show first few fixes
                        print(f"  Line {line_num}: Added method='{method_name}' to sample_id={record.get('sample_id', 'N/A')}")

                records.append(record)

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                records.append(line)  # Keep original line on error

    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            if isinstance(record, dict):
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            else:
                f.write(record + '\n')

    print(f"\nSummary:")
    print(f"  Total records processed: {total_count}")
    print(f"  Records fixed: {fixed_count}")
    print(f"  Output written to: {output_path}")

    return fixed_count


def main():
    parser = argparse.ArgumentParser(
        description="Fix missing 'method' field in JSONL experiment result files"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input JSONL file"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="SoO",
        help="Method name to add (default: SoO)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: overwrite input file)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create a backup of the original file"
    )
    parser.add_argument(
        "--reorder",
        action="store_true",
        help="Also fix position of existing method fields (default: False)"
    )

    args = parser.parse_args()

    try:
        fix_missing_method(
            input_file=args.input_file,
            method_name=args.method,
            output_file=args.output,
            backup=not args.no_backup,
            reorder=args.reorder
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
