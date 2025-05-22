

#!/usr/bin/env python3
"""
inspect_ht.py

A tiny helper script that prints the first and last few rows of a Parquet file.

Usage:
    python inspect_ht.py <path_to_parquet>
"""

import sys
from pathlib import Path
import argparse

import pandas as pd


def main() -> None:
    """
    Print the head and/or tail of a Parquet file.

    Flags
    -----
    -H / --head   : Show only the first N rows
    -T / --tail   : Show only the last N rows
    -N / --num    : Number of rows to show (default: 5)
    """
    parser = argparse.ArgumentParser(
        prog="inspect_ht.py",
        description="Print head and/or tail of a Parquet file.",
    )
    parser.add_argument(
        "parquet_path",
        type=Path,
        help="Path to the Parquet file to inspect.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-H",
        "--head",
        action="store_true",
        help="Show only the first N rows.",
    )
    group.add_argument(
        "-T",
        "--tail",
        action="store_true",
        help="Show only the last N rows.",
    )

    parser.add_argument(
        "-N",
        "--num",
        type=int,
        default=5,
        metavar="N",
        help="Number of rows to show (default: 5).",
    )

    args = parser.parse_args()

    # Validate Parquet file
    if not args.parquet_path.is_file():
        print(f"Error: {args.parquet_path} does not exist or is not a file.")
        sys.exit(1)

    try:
        df = pd.read_parquet(args.parquet_path)
    except Exception as exc:
        print(f"Failed to read Parquet file: {exc}")
        sys.exit(1)

    n = max(args.num, 0)

    # Display output based on flags
    if not args.tail:  # Show head unless tail-only requested
        print(f"\n--- HEAD (first {n} rows) ---")
        print(df.head(n))

    if not args.head:  # Show tail unless head-only requested
        print(f"\n--- TAIL (last {n} rows) ---")
        print(df.tail(n))

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()