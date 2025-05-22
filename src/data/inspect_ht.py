

#!/usr/bin/env python3
"""
inspect_ht.py

A tiny helper script that prints the first and last few rows of a Parquet file.

Usage:
    python inspect_ht.py <path_to_parquet>
"""

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python inspect_ht.py <path_to_parquet>")
        sys.exit(1)

    parquet_path = Path(sys.argv[1])

    if not parquet_path.is_file():
        print(f"Error: {parquet_path} does not exist or is not a file.")
        sys.exit(1)

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        print(f"Failed to read Parquet file: {exc}")
        sys.exit(1)

    print("\n--- HEAD (first 5 rows) ---")
    print(df.head())

    print("\n--- TAIL (last 5 rows) ---")
    print(df.tail())

    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()