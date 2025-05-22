#!/usr/bin/env python3
"""
truncate_data.py

Trim leading NaNs (and the initial open position, if any) from every
<symbol>_<start>_<end>_features.parquet file inside *processed/dataset* and
apply the same cut to the corresponding labels parquet in *processed/labels*.

Rules
-----
1. Determine **cut_count** as the maximum number of consecutive NaNs from
   the top of *any* column in the features file.
2. If the row *after* that cut still has `openposition == True`, keep
   extending the cut until the first row whose `openposition` is **False**.
3. Drop those rows from *both* the features file **and** its label file.
4. Overwrite the original files in‑place.

Usage
-----
    python truncate_data.py
    # or customise paths
    python truncate_data.py -d processed/dataset -l processed/labels
"""
from __future__ import annotations

import argparse
import re
import sys
from glob import glob
from pathlib import Path

import pandas as pd


def consecutive_leading_nans(df: pd.DataFrame) -> int:
    """Return the largest run of leading NaNs among all columns."""
    # For each column, find index of first non‑NaN
    def first_valid(col) -> int:
        idx = col.first_valid_index()
        # If entire column is NaN, cut everything
        return idx if idx is not None else len(col)

    leading = [first_valid(df[col]) for col in df.columns]
    return max(leading)


def derive_label_file(
    labels_dir: Path, symbol: str, start: str, end: str
) -> Path | None:
    """Return the first labels parquet matching symbol & date range."""
    # Example: BTCUSDT_2020-01-01_2025-05-15_L30_P0.0020.parquet
    pattern = f"{symbol}_{start}_{end}_*.parquet"
    matches = sorted(Path(labels_dir).glob(pattern))
    return matches[0] if matches else None


def truncate_pair(
    dataset_path: Path,
    labels_path: Path,
) -> None:
    """Truncate dataset & labels in‑place according to rules."""
    df = pd.read_parquet(dataset_path)

    # RULE 1: base cut by leading NaNs
    cut_count = consecutive_leading_nans(df)

    # RULE 2: extend cut to cover initial open position, if any
    while cut_count < len(df) and df.loc[cut_count, "openposition"]:
        cut_count += 1

    if cut_count == 0:
        print(f"[{dataset_path.name}] nothing to truncate")
        return

    print(
        f"[{dataset_path.name}] trimming {cut_count} rows "
        f"(NaNs + open position) ..."
    )

    # New start timestamp
    new_start_time = df.loc[cut_count, "OpenTime"]

    # Apply cut
    df = df.iloc[cut_count:].reset_index(drop=True)
    df.to_parquet(dataset_path, index=False)

    # Cut labels
    lbl_df = pd.read_parquet(labels_path)
    lbl_trim = lbl_df[lbl_df["opentime"] >= new_start_time].reset_index(drop=True)
    if len(lbl_df) - len(lbl_trim) != cut_count:
        # Just warn; we still overwrite for alignment
        print(
            f"   ⚠ label rows dropped = {len(lbl_df) - len(lbl_trim)}, "
            f"feature rows dropped = {cut_count}"
        )
    lbl_trim.to_parquet(labels_path, index=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trim leading NaNs in datasets.")
    parser.add_argument(
        "-d",
        "--dataset-dir",
        type=Path,
        default=Path("processed/dataset"),
        help="Directory containing *_features.parquet files",
    )
    parser.add_argument(
        "-l",
        "--labels-dir",
        type=Path,
        default=Path("processed/labels"),
        help="Directory containing labels parquet files",
    )
    args = parser.parse_args(argv)

    if not args.dataset_dir.is_dir():
        sys.exit(f"Dataset directory {args.dataset_dir} not found.")
    if not args.labels_dir.is_dir():
        sys.exit(f"Labels directory {args.labels_dir} not found.")

    dataset_files = sorted(args.dataset_dir.glob("*_features.parquet"))

    # Regex to capture symbol and dates
    pat = re.compile(
        r"(?P<sym>[^_]+)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})_features\.parquet$"
    )

    for ds_path in dataset_files:
        m = pat.search(ds_path.name)
        if not m:
            print(f"Skipping unrecognised dataset file: {ds_path.name}")
            continue

        symbol = m["sym"]
        start = m["start"]
        end = m["end"]
        lbl_path = derive_label_file(args.labels_dir, symbol, start, end)
        if lbl_path is None:
            print(f"⚠ No matching labels parquet for {ds_path.name}; skipping.")
            continue

        try:
            truncate_pair(ds_path, lbl_path)
        except Exception as exc:
            print(f"Error processing {ds_path.name}: {exc}")


if __name__ == "__main__":
    main()
