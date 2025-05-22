#!/usr/bin/env python3
"""
data_validation.py

Utility script that scans the *data/processed/labels* folder for every
labelled dataset (``*.parquet``) and, for each one, simulates a simple
all‑in trading strategy driven by the labels.

For every trade:
    • We start with a virtual account of 100 units of currency.
    • When the label signals a trade (non‑zero action), we enter a
      position at the *open* price of that candle.
        – ``action > 0`` ⇒ go **long** (buy).
        – ``action < 0`` ⇒ go **short** (sell short).
    • We close the position at the *open* price of the next candle that
      contains any non‑zero label (i.e. the next trading signal) or, if
      no further signal exists, at the last candle.
    • After closing, the entire resulting balance is reinvested in the
      next trade.
The script plots the account value through time (compounded ROI) and
saves the figure alongside the label file, named
``<original_file_without_ext>_roi.png``.
The script is self‑contained—simply run:

    python data_validation.py

Dependencies: pandas, pyarrow, matplotlib
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # non‑interactive backend for faster saves

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import numpy as np

# Column names mapping – adjust here if the datasets change
LABEL_TIMESTAMP_COL = "opentime"   # in label parquet
RAW_TIMESTAMP_COL   = "OpenTime"   # in raw parquet

# Enumerated label values present in the label column
LABEL_VALUES = {"OPEN_LONG", "OPEN_SHORT", "CLOSE", "HOLD"}


# --------------------------------------------------------------------------- #
#  Core processing helpers
# --------------------------------------------------------------------------- #
def _locate_raw_file(label_file: Path) -> Path | None:
    """
    Infer the matching raw dataset path from a label file path.
    Example:
        labels/BTCUSDT_2020-01-01_2025-05-15_L30_P0.0020.parquet
        → raw/BTCUSDT_2020-01-01_2025-05-15.parquet
    """
    base_name = label_file.name.split("_L")[0]  # e.g. BTCUSDT_2020-01-01_2025-05-15
    raw_path = label_file.parents[2] / "raw" / f"{base_name}.parquet"
    return raw_path if raw_path.exists() else None






def _pick_label_column(df: pd.DataFrame) -> str | None:
    """
    Detect the column whose values are drawn from the expected `LABEL_VALUES`.
    """
    for col in df.columns:
        values = set(df[col].dropna().unique())
        if values and values.issubset(LABEL_VALUES):
            return col
    return None


def _pick_open_column(df: pd.DataFrame) -> str | None:
    """
    Return the name of a column that looks like an OHLC "open" price.
    Tries common variants in a priority order; falls back to the first float
    column if only a single candidate exists.
    """
    preferred = ["open", "Open", "open_price", "OpenPrice", "o", "OpenPriceUSD"]
    for col in preferred:
        if col in df.columns:
            return col
    # Fallback: single float column?
    float_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(float_cols) == 1:
        return float_cols[0]
    return None


def _ensure_utc_datetime(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Convert column *col* of *df* to UTC-aware pandas Timestamps.
    Handles integer epoch milliseconds as well as string/naive datetimes.
    """
    series = df[col]
    if np.issubdtype(series.dtype, np.integer):
        return pd.to_datetime(series, unit="ms", utc=True)
    # If already datetime but tz-naive, make it UTC
    if np.issubdtype(series.dtype, np.datetime64):
        out = pd.to_datetime(series, utc=True)
        if out.dt.tz is None:
            out = out.dt.tz_localize("UTC")
        return out
    # Fall back: parse strings
    return pd.to_datetime(series, utc=True, errors="coerce")


def _simulate_trades(df: pd.DataFrame, label_col: str) -> pd.Series:
    """
    Simulate a fixed-stake (100 units per trade) strategy where:
        * OPEN_LONG  → open long
        * OPEN_SHORT → open short
        * CLOSE      → close any open position
        * HOLD       → no action
    The position is always closed at the *next* CLOSE label, or at the final
    candle if no further CLOSE is encountered.
    The strategy always risks a fixed stake (100 units) per trade and does not compound.
    """
    STAKE = 100.0
    capital = STAKE
    timestamps, balances = [], []

    position = None            # None | "long" | "short"
    entry_price = None

    # Progress bar per‑row for finer granularity
    iterator = tqdm(
        df.iterrows(),
        total=len(df),
        desc="Simulating trades",
        unit="row",
        leave=False,
    )
    for ts, row in iterator:
        open_px = row["open"]
        label   = row[label_col]

        timestamps.append(ts)
        balances.append(capital)

        # Open a position if flat
        if position is None:
            if label == "OPEN_LONG":
                position, entry_price = "long", open_px
            elif label == "OPEN_SHORT":
                position, entry_price = "short", open_px
            continue  # go to next row

        # We have an open position → check for close signal
        if label == "CLOSE":
            if position == "long":
                pct_change = (open_px - entry_price) / entry_price
            else:  # short
                pct_change = (entry_price - open_px) / entry_price
            capital += STAKE * pct_change
            position, entry_price = None, None

    # If a position is still open at the end → close at last price
    if position is not None:
        last_price = df["open"].iloc[-1]
        if position == "long":
            pct_change = (last_price - entry_price) / entry_price
        else:
            pct_change = (entry_price - last_price) / entry_price
        capital += STAKE * pct_change
        timestamps.append(df.index[-1])
        balances.append(capital)

    return pd.Series(balances, index=pd.to_datetime(timestamps), name="capital")


def _plot_roi(series: pd.Series, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots()
    series.plot(ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Account value %")
    ax.set_xlabel("Time")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main driver
# --------------------------------------------------------------------------- #
def main() -> None:
    script_dir = Path(__file__).resolve().parent            # …/src/data
    labels_dir = script_dir / "processed" / "labels"        # …/src/data/processed/labels

    if not labels_dir.exists():
        sys.stderr.write(f"[ERROR] Labels directory not found: {labels_dir}\n")
        sys.exit(1)

    label_files = sorted(labels_dir.glob("*.parquet"))
    if not label_files:
        sys.stderr.write(f"[WARNING] No label files found in {labels_dir}\n")
        return

    for label_path in tqdm(label_files, desc="Processing label files", unit="file"):
        print(f"[INFO] Processing {label_path.name}")

        raw_path = _locate_raw_file(label_path)
        if raw_path is None:
            sys.stderr.write(f"  ↳ matching raw dataset not found, skipping\n")
            continue

        # Load data
        label_df = pd.read_parquet(label_path)
        raw_df = pd.read_parquet(raw_path)

        # Ensure uniform timestamp index
        label_df[LABEL_TIMESTAMP_COL] = _ensure_utc_datetime(label_df, LABEL_TIMESTAMP_COL)
        raw_df[RAW_TIMESTAMP_COL]     = _ensure_utc_datetime(raw_df, RAW_TIMESTAMP_COL)

        label_df.set_index(LABEL_TIMESTAMP_COL, inplace=True)
        raw_df.set_index(RAW_TIMESTAMP_COL, inplace=True)

        # Align on timestamp and keep open price & label column
        label_col = _pick_label_column(label_df)
        if label_col is None:
            sys.stderr.write(f"  ↳ no recognised label column found, skipping\n")
            continue

        # Identify open‑price column in raw data
        open_col = _pick_open_column(raw_df)
        if open_col is None:
            sys.stderr.write("  ↳ could not find an 'open' price column, skipping\n")
            continue

        joined = (
            label_df[[label_col]]
            .join(raw_df[[open_col]].rename(columns={open_col: "open"}), how="inner")
            .dropna(subset=[label_col, "open"])
        )

        roi_series = _simulate_trades(joined, label_col)

        # --------------------------------------------------------------- #
        #  Debug printout – inspect first/last few simulated balances
        # --------------------------------------------------------------- #
        print("  ↳ ROI series (head):")
        print(roi_series.head(10).to_string())
        print("  ↳ ROI series (tail):")
        print(roi_series.tail(10).to_string())

        # Down‑sample to one point per day (end‑of‑day balance) for plotting
        roi_daily = (
            roi_series.resample("D").last().dropna()
        )  # keeps ~1.8 k points over 5 yrs instead of hundreds of thousands

        out_png = label_path.with_suffix("").with_name(label_path.stem + "_roi.png")
        _plot_roi(
            roi_daily,
            title=f"Non Compounded ROI — {label_path.stem}",
            out_path=out_png,
        )
        print(f"  ↳ saved ROI chart → {out_png.relative_to(labels_dir.parent.parent)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
