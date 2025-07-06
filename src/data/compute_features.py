import pandas as pd
import numpy as np
import datetime as dt
import argparse
import talib
import glob
import os

parser = argparse.ArgumentParser(description="Compute Features")
parser.add_argument("-S", "--startDate", type=str, required=True, help="Start date in format YYYY-MM-DD")
parser.add_argument("-E", "--endDate", type=str, required=True, help="End date in format YYYY-MM-DD")
parser.add_argument("-C", "--symbol", type=str, required=True, help="Trading symbol")
args = parser.parse_args()

# Parse dates
startDate = dt.datetime.strptime(args.startDate, "%Y-%m-%d")
endDate = dt.datetime.strptime(args.endDate, "%Y-%m-%d")
symbol = args.symbol

loadfile = symbol + '_' + startDate.strftime("%Y-%m-%d") + '_' + endDate.strftime("%Y-%m-%d") + ".parquet"
savefile = symbol + '_' + startDate.strftime("%Y-%m-%d") + '_' + endDate.strftime("%Y-%m-%d") + "_features.parquet"

df = pd.read_parquet("./raw/" + loadfile)

# ---------- Vectorised feature computation (significantly faster than the row‑by‑row loop) ----------
df2 = pd.DataFrame(index=df.index)

df2["return_pct"]       = (df["Close"] - df["Open"]) / df["Open"]
df2["high_low_ratio"]   = (df["High"]  - df["Low"])  / df["Open"]
df2["candle_body"]      = (df["Close"] - df["Open"]).abs()
df2["candle_direction"] = np.where(df["Close"] > df["Open"], 1, -1)
df2["upper_wick"]       = df["High"] - np.maximum(df["Open"], df["Close"])
df2["lower_wick"]       = np.minimum(df["Open"], df["Close"]) - df["Low"]

nonzero_body = df2["candle_body"] != 0
df2["upper_wick_ratio"] = np.where(nonzero_body, df2["upper_wick"] / df2["candle_body"], 0.0)
df2["lower_wick_ratio"] = np.where(nonzero_body, df2["lower_wick"] / df2["candle_body"], 0.0)
df2["wick_to_body_ratio"] = np.where(
    nonzero_body,
    (df2["upper_wick"] + df2["lower_wick"]) / df2["candle_body"],
    0.0
)

df2["price_change"] = df["Close"] - df["Open"]

nonzero_volume = df["Volume"] != 0
df2["buy_ratio"]          = np.where(nonzero_volume, df["TakerBuyBase"] / df["Volume"], 0.0)
df2["quote_volume_ratio"] = np.where(nonzero_volume, df["QuoteVolume"] / df["Volume"], 0.0)
df2["trade_density"]      = np.where(nonzero_volume, df["NumTrades"] / df["Volume"], 0.0)

nonzero_quote_volume = df["QuoteVolume"] != 0
df2["taker_buy_intensity"] = np.where(
    nonzero_quote_volume, df["TakerBuyBase"] / df["QuoteVolume"], 0.0
)

nonzero_num_trades = df["NumTrades"] != 0

# Technical indicators using TA-Lib
df2["rsi_6"] = talib.RSI(df["Close"], timeperiod=6)
df2["rsi_14"] = talib.RSI(df["Close"], timeperiod=14)


# Attach the timestamp so downstream steps can verify continuity, join labels, etc.
df2.insert(0, "OpenTime", df["OpenTime"])

# ---------- derive openposition flag from labels ----------
# Locate the corresponding labels file in ./processed/labels
label_pattern = f"{symbol}_{startDate.strftime('%Y-%m-%d')}_{endDate.strftime('%Y-%m-%d')}_*.parquet"
label_files = glob.glob(os.path.join("./processed/labels", label_pattern))
if len(label_files) == 0:
    raise FileNotFoundError(f"Could not find labels parquet matching: {label_pattern}")
# We expect exactly one matching file for the symbol/date range; take the first match
labels_df = pd.read_parquet(label_files[0])

# Build the openposition flag: True when a position was already open at the **start** of the bar
open_state = False
open_flags = []
for lbl in labels_df["label"]:
    open_flags.append(open_state)
    if lbl.startswith("OPEN"):
        open_state = True
    elif lbl.startswith("CLOSE"):
        open_state = False
labels_df["openposition"] = open_flags

# Align column names with features dataframe for merging
labels_df.rename(columns={"opentime": "OpenTime"}, inplace=True)

# Merge the flag into the features dataframe on timestamp
df2 = df2.merge(labels_df[["OpenTime", "openposition"]], on="OpenTime", how="left")

# Any rows without a corresponding label are assumed to have no open position
df2["openposition"].fillna(False, inplace=True)

# ---------- final column order ----------
df2 = df2[
    [
        "OpenTime", "openposition",
        "return_pct", "high_low_ratio","candle_direction",
        "upper_wick", "lower_wick", "upper_wick_ratio", "lower_wick_ratio",
        "wick_to_body_ratio","buy_ratio", "quote_volume_ratio",
        "trade_density", "taker_buy_intensity","rsi_6", "rsi_14"
    ]
]
meta_cols = ["OpenTime", "openposition"]
feature_cols = [col for col in df2.columns if col not in meta_cols]

# Manual z-score normalization
feature_means = df2[feature_cols].mean()
feature_stds = df2[feature_cols].std()
df2[feature_cols] = df2[feature_cols].clip(lower=-10, upper=10)
# Avoid divide-by-zero
feature_stds = feature_stds.replace(0, 1)

# Standardize
df2[feature_cols] = (df2[feature_cols] - feature_means) / feature_stds
df2.replace([np.inf, -np.inf], 0.0, inplace=True)
df2.fillna(0.0, inplace=True)
# ---------- export ----------
df2.to_parquet("./processed/dataset/" + savefile, index=False)
print(f"✅ Exported data to ./processed/dataset/{savefile}")
