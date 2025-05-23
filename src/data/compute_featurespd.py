import pandas as pd
import numpy as np
import datetime as dt
import argparse
import pandas_ta as ta
import glob
import os

# ---------- helper: safe division ----------
def _safe_div(numer, denom):
    """Vectorised division that returns 0 where `denom` is 0 to avoid infinities."""
    return np.where(denom == 0, 0, numer / denom)

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
df2["avg_trade_size"] = np.where(nonzero_num_trades, df["Volume"] / df["NumTrades"], 0.0)

# Technical indicators using pandas-ta
df2["ema_8"]  = ta.ema(df["Close"], length=8)
df2["ema_21"] = ta.ema(df["Close"], length=21)
df2["rsi_6"]  = ta.rsi(df["Close"], length=6)
df2["rsi_14"] = ta.rsi(df["Close"], length=14)
df2["vwap"] = (df["QuoteVolume"] / df["Volume"]).where(df["Volume"] != 0, 0.0)


# ---------- v2 proportional & normalised features ----------
# Candle range (high‑low)
df2["candle_range"] = df["High"] - df["Low"]

# Percentages of total range
df2["candle_body_pct"] = _safe_div(df2["candle_body"], df2["candle_range"])
df2["upper_wick_pct"]  = _safe_div(df2["upper_wick"],  df2["candle_range"])
df2["lower_wick_pct"]  = _safe_div(df2["lower_wick"],  df2["candle_range"])

# Core price ratios
df2["close_open_ratio"] = _safe_div(df["Close"], df["Open"])
df2["high_close_ratio"] = _safe_div(df["High"],  df["Close"])
df2["low_close_ratio"]  = _safe_div(df["Low"],   df["Close"])

# EMA & VWAP relative ratios
df2["close_ema8_ratio"]   = _safe_div(df["Close"], df2["ema_8"])
df2["close_ema21_ratio"]  = _safe_div(df["Close"], df2["ema_21"])
df2["ema8_ema21_ratio"]   = _safe_div(df2["ema_8"], df2["ema_21"])
df2["close_vwap_ratio"]   = _safe_div(df["Close"], df2["vwap"])

# Volume normalisation (24‑hour rolling statistics; assumes 1‑min candles ⇒ 1440 rows)
_window_24h = 60 * 24
df2["volume_mean_24h"]   = df["Volume"].rolling(_window_24h, min_periods=1).mean()
df2["volume_std_24h"]    = df["Volume"].rolling(_window_24h, min_periods=1).std(ddof=0)
df2["volume_zscore_24h"] = _safe_div(df["Volume"] - df2["volume_mean_24h"], df2["volume_std_24h"])

# Return volatility over 30‑minute window
df2["return_pct_vol_30m"] = df2["return_pct"].rolling(30, min_periods=1).std(ddof=0)

# Cyclical time‑of‑day encodings
dt_series = pd.to_datetime(df["OpenTime"], unit="ms", utc=True)
df2["minute_sin"] = np.sin(2 * np.pi * dt_series.dt.minute / 60)
df2["minute_cos"] = np.cos(2 * np.pi * dt_series.dt.minute / 60)
df2["hour_sin"]   = np.sin(2 * np.pi * dt_series.dt.hour   / 24)
df2["hour_cos"]   = np.cos(2 * np.pi * dt_series.dt.hour   / 24)


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
        "return_pct", "high_low_ratio", "candle_body", "candle_direction",
        "upper_wick", "lower_wick", "upper_wick_ratio", "lower_wick_ratio",
        "wick_to_body_ratio", "price_change", "buy_ratio", "quote_volume_ratio",
        "trade_density", "taker_buy_intensity", "avg_trade_size",
        "ema_8", "ema_21", "rsi_6", "rsi_14", "vwap",
        # v2 proportional & normalised features
        "candle_range", "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
        "close_open_ratio", "high_close_ratio", "low_close_ratio",
        "close_ema8_ratio", "close_ema21_ratio", "ema8_ema21_ratio", "close_vwap_ratio",
        "volume_mean_24h", "volume_std_24h", "volume_zscore_24h",
        "return_pct_vol_30m",
        "minute_sin", "minute_cos", "hour_sin", "hour_cos"
    ]
]

# ---------- export ----------
df2.to_parquet("./processed/dataset/" + savefile, index=False)
print(f"✅ Exported data to ./processed/dataset/{savefile}")