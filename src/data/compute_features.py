import pandas as pd
import numpy as np
import datetime as dt
import argparse
import talib

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

# Technical indicators using TA-Lib
df2["ema_8"] = talib.EMA(df["Close"], timeperiod=8)
df2["ema_21"] = talib.EMA(df["Close"], timeperiod=21)
df2["rsi_6"] = talib.RSI(df["Close"], timeperiod=6)
df2["rsi_14"] = talib.RSI(df["Close"], timeperiod=14)
df2["vwap"] = (df["QuoteVolume"] / df["Volume"]).where(df["Volume"] != 0, 0.0)

# Maintain original column order for compatibility
df2 = df2[
    [
        "return_pct", "high_low_ratio", "candle_body", "candle_direction",
        "upper_wick", "lower_wick", "upper_wick_ratio", "lower_wick_ratio",
        "wick_to_body_ratio", "price_change", "buy_ratio", "quote_volume_ratio",
        "trade_density", "taker_buy_intensity", "avg_trade_size",
        "ema_8", "ema_21", "rsi_6", "rsi_14", "vwap"
    ]
]

# ---------- export ----------
df2.to_parquet("./processed/dataset/" + savefile, index=False)
print(f"✅ Exported data to ./processed/dataset/{savefile}")