import pandas as pd
import numpy as np
import datetime as dt
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Compute Features")
parser.add_argument("-S", "--startDate", type=str, required=True, help="Start date in format YYYY-MM-DD")
parser.add_argument("-E", "--endDate", type=str, required=True, help="End date in format YYYY-MM-DD")
parser.add_argument("-C", "--symbol", type=str, required=True, help="Trading symbol")
args = parser.parse_args()

# Parse dates
startDate = dt.datetime.strptime(args.startDate, "%Y-%m-%d")
endDate = dt.datetime.strptime(args.endDate, "%Y-%m-%d")
symbol = args.symbol

loadfile = symbol + '_' + startDate.strftime("%Y-%m-%d") + '_' + endDate.strftime("%Y-%m-%d") + ".csv"
savefile = symbol + '_' + startDate.strftime("%Y-%m-%d") + '_' + endDate.strftime("%Y-%m-%d") + "_features.csv"

df = pd.read_csv("./raw/" + loadfile)

df2 = pd.DataFrame(columns=["return_pct", "high_low_ratio",
                            "candle_body", "candle_direction", "upper_wick",
                            "lower_wick", "upper_wick_ratio", "lower_wick_ratio",
                            "wick_to_body_ratio", "price_change", "buy_ratio",
                            "quote_volume_ratio", "trade_density", "taker_buy_intensity",
                            "avg_trade_size"])

for index, row in tqdm(df.iterrows(), total=len(df), desc="Computing Features"):
    Close = df.at[index, "Close"]
    Open = df.at[index, "Open"]
    High = df.at[index, "High"]
    Low = df.at[index, "Low"]
    Volume = df.at[index, "Volume"]
    QuoteVolume = df.at[index, "QuoteVolume"]
    TakerBuyBase = df.at[index, "TakerBuyBase"]
    NumTrades = df.at[index, "NumTrades"]
    # return_pct
    df2.loc[index, df2.columns[0]] = (Close - Open) / Open
    # high_low_ratio
    df2.loc[index, df2.columns[1]] = (High - Low) / Open
    # candle_body
    df2.loc[index, df2.columns[2]] = np.abs(Close - Open)
    # candle_direction
    if Close > Open:
        df2.loc[index, df2.columns[3]] = 1
    else:
        df2.loc[index, df2.columns[3]] = -1
    # upper_wick
    df2.loc[index, df2.columns[4]] = High - max(Open, Close)
    # lower_wick
    df2.loc[index, df2.columns[5]] = min(Open, Close) - Low

    if df2.at[index, "candle_body"] != 0:
        # upper_wick_ratio
        df2.loc[index, df2.columns[6]] = df2.at[index, "upper_wick"] / df2.at[index, "candle_body"]
        # lower_wick_ratio
        df2.loc[index, df2.columns[7]] = df2.at[index, "lower_wick"] / df2.at[index, "candle_body"]
        # wick_to_body_ratio
        df2.loc[index, df2.columns[8]] = (df2.at[index, "upper_wick"] + df2.at[index, "lower_wick"]) / df2.at[
            index, "candle_body"]
    else:
        df2.loc[index, df2.columns[6]] = 0.0
        df2.loc[index, df2.columns[7]] = 0.0
        df2.loc[index, df2.columns[8]] = 0.0

    # price_chage
    df2.loc[index, df2.columns[9]] = Close - Open

    if Volume != 0:
        # buy_ratio
        df2.loc[index, df2.columns[10]] = TakerBuyBase / Volume
        # quote_volume_ratio
        df2.loc[index, df2.columns[11]] = QuoteVolume / Volume
        # trade_density
        df2.loc[index, df2.columns[12]] = NumTrades / Volume
    else :
        df2.loc[index, df2.columns[10]] = 0.0
        df2.loc[index, df2.columns[11]] = 0.0
        df2.loc[index, df2.columns[12]] = 0.0

    if QuoteVolume != 0:
        # taker_buy_intensity
        df2.loc[index, df2.columns[13]] = TakerBuyBase / QuoteVolume
    else:
        df2.loc[index, df2.columns[13]] = 0.0

    if NumTrades != 0:
        # avg_trade_size
        df2.loc[index, df2.columns[14]] = Volume / NumTrades
    else:
        df2.loc[index, df2.columns[14]] = 0.0



df2.to_csv("./features/" + loadfile, index=False)
print("✅ Exported data to ./features/" + loadfile)