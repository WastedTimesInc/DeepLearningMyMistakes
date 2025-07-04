import requests
import datetime as dt
import pandas as pd
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Fetch Binance Futures Kline data")
parser.add_argument("-S", "--startDate", type=str, required=True, help="Start date in format YYYY-MM-DD")
parser.add_argument("-E", "--endDate", type=str, required=True, help="End date in format YYYY-MM-DD")
parser.add_argument("-C", "--symbol", type=str, required=True, help="Trading symbol")
args = parser.parse_args()

# Parse dates
startDate = dt.datetime.strptime(args.startDate, "%Y-%m-%d")
endDate = dt.datetime.strptime(args.endDate, "%Y-%m-%d")
symbol = args.symbol

url = "https://fapi.binance.com/fapi/v1/klines"

# Convert to milliseconds
startUTC = int(startDate.timestamp() * 1000)
endUTC = int(endDate.timestamp() * 1000)

# Settings
maxRequests = 1000
interval_ms = 60 * 1000            # 1 minute in ms
numKlines = (endUTC - startUTC) // interval_ms
numRequests = numKlines // maxRequests

currentUTC = startUTC
df2 = pd.DataFrame()



for request in tqdm(range(int(numRequests)), desc="Fetching Klines"):
    params = {
        "symbol": symbol,
        "interval": "1m",
        "startTime": currentUTC,
        "limit": maxRequests
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        if not data:
            print(f"No data returned at {dt.datetime.fromtimestamp(currentUTC / 1000)}")

            # ---------- save current chunk (if any) ----------
            if not df2.empty:
                # assign proper column names and dtypes before export
                df2.columns = [
                    "OpenTime", "Open", "High", "Low", "Close", "Volume",
                    "CloseTime", "QuoteVolume", "NumTrades",
                    "TakerBuyBase", "TakerBuyQuote", "Ignore"
                ]
                numeric_cols = [
                    "Open", "High", "Low", "Close", "Volume",
                    "QuoteVolume", "NumTrades", "TakerBuyBase", "TakerBuyQuote"
                ]
                df2[numeric_cols] = df2[numeric_cols].apply(pd.to_numeric, errors="coerce")

                first_dt = dt.datetime.fromtimestamp(df2.iloc[0]['OpenTime'] / 1000).strftime("%Y-%m-%d_%H-%M")
                last_dt = dt.datetime.fromtimestamp(df2.iloc[-1]['OpenTime'] / 1000).strftime("%Y-%m-%d_%H-%M")
                name = f"{symbol}_{first_dt}_{last_dt}.parquet"
                df2.to_parquet(f"./raw/{name}", index=False)
                print(f"✅ Exported partial data to ./raw/{name}")

                # start a fresh dataset
                df2 = pd.DataFrame()

            # skip the entire chunk that returned no data (1000 candles)
            currentUTC += interval_ms * maxRequests
            continue
        df = pd.DataFrame(data)
        df2 = pd.concat([df2, df], ignore_index=True)
        currentUTC = data[-1][0] + interval_ms
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(response.text)
        break

    # time.sleep(0.1)

df2.columns = [
    "OpenTime", "Open", "High", "Low", "Close", "Volume",
    "CloseTime", "QuoteVolume", "NumTrades",
    "TakerBuyBase", "TakerBuyQuote", "Ignore"
]
# ---------- cast numeric columns ----------
numeric_cols = [
    "Open", "High", "Low", "Close", "Volume",
    "QuoteVolume", "NumTrades", "TakerBuyBase", "TakerBuyQuote"
]
df2[numeric_cols] = df2[numeric_cols].apply(pd.to_numeric, errors="coerce")

name = symbol + '_' + startDate.strftime("%Y-%m-%d") + '_' + endDate.strftime("%Y-%m-%d") + ".parquet"
df2.to_parquet("./raw/" + name, index=False)
print("✅ Exported data to ./raw/" + name)
