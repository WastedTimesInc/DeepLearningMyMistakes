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
            print(f"No data returned at {dt.datetime(currentUTC / 1000)}")
            break
        df = pd.DataFrame(data)
        df2 = pd.concat([df2, df], ignore_index=True)
        currentUTC = data[-1][0] + interval_ms
    else:
        print(f"❌ Request failed: {response.status_code}")
        print(response.text)
        break

    time.sleep(0.1)

df2.columns = [
    "OpenTime", "Open", "High", "Low", "Close", "Volume",
    "CloseTime", "QuoteVolume", "NumTrades",
    "TakerBuyBase", "TakerBuyQuote", "Ignore"
]

name = symbol + '_' + startDate.strftime("%Y-%m-%d") + '_' + endDate.strftime("%Y-%m-%d") + ".csv"
df2.to_csv("./raw/" + name, index=False)
print("✅ Exported data to ./raw/" + name)