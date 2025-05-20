import requests
import datetime as dt
import pandas as pd
import time
from tqdm import tqdm

url = "https://fapi.binance.com/fapi/v1/klines"

# Dates in datetime
startDate = dt.datetime(2020, 1, 1)
endDate = dt.datetime(2020, 2, 1)

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
        "symbol": "BTCUSDT",
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

    time.sleep(0.5)

df2.columns = [
    "OpenTime", "Open", "High", "Low", "Close", "Volume",
    "CloseTime", "QuoteVolume", "NumTrades",
    "TakerBuyBase", "TakerBuyQuote", "Ignore"
]

df2.to_csv("./raw/binancefull.csv", index=False)
print("✅ Exported data to ./raw/binancefull.csv")