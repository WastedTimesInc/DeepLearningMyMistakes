import argparse
import pandas as pd
import numpy as np
import os
import datetime as dt
from tqdm import tqdm
import pandas_ta as ta

def main():
    parser = argparse.ArgumentParser(description="Label data with multiple trading strategies.")
    parser.add_argument("-S", "--startDate", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("-E", "--endDate",   required=True, help="End date   YYYY-MM-DD")
    parser.add_argument("-C", "--symbol",   required=True, help="Trading symbol, e.g. BTCUSDT")
    parser.add_argument("-L", "--lookahead", type=int, default=20, help="Look‑ahead window in candles")
    parser.add_argument("-P", "--min_profit", type=float, default=0.002, help="Minimum profit threshold")
    args = parser.parse_args()

    # Parse
    startDate = dt.datetime.strptime(args.startDate, "%Y-%m-%d")
    endDate   = dt.datetime.strptime(args.endDate,   "%Y-%m-%d")
    symbol    = args.symbol.upper()
    lookahead = args.lookahead
    min_profit = args.min_profit

    raw_dir = "raw"
    processed_dir = "processed/labels"
    os.makedirs(processed_dir, exist_ok=True)

    loadfile = f"{symbol}_{startDate.strftime('%Y-%m-%d')}_{endDate.strftime('%Y-%m-%d')}.parquet"
    savefile = f"{symbol}_{startDate.strftime('%Y-%m-%d')}_{endDate.strftime('%Y-%m-%d')}_L{lookahead}_P{min_profit:.4f}.parquet"

    raw_path = os.path.join(raw_dir, loadfile)
    out_path = os.path.join(processed_dir, savefile)

    print(f"Loading {raw_path} ...")
    df = pd.read_parquet(raw_path)
    # Normalize columns to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Compute indicators
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    volume = df['volume'].values
    buy_vol = df['buy_volume'].values if 'buy_volume' in df.columns else (
        df['takerbuybase'].values if 'takerbuybase' in df.columns else np.full_like(volume, np.nan)
    )

    df['ema8']  = ta.ema(df['close'], length=8)
    df['ema21'] = ta.ema(df['close'], length=21)
    df['rsi14'] = ta.rsi(df['close'], length=14)
    macd_df = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_hist'] = macd_df['MACDh_12_26_9']
    bb = ta.bbands(df['close'], length=20)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    # VWAP
    typical_price = (high + low + close) / 3
    vwap = np.cumsum(typical_price * volume) / np.maximum(np.cumsum(volume), 1e-9)
    df['vwap'] = vwap
    # Rolling volume
    df['vol_roll5'] = pd.Series(volume).rolling(5).mean().values
    # Buy ratio
    if not np.all(np.isnan(buy_vol)):
        df['buy_ratio'] = buy_vol / (volume + 1e-9)
    else:
        df['buy_ratio'] = np.nan
    # Close‑to‑close percentage return and its acceleration
    df['close_ret'] = pd.Series(close).pct_change().fillna(0).values
    df['acceleration'] = df['close_ret'].diff().fillna(0).values
    # High/low pct
    df['high_low_pct'] = (high - low) / np.maximum(low, 1e-9)

    # Strategies (10)
    strategies = []
    # 1. RSI reversal
    strategies.append(lambda i: 1 if df['rsi14'].iloc[i] < 30 else (-1 if df['rsi14'].iloc[i] > 70 else 0))
    # 2. EMA cross
    strategies.append(lambda i: 1 if df['ema8'].iloc[i] > df['ema21'].iloc[i] else (-1 if df['ema8'].iloc[i] < df['ema21'].iloc[i] else 0))
    # 3. Bollinger bands
    strategies.append(lambda i: 1 if close[i] < df['bb_lower'].iloc[i] else (-1 if close[i] > df['bb_upper'].iloc[i] else 0))
    # 4. MACD hist cross
    strategies.append(lambda i: 1 if df['macd_hist'].iloc[i] > 0 else (-1 if df['macd_hist'].iloc[i] < 0 else 0))
    # 5. VWAP deviation
    strategies.append(lambda i: 1 if close[i] < df['vwap'].iloc[i] * 0.995 else (-1 if close[i] > df['vwap'].iloc[i] * 1.005 else 0))
    # 6. Volume spike + return direction (handled vectorized later)
    strategies.append(None)
    # 7. Buy ratio extreme
    strategies.append(lambda i: 1 if (not np.isnan(df['buy_ratio'].iloc[i])) and (df['buy_ratio'].iloc[i] > 0.7) else (-1 if (not np.isnan(df['buy_ratio'].iloc[i])) and (df['buy_ratio'].iloc[i] < 0.3) else 0))
    # 8. Acceleration up/down
    strategies.append(lambda i: 1 if df['acceleration'].iloc[i] > 0.001 else (-1 if df['acceleration'].iloc[i] < -0.001 else 0))
    # 9. High/low pct & candle direction
    strategies.append(lambda i: 1 if (df['high_low_pct'].iloc[i] > 0.01 and close[i] > open_[i]) else (-1 if (df['high_low_pct'].iloc[i] > 0.01 and close[i] < open_[i]) else 0))
    # 10. Alt blind (toggle): handled separately

    # Pre-compute volume spike candidate indices to avoid looping over every row
    mean_vol = np.nanmean(df['vol_roll5'])
    vol_long_flags = (df['vol_roll5'] > 1.5 * mean_vol) & (close > open_)
    vol_short_flags = (df['vol_roll5'] > 1.5 * mean_vol) & (close < open_)

    trade_candidates = []
    # For progress bar, use tqdm
    for strat_idx, strat in enumerate(strategies + [None]):  # Last is alt blind
        strat_name = (
            [
                "RSI reversal", "EMA cross", "Bollinger bands", "MACD hist cross",
                "VWAP deviation", "Volume spike", "Buy ratio", "Acceleration", "High/Low pct", "Alt blind"
            ][strat_idx]
        )
        if strat_idx == 5:  # volume spike handled vectorized
            strat_name = "Volume spike"
            long_idxs = np.where(vol_long_flags.values)[0]
            short_idxs = np.where(vol_short_flags.values)[0]
            # Combine and sort
            cand_idxs = np.sort(np.concatenate([long_idxs, short_idxs]))
            direction_map = {idx: 1 for idx in long_idxs}
            direction_map.update({idx: -1 for idx in short_idxs})
            last_close_idx = -1
            for i in cand_idxs:
                if i <= last_close_idx or i + lookahead >= len(df):
                    continue
                dir_ = direction_map[i]
                end = i + lookahead
                profit = (close[end] - close[i]) / close[i] if dir_ == 1 else (close[i] - close[end]) / close[i]
                trade_candidates.append({
                    "start": i,
                    "end": end,
                    "direction": dir_,
                    "profit": profit,
                    "strategy": strat_name
                })
                last_close_idx = end
            continue  # Skip normal per-row loop
        if strat is None:
            # Alt blind: alternate long/short every lookahead
            direction = 1
            i = 0
            while i < len(df) - lookahead:
                end = min(i + lookahead, len(df) - 1)
                profit = (close[end] - close[i]) / close[i] if direction == 1 else (close[i] - close[end]) / close[i]
                trade_candidates.append({
                    "start": i,
                    "end": end,
                    "direction": direction,
                    "profit": profit,
                    "strategy": strat_name
                })
                direction *= -1
                i += lookahead
            continue
        last_close_idx = -1
        pbar = tqdm(range(len(df) - lookahead), desc=f"Strategy {strat_name}", leave=False)
        for i in pbar:
            if i <= last_close_idx:
                continue
            dir_ = strat(i)
            if dir_ == 0:
                continue
            end = i + lookahead
            if end >= len(df):
                continue
            profit = (close[end] - close[i]) / close[i] if dir_ == 1 else (close[i] - close[end]) / close[i]
            trade_candidates.append({
                "start": i,
                "end": end,
                "direction": dir_,
                "profit": profit,
                "strategy": strat_name
            })
            last_close_idx = end

    # --- Stage 1: invert losing trades so that all profits are positive ---
    for t in trade_candidates:
        if t['profit'] < 0:
            t['direction'] *= -1
            t['profit']    *= -1

    # Filter by min_profit
    filtered_trades = []
    for t in tqdm(trade_candidates, desc="Min‑profit filter", leave=False):
        if t['profit'] >= min_profit:
            filtered_trades.append(t)
    print(f"Total trades before filtering: {len(trade_candidates)}")
    print(f"Trades after min_profit filter: {len(filtered_trades)}")

    # Merge: keep non-overlapping, sort by profit desc
    filtered_trades.sort(key=lambda x: -x['profit'])
    occupied = set()
    selected_trades = []
    for t in tqdm(filtered_trades, desc="Non‑overlap merge", leave=False):
        # Check overlap
        overlap = False
        for idx in range(t['start'], t['end'] + 1):
            if idx in occupied:
                overlap = True
                break
        if not overlap:
            selected_trades.append(t)
            for idx in range(t['start'], t['end'] + 1):
                occupied.add(idx)

    print(f"Selected non-overlapping trades: {len(selected_trades)}")
    # Build labels
    final_labels = ["HOLD"] * len(df)
    for t in selected_trades:
        if t['direction'] == 1:
            final_labels[t['start']] = "OPEN_LONG"
            final_labels[t['end']] = "CLOSE"
        elif t['direction'] == -1:
            final_labels[t['start']] = "OPEN_SHORT"
            final_labels[t['end']] = "CLOSE"

    df['label'] = final_labels
    df['label_strategy'] = "NONE"
    for t in selected_trades:
        df.loc[t['start'], 'label_strategy'] = t['strategy']
        df.loc[t['end'], 'label_strategy'] = t['strategy']

    # Only keep timestamp and label in the final output
    print(f"Saving labeled data (OpenTime + label) to {out_path} ...")
    # Determine the timestamp column name (after lowercase normalisation)
    ts_col = next((c for c in ("opentime", "open_time", "timestamp", "time") if c in df.columns), None)
    if ts_col is None:
        raise KeyError("Timestamp column not found – cannot save labels without it.")
    df_out = df[[ts_col, "label"]].copy()
    df_out.to_parquet(out_path)
    n_long = sum(1 for t in selected_trades if t['direction'] == 1)
    n_short = sum(1 for t in selected_trades if t['direction'] == -1)

    profits = [t['profit'] for t in selected_trades]
    total_days = (endDate - startDate).days or 1
    trade_density = len(selected_trades) / len(df)
    trades_per_day = len(selected_trades) / total_days
    max_gain = max(profits) if profits else 0
    min_gain = min(profits) if profits else 0
    avg_gain = np.mean(profits) if profits else 0

    print(f"Labeling complete. {n_long} long, {n_short} short, {len(selected_trades)} total trades.")
    print(f"Trade density (trades/candle): {trade_density:.6f}")
    print(f"Trades per day: {trades_per_day:.2f}")
    print(f"Profit stats  →  max: {max_gain:.4f}   min: {min_gain:.4f}   avg: {avg_gain:.4f}")
    print(f"Params: lookahead={lookahead}, min_profit={min_profit}")

if __name__ == "__main__":
    main()