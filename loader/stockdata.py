import pandas as pd
import yfinance as yf
import logging
import json

logging.basicConfig(level=logging.INFO)

def load(ticker):
    DATA_PATH = f"data/{ticker}.csv"
    Ticker = yf.Ticker(ticker)
    intervals = [
        ("1m", "7d"),
        ("2m", "60d"),
        #("5m", "60d"),
        #("15m", "60d"),
        #("30m", "60d"),
        ("60m", "730d"),
        ("1d", "max"),
    ]
    full_data = []
    for interval, period in intervals:
        try:
            data = Ticker.history(interval=interval, period=period)
            if not data.empty:
                data = data.copy()
                data["Interval"] = interval
                full_data.append(data)
                logging.info(f"Loading: {interval} / {period} for {ticker}")
        except Exception as e:
            logging.error(f"Error for {interval}: {e}")
            continue
    if not full_data:
        logging.warning("No data found. Are you giving a wrong ticker?")
        return None
    # Fix first two lines in file but no more need
    #with open(DATA_PATH, 'r') as f:
        #lines = f.readlines()
    #with open(DATA_PATH, 'w') as f:
        #for i, line in enumerate(lines):
            #if i not in [1, 2]:  # skip line 1 and 2
                #f.write(line)
    # Comment: Vielleicht hat yfinanz das schon weggeworfen? Ich weiss nix.
    combined = pd.concat(full_data)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    combined.reset_index(inplace=True)
    combined.to_csv(DATA_PATH, index=False)
    logging.info(f"Saved im {DATA_PATH}")

def update(ticker, time):
    # The `time` is Javascript time.
    logging.info(f"Updating for {ticker} ...")
    dt = pd.to_datetime(time, format="%Y-%m-%dT%H:%M:%S.000Z", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    delta = now - dt
    logging.info(f"Time since last update: {delta}")

    if delta.total_seconds() < 60:
        logging.info("No update needed")
        return
    
    if delta < pd.Timedelta(days=7):
        delta_str = f"{int(delta.total_seconds() // 60)}m"
        intervals = [
            ("1m", delta_str)
        ]
    elif delta < pd.Timedelta(days=60):
        delta_str = f"{int(delta.total_seconds() // 60)}m"
        intervals = [
            ("1m", "7d"),
            ("2m", delta_str)
        ]   
    elif delta < pd.Timedelta(days=730):
        delta_str = f"{int(delta.total_seconds() // 60)}m"
        intervals = [
            ("1m", "7d"),
            ("2m", "60d"),
            ("60m", delta_str)
        ]
    else:
        delta_str = f"{int(delta.total_seconds() // 86400)}d"
        intervals = [
            ("1m", "7d"),
            ("2m", "60d"),
            ("60m", "730d"),
            ("1d", delta_str)
        ]
    Ticker = yf.Ticker(ticker)
    full_data = []
    for interval, period in intervals:
        try:
            data = Ticker.history(interval=interval, period=period)
            if not data.empty:
                data = data.copy()
                data["Interval"] = interval
                full_data.append(data)
                logging.info(f"Loading: {interval} / {period} for {ticker}")
        except Exception as e:
            logging.error(f"Error for {interval}: {e}")
            continue
    if not full_data:
        logging.warning("No data found. Are you giving a wrong ticker?")
        return None
    combined = pd.concat(full_data)
    combined = combined[~combined.index.duplicated(keep="first")]
    combined = combined.sort_index()
    combined.reset_index(inplace=True)
    combined['index'] = pd.to_datetime(combined['index'], utc=True)
    combined.set_index('index', inplace=True)
    combined.reset_index(inplace=True)
    combined['index'] = combined['index'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    desired_order = [
        "Close", "Dividends", "High", "Interval", "Low", "Open", "Stock Splits", "Volume", "index"
    ]
    columns = [col for col in desired_order if col in combined.columns]
    combined = combined[columns]
    return json.loads(combined.to_json(orient='records'))