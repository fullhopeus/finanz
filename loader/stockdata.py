import pandas as pd
import yfinance as yf
import logging
import json
import os

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

def read(ticker, time):
    DATA_PATH = f"data/{ticker}.csv"
    if not os.path.exists(DATA_PATH):
        logging.info(f"Data for {ticker} not found, loading ...")
        load(ticker)
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    dt = pd.to_datetime(time, format="%Y-%m-%dT%H:%M:%S.000Z", utc=True)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df = df[df.index > dt]
    df.reset_index(inplace=True)
    df.rename(columns={df.columns[0]: 'index'}, inplace=True)
    df['index'] = pd.to_datetime(df['index'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    desired_order = [
        "Close", "Dividends", "High", "Interval", "Low", "Open", "Stock Splits", "Volume", "index"
    ]
    columns = [col for col in desired_order if col in df.columns]
    df = df[columns]
    return json.loads(df.to_json(orient='records'))

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

    delta_minutes = int(delta.total_seconds() // 60)
    delta_days = delta.days
    intervals = []

    if delta < pd.Timedelta(days=7):
        if delta_minutes < 60:
            delta_str = f"{delta_minutes}m"
        elif delta_minutes < 1440:
            delta_str = "1d"
        else:
            delta_str = f"{delta_minutes // 1440}d"
        intervals = [("1m", delta_str)]
    elif delta < pd.Timedelta(days=60):
        delta_str = f"{delta_minutes // 1440}d"
        intervals = [
            ("1m", "7d"),
            ("2m", delta_str),
        ]
    elif delta < pd.Timedelta(days=730):
        delta_str = f"{delta_minutes // 1440}d"
        intervals = [
            ("1m", "7d"),
            ("2m", "60d"),
            ("60m", delta_str)
        ]
    else:
        # `d` haben keine Begrenzung
        delta_days = int(delta.total_seconds() // 86400)
        intervals = [
            ("1m", "7d"),
            ("2m", "60d"),
            ("60m", "730d"),
            ("1d", f"{delta_days}d")
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
    combined = combined[combined.index > dt]
    combined.reset_index(inplace=True)
    combined.rename(columns={combined.columns[0]: 'index'}, inplace=True)
    combined['index'] = pd.to_datetime(combined['index'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
    desired_order = [
        "Close", "Dividends", "High", "Interval", "Low", "Open", "Stock Splits", "Volume", "index"
    ]
    columns = [col for col in desired_order if col in combined.columns]
    combined = combined[columns]
    return json.loads(combined.to_json(orient='records'))