import pandas as pd
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)

def load(ticker):
    DATA_PATH = f"data/{ticker}.csv"
    Ticker = yf.Ticker(ticker)
    intervals = [
        ("1m", "7d"),
        ("2m", "60d"),
        ("5m", "60d"),
        ("15m", "60d"),
        ("30m", "60d"),
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

def update(ticker):
    logging.info(f"Updating for {ticker} ...")
    load(ticker)
    logging.info("Update finished.")