import yfinance as yf
import pandas as pd
import os

os.makedirs('data', exist_ok=True)
ticker = input("Stock ticker (e.g. AAPL, GOOGL): ").upper()
start_date = input("Start date (YYYY-MM-DD): ")
end_date = input("End date (YYYY-MM-DD): ")

try:
    stock = yf.download(ticker, start=start_date, end=end_date)
    
    if stock.empty:
        print("Nichts gefunden (tut mir sehr leid)")
    else:
        stock.reset_index(inplace=True)
        stock['Stock'] = ticker
        stock = stock[['Date', 'Stock', 'Close']]
        file_path = './data/yf_data.csv'
        stock.to_csv(file_path, index=False)
        df = pd.read_csv(file_path)
        # -------BUG FIX-------
        with open(file_path, 'r') as f:
            lines = f.readlines()
        with open(file_path, 'w') as f:
            for i, line in enumerate(lines):
                if i != 1:
                    f.write(line)
        print(f"Finished saving path: {file_path}")
except Exception as e:
    print(f"Error: {e}")

