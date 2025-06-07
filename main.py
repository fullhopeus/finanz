from flask import Flask, request, jsonify
app = Flask(__name__)
import pandas as pd
import os
import loader.stockdata as dl
@app.route('/api/stock/data/<ticker>', methods=['GET'])
def read_stock_data(ticker):
    try:
        filepath = f"data/{ticker}.csv"
        if not os.path.exists(filepath):
            dl.update(ticker)
            if not os.path.exists(filepath):
                return jsonify({"error": "Stock data not found"}), 404
        else:
            dl.update(ticker)
        stock_df = pd.read_csv(filepath)
        stock_df['index'] = pd.to_datetime(stock_df['index'], utc=True)
        stock_df.set_index('index', inplace=True)
        stock_df.reset_index(inplace=True)
        stock_df['index'] = stock_df['index'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        return jsonify(stock_df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == "__main__":
    app.run(debug=True)