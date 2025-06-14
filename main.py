from flask import Flask, request, jsonify
import pandas as pd
import os
import loader.stockdata as dl

app = Flask(__name__)

@app.route('/api/stock/data/<ticker>', methods=['GET'])
def read_stock_data(ticker):
    try:
        filepath = f"data/{ticker}.csv"
        if not os.path.exists(filepath):
            dl.load(ticker)
            if not os.path.exists(filepath):
                return jsonify({"error": "Stock data not found"}), 404
        else:
            dl.load(ticker)
        stock_df = pd.read_csv(filepath)
        stock_df['index'] = pd.to_datetime(stock_df['index'], utc=True)
        stock_df['index'] = stock_df['index'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        return jsonify(stock_df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stock/data/<ticker>/<time>', methods=['GET'])
def real_time_stock_data(ticker, time):
    json_data = dl.update(ticker, time)
    if json_data is None:
        return jsonify({"error": "Stock data not found"}), 404
    return jsonify(json_data)

@app.route('/api/stock/data/update/<ticker>/<time>', methods=['GET'])
def update_stock_data(ticker, time):
    json_data = dl.read(ticker, time)
    if json_data is None:
        return jsonify({"error": "Stock data not found"}), 404
    return jsonify(json_data)

if __name__ == "__main__":
    app.run(debug=True)