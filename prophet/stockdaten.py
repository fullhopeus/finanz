import pandas as pd
import numpy as np
import random
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)

# --- Configs ---
SPLIT_RATIO = 0.85
SEQ_LEN = 10
EPOCHS = 100
LR = 0.01
FUTURE_DAYS = 90
DATA_PATH = "data/stock.csv"

def load_stock_data(ticker):
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
                logging.info(f"Loading: {interval} / {period} für {ticker}")
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

# Uncomment to use
load_stock_data("AAPL")

# --- Data Prep ---
df = pd.read_csv(DATA_PATH)
price_data = df[['Close']].values
data_len = len(price_data)

train_size = int(data_len * SPLIT_RATIO)
train_data = price_data[:train_size]
test_data = price_data[train_size:]

scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.fit_transform(test_data)

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X).astype(float), np.array(y).astype(float)

X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test, y_test_scaled = create_sequences(test_scaled, SEQ_LEN)
y_test = scaler.inverse_transform(y_test_scaled)

# --- Model Definition ---
def build_model():
    tf.random.set_seed(random.random() * 100)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(SEQ_LEN,)),
        tf.keras.layers.Dense(49, activation="relu"),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(29, activation="relu"),
        tf.keras.layers.Dropout(0.02),
        tf.keras.layers.Dense(21, activation="relu"),
        tf.keras.layers.Dropout(0.0),
        tf.keras.layers.Dense(1, activation="linear")
    ])
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR)
    )
    logging.info("Starting Modeltraining...")
    model.fit(X_train, y_train, epochs=EPOCHS, verbose=0)
    logging.info("Modeltraining ends.")
    return model

# --- Prediction ---
def predict_model(model):
    logging.info("Starting prediction on testing data...")
    predictions = model.predict(X_test)
    logging.info("Prediction finished.")
    return scaler.inverse_transform(predictions)

# --- Evaluation ---
def evaluate_model(predictions):
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    accuracy = 1 - mape
    logging.info(f"Evaluation: MAE={mae}, MAPE={mape}, Accuracy={accuracy}")
    return mae, mape, accuracy

# --- Run Multiple Models ---
def run_trials(n_trials):
    total_mae = total_mape = total_acc = 0
    final_predictions = None
    for i in range(n_trials):
        logging.info(f"Starting training run {i+1}/{n_trials}")
        model = build_model()
        preds = predict_model(model)
        mae, mape, acc = evaluate_model(preds)
        total_mae += mae
        total_mape += mape
        total_acc += acc
        final_predictions = preds  # Keep last model's predictions
    logging.info("All training runs finished.")
    return total_mae/n_trials, total_mape/n_trials, total_acc/n_trials, final_predictions

mae, mape, acc, predictions = run_trials(1)
logging.info(f"Mean Absolute Error = {mae}")
logging.info(f"Mean Absolute Percentage Error = {mape}")
logging.info(f"Accuracy = {acc}")
logging.debug(f"Predictions: {predictions.tolist()}")

# --- Plot Predictions ---
stock_df = pd.read_csv(DATA_PATH)
date_series = pd.to_datetime(stock_df[stock_df.columns[0]], utc=True)
actual_close = stock_df['Close'].values

prediction_dates = date_series[-len(predictions):]
logging.info("Plotting predictions vs actual data...")
plt.figure(figsize=(12, 6))
plt.plot(date_series, actual_close, label='Aktuelle Daten')
plt.plot(prediction_dates, predictions, label='Vorhersagen')
plt.xlabel('Zeit')
plt.ylabel('Aktienkurs')
plt.title('Aktienkursvorhersage vs. Aktuelle Daten')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()
logging.info("Plotting finished.")

# --- Predict 90 Days Into the Future ---
logging.info("Starting 90-day forcast...")
last_seq = test_scaled[-SEQ_LEN:].flatten()
future_preds = []
future_dates = []
last_known_date = date_series.iloc[-1]

future_model = build_model()
for i in range(FUTURE_DAYS):
    input_seq = last_seq.reshape(1, -1)
    next_val = future_model.predict(input_seq)  # DON'T TOUCH THIS LINE - CRUCIAL
    next_val_inv = scaler.inverse_transform(next_val)[0, 0]
    future_preds.append(next_val_inv)

    last_known_date += pd.Timedelta(days=1)
    future_dates.append(last_known_date)
    last_seq = np.append(last_seq[1:], scaler.transform([[next_val_inv]]))
logging.info("90-day forcast completed.")
plt.figure(figsize=(12, 6))
plt.plot(date_series, actual_close, label='Aktuelle Daten')
plt.plot(prediction_dates, predictions, label='Vorhersagen')
plt.plot(future_dates, future_preds, label='90 Tage Prognose', linestyle='dashed')
plt.xlabel('Zeit')
plt.ylabel('Aktienkurs')
plt.title('Aktienkursvorhersage vs. Aktuelle Daten')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()
logging.info("Plot for 90-day forcast completed.")
