import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
import yfinance as yf
def vorhersage_mlp(ticker):
    end = pd.to_datetime('today').strftime('%Y-%m-%d')
    stock_data = yf.download(ticker, "2000-01-01", end, auto_adjust=False)
    df = pd.DataFrame(stock_data)
    df.to_csv("data/stock.csv")
    df = pd.read_csv("data/stock.csv")
    # -------BUG FIX-------
    with open("data/stock.csv", 'r') as f:
        lines = f.readlines()
    with open("data/stock.csv", 'w') as f:
        for i, line in enumerate(lines):
            if i != 1 and i != 2:
                # Uberspringen der ersten 2 Zeilen (Mit Ticker und Date)
                f.write(line)
#vorhersage_mlp("AAPL")
split = (0.85)
sequence_length = 10
epochs = 100
learning_rate = 0.01

# loading stock price data
stock_data = pd.read_csv("data/stock.csv")
column = ['Close']

len_stock_data = stock_data.shape[0]


# splitting data to train and test
train_examples = int(len_stock_data * split)
train = stock_data.get(column).values[:train_examples]
test = stock_data.get(column).values[train_examples:]
len_train = train.shape[0]
len_test = test.shape[0]


# normalizing data
scaler = MinMaxScaler()
train, test = scaler.fit_transform(train), scaler.fit_transform(test)


# splitting training data to x and y
X_train = []
for i in range(len_train - sequence_length):
    X_train.append(train[i : i + sequence_length])
X_train = np.array(X_train).astype(float)
y_train = np.array(train[sequence_length:]).astype(float)

# splitting testing data to x and y
X_test = []
for i in range(len_test - sequence_length):
    X_test.append(test[i : i + sequence_length])
X_test = np.array(X_test).astype(float)
y_test = np.array(test[sequence_length:]).astype(float)


#creating MLP model
def model_create():
    tf.random.set_seed(random.random() * 100)
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input(shape = (X_train.shape[1],)),
            tf.keras.layers.Dense(units = 50, activation = "relu"),
            tf.keras.layers.Dropout(0.0),
            tf.keras.layers.Dense(units = 29, activation = "relu"),
            tf.keras.layers.Dropout(0.04),
            tf.keras.layers.Dense(units = 18, activation = "relu"),
            tf.keras.layers.Dropout(0.0),
            tf.keras.layers.Dense(units = 1, activation = "linear")
        ]
    )

    model.compile(
        loss = tf.keras.losses.MeanSquaredError,
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
    )


    model.fit(
        X_train, y_train,
        epochs = epochs,
    )
    return model


# inverting normaliztion
y_test = scaler.inverse_transform(y_test)



# prediction on test set
def predict(model):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions


# evaluation
def evaluate(predictions):
    mae = mean_absolute_error(predictions, y_test)
    mape = mean_absolute_percentage_error(predictions, y_test)
    return mae, mape, (1 - mape)


# trial runs
def run_model(n):
    total_mae = total_mape = total_acc = 0
    for i in range(n):
        model = model_create()
        predictions = predict(model)
        mae, mape, acc = evaluate(predictions)
        total_mae += mae
        total_mape += mape 
        total_acc += acc 
    return (total_mae / n), (total_mape / n), (total_acc / n), predictions.tolist()

mae, mape, acc, preds = run_model(1)

import matplotlib.pyplot as plt

print(f"Mean Absolute Error = {mae}")
print(f"Mean Absolute Percentage Error = {mape}%")
print(f"Accuracy = {acc}")
print(preds)
# Plot predictions vs actual data
stock_df = pd.read_csv("data/stock.csv")
dates = pd.to_datetime(stock_df['Price'])
actual = stock_df['Close'].values

# Die letzten len(preds) Datumswerte für die Vorhersage
pred_dates = dates[-len(preds):]
plt.figure(figsize=(12, 6))
plt.plot(dates, actual, label='Aktuelle Daten')
plt.plot(pred_dates, preds, label='Vorhersagen')
plt.xlabel('Zeit')
plt.ylabel('Aktienkurs')
plt.title('Aktienkursvorhersage vs. Aktuelle Daten')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()


# --- 90 Tage in die Zukunft vorhersagen ---
future_steps = 90
last_sequence = test[-sequence_length:].flatten()
future_preds = []
future_dates = []
last_date = dates.iloc[-1]
for i in range(future_steps):
    input_seq = last_sequence.reshape(1, -1)
    next_pred = model_create().predict(input_seq)
    next_pred_inv = scaler.inverse_transform(next_pred)[0, 0]
    future_preds.append(next_pred_inv)
    # Datum um einen Tag erhöhen (Börsentage beachten wir hier nicht)
    last_date += pd.Timedelta(days=1)
    future_dates.append(last_date)
    # Update sequence
    last_sequence = np.append(last_sequence[1:], scaler.transform([[next_pred_inv]]))

plt.figure(figsize=(12, 6))
plt.plot(dates, actual, label='Aktuelle Daten')
plt.plot(pred_dates, preds, label='Vorhersagen')
plt.plot(future_dates, future_preds, label='90 Tage Prognose', linestyle='dashed')
plt.xlabel('Zeit')
plt.ylabel('Aktienkurs')
plt.title('Aktienkursvorhersage vs. Aktuelle Daten')
plt.legend()
plt.gcf().autofmt_xdate()
plt.show()