import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
import yfinance as yf

split = 0.85
sequence_length = 10
epochs = 100
learning_rate = 0.01

stock_data = pd.read_csv("data/stock.csv")
column = ['Close']
len_stock_data = stock_data.shape[0]

train_examples = int(len_stock_data * split)
train = stock_data.get(column).values[:train_examples]
test = stock_data.get(column).values[train_examples:]
len_train = train.shape[0]
len_test = test.shape[0]

scaler = MinMaxScaler()
train, test = scaler.fit_transform(train), scaler.fit_transform(test)

X_train = [train[i : i + sequence_length] for i in range(len_train - sequence_length)]
X_train = np.array(X_train).astype(float)
y_train = np.array(train[sequence_length:]).astype(float)

X_test = [test[i : i + sequence_length] for i in range(len_test - sequence_length)]
X_test = np.array(X_test).astype(float)
y_test = np.array(test[sequence_length:]).astype(float)

y_test = scaler.inverse_transform(y_test)

def predict(model):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

def evaluate(predictions):
    mae = mean_absolute_error(predictions, y_test)
    mape = mean_absolute_percentage_error(predictions, y_test)
    return mae, mape, (1 - mape)

def model_create(units1, dropout1, units2, dropout2, units3, dropout3):
    tf.random.set_seed(random.random() * 100)
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(units=units1, activation="relu"),
        tf.keras.layers.Dropout(dropout1),
        tf.keras.layers.Dense(units=units2, activation="relu"),
        tf.keras.layers.Dropout(dropout2),
        tf.keras.layers.Dense(units=units3, activation="relu"),
        tf.keras.layers.Dropout(dropout3),
        tf.keras.layers.Dense(units=1, activation="linear")
    ])
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, min_delta=0.0001, mode='min', restore_best_weights=True)
    model.fit(X_train, y_train, epochs=epochs, verbose=0, callbacks=[early_stop])
    return model

best_loss = float('inf')
best_params = None

for u1 in range(47, 53, 1):
    for d1 in np.arange(0.0, 0.3, 0.1):
        for u2 in range(28, 32, 1):
            for d2 in np.arange(0.03, 0.07, 0.01):
                for u3 in range(18, 22, 1):
                    for d3 in np.arange(0.0, 0.02, 0.01):
                        model = model_create(u1, d1, u2, d2, u3, d3)
                        predictions = predict(model)
                        mae, mape, acc = evaluate(predictions)
                        if mae < best_loss:
                            best_loss = mae
                            best_params = (u1, d1, u2, d2, u3, d3)
                            print(f"New best MAE: {mae:.4f} with params: {best_params}")
                        if mae > 0.005:
                            break

print(f"Best MAE: {best_loss:.4f} with parameters: {best_params}")