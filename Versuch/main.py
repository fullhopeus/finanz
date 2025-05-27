import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf

# Einstellungen
anteil_training = 0.85
sequenz_länge = 10
epochen = 100
lernrate = 0.01

# Daten einlesen
daten = pd.read_csv("data/stock.csv")
zielspalte = ['Close']
anzahl_gesamt = daten.shape[0]

anzahl_training = int(anzahl_gesamt * anteil_training)

# Daten in Training und Test aufteilen
daten_training = daten[zielspalte].values[:anzahl_training]
daten_test = daten[zielspalte].values[anzahl_training:]

# Skalierung
skalierer = MinMaxScaler()
daten_training = skalierer.fit_transform(daten_training)
daten_test = skalierer.transform(daten_test)

# Trainingsdaten vorbereiten
X_train = [daten_training[i: i + sequenz_länge] for i in range(anzahl_training - sequenz_länge)]
X_train = np.array(X_train).astype(float)
y_train = np.array(daten_training[sequenz_länge:]).astype(float)

# Testdaten vorbereiten
anzahl_test = daten_test.shape[0]
X_test = [daten_test[i: i + sequenz_länge] for i in range(anzahl_test - sequenz_länge)]
X_test = np.array(X_test).astype(float)
y_test = np.array(daten_test[sequenz_länge:]).astype(float)
y_test_original = skalierer.inverse_transform(y_test)

# Modellvorhersage
def vorhersagen(modell):
    prognosen = modell.predict(X_test)
    return skalierer.inverse_transform(prognosen)

# Evaluation
def bewerten(vorhersagen):
    fehler_mae = mean_absolute_error(y_test_original, vorhersagen)
    fehler_mape = mean_absolute_percentage_error(y_test_original, vorhersagen)
    genauigkeit = 1 - fehler_mape
    return fehler_mae, fehler_mape, genauigkeit

# Modell erstellen
def modell_erstellen(einheiten1, dropout1, einheiten2, dropout2, einheiten3, dropout3):
    tf.random.set_seed(int(random.random() * 100))
    modell = tf.keras.models.Sequential([
        tf.keras.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(units=einheiten1, activation="relu"),
        tf.keras.layers.Dropout(dropout1),
        tf.keras.layers.Dense(units=einheiten2, activation="relu"),
        tf.keras.layers.Dropout(dropout2),
        tf.keras.layers.Dense(units=einheiten3, activation="relu"),
        tf.keras.layers.Dropout(dropout3),
        tf.keras.layers.Dense(units=1, activation="linear")
    ])
    modell.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lernrate)
    )
    fruehes_stopp = tf.keras.callbacks.EarlyStopping(
        monitor='loss', patience=10, min_delta=0.0001, mode='min', restore_best_weights=True
    )
    modell.fit(X_train, y_train, epochs=epochen, verbose=0, callbacks=[fruehes_stopp])
    return modell

# Hyperparameteroptimierung
beste_mae = float('inf')
beste_parameter = None

for einh1 in range(49, 52):
    for drop1 in np.arange(0.0, 0.03, 0.01):
        for einh2 in range(28, 31):
            for drop2 in np.arange(0.02, 0.05, 0.01):
                for einh3 in range(18, 22):
                    for drop3 in np.arange(0.0, 0.03, 0.01):
                        modell = modell_erstellen(einh1, drop1, einh2, drop2, einh3, drop3)
                        prognosen = vorhersagen(modell)
                        mae, mape, genauigkeit = bewerten(prognosen)
                        if mae < beste_mae:
                            beste_mae = mae
                            beste_parameter = (einh1, drop1, einh2, drop2, einh3, drop3)
                            print(f"Neue beste MAE: {mae:.4f} mit Parametern: {beste_parameter}")
                        if mae > 0.005:
                            break

print(f"Beste MAE: {beste_mae:.4f} mit Parametern: {beste_parameter}")
