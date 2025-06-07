import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

TICKER = "GOOGL"
DATA_PATH = f"data/{TICKER}.csv"
df = pd.read_csv(DATA_PATH)

def vorhersage_prophet(b):
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['index'], utc=True).dt.tz_localize(None)  # Zeitzone entfernen
    df['y'] = df['Close']
    df = df[['ds', 'y']]
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    vorhersage = model.predict(future)
    vorhersage = vorhersage[vorhersage['ds'] > df['ds'].max()]
    # return vorhersage

    img = model.plot(vorhersage)
    plt.show()

    img = model.plot_components(vorhersage)
    plt.show()

vorhersage_prophet()