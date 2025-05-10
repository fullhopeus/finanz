import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

df = pd.read_csv('data/yf_data.csv')
df['ds'] = pd.to_datetime(df['Date'])
df['y'] = df['Close']
df = df[['ds', 'y']]
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=30)
vorhersage = model.predict(future)
vorhersage = vorhersage[vorhersage['ds'] > df['ds'].max()]
img = model.plot(vorhersage)
plt.show()

img = model.plot_components(vorhersage)
plt.show()