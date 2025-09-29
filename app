import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from prophet import Prophet

# -----------------------------
# 1. Download Apple stock data
# -----------------------------
print("Downloading Apple stock data...")
data = yf.download("AAPL", start="2015-01-01", end="2025-01-01")[["Close"]]
data = data.rename(columns={"Close": "Price"})
data.dropna(inplace=True)

# Ensure datetime index and fill missing business days
data.index = pd.to_datetime(data.index)
data = data.asfreq('B')
data['Price'].interpolate(method='linear', inplace=True)

# -----------------------------
# 2. Train/Test split
# -----------------------------
train_size = int(len(data) * 0.9)
train, test = data.iloc[:train_size], data.iloc[train_size:]

train_log = np.log(train["Price"])
test_log = np.log(test["Price"])

# -----------------------------
# 3. ARIMA model
# -----------------------------
print("Training ARIMA model...")
arima_model = pm.auto_arima(
    train_log,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action="ignore",
    max_p=5,
    max_q=5,
    d=None
)

arima_forecast_log, arima_conf_log = arima_model.predict(n_periods=len(test_log), return_conf_int=True)
arima_forecast = np.exp(arima_forecast_log.values)  # convert Series to array
arima_conf_lower = np.exp(arima_conf_log[:, 0])
arima_conf_upper = np.exp(arima_conf_log[:, 1])

# -----------------------------
# 4. Prophet model
# -----------------------------
print("Training Prophet model...")
# Prepare Prophet DataFrame
prophet_df = train.reset_index()[['Price']].copy()
prophet_df['ds'] = train.index  # datetime column
prophet_df['y'] = prophet_df['Price'].astype(float)  # numeric column
prophet_df = prophet_df[['ds', 'y']]

# Train Prophet model
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_df)

# Forecast with Prophet
future = prophet_model.make_future_dataframe(periods=len(test), freq='B')
forecast = prophet_model.predict(future)

prophet_forecast = forecast['yhat'].values[-len(test):].flatten()
prophet_lower = forecast['yhat_lower'].values[-len(test):].flatten()
prophet_upper = forecast['yhat_upper'].values[-len(test):].flatten()

# -----------------------------
# 5. Evaluation (RMSE & MAE)
# -----------------------------
y_true = test["Price"].values
prophet_forecast = forecast['yhat'].values[-len(test):]
prophet_lower = forecast['yhat_lower'].values[-len(test):]
prophet_upper = forecast['yhat_upper'].values[-len(test):]


arima_rmse = np.sqrt(mean_squared_error(y_true, arima_forecast))
arima_mae = mean_absolute_error(y_true, arima_forecast)

prophet_rmse = np.sqrt(mean_squared_error(y_true, prophet_forecast))
prophet_mae = mean_absolute_error(y_true, prophet_forecast)

print(f"ARIMA RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}")
print(f"Prophet RMSE: {prophet_rmse:.2f}, MAE: {prophet_mae:.2f}")

# -----------------------------
# 5b. Accuracy using MAPE
# -----------------------------
arima_mape = np.mean(np.abs((y_true - arima_forecast) / y_true)) * 100
prophet_mape = np.mean(np.abs((y_true - prophet_forecast) / y_true)) * 100

arima_accuracy = 100 - arima_mape
prophet_accuracy = 100 - prophet_mape

print(f"ARIMA Accuracy: {arima_accuracy:.2f}%")
print(f"Prophet Accuracy: {prophet_accuracy:.2f}%")

# -----------------------------
# 6. Plot comparison
# -----------------------------
plt.figure(figsize=(14,7))
plt.plot(train.index, train["Price"], label="Train")
plt.plot(test.index, test["Price"], label="Test")
plt.plot(test.index, arima_forecast, label="ARIMA Forecast", color="red")
plt.fill_between(test.index, arima_conf_lower, arima_conf_upper, color='pink', alpha=0.3)
plt.plot(test.index, prophet_forecast, label="Prophet Forecast", color="green")
plt.fill_between(test.index, prophet_lower, prophet_upper, color='lightgreen', alpha=0.3)
plt.title("Apple Stock Price Prediction: ARIMA vs Prophet")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

