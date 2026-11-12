# Time Series Exercise 8: Modelling and Forecasting with ARMA Processes

## Name: Aravindan T

## Roll No: 2305001003

## Department: CSE

## AIM

To model and forecast a time series using an ARMA(p,q) process.

## PROGRAM

### Python Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set random seed
np.random.seed(42)

# Define AR and MA parameters
ar = np.array([1, -0.5, -0.2])
ma = np.array([1, 0.4])

# Generate ARMA(2,1) time series
arma_process = ArmaProcess(ar, ma)
data = arma_process.generate_sample(nsample=200)
series = pd.Series(data)

# Plot simulated time series
plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("Simulated ARMA(2,1) Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# Plot ACF and PACF
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

plot_acf(series, lags=20, ax=ax[0])
plot_pacf(series, lags=20, ax=ax[1])

plt.show()

# Split data into training and testing sets
train_size = int(len(series) * 0.8)
train = series[:train_size]
test = series[train_size:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

# Fit AR(1) model
model_ar1 = ARIMA(train, order=(1, 0, 0))
fit_ar1 = model_ar1.fit()

# Fit MA(1) model
model_ma1 = ARIMA(train, order=(0, 0, 1))
fit_ma1 = model_ma1.fit()

# Fit ARMA(1,1) model
model_arma11 = ARIMA(train, order=(1, 0, 1))
fit_arma11 = model_arma11.fit()

# Fit ARMA(2,1) model
model_arma21 = ARIMA(train, order=(2, 0, 1))
fit_arma21 = model_arma21.fit()

# Compare model information criteria
print("\n--- Model Information Criteria Comparison ---")

print(f"AR(1)    | AIC: {fit_ar1.aic:.2f} | BIC: {fit_ar1.bic:.2f}")
print(f"MA(1)    | AIC: {fit_ma1.aic:.2f} | BIC: {fit_ma1.bic:.2f}")
print(f"ARMA(1,1)| AIC: {fit_arma11.aic:.2f} | BIC: {fit_arma11.bic:.2f}")
print(f"ARMA(2,1)| AIC: {fit_arma21.aic:.2f} | BIC: {fit_arma21.bic:.2f}")

# Forecast test data
forecast = fit_arma21.forecast(steps=len(test))

# Calculate forecast error metrics
mae = mean_absolute_error(test, forecast)
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)

print("\n--- Forecast Error Metrics for ARMA(2,1) ---")
print("Mean Absolute Error (MAE):", round(mae, 4))
print("Mean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))

# Plot actual and forecasted values
plt.figure(figsize=(12, 5))

plt.plot(train, label="Training Data")
plt.plot(test, label="Actual Test Data")
plt.plot(
    test.index,
    forecast,
    label="ARMA(2,1) Forecast",
    color="red"
)

plt.title("ARMA(2,1) Forecast vs Actual Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
```

## OUTPUT
<img width="940" height="365" alt="image" src="https://github.com/user-attachments/assets/4b6050c5-4f68-46f8-ab67-c172024d7be2" />
<img width="940" height="680" alt="image" src="https://github.com/user-attachments/assets/5bbe7fd5-ad94-45a1-a897-c8776ea03596" />
<img width="940" height="1114" alt="image" src="https://github.com/user-attachments/assets/2c891966-0cf5-48b2-915f-66ab277fac3f" />

## RESULT
Thus, the ARMA(2,1) model was successfully implemented using Python. The time series was modelled, forecast values were generated, and model performance was evaluated using MAE, MSE, and RMSE.

