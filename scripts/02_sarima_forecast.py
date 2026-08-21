"""
Task 4 - Step 2: SARIMA Forecasting Model with Confidence Intervals
Progree Data Science Internship

WHAT IS SARIMA?
SARIMA = Seasonal AutoRegressive Integrated Moving Average.
It's an extension of ARIMA that explicitly handles seasonal patterns
(like our yearly travel cycle). It has two sets of parameters:

  Non-seasonal:  (p, d, q)
    p = AutoRegressive order   -> how many past values predict the next value
    d = Differencing order     -> how many times we difference to remove trend
    q = Moving Average order   -> how many past forecast errors we use

  Seasonal:      (P, D, Q, s)
    Same idea as above, but applied at the seasonal lag (s = 12 for monthly data)
    D = seasonal differencing order (we found in Step 1 that lag=12 differencing
        was needed to reach stationarity, so D=1)

We'll train on most of the data, hold out the last 24 months to test
accuracy, then forecast forward with confidence intervals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------
df = pd.read_csv("data/airline_passengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
ts = df["Passengers"]
ts.index.freq = "MS"  # tell pandas this is Month-Start frequency data

# ---------------------------------------------------------
# STEP 2: Train/Test split
# We hold out the LAST 24 months as "test" data - the model never
# sees these during training. Then we check how close its forecast
# is to what actually happened. This is standard practice to prove
# a model actually generalizes, not just memorizes.
# ---------------------------------------------------------
train = ts[:-24]
test = ts[-24:]
print(f"Train size: {len(train)} months | Test size: {len(test)} months")

# ---------------------------------------------------------
# STEP 3: Fit the SARIMA model
# order=(1,1,1)          -> non-seasonal: p=1, d=1, q=1
# seasonal_order=(1,1,1,12) -> seasonal: P=1, D=1, Q=1, s=12
#
# We use d=1 and D=1 because Step 1 showed we needed BOTH regular AND
# seasonal (lag=12) differencing to reach stationarity.
# ---------------------------------------------------------
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
fitted_model = model.fit(disp=False)
print(fitted_model.summary())

# ---------------------------------------------------------
# STEP 4: Forecast over the test period (with confidence intervals)
# get_forecast() gives us both the point prediction AND the
# confidence interval (a range we're 95% confident the true value
# falls within).
# ---------------------------------------------------------
forecast_result = fitted_model.get_forecast(steps=len(test))
forecast_mean = forecast_result.predicted_mean
conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval

# ---------------------------------------------------------
# STEP 5: Evaluate accuracy against real held-out values
# MAE  = Mean Absolute Error -> average size of error, in same units as data
# RMSE = Root Mean Squared Error -> penalizes big errors more heavily
# ---------------------------------------------------------
mae = mean_absolute_error(test, forecast_mean)
rmse = np.sqrt(mean_squared_error(test, forecast_mean))
print(f"\nTest MAE : {mae:.2f} passengers")
print(f"Test RMSE: {rmse:.2f} passengers")

# ---------------------------------------------------------
# STEP 6: Plot actual vs forecast with confidence band
# ---------------------------------------------------------
plt.figure(figsize=(11, 6))
plt.plot(train.index, train, label="Training Data", color="steelblue")
plt.plot(test.index, test, label="Actual (Test)", color="black", linewidth=2)
plt.plot(test.index, forecast_mean, label="SARIMA Forecast", color="crimson", linestyle="--")
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                  color="crimson", alpha=0.15, label="95% Confidence Interval")
plt.title("SARIMA Forecast vs Actual - Airline Passengers")
plt.xlabel("Year")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("outputs/forecast_plot.png", dpi=150, bbox_inches="tight")
print("\nSaved forecast_plot.png")

# ---------------------------------------------------------
# STEP 7: Forecast BEYOND the dataset (future, unseen months)
# Now refit on the FULL dataset (train+test) and forecast forward
# 24 new months with no ground truth to compare - a true future prediction.
# ---------------------------------------------------------
full_model = SARIMAX(
    ts,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

future_steps = 24
future_forecast = full_model.get_forecast(steps=future_steps)
future_mean = future_forecast.predicted_mean
future_conf = future_forecast.conf_int(alpha=0.05)

plt.figure(figsize=(11, 6))
plt.plot(ts.index, ts, label="Historical Data", color="steelblue")
plt.plot(future_mean.index, future_mean, label="Future Forecast (next 24 months)", color="darkorange", linestyle="--")
plt.fill_between(future_mean.index, future_conf.iloc[:, 0], future_conf.iloc[:, 1],
                  color="darkorange", alpha=0.15, label="95% Confidence Interval")
plt.title("SARIMA Future Forecast - Next 24 Months Beyond Dataset")
plt.xlabel("Year")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("outputs/future_forecast_plot.png", dpi=150, bbox_inches="tight")
print("Saved future_forecast_plot.png")

# ---------------------------------------------------------
# STEP 8: Save forecast values to CSV (useful for whitepaper appendix)
# ---------------------------------------------------------
results_df = pd.DataFrame({
    "Actual": test,
    "Forecast": forecast_mean,
    "Lower_CI": conf_int.iloc[:, 0],
    "Upper_CI": conf_int.iloc[:, 1]
})
results_df.to_csv("outputs/forecast_vs_actual.csv")
print("Saved forecast_vs_actual.csv")

future_df = pd.DataFrame({
    "Forecast": future_mean,
    "Lower_CI": future_conf.iloc[:, 0],
    "Upper_CI": future_conf.iloc[:, 1]
})
future_df.to_csv("outputs/future_forecast.csv")
print("Saved future_forecast.csv")

with open("outputs/model_metrics.txt", "w") as f:
    f.write("SARIMA MODEL EVALUATION\n")
    f.write("========================\n")
    f.write(f"Order: (1,1,1), Seasonal Order: (1,1,1,12)\n")
    f.write(f"Test MAE : {mae:.2f}\n")
    f.write(f"Test RMSE: {rmse:.2f}\n")
    f.write(f"AIC: {fitted_model.aic:.2f}\n")
    f.write(f"BIC: {fitted_model.bic:.2f}\n")
print("Saved model_metrics.txt")
