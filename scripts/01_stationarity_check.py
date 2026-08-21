"""
Task 4 - Step 1: Stationarity Check + Seasonal Differencing
Progree Data Science Internship

WHAT IS "STATIONARITY"?
A time series is "stationary" if its statistical properties (mean, variance)
stay roughly constant over time - no upward/downward trend, no changing
seasonal swings. Most forecasting models (like ARIMA/SARIMA) ASSUME the
data is stationary, or they need to be told how much "differencing" to
apply to make it stationary first. So checking this is a required step
before building the model.
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------
df = pd.read_csv("data/airline_passengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)
ts = df["Passengers"]

print("Data loaded. Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# STEP 2: Plot the raw series
# Just LOOKING at the plot already tells us a lot:
# - Is there a trend (line going up/down overall)?
# - Is there seasonality (repeating pattern every 12 months)?
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(ts, color="steelblue")
plt.title("Monthly Airline Passengers (1949-1960) - Raw Data")
plt.xlabel("Year")
plt.ylabel("Passengers (thousands)")
plt.grid(alpha=0.3)
plt.savefig("outputs/raw_series_plot.png", dpi=150, bbox_inches="tight")
print("\nSaved raw_series_plot.png")

# ---------------------------------------------------------
# STEP 3: Seasonal Decomposition
# This splits the series into 3 parts:
# - Trend: the long-term overall direction
# - Seasonal: the repeating yearly pattern
# - Residual: whatever randomness is left over
# ---------------------------------------------------------
decomposition = seasonal_decompose(ts, model="multiplicative", period=12)
fig = decomposition.plot()
fig.set_size_inches(10, 8)
fig.savefig("outputs/seasonal_decomposition.png", dpi=150, bbox_inches="tight")
print("Saved seasonal_decomposition.png")

# ---------------------------------------------------------
# STEP 4: Augmented Dickey-Fuller (ADF) Test
# This is the FORMAL statistical test for stationarity.
#
# HOW TO READ THE RESULT (important for exams):
# - Null Hypothesis (H0): the series is NON-stationary
# - If p-value < 0.05  -> REJECT H0 -> series IS stationary
# - If p-value >= 0.05 -> FAIL to reject H0 -> series is NOT stationary
# ---------------------------------------------------------
def run_adf_test(series, label):
    result = adfuller(series.dropna())
    print(f"\n--- ADF Test: {label} ---")
    print(f"ADF Statistic : {result[0]:.4f}")
    print(f"p-value       : {result[1]:.4f}")
    print(f"Critical Values: {result[4]}")
    if result[1] < 0.05:
        print(">> Result: Series IS stationary (reject H0)")
    else:
        print(">> Result: Series is NOT stationary (fail to reject H0)")
    return result[1]

p_raw = run_adf_test(ts, "Raw Series")

# ---------------------------------------------------------
# STEP 5: Apply Seasonal Differencing
# "Differencing" means subtracting the value from a previous period
# to remove trend/seasonality.
#
# - First-order differencing: today - yesterday (removes trend)
# - Seasonal differencing (lag=12 for monthly data): today - same
#   month last year (removes yearly seasonal pattern)
# ---------------------------------------------------------
ts_diff1 = ts.diff(1).dropna()               # regular differencing (removes trend)
ts_seasonal_diff = ts.diff(12).dropna()       # seasonal differencing (removes yearly cycle)
ts_both_diff = ts.diff(1).diff(12).dropna()   # both combined

p_diff1 = run_adf_test(ts_diff1, "After 1st-order Differencing")
p_seasonal = run_adf_test(ts_seasonal_diff, "After Seasonal (lag=12) Differencing")
p_both = run_adf_test(ts_both_diff, "After Both (trend + seasonal) Differencing")

# ---------------------------------------------------------
# STEP 6: Plot the differenced series for visual comparison
# ---------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
axes[0].plot(ts_diff1, color="darkorange")
axes[0].set_title("After 1st-order Differencing (removes trend)")
axes[1].plot(ts_seasonal_diff, color="seagreen")
axes[1].set_title("After Seasonal Differencing, lag=12 (removes yearly cycle)")
axes[2].plot(ts_both_diff, color="crimson")
axes[2].set_title("After Both Differencing Steps (trend + season removed)")
for ax in axes:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/differencing_comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved differencing_comparison.png")

# Save a plain-text summary of results for the whitepaper later
with open("outputs/stationarity_test_results.txt", "w") as f:
    f.write("STATIONARITY TEST SUMMARY (ADF Test)\n")
    f.write("=====================================\n")
    f.write(f"Raw series p-value: {p_raw:.4f} -> {'Stationary' if p_raw < 0.05 else 'Non-stationary'}\n")
    f.write(f"After 1st-order differencing p-value: {p_diff1:.4f} -> {'Stationary' if p_diff1 < 0.05 else 'Non-stationary'}\n")
    f.write(f"After seasonal differencing (lag=12) p-value: {p_seasonal:.4f} -> {'Stationary' if p_seasonal < 0.05 else 'Non-stationary'}\n")
    f.write(f"After both differencing steps p-value: {p_both:.4f} -> {'Stationary' if p_both < 0.05 else 'Non-stationary'}\n")

print("\nAll results saved to outputs/stationarity_test_results.txt")
