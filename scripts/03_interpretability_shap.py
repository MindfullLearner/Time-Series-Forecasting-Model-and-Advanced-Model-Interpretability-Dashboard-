"""
Task 4 - Step 3: Model Interpretability using SHAP
Progree Data Science Internship

WHY DO WE NEED THIS STEP SEPARATELY FROM SARIMA?
SARIMA predicts the next value using ONLY the series' own past values
(it doesn't have separate "features" like income or temperature).
So there's nothing meaningful for SHAP to explain inside SARIMA itself.

To satisfy the "model interpretability wrapper (SHAP or LIME) to measure
exact feature importance weights" requirement, we build a SEPARATE
feature-based regression model (Random Forest) that predicts passenger
counts using engineered features such as:
    - month number (captures seasonality)
    - year (captures trend)
    - lag_1  (passengers 1 month ago)
    - lag_12 (passengers 12 months ago - same month last year)
    - rolling_mean_3 (average of last 3 months)

Then we use SHAP to explain WHICH of these features matters most when
predicting passenger counts - this is the "interpretability layer."
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------
df = pd.read_csv("data/airline_passengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.set_index("Month", inplace=True)

# ---------------------------------------------------------
# STEP 2: Feature Engineering
# We create new columns (features) from the raw series that a
# normal ML model (which doesn't understand "time" on its own)
# can actually learn from.
# ---------------------------------------------------------
df["month_num"] = df.index.month
df["year"] = df.index.year
df["lag_1"] = df["Passengers"].shift(1)
df["lag_12"] = df["Passengers"].shift(12)
df["rolling_mean_3"] = df["Passengers"].shift(1).rolling(window=3).mean()

df.dropna(inplace=True)  # drop early rows that don't have lag/rolling data yet

feature_cols = ["month_num", "year", "lag_1", "lag_12", "rolling_mean_3"]
X = df[feature_cols]
y = df["Passengers"]

print("Feature table preview:")
print(df[feature_cols + ["Passengers"]].head())

# ---------------------------------------------------------
# STEP 3: Train/test split and fit a Random Forest
# Random Forest is used here (not SARIMA) because SHAP works
# naturally with tree-based models and clearly shows feature
# importance per prediction.
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # shuffle=False keeps time order intact
)

rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train)

preds = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"\nRandom Forest Test MAE: {mae:.2f} passengers")

# ---------------------------------------------------------
# STEP 4: Apply SHAP to explain the model's predictions
# SHAP (SHapley Additive exPlanations) assigns each feature a
# "contribution score" for each prediction, based on game theory:
# it measures how much each feature pushes the prediction up or
# down compared to the average prediction.
# ---------------------------------------------------------
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# ---------------------------------------------------------
# STEP 5: Summary Plot
# Shows, across ALL test predictions, which features matter most
# overall (ranked top to bottom) and whether high/low values of
# that feature push predictions up or down.
# ---------------------------------------------------------
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_summary_plot.png", dpi=150, bbox_inches="tight")
print("Saved shap_summary_plot.png")

# ---------------------------------------------------------
# STEP 6: Bar plot of average absolute SHAP value per feature
# Simplest way to answer "which feature matters most overall?"
# ---------------------------------------------------------
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("outputs/shap_importance_bar.png", dpi=150, bbox_inches="tight")
print("Saved shap_importance_bar.png")

# ---------------------------------------------------------
# STEP 7: Explain ONE single prediction in detail (waterfall plot)
# Good for walking through "how did the model arrive at THIS
# specific prediction" - useful as a worked example in your report.
# ---------------------------------------------------------
sample_index = 0

# Handle both old and new SHAP versions where expected_value
# may be a single number or a 1-element array
base_value = explainer.expected_value
if hasattr(base_value, "__len__"):
    base_value = base_value[0]

plt.figure()
shap.plots._waterfall.waterfall_legacy(
    base_value,
    shap_values[sample_index],
    feature_names=feature_cols,
    show=False
)
plt.tight_layout()
plt.savefig("outputs/shap_single_prediction_waterfall.png", dpi=150, bbox_inches="tight")
print("Saved shap_single_prediction_waterfall.png")

# ---------------------------------------------------------
# STEP 8: Save mean absolute SHAP values as a table (for whitepaper)
# ---------------------------------------------------------
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Mean_Abs_SHAP_Value": mean_abs_shap
}).sort_values("Mean_Abs_SHAP_Value", ascending=False)

importance_df.to_csv("outputs/shap_feature_importance.csv", index=False)
print("\nFeature importance ranking:")
print(importance_df)
print("\nSaved shap_feature_importance.csv")
