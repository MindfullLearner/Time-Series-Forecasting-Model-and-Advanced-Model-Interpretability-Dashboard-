#  Time-Series Forecasting + Model Interpretability

## Folder structure
```
task4/
├── data/
│   └── airline_passengers.csv
├── outputs/                        (created automatically when scripts run)
├── 01_stationarity_check.py
├── 02_sarima_forecast.py
├── 03_interpretability_shap.py
└── requirements.txt
```

## Setup (run once)

```bash
pip install -r requirements.txt
```

## How to run (run in this exact order, from inside the `task4` folder)

```bash
cd task4
python 01_stationarity_check.py
python 02_sarima_forecast.py
python 03_interpretability_shap.py
```

Each script prints its results to the terminal AND saves plots/CSVs into the `outputs/` folder.

## What each script does

1. **01_stationarity_check.py**
   - Loads and plots the raw series
   - Decomposes it into trend / seasonal / residual
   - Runs the ADF stationarity test on raw data and after differencing
   - Saves: `raw_series_plot.png`, `seasonal_decomposition.png`, `differencing_comparison.png`, `stationarity_test_results.txt`

2. **02_sarima_forecast.py**
   - Splits data into train/test (last 24 months held out)
   - Fits a SARIMA(1,1,1)(1,1,1,12) model
   - Forecasts the test period with 95% confidence intervals, computes MAE/RMSE
   - Refits on full data and forecasts 24 months into the true future
   - Saves: `forecast_plot.png`, `future_forecast_plot.png`, `forecast_vs_actual.csv`, `future_forecast.csv`, `model_metrics.txt`

3. **03_interpretability_shap.py**
   - Builds a separate feature-based Random Forest model (month, year, lag_1, lag_12, rolling_mean_3) since SARIMA itself has no "features" for SHAP to explain
   - Applies SHAP to rank which features matter most
   - Saves: `shap_summary_plot.png`, `shap_importance_bar.png`, `shap_single_prediction_waterfall.png`, `shap_feature_importance.csv`

## Notes
- If `01_stationarity_check.py` or `02_sarima_forecast.py` complain about a missing `outputs/` folder, just create it manually first: `mkdir outputs`
- All scripts assume they're run from inside the `task4/` folder (they use relative paths like `data/airline_passengers.csv`)
- Dataset source: https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv
