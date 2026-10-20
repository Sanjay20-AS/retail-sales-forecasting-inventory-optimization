"""Forecast model training and evaluation utilities."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .features import FEATURE_COLUMNS

def calculate_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    denominator = np.abs(actual).sum()
    wape = float(np.abs(actual - predicted).sum() / denominator * 100) if denominator else float("nan")
    return {"MAE": float(mae), "RMSE": rmse, "WAPE_percent": wape}

def train_and_evaluate(features: pd.DataFrame):
    """Chronological split; returns model, forecasts, metrics, and training data."""
    dates = np.sort(features["date"].unique())
    split_date = dates[int(len(dates) * 0.80)]
    train = features[features["date"] < split_date].copy()
    test = features[features["date"] >= split_date].copy()
    if train.empty or test.empty:
        raise ValueError("Not enough data for a chronological train/test split.")
    model = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    model.fit(train[FEATURE_COLUMNS], train["units_sold"])
    test["rf_forecast"] = np.maximum(0, model.predict(test[FEATURE_COLUMNS]))
    test["baseline_forecast"] = test["rolling_mean_7"].clip(lower=0)
    metrics = []
    for model_name, col in [("RandomForest", "rf_forecast"), ("MovingAverage7", "baseline_forecast")]:
        metrics.append({"model": model_name, **calculate_metrics(test["units_sold"], test[col])})
    return model, train, test, pd.DataFrame(metrics), split_date
