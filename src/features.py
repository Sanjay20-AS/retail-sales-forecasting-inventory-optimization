"""Feature engineering that keeps each store-SKU time series separate."""
import pandas as pd

GROUP_COLS = ["store_id", "sku_id"]

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(GROUP_COLS + ["date"]).reset_index(drop=True)
    out["day_of_week"] = out["date"].dt.dayofweek
    out["month"] = out["date"].dt.month
    out["day_of_year"] = out["date"].dt.dayofyear
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["day_of_month"] = out["date"].dt.day
    grouped = out.groupby(GROUP_COLS, sort=False)["units_sold"]
    out["lag_1"] = grouped.shift(1)
    out["lag_7"] = grouped.shift(7)
    out["rolling_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=7).mean())
    out["rolling_mean_28"] = grouped.transform(lambda s: s.shift(1).rolling(28, min_periods=28).mean())
    out["rolling_std_28"] = grouped.transform(lambda s: s.shift(1).rolling(28, min_periods=28).std())
    return out.dropna(subset=["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_28", "rolling_std_28"]).reset_index(drop=True)

FEATURE_COLUMNS = [
    "unit_price", "promotion", "day_of_week", "month", "day_of_year",
    "is_weekend", "day_of_month", "lag_1", "lag_7",
    "rolling_mean_7", "rolling_mean_28", "rolling_std_28"
]
