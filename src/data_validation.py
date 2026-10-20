"""Basic quality checks for retail sales input data."""
import pandas as pd

REQUIRED_COLUMNS = {
    "date", "store_id", "sku_id", "category", "unit_price",
    "promotion", "units_sold", "on_hand_units", "lead_time_days"
}

def validate_sales_data(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Sales dataset is empty.")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Required columns contain missing values.")
    if (df["units_sold"] < 0).any():
        raise ValueError("units_sold cannot be negative.")
    if (df["unit_price"] <= 0).any():
        raise ValueError("unit_price must be positive.")
    if (df["lead_time_days"] < 1).any():
        raise ValueError("lead_time_days must be at least 1.")
    if not set(df["promotion"].unique()).issubset({0, 1}):
        raise ValueError("promotion must contain only 0 or 1.")
