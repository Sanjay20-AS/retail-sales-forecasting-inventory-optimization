"""Run the end-to-end retail forecasting and inventory pipeline."""
from pathlib import Path
import pandas as pd
from src.data_validation import validate_sales_data
from src.features import add_time_features
from src.forecasting import train_and_evaluate
from src.inventory import build_recommendations

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "retail_sales.csv"
REPORTS = ROOT / "reports"

def main():
    REPORTS.mkdir(exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found. Run: python generate_data.py")
    raw = pd.read_csv(DATA_PATH)
    validate_sales_data(raw)
    featured = add_time_features(raw)
    model, train, test, metrics, split_date = train_and_evaluate(featured)
    # Model forecasts are evaluated on a later time period.
    metrics.to_csv(REPORTS / "model_metrics.csv", index=False)
    forecasts = test[["date", "store_id", "sku_id", "category", "units_sold",
                      "rf_forecast", "baseline_forecast", "on_hand_units"]].copy()
    forecasts["date"] = forecasts["date"].astype(str)
    forecasts.to_csv(REPORTS / "test_forecasts.csv", index=False)
    recommendations = build_recommendations(train, test)
    recommendations.to_csv(REPORTS / "inventory_recommendations.csv", index=False)
    summary = raw.groupby(["store_id", "sku_id", "category"], as_index=False).agg(
        total_units_sold=("units_sold", "sum"),
        avg_daily_units=("units_sold", "mean"),
        average_price=("unit_price", "mean"),
        promotion_days=("promotion", "sum"),
        latest_on_hand=("on_hand_units", "last")
    )
    summary.to_csv(REPORTS / "sku_summary.csv", index=False)
    print(f"Chronological split starts: {pd.Timestamp(split_date).date()}")
    print("\nModel metrics (lower is better):")
    print(metrics.to_string(index=False))
    print(f"\nSaved reports to: {REPORTS}")

if __name__ == "__main__":
    main()
