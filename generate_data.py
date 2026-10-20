"""Generate reproducible synthetic retail daily demand data."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
OUTPUT = Path(__file__).resolve().parent / "data" / "retail_sales.csv"

def generate_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    products = [
        ("SKU-1001", "Beverages", 42, 1.8, 0.22, 3),
        ("SKU-1002", "Snacks", 35, 2.4, 0.18, 4),
        ("SKU-1003", "Household", 18, 6.5, 0.12, 7),
        ("SKU-1004", "Personal Care", 22, 4.2, 0.15, 5),
        ("SKU-1005", "Dairy", 48, 1.4, 0.25, 2),
        ("SKU-1006", "Packaged Food", 31, 3.1, 0.16, 4),
        ("SKU-1007", "Stationery", 14, 2.8, 0.10, 6),
        ("SKU-1008", "Pet Care", 11, 7.2, 0.13, 8),
    ]
    rows = []
    for store_num in range(1, 4):
        store = f"STORE-{store_num:02d}"
        store_factor = [0.88, 1.0, 1.16][store_num - 1]
        for sku, category, base_demand, price, promo_lift, lead_time in products:
            sku_offset = rng.uniform(-0.05, 0.05)
            for i, date in enumerate(dates):
                dow = date.dayofweek
                weekend_factor = 1.18 if dow >= 5 else 1.0
                annual = 1 + 0.12 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                month_end = 1.10 if date.is_month_end else 1.0
                promotion = int(rng.random() < (0.10 if date.month not in [11, 12] else 0.17))
                effective_price = round(price * (0.90 if promotion else 1.0), 2)
                trend = 1 + 0.00025 * i
                expected = base_demand * store_factor * (1 + sku_offset) * weekend_factor * annual * month_end * trend
                expected *= (1 + promo_lift * promotion)
                expected *= (price / effective_price) ** 0.25
                units = int(rng.poisson(max(expected, 0.1)))
                # Inventory is a simulated snapshot, not a real inventory ledger.
                on_hand = max(0, int(rng.normal(expected * (lead_time + 3), max(2, expected * 1.5))))
                rows.append((date.date().isoformat(), store, sku, category, effective_price,
                             promotion, units, on_hand, lead_time))
    return pd.DataFrame(rows, columns=[
        "date", "store_id", "sku_id", "category", "unit_price",
        "promotion", "units_sold", "on_hand_units", "lead_time_days"
    ])

if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = generate_data()
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df):,} rows to {OUTPUT}")
