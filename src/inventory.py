"""Transparent inventory planning calculations."""
import math
import numpy as np
import pandas as pd

def calculate_inventory_policy(avg_daily_demand: float, residual_std: float,
                               lead_time_days: int, review_period_days: int = 7,
                               service_z: float = 1.65, current_stock: float = 0) -> dict:
    """Calculate safety stock, reorder point, target stock, and suggested order."""
    if avg_daily_demand < 0 or residual_std < 0:
        raise ValueError("Demand and residual standard deviation must be non-negative.")
    if lead_time_days < 1 or review_period_days < 1:
        raise ValueError("Lead time and review period must be at least one day.")
    if service_z < 0:
        raise ValueError("service_z must be non-negative.")
    safety_stock = service_z * residual_std * math.sqrt(lead_time_days)
    reorder_point = avg_daily_demand * lead_time_days + safety_stock
    target_stock = avg_daily_demand * (lead_time_days + review_period_days) + safety_stock
    order_qty = max(0, math.ceil(target_stock - max(0, current_stock)))
    return {
        "avg_daily_demand": round(float(avg_daily_demand), 2),
        "residual_std": round(float(residual_std), 2),
        "safety_stock": int(math.ceil(safety_stock)),
        "reorder_point": int(math.ceil(reorder_point)),
        "target_stock": int(math.ceil(target_stock)),
        "current_stock": int(max(0, current_stock)),
        "recommended_order_qty": int(order_qty),
        "reorder_now": bool(current_stock <= reorder_point),
    }

def build_recommendations(train: pd.DataFrame, test: pd.DataFrame,
                          review_period_days: int = 7, service_z: float = 1.65) -> pd.DataFrame:
    """Build SKU-store recommendations using training history and last observed stock."""
    rows = []
    keys = ["store_id", "sku_id", "category"]
    for (store, sku, category), hist in train.groupby(keys):
        test_group = test[(test.store_id == store) & (test.sku_id == sku)]
        if hist.empty:
            continue
        avg_demand = float(hist["units_sold"].mean())
        residual_std = float(hist["units_sold"].std(ddof=1) or 0)
        if not test_group.empty:
            residuals = test_group["units_sold"] - test_group["rf_forecast"]
            # This is holdout error for reporting only; use training variability for policy.
            # Use historical daily demand variability here to avoid test-label leakage.
            residual_std = float(hist["units_sold"].std(ddof=1) or 0)
            stock = int(test_group.sort_values("date").iloc[-1]["on_hand_units"])
        else:
            stock = int(hist.sort_values("date").iloc[-1]["on_hand_units"])
        lead = int(hist["lead_time_days"].median())
        policy = calculate_inventory_policy(avg_demand, residual_std, lead,
                                             review_period_days, service_z, stock)
        rows.append({"store_id": store, "sku_id": sku, "category": category,
                     "lead_time_days": lead, **policy})
    return pd.DataFrame(rows)
