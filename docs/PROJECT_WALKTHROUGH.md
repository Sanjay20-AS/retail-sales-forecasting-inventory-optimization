# Project Walkthrough

## 1. Business framing
The system supports two linked decisions:
1. What might each store sell for each SKU in the near future?
2. Given demand, supplier lead time, and current simulated stock, which items may need replenishment?

A forecast is not automatically an order. Inventory decisions also depend on current inventory position, incoming purchase orders, supplier constraints, and business rules.

## 2. Dataset dictionary
| Field | Meaning |
|---|---|
| date | Calendar date of the observation |
| store_id | Synthetic store identifier |
| sku_id | Synthetic product identifier |
| category | Product category |
| unit_price | Simulated selling price |
| promotion | 1 if a promotion is active, else 0 |
| units_sold | Simulated units sold that day |
| on_hand_units | Simulated stock snapshot |
| lead_time_days | Assumed supplier lead time |

## 3. Forecasting design
The project uses calendar features, lag-1 and lag-7 demand, and shifted rolling means/std. Shifting ensures that rolling values for a date only use prior observations. A chronological split is used instead of random splitting because random splitting can leak future patterns into training.

Models:
- Moving-average baseline: last seven days' mean.
- Random Forest: nonlinear tabular model using price, promotion, calendar, and lag features.

Metrics:
- MAE: average absolute error in units.
- RMSE: penalizes larger errors more heavily.
- WAPE: sum of absolute errors divided by total actual demand, expressed as a percentage.

## 4. Inventory logic
Safety stock uses a normal approximation based on variability and lead time. The implementation is intentionally transparent so interviewers can inspect assumptions. The provided stock value is a simulated snapshot, not a live inventory position.

## 5. Suggested experiments
1. Change the random seed and compare metrics.
2. Compare Random Forest against the baseline by SKU and store.
3. Change `service_z` in `src/inventory.py` to see the effect of service level on safety stock.
4. Compare 7-day and 28-day moving averages.
5. Add holiday flags or a simulated stockout indicator.
6. Add a category-level performance report and error analysis.
7. Use rolling-origin backtesting rather than a single holdout.

## 6. Proof of work
Save genuine screenshots of:
- dashboard model metrics,
- actual vs forecast line chart,
- filtered replenishment recommendations,
- terminal output from tests.

Write a short `EXPERIMENTS.md` with the actual values you observed. Explain what improved, what did not, and why. Do not claim the model is production-ready or that it reduces costs unless you evaluate that claim with a defined baseline and cost model.
