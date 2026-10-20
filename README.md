# Retail Sales Forecasting & Inventory Optimization System

A portfolio-ready retail analytics project that forecasts product-level demand and turns forecasts into practical replenishment recommendations. It uses a reproducible synthetic retail dataset, so no company data or retail system access is required.

## Business problem
Retailers need to estimate upcoming demand while balancing two costly outcomes:
- **Stockouts:** lost sales and poor customer experience.
- **Overstock:** excess cash tied up in inventory, carrying costs, and potential waste.

This project builds a small end-to-end workflow: data generation → validation → exploratory analysis → demand forecasting → inventory policy → business dashboard.

## Features
- Reproducible daily sales data for multiple stores and SKUs
- Calendar, lag, rolling-average, price, promotion, and seasonality features
- Time-ordered train/test split to avoid leakage
- Naive baseline and Random Forest forecasting models
- MAE, RMSE, and WAPE evaluation
- Safety stock, reorder point, target stock, and order-quantity recommendations
- Streamlit dashboard for exploring forecast and inventory outputs
- Unit tests and a GitHub Actions workflow
- Clear limitations and interview talking points

## Quick start (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

## Quick start (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

The pipeline writes results to `reports/`:
- `model_metrics.csv`
- `test_forecasts.csv`
- `inventory_recommendations.csv`
- `sku_summary.csv`

The app reads these generated outputs. Run the pipeline first.

## Dataset
`data/retail_sales.csv` is synthetic, generated with a fixed seed. It represents daily demand for several SKUs across stores and includes date, units sold, price, promotion, product category, store, and on-hand stock. It is not real company data and should never be described as real retail transaction data.

Regenerate it:
```bash
python generate_data.py
```

## Pipeline overview
1. Validate columns, types, missing values, and non-negative sales.
2. Build calendar and lagged demand features within each store-SKU series.
3. Train a baseline forecast and Random Forest regressor using a chronological split.
4. Evaluate forecasts on the holdout period.
5. Estimate demand during lead time and calculate safety stock / reorder point.
6. Export transparent replenishment recommendations for business review.

## Core inventory logic
For each store-SKU:
- Average daily demand is estimated from training history.
- Demand variability is estimated from daily residuals (actual minus forecast) on the training period.
- `safety_stock = z × residual_std × sqrt(lead_time_days)`
- `reorder_point = average_daily_demand × lead_time_days + safety_stock`
- `target_stock = average_daily_demand × (lead_time_days + review_period_days) + safety_stock`
- `recommended_order_qty = max(0, ceil(target_stock - current_stock))`

The default service-level factor is `z = 1.65` (roughly a 95% one-sided normal service target under assumptions). This is a planning approximation, not a guarantee. Current stock is simulated and should be replaced by reliable inventory-on-hand data in a production deployment.

## Important modeling choices
- The test period is later than the training period.
- Lag and rolling features are shifted so the current day's sales are not used to predict that same day.
- The Random Forest is compared with a simple moving-average baseline.
- The pipeline reports WAPE alongside MAE/RMSE because it is interpretable across SKU volume, while noting WAPE can be unstable or undefined if total actual demand is zero.
- No future test labels are used to create test-time lag features. In this educational project, features are created from historical observed demand; a real multi-step forecast should recursively feed predictions forward or use a dedicated forecasting design.

## Limitations / production improvements
- Synthetic data does not represent every real retail pattern.
- Promotions and price effects are simulated, not causal estimates.
- Inventory positions do not include purchase orders, reservations, returns, shrinkage, or supplier constraints.
- Normal-demand safety stock is an approximation; intermittent demand may need Croston-style methods or quantile forecasts.
- The model is a portfolio demonstration, not a validated operational ordering system.
- For production: monitor drift, backtest by SKU/store, add holidays and supplier calendars, model lost sales, account for minimum order quantities/case packs, and require planner approval.

## Repository map
```text
retail-sales-forecasting-inventory-optimization/
├── app.py
├── run_pipeline.py
├── generate_data.py
├── requirements.txt
├── data/retail_sales.csv
├── src/
│   ├── data_validation.py
│   ├── features.py
│   ├── forecasting.py
│   └── inventory.py
├── tests/
│   └── test_inventory.py
├── reports/
│   └── (generated outputs)
└── .github/workflows/tests.yml
```

## GitHub publishing checklist
1. Create a public repository named `retail-sales-forecasting-inventory-optimization`.
2. Extract this ZIP and open the project folder in VS Code.
3. Run the pipeline and dashboard locally; inspect the generated CSVs.
4. Ensure `.venv/`, cache files, and secrets are not committed.
5. Initialize and push:
   ```bash
   git init
   git add .
   git commit -m "Build retail demand forecasting and inventory optimization pipeline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/retail-sales-forecasting-inventory-optimization.git
   git push -u origin main
   ```
6. Add screenshots of the dashboard to `reports/screenshots/` after running it. Do not fabricate screenshots or metrics.
7. Add repository topics: `retail-analytics`, `demand-forecasting`, `inventory-optimization`, `machine-learning`, `python`, `streamlit`.
8. Pin the repository on your GitHub profile after the README and outputs are complete.

## Interview explanation (60–90 seconds)
“I built a retail demand forecasting and replenishment prototype using reproducible synthetic daily sales across stores and SKUs. I validated the data, created lag and rolling features without using current-day sales, and used a chronological holdout to compare a moving-average baseline against a Random Forest model. I evaluated error with MAE, RMSE, and WAPE, then converted demand and forecast-error variability into safety stock, reorder points, and suggested order quantities. The dashboard lets a user inspect forecasts and inventory risk. Because the data is simulated, I present it as a prototype and discuss production gaps such as purchase orders, case packs, intermittent demand, and forecast monitoring.”

## Resume bullet (edit after running)
- Built a Python/Scikit-learn retail analytics pipeline that forecasts SKU-store demand, evaluates models using MAE/RMSE/WAPE, and generates safety-stock, reorder-point, and replenishment recommendations in a Streamlit dashboard using synthetic retail data.

Only include numerical improvements if you have measured them yourself from the generated `model_metrics.csv`.
