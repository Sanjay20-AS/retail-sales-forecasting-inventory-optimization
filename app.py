
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.data_validation import validate_sales_data
from src.features import add_time_features
from src.forecasting import train_and_evaluate
from src.inventory import build_recommendations

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = ROOT_DIR / "data" / "retail_sales.csv"

st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# 2. CUSTOM DASHBOARD THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background: #101827;
        border-right: 1px solid #263449;
    }

    .hero {
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 24px;
        border: 1px solid #2b5263;
        background: linear-gradient(120deg, #172554, #123047 60%, #134e4a);
    }

    .hero h1 {
        color: #f8fafc;
        font-size: 30px;
        font-weight: 800;
        margin: 5px 0 10px 0;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 14px;
        margin: 0;
    }

    .eyebrow {
        color: #5eead4;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
    }

    .kpi {
        background: #131e30;
        border: 1px solid #263449;
        border-radius: 15px;
        padding: 18px;
        min-height: 118px;
    }

    .kpi-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 25px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .kpi-note {
        color: #94a3b8;
        font-size: 11px;
        margin-top: 8px;
    }

    .section-subtitle {
        color: #94a3b8;
        font-size: 12px;
        margin-bottom: 14px;
    }

    div[data-testid="stMetric"] {
        background: #131e30;
        border: 1px solid #263449;
        border-radius: 12px;
        padding: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    div[data-testid="stTabs"] button {
        color: #cbd5e1;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #5eead4;
        border-bottom-color: #5eead4;
    }

    .footer {
        color: #64748b;
        font-size: 11px;
        text-align: center;
        padding: 25px 0 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. DATA AND MODEL PIPELINE
# =========================================================
@st.cache_data(show_spinner=False)
def load_sales_data(path):
    df = pd.read_csv(path)
    validate_sales_data(df)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def run_pipeline(data):
    """
    Use the existing project APIs:
      features.add_time_features()
      forecasting.train_and_evaluate()
      inventory.build_recommendations()
    """
    featured = add_time_features(data.copy())

    model, train, test, metrics, split_date = train_and_evaluate(
        featured
    )

    inventory = build_recommendations(
        train,
        test,
        review_period_days=7,
        service_z=1.65,
    )

    return train, test, metrics, inventory, split_date


def money(value):
    return f"{value:,.2f}"


def render_kpi(label, value, note):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=350):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter"),
        margin=dict(l=15, r=15, t=55, b=15),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#263449",
        tickfont=dict(color="#94a3b8"),
    )
    fig.update_yaxes(
        gridcolor="#263449",
        zerolinecolor="#263449",
        tickfont=dict(color="#94a3b8"),
    )
    return fig


# =========================================================
# 4. LOAD DATA
# =========================================================
with st.sidebar:
    st.markdown("## 📊 Retail Intelligence")
    st.caption("Demand forecasting & inventory planning")
    st.divider()

    st.markdown("### Workspace")
    st.caption("Local retail simulation")

    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("### About")
    st.caption("Random Forest forecasting")
    st.caption("7-day moving-average baseline")
    st.caption("Safety stock and reorder planning")

if not DATA_PATH.exists():
    st.error(f"Dataset not found: `{DATA_PATH}`")
    st.info(
        "Run `python generate_data.py` from the project root, "
        "then restart Streamlit."
    )
    st.stop()

try:
    with st.spinner("Preparing sales analytics and inventory recommendations..."):
        sales = load_sales_data(str(DATA_PATH))
        train_df, test_df, metrics, inventory, split_date = run_pipeline(
            sales
        )
except Exception as exc:
    st.error("The analytics pipeline could not be completed.")
    st.exception(exc)
    st.info(
        "Check that the function names and column names in your "
        "`src` modules match the versions you shared."
    )
    st.stop()

# =========================================================
# 5. HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">RETAIL ANALYTICS PLATFORM</div>
        <h1>Forecast demand. Optimize inventory.</h1>
        <p>
            Explore sales trends, evaluate forecasting models, and identify
            replenishment needs through an interactive retail dashboard.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 6. INTERACTIVE FILTERS
# =========================================================
st.markdown("### 🎛️ Dashboard filters")
f1, f2, f3 = st.columns(3)

with f1:
    all_stores = sorted(sales["store_id"].dropna().unique().tolist())
    selected_stores = st.multiselect(
        "Select stores",
        all_stores,
        default=all_stores,
    )

with f2:
    all_skus = sorted(sales["sku_id"].dropna().unique().tolist())
    selected_skus = st.multiselect(
        "Select products",
        all_skus,
        default=all_skus,
    )

with f3:
    min_date = sales["date"].min().date()
    max_date = sales["date"].max().date()

    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

filtered = sales.copy()

filtered = filtered[
    filtered["store_id"].isin(selected_stores)
    & filtered["sku_id"].isin(selected_skus)
]

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered = filtered[
        filtered["date"].dt.date.between(start_date, end_date)
    ]

if filtered.empty:
    st.warning("No sales records match the selected filters.")
    st.stop()

# =========================================================
# 7. BUSINESS SNAPSHOT
# =========================================================
total_units = float(filtered["units_sold"].sum())
store_count = filtered["store_id"].nunique()
sku_count = filtered["sku_id"].nunique()
active_days = filtered["date"].nunique()

revenue = None
if "revenue" in filtered.columns:
    revenue = pd.to_numeric(
        filtered["revenue"], errors="coerce"
    ).sum()
elif "unit_price" in filtered.columns:
    revenue = (
        pd.to_numeric(filtered["unit_price"], errors="coerce")
        * pd.to_numeric(filtered["units_sold"], errors="coerce")
    ).sum()

metrics_df = pd.DataFrame(metrics)

best_model = "N/A"
best_wape = None

if (
    not metrics_df.empty
    and "model" in metrics_df.columns
    and "WAPE_percent" in metrics_df.columns
):
    valid = metrics_df.dropna(subset=["WAPE_percent"])
    if not valid.empty:
        best_row = valid.loc[valid["WAPE_percent"].idxmin()]
        best_model = str(best_row["model"])
        best_wape = float(best_row["WAPE_percent"])

st.markdown("### Business snapshot")
st.markdown(
    '<div class="section-subtitle">Key indicators for the selected data</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    render_kpi("Units sold", f"{total_units:,.0f}", "Filtered sales volume")

with k2:
    render_kpi(
        "Sales value",
        f"{revenue:,.0f}" if revenue is not None else "N/A",
        "Estimated from available data",
    )

with k3:
    render_kpi("Stores", f"{store_count}", "Selected stores")

with k4:
    render_kpi("Products", f"{sku_count}", "Unique SKUs")

with k5:
    render_kpi(
        "Best model WAPE",
        f"{best_wape:.2f}%" if best_wape is not None else "N/A",
        best_model,
    )

st.write("")

# =========================================================
# 8. MAIN TABS
# =========================================================
tab_overview, tab_forecast, tab_inventory, tab_data = st.tabs(
    [
        "📈 Sales overview",
        "🔮 Forecasting",
        "📦 Inventory planning",
        "🗂️ Data explorer",
    ]
)

# =========================================================
# 9. SALES OVERVIEW
# =========================================================
with tab_overview:
    st.markdown("### Sales performance")

    left, right = st.columns([1.6, 1])

    with left:
        daily_sales = (
            filtered.groupby("date", as_index=False)["units_sold"]
            .sum()
            .sort_values("date")
        )

        fig = px.area(
            daily_sales,
            x="date",
            y="units_sold",
            title="Daily demand trend",
            labels={"date": "Date", "units_sold": "Units sold"},
        )
        fig.update_traces(
            line_color="#2dd4bf",
            fillcolor="rgba(45,212,191,0.12)",
        )
        st.plotly_chart(
            chart_layout(fig, 370),
            use_container_width=True,
        )

    with right:
        top_products = (
            filtered.groupby("sku_id", as_index=False)["units_sold"]
            .sum()
            .nlargest(10, "units_sold")
            .sort_values("units_sold")
        )

        fig = px.bar(
            top_products,
            x="units_sold",
            y="sku_id",
            orientation="h",
            title="Top 10 products",
            labels={"units_sold": "Units sold", "sku_id": "SKU"},
        )
        fig.update_traces(marker_color="#818cf8")
        st.plotly_chart(
            chart_layout(fig, 370),
            use_container_width=True,
        )

    left2, right2 = st.columns(2)

    with left2:
        store_sales = (
            filtered.groupby("store_id", as_index=False)["units_sold"]
            .sum()
            .sort_values("units_sold", ascending=False)
        )

        fig = px.bar(
            store_sales,
            x="store_id",
            y="units_sold",
            title="Store performance",
            labels={"store_id": "Store", "units_sold": "Units sold"},
            color="units_sold",
            color_continuous_scale="Teal",
        )
        st.plotly_chart(
            chart_layout(fig, 320),
            use_container_width=True,
        )

    with right2:
        if "category" in filtered.columns:
            category_sales = (
                filtered.groupby("category", as_index=False)["units_sold"]
                .sum()
                .sort_values("units_sold", ascending=False)
            )

            
            fig = px.pie(
                category_sales,
                names="category",
                values="units_sold",
                hole=0.6,
            )

            fig.update_layout(
                title=dict(
                    text="Sales by category",
                    x=0.02,
                    xanchor="left",
                    y=0.98,
                    yanchor="top",
                ),
                margin=dict(l=15, r=15, t=85, b=15),
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont=dict(size=11),
            )

            st.plotly_chart(
                chart_layout(fig, 360),
                use_container_width=True,
            )

        else:
            st.info("The dataset does not contain a category column.")

    st.caption(
        f"Data coverage: {active_days:,} distinct sales dates in the current selection."
    )

# =========================================================
# 10. FORECASTING AND MODEL EVALUATION
# =========================================================
with tab_forecast:
    st.markdown("### Forecast model performance")
    st.caption(
        f"Chronological evaluation split date: {pd.Timestamp(split_date).date()}"
    )

    if not metrics_df.empty:
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
        )

        if {"model", "WAPE_percent"}.issubset(metrics_df.columns):
            plot_metrics = metrics_df.copy()
            plot_metrics["WAPE_percent"] = pd.to_numeric(
                plot_metrics["WAPE_percent"], errors="coerce"
            )
            plot_metrics = plot_metrics.dropna(subset=["WAPE_percent"])

            if not plot_metrics.empty:
                fig = px.bar(
                    plot_metrics.sort_values("WAPE_percent"),
                    x="WAPE_percent",
                    y="model",
                    orientation="h",
                    title="Weighted absolute percentage error",
                    labels={
                        "WAPE_percent": "WAPE (%) — lower is better",
                        "model": "Model",
                    },
                    text="WAPE_percent",
                )
                fig.update_traces(
                    marker_color="#2dd4bf",
                    texttemplate="%{text:.2f}%",
                    textposition="outside",
                )
                st.plotly_chart(
                    chart_layout(fig, 300),
                    use_container_width=True,
                )

                baseline = plot_metrics[
                    plot_metrics["model"] == "MovingAverage7"
                ]
                rf = plot_metrics[
                    plot_metrics["model"] == "RandomForest"
                ]

                if not baseline.empty and not rf.empty:
                    base_wape = float(baseline.iloc[0]["WAPE_percent"])
                    rf_wape = float(rf.iloc[0]["WAPE_percent"])

                    if base_wape > 0:
                        improvement = (
                            (base_wape - rf_wape) / base_wape
                        ) * 100

                        st.metric(
                            "Random Forest improvement over baseline",
                            f"{improvement:.2f}%",
                            help=(
                                "Relative reduction in WAPE compared with "
                                "the 7-day moving-average baseline."
                            ),
                        )
    else:
        st.warning("No model metrics are available.")

    st.markdown("### Actual demand vs predictions")

    if not test_df.empty:
        daily_forecast = (
            test_df.groupby("date", as_index=False)
            .agg(
                Actual=("units_sold", "sum"),
                RandomForest=("rf_forecast", "sum"),
                MovingAverage7=("baseline_forecast", "sum"),
            )
            .sort_values("date")
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily_forecast["date"],
                y=daily_forecast["Actual"],
                name="Actual demand",
                mode="lines",
                line=dict(color="#60a5fa", width=2.5),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily_forecast["date"],
                y=daily_forecast["RandomForest"],
                name="Random Forest",
                mode="lines",
                line=dict(color="#2dd4bf", width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily_forecast["date"],
                y=daily_forecast["MovingAverage7"],
                name="7-day baseline",
                mode="lines",
                line=dict(color="#fbbf24", width=2, dash="dash"),
            )
        )

        fig.update_layout(title="Holdout period — daily aggregated demand")
        st.plotly_chart(
            chart_layout(fig, 400),
            use_container_width=True,
        )

        st.download_button(
            "⬇️ Download forecast results",
            data=test_df.to_csv(index=False).encode("utf-8"),
            file_name="forecast_results.csv",
            mime="text/csv",
        )

        with st.expander("View row-level predictions"):
            st.dataframe(
                test_df,
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("How to interpret the evaluation"):
        st.markdown(
            """
            - **MAE:** Average absolute prediction error in units.
            - **RMSE:** Gives greater weight to large prediction errors.
            - **WAPE:** Total absolute prediction error divided by total actual demand.
            - Lower values are generally better.
            - The Random Forest is compared with a simple 7-day moving-average baseline.
            - Results are from a synthetic-data simulation and are not proof of performance on real retail operations.
            """
        )

# =========================================================
# 11. INVENTORY OPTIMIZATION
# =========================================================
with tab_inventory:
    st.markdown("### Inventory planning and replenishment")

    inv_view = inventory.copy()

    if not inv_view.empty:
        inv_view = inv_view[
            inv_view["store_id"].isin(selected_stores)
            & inv_view["sku_id"].isin(selected_skus)
        ]

    if inv_view.empty:
        st.info("No inventory recommendations match your filters.")
    else:
        inv_view["reorder_now"] = (
            inv_view["reorder_now"].astype(bool)
        )

        reorder_count = int(inv_view["reorder_now"].sum())
        order_qty = int(inv_view["recommended_order_qty"].sum())
        total_stock = int(inv_view["current_stock"].sum())
        mean_safety = float(inv_view["safety_stock"].mean())

        a, b, c, d = st.columns(4)

        a.metric("Store-SKU combinations", f"{len(inv_view):,}")
        b.metric("Reorder alerts", f"{reorder_count:,}")
        c.metric("Suggested order units", f"{order_qty:,}")
        d.metric("Average safety stock", f"{mean_safety:.1f}")

        left, right = st.columns(2)

        with left:
            status_df = pd.DataFrame(
                {
                    "Status": [
                        "Reorder now",
                        "Above reorder point",
                    ],
                    "Products": [
                        reorder_count,
                        len(inv_view) - reorder_count,
                    ],
                }
            )

            fig = px.pie(
                status_df,
                names="Status",
                values="Products",
                hole=0.6,
                title="Replenishment status",
                color="Status",
                color_discrete_map={
                    "Reorder now": "#fb7185",
                    "Above reorder point": "#2dd4bf",
                },
            )
            st.plotly_chart(
                chart_layout(fig, 330),
                use_container_width=True,
            )

        with right:
            top_orders = (
                inv_view.nlargest(10, "recommended_order_qty")
                .sort_values("recommended_order_qty")
            )

            fig = px.bar(
                top_orders,
                x="recommended_order_qty",
                y="sku_id",
                color="reorder_now",
                orientation="h",
                title="Products with largest suggested orders",
                labels={
                    "recommended_order_qty": "Suggested order quantity",
                    "sku_id": "SKU",
                    "reorder_now": "Reorder now",
                },
                color_discrete_map={
                    True: "#fb7185",
                    False: "#2dd4bf",
                },
            )
            st.plotly_chart(
                chart_layout(fig, 330),
                use_container_width=True,
            )

        st.markdown("#### Reorder point vs current stock")

        stock_chart = inv_view.copy()

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=stock_chart["sku_id"].astype(str),
                y=stock_chart["current_stock"],
                name="Current stock",
                marker_color="#60a5fa",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=stock_chart["sku_id"].astype(str),
                y=stock_chart["reorder_point"],
                name="Reorder point",
                mode="markers",
                marker=dict(color="#fb7185", size=10, symbol="diamond"),
            )
        )

        fig.update_layout(
            title="Stock position by SKU",
            xaxis_title="SKU",
            yaxis_title="Units",
            barmode="group",
        )

        st.plotly_chart(
            chart_layout(fig, 350),
            use_container_width=True,
        )

        st.markdown("#### Replenishment recommendation table")

        display_cols = [
            "store_id",
            "sku_id",
            "category",
            "lead_time_days",
            "avg_daily_demand",
            "current_stock",
            "residual_std",
            "safety_stock",
            "reorder_point",
            "target_stock",
            "recommended_order_qty",
            "reorder_now",
        ]

        display_cols = [
            col for col in display_cols if col in inv_view.columns
        ]

        st.dataframe(
            inv_view[display_cols].sort_values(
                ["reorder_now", "recommended_order_qty"],
                ascending=[False, False],
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇️ Download inventory recommendations",
            data=inv_view.to_csv(index=False).encode("utf-8"),
            file_name="inventory_recommendations.csv",
            mime="text/csv",
        )

        st.caption(
            "Reorder recommendations use historical demand variability, "
            "lead time, review period, and simulated stock levels. "
            "Validate the assumptions before making real purchasing decisions."
        )

# =========================================================
# 12. DATA EXPLORER
# =========================================================
with tab_data:
    st.markdown("### Explore the filtered sales records")
    st.caption("Inspect and download the records matching your filters.")

    st.write(f"**{len(filtered):,} rows** match your current selection.")

    st.dataframe(
        filtered.head(5000),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇️ Download filtered sales data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_retail_sales.csv",
        mime="text/csv",
    )

    with st.expander("Dataset schema and data quality"):
        schema = pd.DataFrame(
            {
                "Column": sales.columns,
                "Data type": [
                    str(dtype) for dtype in sales.dtypes
                ],
                "Missing values": [
                    int(sales[column].isna().sum())
                    for column in sales.columns
                ],
            }
        )

        st.dataframe(
            schema,
            use_container_width=True,
            hide_index=True,
        )

# =========================================================
# 13. FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        RETAIL INTELLIGENCE · DEMAND FORECASTING & INVENTORY OPTIMIZATION<br>
        Educational decision-support prototype using synthetic retail data.
    </div>
    """,
    unsafe_allow_html=True,
)
