
"""Retail Sales Forecasting & Inventory Optimization Dashboard.

Run from the project root:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_validation import validate_sales_data
from src.features import add_time_features
from src.forecasting import train_and_evaluate
from src.inventory import build_recommendations


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "retail_sales.csv"

st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Styling --------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,.18);
    }
    .hero {
        padding: 1.4rem;
        border-radius: 16px;
        background: linear-gradient(115deg, #102a43, #176b87);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {
        color: white;
        margin: 0 0 .4rem 0;
        font-size: 1.9rem;
    }
    .hero p {
        color: #e6f3f8;
        margin: 0;
    }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.2);
        padding: .8rem 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def chart_layout(fig, height=340):
    """Apply consistent chart spacing."""
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=55, b=15),
        font=dict(size=12),
        legend_title_text="",
    )
    return fig


@st.cache_data(show_spinner=False)
def load_data(path_string: str, modified_time: float) -> pd.DataFrame:
    """Load and validate the CSV dataset."""
    del modified_time
    df = pd.read_csv(path_string, parse_dates=["date"])
    validate_sales_data(df)
    return df


@st.cache_data(
    show_spinner="Training forecasting model and preparing inventory..."
)
def run_pipeline(data: pd.DataFrame):
    """Prepare features, evaluate forecasts, and build inventory policies."""
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


# -------------------- Load dataset --------------------

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found: {DATA_PATH}\n\n"
        "Run `python generate_data.py` from your project folder."
    )
    st.stop()

try:
    source_data = load_data(
        str(DATA_PATH),
        DATA_PATH.stat().st_mtime,
    )

    train, forecasts, metrics, inventory, split_date = run_pipeline(
        source_data
    )

except Exception as exc:
    st.error(f"Could not prepare the dashboard: {exc}")
    st.info(
        "Check your dependencies and ensure "
        "data/retail_sales.csv contains the required columns."
    )
    st.stop()


# -------------------- Sidebar navigation --------------------

with st.sidebar:
    st.markdown("## 📦 Retail Intelligence")
    st.caption("Forecasting & Inventory Platform")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Sales Overview",
            "Forecasting",
            "Inventory Optimization",
            "Data Explorer",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Filters")

    store_options = [
        "All stores"
    ] + sorted(source_data["store_id"].astype(str).unique().tolist())

    selected_store = st.selectbox(
        "Store",
        store_options,
    )

    sku_options = [
        "All products"
    ] + sorted(source_data["sku_id"].astype(str).unique().tolist())

    selected_sku = st.selectbox(
        "Product / SKU",
        sku_options,
    )

    date_min = source_data["date"].min().date()
    date_max = source_data["date"].max().date()

    selected_dates = st.date_input(
        "Date range",
        value=(date_min, date_max),
        min_value=date_min,
        max_value=date_max,
    )

    if st.button("↻ Refresh dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.caption(
        "Portfolio prototype • Synthetic data • "
        "Review recommendations before use"
    )


# -------------------- Apply filters --------------------

# Important: source_data is the actual loaded dataset.
# Do not use an undefined variable named data in page sections.

filtered_sales = source_data.copy()

if selected_store != "All stores":
    filtered_sales = filtered_sales[
        filtered_sales["store_id"].astype(str) == selected_store
    ]

if selected_sku != "All products":
    filtered_sales = filtered_sales[
        filtered_sales["sku_id"].astype(str) == selected_sku
    ]

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_date, end_date = selected_dates

    filtered_sales = filtered_sales[
        filtered_sales["date"].dt.date.between(
            start_date,
            end_date,
        )
    ]


filtered_forecasts = forecasts.copy()

if selected_store != "All stores":
    filtered_forecasts = filtered_forecasts[
        filtered_forecasts["store_id"].astype(str) == selected_store
    ]

if selected_sku != "All products":
    filtered_forecasts = filtered_forecasts[
        filtered_forecasts["sku_id"].astype(str) == selected_sku
    ]

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    filtered_forecasts = filtered_forecasts[
        filtered_forecasts["date"].dt.date.between(
            start_date,
            end_date,
        )
    ]


filtered_inventory = inventory.copy()

if selected_store != "All stores":
    filtered_inventory = filtered_inventory[
        filtered_inventory["store_id"].astype(str) == selected_store
    ]

if selected_sku != "All products":
    filtered_inventory = filtered_inventory[
        filtered_inventory["sku_id"].astype(str) == selected_sku
    ]


# -------------------- Page header --------------------

st.markdown(
    f"""
    <div class="hero">
        <h1>{page}</h1>
        <p>
            Data-driven retail decisions with demand forecasts
            and inventory planning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Source period: {date_min:%d %b %Y} – {date_max:%d %b %Y}"
    f" · Forecast holdout begins "
    f"{pd.Timestamp(split_date).date():%d %b %Y}"
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <h1>Welcome to Retail Intelligence</h1>
            <p>
                Explore retail sales, forecast product demand,
                and plan inventory using data-driven insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Use the filtered dataset, not an undefined `data` variable.
    home_sales = filtered_sales
    home_inventory = filtered_inventory

    total_units = (
        int(home_sales["units_sold"].sum())
        if not home_sales.empty else 0
    )

    sales_value = (
        float(
            (
                home_sales["units_sold"]
                * home_sales["unit_price"]
            ).sum()
        )
        if not home_sales.empty else 0.0
    )

    store_count = (
        int(home_sales["store_id"].nunique())
        if not home_sales.empty else 0
    )

    sku_count = (
        int(home_sales["sku_id"].nunique())
        if not home_sales.empty else 0
    )

    reorder_count = (
        int(home_inventory["reorder_now"].sum())
        if not home_inventory.empty else 0
    )

    st.markdown("### At a glance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total units sold",
        f"{total_units:,}",
    )

    c2.metric(
        "Estimated sales value",
        f"₹{sales_value:,.0f}",
    )

    c3.metric(
        "Stores in selection",
        f"{store_count:,}",
    )

    c4.metric(
        "Products to reorder",
        f"{reorder_count:,}",
    )

    st.markdown("### Your analytics workspace")

    left, right = st.columns(2)

    with left:
        st.markdown("#### 📊 Sales Overview")
        st.write(
            "Review daily sales trends, compare store performance, "
            "explore product rankings, and analyze category sales."
        )

        st.markdown("#### 📈 Forecasting")
        st.write(
            "Compare Random Forest forecasts with a 7-day "
            "moving-average baseline using MAE, RMSE, and WAPE."
        )

    with right:
        st.markdown("#### 📦 Inventory Optimization")
        st.write(
            "Review reorder alerts, safety stock, reorder points, "
            "and suggested order quantities for each store and SKU."
        )

        st.markdown("#### 🗂️ Data Explorer")
        st.write(
            "Inspect sales records, review dataset quality, "
            "and export filtered data to CSV."
        )

    st.markdown("### Quick insights")

    if home_sales.empty:
        st.info(
            "No sales records match the selected filters. "
            "Adjust the store, product, or date range in the sidebar."
        )

    else:
        insight_cols = st.columns(3)

        sku_sales = home_sales.groupby("sku_id")["units_sold"].sum()
        store_sales = home_sales.groupby("store_id")["units_sold"].sum()
        category_sales = home_sales.groupby("category")["units_sold"].sum()

        top_product = sku_sales.idxmax()
        top_store = store_sales.idxmax()
        top_category = category_sales.idxmax()

        insight_cols[0].metric(
            "Top-selling SKU",
            str(top_product),
            f"{int(sku_sales.max()):,} units",
        )

        insight_cols[1].metric(
            "Highest-selling store",
            str(top_store),
            f"{int(store_sales.max()):,} units",
        )

        insight_cols[2].metric(
            "Leading category",
            str(top_category),
        )

    st.info(
        "Choose a section from the sidebar to explore detailed analytics. "
        "Home metrics follow your current sidebar filters."
    )


# ============================================================
# SALES OVERVIEW
# ============================================================

elif page == "Sales Overview":

    total_units = (
        int(filtered_sales["units_sold"].sum())
        if not filtered_sales.empty else 0
    )

    sales_value = (
        float(
            (
                filtered_sales["units_sold"]
                * filtered_sales["unit_price"]
            ).sum()
        )
        if not filtered_sales.empty else 0.0
    )

    active_skus = int(filtered_sales["sku_id"].nunique())
    reorder_count = (
        int(filtered_inventory["reorder_now"].sum())
        if not filtered_inventory.empty else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Units sold", f"{total_units:,}")
    c2.metric("Estimated sales value", f"₹{sales_value:,.0f}")
    c3.metric("Products in selection", f"{active_skus:,}")
    c4.metric("Products to reorder", f"{reorder_count:,}")

    if filtered_sales.empty:
        st.info("No sales records match these filters.")

    else:
        daily_sales = (
            filtered_sales.groupby("date", as_index=False)
            .agg(units_sold=("units_sold", "sum"))
        )

        left, right = st.columns([1.6, 1])

        with left:
            st.markdown("#### Daily sales trend")

            fig = px.area(
                daily_sales,
                x="date",
                y="units_sold",
                labels={
                    "date": "Date",
                    "units_sold": "Units sold",
                },
            )

            fig.update_traces(line_width=2)

            st.plotly_chart(
                chart_layout(fig, 340),
                use_container_width=True,
            )

        with right:
            # Use a Streamlit heading separately to prevent overlap.
            st.markdown("#### Sales by category")

            category_df = (
                filtered_sales.groupby("category", as_index=False)
                ["units_sold"].sum()
                .sort_values("units_sold", ascending=False)
            )

            fig = px.pie(
                category_df,
                names="category",
                values="units_sold",
                hole=0.58,
            )

            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont=dict(size=11),
            )

            fig.update_layout(
                title=None,
                margin=dict(l=10, r=10, t=10, b=10),
                height=340,
                showlegend=False,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        left, right = st.columns(2)

        with left:
            st.markdown("#### Top products by units sold")

            top_skus = (
                filtered_sales.groupby("sku_id", as_index=False)
                ["units_sold"].sum()
                .nlargest(10, "units_sold")
                .sort_values("units_sold")
            )

            fig = px.bar(
                top_skus,
                x="units_sold",
                y="sku_id",
                orientation="h",
                labels={
                    "sku_id": "SKU",
                    "units_sold": "Units sold",
                },
            )

            st.plotly_chart(
                chart_layout(fig, 340),
                use_container_width=True,
            )

        with right:
            st.markdown("#### Store performance")

            store_df = (
                filtered_sales.groupby("store_id", as_index=False)
                ["units_sold"].sum()
                .sort_values("units_sold", ascending=False)
            )

            fig = px.bar(
                store_df,
                x="store_id",
                y="units_sold",
                labels={
                    "store_id": "Store",
                    "units_sold": "Units sold",
                },
            )

            st.plotly_chart(
                chart_layout(fig, 340),
                use_container_width=True,
            )


# ============================================================
# FORECASTING
# ============================================================

elif page == "Forecasting":

    st.markdown("#### Model performance on the chronological holdout")
    st.caption("Lower values are better. WAPE means weighted absolute percentage error.")

    st.dataframe(
        metrics.style.format(
            {
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}",
                "WAPE_percent": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    metric_col, _ = st.columns([1, 2])

    rf_row = metrics[metrics["model"] == "RandomForest"]
    baseline_row = metrics[metrics["model"] == "MovingAverage7"]

    if not rf_row.empty and not baseline_row.empty:
        rf_wape = float(rf_row.iloc[0]["WAPE_percent"])
        base_wape = float(baseline_row.iloc[0]["WAPE_percent"])

        improvement = (
            (base_wape - rf_wape) / base_wape * 100
            if base_wape else 0.0
        )

        metric_col.metric(
            "Random Forest WAPE improvement",
            f"{improvement:.1f}%",
            "vs. 7-day moving average",
        )

    if filtered_forecasts.empty:
        st.info(
            "No holdout forecast rows match these filters. "
            "Try a different product or date range."
        )

    else:
        daily = (
            filtered_forecasts.groupby("date", as_index=False)[
                ["units_sold", "rf_forecast", "baseline_forecast"]
            ].sum()
        )

        st.markdown("#### Actual vs. predicted demand")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["units_sold"],
                mode="lines",
                name="Actual",
                line=dict(width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["rf_forecast"],
                mode="lines",
                name="Random Forest",
                line=dict(width=2),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=daily["date"],
                y=daily["baseline_forecast"],
                mode="lines",
                name="7-day moving average",
                line=dict(width=1.5, dash="dash"),
            )
        )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Units sold",
        )

        st.plotly_chart(
            chart_layout(fig, 400),
            use_container_width=True,
        )

        st.markdown("#### Forecast records")

        display_forecasts = filtered_forecasts.sort_values(
            "date",
            ascending=False,
        )

        st.dataframe(
            display_forecasts,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download forecast CSV",
            display_forecasts.to_csv(index=False).encode("utf-8"),
            file_name="retail_forecasts_filtered.csv",
            mime="text/csv",
        )


# ============================================================
# INVENTORY OPTIMIZATION
# ============================================================

elif page == "Inventory Optimization":

    if filtered_inventory.empty:
        st.info("No inventory recommendations match these filters.")

    else:
        reorder_count = int(filtered_inventory["reorder_now"].sum())

        total_order_qty = int(
            filtered_inventory["recommended_order_qty"].sum()
        )

        avg_lead = float(
            filtered_inventory["lead_time_days"].mean()
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Reorder alerts", f"{reorder_count:,}")
        c2.metric("Suggested units to order", f"{total_order_qty:,}")
        c3.metric("Average supplier lead time", f"{avg_lead:.1f} days")

        left, right = st.columns(2)

        with left:
            st.markdown("#### Reorder status")

            status_df = pd.DataFrame(
                {
                    "Status": [
                        "Reorder now",
                        "No immediate reorder",
                    ],
                    "Products": [
                        reorder_count,
                        len(filtered_inventory) - reorder_count,
                    ],
                }
            )

            fig = px.pie(
                status_df,
                names="Status",
                values="Products",
                hole=0.55,
            )

            fig.update_layout(
                title=None,
                height=320,
                margin=dict(l=10, r=10, t=10, b=10),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        with right:
            st.markdown("#### Recommended order quantity by SKU")

            order_df = (
                filtered_inventory.nlargest(
                    12,
                    "recommended_order_qty",
                )
                .sort_values("recommended_order_qty")
            )

            fig = px.bar(
                order_df,
                x="recommended_order_qty",
                y="sku_id",
                color="store_id",
                orientation="h",
                labels={
                    "recommended_order_qty": "Units to order",
                    "sku_id": "SKU",
                    "store_id": "Store",
                },
            )

            st.plotly_chart(
                chart_layout(fig, 320),
                use_container_width=True,
            )

        st.markdown("#### Current stock vs. reorder point")

        stock_df = filtered_inventory.copy()

        stock_df["item"] = (
            stock_df["store_id"].astype(str)
            + " · "
            + stock_df["sku_id"].astype(str)
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=stock_df["item"],
                y=stock_df["current_stock"],
                name="Current stock",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=stock_df["item"],
                y=stock_df["reorder_point"],
                mode="markers",
                name="Reorder point",
                marker=dict(size=9, symbol="diamond"),
            )
        )

        fig.update_layout(
            xaxis_title="Store · SKU",
            yaxis_title="Units",
            barmode="group",
        )

        st.plotly_chart(
            chart_layout(fig, 360),
            use_container_width=True,
        )

        st.markdown("#### Inventory recommendations")

        inv_view = filtered_inventory.sort_values(
            ["reorder_now", "recommended_order_qty"],
            ascending=[False, False],
        )

        st.dataframe(
            inv_view,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download inventory recommendations CSV",
            inv_view.to_csv(index=False).encode("utf-8"),
            file_name="inventory_recommendations_filtered.csv",
            mime="text/csv",
        )

        with st.expander("How to interpret these recommendations"):
            st.markdown(
                "- **Safety stock:** buffer against demand variability.\n"
                "- **Reorder point:** estimated stock level that triggers replenishment.\n"
                "- **Target stock:** demand coverage for lead time plus review period, including safety stock.\n"
                "- **Recommended order quantity:** estimated quantity needed to reach target stock."
            )

        st.warning(
            "Prototype limitation: this does not account for open purchase "
            "orders, reserved inventory, case-pack sizes, minimum order "
            "quantities, shelf life, or supplier capacity. Review "
            "recommendations before acting."
        )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.markdown("#### Filtered sales dataset")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows in selection",
        f"{len(filtered_sales):,}",
    )

    c2.metric(
        "Columns",
        f"{filtered_sales.shape[1]:,}",
    )

    c3.metric(
        "Missing values",
        f"{int(filtered_sales.isna().sum().sum()):,}",
    )

    st.dataframe(
        filtered_sales.sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered sales CSV",
        filtered_sales.to_csv(index=False).encode("utf-8"),
        file_name="retail_sales_filtered.csv",
        mime="text/csv",
    )

    with st.expander("Dataset schema"):
        schema = pd.DataFrame(
            {
                "Column": filtered_sales.columns,
                "Data type": [
                    str(dtype)
                    for dtype in filtered_sales.dtypes
                ],
            }
        )

        st.dataframe(
            schema,
            use_container_width=True,
            hide_index=True,
        )


# -------------------- Footer --------------------

st.divider()

st.caption(
    "Retail Sales Forecasting & Inventory Optimization "
    "• Student portfolio prototype • Synthetic data only"
)
