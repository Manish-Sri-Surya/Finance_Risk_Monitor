from __future__ import annotations

import base64
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st

from src.finance_monitor import (
    analyze_transactions,
    category_spend,
    generate_pdf_report,
    load_transactions,
    monthly_spend,
    recommend_budgets,
    risk_alerts,
    summarize,
)


st.set_page_config(
    page_title="AI Personal Finance Risk Monitor",
    page_icon="",
    layout="wide",
)


APP_DIR = Path(__file__).parent


def asset_as_data_uri(path: Path) -> str:
    mime = "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def load_demo_data() -> pd.DataFrame:
    return load_transactions(APP_DIR / "data" / "sample_transactions.csv")


background_uri = asset_as_data_uri(APP_DIR / "assets" / "finance-dashboard-bg.png")

st.markdown(
    f"""
    <style>
    @keyframes fadeLift {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes softPulse {{
        0%, 100% {{ box-shadow: 0 0 0 rgba(216, 176, 91, 0); }}
        50% {{ box-shadow: 0 0 32px rgba(216, 176, 91, 0.16); }}
    }}

    @keyframes scanLine {{
        from {{ transform: translateX(-30%); opacity: 0; }}
        35% {{ opacity: 1; }}
        to {{ transform: translateX(120%); opacity: 0; }}
    }}

    .stApp {{
        background:
            linear-gradient(180deg, rgba(7, 13, 18, 0.58), rgba(7, 13, 18, 0.82)),
            url("{background_uri}") center center / cover fixed no-repeat;
        color: #eef5f2;
    }}

    .block-container {{
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1240px;
        animation: fadeLift 520ms ease-out both;
    }}

    [data-testid="stSidebar"] {{
        background: rgba(10, 20, 26, 0.86);
        border-right: 1px solid rgba(137, 185, 179, 0.18);
        backdrop-filter: blur(18px);
    }}

    [data-testid="stSidebar"] * {{
        color: #e8f1ee;
    }}

    .finance-hero {{
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(160, 199, 191, 0.22);
        background: rgba(9, 20, 25, 0.72);
        backdrop-filter: blur(18px);
        border-radius: 8px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.25rem;
        animation: fadeLift 620ms ease-out both;
    }}

    .finance-hero::after {{
        content: "";
        position: absolute;
        inset: 0;
        width: 38%;
        background: linear-gradient(90deg, transparent, rgba(216, 176, 91, 0.12), transparent);
        animation: scanLine 5.4s ease-in-out infinite;
        pointer-events: none;
    }}

    .finance-hero h1 {{
        position: relative;
        margin: 0 0 0.35rem 0;
        color: #f6fbf8;
        font-size: clamp(2rem, 4vw, 3.3rem);
        letter-spacing: 0;
        line-height: 1.04;
        z-index: 1;
    }}

    .finance-hero p {{
        position: relative;
        max-width: 760px;
        margin: 0;
        color: #b8cbc6;
        font-size: 1rem;
        line-height: 1.55;
        z-index: 1;
    }}

    [data-testid="stMetric"] {{
        min-height: 118px;
        background: rgba(10, 22, 27, 0.76);
        border: 1px solid rgba(139, 193, 183, 0.22);
        border-radius: 8px;
        padding: 1rem 1.05rem;
        backdrop-filter: blur(16px);
        animation: fadeLift 700ms ease-out both;
        transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
    }}

    [data-testid="stMetric"]:hover {{
        transform: translateY(-3px);
        border-color: rgba(216, 176, 91, 0.48);
        background: rgba(12, 30, 35, 0.84);
    }}

    [data-testid="stMetricLabel"] p {{
        color: #a8bbb6;
        font-size: 0.82rem;
    }}

    [data-testid="stMetricValue"] {{
        color: #ffffff;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(9, 20, 25, 0.72);
        border: 1px solid rgba(216, 176, 91, 0.28);
        border-radius: 8px;
        backdrop-filter: blur(18px);
        animation: softPulse 5s ease-in-out infinite;
    }}

    [data-testid="stVerticalBlockBorderWrapper"] h3 {{
        color: #ffd986;
    }}

    [data-testid="stTabs"] button {{
        color: #d9e7e3;
    }}

    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: #ffffff;
        border-bottom-color: #d8b05b;
    }}

    [data-testid="stDataFrame"], [data-testid="stTable"], [data-testid="stChart"] {{
        border-radius: 8px;
        overflow: hidden;
    }}

    .stButton > button,
    .stDownloadButton > button {{
        border-radius: 8px;
        border: 1px solid rgba(216, 176, 91, 0.44);
        background: rgba(14, 33, 38, 0.84);
        color: #f7fbf8;
        transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover {{
        transform: translateY(-2px);
        border-color: rgba(248, 215, 139, 0.85);
        background: rgba(26, 56, 60, 0.94);
        color: #ffffff;
    }}

    h2, h3 {{
        color: #f4fbf8;
        letter-spacing: 0;
    }}

    div[data-testid="stMarkdownContainer"] code {{
        color: #f8d78b;
        background: rgba(216, 176, 91, 0.12);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <section class="finance-hero">
        <h1>AI Personal Finance Risk Monitor</h1>
        <p>Upload bank or UPI transactions to classify expenses, detect risk, forecast spend, and generate reports.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Statement")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_demo = st.toggle("Use sample data", value=uploaded is None)
    st.markdown(
        "Required columns: `date`, `description`, `amount`. Optional column: `type` with values like debit or credit."
    )

try:
    raw_df = load_transactions(uploaded) if uploaded is not None else load_demo_data()
    if uploaded is None and not use_demo:
        st.info("Upload a CSV statement to begin.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read statement: {exc}")
    st.stop()

df = analyze_transactions(raw_df)
summary = summarize(df)
alerts = risk_alerts(df, summary)
category_df = category_spend(df)
monthly_df = monthly_spend(df)
budget_df = recommend_budgets(df)

metric_cols = st.columns(5)
metric_cols[0].metric("Total Spend", f"{summary.total_spend:,.0f}")
metric_cols[1].metric("Total Income", f"{summary.total_income:,.0f}")
metric_cols[2].metric("Net Cashflow", f"{summary.net_cashflow:,.0f}")
metric_cols[3].metric("Next Month Forecast", f"{summary.predicted_next_month_spend:,.0f}")
metric_cols[4].metric("Risk Flags", summary.anomaly_count + summary.duplicate_count + summary.frequency_alert_count)

alert_container = st.container(border=True)
with alert_container:
    st.subheader("Financial Risk Alerts")
    for alert in alerts:
        st.write(f"- {alert}")

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Monthly Spending Trend")
    st.line_chart(monthly_df, x="month", y="spend", height=300)

with right:
    st.subheader("Category-wise Spending")
    st.bar_chart(category_df, x="category", y="expense_amount", height=300)

tab_risk, tab_budget, tab_data = st.tabs(["Risk Monitor", "Budgets", "Transactions"])

with tab_risk:
    flagged = df[df["risk_level"].ne("Low")].copy()
    if flagged.empty:
        st.success("No unusual or repeated transactions were detected.")
    else:
        st.dataframe(
            flagged[
                [
                    "date",
                    "description",
                    "category",
                    "expense_amount",
                    "risk_level",
                    "is_amount_anomaly",
                    "is_duplicate_payment",
                    "is_frequency_risk",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

with tab_budget:
    st.dataframe(budget_df, use_container_width=True, hide_index=True)
    st.bar_chart(budget_df, x="category", y="recommended_budget", height=300)

with tab_data:
    st.dataframe(df, use_container_width=True, hide_index=True)

download_cols = st.columns(2)
csv_bytes = df.to_csv(index=False).encode("utf-8")
download_cols[0].download_button(
    "Download analyzed CSV",
    data=csv_bytes,
    file_name="analyzed_transactions.csv",
    mime="text/csv",
)

try:
    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        report_path = generate_pdf_report(df, summary, alerts, tmp.name)
    download_cols[1].download_button(
        "Download monthly PDF report",
        data=Path(report_path).read_bytes(),
        file_name="finance_risk_report.pdf",
        mime="application/pdf",
    )
except Exception as exc:
    download_cols[1].warning(f"PDF generation unavailable: {exc}")
