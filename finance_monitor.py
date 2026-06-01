from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


CATEGORY_KEYWORDS = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "food", "dining", "pizza"],
    "Groceries": ["grocery", "dmart", "bigbasket", "blinkit", "zepto", "supermarket"],
    "Transport": ["uber", "ola", "metro", "fuel", "petrol", "diesel", "rapido"],
    "Shopping": ["amazon", "flipkart", "myntra", "shopping", "store", "mall"],
    "Utilities": ["electricity", "water", "gas", "broadband", "wifi", "mobile", "recharge"],
    "Rent": ["rent", "landlord", "lease"],
    "Healthcare": ["pharmacy", "hospital", "doctor", "clinic", "medical"],
    "Entertainment": ["netflix", "prime", "spotify", "movie", "bookmyshow", "gaming"],
    "Education": ["course", "tuition", "udemy", "coursera", "school", "college"],
    "Investment": ["mutual fund", "sip", "zerodha", "groww", "stocks", "nps"],
    "Income": ["salary", "refund", "cashback", "interest", "dividend"],
}

REQUIRED_COLUMNS = {"date", "description", "amount"}


@dataclass
class FinanceSummary:
    total_spend: float
    total_income: float
    net_cashflow: float
    predicted_next_month_spend: float
    anomaly_count: int
    duplicate_count: int
    frequency_alert_count: int


def load_transactions(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"CSV is missing required columns: {missing_text}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["description"] = df["description"].fillna("Unknown").astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["date", "amount"]).sort_values("date")

    if "type" not in df.columns:
        df["type"] = np.where(df["amount"] >= 0, "credit", "debit")
    else:
        df["type"] = df["type"].fillna("").astype(str).str.lower()
        df.loc[df["type"].eq(""), "type"] = np.where(df["amount"] >= 0, "credit", "debit")

    df["signed_amount"] = df["amount"]
    df["expense_amount"] = np.where(df["signed_amount"] < 0, df["signed_amount"].abs(), 0.0)
    debit_mask = df["type"].str.contains("debit|withdrawal|paid|dr", case=False, na=False)
    df.loc[debit_mask, "expense_amount"] = df.loc[debit_mask, "amount"].abs()
    df.loc[debit_mask, "signed_amount"] = -df.loc[debit_mask, "amount"].abs()

    credit_mask = df["type"].str.contains("credit|deposit|received|cr", case=False, na=False)
    df.loc[credit_mask, "signed_amount"] = df.loc[credit_mask, "amount"].abs()
    df.loc[credit_mask, "expense_amount"] = 0.0

    df["category"] = df["description"].map(categorize_description)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["day"] = df["date"].dt.date.astype(str)
    return df.reset_index(drop=True)


def categorize_description(description: str) -> str:
    text = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "Other"


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    spend = df[df["expense_amount"] > 0].copy()
    df["is_amount_anomaly"] = False
    df["anomaly_score"] = 0.0

    if len(spend) >= 8:
        features = pd.DataFrame(
            {
                "expense_amount": spend["expense_amount"],
                "day_of_month": spend["date"].dt.day,
                "category_code": spend["category"].astype("category").cat.codes,
            }
        )
        model = IsolationForest(contamination=0.12, random_state=42)
        predictions = model.fit_predict(features)
        scores = model.decision_function(features)
        df.loc[spend.index, "is_amount_anomaly"] = predictions == -1
        df.loc[spend.index, "anomaly_score"] = -scores
    else:
        threshold = spend["expense_amount"].median() * 3 if not spend.empty else np.inf
        df.loc[spend.index, "is_amount_anomaly"] = spend["expense_amount"] > threshold
        df.loc[spend.index, "anomaly_score"] = spend["expense_amount"] / max(threshold, 1)

    return df


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    same_day = df.duplicated(subset=["day", "description", "expense_amount"], keep=False)
    nearby_repeat = pd.Series(False, index=df.index)
    spend = df[df["expense_amount"] > 0].sort_values("date")
    for _, group in spend.groupby(["description", "expense_amount"]):
        if len(group) < 2:
            continue
        day_gap = group["date"].diff().dt.days.fillna(999)
        repeated_indices = group.index[day_gap <= 2]
        previous_indices = group.index[group.index.get_indexer(repeated_indices) - 1]
        nearby_repeat.loc[repeated_indices] = True
        nearby_repeat.loc[previous_indices] = True
    duplicated = (same_day | nearby_repeat) & (df["expense_amount"] > 0)
    df["is_duplicate_payment"] = duplicated
    return df


def detect_frequency_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    daily_counts = (
        df[df["expense_amount"] > 0]
        .groupby(["day", "category"])
        .size()
        .rename("daily_category_count")
        .reset_index()
    )
    df = df.merge(daily_counts, on=["day", "category"], how="left")
    df["daily_category_count"] = df["daily_category_count"].fillna(0).astype(int)
    df["is_frequency_risk"] = df["daily_category_count"] >= 4
    return df


def analyze_transactions(df: pd.DataFrame) -> pd.DataFrame:
    analyzed = detect_anomalies(df)
    analyzed = detect_duplicates(analyzed)
    analyzed = detect_frequency_risk(analyzed)
    analyzed["risk_level"] = np.select(
        [
            analyzed["is_amount_anomaly"] & analyzed["is_duplicate_payment"],
            analyzed["is_amount_anomaly"] | analyzed["is_frequency_risk"],
            analyzed["is_duplicate_payment"],
        ],
        ["Critical", "High", "Medium"],
        default="Low",
    )
    return analyzed


def monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("month", as_index=False)["expense_amount"]
        .sum()
        .rename(columns={"expense_amount": "spend"})
    )


def predict_next_month_spend(df: pd.DataFrame) -> float:
    monthly = monthly_spend(df)
    if monthly.empty:
        return 0.0
    if len(monthly) < 2:
        return float(monthly["spend"].mean())

    x = np.arange(len(monthly)).reshape(-1, 1)
    y = monthly["spend"].to_numpy()
    model = LinearRegression().fit(x, y)
    prediction = model.predict([[len(monthly)]])[0]
    return float(max(prediction, 0.0))


def recommend_budgets(df: pd.DataFrame) -> pd.DataFrame:
    category_monthly = (
        df[df["expense_amount"] > 0]
        .groupby(["month", "category"], as_index=False)["expense_amount"]
        .sum()
    )
    if category_monthly.empty:
        return pd.DataFrame(columns=["category", "avg_monthly_spend", "recommended_budget"])

    budgets = (
        category_monthly.groupby("category", as_index=False)["expense_amount"]
        .mean()
        .rename(columns={"expense_amount": "avg_monthly_spend"})
    )
    budgets["recommended_budget"] = (budgets["avg_monthly_spend"] * 0.9).round(2)
    return budgets.sort_values("recommended_budget", ascending=False)


def summarize(df: pd.DataFrame) -> FinanceSummary:
    total_spend = float(df["expense_amount"].sum())
    total_income = float(df.loc[df["signed_amount"] > 0, "signed_amount"].sum())
    return FinanceSummary(
        total_spend=total_spend,
        total_income=total_income,
        net_cashflow=total_income - total_spend,
        predicted_next_month_spend=predict_next_month_spend(df),
        anomaly_count=int(df["is_amount_anomaly"].sum()),
        duplicate_count=int(df["is_duplicate_payment"].sum()),
        frequency_alert_count=int(df["is_frequency_risk"].sum()),
    )


def risk_alerts(df: pd.DataFrame, summary: FinanceSummary) -> list[str]:
    alerts: list[str] = []
    if summary.anomaly_count:
        alerts.append(f"{summary.anomaly_count} unusual transaction amount(s) detected.")
    if summary.duplicate_count:
        alerts.append(f"{summary.duplicate_count} possible duplicate payment(s) found.")
    if summary.frequency_alert_count:
        alerts.append(f"{summary.frequency_alert_count} suspicious frequency alert(s) found.")
    if summary.predicted_next_month_spend > summary.total_income * 0.75 and summary.total_income > 0:
        alerts.append("Predicted spending is above 75% of observed income.")
    if summary.net_cashflow < 0:
        alerts.append("Net cashflow is negative for the uploaded period.")
    return alerts or ["No major financial risk alerts found."]


def category_spend(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["expense_amount"] > 0]
        .groupby("category", as_index=False)["expense_amount"]
        .sum()
        .sort_values("expense_amount", ascending=False)
    )


def generate_pdf_report(
    df: pd.DataFrame,
    summary: FinanceSummary,
    alerts: Iterable[str],
    output_path: str | Path,
) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ModuleNotFoundError:
        return _generate_basic_pdf_report(df, summary, alerts, output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = [
        Paragraph("Personal Finance Risk Report", styles["Title"]),
        Paragraph(datetime.now().strftime("Generated on %d %b %Y, %I:%M %p"), styles["Normal"]),
        Spacer(1, 12),
    ]

    summary_rows = [
        ["Metric", "Value"],
        ["Total spend", f"{summary.total_spend:,.2f}"],
        ["Total income", f"{summary.total_income:,.2f}"],
        ["Net cashflow", f"{summary.net_cashflow:,.2f}"],
        ["Predicted next month spend", f"{summary.predicted_next_month_spend:,.2f}"],
        ["Amount anomalies", str(summary.anomaly_count)],
        ["Duplicate payments", str(summary.duplicate_count)],
        ["Frequency alerts", str(summary.frequency_alert_count)],
    ]
    story.append(_styled_table(summary_rows))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Risk Alerts", styles["Heading2"]))
    for alert in alerts:
        story.append(Paragraph(f"- {alert}", styles["Normal"]))
    story.append(Spacer(1, 12))

    high_risk = df[df["risk_level"].isin(["Critical", "High", "Medium"])].copy()
    high_risk = high_risk.sort_values(["risk_level", "expense_amount"], ascending=[True, False])
    rows = [["Date", "Description", "Category", "Amount", "Risk"]]
    for _, row in high_risk.head(12).iterrows():
        rows.append(
            [
                row["date"].strftime("%Y-%m-%d"),
                row["description"][:34],
                row["category"],
                f"{row['expense_amount']:,.2f}",
                row["risk_level"],
            ]
        )
    story.append(Paragraph("Flagged Transactions", styles["Heading2"]))
    story.append(_styled_table(rows))
    doc.build(story)
    return output_path


def _styled_table(rows: list[list[str]]) -> Table:
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203040")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d6d9de")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f6f8fa")]),
            ]
        )
    )
    return table


def _generate_basic_pdf_report(
    df: pd.DataFrame,
    summary: FinanceSummary,
    alerts: Iterable[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Personal Finance Risk Report",
        datetime.now().strftime("Generated on %d %b %Y, %I:%M %p"),
        "",
        f"Total spend: {summary.total_spend:,.2f}",
        f"Total income: {summary.total_income:,.2f}",
        f"Net cashflow: {summary.net_cashflow:,.2f}",
        f"Predicted next month spend: {summary.predicted_next_month_spend:,.2f}",
        f"Amount anomalies: {summary.anomaly_count}",
        f"Duplicate payments: {summary.duplicate_count}",
        f"Frequency alerts: {summary.frequency_alert_count}",
        "",
        "Risk Alerts",
    ]
    lines.extend(f"- {alert}" for alert in alerts)
    lines.extend(["", "Flagged Transactions"])

    flagged = df[df["risk_level"].isin(["Critical", "High", "Medium"])].head(12)
    for _, row in flagged.iterrows():
        lines.append(
            f"{row['date'].strftime('%Y-%m-%d')} | {row['description'][:38]} | "
            f"{row['category']} | {row['expense_amount']:,.2f} | {row['risk_level']}"
        )

    _write_minimal_pdf(output_path, lines)
    return output_path


def _write_minimal_pdf(path: Path, lines: list[str]) -> None:
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    y = 800
    commands = ["BT", "/F1 12 Tf"]
    for line in lines[:45]:
        commands.append(f"50 {y} Td ({esc(line)}) Tj")
        commands.append(f"-50 -18 Td")
        y -= 18
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{i} 0 obj\n".encode())
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(bytes(pdf))
