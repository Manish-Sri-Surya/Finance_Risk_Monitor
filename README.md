# AI Personal Finance Risk and Fraud Monitor

A Python project that analyzes bank or UPI transaction CSVs, classifies expenses, detects suspicious activity, predicts next-month spending, recommends category budgets, and generates a monthly PDF risk report.

## Features

- CSV statement upload with flexible debit/credit handling
- Expense categorization from merchant descriptions
- Unusual amount detection with `IsolationForest`
- Duplicate payment detection
- Suspicious transaction frequency detection
- Monthly spending forecast with linear regression
- Category-wise budget recommendation
- Streamlit dashboard
- FastAPI analysis endpoint
- PDF finance risk report

## CSV Format

Required columns:

```csv
date,description,amount
2026-05-08,Unknown International Merchant,38500
```

Optional column:

```csv
type
```

Use values such as `debit`, `credit`, `paid`, or `received`. If `type` is missing, negative amounts are treated as expenses and positive amounts as income.

## Run Dashboard

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the Streamlit URL shown in the terminal. The dashboard loads `data/sample_transactions.csv` by default.

## Run API

```bash
uvicorn api:app --reload
```

Analyze a statement:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@data/sample_transactions.csv"
```

## Project Structure

```text
finance-risk-monitor/
  app.py
  api.py
  requirements.txt
  assets/finance-dashboard-bg.png
  data/sample_transactions.csv
  src/finance_monitor.py
```

## Resume Bullet

Developed a personal finance risk monitor that classifies transactions, detects anomalies, predicts monthly expenses, recommends budgets, and generates financial risk reports using Python, pandas, scikit-learn, FastAPI, and Streamlit.
