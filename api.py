from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.finance_monitor import analyze_transactions, load_transactions, recommend_budgets, risk_alerts, summarize

app = FastAPI(title="AI Personal Finance Risk Monitor API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    try:
        raw_df = load_transactions(file.file)
        df = analyze_transactions(raw_df)
        summary = summarize(df)
        budgets = recommend_budgets(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    flagged = df[df["risk_level"].ne("Low")]
    return {
        "summary": summary.__dict__,
        "alerts": risk_alerts(df, summary),
        "budgets": budgets.to_dict(orient="records"),
        "flagged_transactions": flagged[
            ["date", "description", "category", "expense_amount", "risk_level"]
        ].astype({"date": str}).to_dict(orient="records"),
    }
