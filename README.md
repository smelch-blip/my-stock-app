# Wealth Architect Pro (NSE)

## What it does
Upload a CSV of NSE tickers and get a table with:
- Technicals: 50DMA, 150DMA, 200DMA
- Fundamentals: Sales growth % (YOY-3 years), Profit growth % (YOY 3years), ROE, ROCE
- Valuation: VAL: PB, VAL: Fair, VAL: MoS Buy, VAL: Method
- Final: Momentum, Recommendation, Reason

## Input file format (minimum)
Your CSV must contain a column like:
- stock symbol OR ticker OR symbol

Optional:
- company

Examples:
INFY
TCS
APOLLOTYRE

The app auto-adds `.NS` for NSE.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
