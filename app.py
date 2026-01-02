import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time

# --- 1. UI & THEME ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
    .stButton>button { background-color: #1d4ed8 !important; color: white !important; width: 100%; height: 3em; border-radius: 8px; }
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VALUATION LOGIC (As per One-Pager) ---
SECTOR_LOGIC = {
    "Financial Services": {"method": "P/B", "target": 2.2},
    "Consumer Defensive": {"method": "P/E", "target": 45.0},
    "Technology": {"type": "P/E", "target": 28.0},
    "Basic Materials": {"method": "CYCLICAL", "target": 12.0},
    "Energy": {"method": "CYCLICAL", "target": 10.0},
    "Industrials": {"method": "DEBT_ADJ", "target": 20.0},
    "ETF": {"method": "NAV", "target": 1.0},
    "Default": {"method": "P/E", "target": 20.0}
}

def analyze_stock(symbol, mos_pct):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        if not info or 'sector' not in info and info.get('quoteType') != 'ETF':
            return None
            
        hist = t.history(period="2y")
        if hist.empty: return None

        ltp = float(hist['Close'].iloc[-1])
        d50 = float(hist['Close'].rolling(50).mean().iloc[-1])
        d200 = float(hist['Close'].rolling(200).mean().iloc[-1])
        
        sector = info.get("sector", "Default")
        if info.get("quoteType") == "ETF": sector = "ETF"
        cfg = SECTOR_LOGIC.get(sector, SECTOR_LOGIC["Default"])
        
        fair_val = 0
        rationale = []

        # Sector-Specific Math
        if cfg.get("method") == "P/B":
            bv = info.get("bookValue", 1)
            roe = info.get("returnOnEquity", 0.12)
            dynamic_target = cfg["target"] * (roe / 0.15) if roe else cfg["target"]
            fair_val = bv * dynamic_target
            rationale.append(f"Bank logic: P/B based on {round(roe*100,1)}% ROE.")
        elif cfg.get("method") == "CYCLICAL":
            norm_eps = info.get("trailingEps", 0) * 0.8 # 20% Normalization Haircut
            fair_val = norm_eps * cfg["target"]
            rationale.append("Cyclical logic: 20% peak-earnings haircut applied.")
        elif cfg.get("method") == "DEBT_ADJ":
            debt_ebitda = info.get("debtToEbitda", 0)
            penalty = 0.8 if debt_ebitda > 3 else 1.0
            fair_val = (info.get("forwardEps") or info.get("trailingEps") or 0) * cfg["target"] * penalty
            if penalty < 1: rationale.append(f"Debt penalty applied (D/E: {round(debt_ebitda,1)}).")
        elif sector == "ETF":
            return {"Ticker": symbol, "Verdict": "📦 ETF", "LTP": ltp, "Fair Price": "N/A", "Strategic Rationale": "ETF/Fund: Fundamental valuation not applicable."}
        else:
            eps = info.get("forwardEps") or info.get("trailingEps") or 0
            fair_val = eps * cfg["target"]
            rationale.append(f"Standard P/E valuation ({cfg['target']}x).")

        # Status Logic
        mos_price = fair_val * (1 - mos_pct/100)
        if ltp <= mos_price:
            if ltp > d200: verdict = "🚀 STRONG BUY"
            elif ltp > d50: verdict = "💎 ACCUMULATE"
            else: verdict = "✋ WATCHLIST"
        elif ltp <= fair_val:
            verdict = "🔥 BREAKOUT" if (ltp > d50 and ltp > d200) else "🟡 HOLD"
        else:
            verdict = "⚠️ AVOID"

        return {
            "Ticker": symbol, "Verdict": verdict, "LTP": round(ltp, 2), "Fair Price": round(fair_val, 2),
            "Profit Growth": f"{round(info.get('earningsGrowth', 0)*100,1)}%" if info.get('earningsGrowth') else "N/A",
            "Strategic Rationale": " ".join(rationale)
        }
    except Exception as e:
        return None

# --- 3. UI LAYOUT ---
st.title("🏛️ Wealth Architect Pro")
st.markdown("### Strategic Market-Aligned Audit")

with st.sidebar:
    st.header("Settings")
    mos = st.slider("Margin of Safety %", 5, 40, 20)
    st.divider()
    st.info("Logic: P/B for Banks, Normalized EPS for PSUs, Debt-Adjusted P/E for Infra.")

uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)

    if ticker_col and st.button("🚀 RUN ANALYSIS"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tickers = df[ticker_col].tolist()
        for i, raw_ticker in enumerate(tickers):
            symbol = str(raw_ticker).strip()
            # Handle Indian Suffix
            if "." not in symbol: symbol += ".NS"
            
            status_text.text(f"Processing {i+1}/{len(tickers)}: {symbol}...")
            res = analyze_stock(symbol, mos)
            if res: results.append(res)
            progress_bar.progress((i + 1) / len(tickers))
            
        status_text.success("Analysis Complete!")
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
