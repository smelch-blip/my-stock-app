import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. UI SETUP ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro")
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
    .stButton>button { background-color: #1d4ed8 !important; color: white !important; width: 100%; height: 3em; border-radius: 8px; }
    .status-box { padding: 10px; border-radius: 5px; background-color: #f1f5f9; border: 1px solid #cbd5e1; margin-bottom: 10px; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ONE-PAGER VALUATION LOGIC ---
SECTOR_DEFAULTS = {
    "Financial Services": {"method": "P/B", "target": 2.2},
    "Basic Materials": {"method": "CYCLICAL", "target": 12.0},
    "Energy": {"method": "CYCLICAL", "target": 10.0},
    "Industrials": {"method": "DEBT_ADJ", "target": 20.0},
    "Technology": {"method": "P/E", "target": 28.0},
    "Consumer Defensive": {"method": "P/E", "target": 45.0},
    "Default": {"method": "P/E", "target": 20.0}
}

def analyze_ticker(sym, mos):
    try:
        ticker = yf.Ticker(sym)
        # Fetching only essential data to avoid timeouts
        info = ticker.info
        if not info or 'quoteType' not in info: return None
        
        hist = ticker.history(period="1y") # Reduced from 2y to 1y for speed
        if hist.empty: return None
        
        ltp = hist['Close'].iloc[-1]
        d50 = hist['Close'].rolling(50).mean().iloc[-1]
        d200 = hist['Close'].rolling(200).mean().iloc[-1]
        
        sector = info.get("sector", "Default")
        if info.get("quoteType") == "ETF":
            return {"Ticker": sym, "Verdict": "📦 ETF", "LTP": round(ltp,2), "Strategic Rationale": "NAV tracking instrument."}
        
        cfg = SECTOR_DEFAULTS.get(sector, SECTOR_DEFAULTS["Default"])
        fair_val = 0
        rat = []

        # One-Pager Method Implementation
        if cfg["method"] == "P/B":
            bv = info.get("bookValue", 1)
            roe = info.get("returnOnEquity", 0.12)
            adj_target = cfg["target"] * (roe / 0.15) if roe else cfg["target"]
            fair_val = bv * adj_target
            rat.append(f"P/B Logic (ROE {round(roe*100,1)}%).")
        elif cfg["method"] == "CYCLICAL":
            eps = info.get("trailingEps", 0) * 0.8 # 20% Normalization Haircut
            fair_val = eps * cfg["target"]
            rat.append("Normalized EPS (20% Haircut).")
        elif cfg["method"] == "DEBT_ADJ":
            debt_ebitda = info.get("debtToEbitda", 0)
            penalty = 0.8 if debt_ebitda > 3 else 1.0
            eps = info.get("forwardEps") or info.get("trailingEps") or 0
            fair_val = eps * cfg["target"] * penalty
            if penalty < 1: rat.append(f"Debt penalty applied ({round(debt_ebitda,1)}x).")
        else:
            eps = info.get("forwardEps") or info.get("trailingEps") or 0
            fair_val = eps * cfg["target"]
            rat.append(f"Standard {cfg['method']} ({cfg['target']}x).")

        mos_price = fair_val * (1 - mos/100)
        
        # Momentum Filter
        if ltp <= mos_price:
            verdict = "🚀 STRONG BUY" if ltp > d200 else ("💎 ACCUMULATE" if ltp > d50 else "✋ WATCHLIST")
        elif ltp <= fair_val:
            verdict = "🔥 BREAKOUT" if (ltp > d50 and ltp > d200) else "🟡 HOLD"
        else:
            verdict = "⚠️ AVOID"

        return {
            "Ticker": sym, "Verdict": verdict, "LTP": round(ltp, 2), "Fair Price": round(fair_val, 2),
            "MoS Buy": round(mos_price, 2), "Rationale": " ".join(rat)
        }
    except:
        return None

# --- 3. THE INTERFACE ---
st.title("🏛️ Wealth Architect Pro")
st.markdown("### Method-Aware Valuation Engine")

with st.sidebar:
    st.header("Parameters")
    mos_val = st.slider("Margin of Safety %", 5, 40, 20)
    st.info("Method: P/B (Banks) | Cyclical (PSUs) | Debt-Adj (Infra)")

uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    tick_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)
    
    if tick_col and st.button("🚀 RUN AUDIT"):
        results = []
        # Diagnostic Console
        console = st.empty()
        progress = st.progress(0)
        
        ticker_list = df[tick_col].dropna().unique().tolist()
        
        for i, t in enumerate(ticker_list):
            clean_t = str(t).strip().upper()
            if "." not in clean_t: clean_t += ".NS" # Auto-suffix for Indian Market
            
            console.markdown(f'<div class="status-box">Processing: <b>{clean_t}</b> ({i+1}/{len(ticker_list)})</div>', unsafe_allow_html=True)
            
            res = analyze_ticker(clean_t, mos_val)
            if res: results.append(res)
            progress.progress((i + 1) / len(ticker_list))
            
        console.empty()
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            st.success(f"Audit Complete. Analyzed {len(results)} stocks.")
    elif not tick_col:
        st.error("CSV must contain a 'stock symbol' or 'ticker' column.")
