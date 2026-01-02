import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. PRO TERMINAL LAYOUT ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro", page_icon="🏛️")

# CSS to clean up the sidebar and buttons
st.markdown("""
    <style>
    section[data-testid="stSidebar"] { width: 320px !important; background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #2e7d32; color: white; font-weight: bold; }
    .stExpander { border: none !important; }
    div[data-testid="stMetricValue"] { font-size: 22px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGIC & BANDS (Hidden from UI) ---
SECTOR_BANDS = {
    "Financial Services": {"type": "PB", "min": 1.0, "max": 4.0},
    "Technology": {"type": "PE", "min": 18, "max": 45},
    "Consumer Defensive": {"type": "PE", "min": 30, "max": 65},
    "Healthcare": {"type": "PE", "min": 25, "max": 50},
    "Basic Materials": {"type": "PE", "min": 6, "max": 18},
    "Default": {"type": "PE", "min": 15, "max": 30},
}

def infer_sector(ind):
    ind = (ind or "").lower()
    if any(k in ind for k in ["bank", "finance", "insurance"]): return "Financial Services"
    if any(k in ind for k in ["software", "it", "tech"]): return "Technology"
    if any(k in ind for k in ["pharma", "health"]): return "Healthcare"
    if any(k in ind for k in ["steel", "metal", "cement", "chem"]): return "Basic Materials"
    return "Default"

@st.cache_data(ttl=3600)
def fetch_data(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.info or {}, t.history(period="2y")
    except: return {}, None

# --- 3. THE ANALYTIC ENGINE ---
def analyze_stock(symbol, mos_pct, ttm_h, coe, g):
    info, hist = fetch_data(symbol)
    if hist is None or hist.empty or len(hist) < 200: return None
    
    ltp = hist['Close'].iloc[-1]
    d200 = hist['Close'].rolling(200).mean().iloc[-1]
    momentum = "Bullish" if ltp > d200 else "Bearish"
    
    sector = info.get("sector") or infer_sector(info.get("industry"))
    cfg = SECTOR_BANDS.get(sector, SECTOR_BANDS["Default"])
    
    # Valuation Logic
    fair_val = None
    if cfg["type"] == "PB":
        book = info.get("bookValue")
        roe = info.get("returnOnEquity", 0.12)
        if book:
            just_pb = max(cfg["min"], min(cfg["max"], (roe - g)/(coe - g) if coe > g else 2.0))
            fair_val = book * just_pb
    else:
        eps = info.get("forwardEps") or (info.get("trailingEps", 0) * ttm_h)
        if eps:
            fair_val = eps * ((cfg["min"] + cfg["max"]) / 2)

    if not fair_val: return None
    mos_price = fair_val * (1 - mos_pct/100)
    
    # Simple Recommendation
    if ltp <= mos_price and momentum == "Bullish": rec = "✅ Strong Buy"
    elif ltp <= fair_val: rec = "🟡 Hold"
    else: rec = "🔴 Overvalued"
    
    return {"Ticker": symbol, "Sector": sector, "LTP": round(ltp, 2), "Fair Price": round(fair_val, 2), 
            "MoS Buy": round(mos_price, 2), "Momentum": momentum, "Recommendation": rec}

# --- 4. THE CLEAN UI ---

# Sidebar Header & Grouping
with st.sidebar:
    st.title("⚙️ Parameters")
    st.markdown("---")
    
    with st.expander("🛡️ Valuation Margins", expanded=True):
        mos = st.slider("Margin of Safety %", 5, 40, 20)
        ttm_h = st.slider("TTM Haircut", 0.5, 1.0, 0.85)

    with st.expander("📈 Bank P/B Inputs", expanded=False):
        coe = st.slider("Cost of Equity", 0.10, 0.20, 0.14)
        g_rate = st.slider("Long-term Growth", 0.03, 0.10, 0.06)

    st.markdown("---")
    st.caption("v3.5 Build | Data: Yahoo Finance")

# Main Area
st.title("🏛️ Wealth Architect Pro")
st.markdown("#### Institutional Portfolio Intelligence")

# 1. Direct Upload
uploaded_file = st.file_uploader("Upload your portfolio CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 2. Key Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Holdings Detected", len(df))
    m2.metric("Target MoS", f"{mos}%")
    m3.metric("Analysis Mode", "Sector-Aware")

    # 3. Big Action Button
    if st.button("🚀 ANALYZE PORTFOLIO"):
        results = []
        # Auto-detect Ticker column
        ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)
        
        if not ticker_col:
            st.error("Could not find a 'Stock Symbol' column in your CSV.")
        else:
            prog = st.progress(0)
            status = st.empty()
            for i, row in df.iterrows():
                sym = str(row[ticker_col]).strip()
                if "." not in sym: sym += ".NS"
                status.text(f"Processing: {sym}")
                res = analyze_stock(sym, mos, ttm_h, coe, g_rate)
                if res: results.append(res)
                prog.progress((i+1)/len(df))
            
            status.empty()
            if results:
                st.subheader("📋 Analysis Results")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                st.download_button("📥 Download Excel Report", res_df.to_csv(index=False), "Analysis.csv")

else:
    st.info("💡 Please upload your PortfolioImportTemplate.csv to begin.")
