import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. HIGH-CONTRAST "PAPER WHITE" UI ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro", page_icon="🏛️")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
    section[data-testid="stSidebar"] label { color: #000000 !important; font-weight: bold !important; }
    .stButton>button { background-color: #1d4ed8 !important; color: white !important; border-radius: 8px; font-weight: bold; width: 100%; height: 3.5em; }
    div[data-testid="stMetricValue"] { color: #1e40af !important; font-weight: 700; }
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HARD-CODED INSTITUTIONAL BENCHMARKS ---
SECTOR_DEFAULTS = {
    "Financial Services": {"type": "PB", "fair": 2.2},  
    "Technology": {"type": "PE", "fair": 28.0},        
    "Consumer Defensive": {"type": "PE", "fair": 45.0}, 
    "Healthcare": {"type": "PE", "fair": 35.0},         
    "Basic Materials": {"type": "PE", "fair": 15.0},    
    "Industrials": {"type": "PE", "fair": 22.0},       
    "Default": {"type": "PE", "fair": 20.0}
}

@st.cache_data(ttl=3600)
def fetch_data(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.info or {}, t.history(period="2y", auto_adjust=False)
    except: return {}, None

def analyze_wealth_engine(symbol, mos_pct):
    info, hist = fetch_data(symbol)
    if not info or hist is None or hist.empty: return None

    # 1. Technicals
    close = hist['Close'].dropna()
    ltp = float(close.iloc[-1])
    d200 = float(close.rolling(200).mean().iloc[-1])
    momentum = "Bullish" if ltp > d200 else "Bearish"

    # 2. Valuation Logic
    sector = info.get("sector", "Default")
    cfg = SECTOR_DEFAULTS.get(sector, SECTOR_DEFAULTS["Default"])
    
    if cfg["type"] == "PB":
        basis_val = info.get("bookValue", 0)
    else:
        basis_val = info.get("forwardEps") or info.get("trailingEps") or 0

    fair_val = basis_val * cfg["fair"]
    mos_price = fair_val * (1 - mos_pct/100)

    # 3. Growth Metrics (YoY = Year over Year for the most recent quarter)
    rev_g = info.get('revenueGrowth')
    prof_g = info.get('earningsGrowth') # YoY change in quarterly profits

    # 4. Rationale & Verdict
    rationale = []
    if ltp <= mos_price:
        verdict = "🚀 BUY"
        rationale.append(f"Undervalued: Trading at >{mos_pct}% discount to sector fair value.")
    elif ltp <= fair_val:
        verdict = "✋ HOLD"
        rationale.append("Fair Value: Trading within reasonable sector multiples.")
    else:
        verdict = "⚠️ AVOID"
        rationale.append("Expensive: Price exceeds conservative sector benchmarks.")

    if prof_g and prof_g > 0.15:
        rationale.append("Strong bottom-line growth.")
    elif prof_g and prof_g < 0:
        rationale.append("Declining profitability.")

    if momentum == "Bearish":
        rationale.append("Negative price trend (Below 200 DMA).")

    return {
        "Ticker": symbol,
        "Verdict": verdict,
        "LTP": round(ltp, 2),
        "Fair Price": round(fair_val, 2),
        "MoS Buy": round(mos_price, 2),
        "Momentum": momentum,
        "Sales Growth (YoY)": f"{round(rev_g*100, 1)}%" if rev_g else "N/A",
        "Profit Growth (YoY)": f"{round(prof_g*100, 1)}%" if prof_g else "N/A",
        "Strategic Rationale": " ".join(rationale)
    }

# --- 3. THE UI ---
with st.sidebar:
    st.title("Audit Settings")
    st.write("Valuation benchmarks are hard-coded for conservative Indian market standards.")
    st.divider()
    mos = st.slider("Required Margin of Safety %", 5, 40, 20)
    st.divider()
    st.caption("Standard Sector Multiples Applied:")
    st.caption("- Banks: 2.2x PB | IT: 28x PE | FMCG: 45x PE")

st.title("🏛️ Wealth Architect Pro")
st.write("Automated Fundamental & Technical Audit")

uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)

    if st.button("🚀 RUN FULL AUDIT"):
        if ticker_col:
            results = []
            status = st.empty()
            prog = st.progress(0)
            
            for i, (_, row) in enumerate(df.iterrows()):
                sym = str(row[ticker_col]).strip()
                if "." not in sym: sym += ".NS"
                status.text(f"Analyzing {sym}...")
                
                res = analyze_wealth_engine(sym, mos)
                if res: results.append(res)
                prog.progress((i+1)/len(df))
            
            status.empty()
            if results:
                st.write("---")
                res_df = pd.DataFrame(results)
                
                # Final Table Display
                cols = ["Ticker", "Verdict", "LTP", "Fair Price", "MoS Buy", "Momentum", "Sales Growth (YoY)", "Profit Growth (YoY)", "Strategic Rationale"]
                st.dataframe(res_df[cols], use_container_width=True, hide_index=True)
                
                st.download_button("📥 Download Report", res_df.to_csv(index=False), "Market_Audit.csv")
        else:
            st.error("Could not find a 'Stock Symbol' column.")
