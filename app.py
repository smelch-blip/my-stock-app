import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. CLEAN TERMINAL UI ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro", page_icon="🏛️")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
    section[data-testid="stSidebar"] label { color: #000000 !important; font-weight: bold !important; font-size: 1.1rem !important; }
    .stButton>button { background-color: #2563eb !important; color: white !important; font-weight: bold; width: 100%; height: 3.5em; border-radius: 10px; }
    div[data-testid="stMetricValue"] { color: #1e40af !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HARD-CODED LOGIC (Institutional Defaults) ---
# We are hard-coding conservative sector multiples here
DEFAULTS = {
    "Financial Services": {"type": "PB", "fair": 2.2},  # Quality Indian Banks trade ~2x Book
    "Technology": {"type": "PE", "fair": 30.0},        # Growth IT average
    "Consumer Defensive": {"type": "PE", "fair": 45.0}, # FMCG Premium
    "Industrials": {"type": "PE", "fair": 25.0},       # Infrastructure/Capital Goods
    "Default": {"type": "PE", "fair": 22.0}
}

@st.cache_data(ttl=3600)
def fetch_data(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.info or {}, t.history(period="2y")
    except: return {}, None

def quick_analyze(symbol, mos_pct, row_data):
    info, hist = fetch_data(symbol)
    if not info or hist is None or hist.empty: return None

    ltp = hist['Close'].iloc[-1]
    d200 = hist['Close'].rolling(200).mean().iloc[-1]
    
    # Valuation Logic
    sector = info.get("sector", "Default")
    cfg = DEFAULTS.get(sector, DEFAULTS["Default"])
    
    if cfg["type"] == "PB":
        val_basis = info.get("bookValue", 0)
    else:
        val_basis = info.get("forwardEps") or info.get("trailingEps") or 0
        
    fair_val = val_basis * cfg["fair"]
    mos_price = fair_val * (1 - mos_pct/100)

    # Simple Verdict
    if ltp <= mos_price and ltp > d200: verdict = "🚀 BUY (Strong)"
    elif ltp <= fair_val: verdict = "✋ HOLD"
    else: verdict = "⚠️ OVERVALUED"

    res = {
        "Ticker": symbol,
        "LTP": round(ltp, 2),
        "Fair Price": round(fair_val, 2),
        "Verdict": verdict,
        "Momentum": "Bullish" if ltp > d200 else "Bearish",
        "Rev Growth": f"{round(info.get('revenueGrowth', 0)*100, 1)}%" if info.get('revenueGrowth') else "N/A"
    }

    # Auto-calculate Portfolio Profit/Loss
    if 'average cost' in row_data:
        cost = float(row_data['average cost'])
        res["My P/L %"] = f"{round(((ltp/cost)-1)*100, 1)}%"
        
    return res

# --- 3. THE UI ---
with st.sidebar:
    st.title("Settings")
    st.write("The engine uses pre-set institutional fair values for each sector.")
    st.divider()
    # The only lever you need
    mos = st.slider("Margin of Safety %", 5, 40, 20, help="How much discount do you want before a 'BUY' signal?")
    st.divider()
    st.caption("Auto-Detecting Symbol, Qty, and Cost from CSV.")

st.title("🏛️ Wealth Architect Pro")
st.markdown("### Simple Portfolio Audit")

uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)

    if st.button("RUN AUDIT"):
        results = []
        bar = st.progress(0)
        status = st.empty()
        
        for i, (idx, row) in enumerate(df.iterrows()):
            sym = str(row[ticker_col]).strip()
            if "." not in sym: sym += ".NS"
            status.text(f"Checking {sym}...")
            
            res = quick_analyze(sym, mos, row)
            if res: results.append(res)
            bar.progress((i+1)/len(df))
            
        status.empty()
        if results:
            st.write("---")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
