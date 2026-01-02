import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. CLEAN HIGH-CONTRAST UI ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro", page_icon="🏛️")

st.markdown("""
    <style>
    /* Force high-contrast dark text on white background */
    .stApp { background-color: #ffffff; color: #000000; }
    
    /* Sidebar: Light grey with dark text */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #dddddd;
    }
    
    /* Ensure all sidebar labels and text are deep black/slate */
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown h1 {
        color: #1a1a1a !important;
        font-weight: 600 !important;
    }

    /* Metric Values: High contrast green/red */
    div[data-testid="stMetricValue"] { color: #1e40af !important; }
    div[data-testid="stMetricLabel"] { color: #4b5563 !important; }

    /* Buttons: Bold blue with white text */
    .stButton>button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }

    /* Dataframe contrast */
    [data-testid="stTable"] { color: #000000 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE ENGINE ---
SECTOR_BANDS = {
    "Financial Services": {"type": "PB", "min": 1.0, "max": 4.0},
    "Technology": {"type": "PE", "min": 18, "max": 45},
    "Consumer Defensive": {"type": "PE", "min": 30, "max": 65},
    "Default": {"type": "PE", "min": 15, "max": 30}
}

@st.cache_data(ttl=3600)
def fetch_data(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.info or {}, t.history(period="2y")
    except: return {}, None

def analyze_stock(symbol, mos_pct, ttm_h, coe, g):
    info, hist = fetch_data(symbol)
    if hist is None or hist.empty or len(hist) < 200: return None
    ltp = hist['Close'].iloc[-1]
    d200 = hist['Close'].rolling(200).mean().iloc[-1]
    
    # Simple logic
    fair_val = (info.get("forwardEps") or (info.get("trailingEps", 0) * ttm_h)) * 20 # Simple PE fallback
    if info.get("sector") == "Financial Services":
        fair_val = (info.get("bookValue") or 1) * 2.0 # Simple PB fallback
    
    mos_price = fair_val * (1 - mos_pct/100)
    rec = "Strong Buy" if ltp <= mos_price else "Hold/Wait"
    
    return {"Ticker": symbol, "LTP": round(ltp, 2), "Fair Price": round(fair_val, 2), 
            "MoS Buy": round(mos_price, 2), "Momentum": "Bullish" if ltp > d200 else "Bearish", 
            "Recommendation": rec}

# --- 3. UI LAYOUT ---

with st.sidebar:
    st.title("Settings")
    st.markdown("---")
    mos = st.slider("Margin of Safety %", 5, 40, 20)
    ttm_h = st.slider("TTM Haircut", 0.5, 1.0, 0.85)
    st.markdown("---")
    st.write("**Bank P/B Settings**")
    coe = st.slider("Cost of Equity", 0.10, 0.20, 0.14)
    g_rate = st.slider("Growth", 0.03, 0.10, 0.06)

st.title("🏛️ Wealth Architect Pro")
st.write("Upload your CSV file below to run the analysis.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    st.write(f"✅ **{len(df)} Stocks Detected**")
    
    if st.button("RUN ANALYSIS"):
        # Auto-detect column
        ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)
        
        if ticker_col:
            results = []
            status = st.empty()
            for i, row in df.iterrows():
                sym = str(row[ticker_col]).strip()
                if "." not in sym: sym += ".NS"
                status.text(f"Processing: {sym}")
                res = analyze_stock(sym, mos, ttm_h, coe, g_rate)
                if res: results.append(res)
            
            status.empty()
            if results:
                st.subheader("Results Table")
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        else:
            st.error("No symbol column found.")
