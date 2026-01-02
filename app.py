import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. MODERN LIGHT UI CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro", page_icon="🏛️")

# Custom CSS for a Clean Light Professional Look
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #ffffff; }
    
    /* Sidebar Styling - Light & High Contrast */
    section[data-testid="stSidebar"] {
        background-color: #f0f2f6 !important; /* Light Grey-Blue */
        border-right: 1px solid #e0e0e0;
        width: 350px !important;
    }
    
    /* Sidebar Text & Titles */
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #1e293b !important;
    }

    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        color: #2e7d32; /* Professional Green */
        font-weight: 700;
    }

    /* Primary Execution Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3.5em;
        background-color: #2563eb; /* Modern Blue */
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }

    /* Clean Dataframe */
    .stDataFrame {
        border: 1px solid #f0f2f6;
        border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. VALUATION ENGINE ---
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

def analyze_stock(symbol, mos_pct, ttm_h, coe, g):
    info, hist = fetch_data(symbol)
    if hist is None or hist.empty or len(hist) < 200: return None
    
    ltp = hist['Close'].iloc[-1]
    d200 = hist['Close'].rolling(200).mean().iloc[-1]
    momentum = "Bullish" if ltp > d200 else "Bearish"
    
    sector = info.get("sector") or infer_sector(info.get("industry"))
    cfg = SECTOR_BANDS.get(sector, SECTOR_BANDS["Default"])
    
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
    
    if ltp <= mos_price and momentum == "Bullish": rec = "✅ Strong Buy"
    elif ltp <= fair_val: rec = "🟡 Hold"
    else: rec = "🔴 Overvalued"
    
    return {"Ticker": symbol, "Sector": sector, "LTP": round(ltp, 2), "Fair Price": round(fair_val, 2), 
            "MoS Buy": round(mos_price, 2), "Momentum": momentum, "Recommendation": rec}

# --- 3. UI LAYOUT (LIGHT THEME) ---

with st.sidebar:
    st.title("⚙️ Engine Controls")
    st.markdown("Adjust your global valuation filters below.")
    st.write("")
    
    with st.expander("🛡️ Margin of Safety", expanded=True):
        mos = st.slider("Target MoS %", 5, 40, 20, help="Discount from Fair Value for a 'Strong Buy'")
        ttm_h = st.slider("TTM EPS Haircut", 0.5, 1.0, 0.85, help="Reduction in PE multiple when using past earnings")

    with st.expander("📈 Bank Valuation (P/B)", expanded=False):
        coe = st.slider("Cost of Equity", 0.10, 0.20, 0.14)
        g_rate = st.slider("Long-term Growth", 0.03, 0.10, 0.06)

    st.divider()
    st.caption("Status: All Systems Nominal")

# Main Display
st.title("🏛️ Wealth Architect Pro")
st.markdown("#### Strategic Portfolio Intelligence Terminal")
st.write("")

# Single, clean upload area
uploaded_file = st.file_uploader("Upload PortfolioImportTemplate.csv", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Dashboard Summary
    c1, c2, c3 = st.columns(3)
    c1.metric("Holdings Detected", len(df))
    c2.metric("Valuation Logic", "Sector-Aware")
    c3.metric("Safety Buffer", f"{mos}%")

    st.write("")
    # Big primary action button
    if st.button("🚀 EXECUTE MULTI-FACTOR ANALYSIS"):
        # Auto-detect column
        ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)
        
        if not ticker_col:
            st.error("Error: Could not find a 'Stock Symbol' column in your file.")
        else:
            results = []
            prog = st.progress(0)
            status = st.empty()
            
            for i, row in df.iterrows():
                sym = str(row[ticker_col]).strip()
                if "." not in sym: sym += ".NS"
                status.text(f"Analyzing: {sym}...")
                res = analyze_stock(sym, mos, ttm_h, coe, g_rate)
                if res: results.append(res)
                prog.progress((i+1)/len(df))
            
            status.empty()
            if results:
                st.write("---")
                st.subheader("📋 Institutional Valuation Report")
                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True, hide_index=True)
                
                # Download
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Export Analysis to CSV", csv, "Wealth_Report.csv", "text/csv")
else:
    st.info("👋 Welcome. Please upload your Portfolio CSV above to begin the automated valuation process.")
