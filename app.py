import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. HIGH-CONTRAST LIGHT UI ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro", page_icon="🏛️")

st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
    section[data-testid="stSidebar"] .stMarkdown p, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown h1 { color: #000000 !important; font-weight: 600 !important; }
    .stButton>button { background-color: #1d4ed8 !important; color: white !important; border-radius: 8px; font-weight: bold; width: 100%; height: 3.5em; }
    div[data-testid="stMetricValue"] { color: #1e40af !important; font-weight: 700; }
    /* Table font size and color */
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CONFIG & HELPERS ---
SECTOR_BANDS = {
    "Financial Services": {"type": "PB", "min": 1.0, "max": 4.0},
    "Technology": {"type": "PE", "min": 18, "max": 45},
    "Consumer Defensive": {"type": "PE", "min": 30, "max": 65},
    "Default": {"type": "PE", "min": 15, "max": 30}
}

def safe_num(x):
    try: return float(x) if x is not None and not np.isnan(x) else None
    except: return None

def r2(x, nd=2): return round(float(x), nd) if x is not None else "N/A"

# --- 3. THE FULL ANALYTICS ENGINE ---
@st.cache_data(ttl=3600)
def fetch_data(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.info or {}, t.history(period="2y", auto_adjust=False)
    except: return {}, None

def analyze_wealth_engine(symbol, mos_pct, ttm_h, coe, g):
    info, hist = fetch_data(symbol)
    if hist is None or hist.empty or len(hist) < 200: return None

    # Technicals
    close = hist['Close'].dropna()
    ltp = float(close.iloc[-1])
    d50 = float(close.rolling(50).mean().iloc[-1])
    d200 = float(close.rolling(200).mean().iloc[-1])
    momentum = "Bullish" if ltp > d50 > d200 else ("Bearish" if ltp < d200 else "Neutral")

    # Fundamentals
    rev_growth = safe_num(info.get('revenueGrowth'))
    op_margins = safe_num(info.get('operatingMargins'))
    roe = safe_num(info.get('returnOnEquity'))
    fcf = safe_num(info.get('freeCashflow'))
    
    # Quality Score (0-100)
    q_score = 0
    if roe and roe > 0.15: q_score += 30
    if rev_growth and rev_growth > 0.12: q_score += 20
    if op_margins and op_margins > 0.15: q_score += 20
    if fcf and fcf > 0: q_score += 30

    # Sector & Valuation
    sector = info.get("sector", "Default")
    cfg = SECTOR_BANDS.get(sector, SECTOR_BANDS["Default"])
    fair_val = None

    if cfg["type"] == "PB":
        book = safe_num(info.get('bookValue'))
        if book:
            just_pb = max(cfg["min"], min(cfg["max"], (roe - g)/(coe - g) if roe and coe > g else 2.0))
            fair_val = book * just_pb
    else:
        eps = safe_num(info.get('forwardEps')) or (safe_num(info.get('trailingEps')) * ttm_h if info.get('trailingEps') else None)
        if eps:
            fair_val = eps * ((cfg["min"] + cfg["max"]) / 2)

    if not fair_val: return None
    mos_price = fair_val * (1 - mos_pct/100)
    
    # Recommendation Logic
    if q_score >= 70 and ltp <= mos_price and momentum != "Bearish": rec = "✅ Strong Buy"
    elif q_score >= 50 and ltp <= fair_val: rec = "🟡 Hold/Watch"
    else: rec = "⚪ Wait/Neutral"

    return {
        "Ticker": symbol,
        "Sector": sector,
        "Quality": q_score,
        "Momentum": momentum,
        "LTP": r2(ltp),
        "50DMA": r2(d50),
        "200DMA": r2(d200),
        "Rev Growth %": f"{r2(rev_growth*100, 1)}%" if rev_growth else "N/A",
        "Op Margin %": f"{r2(op_margins*100, 1)}%" if op_margins else "N/A",
        "Fair Price": r2(fair_val),
        "MoS Buy": r2(mos_price),
        "Recommendation": rec
    }

# --- 4. THE UI ---
with st.sidebar:
    st.title("Settings")
    st.markdown("---")
    mos = st.slider("Margin of Safety %", 5, 40, 20)
    ttm_h = st.slider("TTM Multiple Haircut", 0.5, 1.0, 0.85)
    st.write("**Bank P/B Parameters**")
    coe = st.slider("Cost of Equity", 0.10, 0.20, 0.14)
    g_rate = st.slider("LT Growth (g)", 0.03, 0.10, 0.06)

st.title("🏛️ Wealth Architect Pro")
st.write("Upload your portfolio for a comprehensive Technical & Fundamental analysis.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Auto-detect Ticker, Qty, and Cost
    ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)
    qty_col = next((c for c in df.columns if "qty" in c or "quantity" in c or "qnty" in c), None)
    cost_col = next((c for c in df.columns if "cost" in c or "avg" in c), None)

    m1, m2 = st.columns(2)
    m1.metric("Stocks Detected", len(df))
    m2.metric("Applied MoS", f"{mos}%")

    if st.button("🚀 EXECUTE FULL SCALE ANALYSIS"):
        if ticker_col:
            results = []
            status = st.empty()
            prog = st.progress(0)
            
            for i, (_, row) in enumerate(df.iterrows()):
                sym = str(row[ticker_col]).strip()
                if "." not in sym: sym += ".NS"
                status.text(f"🔍 Deep Dive: {sym}...")
                
                res = analyze_wealth_engine(sym, mos, ttm_h, coe, g_rate)
                if res:
                    # Append Portfolio data if columns exist
                    if cost_col and qty_col:
                        buy_price = safe_num(row[cost_col])
                        if buy_price:
                            res["My Cost"] = r2(buy_price)
                            res["P/L %"] = f"{r2(((float(res['LTP']) / buy_price) - 1) * 100, 1)}%"
                    results.append(res)
                prog.progress((i+1)/len(df))
            
            status.empty()
            if results:
                st.write("### 📊 Comprehensive Market Report")
                res_df = pd.DataFrame(results)
                
                # Reorder columns to put most important info first
                base_cols = ["Ticker", "Recommendation", "Quality", "Momentum", "LTP", "Fair Price", "MoS Buy"]
                port_cols = ["My Cost", "P/L %"] if "P/L %" in res_df.columns else []
                tech_cols = ["50DMA", "200DMA", "Rev Growth %", "Op Margin %"]
                
                final_cols = base_cols + port_cols + tech_cols
                st.dataframe(res_df[final_cols], use_container_width=True, hide_index=True)
                
                csv = res_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Excel Report", csv, "Wealth_Analysis.csv")
        else:
            st.error("Missing 'Stock Symbol' column in CSV.")
