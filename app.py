import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time

# --- 1. APP CONFIGURATION & STYLING ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro", page_icon="🏛️")

# Professional Terminal CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #1e293b; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .stButton>button { border-radius: 5px; height: 3em; width: 100%; font-weight: bold; }
    [data-testid="stSidebar"] { background-color: #0f172a; }
    .stExpander { border: none; box-shadow: 0 2px 4px rgba(0,0,0,0.05); background: white; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SECTOR BANDS & LOGIC ---
SECTOR_BANDS = {
    "Financial Services": {"type": "PB", "min": 1.0, "max": 4.0},
    "Technology": {"type": "PE", "min": 18, "max": 45},
    "Consumer Defensive": {"type": "PE", "min": 30, "max": 65},
    "Consumer Cyclical": {"type": "PE", "min": 20, "max": 45},
    "Industrials": {"type": "PE", "min": 15, "max": 35},
    "Basic Materials": {"type": "PE", "min": 6, "max": 18},
    "Healthcare": {"type": "PE", "min": 25, "max": 50},
    "Utilities": {"type": "PE", "min": 10, "max": 20},
    "Energy": {"type": "PE", "min": 8, "max": 15},
    "Communication Services": {"type": "PE", "min": 15, "max": 30},
    "Real Estate": {"type": "PE", "min": 15, "max": 35},
    "Default": {"type": "PE", "min": 15, "max": 30},
}

def infer_sector_from_industry(industry: str) -> str:
    ind = (industry or "").lower()
    if any(k in ind for k in ["bank", "nbfc", "finance", "financial", "insurance", "asset", "broking"]): return "Financial Services"
    if any(k in ind for k in ["software", "it services", "technology", "semiconductor"]): return "Technology"
    if any(k in ind for k in ["pharma", "hospital", "healthcare", "diagnostic"]): return "Healthcare"
    if any(k in ind for k in ["power", "utility", "electric", "water", "gas"]): return "Utilities"
    if any(k in ind for k in ["oil", "gas", "refining", "energy"]): return "Energy"
    if any(k in ind for k in ["cement", "steel", "metal", "mining", "chemical", "materials"]): return "Basic Materials"
    if any(k in ind for k in ["telecom", "media", "entertainment"]): return "Communication Services"
    if any(k in ind for k in ["real estate", "reit", "property"]): return "Real Estate"
    if any(k in ind for k in ["fmcg", "beverage", "foods", "consumer", "retail", "apparel"]): return "Consumer Defensive"
    if any(k in ind for k in ["auto", "hotel", "travel", "leisure"]): return "Consumer Cyclical"
    if any(k in ind for k in ["industrial", "engineering", "capital goods", "construction"]): return "Industrials"
    return "Default"

# --- 3. HELPER FUNCTIONS ---
def safe_num(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return None
        return float(x)
    except: return None

def r2(x, nd=2):
    return round(float(x), nd) if x is not None else None

def color_decision(val):
    s = str(val)
    if "Strong Buy" in s: return "color:#28a745;font-weight:700"
    if "Avoid" in s or "Bearish" in s: return "color:#dc3545;font-weight:700"
    if "Hold" in s or "Watch" in s: return "color:#ffc107;font-weight:700"
    return "color:#6c757d"

def normalize_symbol(sym, suffix=".NS"):
    s = str(sym).strip().upper()
    if not s or s == "NAN": return ""
    return s if s.endswith((".NS", ".BO")) else s + suffix

# --- 4. DATA ENGINE ---
@st.cache_data(ttl=3600)
def fetch_yf(symbol):
    try:
        t = yf.Ticker(symbol)
        return t.info or {}, t.history(period="2y", auto_adjust=False)
    except: return {}, None

def analyze_wealth_engine(symbol, mos_pct, ttm_haircut, coe, g):
    info, hist = fetch_yf(symbol)
    if hist is None or hist.empty or len(hist) < 210: return None

    close = hist['Close'].dropna()
    ltp = float(close.iloc[-1])
    d50 = float(close.rolling(50).mean().iloc[-1])
    d200 = float(close.rolling(200).mean().iloc[-1])

    # Momentum Logic
    if ltp > d50 > d200: momentum = "Bullish"
    elif ltp < d200 or d50 < d200: momentum = "Bearish"
    else: momentum = "Neutral"

    # Sector & Fundamentals
    sector = info.get("sector") or infer_sector_from_industry(info.get("industry"))
    sector_cfg = SECTOR_BANDS.get(sector, SECTOR_BANDS["Default"])
    
    roe = safe_num(info.get('returnOnEquity'))
    roe_pct = roe * 100 if roe else 0
    
    ebitda, debt, cash = safe_num(info.get('ebitda')), safe_num(info.get('totalDebt')), safe_num(info.get('totalCash'))
    net_debt_ebitda = (debt - cash) / ebitda if ebitda and ebitda > 0 and debt is not None else None

    # Hard Stops
    reasons = []
    if net_debt_ebitda and net_debt_ebitda > 3.0 and sector != "Utilities": reasons.append("High Debt")
    if (safe_num(info.get('payoutRatio')) or 0) > 1.0: reasons.append("Dividend Unsafe")

    # Quality Score
    q_score = 0
    if roe_pct > 15: q_score += 30
    if (safe_num(info.get('revenueGrowth')) or 0) > 0.12: q_score += 20
    if (safe_num(info.get('operatingMargins')) or 0) > 0.15: q_score += 20
    if (safe_num(info.get('freeCashflow')) or 0) > 0: q_score += 30

    # Valuation
    fair_val = None
    if sector_cfg["type"] == "PB":
        book = safe_num(info.get('bookValue'))
        if book and book > 0:
            just_pb = (roe - g) / (coe - g) if (coe - g) > 0 and roe else (sector_cfg["min"] + sector_cfg["max"])/2
            if roe_pct > 18: just_pb *= 1.10
            elif roe_pct < 10: just_pb *= 0.85
            just_pb = max(sector_cfg["min"], min(sector_cfg["max"], just_pb))
            fair_val = book * just_pb
    else:
        eps = safe_num(info.get('forwardEps')) or (safe_num(info.get('trailingEps')) * ttm_haircut if info.get('trailingEps') else None)
        if eps:
            base_pe = (sector_cfg["min"] + sector_cfg["max"]) / 2
            peg = safe_num(info.get('pegRatio'))
            if peg:
                if peg < 1.2: base_pe *= 1.10
                elif peg > 2.0: base_pe *= 0.85
            base_pe = max(sector_cfg["min"], min(sector_cfg["max"], base_pe))
            fair_val = eps * base_pe

    if not fair_val: return None
    mos_price = fair_val * (1 - mos_pct/100)

    # Final Recommendation
    if reasons: rec = f"Avoid ({', '.join(reasons)})"
    elif q_score >= 70 and ltp <= mos_price and momentum != "Bearish": rec = "Strong Buy / Add"
    elif q_score >= 50 and ltp <= fair_val: rec = "Hold / Watch"
    else: rec = "Neutral / Wait"

    return {
        "Ticker": symbol, "Sector": sector, "Quality": q_score, "Momentum": momentum,
        "LTP": r2(ltp), "Fair Price": r2(fair_val), "MoS Buy": r2(mos_price),
        "ROE %": r2(roe_pct, 1), "Debt/EBIT": r2(net_debt_ebitda), "Recommendation": rec
    }

# --- 5. UI LAYOUT ---
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.title("🏛️ Wealth Architect Pro")
    st.caption("Strategic Multi-Factor Portfolio Intelligence")
with col_t2:
    st.write("") 
    st.info("System Status: Operational")

st.divider()

# Sidebar Controls
st.sidebar.header("🎯 Strategy Settings")
suffix = st.sidebar.selectbox("Market Suffix", [".NS (NSE)", ".BO (BSE)"])
default_suffix = ".NS" if ".NS" in suffix else ".BO"
mos = st.sidebar.slider("Margin of Safety %", 5, 40, 20)
ttm_h = st.sidebar.slider("TTM EPS Haircut", 0.6, 1.0, 0.85)
coe = st.sidebar.slider("Cost of Equity", 0.08, 0.20, 0.14)
g_rate = st.sidebar.slider("Long-term Growth", 0.03, 0.12, 0.06)

# Main Body
c1, c2 = st.columns([2, 3])
with c1:
    st.subheader("📂 Import Portfolio")
    uploaded_file = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"], label_visibility="collapsed")
with c2:
    st.subheader("🎯 Key Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("MoS Target", f"{mos}%")
    m2.metric("Min Quality", "70/100")
    m3.metric("Valuation", "Sector-Aware")

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    df_raw.columns = [c.lower().strip() for c in df_raw.columns]
    
    with st.expander("⚙️ Configure Data Mapping", expanded=True):
        col_m1, col_m2, col_m3 = st.columns(3)
        sym_col = col_m1.selectbox("Ticker Column", [c for c in df_raw.columns if "symbol" in c or "ticker" in c])
        cost_col = col_m2.selectbox("Avg Cost Column", ["(none)"] + [c for c in df_raw.columns if "cost" in c or "avg" in c])
        run_btn = col_m3.button("🚀 EXECUTE ANALYSIS", type="primary")

    if run_btn:
        results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, row in df_raw.iterrows():
            sym = normalize_symbol(row[sym_col], default_suffix)
            status.text(f"Analyzing {i+1}/{len(df_raw)}: {sym}")
            data = analyze_wealth_engine(sym, mos, ttm_h, coe, g_rate)
            if data:
                if cost_col != "(none)":
                    data["Portfolio Cost"] = r2(row[cost_col])
                    data["P/L %"] = r2(((data["LTP"] / data["Portfolio Cost"]) - 1) * 100)
                results.append(data)
            progress.progress((i + 1) / len(df_raw))
        
        status.success("✅ Analysis Complete")
        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.style.applymap(color_decision, subset=['Recommendation']), use_container_width=True)
            st.download_button("📥 Download Report", res_df.to_csv(index=False), "Wealth_Report.csv", "text/csv")
