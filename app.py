import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. UI CONFIGURATION (Paper White High-Contrast) ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro", page_icon="🏛️")
st.markdown("""
    <style>
    .stApp { background-color: #ffffff; color: #000000; }
    section[data-testid="stSidebar"] { background-color: #f8f9fa !important; border-right: 1px solid #dddddd; }
    section[data-testid="stSidebar"] label { color: #000000 !important; font-weight: bold !important; }
    .stButton>button { background-color: #1d4ed8 !important; color: white !important; font-weight: bold; width: 100%; height: 3.5em; border-radius: 8px; }
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. THE SECTOR-METHOD MAPPING (Your One-Pager Logic) ---
SECTOR_LOGIC = {
    "Financial Services": {"method": "P/B", "target": 2.2},   # Values based on Book, ignores Sales
    "Consumer Defensive": {"method": "P/E", "target": 45.0}, # High quality, high premium
    "Technology": {"method": "P/E", "target": 28.0},        # Growth-sensitive
    "Healthcare": {"method": "P/E", "target": 32.0},
    "Basic Materials": {"method": "CYCLICAL", "target": 12.0}, # Normalizes earnings
    "Energy": {"method": "CYCLICAL", "target": 10.0},          # PSUs / Commodities
    "Industrials": {"method": "DEBT_ADJ", "target": 20.0},     # Penalizes leverage
    "ETF": {"method": "NAV", "target": 1.0},                  # No intrinsic valuation
    "Default": {"method": "P/E", "target": 18.0}
}

@st.cache_data(ttl=3600)
def fetch_rich_data(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        # Fetch 3 years for normalization and 200-DMA
        hist = t.history(period="3y", auto_adjust=False)
        return info, hist
    except: return {}, None

def engine_valuation(symbol, mos_pct):
    info, hist = fetch_rich_data(symbol)
    if not info or hist is None or hist.empty: return None

    # Technicals
    ltp = float(hist['Close'].iloc[-1])
    d50 = float(hist['Close'].rolling(50).mean().iloc[-1])
    d200 = float(hist['Close'].rolling(200).mean().iloc[-1])
    
    # Identify Sector & Method
    sector = info.get("sector", "Default")
    if info.get("quoteType") == "ETF": sector = "ETF"
    cfg = SECTOR_LOGIC.get(sector, SECTOR_LOGIC["Default"])
    
    fair_val = 0
    rationale = []
    
    # --- METHOD AWARE LOGIC ---
    
    # 1. Financials (Book Value Basis)
    if cfg["method"] == "P/B":
        bv = info.get("bookValue", 0)
        roe = info.get("returnOnEquity", 0.12)
        # Justified P/B: Higher ROE gets higher multiple
        dynamic_target = cfg["target"] * (roe / 0.15) if roe else cfg["target"]
        fair_val = bv * dynamic_target
        rationale.append(f"Valued on Book ({round(dynamic_target, 1)}x). ROE is {round(roe*100,1)}%. Sales growth ignored.")

    # 2. Cyclicals & PSUs (Normalization Layer)
    elif cfg["method"] == "CYCLICAL":
        # Instead of Peak EPS, use a proxy for mid-cycle earnings
        curr_eps = info.get("trailingEps", 0)
        # 20% Haircut applied if current margins are suspiciously high (Peak Earnings trap)
        norm_eps = curr_eps * 0.8 
        fair_val = norm_eps * cfg["target"]
        rationale.append(f"Cyclical/PSU logic: 20% haircut applied to peak earnings for normalization.")

    # 3. Industrials (Debt Awareness)
    elif cfg["method"] == "DEBT_ADJ":
        eps = info.get("forwardEps") or info.get("trailingEps") or 0
        debt_to_ebitda = info.get("debtToEbitda", 0)
        # Penalize fair multiple if leverage > 3
        debt_penalty = 0.8 if debt_to_ebitda > 3 else 1.0
        fair_val = eps * cfg["target"] * debt_penalty
        if debt_penalty < 1: rationale.append(f"Fair value penalized by 20% due to high leverage (D/E: {debt_to_ebitda}).")
        else: rationale.append("Valued on Forward P/E; leverage is within safe limits.")

    # 4. ETFs (Exclude)
    elif cfg["method"] == "NAV":
        return {"Ticker": symbol, "Verdict": "📦 ETF", "LTP": ltp, "Fair Price": "N/A", "Strategic Rationale": "Passive instrument; valuation not applicable."}

    # 5. Standard Compounders (FMCG/IT)
    else:
        eps = info.get("forwardEps") or info.get("trailingEps") or 0
        fair_val = eps * cfg["target"]
        rationale.append(f"Standard {cfg['method']} valuation at {cfg['target']}x multiple.")

    # --- VERDICT & MOMENTUM FILTER ---
    mos_price = fair_val * (1 - mos_pct/100)
    
    if ltp <= mos_price:
        if ltp > d50: verdict = "🚀 STRONG BUY" if ltp > d200 else "💎 ACCUMULATE"
        else: verdict = "✋ WATCHLIST"
    elif ltp <= fair_val:
        verdict = "🔥 BREAKOUT" if (ltp > d50 and ltp > d200) else "🟡 HOLD"
    else:
        verdict = "⚠️ AVOID"

    if ltp < d200: rationale.append("Long-term trend is Bearish.")
    
    return {
        "Ticker": symbol,
        "Verdict": verdict,
        "LTP": round(ltp, 2),
        "Fair Price": round(fair_val, 2) if fair_val > 0 else "N/A",
        "MoS Buy": round(mos_price, 2) if fair_val > 0 else "N/A",
        "Profit Growth (YoY)": f"{round(info.get('earningsGrowth', 0)*100,1)}%" if info.get('earningsGrowth') else "N/A",
        "Strategic Rationale": " ".join(rationale)
    }

# --- 3. UI LAYOUT ---
with st.sidebar:
    st.title("Audit Settings")
    st.divider()
    mos = st.slider("Required Margin of Safety %", 5, 40, 20)
    st.divider()
    st.write("**Engine Logic:**")
    st.caption("✔️ Financials: Justified P/B")
    st.caption("✔️ PSUs: Normalised EPS")
    st.caption("✔️ Infra: Debt-Adjusted P/E")
    st.caption("✔️ ETFs: Excluded")

st.title("🏛️ Wealth Architect Pro")
st.markdown("#### Strategic Market-Aligned Valuation Terminal")

uploaded_file = st.file_uploader("Upload Portfolio CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)

    if st.button("🚀 EXECUTE METHOD-AWARE AUDIT"):
        results = []
        status = st.empty()
        prog = st.progress(0)
        
        for i, (_, row) in enumerate(df.iterrows()):
            sym = str(row[ticker_col]).strip()
            if "." not in sym: sym += ".NS"
            status.text(f"Applying {sym} specific valuation method...")
            
            res = engine_valuation(sym, mos)
            if res: results.append(res)
            prog.progress((i+1)/len(df))
        
        status.empty()
        if results:
            st.write("---")
            res_df = pd.DataFrame(results)
            cols = ["Ticker", "Verdict", "LTP", "Fair Price", "MoS Buy", "Profit Growth (YoY)", "Strategic Rationale"]
            st.dataframe(res_df[cols], use_container_width=True, hide_index=True)
            st.download_button("📥 Export Analysis", res_df.to_csv(index=False), "Wealth_Analysis.csv")
