import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. THE BRAINS: SECTOR-SPECIFIC VALUATION METHODS ---
# As per your One-Pager: Method determines the Multiple.
SECTOR_LOGIC = {
    "Financial Services": {"method": "P/B", "target": 2.2},   # Values based on Book
    "Consumer Defensive": {"method": "P/E", "target": 45.0}, # Brand Moats
    "Technology": {"method": "P/E", "target": 28.0},        # Growth-sensitive
    "Basic Materials": {"method": "CYCLICAL", "target": 12.0}, # Uses 20% Normalization Haircut
    "Energy": {"method": "CYCLICAL", "target": 10.0},          # PSU/Commodity logic
    "Industrials": {"method": "DEBT_ADJ", "target": 20.0},     # Penalizes leverage
    "ETF": {"method": "NAV", "target": 1.0},                  # Bypasses valuation
    "Default": {"method": "P/E", "target": 20.0}
}

@st.cache_data(ttl=3600)
def fetch_rich_data(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        hist = t.history(period="2y", auto_adjust=False)
        return info, hist
    except: return {}, None

def engine_valuation(symbol, mos_pct):
    info, hist = fetch_rich_data(symbol)
    if not info or hist is None or hist.empty or len(hist) < 200: return None

    # Technicals (200-DMA for Bullish/Bearish check)
    ltp = float(hist['Close'].iloc[-1])
    d200 = float(hist['Close'].rolling(200).mean().iloc[-1])
    d50 = float(hist['Close'].rolling(50).mean().iloc[-1])
    
    sector = info.get("sector", "Default")
    if info.get("quoteType") == "ETF": sector = "ETF"
    cfg = SECTOR_LOGIC.get(sector, SECTOR_LOGIC["Default"])
    
    fair_val = 0
    rationale = []
    
    # --- APPLYING YOUR ONE-PAGER PRINCIPLES ---
    if cfg["method"] == "P/B":
        bv = info.get("bookValue", 0)
        roe = info.get("returnOnEquity", 0.12)
        # Dynamic P/B: High ROE banks get higher multiple ceiling
        dynamic_target = cfg["target"] * (roe / 0.15) if roe else cfg["target"]
        fair_val = bv * dynamic_target
        rationale.append(f"Bank Logic: Valued on Book ({round(dynamic_target, 1)}x). Sales ignored.")

    elif cfg["method"] == "CYCLICAL":
        # PSU/Commodity: Avoid Peak Earnings Trap with a 20% Normalization Haircut
        curr_eps = info.get("trailingEps", 0)
        norm_eps = curr_eps * 0.8 
        fair_val = norm_eps * cfg["target"]
        rationale.append("Cyclical/PSU Logic: 20% haircut applied to normalize peak earnings.")

    elif cfg["method"] == "DEBT_ADJ":
        eps = info.get("forwardEps") or info.get("trailingEps") or 0
        debt_to_ebitda = info.get("debtToEbitda", 0)
        # Debt Penalty: Slashes Fair Value if leverage is > 3x
        debt_penalty = 0.8 if debt_to_ebitda > 3 else 1.0
        fair_val = eps * cfg["target"] * debt_penalty
        if debt_penalty < 1: rationale.append(f"Debt Warning: Value slashed 20% due to high leverage ({round(debt_to_ebitda,1)}x).")

    elif cfg["method"] == "NAV":
        return {"Ticker": symbol, "Verdict": "📦 ETF", "LTP": ltp, "Fair Price": "N/A", "Strategic Rationale": "ETF: Tracking NAV only."}

    else:
        eps = info.get("forwardEps") or info.get("trailingEps") or 0
        fair_val = eps * cfg["target"]
        rationale.append(f"Standard {cfg['method']} at {cfg['target']}x.")

    # --- THE VERDICT (Valuation + Momentum Filter) ---
    mos_price = fair_val * (1 - mos_pct/100)
    
    if ltp <= mos_price:
        if ltp > d200: verdict = "🚀 STRONG BUY"
        elif ltp > d50: verdict = "💎 ACCUMULATE"
        else: verdict = "✋ WATCHLIST"
    elif ltp <= fair_val:
        verdict = "🔥 BREAKOUT" if (ltp > d50 and ltp > d200) else "🟡 HOLD"
    else:
        verdict = "⚠️ AVOID"

    if ltp < d200: rationale.append("Below 200-DMA (Bearish Trend).")
    
    return {
        "Ticker": symbol, "Verdict": verdict, "LTP": round(ltp, 2), "Fair Price": round(fair_val, 2),
        "MoS Buy": round(mos_price, 2), "Profit Growth": f"{round(info.get('earningsGrowth', 0)*100,1)}%" if info.get('earningsGrowth') else "N/A",
        "Strategic Rationale": " ".join(rationale)
    }

# --- 3. UI LAYOUT ---
st.title("🏛️ Wealth Architect Pro")
st.sidebar.title("Settings")
mos = st.sidebar.slider("Margin of Safety %", 5, 40, 20)

uploaded_file = st.file_uploader("Drop your Portfolio CSV here", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)

    if st.button("🚀 EXECUTE AUDIT"):
        results = [engine_valuation(str(row[ticker_col]).strip() + (".NS" if "." not in str(row[ticker_col]) else ""), mos) for _, row in df.iterrows()]
        res_df = pd.DataFrame([r for r in results if r])
        st.dataframe(res_df, use_container_width=True, hide_index=True)
