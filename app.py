import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- 1. HIGH-CONTRAST "PAPER WHITE" UI CONFIG ---
st.set_page_config(layout="wide", page_title="Wealth Architect Pro", page_icon="🏛️")

st.markdown("""
    <style>
    /* Pure White Background & Crisp Black Text */
    .stApp { background-color: #ffffff; color: #000000; }
    
    /* Sidebar Styling: Light Grey with Deep Black Labels */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #dddddd;
    }
    section[data-testid="stSidebar"] label {
        color: #000000 !important;
        font-weight: bold !important;
    }

    /* Professional Blue Action Button */
    .stButton>button {
        background-color: #1d4ed8 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: bold;
        width: 100%;
        height: 3.5em;
        border: none;
    }

    /* Clean Table Borders */
    [data-testid="stDataFrame"] { border: 1px solid #e5e7eb; border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HARD-CODED SECTOR BENCHMARKS (Indian Market Standards) ---
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
        # Fetch 2 years of data to calculate 200-DMA
        return t.info or {}, t.history(period="2y", auto_adjust=False)
    except: return {}, None

# --- 3. THE ANALYTIC ENGINE (Valuation + 50/200 DMA Logic) ---
def analyze_wealth_engine(symbol, mos_pct):
    info, hist = fetch_data(symbol)
    if not info or hist is None or hist.empty or len(hist) < 200: return None

    # Technical Benchmarks
    close = hist['Close'].dropna()
    ltp = float(close.iloc[-1])
    d50 = float(close.rolling(50).mean().iloc[-1])   # Short-term Trend
    d200 = float(close.rolling(200).mean().iloc[-1]) # Long-term Trend
    
    # Valuation Logic
    sector = info.get("sector", "Default")
    cfg = SECTOR_DEFAULTS.get(sector, SECTOR_DEFAULTS["Default"])
    
    if cfg["type"] == "PB":
        basis_val = info.get("bookValue", 0)
    else:
        basis_val = info.get("forwardEps") or info.get("trailingEps") or 0

    fair_val = basis_val * cfg["fair"]
    mos_price = fair_val * (1 - mos_pct/100)

    # Growth Metrics (Year-over-Year)
    prof_g = info.get('earningsGrowth') 
    rev_g = info.get('revenueGrowth')

    # --- ADVANCED VERDICT ENGINE ---
    rationale = []
    
    if ltp <= mos_price:
        if ltp > d200:
            verdict = "🚀 STRONG BUY"
            rationale.append(f"Deep Value & Bullish: Price is >{mos_pct}% below fair value and trending up.")
        elif ltp > d50:
            verdict = "💎 ACCUMULATE"
            rationale.append("Early Reversal: Deep value detected and short-term trend (50-DMA) has turned up.")
        else:
            verdict = "✋ WATCHLIST"
            rationale.append("Value Trap Warning: Stock is cheap but still in a free-fall (below 50-DMA).")
            
    elif ltp <= fair_val:
        if ltp > d50 and ltp > d200:
            verdict = "🔥 BREAKOUT"
            rationale.append("Momentum Play: Price just cleared key averages; likely to trend toward new highs.")
        else:
            verdict = "🟡 HOLD"
            rationale.append("Fair Value: Trading within expected sector ranges with neutral momentum.")
    else:
        verdict = "⚠️ AVOID"
        rationale.append("Overvalued: Current price significantly exceeds conservative sector benchmarks.")

    # Quality Check for Rationale
    if prof_g and prof_g > 0.15:
        rationale.append(f"Growth is strong ({round(prof_g*100,0)}% YoY Profit).")
    elif prof_g and prof_g < 0:
        rationale.append("Caution: Quarterly earnings are shrinking.")

    return {
        "Ticker": symbol,
        "Verdict": verdict,
        "LTP": round(ltp, 2),
        "Fair Price": round(fair_val, 2),
        "MoS Buy": round(mos_price, 2),
        "50-DMA": round(d50, 2),
        "200-DMA": round(d200, 2),
        "Profit Growth (YoY)": f"{round(prof_g*100, 1)}%" if prof_g else "N/A",
        "Strategic Rationale": " ".join(rationale)
    }

# --- 4. THE UI LAYOUT ---
with st.sidebar:
    st.title("Audit Settings")
    st.write("Valuations are based on conservative Institutional Benchmarks.")
    st.divider()
    
    # Margin of Safety Lever
    mos = st.slider("Target Margin of Safety %", 5, 40, 20, 
                    help="How much discount from Fair Value is required for a Buy signal?")
    
    st.divider()
    st.caption("Standard Sector Multiples Applied:")
    st.caption("- Banks: 2.2x PB")
    st.caption("- Tech: 28x PE")
    st.caption("- FMCG: 45x PE")

st.title("🏛️ Wealth Architect Pro")
st.markdown("#### Strategic Portfolio Audit: Valuation, Momentum & Growth")

# File Upload
uploaded_file = st.file_uploader("Upload your Portfolio CSV", type=["csv"], label_visibility="collapsed")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Auto-find the ticker column
    ticker_col = next((c for c in df.columns if "symbol" in c or "ticker" in c), None)

    if st.button("🚀 EXECUTE MULTI-FACTOR AUDIT"):
        if ticker_col:
            results = []
            status = st.empty()
            prog = st.progress(0)
            
            for i, (_, row) in enumerate(df.iterrows()):
                sym = str(row[ticker_col]).strip()
                # Ensure it has the Yahoo Finance suffix for Indian Stocks
                if "." not in sym: sym += ".NS"
                
                status.text(f"Auditing Technicals & Fundamentals for {sym}...")
                
                res = analyze_wealth_engine(sym, mos)
                if res: results.append(res)
                prog.progress((i+1)/len(df))
            
            status.empty()
            if results:
                st.write("---")
                res_df = pd.DataFrame(results)
                
                # Define Column Order for the Results Table
                final_cols = [
                    "Ticker", "Verdict", "LTP", "Fair Price", "MoS Buy", 
                    "50-DMA", "200-DMA", "Profit Growth (YoY)", "Strategic Rationale"
                ]
                
                # Display high-contrast table
                st.dataframe(res_df[final_cols], use_container_width=True, hide_index=True)
                
                # Download Result
                st.download_button("📥 Export Audit to CSV", res_df.to_csv(index=False), "Wealth_Audit_Report.csv")
        else:
            st.error("Error: Could not find a 'Stock Symbol' column in your file. Please check your CSV header.")
else:
    st.info("💡 Please upload your portfolio CSV (containing 'Stock Symbol') to begin the audit.")
