import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(layout="wide", page_title="Wealth Architect Pro")
st.title("🏛️ Strategic Portfolio Analyst")

# --- CUSTOM STYLING FUNCTION ---
def apply_heatmap(df):
    def color_val_flag(val):
        if val == "Cheap (>= MoS)": return 'background-color: #d4edda; color: #155724' # Green
        if val == "Fair": return 'background-color: #fff3cd; color: #856404' # Yellow
        if val == "Expensive": return 'background-color: #f8d7da; color: #721c24' # Red
        return ''

    def color_rec(val):
        if "Strong Buy" in str(val) or "Add" in str(val): return 'font-weight: bold; color: #28a745'
        if "Sell" in str(val): return 'font-weight: bold; color: #dc3545'
        return ''

    return df.style.applymap(color_val_flag, subset=['VAL: Valuation Flag']) \
                   .applymap(color_rec, subset=['FINAL: Recommendation'])

# --- HELPERS ---
def fmt(x, nd=2):
    return round(float(x), nd) if (x is not None and not pd.isna(x)) else "N/A"

@st.cache_data(ttl=3600)
def fetch_data(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        hist = t.history(period="2y")
        # Sector is fetched directly from info
        sector = info.get('sector', 'Unknown')
        return info, hist, sector
    except: return {}, pd.DataFrame(), "Unknown"

# --- MAIN APP ---
uploaded_file = st.file_uploader("Upload Portfolio", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    st.sidebar.header("Valuation Settings")
    global_fair_pe = st.sidebar.number_input("Default Fair P/E", value=25.0)
    mos_pct = st.sidebar.slider("Margin of Safety %", 0, 50, 15)

    if st.button("🚀 Run Sector-Based Analysis"):
        results = []
        prog = st.progress(0)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            sym = str(row.get('stock symbol', '')).strip()
            if not sym: continue
            if not sym.endswith((".NS", ".BO")): sym += ".NS"

            info, hist, sector = fetch_data(sym)
            
            # Technicals
            ltp = hist['Close'].iloc[-1] if not hist.empty else None
            d200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) > 200 else None
            
            # Fundamentals
            eps = info.get('forwardEps') or info.get('trailingEps')
            curr_pe = info.get('trailingPE')
            
            # Valuation Logic
            fair_price = eps * global_fair_pe if eps else None
            mos_price = fair_price * (1 - mos_pct/100) if fair_price else None
            
            val_flag = "NA"
            if ltp and fair_price:
                if ltp <= mos_price: val_flag = "Cheap (>= MoS)"
                elif ltp <= fair_price: val_flag = "Fair"
                else: val_flag = "Expensive"

            results.append({
                "Company": row.get('company name', sym),
                "Sector": sector,
                "LTP": fmt(ltp),
                "Current P/E": fmt(curr_pe),
                "VAL: Fair Price": fmt(fair_price),
                "VAL: MoS Buy": fmt(mos_price),
                "VAL: Valuation Flag": val_flag,
                "FINAL: Recommendation": "Buy/Add" if val_flag == "Cheap (>= MoS)" and ltp > d200 else "Hold/Watch"
            })
            prog.progress((i + 1) / len(df))

        final_df = pd.DataFrame(results)
        st.subheader("Interactive Portfolio Heatmap")
        st.dataframe(apply_heatmap(final_df), use_container_width=True)
