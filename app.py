import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

st.set_page_config(layout="wide", page_title="Wealth Architect Pro")
st.title("🏛️ Strategic Portfolio & Valuation Analyst")

# --- CUSTOM HEATMAP STYLING ---
def apply_heatmap(df):
    def color_val_flag(val):
        if val == "Cheap (>= MoS)": return 'background-color: #d4edda; color: #155724'
        if val == "Fair": return 'background-color: #fff3cd; color: #856404'
        if val == "Expensive": return 'background-color: #f8d7da; color: #721c24'
        return ''

    def color_rec(val):
        if "Buy" in str(val): return 'color: #28a745; font-weight: bold'
        if "Sell" in str(val): return 'color: #dc3545; font-weight: bold'
        return ''

    return df.style.applymap(color_val_flag, subset=['VAL: Valuation Flag']) \
                   .applymap(color_rec, subset=['FINAL: Recommendation'])

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def fetch_full_data(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.info
        hist = t.history(period="2y")
        if hist.empty: return None
        
        # Fundamental Data
        eps = info.get('forwardEps') or info.get('trailingEps')
        sector = info.get('sector', 'Unknown')
        pe = info.get('trailingPE')
        
        # Technicals
        ltp = hist['Close'].iloc[-1]
        d50 = hist['Close'].rolling(50).mean().iloc[-1]
        d200 = hist['Close'].rolling(200).mean().iloc[-1]
        
        return {"ltp": ltp, "d50": d50, "d200": d200, "eps": eps, "sector": sector, "pe": pe}
    except: return None

# --- UI ---
uploaded_file = st.file_uploader("Upload Portfolio", type=["csv", "xlsx"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = [c.lower().strip() for c in df.columns]
    
    st.sidebar.header("Valuation Controls")
    fair_pe = st.sidebar.number_input("Standard Fair P/E", value=25.0)
    mos_pct = st.sidebar.slider("Margin of Safety %", 0, 50, 15)

    if st.button("Run Full Sector & Valuation Analysis"):
        results = []
        with st.spinner("Analyzing Market Fundamentals..."):
            for i, row in df.iterrows():
                sym = str(row.get('stock symbol', '')).strip()
                if not sym: continue
                if not sym.endswith((".NS", ".BO")): sym += ".NS"

                data = fetch_full_data(sym)
                if data:
                    # Valuation Calculations
                    fair_price = data['eps'] * fair_pe if data['eps'] else None
                    mos_price = fair_price * (1 - mos_pct/100) if fair_price else None
                    
                    val_flag = "NA"
                    if data['ltp'] and fair_price:
                        if data['ltp'] <= mos_price: val_flag = "Cheap (>= MoS)"
                        elif data['ltp'] <= fair_price: val_flag = "Fair"
                        else: val_flag = "Expensive"

                    results.append({
                        "Company": row.get('company name', sym),
                        "Sector": data['sector'],
                        "LTP": round(data['ltp'], 2),
                        "P/E": round(data['pe'], 2) if data['pe'] else "N/A",
                        "VAL: Fair Price": round(fair_price, 2) if fair_price else "N/A",
                        "VAL: MoS Buy": round(mos_price, 2) if mos_price else "N/A",
                        "VAL: Valuation Flag": val_flag,
                        "FINAL: Recommendation": "Strong Buy (Value)" if val_flag == "Cheap (>= MoS)" and data['ltp'] > data['d200'] else "Hold/Caution"
                    })
            
            if results:
                st.dataframe(apply_heatmap(pd.DataFrame(results)), use_container_width=True)
