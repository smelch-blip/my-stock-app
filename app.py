import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time

st.set_page_config(layout="wide", page_title="Wealth Architect Pro")
st.title("🏛️ Strategic Portfolio & Valuation Analyst")

# --- CUSTOM HEATMAP STYLING ---
def apply_heatmap(df):
    def color_val_flag(val):
        if val == "Cheap (>= MoS)": return 'background-color: #d4edda; color: #155724' # Green
        if val == "Fair": return 'background-color: #fff3cd; color: #856404' # Yellow
        if val == "Expensive": return 'background-color: #f8d7da; color: #721c24' # Red
        return ''

    def color_rec(val):
        if "Buy" in str(val): return 'color: #28a745; font-weight: bold'
        if "Sell" in str(val): return 'color: #dc3545; font-weight: bold'
        return ''

    if df.empty:
        return df
    
    return df.style.applymap(color_val_flag, subset=['VAL: Valuation Flag']) \
                   .applymap(color_rec, subset=['FINAL: Recommendation'])

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def fetch_full_data(symbol):
    try:
        t = yf.Ticker(symbol)
        # Fetching info and history with a timeout safety
        info = t.info
        hist = t.history(period="2y")
        
        if hist.empty:
            return None
        
        # Extracting Data
        eps = info.get('forwardEps') or info.get('trailingEps')
        sector = info.get('sector', 'Unknown')
        pe = info.get('trailingPE')
        ltp = hist['Close'].iloc[-1]
        d200 = hist['Close'].rolling(200).mean().iloc[-1]
        
        return {"ltp": ltp, "d200": d200, "eps": eps, "sector": sector, "pe": pe}
    except Exception as e:
        return None

# --- UI ---
uploaded_file = st.file_uploader("Upload Portfolio (Excel or CSV)", type=["csv", "xlsx"])

if uploaded_file:
    # Read file
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        df.columns = [c.lower().strip() for c in df.columns]
        st.success(f"Loaded {len(df)} rows.")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
    
    # Sidebar Controls
    st.sidebar.header("Valuation Controls")
    fair_pe = st.sidebar.number_input("Standard Fair P/E", value=25.0)
    mos_pct = st.sidebar.slider("Margin of Safety %", 0, 50, 15)

    if st.button("Run Full Sector & Valuation Analysis"):
        results = []
        
        # Debugging / Progress Indicators
        progress_bar = st.progress(0)
        status_text = st.empty() 
        
        total_rows = len(df)
        
        for i, (idx, row) in enumerate(df.iterrows()):
            # Get symbol
            sym = str(row.get('stock symbol', '')).strip()
            if not sym or sym == 'nan':
                continue
            
            # Auto-add suffix if missing
            if not sym.endswith((".NS", ".BO")):
                sym += ".NS"

            # Update Debug Status
            status_text.text(f"🔍 Analyzing {i+1}/{total_rows}: {sym}...")
            progress_bar.progress((i + 1) / total_rows)

            # Fetch Data
            data = fetch_full_data(sym)
            
            if data:
                # Calculations
                f_eps = data['eps']
                ltp = data['ltp']
                
                fair_price = f_eps * fair_pe if f_eps else None
                mos_price = fair_price * (1 - mos_pct/100) if fair_price else None
                
                val_flag = "NA"
                if ltp and fair_price:
                    if ltp <= mos_price: val_price_status = "Cheap (>= MoS)"
                    elif ltp <= fair_price: val_price_status = "Fair"
                    else: val_price_status = "Expensive"
                    val_flag = val_price_status

                results.append({
                    "Company": row.get('company name', sym),
                    "Sector": data['sector'],
                    "LTP": round(ltp, 2),
                    "P/E": round(data['pe'], 2) if data['pe'] else "N/A",
                    "VAL: Fair Price": round(fair_price, 2) if fair_price else "N/A",
                    "VAL: MoS Buy": round(mos_price, 2) if mos_price else "N/A",
                    "VAL: Valuation Flag": val_flag,
                    "FINAL: Recommendation": "Strong Buy" if val_flag == "Cheap (>= MoS)" and ltp > data['d200'] else "Hold/Watch"
                })
            else:
                # If a stock fails, we add a row with "N/A" so the app doesn't break
                results.append({
                    "Company": row.get('company name', sym),
                    "Sector": "Not Found",
                    "LTP": 0, "P/E": "N/A", "VAL: Fair Price": "N/A", "VAL: MoS Buy": "N/A",
                    "VAL: Valuation Flag": "Invalid Symbol",
                    "FINAL: Recommendation": "Check Symbol"
                })

        # Final Display
        status_text.text("✅ Analysis Complete!")
        if results:
            res_df = pd.DataFrame(results)
            st.dataframe(apply_heatmap(res_df), use_container_width=True)
        else:
            st.error("No data could be retrieved. Please check your stock symbols.")
