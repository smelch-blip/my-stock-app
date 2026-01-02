import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(layout="wide", page_title="Stock Analyst")

st.title("📈 Portfolio Analysis Dashboard")

# Upload Section
uploaded_file = st.file_uploader("Upload your portfolio (Excel or CSV)", type=["csv", "xlsx"])

def get_stock_metrics(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # Fetch 10 years of data for long-term growth and DMAs
        hist = ticker.history(period="10y")
        if hist.empty: return None
        
        ltp = hist['Close'].iloc[-1]
        d50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        d150 = hist['Close'].rolling(window=150).mean().iloc[-1]
        d200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # Simple Recommendation logic
        rec = "Hold"
        if ltp > d50 > d200: rec = "Strong Buy"
        elif ltp < d200: rec = "Sell/Caution"
            
        return round(ltp, 2), round(d50, 2), round(d150, 2), round(d200, 2), rec
    except:
        return None, None, None, None, "Error"

if uploaded_file:
    # Read the file
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.success("File Uploaded Successfully!")
    
    if st.button("Run Analysis"):
        with st.spinner('Fetching market data...'):
            results = []
            for index, row in df.iterrows():
                # We use the symbol column from your file
                sym = str(row['stock symbol']).strip()
                ltp, d50, d150, d200, rec = get_stock_metrics(sym)
                
                results.append({
                    "Share Name": row['company name'],
                    "Qty": row['qnty'],
                    "LTP": ltp,
                    "50 DMA": d50,
                    "150 DMA": d150,
                    "200 DMA": d200,
                    "Recommendation": rec
                })
            
            st.table(pd.DataFrame(results))
