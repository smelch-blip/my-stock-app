import streamlit as st
import pandas as pd
import yfinance as yf

st.set_page_config(layout="wide", page_title="Stock Analyst")
st.title("📈 Portfolio Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your portfolio (Excel or CSV)", type=["csv", "xlsx"])

def get_stock_metrics(symbol):
    try:
        ticker = yf.Ticker(str(symbol).strip())
        # We fetch 10y for growth and DMAs
        hist = ticker.history(period="10y")
        
        if hist.empty or len(hist) < 200:
            return None, None, None, None, "Data Missing"
        
        ltp = hist['Close'].iloc[-1]
        d50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        d150 = hist['Close'].rolling(window=150).mean().iloc[-1]
        d200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        rec = "Hold"
        if ltp > d50 > d200: rec = "Strong Buy"
        elif ltp < d200: rec = "Sell/Caution"
            
        return round(ltp, 2), round(d50, 2), round(d150, 2), round(d200, 2), rec
    except Exception:
        return None, None, None, None, "Invalid Symbol"

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.success("File Uploaded Successfully!")
    
    # Ensure columns are lowercase for the code to find them
    df.columns = [c.lower().strip() for c in df.columns]

    if st.button("Run Analysis"):
        with st.spinner('Analyzing stocks...'):
            results = []
            for index, row in df.iterrows():
                # Check if 'stock symbol' exists in your file
                if 'stock symbol' in df.columns:
                    sym = row['stock symbol']
                    ltp, d50, d150, d200, rec = get_stock_metrics(sym)
                    
                    results.append({
                        "Share Name": row.get('company name', sym),
                        "Qty": row.get('qnty', 0),
                        "LTP": ltp,
                        "50 DMA": d50,
                        "150 DMA": d150,
                        "200 DMA": d200,
                        "Recommendation": rec
                    })
                else:
                    st.error("Error: Could not find a column named 'stock symbol' in your file.")
                    break
            
            if results:
                st.table(pd.DataFrame(results))
